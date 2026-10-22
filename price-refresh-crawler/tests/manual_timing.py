"""Opt-in timing against the EXISTING host-local PostgreSQL; never unittest-discovered.

From price-refresh-crawler (no configuration files are edited):
  python3 tests/manual_timing.py --docker --db-env-file ../.env --preflight
  python3 tests/manual_timing.py --docker --db-env-file ../.env --live --scope first
  python3 tests/manual_timing.py --docker --db-env-file ../.env --live --scope all

Run tests/test_manual_timing.py offline BEFORE --preflight/--live. --live updates
only existing local prices through the real writer, never recommendation flags,
and adds PENDING outbox records for changed recommended products.
Successful first-category writes are NOT reset before the full run.
Docker startup, imports and the tool's read-only preflight are excluded from the
perf_counter run_apply measurement and included in container_wall_seconds.
run_apply never sends to the backend; the boundary is the result emit after the
last price/outbox commit. --live requires the price_change_outbox table, otherwise
run_apply stops at its own preflight. A failed live lookup aborts measurement
immediately (after the scanner's existing finally/1-second wait), with no retry/next sort.
"""

import argparse
import hashlib
import http.client
import json
import os
import platform
import ssl
import subprocess
import sys
import time
from contextlib import ExitStack, contextmanager
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch
from urllib.parse import parse_qsl, urlsplit, urlunsplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from price_refresh.apply import run_apply
from price_refresh.backend import BackendClient
from price_refresh.logging_config import configure_logging
from price_refresh.parser import listing_items
from price_refresh.repository import DatabaseSettings
from price_refresh.scanner import collect_prices
from price_refresh.settings import Settings
from price_refresh.source.categories import CATEGORIES
from price_refresh.source.client import LISTING_ENDPOINT, TwentyNineCmClient
from price_refresh.writer import PostgresApplySession

IMAGE = "lookddak-price-refresh:py3.13.16"
LISTING_HOST = urlsplit(LISTING_ENDPOINT).hostname
LISTING_PATH = urlsplit(LISTING_ENDPOINT).path + "?colorchipVariant=treatment"


class TimingAbort(BaseException):
    """A test supervisor abort; not swallowed by the scanner/apply error handlers."""


@contextmanager
def block_backend_network(stats, *, allow_listing=False):
    """Fail closed BEFORE connect/request. Only the exact listing host/path allowed."""
    original_http = http.client.HTTPConnection.connect
    original_https = http.client.HTTPSConnection.connect
    original_request = http.client.HTTPConnection.request
    original_response = http.client.HTTPConnection.getresponse

    def check(connection):
        if not allow_listing or connection.host != LISTING_HOST or connection.port != 443:
            stats["blocked_http_attempts"] += 1
            raise TimingAbort("unexpected_http_transport")

    def connect_http(connection):
        check(connection)
        return original_http(connection)

    def connect_https(connection):
        check(connection)
        return original_https(connection)

    def request(connection, method, url, *args, **kwargs):
        check(connection)
        if method != "POST" or url != LISTING_PATH:
            stats["blocked_http_attempts"] += 1
            raise TimingAbort("unexpected_http_request")
        stats["listing_http_request_attempts"] += 1
        return original_request(connection, method, url, *args, **kwargs)

    def getresponse(connection):
        check(connection)
        response = original_response(connection)
        status = str(response.status)
        stats["http_status_counts"][status] = stats["http_status_counts"].get(status, 0) + 1
        return response

    def actual_backend_send(*_args, **_kwargs):
        stats["blocked_http_attempts"] += 1
        raise TimingAbort("real_backend_client_forbidden")

    with ExitStack() as stack:
        stack.enter_context(patch.object(http.client.HTTPConnection, "connect", connect_http))
        stack.enter_context(patch.object(http.client.HTTPSConnection, "connect", connect_https))
        stack.enter_context(patch.object(http.client.HTTPConnection, "request", request))
        stack.enter_context(patch.object(http.client.HTTPConnection, "getresponse", getresponse))
        stack.enter_context(patch.object(BackendClient, "send", actual_backend_send))
        yield


def new_stats():
    return {"listing_fetch_attempts": 0, "listing_http_request_attempts": 0,
            "listing_error_count": 0, "http_status_counts": {}, "blocked_http_attempts": 0,
            "actual_backend_http_requests": 0}


def measure(session, collect, *, stats=None, clock=time.perf_counter):
    marks, stats = {}, stats if stats is not None else new_stats()
    captured = {}

    def timed_collect():
        marks["scan_start"] = clock()
        try:
            scan = collect()
            stats["completed_categories"] = len(scan.categories)
            stats["scanner_page_requests"] = sum(category.page_requests for category in scan.categories)
            stats["scanner_observed_codes"] = len(scan.prices)
            return scan
        finally:
            marks["scan_end"] = clock()

    def emit(payload):
        marks["storage_end"] = clock()  # after the last price/outbox commit, still under the lock
        captured.update(payload)

    aborted = False
    started_at = datetime.now(timezone(timedelta(hours=9))).isoformat(timespec="milliseconds")
    marks["start"] = clock()  # Immediately before the existing operating orchestrator.
    try:
        run_apply(session, timed_collect, emit=emit)
    except TimingAbort:
        aborted = True  # run_apply's existing finally already closes/unlocks.
    marks["end"] = clock()

    def interval(start, end):
        return (round(marks[end] - marks[start], 6)
                if start in marks and end in marks else None)

    # emit also runs from run_apply's finally on aborts/preflight failures.
    reached = not aborted and captured.get("status") in {"completed", "partial_failure"}
    return {
        "started_at_kst": started_at, "run_apply_status": captured.get("status"),
        "measurement_aborted": aborted,
        "boundary": "price_and_outbox_storage_complete" if reached else "not_reached",
        "start_to_boundary_seconds": interval("start", "storage_end") if reached else None,
        "segments_seconds": {
            "lock_preflight_and_initial_db_read": interval("start", "scan_start"),
            "actual_collection": interval("scan_start", "scan_end"),
            "comparison_price_and_outbox_storage": interval("scan_end", "storage_end") if reached else None,
        },
        "program_run_apply_total_seconds": interval("start", "end"),
        "docker_startup_included_in_boundary": False,
        "target_count": captured.get("target_count"), "confirmed_count": captured.get("confirmed_count"),
        "unconfirmed_count": captured.get("unconfirmed_count"),
        "local_price_commit_count": captured.get("updated_count", 0),
        "local_outbox_record_count": captured.get("outbox_record_count", 0),
        "local_price_failure_count": captured.get("failed_count", 0),
        "local_commit_unknown_count": captured.get("unknown_count", 0),
        "readiness_issues": captured.get("readiness_issues", []),
        "network": stats,
        "session_connection_closed": session.connection is None or session.connection.closed,
    }


def inspect_local_db(url):
    """Read-only metadata/counts, no DDL/DML, no raw driver errors or DSN output."""
    import psycopg
    from psycopg.conninfo import conninfo_to_dict
    config = conninfo_to_dict(url)
    if (os.environ.get("PRICE_REFRESH_TIMING_LOCAL_DB_CONFIRMED") != "1"
            or config.get("host") != "host.docker.internal"
            or any(key in config for key in ("hostaddr", "service", "servicefile"))):
        raise ValueError("local_db_not_verified")
    with psycopg.connect(url, connect_timeout=5, autocommit=False) as connection:
        connection.read_only = True
        with connection.cursor() as cursor:
            cursor.execute("SELECT to_regclass('products')::oid, inet_server_port(), version()")
            table_oid, port, version = cursor.fetchone()
            if table_oid is None:
                return {"ready": False, "reason": "products_table_missing"}
            if port != int(config.get("port", "5432")):
                raise ValueError("local_db_port_mismatch")
            cursor.execute("SELECT attname, atttypid::regtype::text FROM pg_attribute "
                           "WHERE attrelid=%s AND attnum>0 AND NOT attisdropped", (table_oid,))
            columns = dict(cursor.fetchall())
            required = {"product_code": "bigint", "price": "integer", "has_been_recommended": "boolean"}
            missing = [key for key, value in required.items() if columns.get(key) != value]
            result = {"ready": False, "connection_read_only": connection.read_only,
                      "postgresql_version": version.split(" on ")[0],
                      "required_columns": {key: columns.get(key) for key in required},
                      "missing_or_wrong_type_columns": missing}
            # Counts that do not depend on recommendation metadata remain useful
            # when the recommendation column is absent. Never synthesize FALSE.
            cursor.execute("SELECT has_table_privilege(%s, 'SELECT')", (table_oid,))
            if cursor.fetchone()[0] and all(columns.get(key) == required[key]
                                           for key in ("product_code", "price")):
                cursor.execute("SELECT count(*), count(DISTINCT product_code) FROM products")
                result["product_rows"], result["unique_codes"] = cursor.fetchone()
            if missing:
                result["reason"] = "required_columns_unavailable"
                return result
            cursor.execute("SELECT has_table_privilege(%s, 'SELECT'), "
                           "has_column_privilege(%s, 'price', 'UPDATE'), "
                           "has_table_privilege('pg_catalog.pg_locks', 'SELECT'), "
                           "has_function_privilege('pg_try_advisory_lock(bigint)', 'EXECUTE')",
                           (table_oid, table_oid))
            result["required_permissions_available"] = all(cursor.fetchone())
            if not result["required_permissions_available"]:
                result["reason"] = "required_permissions_missing"
                return result
            cursor.execute("SELECT count(*), count(DISTINCT product_code), "
                           "count(*) FILTER (WHERE has_been_recommended IS TRUE), "
                           "count(*) FILTER (WHERE has_been_recommended IS NULL) FROM products")
            count, unique, recommended, null_flags = cursor.fetchone()
            result.update(product_rows=count, unique_codes=unique, recommended_rows=recommended,
                          null_recommendation_rows=null_flags, ready=count == unique and null_flags == 0)
            if not result["ready"]:
                result["reason"] = "ambiguous_product_rows"
            return result


def live_measure(url, scope):
    settings = Settings.from_env()
    if asdict(settings) != asdict(Settings()):
        raise ValueError("measurement_requires_unchanged_default_policy")
    session = PostgresApplySession(DatabaseSettings(url))
    client = TwentyNineCmClient(settings)
    categories = CATEGORIES[:1] if scope == "first" else CATEGORIES
    stats = new_stats()
    original_fetch = client.fetch_listing

    def fetch(**kwargs):
        stats["listing_fetch_attempts"] += 1
        try:
            return original_fetch(**kwargs)
        except Exception as error:
            stats["listing_error_count"] += 1
            stats["failure"] = {"stage": "listing_fetch", "exception_type": type(error).__name__,
                                "sort": kwargs["sort"], "page": kwargs["page"], "small_id": kwargs["smallId"]}
            raise TimingAbort("listing_fetch_failed") from None

    def parse(payload):
        try:
            return listing_items(payload)
        except Exception as error:
            stats["listing_error_count"] += 1
            stats["failure"] = {"stage": "listing_response_parser", "exception_type": type(error).__name__}
            raise TimingAbort("listing_response_failed") from None

    with block_backend_network(stats, allow_listing=True), \
            patch.object(client, "fetch_listing", fetch), \
            patch("price_refresh.scanner.listing_items", parse):
        result = measure(session, lambda: collect_prices(client, categories, settings), stats=stats)
    result.update(scope=scope, configured_categories=len(categories), request_policy=asdict(settings))
    return result


def read_host_db_config(filename):
    values = {}
    for line in filename.read_text().splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            key, value = line.split("=", 1)
            values[key.strip().removeprefix("export ")] = value.strip().strip("\"'")
    raw = values.get("DATABASE_URL", "")
    parts = urlsplit(raw)
    if (parts.scheme not in {"postgres", "postgresql"} or parts.hostname not in {"localhost", "127.0.0.1", "::1"}
            or any(key.lower() in {"host", "hostaddr", "port", "service", "servicefile", "dbname"}
                   for key, _ in parse_qsl(parts.query))):
        raise ValueError("host_db_config_not_loopback")
    port = parts.port or 5432
    check = subprocess.run(["/usr/sbin/lsof", "-nP", "-iTCP:" + str(port), "-sTCP:LISTEN", "-Fpc"],
                           capture_output=True, text=True)
    listeners = [line[1:] for line in check.stdout.splitlines() if line.startswith("c")]
    if check.returncode or not listeners or any("postgres" not in name.lower() for name in listeners):
        raise ValueError("local_postgresql_listener_not_verified")
    userinfo = parts.netloc.rpartition("@")[0] + "@" if "@" in parts.netloc else ""
    remapped = urlunsplit((parts.scheme, userinfo + "host.docker.internal:" + str(port), parts.path,
                          parts.query, parts.fragment))
    return remapped


def protected_hashes():
    files = list((ROOT / "price_refresh").rglob("*.py"))
    files += [ROOT / name for name in ("README.md", "Dockerfile", "requirements.txt", ".env.example")]
    return {str(file.relative_to(ROOT)): hashlib.sha256(file.read_bytes()).hexdigest() for file in files}


def docker_run(args):
    url = read_host_db_config(args.db_env_file)
    environment = dict(os.environ, PRICE_REFRESH_TIMING_DATABASE_URL=url,
                       PRICE_REFRESH_TIMING_LOCAL_DB_CONFIRMED="1")
    name = "price-refresh-timing-" + args.scope + "-" + str(os.getpid())
    command = ["docker", "run", "--rm", "--name", name, "--platform", "linux/arm64", "--network", "bridge",
               "--read-only", "--mount", f"type=bind,src={ROOT / 'price_refresh'},dst=/app/price_refresh,readonly",
               "--mount", f"type=bind,src={ROOT / 'tests'},dst=/app/tests,readonly",
               "--env", "PRICE_REFRESH_TIMING_DATABASE_URL", "--env", "PRICE_REFRESH_TIMING_LOCAL_DB_CONFIRMED",
               IMAGE, "tests/manual_timing.py", "--scope", args.scope, "--live" if args.live else "--preflight"]
    baseline = protected_hashes()
    started = time.perf_counter()
    try:
        process = subprocess.run(command, env=environment, stdout=subprocess.PIPE, text=True)
    except BaseException:
        subprocess.run(["docker", "rm", "-f", name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        raise
    total = time.perf_counter() - started
    if not process.stdout.strip():
        raise ValueError("measurement_container_no_result")
    result = json.loads(process.stdout)
    result.update(container_wall_seconds=round(total, 6), docker_network="bridge",
                  host_local_postgresql_listener_verified=True,
                  implementation_and_configuration_unchanged=baseline == protected_hashes())
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return process.returncode


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--docker", action="store_true")
    parser.add_argument("--db-env-file", type=Path, default=ROOT.parent / ".env")
    parser.add_argument("--scope", choices=("first", "all"), default="first")
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--preflight", action="store_true")
    modes.add_argument("--live", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.docker:
            return docker_run(args)
        configure_logging()
        url = os.environ.get("PRICE_REFRESH_TIMING_DATABASE_URL", "")
        inspection = inspect_local_db(url)
        result = {"db_preflight": inspection, "python": platform.python_version(),
                  "tls": ssl.OPENSSL_VERSION, "architecture": platform.machine()}
        if not inspection["ready"] or args.preflight:
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0 if inspection["ready"] else 1
        result["measurement"] = live_measure(url, args.scope)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        measurement = result["measurement"]
        return 0 if (not measurement["measurement_aborted"]
                     and measurement["run_apply_status"] == "completed"
                     and measurement["network"]["blocked_http_attempts"] == 0) else 1
    except Exception as error:
        # Never echo an exception message: libpq/URL/parser errors may contain a DSN.
        print(json.dumps({"status": "blocked_or_failed", "exception_type": type(error).__name__}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
