"""Opt-in four-case LOCAL DB test; restore all sampled prices and flags.

Run from price-refresh-crawler:
  python3 -B tests/manual_recommendation_filter.py --docker --run-local
Never discovers or runs automatically with unittest. No external HTTP is allowed.
"""

import argparse
import http.client
import json
import os
import subprocess
import sys
import tempfile
import time
from contextlib import ExitStack, contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from manual_timing import IMAGE, inspect_local_db, protected_hashes, read_host_db_config
from price_refresh.apply import run_apply
from price_refresh.backend import BackendClient, BackendSettings, ENDPOINT, SUCCESS
from price_refresh.delivery import BackendDelivery
from price_refresh.logging_config import configure_logging
from price_refresh.repository import DatabaseSettings
from price_refresh.scanner import ScanResult
from price_refresh.writer import PostgresApplySession


def require(condition, message):
    if not condition:
        raise AssertionError(message)


@contextmanager
def deny_real_http(attempts):
    def deny(*_args, **_kwargs):
        attempts.append("blocked_real_http")
        raise AssertionError("real_http_forbidden")

    with ExitStack() as stack:
        for cls, method in ((http.client.HTTPConnection, "connect"),
                            (http.client.HTTPSConnection, "connect"),
                            (http.client.HTTPConnection, "request")):
            stack.enter_context(patch.object(cls, method, deny))
        yield


class RecordingConnection:
    """In-memory transport; the REAL BackendClient still serializes/parses JSON."""

    def __init__(self, session, requests):
        self.session, self.requests = session, requests
        self.closed = False
        self.read_timeout = None
        self.sock = SimpleNamespace(settimeout=self.settimeout)

    def settimeout(self, value):
        self.read_timeout = value

    def connect(self):
        self.session.assert_delivery_ready()

    def request(self, method, path, *, body, headers):
        self.session.assert_delivery_ready()
        with self.session.connection.cursor() as cursor:
            cursor.execute("SELECT product_code, price, has_been_recommended FROM products "
                           "WHERE product_code = ANY(%s) ORDER BY product_code",
                           ([int(row["product_code"]) for row in self.session.originals],))
            actual = [(str(code), price, flag) for code, price, flag in cursor.fetchall()]
        expected = [(row["product_code"], row["price"] + (1 if index < 2 else 0), index in (0, 2))
                    for index, row in enumerate(self.session.originals)]
        require(actual == expected, "unexpected_database_state_before_send")
        require(self.read_timeout == 20, "unexpected_read_timeout")
        require(headers.get("Content-Type") == "application/json", "missing_json_header")
        self.body = json.loads(body.decode("utf-8"))
        self.requests.append({"method": method, "path": path, "body": self.body})

    def getresponse(self):
        payload = {"code": SUCCESS, "message": "SIMULATED test response", "data": {
            "received_count": len(self.body["products"]),
            "failed_product_codes": [], "not_found_product_codes": []}}

        class Response:
            status = 200

            def __enter__(self):
                return self

            def __exit__(self, *_):
                pass

            def read(self):
                return json.dumps(payload).encode("utf-8")

        return Response()

    def close(self):
        self.closed = True


class RestoringLocalSession(PostgresApplySession):
    """Only test setup/cleanup differ; real reads, writer and locks are retained."""

    def __init__(self, url, backup_file):
        super().__init__(DatabaseSettings(url))
        self.backup_file = backup_file
        self.originals = []
        self.restored = False
        self.restore_error = None

    def open(self):
        if not super().open():
            return False
        with self.connection.transaction():
            with self.connection.cursor() as cursor:
                cursor.execute("SELECT product_code, price, has_been_recommended FROM products "
                               "WHERE price >= 0 AND price < 2147483647 "
                               "AND has_been_recommended IS NOT NULL "
                               "ORDER BY product_code LIMIT 4 FOR UPDATE")
                rows = cursor.fetchall()
                require(len(rows) == 4, "four_existing_products_required")
                self.originals = [{"product_code": str(code), "price": price,
                                   "has_been_recommended": flag} for code, price, flag in rows]
                # Persist a credential-free recovery snapshot BEFORE any test write.
                with self.backup_file.open("x", encoding="utf-8") as output:
                    json.dump(self.originals, output, ensure_ascii=False, indent=2)
                    output.flush()
                    os.fsync(output.fileno())
                for index, row in enumerate(self.originals):
                    cursor.execute("UPDATE products SET has_been_recommended = %s "
                                   "WHERE product_code = %s", (index in (0, 2), int(row["product_code"])))
                    require(cursor.rowcount == 1, "unexpected_setup_row_count")
        return True

    def simulated_collection(self):
        require(len(self.originals) == 4, "setup_not_complete")
        return ScanResult(prices={row["product_code"]: row["price"] + (1 if index < 2 else 0)
                                  for index, row in enumerate(self.originals)})

    def close(self):
        try:
            if self.originals and not self.restored:
                self.assert_lock()
                with self.connection.transaction():
                    with self.connection.cursor() as cursor:
                        for index, row in enumerate(self.originals):
                            cursor.execute("SELECT price, has_been_recommended FROM products "
                                           "WHERE product_code = %s FOR UPDATE", (int(row["product_code"]),))
                            current = cursor.fetchone()
                            expected_price = row["price"] + (1 if index < 2 else 0)
                            require(current is not None and current[0] in (row["price"], expected_price),
                                    "unexpected_price_during_restore")
                            cursor.execute("UPDATE products SET price = %s, has_been_recommended = %s "
                                           "WHERE product_code = %s",
                                           (row["price"], row["has_been_recommended"], int(row["product_code"])))
                        codes = [int(row["product_code"]) for row in self.originals]
                        cursor.execute("SELECT product_code, price, has_been_recommended FROM products "
                                       "WHERE product_code = ANY(%s) ORDER BY product_code", (codes,))
                        restored = [(str(code), price, flag) for code, price, flag in cursor.fetchall()]
                        expected = [(row["product_code"], row["price"], row["has_been_recommended"])
                                    for row in self.originals]
                        require(restored == expected, "restore_verification_failed")
                self.restored = True  # Only after confirmed COMMIT.
        except Exception:
            self.restore_error = "restore_failed_see_original_rows_json"
        finally:
            super().close()


def run_local(url, result_dir):
    inspection = inspect_local_db(url)
    require(inspection["ready"], "local_db_preflight_failed")
    session = RestoringLocalSession(url, result_dir / "original_rows.json")
    requests, connections, network_attempts = [], [], []

    def factory(host, *, port, timeout):
        require((host, port, timeout) == ("backend-disabled.invalid", None, 5), "unexpected_backend_config")
        connection = RecordingConnection(session, requests)
        connections.append(connection)
        return connection

    client = BackendClient(BackendSettings("https://backend-disabled.invalid", {"X-Test-Only": "local"}),
                           connection_factory=factory)
    started = time.perf_counter()
    with deny_real_http(network_attempts):
        report = run_apply(session, session.simulated_collection, delivery=BackendDelivery(client))
    require(session.restored, session.restore_error or "original_rows_not_restored")
    require(session.connection.closed and not session.acquired, "session_not_closed")
    require(report.status == "completed", "apply_not_completed")
    a, b, c, d = session.originals
    expected_product = {"product_code": a["product_code"], "old_price": a["price"],
                        "new_price": a["price"] + 1}
    expected_request = {"method": "PATCH", "path": ENDPOINT,
                        "body": {"changed_count": 1, "products": [expected_product]}}
    require(requests == [expected_request], "unexpected_request_body_or_count")
    require({change.product_code for change in report.saved_changes} == {a["product_code"], b["product_code"]},
            "unexpected_saved_products")
    require(report.backend["target_count"] == 1 and report.backend["retry_request_count"] == 0,
            "unexpected_delivery_result")
    require(not network_attempts and all(connection.closed for connection in connections),
            "network_or_cleanup_failure")
    names = ("changed_recommended", "changed_not_recommended", "same_recommended", "same_not_recommended")
    cases = [{"case": name, "product_code": row["product_code"], "test_recommended": index in (0, 2),
              "price_changed": index < 2, "saved": index < 2, "included_in_request": index == 0}
             for index, (name, row) in enumerate(zip(names, session.originals))]
    return {"status": "passed", "scope": "local_db_functional_test_not_crawl_timing",
            "cases": cases, "local_price_commits_before_restore": len(report.saved_changes),
            "recommended_delivery_target_count": report.backend["target_count"],
            "captured_requests": requests, "acknowledgements_are_simulated": True,
            "actual_29cm_requests": 0, "actual_backend_http_requests": 0,
            "blocked_http_attempts": len(network_attempts), "original_prices_and_flags_restored": session.restored,
            "session_closed": session.connection.closed, "lock_released": not session.acquired,
            "functional_test_seconds": round(time.perf_counter() - started, 6)}


def docker_run(args):
    url = read_host_db_config(args.db_env_file)
    result_dir = Path(tempfile.mkdtemp(prefix="price-refresh-recommendation-", dir="/private/tmp"))
    baseline = protected_hashes()
    env = dict(os.environ, PRICE_REFRESH_TIMING_DATABASE_URL=url,
               PRICE_REFRESH_TIMING_LOCAL_DB_CONFIRMED="1")
    name = "price-refresh-recommendation-" + str(os.getpid())
    command = ["docker", "run", "--rm", "--name", name, "--platform", "linux/arm64", "--network", "bridge",
               "--read-only", "--tmpfs", "/tmp:rw,nosuid,nodev,size=64m", "--env", "PYTHONDONTWRITEBYTECODE=1",
               "--mount", f"type=bind,src={ROOT / 'price_refresh'},dst=/app/price_refresh,readonly",
               "--mount", f"type=bind,src={ROOT / 'tests'},dst=/app/tests,readonly",
               "--mount", f"type=bind,src={result_dir},dst=/artifacts",
               "--env", "PRICE_REFRESH_TIMING_DATABASE_URL", "--env", "PRICE_REFRESH_TIMING_LOCAL_DB_CONFIRMED",
               IMAGE, "tests/manual_recommendation_filter.py", "--run-local", "--result-dir", "/artifacts"]
    try:
        process = subprocess.run(command, env=env, stdout=subprocess.PIPE, text=True)
    except BaseException:
        subprocess.run(["docker", "rm", "-f", name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        raise
    result = json.loads(process.stdout) if process.stdout.strip() else {"status": "container_no_result"}
    result.update(implementation_and_configuration_unchanged=baseline == protected_hashes(),
                  recovery_snapshot=str(result_dir / "original_rows.json"))
    (result_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return process.returncode


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--docker", action="store_true")
    parser.add_argument("--run-local", action="store_true", required=True)
    parser.add_argument("--db-env-file", type=Path, default=ROOT.parent / ".env")
    parser.add_argument("--result-dir", type=Path, default=Path("/tmp"))
    args = parser.parse_args(argv)
    try:
        if args.docker:
            return docker_run(args)
        configure_logging()
        result = run_local(os.environ.get("PRICE_REFRESH_TIMING_DATABASE_URL", ""), args.result_dir)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except Exception as error:
        # No raw DB/URL exception messages or credentials.
        print(json.dumps({"status": "failed", "exception_type": type(error).__name__}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
