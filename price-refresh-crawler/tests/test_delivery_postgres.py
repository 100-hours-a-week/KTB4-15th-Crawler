"""Opt-in outbox delivery runs against the disposable test database and a fake/loopback backend.

Never reads DATABASE_URL and never reaches a real backend. Time is injected; nothing sleeps.
These are simulations: they do not prove every ordering or failure of a real remote backend.
"""

import json
import os
import threading
import unittest
import uuid
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from unittest.mock import patch

from price_refresh import outbox
from price_refresh.apply import run_apply
from price_refresh.backend import BackendClient, BackendError, BackendSettings, HttpResult
from price_refresh.delivery import DeliverySettings, run_delivery
from price_refresh.delivery_control import fetch_delivery_status, run_resume
from price_refresh.outbox import ATTEMPT_UNCONFIRMED, AUTH_FAILED
from price_refresh.repository import DatabaseSettings
from price_refresh.scanner import ScanResult
from price_refresh.writer import DeliverySession, PostgresApplySession

TEST_URL = os.getenv("PRICE_REFRESH_TEST_DATABASE_URL", "")
SQL = Path(__file__).resolve().parents[1] / "sql" / "create_price_change_outbox.sql"
T0 = datetime(2026, 10, 1, 2, 0, tzinfo=timezone.utc)
SUCCESS, PARTIAL = "PRODUCT_PRICE_UPDATE_SUCCESS", "PRODUCT_PRICE_UPDATE_PARTIAL_SUCCESS"


class Crash(BaseException):
    """Simulated process death; not swallowed by ordinary error handling."""


class FakeBackend:
    """Applies the agreed BE rules to its own price table; may be scripted per request."""

    def __init__(self, prices=None, *, batch_size=100, script=(), on_send=None):
        self.prices = dict(prices or {})
        self.settings = BackendSettings("https://backend.invalid", {"X-Test-Only": "test-value"}, batch_size)
        self.script, self.on_send, self.calls = list(script), on_send, []

    def apply(self, body):
        not_found, conflicts = [], []
        for product in body["products"]:
            code, old, new = product["product_code"], product["old_price"], product["new_price"]
            if code not in self.prices:
                not_found.append(code)
            elif self.prices[code] == old:
                self.prices[code] = new
            elif self.prices[code] != new:
                conflicts.append({"product_code": code, "current_price": self.prices[code]})
        return {"code": PARTIAL if not_found or conflicts else SUCCESS, "message": "test",
                "data": {"received_count": len(body["products"]), "not_found_product_codes": not_found,
                         "price_conflicts": conflicts}}

    def send(self, products):
        body = {"changed_count": len(products), "products": json.loads(json.dumps(products))}
        self.calls.append(body)
        if self.on_send:
            self.on_send(body)
        step = self.script.pop(0) if self.script else None
        if isinstance(step, BaseException):
            raise step
        if callable(step):
            return step(body)
        if isinstance(step, HttpResult):
            return step
        return HttpResult(200, self.apply(body))


@unittest.skipUnless(TEST_URL, "requires isolated PRICE_REFRESH_TEST_DATABASE_URL; never DATABASE_URL")
class PostgresDeliveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import psycopg
        from psycopg.conninfo import conninfo_to_dict
        config = conninfo_to_dict(TEST_URL)
        if (config.get("dbname") != "price_refresh_test"
                or config.get("host") != "price-refresh-test-db"):
            raise RuntimeError("Refusing integration tests outside the dedicated Docker test database")
        cls.psycopg = psycopg

    def setUp(self):
        self.admin = self.psycopg.connect(TEST_URL, autocommit=True, connect_timeout=5)
        self.addCleanup(self.admin.close)
        self.admin.execute("DROP TABLE IF EXISTS price_change_outbox, products CASCADE")
        self.admin.execute(
            "CREATE TABLE products (id BIGSERIAL PRIMARY KEY, product_code BIGINT NOT NULL UNIQUE, "
            "price INTEGER NOT NULL, has_been_recommended BOOLEAN NOT NULL DEFAULT FALSE)")
        self.admin.execute("INSERT INTO products (product_code, price) "
                           "SELECT code, 1000 FROM generate_series(1, 400) AS code")
        self.admin.execute(SQL.read_text(encoding="utf-8"))
        self.database = DatabaseSettings(TEST_URL)

    def add(self, code, old=100, new=200, *, status="PENDING", next_at=None):
        return self.admin.execute(
            "INSERT INTO price_change_outbox (scan_run_id, product_code, original_old_price, old_price, "
            "new_price, status, next_attempt_at) VALUES (%s, %s, %s, %s, %s, %s, %s) RETURNING id",
            (uuid.uuid4(), code, old, old, new, status,
             None if status in ("BLOCKED", "NOT_FOUND") else next_at or T0 - timedelta(minutes=1))).fetchone()[0]

    def rows(self):
        return {row[0]: outbox.record_from_row(row) for row in self.admin.execute(
            f"SELECT {', '.join(outbox.COLUMNS)} FROM price_change_outbox ORDER BY id").fetchall()}

    def status(self, record_id):
        return self.rows()[record_id].status

    def deliver(self, backend, *, now=T0, settings=None, session=None):
        session = session or DeliverySession(self.database)
        self.addCleanup(session.close)
        self.session = session
        emitted = []
        report = run_delivery(session, backend, settings or DeliverySettings(), clock=lambda: now,
                              emit=emitted.append)
        self.assertEqual(emitted[0]["status"], report.status)
        return report

    def products_unchanged(self):
        self.assertEqual(self.admin.execute("SELECT count(*) FROM products WHERE price <> 1000").fetchone()[0], 0)

    # Basic delivery

    def test_97_success_2_conflicts_1_not_found_in_one_request(self):
        ids = {code: self.add(code) for code in range(1, 101)}
        prices = {str(code): 100 for code in range(1, 101)}
        prices.update({"10": 150, "20": 160})
        del prices["30"]
        backend = FakeBackend(prices)
        report = self.deliver(backend)
        self.assertEqual(len(backend.calls), 1)
        self.assertEqual(backend.calls[0]["changed_count"], 100)
        self.assertEqual((report.status, report.selected_count, report.attempted_count, report.http_request_count,
                          report.succeeded_count, report.conflict_retry_count, report.not_found_count),
                         ("completed", 100, 100, 1, 97, 2, 1))
        rows = self.rows()
        self.assertEqual(rows[ids[30]].status, "NOT_FOUND")
        for code, price in ((10, 150), (20, 160)):
            record = rows[ids[code]]
            self.assertEqual((record.status, record.original_old_price, record.old_price, record.new_price,
                              record.last_backend_price, record.attempt_count, record.next_attempt_at),
                             ("RETRY_WAIT", 100, price, 200, price, 1, T0 + timedelta(seconds=300)))
        self.assertEqual(sum(r.status == "DONE" for r in rows.values()), 97)
        self.products_unchanged()
        # Conflicts are retried with the adjusted old_price in a later run, then succeed.
        later = self.deliver(backend, now=T0 + timedelta(seconds=300))
        self.assertEqual(later.succeeded_count, 2)
        self.assertEqual([p["old_price"] for p in backend.calls[1]["products"]], [150, 160])

    def test_multiple_batches_limit_and_remaining_records_kept(self):
        for code in range(1, 251):
            self.add(code)
        backend = FakeBackend({str(code): 100 for code in range(1, 251)})
        report = self.deliver(backend, settings=DeliverySettings(max_requests=2))
        self.assertEqual([len(call["products"]) for call in backend.calls], [100, 100])
        self.assertEqual((report.status, report.batch_limit_reached, report.succeeded_count), ("completed", True, 200))
        self.assertEqual(sum(r.status == "PENDING" for r in self.rows().values()), 50)
        rest = self.deliver(backend)
        self.assertEqual((rest.succeeded_count, rest.batch_limit_reached), (50, False))
        self.assertEqual([len(call["products"]) for call in backend.calls], [100, 100, 50])

    def test_invalid_records_are_blocked_and_excluded(self):
        bad = self.add(1, old=0)
        good = self.add(2)
        backend = FakeBackend({"2": 100})
        report = self.deliver(backend)
        self.assertEqual([p["product_code"] for p in backend.calls[0]["products"]], ["2"])
        self.assertEqual(backend.calls[0]["changed_count"], 1)
        rows = self.rows()
        self.assertEqual((rows[bad].status, rows[bad].last_error_code, rows[bad].attempt_count),
                         ("BLOCKED", "LOCAL_INVALID_PRICE", 0))
        self.assertEqual(rows[good].status, "DONE")
        self.assertEqual((report.status, report.invalid_blocked_count), ("completed_with_errors", 1))

    def test_all_invalid_means_no_http(self):
        self.add(1, old=0)
        self.add(2, old=0, new=5)
        backend = FakeBackend()
        report = self.deliver(backend)
        self.assertEqual((backend.calls, report.http_request_count, report.invalid_blocked_count), ([], 0, 2))

    def test_attempt_mark_mismatch_rolls_back_and_sends_nothing(self):
        ids = [self.add(1), self.add(2)]
        original = outbox.mark_attempts
        with patch("price_refresh.delivery.mark_attempts", lambda *a, **k: original(*a, **k)[:1]):
            backend = FakeBackend({"1": 100, "2": 100})
            report = self.deliver(backend)
        self.assertEqual((backend.calls, report.status, report.error), ([], "failed", "attempt_not_recorded"))
        self.assertEqual([self.rows()[i].attempt_count for i in ids], [0, 0])

    def test_same_prepared_batch_is_sent_and_parsed(self):
        self.add(1)
        seen = []
        from price_refresh import delivery
        original = delivery.parse_sync_response

        def spy(response, batch):
            seen.append(batch)
            return original(response, batch)

        backend = FakeBackend({"1": 100})
        with patch("price_refresh.delivery.parse_sync_response", spy):
            self.deliver(backend)
        self.assertEqual(seen[0].body(), backend.calls[0])

    # Per-product order

    def test_product_order_across_runs_and_independence(self):
        a1, a2 = self.add(1, 100, 200), self.add(1, 200, 300)
        b1 = self.add(2)
        held, after_held = self.add(3, status="BLOCKED"), self.add(3)
        gone, after_gone = self.add(4, status="NOT_FOUND"), self.add(4)
        backend = FakeBackend({"1": 100, "2": 100, "4": 100})
        report = self.deliver(backend)
        sent = [[p["product_code"] for p in call["products"]] for call in backend.calls]
        self.assertEqual(sent, [["1", "2", "4"], ["1"]])     # A's next change only after A's first is DONE
        self.assertEqual(backend.calls[1]["products"][0]["old_price"], 200)
        rows = self.rows()
        self.assertEqual([rows[i].status for i in (a1, a2, b1, after_gone)], ["DONE"] * 4)
        self.assertEqual((rows[held].status, rows[after_held].status), ("BLOCKED", "PENDING"))
        self.assertEqual(report.http_request_count, 2)

    def test_timeout_holds_later_record_until_retried(self):
        first, second = self.add(1, 100, 200), self.add(1, 200, 300)
        backend = FakeBackend({"1": 100}, script=[BackendError("backend_timeout", phase="response")])
        report = self.deliver(backend)
        self.assertEqual((len(backend.calls), report.status, report.error_retry_count), (1, "completed_with_errors", 1))
        self.assertEqual(self.rows()[first].last_error_code, "BACKEND_TIMEOUT")
        early = self.deliver(backend, now=T0 + timedelta(seconds=299))
        self.assertEqual((early.http_request_count, len(backend.calls)), (0, 1))
        self.deliver(backend, now=T0 + timedelta(seconds=300))
        self.assertEqual([call["products"][0]["new_price"] for call in backend.calls], [200, 200, 300])
        self.assertEqual([self.status(i) for i in (first, second)], ["DONE", "DONE"])

    # Errors and stops

    def test_server_error_and_invalid_response_keep_prices_and_retry_later(self):
        record = self.add(1)
        backend = FakeBackend({"1": 100}, script=[
            HttpResult(500, {"code": "INTERNAL_SERVER_ERROR", "message": "m", "data": None}),
            HttpResult(200, {"code": SUCCESS, "message": "m", "data": {"received_count": 1}}),
            HttpResult(200, {"code": PARTIAL, "message": "m", "data": {
                "received_count": 1, "not_found_product_codes": [],
                "price_conflicts": [{"product_code": "1", "current_price": 100}]}}),
        ])
        for minutes, code in ((0, "SERVER_ERROR"), (5, "INVALID_RESPONSE_FIELDS"), (15, "INVALID_RESPONSE_PRICE")):
            report = self.deliver(backend, now=T0 + timedelta(minutes=minutes))
            row = self.rows()[record]
            self.assertEqual((report.status, report.succeeded_count, row.status, row.last_error_code, row.old_price),
                             ("completed_with_errors", 0, "RETRY_WAIT", code, 100))
        self.assertEqual(self.rows()[record].attempt_count, 3)
        self.assertEqual(self.deliver(backend, now=T0 + timedelta(minutes=35)).succeeded_count, 1)

    def test_request_rejected_blocks_only_that_request(self):
        first = [self.add(code) for code in (1, 2)]
        second = [self.add(code) for code in (3, 4)]
        backend = FakeBackend({str(c): 100 for c in range(1, 5)}, batch_size=2, script=[
            HttpResult(400, {"code": "INVALID_REQUEST", "message": "m", "data": None})])
        report = self.deliver(backend)
        self.assertEqual(len(backend.calls), 2)
        rows = self.rows()
        self.assertEqual([rows[i].status for i in first], ["BLOCKED", "BLOCKED"])
        self.assertEqual([rows[i].last_error_code for i in first], ["INVALID_REQUEST", "INVALID_REQUEST"])
        self.assertEqual([rows[i].status for i in second], ["DONE", "DONE"])
        self.assertEqual((report.status, report.request_rejected_blocked_count), ("completed_with_errors", 2))
        self.assertFalse(fetch_delivery_status(self.database).stopped)

    def test_auth_failure_stops_now_and_in_later_runs_until_resumed(self):
        first = [self.add(code) for code in (1, 2)]
        later = [self.add(code) for code in (3, 4)]
        backend = FakeBackend({str(c): 100 for c in range(1, 6)}, batch_size=2,
                              script=[HttpResult(401, "<html>denied</html>")])
        report = self.deliver(backend)
        self.assertEqual((len(backend.calls), report.status, report.auth_stop_saved), (1, "auth_stopped", True))
        rows = self.rows()
        self.assertEqual([(rows[i].status, rows[i].last_error_code, rows[i].last_http_status) for i in first],
                         [("RETRY_WAIT", AUTH_FAILED, 401)] * 2)
        self.assertEqual([rows[i].status for i in later], ["PENDING", "PENDING"])
        self.add(5)   # a new day's record cannot bypass the stop
        for minutes in (10, 600):
            again = self.deliver(backend, now=T0 + timedelta(minutes=minutes))
            self.assertEqual((again.status, again.http_request_count, len(backend.calls)), ("auth_stopped", 0, 1))
        resume_session = DeliverySession(self.database)
        self.addCleanup(resume_session.close)
        self.assertEqual(run_resume(resume_session, now=T0 + timedelta(hours=1)).status, "resumed")
        resumed = self.deliver(backend, now=T0 + timedelta(hours=1))
        self.assertEqual((resumed.status, resumed.succeeded_count), ("completed", 5))

    def test_auth_failure_not_saved_still_stops_sending(self):
        for code in (1, 2, 3, 4):
            self.add(code)
        backend = FakeBackend(batch_size=2, script=[BackendError("backend_invalid_json", phase="json", http_status=403)])
        with patch("price_refresh.delivery.mark_auth_failed", side_effect=outbox.OutboxError("auth_failure_not_recorded")):
            report = self.deliver(backend)
        self.assertEqual((len(backend.calls), report.status, report.auth_stop_saved, report.error),
                         (1, "auth_stop_not_saved", False, "auth_failure_not_recorded"))
        self.assertFalse(fetch_delivery_status(self.database).stopped)   # no persistent stop is claimed
        self.assertEqual(sum(r.last_error_code == ATTEMPT_UNCONFIRMED for r in self.rows().values()), 2)

    def test_result_save_failure_stops_and_rolls_back_that_result(self):
        ids = [self.add(code) for code in (1, 2, 3, 4)]
        backend = FakeBackend({str(c): 100 for c in range(1, 5)}, batch_size=2)
        original = outbox.mark_done
        calls = []

        def second_fails(*args, **kwargs):
            calls.append(1)
            return None if len(calls) == 2 else original(*args, **kwargs)

        with patch("price_refresh.delivery.mark_done", second_fails):
            report = self.deliver(backend)
        self.assertEqual((len(backend.calls), report.status, report.result_save_failed, report.error),
                         (1, "failed", True, "result_not_saved"))
        rows = self.rows()
        self.assertEqual([rows[i].last_error_code for i in ids[:2]], [ATTEMPT_UNCONFIRMED] * 2)  # rolled back
        self.assertEqual([rows[i].status for i in ids[2:]], ["PENDING", "PENDING"])
        retried = self.deliver(backend, now=T0 + timedelta(seconds=300))   # BE already has new_price
        self.assertEqual((retried.status, retried.succeeded_count), ("completed", 4))

    # Locks, transactions, crashes

    def test_competing_delivery_or_resume_lock_skips_but_collection_lock_does_not(self):
        self.add(1)
        holder = DeliverySession(self.database)
        self.addCleanup(holder.close)
        self.assertTrue(holder.open())
        backend = FakeBackend({"1": 100})
        self.assertEqual(self.deliver(backend).status, "skipped_concurrent")
        self.assertEqual(run_resume(DeliverySession(self.database), now=T0).status, "skipped_concurrent")
        holder.close()
        collector = PostgresApplySession(self.database)
        self.addCleanup(collector.close)
        self.assertTrue(collector.open())
        self.assertEqual(self.deliver(backend).succeeded_count, 1)
        collector.assert_lock()

    def test_http_happens_with_committed_attempt_and_no_open_transaction(self):
        record = self.add(1)

        def check(_body):
            self.assertEqual(self.session.connection.info.transaction_status,
                             self.psycopg.pq.TransactionStatus.IDLE)
            row = self.rows()[record]
            self.assertEqual((row.attempt_count, row.last_error_code), (1, ATTEMPT_UNCONFIRMED))

        backend = FakeBackend({"1": 100}, on_send=check)
        self.assertEqual(self.deliver(backend).succeeded_count, 1)

    def test_lost_session_during_http_stops_and_keeps_attempt(self):
        ids = [self.add(code) for code in (1, 2, 3, 4)]
        backend = FakeBackend({str(c): 100 for c in range(1, 5)}, batch_size=2,
                              on_send=lambda _: self.admin.execute("SELECT pg_terminate_backend(%s)",
                                                                   (self.session.backend_pid,)))
        report = self.deliver(backend)
        self.assertEqual(len(backend.calls), 1)
        self.assertIn(report.status, {"aborted", "failed"})
        rows = self.rows()
        self.assertEqual([rows[i].last_error_code for i in ids[:2]], [ATTEMPT_UNCONFIRMED] * 2)
        self.assertEqual([rows[i].status for i in ids[2:]], ["PENDING", "PENDING"])

    def test_crash_after_attempt_before_http_is_reselected_after_lease(self):
        record = self.add(1)
        backend = FakeBackend({"1": 100}, script=[Crash()])
        with self.assertRaises(Crash):
            self.deliver(backend)
        self.assertEqual(backend.prices["1"], 100)          # crashed before the request was applied
        row = self.rows()[record]
        self.assertEqual((row.attempt_count, row.next_attempt_at), (1, T0 + timedelta(seconds=300)))
        self.assertEqual(self.deliver(backend, now=T0 + timedelta(seconds=299)).http_request_count, 0)
        self.assertEqual(self.deliver(backend, now=T0 + timedelta(seconds=300)).succeeded_count, 1)

    def test_backend_applied_then_crash_before_result_is_success_on_retry(self):
        record = self.add(1)
        backend = FakeBackend({"1": 100})

        def applied_then_crash(body):
            backend.apply(body)
            raise Crash()

        backend.script = [applied_then_crash]
        with self.assertRaises(Crash):
            self.deliver(backend)
        self.assertEqual((backend.prices["1"], self.status(record)), (200, "RETRY_WAIT"))
        retried = self.deliver(backend, now=T0 + timedelta(seconds=300))
        self.assertEqual((retried.succeeded_count, self.status(record)), (1, "DONE"))
        self.assertEqual(self.rows()[record].attempt_count, 2)

    def test_failures_are_not_retried_in_the_same_run(self):
        for code in range(1, 6):
            self.add(code)
        backend = FakeBackend({str(c): 999 for c in range(1, 6)}, batch_size=2)   # every product conflicts
        report = self.deliver(backend, settings=DeliverySettings(max_requests=50))
        self.assertEqual((len(backend.calls), report.conflict_retry_count, report.batch_limit_reached), (3, 5, False))

    def test_nothing_due_after_run_start_is_selected_in_the_same_run(self):
        for code in (1, 2, 3):
            self.add(code)
        ticks = iter(range(0, 10000, 10))
        clock = lambda: T0 + timedelta(seconds=next(ticks))   # 10 s per call; retries due 1 s later

        def insert_new(_body):
            self.admin.execute(
                "INSERT INTO price_change_outbox (scan_run_id, product_code, original_old_price, old_price, "
                "new_price, next_attempt_at) VALUES (%s, 9, 100, 100, 200, %s)", (uuid.uuid4(), T0 + timedelta(seconds=1)))

        backend = FakeBackend({str(c): 999 for c in (1, 2, 3, 9)}, batch_size=1, on_send=insert_new)
        session = DeliverySession(self.database)
        self.addCleanup(session.close)
        report = run_delivery(session, backend, DeliverySettings(retry_delay_seconds=1, max_requests=100),
                              clock=clock)
        self.assertEqual([call["products"][0]["product_code"] for call in backend.calls], ["1", "2", "3"])
        self.assertEqual((report.conflict_retry_count, report.batch_limit_reached), (3, False))

    def test_batch_cap_bounds_runs_that_send_nothing(self):
        for code in range(1, 8):
            self.add(code, old=0)                    # every record invalid: no HTTP at all
        backend = FakeBackend(batch_size=2)
        report = self.deliver(backend, settings=DeliverySettings(max_requests=10, max_batches=2))
        self.assertEqual((report.batch_count, report.invalid_blocked_count, report.http_request_count,
                          report.batch_limit_reached, report.limit_reason), (2, 4, 0, True, "max_batches"))
        self.assertEqual(sum(r.status == "PENDING" for r in self.rows().values()), 3)

    def test_delivery_status_and_resume_need_only_outbox_select_update(self):
        role = f"price_refresh_delivery_{uuid.uuid4().hex[:8]}"
        self.admin.execute(f"CREATE ROLE {role} LOGIN PASSWORD 'delivery-test-only'")
        self.addCleanup(self.admin.execute, f"DROP ROLE {role}")
        self.addCleanup(self.admin.execute, f"DROP OWNED BY {role}")
        self.admin.execute(f"GRANT USAGE ON SCHEMA public TO {role}")
        self.admin.execute(f"GRANT SELECT, UPDATE ON price_change_outbox TO {role}")
        from psycopg.conninfo import conninfo_to_dict
        config = conninfo_to_dict(TEST_URL)
        limited = DatabaseSettings(f"postgresql://{role}:delivery-test-only@{config['host']}:"
                                   f"{config.get('port', '5432')}/{config['dbname']}")
        ids = [self.add(code) for code in (1, 2, 3)]
        backend = FakeBackend({"1": 100, "2": 100, "3": 100}, batch_size=1,
                              script=[None, HttpResult(403, None)])
        session = DeliverySession(limited)
        self.addCleanup(session.close)
        report = run_delivery(session, backend, DeliverySettings(), clock=lambda: T0)
        self.assertEqual((report.succeeded_count, report.status), (1, "auth_stopped"))
        self.assertTrue(fetch_delivery_status(limited).stopped)
        self.assertEqual(run_resume(DeliverySession(limited), now=T0 + timedelta(hours=1)).status, "resumed")
        again = run_delivery(DeliverySession(limited), backend, DeliverySettings(), clock=lambda: T0 + timedelta(hours=1))
        self.assertEqual((again.status, again.succeeded_count), ("completed", 2))
        self.assertEqual([self.status(i) for i in ids], ["DONE"] * 3)

    def test_collection_then_delivery_end_to_end(self):
        self.admin.execute("UPDATE products SET has_been_recommended = TRUE WHERE product_code IN (1, 2)")
        applied = run_apply(PostgresApplySession(self.database), lambda: ScanResult(prices={"1": 1100, "2": 1200, "3": 1300}))
        self.assertEqual((applied.status, applied.to_dict()["updated_count"], applied.to_dict()["outbox_record_count"]),
                         ("completed", 3, 2))
        backend = FakeBackend({"1": 1000, "2": 1000}, script=[BackendError("backend_timeout", phase="response")])
        now = datetime.now(timezone.utc) + timedelta(minutes=1)   # records were queued at the DB's current time
        failed = self.deliver(backend, now=now)
        self.assertEqual((failed.status, failed.error_retry_count), ("completed_with_errors", 2))
        prices = dict(self.admin.execute("SELECT product_code, price FROM products WHERE product_code <= 3").fetchall())
        self.assertEqual(prices, {1: 1100, 2: 1200, 3: 1300})          # delivery failure never reverts AI prices
        later = self.deliver(backend, now=now + timedelta(seconds=300))
        self.assertEqual((later.status, later.succeeded_count), ("completed", 2))
        self.assertEqual(backend.prices, {"1": 1100, "2": 1200})
        self.assertEqual([p["old_price"] for p in backend.calls[1]["products"]], [1000, 1000])

    def test_real_client_over_loopback_http(self):
        record = self.add(7)
        state = FakeBackend({"7": 100})
        requests = []

        class Handler(BaseHTTPRequestHandler):
            def do_PATCH(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                requests.append((self.path, self.headers.get("X-Test-Only")))
                raw = json.dumps(state.apply(body)).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def log_message(self, *_):
                pass

        server = HTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            client = BackendClient(BackendSettings(f"http://127.0.0.1:{server.server_address[1]}",
                                                   {"X-Test-Only": "test-value"}))
            report = self.deliver(client)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)
        self.assertEqual((requests, report.succeeded_count, self.status(record)),
                         ([("/api/v1/internal/products", "test-value")], 1, "DONE"))


if __name__ == "__main__":
    unittest.main()
