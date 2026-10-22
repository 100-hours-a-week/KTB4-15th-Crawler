"""Opt-in authentication stop/resume tests ONLY for the disposable test database.

Never reads DATABASE_URL from the environment of the caller; time is passed explicitly.
"""

import io
import json
import os
import unittest
import uuid
from contextlib import redirect_stdout
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from price_refresh import outbox
from price_refresh.apply import run_apply
from price_refresh.delivery_control import DeliveryControlError, fetch_delivery_status, run_resume
from price_refresh.main import main
from price_refresh.outbox import (
    AUTH_FAILED, OutboxError, delivery_stop_status, mark_attempts, mark_auth_failed, select_due,
)
from price_refresh.repository import DatabaseSettings
from price_refresh.scanner import ScanResult
from price_refresh.writer import DeliverySession, PostgresApplySession

TEST_URL = os.getenv("PRICE_REFRESH_TEST_DATABASE_URL", "")
SQL = Path(__file__).resolve().parents[1] / "sql" / "create_price_change_outbox.sql"
T0 = datetime(2026, 10, 22, 2, 0, tzinfo=timezone.utc)
LATER = T0 + timedelta(minutes=5)
RESUME = T0 + timedelta(hours=1)


@unittest.skipUnless(TEST_URL, "requires isolated PRICE_REFRESH_TEST_DATABASE_URL; never DATABASE_URL")
class PostgresDeliveryControlTests(unittest.TestCase):
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
                           "SELECT code, 1000 FROM generate_series(1, 20) AS code")
        self.admin.execute(SQL.read_text(encoding="utf-8"))
        self.connection = self.psycopg.connect(TEST_URL, connect_timeout=5)
        self.addCleanup(self.connection.close)
        self.settings = DatabaseSettings(TEST_URL)

    def add(self, code, *, status="PENDING", next_at=T0 - timedelta(minutes=1), error_code=None,
            http_status=None, old=100, new=200):
        return self.admin.execute(
            "INSERT INTO price_change_outbox (scan_run_id, product_code, original_old_price, old_price, "
            "new_price, status, next_attempt_at, completed_at, last_error_code, last_http_status) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s) RETURNING id",
            (uuid.uuid4(), code, old, old, new, status, next_at, T0 if status == "DONE" else None,
             error_code, http_status)).fetchone()[0]

    def rows(self):
        return {row[0]: outbox.record_from_row(row) for row in self.admin.execute(
            f"SELECT {', '.join(outbox.COLUMNS)} FROM price_change_outbox ORDER BY id").fetchall()}

    def attempted(self, *codes):
        ids = [self.add(code) for code in codes]
        mark_attempts(self.connection, ids, now=T0, next_attempt_at=LATER)
        self.connection.commit()
        return ids

    def stop_with_auth_failure(self, *codes, http_status=401):
        ids = self.attempted(*codes)
        with self.connection.transaction():
            mark_auth_failed(self.connection, ids, now=T0, next_attempt_at=LATER, http_status=http_status)
        return ids

    def session(self, cls=DeliverySession):
        session = cls(self.settings)
        self.addCleanup(session.close)
        return session

    # Stop status

    def test_status_allowed_and_ordinary_retry_wait_is_not_a_stop(self):
        self.add(1, status="RETRY_WAIT", error_code="SERVER_ERROR", http_status=500)
        self.add(2, status="RETRY_WAIT", error_code="BACKEND_TIMEOUT")
        self.add(3, status="DONE", next_at=None)
        stop = fetch_delivery_status(self.settings)
        self.assertEqual((stop.stopped, stop.auth_failed_count, stop.http_statuses), (False, 0, ()))

    def test_any_auth_failure_stops_even_future_or_outside_a_batch(self):
        for code in range(1, 20):
            self.add(code)                                    # 19 due records, plenty for a batch
        self.add(20, status="RETRY_WAIT", next_at=T0 + timedelta(days=30), error_code=AUTH_FAILED,
                 http_status=403)
        self.assertNotIn(20, [r.product_code for r in select_due(self.connection, now=T0, limit=5)])
        self.connection.rollback()
        stop = fetch_delivery_status(self.settings)
        self.assertEqual((stop.stopped, stop.auth_failed_count, stop.http_statuses), (True, 1, (403,)))

    def test_status_check_failure_is_not_allowed(self):
        self.admin.execute("DROP TABLE price_change_outbox")
        with self.assertRaises(DeliveryControlError):
            fetch_delivery_status(self.settings)
        with self.assertRaises(self.psycopg.Error):
            delivery_stop_status(self.connection)
        self.connection.rollback()

    # Recording the failure

    def test_auth_failure_changes_only_the_attempted_request_rows(self):
        ids = self.attempted(1, 2)
        unsent = self.add(3)
        before = self.rows()
        with self.connection.transaction():
            records = mark_auth_failed(self.connection, ids, now=T0, next_attempt_at=LATER, http_status=401)
        after = self.rows()
        self.assertEqual([r.id for r in records], ids)
        for record_id in ids:
            self.assertEqual((after[record_id].status, after[record_id].last_error_code,
                              after[record_id].last_http_status), ("RETRY_WAIT", AUTH_FAILED, 401))
            self.assertEqual((after[record_id].attempt_count, after[record_id].original_old_price,
                              after[record_id].old_price, after[record_id].new_price),
                             (1, 100, 100, 200))
        self.assertEqual(after[unsent], before[unsent])
        other = self.psycopg.connect(TEST_URL, connect_timeout=5)   # a new connection sees the stop
        self.addCleanup(other.close)
        self.assertEqual(delivery_stop_status(other).auth_failed_count, 2)

    def test_only_401_and_403(self):
        ids = self.attempted(1)
        for status in (400, 404, 500, None, True, "401", 401.0):
            with self.subTest(status=status), self.assertRaises(OutboxError):
                mark_auth_failed(self.connection, ids, now=T0, next_attempt_at=LATER, http_status=status)
        self.connection.rollback()
        self.assertEqual(self.rows()[ids[0]].last_error_code, outbox.ATTEMPT_UNCONFIRMED)

    def test_partial_update_raises_and_rolls_back_whole_request(self):
        attempted = self.attempted(1)
        never_sent = self.add(2)
        done = self.add(3, status="DONE", next_at=None)
        for ids in ([attempted[0], never_sent], [attempted[0], done], [attempted[0], 999999]):
            with self.subTest(ids=ids):
                with self.assertRaises(OutboxError) as raised:
                    with self.connection.transaction():
                        mark_auth_failed(self.connection, ids, now=T0, next_attempt_at=LATER, http_status=401)
                self.assertEqual(raised.exception.code, "auth_failure_not_recorded")
                self.assertFalse(fetch_delivery_status(self.settings).stopped)
                self.assertEqual(self.rows()[attempted[0]].last_error_code, outbox.ATTEMPT_UNCONFIRMED)

    # Manual resume

    def test_resume_releases_only_auth_stop_and_keeps_history(self):
        stopped = self.stop_with_auth_failure(1, 2, http_status=403)
        ordinary = self.add(3, status="RETRY_WAIT", next_at=T0 + timedelta(days=1), error_code="SERVER_ERROR",
                            http_status=500)
        others = [self.add(4, status=s, next_at=None) for s in ("DONE", "NOT_FOUND", "BLOCKED")]
        self.admin.execute("UPDATE price_change_outbox SET last_backend_price = 90, old_price = 90 "
                           "WHERE id = %s", (stopped[0],))
        before = self.rows()
        emitted = []
        with patch("http.client.HTTPConnection.connect", side_effect=AssertionError("no HTTP")), \
                patch("http.client.HTTPSConnection.connect", side_effect=AssertionError("no HTTPS")):
            report = run_resume(self.session(), now=RESUME, emit=emitted.append)
        after = self.rows()
        self.assertEqual((report.status, report.auth_failed_count, report.http_statuses, report.resumed_ids),
                         ("resumed", 2, [403], stopped))
        self.assertEqual(emitted[0]["resumed_count"], 2)
        for record_id in stopped:
            old, new = before[record_id], after[record_id]
            self.assertEqual((new.status, new.last_error_code, new.last_error_message, new.next_attempt_at,
                              new.updated_at), ("RETRY_WAIT", None, None, RESUME, RESUME))
            self.assertEqual((new.last_http_status, new.last_attempt_at, new.attempt_count, new.original_old_price,
                              new.old_price, new.new_price, new.last_backend_price),
                             (old.last_http_status, old.last_attempt_at, old.attempt_count, old.original_old_price,
                              old.old_price, old.new_price, old.last_backend_price))
        for record_id in [ordinary, *others]:
            self.assertEqual(after[record_id], before[record_id])
        self.assertFalse(fetch_delivery_status(self.settings).stopped)
        due = [r.id for r in select_due(self.connection, now=RESUME)]
        self.connection.rollback()
        self.assertTrue(set(stopped) <= set(due))
        self.assertEqual(self.admin.execute("SELECT count(*) FROM products WHERE price <> 1000").fetchone()[0], 0)

    def test_resume_without_stop_changes_nothing(self):
        self.add(1, status="RETRY_WAIT", error_code="SERVER_ERROR")
        before = self.rows()
        report = run_resume(self.session(), now=RESUME)
        self.assertEqual((report.status, report.auth_failed_count, report.resumed_ids), ("already_allowed", 0, []))
        self.assertEqual(self.rows(), before)

    def test_competing_delivery_lock_prevents_resume_but_collection_lock_does_not(self):
        self.stop_with_auth_failure(1)
        holder = self.session()
        self.assertTrue(holder.open())
        report = run_resume(self.session(), now=RESUME)
        self.assertEqual(report.status, "skipped_concurrent")
        self.assertTrue(fetch_delivery_status(self.settings).stopped)
        holder.close()
        collector = self.session(PostgresApplySession)
        self.assertTrue(collector.open())
        self.assertEqual(run_resume(self.session(), now=RESUME).status, "resumed")
        collector.assert_lock()

    def test_lock_or_connection_loss_keeps_the_stop(self):
        self.stop_with_auth_failure(1)
        from price_refresh import delivery_control
        original = delivery_control.release_auth_stop
        for loss in ("unlock", "terminate"):
            with self.subTest(loss=loss):
                session = self.session()

                def lose_then_release(connection, *, now):
                    if loss == "unlock":
                        # The UPDATE succeeds, but the lock check before COMMIT must roll it back.
                        connection.execute("SELECT pg_advisory_unlock_all()")
                    else:
                        self.admin.execute("SELECT pg_terminate_backend(%s)", (session.backend_pid,))
                    return original(connection, now=now)

                with patch("price_refresh.delivery_control.release_auth_stop", lose_then_release):
                    report = run_resume(session, now=RESUME)
                self.assertEqual(report.status, "failed")
                self.assertTrue(fetch_delivery_status(self.settings).stopped)
                self.assertEqual(self.rows()[1].last_error_code, AUTH_FAILED)

    # Separation from collection

    def test_collection_still_writes_prices_and_outbox_during_stop_and_does_not_clear_it(self):
        self.stop_with_auth_failure(1)
        self.admin.execute("UPDATE products SET has_been_recommended = TRUE WHERE product_code = 5")
        report = run_apply(self.session(PostgresApplySession), lambda: ScanResult(prices={"5": 1500}))
        self.assertEqual((report.status, report.to_dict()["outbox_record_count"]), ("completed", 1))
        self.assertEqual(self.admin.execute("SELECT price FROM products WHERE product_code = 5").fetchone()[0], 1500)
        stop = fetch_delivery_status(self.settings)
        self.assertEqual((stop.stopped, stop.auth_failed_count), (True, 1))

    def test_cli_status_and_resume_against_database_without_backend_settings(self):
        self.stop_with_auth_failure(1, 2)
        env = {"DATABASE_URL": TEST_URL}
        blocked = [patch("http.client.HTTPConnection.connect", side_effect=AssertionError("no HTTP")),
                   patch("http.client.HTTPSConnection.connect", side_effect=AssertionError("no HTTPS")),
                   patch("price_refresh.main.collect_prices", side_effect=AssertionError("no 29CM"))]
        for item in blocked:
            item.start()
            self.addCleanup(item.stop)
        results = []
        for args in (["--delivery-status"], ["--delivery-status"], ["--resume-delivery"], ["--delivery-status"],
                     ["--resume-delivery"]):
            output = io.StringIO()
            with patch.dict("os.environ", env, clear=True), redirect_stdout(output):
                code = main(args)
            results.append((code, json.loads(output.getvalue())["status"]))
        self.assertEqual(results, [(3, "stopped"), (3, "stopped"), (0, "resumed"), (0, "allowed"),
                                   (0, "already_allowed")])


if __name__ == "__main__":
    unittest.main()
