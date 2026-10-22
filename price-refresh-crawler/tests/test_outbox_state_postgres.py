"""Opt-in outbox selection/state tests ONLY for the disposable test database.

Never reads DATABASE_URL. Time is always passed explicitly; nothing sleeps.
"""

import os
import unittest
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

from price_refresh import outbox
from price_refresh.outbox import (
    ATTEMPT_UNCONFIRMED, OutboxError, mark_attempts, mark_blocked, mark_done, mark_not_found,
    mark_price_conflict, mark_retry, select_due,
)

TEST_URL = os.getenv("PRICE_REFRESH_TEST_DATABASE_URL", "")
SQL = Path(__file__).resolve().parents[1] / "sql" / "create_price_change_outbox.sql"
T0 = datetime(2026, 10, 22, 2, 0, tzinfo=timezone.utc)
LATER = T0 + timedelta(minutes=5)


@unittest.skipUnless(TEST_URL, "requires isolated PRICE_REFRESH_TEST_DATABASE_URL; never DATABASE_URL")
class PostgresOutboxStateTests(unittest.TestCase):
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
                           "SELECT code, 1000 FROM generate_series(1, 300) AS code")
        self.admin.execute(SQL.read_text(encoding="utf-8"))
        self.connection = self.psycopg.connect(TEST_URL, connect_timeout=5)
        self.addCleanup(self.connection.close)

    def add(self, code, *, status="PENDING", next_at=T0 - timedelta(minutes=1), old=100, new=200,
            completed=None, error_code=None, http_status=None, message=None):
        return self.admin.execute(
            "INSERT INTO price_change_outbox (scan_run_id, product_code, original_old_price, old_price, "
            "new_price, status, next_attempt_at, completed_at, last_error_code, last_http_status, "
            "last_error_message) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) RETURNING id",
            (uuid.uuid4(), code, old, old, new, status, next_at,
             T0 if status == "DONE" and completed is None else completed,
             error_code, http_status, message)).fetchone()[0]

    def row(self, record_id):
        return outbox.record_from_row(self.admin.execute(
            f"SELECT {', '.join(outbox.COLUMNS)} FROM price_change_outbox WHERE id = %s",
            (record_id,)).fetchone())

    def due(self, now=T0, limit=100):
        records = select_due(self.connection, now=now, limit=limit)
        self.connection.rollback()
        return [record.id for record in records]

    def attempted(self, code=1, **kwargs):
        record_id = self.add(code, **kwargs)
        self.assertEqual([r.id for r in mark_attempts(self.connection, [record_id], now=T0,
                                                       next_attempt_at=LATER)], [record_id])
        self.connection.commit()
        return record_id

    def products_unchanged(self):
        self.assertEqual(self.admin.execute("SELECT count(*) FROM products WHERE price <> 1000").fetchone()[0], 0)

    # Selection

    def test_only_due_pending_and_retry_wait_records(self):
        past = self.add(1, next_at=T0 - timedelta(minutes=1))
        boundary = self.add(2, status="RETRY_WAIT", next_at=T0)
        self.add(3, next_at=T0 + timedelta(seconds=1))
        self.add(4, status="DONE", next_at=None)
        self.add(5, status="NOT_FOUND", next_at=None)
        self.add(6, status="BLOCKED", next_at=None)
        self.assertEqual(self.due(), [past, boundary])

    def test_product_order_blockers_and_independent_products(self):
        self.add(1, status="RETRY_WAIT", next_at=T0 + timedelta(days=1))   # future head blocks
        self.add(1)
        self.add(2, status="BLOCKED", next_at=None)
        self.add(2)
        self.add(3, status="DONE", next_at=None)
        after_done = self.add(3)
        self.add(4, status="NOT_FOUND", next_at=None)
        after_not_found = self.add(4)
        head = self.add(5, next_at=T0 - timedelta(minutes=2))
        self.add(5, next_at=T0 - timedelta(minutes=3))   # later record, earlier schedule: still waits
        independent = self.add(6)
        self.assertEqual(sorted(self.due()), sorted([after_done, after_not_found, head, independent]))

    def test_limit_applies_after_product_order_and_codes_are_unique(self):
        for code in range(1, 101):
            self.add(code, status="RETRY_WAIT", next_at=T0 + timedelta(hours=1))
            self.add(code, next_at=T0 - timedelta(hours=1))   # earliest due, but blocked
        normal = [self.add(code, next_at=T0 - timedelta(minutes=code)) for code in range(101, 106)]
        self.assertEqual(self.due(limit=5), list(reversed(normal)))
        for code in range(106, 300):
            self.add(code)
        records = select_due(self.connection, now=T0, limit=100)
        self.connection.rollback()
        self.assertEqual(len(records), 100)
        self.assertEqual(len({r.product_code for r in records}), 100)
        for limit in (0, 101, -1, True, 1.5, "10", None):
            with self.subTest(limit=limit), self.assertRaises(OutboxError) as raised:
                select_due(self.connection, now=T0, limit=limit)
            self.assertEqual(raised.exception.code, "invalid_outbox_limit")

    def test_order_is_next_attempt_then_id_and_repeatable(self):
        ids = [self.add(code, next_at=T0 - timedelta(minutes=1)) for code in (9, 3, 7)]
        earlier = self.add(8, next_at=T0 - timedelta(minutes=2))
        self.assertEqual(self.due(), [earlier, *ids])
        self.assertEqual(self.due(), self.due())

    def test_time_must_be_timezone_aware(self):
        with self.assertRaises(OutboxError):
            select_due(self.connection, now=T0.replace(tzinfo=None))

    # Attempts

    def test_attempt_increments_once_clears_previous_result_and_survives_crash(self):
        record_id = self.add(1, status="RETRY_WAIT", next_at=T0, error_code="SERVER_ERROR",
                             http_status=500, message="previous")
        later = self.add(1)
        marked = mark_attempts(self.connection, [record_id], now=T0, next_attempt_at=LATER)
        self.connection.commit()   # process "dies" here: no result is stored
        record = self.row(record_id)
        self.assertEqual(marked, [record])
        self.assertEqual((record.status, record.attempt_count, record.last_attempt_at, record.next_attempt_at,
                          record.last_error_code, record.last_http_status, record.last_error_message,
                          record.updated_at), ("RETRY_WAIT", 1, T0, LATER, ATTEMPT_UNCONFIRMED, None, None, T0))
        self.assertEqual(self.due(now=LATER - timedelta(microseconds=1)), [])
        self.assertEqual(self.due(now=LATER), [record_id])
        self.assertNotIn(later, self.due(now=LATER + timedelta(days=1)))
        self.assertEqual(mark_done(self.connection, record_id, now=LATER, http_status=200).attempt_count, 1)
        self.connection.commit()
        self.assertEqual(self.row(record_id).attempt_count, 1)

    def test_attempt_only_marks_sendable_records_and_validates_input(self):
        sendable = self.add(1)
        finished = [self.add(2, status=s, next_at=None) for s in ("DONE", "NOT_FOUND", "BLOCKED")]
        marked = mark_attempts(self.connection, [sendable, *finished, 999999], now=T0, next_attempt_at=LATER)
        self.assertEqual([r.id for r in marked], [sendable])
        self.connection.rollback()
        for ids, kwargs in (([], {}), ([1, 1], {}), ([0], {}), ([True], {}), (list(range(1, 102)), {}),
                            ([1], {"next_attempt_at": T0}), ([1], {"now": T0.replace(tzinfo=None)})):
            arguments = {"now": T0, "next_attempt_at": LATER, **kwargs}
            with self.subTest(ids=ids[:3], kwargs=kwargs), self.assertRaises(OutboxError):
                mark_attempts(self.connection, ids, **arguments)

    # Results

    def test_done_completes_and_releases_next_record(self):
        record_id = self.attempted()
        later = self.add(1)
        self.assertEqual(self.due(now=LATER), [record_id])
        record = mark_done(self.connection, record_id, now=LATER, http_status=200)
        self.connection.commit()
        self.assertEqual((record.status, record.completed_at, record.next_attempt_at, record.last_error_code),
                         ("DONE", LATER, None, None))
        self.assertEqual(self.due(now=LATER), [later])
        self.products_unchanged()

    def test_retry_keeps_request_prices(self):
        record_id = self.attempted(old=100, new=200)
        record = mark_retry(self.connection, record_id, now=T0, next_attempt_at=LATER + timedelta(minutes=5),
                            error_code="BACKEND_TIMEOUT", error_message="read timeout", http_status=None)
        self.connection.commit()
        self.assertEqual((record.status, record.original_old_price, record.old_price, record.new_price,
                          record.attempt_count, record.completed_at, record.last_error_code,
                          record.last_error_message, record.next_attempt_at),
                         ("RETRY_WAIT", 100, 100, 200, 1, None, "BACKEND_TIMEOUT", "read timeout",
                          LATER + timedelta(minutes=5)))
        self.assertIsNone(mark_retry(self.connection, record_id, now=T0, next_attempt_at=LATER,
                                     error_code="AGAIN"))   # result already stored
        self.products_unchanged()

    def test_price_conflict_adjusts_only_old_price_and_backend_price(self):
        record_id = self.attempted(old=69000, new=75000)
        record = mark_price_conflict(self.connection, record_id, now=T0, next_attempt_at=LATER,
                                     backend_price=65000, http_status=200)
        self.connection.commit()
        self.assertEqual((record.original_old_price, record.old_price, record.new_price, record.last_backend_price,
                          record.status, record.last_error_code, record.attempt_count),
                         (69000, 65000, 75000, 65000, "RETRY_WAIT", "PRICE_CONFLICT", 1))
        self.products_unchanged()
        again = self.attempted(code=2, old=69000, new=75000)
        self.assertIsNone(mark_price_conflict(self.connection, again, now=T0, next_attempt_at=LATER,
                                              backend_price=75000))
        self.connection.rollback()
        self.assertEqual(self.row(again).old_price, 69000)
        for price in (-1, True, 1.5, "65000", None, 2 ** 31):
            with self.subTest(price=price), self.assertRaises(OutboxError) as raised:
                mark_price_conflict(self.connection, again, now=T0, next_attempt_at=LATER, backend_price=price)
            self.assertEqual(raised.exception.code, "invalid_backend_price")

    def test_not_found_releases_next_record_and_blocked_holds_it(self):
        missing = self.attempted(code=1)
        after_missing = self.add(1)
        record = mark_not_found(self.connection, missing, now=T0, http_status=200)
        self.connection.commit()
        self.assertEqual((record.status, record.next_attempt_at, record.completed_at), ("NOT_FOUND", None, None))
        held = self.add(2)                     # blocked before any attempt (local validation)
        after_held = self.add(2)
        blocked = mark_blocked(self.connection, held, now=T0, error_code="INVALID_LOCAL_PRICE",
                               error_message="old_price must be positive")
        self.connection.commit()
        self.assertEqual((blocked.status, blocked.next_attempt_at, blocked.completed_at, blocked.last_error_code),
                         ("BLOCKED", None, None, "INVALID_LOCAL_PRICE"))
        due = self.due(now=LATER)
        self.assertIn(after_missing, due)
        self.assertNotIn(after_held, due)
        self.assertNotIn(missing, due)

    def test_blocked_rejects_authentication_failures(self):
        record_id = self.attempted()
        for kwargs in ({"error_code": outbox.AUTH_FAILED}, {"error_code": "REJECTED", "http_status": 401},
                       {"error_code": "REJECTED", "http_status": 403}):
            with self.subTest(kwargs=kwargs), self.assertRaises(OutboxError):
                mark_blocked(self.connection, record_id, now=T0, **kwargs)
        self.assertEqual(self.row(record_id).status, "RETRY_WAIT")

    def test_results_never_reopen_or_skip_attempt(self):
        pending = self.add(1)
        done = self.add(2, status="DONE", next_at=None)
        not_found = self.add(3, status="NOT_FOUND", next_at=None)
        blocked = self.add(4, status="BLOCKED", next_at=None)
        calls = [
            lambda i: mark_done(self.connection, i, now=T0),
            lambda i: mark_retry(self.connection, i, now=T0, next_attempt_at=LATER, error_code="X"),
            lambda i: mark_price_conflict(self.connection, i, now=T0, next_attempt_at=LATER, backend_price=1),
            lambda i: mark_not_found(self.connection, i, now=T0),
        ]
        for record_id in (pending, done, not_found, blocked, 999999):
            for call in calls:
                with self.subTest(record_id=record_id):
                    self.assertIsNone(call(record_id))
        for record_id in (done, not_found, blocked, 999999):
            self.assertIsNone(mark_blocked(self.connection, record_id, now=T0, error_code="X"))
        self.connection.rollback()
        self.assertEqual([self.row(i).status for i in (pending, done, not_found, blocked)],
                         ["PENDING", "DONE", "NOT_FOUND", "BLOCKED"])

    def test_invalid_result_arguments_are_rejected_before_sql(self):
        record_id = self.attempted()
        cases = [
            lambda: mark_retry(self.connection, record_id, now=T0, next_attempt_at=LATER, error_code=""),
            lambda: mark_retry(self.connection, record_id, now=T0, next_attempt_at=LATER, error_code="a b"),
            lambda: mark_retry(self.connection, record_id, now=T0, next_attempt_at=LATER, error_code="x" * 65),
            lambda: mark_retry(self.connection, record_id, now=T0, next_attempt_at=LATER,
                               error_code=ATTEMPT_UNCONFIRMED),
            lambda: mark_retry(self.connection, record_id, now=T0, next_attempt_at=LATER, error_code="X",
                               error_message="line\nbreak"),
            lambda: mark_retry(self.connection, record_id, now=T0, next_attempt_at=LATER, error_code="X",
                               error_message="x" * 501),
            lambda: mark_retry(self.connection, record_id, now=T0, next_attempt_at=T0, error_code="X"),
            lambda: mark_done(self.connection, record_id, now=T0, http_status=99),
            lambda: mark_done(self.connection, record_id, now=T0, http_status=True),
            lambda: mark_done(self.connection, "1", now=T0),
        ]
        for index, call in enumerate(cases):
            with self.subTest(case=index), self.assertRaises(OutboxError):
                call()
        self.assertEqual(self.row(record_id).last_error_code, ATTEMPT_UNCONFIRMED)

    # Transactions

    def test_caller_owns_transaction(self):
        record_id = self.add(1)
        mark_attempts(self.connection, [record_id], now=T0, next_attempt_at=LATER)
        self.assertEqual(self.row(record_id).attempt_count, 0)   # not committed by the function
        self.connection.rollback()
        self.assertEqual((self.row(record_id).status, self.row(record_id).attempt_count), ("PENDING", 0))
        record_id = self.attempted(code=2)
        mark_done(self.connection, record_id, now=T0)
        self.connection.rollback()
        self.assertEqual(self.row(record_id).status, "RETRY_WAIT")
        self.assertFalse(self.connection.closed)

    def test_database_errors_keep_sqlstate(self):
        self.admin.execute("DROP TABLE price_change_outbox")
        with self.assertRaises(self.psycopg.Error) as raised:
            select_due(self.connection, now=T0)
        self.assertEqual(raised.exception.sqlstate, "42P01")
        self.connection.rollback()


if __name__ == "__main__":
    unittest.main()
