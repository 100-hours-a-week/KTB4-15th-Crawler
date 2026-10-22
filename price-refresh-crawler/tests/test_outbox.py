"""Offline outbox model and INSERT contract; no driver or database required."""

import unittest
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone

from price_refresh import outbox
from price_refresh.outbox import OutboxError, OutboxRecord, insert_pending, record_from_row

RUN = uuid.UUID("00000000-0000-4000-8000-000000000001")
NOW = datetime(2026, 10, 22, 2, 0, tzinfo=timezone.utc)


def row(**overrides):
    values = dict(id=7, scan_run_id=RUN, product_code=123, original_old_price=89000,
                  old_price=89000, new_price=79000, status="PENDING", attempt_count=0,
                  next_attempt_at=NOW, last_attempt_at=None, last_backend_price=None,
                  last_http_status=None, last_error_code=None, last_error_message=None,
                  created_at=NOW, updated_at=NOW, completed_at=None)
    values.update(overrides)
    return tuple(values[name] for name in outbox.COLUMNS)


class FakeConnection:
    """Records statements; any commit/rollback/close by the function is a failure."""

    def __init__(self, results):
        self.results = list(results)
        self.statements = []

    @contextmanager
    def cursor(self):
        connection = self

        class Cursor:
            def execute(self, sql, params=None):
                connection.statements.append((sql, params))

            def fetchone(self):
                return connection.results.pop(0)

        yield Cursor()

    def commit(self):
        raise AssertionError("insert_pending must not commit")

    def rollback(self):
        raise AssertionError("insert_pending must not roll back")

    def close(self):
        raise AssertionError("insert_pending must not close the caller's connection")


class OutboxModelTests(unittest.TestCase):
    def test_model_has_the_seventeen_table_columns_in_order(self):
        self.assertEqual(outbox.COLUMNS, (
            "id", "scan_run_id", "product_code", "original_old_price", "old_price", "new_price",
            "status", "attempt_count", "next_attempt_at", "last_attempt_at", "last_backend_price",
            "last_http_status", "last_error_code", "last_error_message", "created_at",
            "updated_at", "completed_at"))
        self.assertEqual(outbox.STATUSES, ("PENDING", "RETRY_WAIT", "DONE", "NOT_FOUND", "BLOCKED"))
        self.assertNotIn(outbox.AUTH_FAILED, outbox.STATUSES)

    def test_row_conversion_normalizes_bigint_code_to_string(self):
        record = record_from_row(row())
        self.assertIsInstance(record, OutboxRecord)
        self.assertEqual((record.id, record.product_code, record.status), (7, "123", "PENDING"))

    def test_auth_failure_is_a_retry_wait_record_with_error_fields(self):
        record = record_from_row(row(status="RETRY_WAIT", attempt_count=1, last_http_status=401,
                                     last_error_code=outbox.AUTH_FAILED))
        self.assertEqual((record.status, record.last_error_code, record.last_http_status),
                         ("RETRY_WAIT", "AUTH_FAILED", 401))

    def test_invalid_rows_are_rejected(self):
        for bad in (None, row()[:-1], row(status="AUTH_FAILED"), row(product_code="123"),
                    row(product_code=0), row(new_price=-1), row(old_price=True),
                    row(scan_run_id=str(RUN)), row(attempt_count=-1),
                    row(status="DONE"), row(completed_at=NOW)):
            with self.subTest(bad=bad), self.assertRaises(OutboxError) as raised:
                record_from_row(bad)
            self.assertEqual(raised.exception.code, "invalid_outbox_row")
        self.assertEqual(record_from_row(row(status="DONE", completed_at=NOW)).status, "DONE")


class InsertPendingTests(unittest.TestCase):
    def test_new_record_uses_parameters_and_same_old_prices_without_commit(self):
        connection = FakeConnection([row()])
        record = insert_pending(connection, RUN, "123", 89000, 79000)
        self.assertEqual(record.id, 7)
        self.assertEqual(len(connection.statements), 1)
        sql, params = connection.statements[0]
        self.assertIn("ON CONFLICT (scan_run_id, product_code) DO NOTHING", sql)
        self.assertNotIn("89000", sql)
        self.assertEqual(params, (RUN, 123, 89000, 89000, 79000))

    def test_string_run_id_is_accepted(self):
        connection = FakeConnection([row()])
        insert_pending(connection, str(RUN), "123", 89000, 79000)
        self.assertEqual(connection.statements[0][1][0], RUN)

    def test_identical_existing_record_is_returned_even_after_old_price_adjustment(self):
        existing = row(id=3, old_price=85000, status="RETRY_WAIT")
        connection = FakeConnection([None, existing])
        record = insert_pending(connection, RUN, "123", 89000, 79000)
        self.assertEqual((record.id, record.old_price), (3, 85000))
        self.assertEqual(connection.statements[1][1], (RUN, 123))

    def test_different_change_for_same_run_and_product_is_an_error(self):
        for existing in (row(new_price=69000), row(original_old_price=99000, old_price=99000)):
            with self.subTest(existing=existing), self.assertRaises(OutboxError) as raised:
                insert_pending(FakeConnection([None, existing]), RUN, "123", 89000, 79000)
            self.assertEqual(raised.exception.code, "outbox_record_conflict")

    def test_invalid_values_never_reach_sql(self):
        cases = [("bad-run", "123", 1, 2), (RUN, "0123", 1, 2), (RUN, 123, 1, 2),
                 (RUN, "9223372036854775808", 1, 2), (RUN, "123", -1, 2), (RUN, "123", 1, 2.0),
                 (RUN, "123", True, 2), (RUN, "123", 5, 5), (RUN, "123", 1, 2147483648)]
        for case in cases:
            connection = FakeConnection([])
            with self.subTest(case=case), self.assertRaises(OutboxError) as raised:
                insert_pending(connection, *case)
            self.assertEqual(raised.exception.code, "invalid_outbox_record")
            self.assertEqual(connection.statements, [])

    def test_zero_price_follows_existing_collection_policy(self):
        connection = FakeConnection([row(original_old_price=0, old_price=0)])
        self.assertEqual(insert_pending(connection, RUN, "123", 0, 79000).old_price, 0)


if __name__ == "__main__":
    unittest.main()
