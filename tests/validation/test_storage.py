"""app/validation/storage.py 의 SQL/변환을 실제 PostgreSQL 없이 검증한다."""

import unittest
from datetime import datetime, timedelta, timezone

from app.validation import storage

CRAWLED_AT = datetime(2026, 9, 22, 12, 0, tzinfo=timezone(timedelta(hours=9)))

_ROW = (
    1001,
    "테스트 상품",
    10000,
    None,
    "https://product.29cm.co.kr/catalog/1001",
    "https://img.29cm.co.kr/item/example.jpg",
    "상의",
    "스웨트셔츠",
    False,
    CRAWLED_AT,
)


class FakeCursor:
    def __init__(self, rows=()):
        self._rows = list(rows)
        self.executed: list[tuple[str, tuple]] = []
        self.rowcount = 0

    def execute(self, sql, params=()):
        self.executed.append((sql, tuple(params)))
        self.rowcount = len(self._rows)

    def fetchall(self):
        return self._rows

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class FakeConnection:
    def __init__(self, rows=()):
        self.committed = False
        self.closed = False
        self._cursor = FakeCursor(rows)

    def cursor(self):
        return self._cursor

    def commit(self):
        self.committed = True

    def close(self):
        self.closed = True


class GetPendingProductsTests(unittest.TestCase):
    def test_returns_products_reconstructed_from_rows(self):
        connection = FakeConnection(rows=[_ROW])

        products = storage.get_pending_products(connection)

        self.assertEqual(len(products), 1)
        self.assertEqual(products[0].product_code, 1001)
        self.assertEqual(products[0].sub_category, "스웨트셔츠")
        self.assertEqual(products[0].crawled_at, CRAWLED_AT)

    def test_filters_by_pending_status(self):
        connection = FakeConnection(rows=[_ROW])

        storage.get_pending_products(connection)

        [(sql, _params)] = connection._cursor.executed
        self.assertIn("validation_status = 'PENDING'", sql)

    def test_no_limit_queries_all_pending_products(self):
        connection = FakeConnection(rows=[_ROW])

        storage.get_pending_products(connection)

        [(sql, params)] = connection._cursor.executed
        self.assertNotIn("LIMIT", sql)
        self.assertEqual(params, ())

    def test_limit_is_passed_as_a_query_parameter(self):
        connection = FakeConnection(rows=[_ROW])

        storage.get_pending_products(connection, limit=10)

        [(sql, params)] = connection._cursor.executed
        self.assertIn("LIMIT", sql)
        self.assertEqual(params, (10,))


class UpdateValidationResultTests(unittest.TestCase):
    def test_updates_status_and_reason_and_commits(self):
        connection = FakeConnection()

        storage.update_validation_result(connection, 1001, "FAIL", "MULTIPLE_GARMENTS")

        [(sql, params)] = connection._cursor.executed
        self.assertIn("UPDATE products", sql)
        self.assertEqual(params, ("FAIL", "MULTIPLE_GARMENTS", 1001))
        self.assertTrue(connection.committed)

    def test_pass_result_has_no_reason(self):
        connection = FakeConnection()

        storage.update_validation_result(connection, 1001, "PASS", None)

        [(_, params)] = connection._cursor.executed
        self.assertIsNone(params[1])


class GetValidatedProductsTests(unittest.TestCase):
    def test_returns_product_with_status_and_reason(self):
        row_with_status = (*_ROW, "PASS", None)
        connection = FakeConnection(rows=[row_with_status])

        validated = storage.get_validated_products(connection)

        [(product, status, reason)] = validated
        self.assertEqual(product.product_code, 1001)
        self.assertEqual(status, "PASS")
        self.assertIsNone(reason)

    def test_only_pass_and_fail_are_selected(self):
        connection = FakeConnection(rows=[])

        storage.get_validated_products(connection)

        [(sql, _)] = connection._cursor.executed
        self.assertIn("IN ('PASS', 'FAIL')", sql)


class CountByValidationStatusTests(unittest.TestCase):
    def test_fills_in_zero_for_missing_statuses(self):
        connection = FakeConnection()
        connection._cursor._rows = [("PASS", 3)]

        counts = storage.count_by_validation_status(connection)

        self.assertEqual(counts, {"PENDING": 0, "PASS": 3, "FAIL": 0})


class DeleteFailedProductsTests(unittest.TestCase):
    def test_deletes_only_fail_status_and_commits(self):
        connection = FakeConnection(rows=[_ROW, _ROW])

        deleted = storage.delete_failed_products(connection)

        [(sql, _)] = connection._cursor.executed
        self.assertIn("DELETE FROM products", sql)
        self.assertIn("validation_status = 'FAIL'", sql)
        self.assertEqual(deleted, 2)
        self.assertTrue(connection.committed)


if __name__ == "__main__":
    unittest.main()
