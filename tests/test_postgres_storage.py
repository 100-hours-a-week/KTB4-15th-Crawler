"""실제 PostgreSQL 없이 postgres_storage의 변환/SQL/batch 로직을 검증한다."""

import os
import unittest
from datetime import datetime, timedelta, timezone

from app.config.database import DatabaseConfigError, get_database_url
from app.models.product import Product
from app.storage.postgres_storage import (
    _INSERT_COLUMNS,
    build_insert_sql,
    product_to_row,
    save_products,
)

CRAWLED_AT = datetime(2026, 9, 19, 12, 0, tzinfo=timezone(timedelta(hours=9)))

def _make_product(code: int, *, color=None, sub_category="반소매 티셔츠") -> Product:
    return Product(
        product_code=code,
        product_name=f"상품 {code}",
        detail_url=f"https://product.29cm.co.kr/catalog/{code}",
        image_url="https://img.29cm.co.kr/item/example.jpg",
        price=10000,
        is_sold_out=False,
        main_category="상의",
        sub_category=sub_category,
        color=color,
        crawled_at=CRAWLED_AT,
    )

class FakeCursor:
    """실제 psycopg cursor 대신 실행된 SQL/params와 rowcount만 기록한다."""

    def __init__(self):
        self.executed: list[tuple[str, list]] = []
        self.rowcount = 0

    def execute(self, sql, params):
        self.executed.append((sql, list(params)))
        self.rowcount = len(params) // len(_INSERT_COLUMNS)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class FakeConnection:
    def __init__(self):
        self.committed = False
        self.closed = False
        self._cursor = FakeCursor()

    def cursor(self):
        return self._cursor

    def commit(self):
        self.committed = True

    def close(self):
        self.closed = True


class ProductToRowTests(unittest.TestCase):
    def test_converts_all_fields_in_column_order(self):
        product = _make_product(1001, color="블랙")
        row = product_to_row(product)
        self.assertEqual(
            row,
            (
                1001,
                "상품 1001",
                10000,
                "블랙",
                "https://product.29cm.co.kr/catalog/1001",
                "https://img.29cm.co.kr/item/example.jpg",
                "상의",
                "반소매 티셔츠",
                False,
                CRAWLED_AT,
            ),
        )
        self.assertEqual(len(row), len(_INSERT_COLUMNS))

    def test_color_none_is_kept_as_none(self):
        product = _make_product(1002, color=None)
        row = product_to_row(product)
        self.assertIsNone(row[_INSERT_COLUMNS.index("color")])

    def test_crawled_at_stays_timezone_aware_datetime(self):
        product = _make_product(1003)
        row = product_to_row(product)
        crawled_at = row[_INSERT_COLUMNS.index("crawled_at")]
        self.assertIsInstance(crawled_at, datetime)
        self.assertIsNotNone(crawled_at.utcoffset())
        self.assertEqual(crawled_at, CRAWLED_AT)


class BuildInsertSqlTests(unittest.TestCase):
    def test_uses_on_conflict_do_nothing_on_product_code(self):
        sql = build_insert_sql(3)
        self.assertIn("ON CONFLICT (product_code) DO NOTHING", sql)
        self.assertNotIn("DO UPDATE", sql)
        self.assertIn("INSERT INTO products", sql)

    def test_has_one_value_group_per_row(self):
        sql = build_insert_sql(3)
        self.assertEqual(sql.count("%s"), 3 * len(_INSERT_COLUMNS))


class SaveProductsTests(unittest.TestCase):
    def test_empty_list_returns_zero_without_connection(self):
        os.environ.pop("DATABASE_URL", None)
        self.assertEqual(save_products([], connection=None), 0)

    def test_splits_into_batches_and_sums_inserted_rows(self):
        products = [_make_product(2000 + i) for i in range(5)]
        conn = FakeConnection()

        inserted = save_products(products, batch_size=2, connection=conn)

        self.assertEqual(inserted, 5)
        row_counts = [len(params) // len(_INSERT_COLUMNS) for _, params in conn._cursor.executed]
        self.assertEqual(row_counts, [2, 2, 1])

    def test_commits_once_and_does_not_close_injected_connection(self):
        products = [_make_product(3000 + i) for i in range(3)]
        conn = FakeConnection()

        save_products(products, batch_size=10, connection=conn)

        self.assertTrue(conn.committed)
        self.assertFalse(conn.closed)

    def test_conflicting_product_codes_are_not_counted_as_inserted(self):
        products = [_make_product(4000 + i) for i in range(4)]
        conn = FakeConnection()

        # 2개는 이미 있어서 충돌(skip)되는 상황을 흉내낸다.
        def execute_with_conflicts(sql, params):
            conn._cursor.executed.append((sql, list(params)))
            conn._cursor.rowcount = max(0, len(params) // len(_INSERT_COLUMNS) - 2)

        conn._cursor.execute = execute_with_conflicts

        inserted = save_products(products, batch_size=10, connection=conn)

        self.assertEqual(inserted, 2)

    def test_rejects_invalid_batch_size(self):
        with self.assertRaises(ValueError):
            save_products([_make_product(5000)], batch_size=0, connection=FakeConnection())


class DatabaseUrlTests(unittest.TestCase):
    def tearDown(self):
        os.environ.pop("DATABASE_URL", None)

    def test_missing_database_url_raises_clear_error(self):
        os.environ.pop("DATABASE_URL", None)
        with self.assertRaises(DatabaseConfigError) as error:
            get_database_url()
        self.assertIn("DATABASE_URL", str(error.exception))

    def test_returns_configured_value(self):
        os.environ["DATABASE_URL"] = "postgresql://user:pw@localhost:5432/dbname"
        self.assertEqual(
            get_database_url(), "postgresql://user:pw@localhost:5432/dbname"
        )

    def test_rejects_non_postgres_scheme(self):
        os.environ["DATABASE_URL"] = "mysql://user:pw@localhost:3306/dbname"
        with self.assertRaises(DatabaseConfigError):
            get_database_url()


if __name__ == "__main__":
    unittest.main()
