import os
import unittest
import uuid
from pathlib import Path

from price_refresh import outbox
from price_refresh.outbox import OutboxError, insert_pending, record_from_row
from price_refresh.readiness import ReadinessIssue, check_readiness

TEST_URL = os.getenv("PRICE_REFRESH_TEST_DATABASE_URL", "")
SQL = Path(__file__).resolve().parents[1] / "sql" / "create_price_change_outbox.sql"
RUN_A = uuid.UUID("00000000-0000-4000-8000-00000000000a")
RUN_B = uuid.UUID("00000000-0000-4000-8000-00000000000b")
ROLE = "price_refresh_outbox_test_role"
ROLE_PASSWORD = "outbox-test-only"


@unittest.skipUnless(TEST_URL, "requires isolated PRICE_REFRESH_TEST_DATABASE_URL; never DATABASE_URL")
class PostgresOutboxTests(unittest.TestCase):
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
        # Mirrors crawling/db/schema.sql: surrogate id, product_code UNIQUE (not PK).
        self.admin.execute(
            "CREATE TABLE products (id BIGSERIAL PRIMARY KEY, product_code BIGINT NOT NULL UNIQUE, "
            "price INTEGER NOT NULL, has_been_recommended BOOLEAN NOT NULL DEFAULT FALSE)")
        self.admin.execute("INSERT INTO products (product_code, price) VALUES (1, 89000), (2, 100)")
        self.admin.execute(SQL.read_text(encoding="utf-8"))

    def connect(self, **kwargs):
        connection = self.psycopg.connect(TEST_URL, connect_timeout=5, **kwargs)
        self.addCleanup(connection.close)
        return connection

    def outbox_rows(self):
        return self.admin.execute(
            f"SELECT {', '.join(outbox.COLUMNS)} FROM price_change_outbox ORDER BY id").fetchall()

    def raises_sqlstate(self, state, sql, params=None):
        with self.assertRaises(self.psycopg.Error) as raised:
            self.admin.execute(sql, params)
        self.assertEqual(raised.exception.sqlstate, state)

    def limited_role(self, grants, *, revoke_public_schema=False):
        role = f"{ROLE}_{uuid.uuid4().hex[:8]}"
        self.admin.execute(f"CREATE ROLE {role} LOGIN PASSWORD '{ROLE_PASSWORD}'")
        self.addCleanup(self.admin.execute, f"DROP ROLE {role}")
        self.addCleanup(self.admin.execute, f"DROP OWNED BY {role}")
        if revoke_public_schema:
            self.admin.execute("REVOKE USAGE ON SCHEMA public FROM PUBLIC")
            self.addCleanup(self.admin.execute, "GRANT USAGE ON SCHEMA public TO PUBLIC")
        for grant in grants:
            self.admin.execute(grant.format(role=role))
        return self.connect(autocommit=True, user=role, password=ROLE_PASSWORD)

    APP_GRANTS = (
        "GRANT USAGE ON SCHEMA public TO {role}",
        "GRANT SELECT ON products TO {role}",
        "GRANT UPDATE (price) ON products TO {role}",
        "GRANT SELECT, INSERT, UPDATE ON price_change_outbox TO {role}",
    )

    def test_sql_matches_model_and_indexes(self):
        columns = [name for (name,) in self.admin.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_schema = 'public' AND table_name = 'price_change_outbox' "
            "ORDER BY ordinal_position").fetchall()]
        self.assertEqual(tuple(columns), outbox.COLUMNS)
        self.assertEqual(check_readiness(self.admin), [])
        indexes = dict(self.admin.execute(
            "SELECT indexname, indexdef FROM pg_indexes WHERE tablename = 'price_change_outbox'").fetchall())
        self.assertIn("NOT_FOUND", indexes["ix_price_change_outbox_unfinished_product"])
        self.assertIn("DONE", indexes["ix_price_change_outbox_unfinished_product"])
        self.assertIn("RETRY_WAIT", indexes["ix_price_change_outbox_due"])
        self.assertIn("completed_at", indexes["ix_price_change_outbox_completed"])
        with self.assertRaises(self.psycopg.Error) as raised:
            self.admin.execute(SQL.read_text(encoding="utf-8"))
        self.assertEqual(raised.exception.sqlstate, "42P07")  # never silently re-applied

    def test_insert_defaults_and_row_conversion(self):
        connection = self.connect()
        record = insert_pending(connection, RUN_A, "1", 89000, 79000)
        connection.commit()
        self.assertEqual((record.product_code, record.original_old_price, record.old_price,
                          record.new_price, record.status, record.attempt_count),
                         ("1", 89000, 89000, 79000, "PENDING", 0))
        self.assertIsNotNone(record.next_attempt_at)
        self.assertIsNone(record.completed_at)
        self.assertEqual(record_from_row(self.outbox_rows()[0]), record)

    def test_duplicates_new_runs_and_conflicts(self):
        connection = self.connect()
        first = insert_pending(connection, RUN_A, "1", 89000, 79000)
        self.assertEqual(insert_pending(connection, RUN_A, "1", 89000, 79000).id, first.id)
        second = insert_pending(connection, RUN_B, "1", 79000, 69000)
        self.assertGreater(second.id, first.id)
        for old, new in ((89000, 69000), (99000, 79000)):
            with self.subTest(old=old, new=new), self.assertRaises(OutboxError) as raised:
                insert_pending(connection, RUN_A, "1", old, new)
            self.assertEqual(raised.exception.code, "outbox_record_conflict")
        connection.commit()
        self.assertEqual([(r[0], r[5]) for r in self.outbox_rows()], [(first.id, 79000), (second.id, 69000)])
        # An old_price adjusted by a later stage still identifies the same change.
        self.admin.execute("UPDATE price_change_outbox SET old_price = 85000 WHERE id = %s", (first.id,))
        self.assertEqual(insert_pending(connection, RUN_A, "1", 89000, 79000).old_price, 85000)
        connection.rollback()

    def test_caller_owns_commit_rollback_and_connection(self):
        connection = self.connect()
        insert_pending(connection, RUN_A, "1", 89000, 79000)
        self.assertFalse(connection.closed)
        self.assertEqual(self.outbox_rows(), [])  # not committed by the function
        connection.rollback()
        self.assertEqual(self.outbox_rows(), [])
        with connection.transaction():
            connection.execute("UPDATE products SET price = 79000 WHERE product_code = 1")
            insert_pending(connection, RUN_A, "1", 89000, 79000)
        self.assertEqual(len(self.outbox_rows()), 1)
        self.assertEqual(self.admin.execute("SELECT price FROM products WHERE product_code = 1").fetchone()[0], 79000)

    def test_database_errors_propagate_with_sqlstate(self):
        connection = self.connect()
        with self.assertRaises(self.psycopg.Error) as raised:
            insert_pending(connection, RUN_A, "999", 1, 2)
        self.assertEqual(raised.exception.sqlstate, "23503")
        connection.rollback()

    def test_foreign_key_status_price_and_completion_constraints(self):
        sql = ("INSERT INTO price_change_outbox (scan_run_id, product_code, original_old_price, "
               "old_price, new_price, status, next_attempt_at, completed_at, last_http_status) "
               "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)")
        ts = "2026-10-22T00:00:00Z"
        cases = [
            ("23503", "fk_price_change_outbox_product", (999, 1, 1, 2, "PENDING", ts, None, None)),
            ("23514", "ck_price_change_outbox_status", (1, 1, 1, 2, "AUTH_FAILED", ts, None, None)),
            ("23514", "ck_price_change_outbox_prices", (1, -1, -1, 2, "PENDING", ts, None, None)),
            ("23514", "ck_price_change_outbox_changed", (1, 2, 1, 2, "PENDING", ts, None, None)),
            ("23514", "ck_price_change_outbox_completion", (1, 1, 1, 2, "DONE", None, None, None)),
            ("23514", "ck_price_change_outbox_completion", (1, 1, 1, 2, "BLOCKED", None, ts, None)),
            ("23514", "ck_price_change_outbox_schedule", (1, 1, 1, 2, "PENDING", None, None, None)),
            ("23514", "ck_price_change_outbox_schedule", (1, 1, 1, 2, "RETRY_WAIT", None, None, None)),
            ("23514", "ck_price_change_outbox_http_status", (1, 1, 1, 2, "PENDING", ts, None, 99)),
            ("23514", "ck_price_change_outbox_product", (-1, 1, 1, 2, "PENDING", ts, None, None)),
        ]
        for state, constraint, params in cases:
            with self.subTest(constraint=constraint, params=params):
                with self.assertRaises(self.psycopg.Error) as raised:
                    self.admin.execute(sql, (RUN_A, *params))
                self.assertEqual((raised.exception.sqlstate, raised.exception.diag.constraint_name),
                                 (state, constraint))
        self.admin.execute(sql, (RUN_A, 1, 1, 1, 2, "DONE", None, ts, None))
        self.admin.execute(sql, (RUN_B, 1, 0, 0, 2, "NOT_FOUND", None, None, None))
        self.admin.execute(sql, (RUN_A, 2, 100, 100, 200, "RETRY_WAIT", ts, None, 401))
        self.admin.execute("UPDATE price_change_outbox SET last_error_code = %s WHERE product_code = 2",
                           (outbox.AUTH_FAILED,))
        self.assertEqual({record_from_row(r).status for r in self.outbox_rows()},
                         {"DONE", "NOT_FOUND", "RETRY_WAIT"})
        self.raises_sqlstate("23503", "DELETE FROM products WHERE product_code = 1")

    def test_readiness_distinguishes_missing_tables_columns_types_and_constraints(self):
        cases = [
            ("ALTER TABLE products DROP COLUMN has_been_recommended",
             [ReadinessIssue("column_missing", "products.has_been_recommended")]),
            ("ALTER TABLE products ALTER COLUMN price TYPE bigint",
             [ReadinessIssue("column_type_mismatch", "products.price")]),
            ("ALTER TABLE price_change_outbox ALTER COLUMN status TYPE varchar(32)",
             [ReadinessIssue("column_type_mismatch", "price_change_outbox.status")]),
            ("ALTER TABLE price_change_outbox ALTER COLUMN scan_run_id DROP NOT NULL",
             [ReadinessIssue("column_definition_mismatch", "price_change_outbox.scan_run_id")]),
            ("ALTER TABLE price_change_outbox DROP CONSTRAINT uq_price_change_outbox_run_product",
             [ReadinessIssue("constraint_missing", "price_change_outbox unique(scan_run_id, product_code)")]),
            ("ALTER TABLE price_change_outbox DROP CONSTRAINT fk_price_change_outbox_product",
             [ReadinessIssue("constraint_missing", "price_change_outbox foreign key(product_code)")]),
            ("DROP TABLE price_change_outbox",
             [ReadinessIssue("table_missing", "public.price_change_outbox")]),
            ("DROP TABLE price_change_outbox; ALTER TABLE products DROP CONSTRAINT products_product_code_key",
             [ReadinessIssue("table_missing", "public.price_change_outbox"),
              ReadinessIssue("constraint_missing", "products unique(product_code)")]),
            ("DROP TABLE products CASCADE",
             [ReadinessIssue("table_missing", "public.products")]),
        ]
        for statement, expected in cases:
            with self.subTest(statement=statement):
                self.setUp()
                self.admin.execute(statement)
                self.assertEqual(check_readiness(self.admin), expected)

    def test_readiness_reports_missing_privileges_for_application_role(self):
        self.assertEqual(check_readiness(self.limited_role(self.APP_GRANTS)), [])
        cases = [
            ([g for g in self.APP_GRANTS if "UPDATE (price)" not in g],
             [ReadinessIssue("privilege_missing", "UPDATE products.price")]),
            ([g for g in self.APP_GRANTS if "price_change_outbox" not in g]
             + ["GRANT SELECT ON price_change_outbox TO {role}"],
             [ReadinessIssue("privilege_missing", "INSERT price_change_outbox"),
              ReadinessIssue("privilege_missing", "UPDATE price_change_outbox")]),
            ([g for g in self.APP_GRANTS if "SELECT ON products" not in g],
             [ReadinessIssue("privilege_missing", "SELECT products")]),
        ]
        for grants, expected in cases:
            with self.subTest(expected=expected):
                self.assertEqual(check_readiness(self.limited_role(grants)), expected)
        no_schema = self.limited_role(self.APP_GRANTS[1:], revoke_public_schema=True)
        self.assertEqual(check_readiness(no_schema),
                         [ReadinessIssue("privilege_missing", "USAGE schema public")])

    def test_application_role_inserts_with_identity_and_no_sequence_grant(self):
        connection = self.limited_role(self.APP_GRANTS)
        with self.assertRaises(self.psycopg.Error):
            connection.execute("DELETE FROM price_change_outbox")
        with connection.transaction():
            connection.execute("SELECT price FROM products WHERE product_code = 1 FOR UPDATE")
            connection.execute("UPDATE products SET price = 79000 WHERE product_code = 1")
            self.assertEqual(insert_pending(connection, RUN_A, "1", 89000, 79000).status, "PENDING")
        self.assertEqual(len(self.outbox_rows()), 1)

    def test_readiness_is_read_only_and_leaves_transaction_to_caller(self):
        insert_pending(self.admin, RUN_A, "1", 89000, 79000)
        before = (self.admin.execute("SELECT * FROM products ORDER BY id").fetchall(), self.outbox_rows())
        connection = self.connect()
        connection.read_only = True  # any write inside the check would fail with 25006
        self.assertEqual(check_readiness(connection), [])
        self.assertFalse(connection.closed)
        self.assertEqual(connection.info.transaction_status, self.psycopg.pq.TransactionStatus.INTRANS)
        connection.rollback()
        after = (self.admin.execute("SELECT * FROM products ORDER BY id").fetchall(), self.outbox_rows())
        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
