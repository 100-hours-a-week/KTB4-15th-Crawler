"""Opt-in tests of sql/cleanup_price_change_outbox.sql ONLY on the disposable test database.

Rows and the cleanup run in one transaction, so CURRENT_TIMESTAMP is fixed and the
5-day boundary is exact. Never reads DATABASE_URL.
"""

import os
import unittest
import uuid
from pathlib import Path

TEST_URL = os.getenv("PRICE_REFRESH_TEST_DATABASE_URL", "")
SQL_DIR = Path(__file__).resolve().parents[1] / "sql"
CREATE = (SQL_DIR / "create_price_change_outbox.sql").read_text(encoding="utf-8")
CLEANUP = (SQL_DIR / "cleanup_price_change_outbox.sql").read_text(encoding="utf-8")
PASSWORD = "cleanup-test-only"


@unittest.skipUnless(TEST_URL, "requires isolated PRICE_REFRESH_TEST_DATABASE_URL; never DATABASE_URL")
class CleanupSqlTests(unittest.TestCase):
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
        self.admin.execute("INSERT INTO products (product_code, price) SELECT g, 1000 FROM generate_series(1, 3000) g")
        self.admin.execute(CREATE)

    def add(self, connection, code, status, *, completed=None, created="0 seconds"):
        """completed/created are SQL intervals before now(); now() == the cleanup's CURRENT_TIMESTAMP."""
        return connection.execute(
            "INSERT INTO price_change_outbox (scan_run_id, product_code, original_old_price, old_price, new_price, "
            "status, next_attempt_at, completed_at, created_at) VALUES (%s, %s, 100, 100, 200, %s, "
            "CASE WHEN %s IN ('PENDING', 'RETRY_WAIT') THEN now() END, "
            "CASE WHEN %s::interval IS NULL THEN NULL ELSE now() - %s::interval END, now() - %s::interval) "
            "RETURNING id", (uuid.uuid4(), code, status, status, completed, completed, created)).fetchone()[0]

    def ids(self, connection):
        return {row[0] for row in connection.execute("SELECT id FROM price_change_outbox").fetchall()}

    def test_only_done_older_than_five_days_since_completion(self):
        with self.psycopg.connect(TEST_URL, connect_timeout=5) as connection:
            deleted = {self.add(connection, 1, "DONE", completed="6 days"),
                       self.add(connection, 2, "DONE", completed="5 days 1 microsecond", created="7 days")}
            kept = {self.add(connection, 3, "DONE", completed="5 days"),                     # exact boundary
                    self.add(connection, 4, "DONE", completed="4 days 23 hours"),
                    self.add(connection, 5, "DONE", completed="1 hour", created="300 days")}  # old row, recent done
            for code, status in enumerate(("PENDING", "RETRY_WAIT", "NOT_FOUND", "BLOCKED"), start=6):
                kept.add(self.add(connection, code, status, created="300 days"))
            before_products = connection.execute("SELECT product_code, price FROM products ORDER BY 1").fetchall()
            self.assertEqual(connection.execute(CLEANUP).fetchone()[0], 2)
            self.assertEqual(self.ids(connection), kept)
            self.assertTrue(deleted.isdisjoint(self.ids(connection)))
            self.assertEqual(connection.execute("SELECT product_code, price FROM products ORDER BY 1").fetchall(),
                             before_products)
            self.assertEqual(connection.execute(CLEANUP).fetchone()[0], 0)   # repeatable

    def test_batches_of_at_most_1000_oldest_first_until_done(self):
        with self.psycopg.connect(TEST_URL, connect_timeout=5) as connection:
            connection.execute(
                "INSERT INTO price_change_outbox (scan_run_id, product_code, original_old_price, old_price, "
                "new_price, status, next_attempt_at, completed_at) "
                "SELECT gen_random_uuid(), g, 100, 100, 200, 'DONE', NULL, now() - interval '10 days' - g * interval '1 second' "
                "FROM generate_series(1, 2500) g")
            kept = self.add(connection, 2501, "DONE", completed="1 day")
            counts = []
            for _ in range(10):   # bounded repetition, as documented
                counts.append(connection.execute(CLEANUP).fetchone()[0])
                if counts[-1] == 0:
                    break
            self.assertEqual(counts, [1000, 1000, 500, 0])
            self.assertEqual(self.ids(connection), {kept})

    def test_first_batch_takes_oldest_completed(self):
        with self.psycopg.connect(TEST_URL, connect_timeout=5) as connection:
            connection.execute(
                "INSERT INTO price_change_outbox (scan_run_id, product_code, original_old_price, old_price, "
                "new_price, status, next_attempt_at, completed_at) "
                "SELECT gen_random_uuid(), g, 100, 100, 200, 'DONE', NULL, now() - interval '6 days' - g * interval '1 minute' "
                "FROM generate_series(1, 1001) g")
            self.assertEqual(connection.execute(CLEANUP).fetchone()[0], 1000)
            self.assertEqual(connection.execute("SELECT product_code FROM price_change_outbox").fetchall(), [(1,)])

    def test_cleanup_role_with_only_select_and_delete(self):
        role = f"price_refresh_cleanup_{uuid.uuid4().hex[:8]}"
        self.admin.execute(f"CREATE ROLE {role} LOGIN PASSWORD '{PASSWORD}'")
        self.addCleanup(self.admin.execute, f"DROP ROLE {role}")
        self.addCleanup(self.admin.execute, f"DROP OWNED BY {role}")
        self.admin.execute(f"GRANT USAGE ON SCHEMA public TO {role}")
        self.admin.execute(f"GRANT SELECT, DELETE ON price_change_outbox TO {role}")
        old = self.add(self.admin, 1, "DONE", completed="30 days")
        recent = self.add(self.admin, 2, "DONE", completed="1 day")
        with self.psycopg.connect(TEST_URL, user=role, password=PASSWORD, autocommit=True,
                                  connect_timeout=5) as connection:
            self.assertEqual(connection.execute(CLEANUP).fetchone()[0], 1)
            with self.assertRaises(self.psycopg.Error) as raised:
                connection.execute("UPDATE price_change_outbox SET attempt_count = 0")
            self.assertEqual(raised.exception.sqlstate, "42501")
        self.assertEqual(self.ids(self.admin), {recent})
        self.assertNotIn(old, self.ids(self.admin))


if __name__ == "__main__":
    unittest.main()
