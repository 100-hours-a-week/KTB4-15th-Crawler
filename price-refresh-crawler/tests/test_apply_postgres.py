import os
import unittest
import uuid
from pathlib import Path
from unittest.mock import Mock, patch

from price_refresh.apply import run_apply
from price_refresh.repository import DatabaseSettings, PostgresPriceRepository
from price_refresh.dry_run import run_dry_run
from price_refresh.scanner import ScanResult, collect_prices
from price_refresh.settings import Settings
from price_refresh.source.categories import CATEGORIES
from price_refresh.source.fixture_client import FixtureClient
from price_refresh.writer import PostgresApplySession

TEST_URL = os.getenv("PRICE_REFRESH_TEST_DATABASE_URL", "")
OUTBOX_SQL = Path(__file__).resolve().parents[1] / "sql" / "create_price_change_outbox.sql"


@unittest.skipUnless(TEST_URL, "requires isolated PRICE_REFRESH_TEST_DATABASE_URL; never DATABASE_URL")
class PostgresApplyTests(unittest.TestCase):
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
            "CREATE TABLE products (product_code BIGINT PRIMARY KEY, price INTEGER NOT NULL, "
            "has_been_recommended BOOLEAN NOT NULL DEFAULT FALSE, "
            "CHECK (product_code != 2 OR price != 999))")
        self.admin.execute("INSERT INTO products (product_code, price) VALUES (1,100),(2,100),(3,100)")
        self.admin.execute(OUTBOX_SQL.read_text(encoding="utf-8"))

    def session(self):
        session = PostgresApplySession(DatabaseSettings(TEST_URL))
        self.addCleanup(session.close)
        return session

    def prices(self):
        return dict(self.admin.execute("SELECT product_code, price FROM products").fetchall())

    def outbox(self):
        return self.admin.execute(
            "SELECT scan_run_id, product_code, original_old_price, old_price, new_price, status "
            "FROM price_change_outbox ORDER BY id").fetchall()

    def test_real_lock_competition_survives_commits_and_rollback(self):
        first = self.session()
        self.assertTrue(first.open())
        pid = first.backend_pid
        run_id = uuid.uuid4().hex
        self.assertEqual(first.write_one("1", 200, run_id).status, "updated")
        self.assertEqual(first.write_one("2", 999, run_id).status, "failed")
        first.assert_lock()
        self.assertEqual(first.backend_pid, pid)
        second = self.session()
        collect = Mock(side_effect=AssertionError("busy lock must not scan"))
        report = run_apply(second, collect)
        self.assertEqual(report.status, "skipped_concurrent")
        collect.assert_not_called()
        self.assertEqual(self.prices(), {1: 200, 2: 100, 3: 100})
        first.close()
        third = self.session()
        self.assertTrue(third.open())

    def test_real_row_rollback_previous_and_next_commits_and_false_flag(self):
        report = run_apply(self.session(), lambda: ScanResult(prices={"1": 200, "2": 999, "3": 0}))
        self.assertEqual(report.status, "partial_failure")
        self.assertEqual([c.product_code for c in report.saved_changes], ["1", "3"])
        self.assertEqual(self.prices(), {1: 200, 2: 100, 3: 0})
        self.assertEqual(report.failed_products[0]["error"], "sqlstate_23514")
        self.assertEqual(self.admin.execute(
            "SELECT COUNT(*) FROM products WHERE has_been_recommended").fetchone()[0], 0)
        self.assertEqual(self.outbox(), [])

    def test_real_fresh_row_price_same_and_missing_after_initial_read(self):
        def collect():
            self.admin.execute("UPDATE products SET price=150 WHERE product_code=1")
            self.admin.execute("UPDATE products SET price=200 WHERE product_code=2")
            self.admin.execute("DELETE FROM products WHERE product_code=3")
            return ScanResult(prices={"1": 200, "2": 200, "3": 200})

        report = run_apply(self.session(), collect)
        self.assertEqual([(c.product_code, c.old_price, c.new_price) for c in report.saved_changes],
                         [("1", 150, 200)])
        self.assertEqual((report.skipped_same_count, report.missing_count), (1, 1))

    def test_real_terminated_lock_session_aborts_before_updates_and_releases_lock(self):
        session = self.session()

        def collect():
            self.assertTrue(self.admin.execute("SELECT pg_terminate_backend(%s)",
                                              (session.backend_pid,)).fetchone()[0])
            return ScanResult(prices={"1": 200, "2": 200})

        report = run_apply(session, collect)
        self.assertEqual(report.status, "aborted")
        self.assertEqual(report.saved_changes, [])
        self.assertEqual(self.prices(), {1: 100, 2: 100, 3: 100})
        self.assertTrue(self.session().open())

    def test_real_explicit_lock_loss_does_not_reacquire_or_write(self):
        session = self.session()

        def collect():
            session.connection.execute("SELECT pg_advisory_unlock_all()")
            return ScanResult(prices={"1": 200})

        report = run_apply(session, collect)
        self.assertEqual(report.error, "lock_session_lost")
        self.assertEqual(self.prices()[1], 100)

    def test_real_disconnect_after_first_commit_stops_later_products(self):
        session = self.session()
        write_one = session.write_one

        def write_then_terminate(code, new_price, scan_run_id):
            result = write_one(code, new_price, scan_run_id)
            if code == "1":
                self.admin.execute("SELECT pg_terminate_backend(%s)", (session.backend_pid,))
            return result

        session.write_one = write_then_terminate
        report = run_apply(session, lambda: ScanResult(prices={"1": 200, "2": 200, "3": 200}))
        self.assertEqual(report.status, "aborted")
        self.assertEqual([c.product_code for c in report.saved_changes], ["1"])
        self.assertEqual(self.prices(), {1: 200, 2: 100, 3: 100})

    def test_real_missing_recommendation_column_or_outbox_blocks_before_scan(self):
        for statement, issue in (
                ("ALTER TABLE products DROP COLUMN has_been_recommended",
                 {"code": "column_missing", "target": "products.has_been_recommended"}),
                ("DROP TABLE price_change_outbox",
                 {"code": "table_missing", "target": "public.price_change_outbox"})):
            with self.subTest(statement=statement):
                self.setUp()
                self.admin.execute(statement)
                collect = Mock(side_effect=AssertionError("must not scan"))
                report = run_apply(self.session(), collect)
                self.assertEqual(report.status, "preflight_failed")
                self.assertEqual(report.readiness_issues, [issue])
                collect.assert_not_called()
                self.assertEqual(self.prices(), {1: 100, 2: 100, 3: 100})
                probe = self.session()
                self.assertTrue(probe.open())  # lock released
                probe.close()

    def test_real_deferred_commit_rejection_rolls_back_and_continues(self):
        self.admin.execute("""
            CREATE OR REPLACE FUNCTION reject_test_commit() RETURNS trigger LANGUAGE plpgsql AS $$
            BEGIN
              IF NEW.product_code = 1 THEN
                RAISE EXCEPTION 'test deferred constraint' USING ERRCODE = '23514';
              END IF;
              RETURN NEW;
            END $$
        """)
        self.admin.execute("""
            CREATE CONSTRAINT TRIGGER reject_commit AFTER UPDATE ON products
            DEFERRABLE INITIALLY DEFERRED FOR EACH ROW EXECUTE FUNCTION reject_test_commit()
        """)
        report = run_apply(self.session(), lambda: ScanResult(prices={"1": 200, "2": 200}))
        self.assertEqual(report.status, "partial_failure")
        self.assertEqual([c.product_code for c in report.saved_changes], ["2"])
        self.assertEqual(self.prices(), {1: 100, 2: 200, 3: 100})
        self.assertEqual(report.to_dict()["unknown_count"], 0)

    def test_real_fixture_scanner_and_readonly_dry_run(self):
        fixture = Path(__file__).parent / "fixtures" / "listing_response.json"

        def collect():
            return collect_prices(FixtureClient(fixture), CATEGORIES[:1], Settings(), sleep=lambda _: None)

        scan = collect()
        code = next(iter(scan.prices))
        price = scan.prices[code]
        self.admin.execute("INSERT INTO products(product_code, price) VALUES (%s,%s) "
                           "ON CONFLICT(product_code) DO UPDATE SET price=EXCLUDED.price", (int(code), price + 1))
        before = self.prices()
        preview = run_dry_run(PostgresPriceRepository(DatabaseSettings(TEST_URL)), collect)
        self.assertEqual(self.prices(), before)
        self.assertEqual(preview["all_change_count"], 1)
        report = run_apply(self.session(), collect)
        self.assertEqual(report.status, "completed")
        self.assertEqual(report.saved_changes[0].product_code, code)
        self.assertEqual(self.prices()[int(code)], price)

    def test_real_recommended_price_and_outbox_commit_together(self):
        self.admin.execute("UPDATE products SET has_been_recommended=TRUE WHERE product_code IN (1,2)")

        def collect():
            self.admin.execute("UPDATE products SET price=150 WHERE product_code=1")
            return ScanResult(prices={"1": 200, "2": 999, "3": 200})

        report = run_apply(self.session(), collect)
        self.assertEqual(report.status, "partial_failure")
        self.assertEqual(self.prices(), {1: 200, 2: 100, 3: 200})
        self.assertEqual(self.outbox(), [(uuid.UUID(report.run_id), 1, 150, 150, 200, "PENDING")])
        self.assertEqual((report.to_dict()["updated_count"], report.to_dict()["outbox_record_count"]), (2, 1))
        self.assertEqual(report.failed_products[0]["category"], "database")

    def test_real_outbox_insert_failure_rolls_back_price(self):
        self.admin.execute("UPDATE products SET has_been_recommended=TRUE")
        self.admin.execute("""
            CREATE OR REPLACE FUNCTION reject_outbox() RETURNS trigger LANGUAGE plpgsql AS $$
            BEGIN
              IF NEW.product_code = 1 THEN
                RAISE EXCEPTION 'test outbox rejection' USING ERRCODE = '23514';
              END IF;
              RETURN NEW;
            END $$
        """)
        self.admin.execute("CREATE TRIGGER reject_outbox BEFORE INSERT ON price_change_outbox "
                           "FOR EACH ROW EXECUTE FUNCTION reject_outbox()")
        report = run_apply(self.session(), lambda: ScanResult(prices={"1": 200, "2": 200}))
        self.assertEqual(self.prices(), {1: 100, 2: 200, 3: 100})
        self.assertEqual([row[1] for row in self.outbox()], [2])
        self.assertEqual(report.failed_products,
                         [{"product_code": "1", "error": "sqlstate_23514", "category": "outbox"}])

    def test_real_conflicting_outbox_record_is_not_swallowed(self):
        run = uuid.UUID("00000000-0000-4000-8000-0000000000aa")
        self.admin.execute("UPDATE products SET has_been_recommended=TRUE")
        self.admin.execute("INSERT INTO price_change_outbox (scan_run_id, product_code, original_old_price, "
                           "old_price, new_price) VALUES (%s, 1, 100, 100, 300)", (run,))
        with patch("price_refresh.apply.uuid.uuid4", return_value=run):
            report = run_apply(self.session(), lambda: ScanResult(prices={"1": 200, "2": 200}))
        self.assertEqual(report.run_id, run.hex)
        self.assertEqual(self.prices(), {1: 100, 2: 200, 3: 100})
        self.assertEqual([(row[1], row[4]) for row in self.outbox()], [(1, 300), (2, 200)])
        self.assertEqual(report.failed_products,
                         [{"product_code": "1", "error": "outbox_record_conflict", "category": "outbox"}])

    def test_real_null_recommendation_flag_fails_only_that_product(self):
        self.admin.execute("ALTER TABLE products ALTER COLUMN has_been_recommended DROP NOT NULL")
        self.admin.execute("UPDATE products SET has_been_recommended = NULL WHERE product_code = 1")
        report = run_apply(self.session(), lambda: ScanResult(prices={"1": 200, "2": 200}))
        self.assertEqual(self.prices(), {1: 100, 2: 200, 3: 100})
        self.assertEqual(report.failed_products[0]["error"], "invalid_recommendation_flag")
        self.assertEqual(self.outbox(), [])

    def test_real_terminated_session_during_recommended_writes_stops(self):
        self.admin.execute("UPDATE products SET has_been_recommended=TRUE")
        session = self.session()
        write_one = session.write_one

        def write_then_terminate(code, new_price, scan_run_id):
            result = write_one(code, new_price, scan_run_id)
            self.admin.execute("SELECT pg_terminate_backend(%s)", (session.backend_pid,))
            return result

        session.write_one = write_then_terminate
        report = run_apply(session, lambda: ScanResult(prices={"1": 200, "2": 200}))
        self.assertEqual(report.status, "aborted")
        self.assertEqual(self.prices(), {1: 200, 2: 100, 3: 100})
        self.assertEqual([row[1] for row in self.outbox()], [1])


if __name__ == "__main__":
    unittest.main()
