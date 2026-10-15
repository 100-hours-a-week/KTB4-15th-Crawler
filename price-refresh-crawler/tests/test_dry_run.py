import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from decimal import Decimal
from pathlib import Path
from unittest.mock import Mock, patch

from price_refresh.comparison import compare_prices
from price_refresh.dry_run import run_dry_run
from price_refresh.main import main
from price_refresh.repository import (
    DatabaseSettings, FixturePriceRepository, PostgresPriceRepository, RepositoryError,
)
from price_refresh.scanner import ScanResult, collect_prices
from price_refresh.settings import Settings
from price_refresh.source.categories import CATEGORIES
from price_refresh.source.client import MOST_REVIEWED, RECOMMENDED

FIXTURES = Path(__file__).parent / "fixtures"
SECRET = "postgresql://private-user:private-password@private-host/private-db"


class ComparisonTests(unittest.TestCase):
    def test_rises_falls_equal_absent_unobserved_and_zero(self):
        result = compare_prices([(1, 100), (2, 100), (3, 100), (4, 100), (6, 100), (7, 0)],
                                {"1": 200, "2": 50, "3": 100, "5": 999, "6": 0, "7": 10})
        self.assertEqual([(r.product_code, r.old_price, r.new_price) for r in result.changes],
                         [("1", 100, 200), ("2", 100, 50), ("6", 100, 0), ("7", 0, 10)])

    def test_exact_numeric_id_normalization_and_deterministic_numeric_order(self):
        result = compare_prices([("09007199254740993", 1), ("0010", 1), (2, 1)],
                                {9007199254740993: 2, "0002": 2, "10": 2})
        self.assertEqual([r.product_code for r in result.changes], ["2", "10", "9007199254740993"])

    def test_invalid_current_price_never_becomes_zero_or_truncated(self):
        for price in (None, -1, 10.5, 10.0, "10", True, Decimal("10")):
            with self.subTest(price=price):
                self.assertEqual(compare_prices([(1, 100)], {"1": price}).changes, [])

    def test_invalid_stored_prices_and_codes_are_excluded_without_coercion(self):
        for value in (None, -1, 10.5, 10.0, "10", True, Decimal("10")):
            with self.subTest(value=value):
                result = compare_prices([(1, value), (True, 10), ("bad", 10)], {"1": 0})
                self.assertEqual(result.changes, [])
                self.assertEqual(result.invalid_stored_rows, 3)

    def test_conflicting_stored_prices_are_not_arbitrarily_selected(self):
        with self.assertRaises(ValueError):
            compare_prices([(1, 100), ("01", 200)], {"1": 50})


class DryRunTests(unittest.TestCase):
    def repository(self, prices, flags):
        repo = Mock()
        repo.fetch_prices.return_value = prices
        repo.fetch_recommendations.return_value = flags
        return repo

    def test_true_targets_only_false_remains_in_all_changes_and_preview_count(self):
        repo = self.repository([(1, 100), (2, 100)], {1: True, "02": False})
        result = run_dry_run(repo, lambda: ScanResult(prices={"1": 200, "2": 50}))
        self.assertEqual(result["all_change_count"], 2)
        self.assertEqual(result["backend_targets"], [result["all_changes"][0]])
        self.assertEqual(result["backend_request_preview"],
                         {"changed_count": 1, "products": result["backend_targets"]})
        self.assertEqual(result["backend_target_count"], 1)
        self.assertEqual(result["recommendation_lookup"]["status"], "success")
        self.assertTrue(all(type(row["product_code"]) is str for row in result["all_changes"]))
        repo.fetch_recommendations.assert_called_once_with(["1", "2"])

    def test_no_changes_skips_recommendation_query(self):
        repo = self.repository([(1, 100)], {})
        result = run_dry_run(repo, lambda: ScanResult(prices={"1": 100}))
        repo.fetch_recommendations.assert_not_called()
        self.assertEqual(result["recommendation_lookup"]["status"], "skipped_no_changes")
        self.assertEqual(result["backend_targets"], [])
        self.assertEqual(result["backend_target_count"], 0)

    def test_successful_zero_targets_differs_from_failure(self):
        repo = self.repository([(1, 100)], {"1": False})
        result = run_dry_run(repo, lambda: ScanResult(prices={"1": 200}))
        self.assertEqual(result["recommendation_lookup"]["status"], "success")
        self.assertEqual(result["backend_targets"], [])
        self.assertEqual(result["backend_request_preview"], {"changed_count": 0, "products": []})

    def test_recommendation_failures_preserve_changes_and_hide_credentials(self):
        for error in (RepositoryError("recommendation_column_missing"), RuntimeError(SECRET)):
            with self.subTest(error_type=type(error).__name__):
                repo = self.repository([(1, 100)], {})
                repo.fetch_recommendations.side_effect = error
                with self.assertLogs("price_refresh.dry_run", level="ERROR") as logs:
                    result = run_dry_run(repo, lambda: ScanResult(prices={"1": 200}))
                self.assertEqual(result["all_change_count"], 1)
                self.assertEqual(result["recommendation_lookup"]["status"], "failed")
                for name in ("backend_targets", "backend_target_count", "backend_request_preview"):
                    self.assertIsNone(result[name])
                self.assertNotIn(SECRET, json.dumps(result) + str(logs.output))

    def test_missing_invalid_or_extra_recommendation_rows_are_not_false(self):
        for flags in ({}, {"1": None}, {"1": 1}, {"1": "true"}, {"1": True, "2": True}):
            with self.subTest(flags=flags):
                repo = self.repository([(1, 100)], flags)
                with self.assertLogs("price_refresh.dry_run", level="ERROR"):
                    result = run_dry_run(repo, lambda: ScanResult(prices={"1": 200}))
                self.assertEqual(result["recommendation_lookup"]["status"], "failed")
                self.assertIsNone(result["backend_targets"])

    def test_read_scan_recommendation_order_and_existing_scanner_maximum(self):
        events = []
        repo = self.repository([(1, 50)], {"1": True})
        repo.fetch_prices.side_effect = lambda: events.append("read_prices") or [(1, 50)]
        repo.fetch_recommendations.side_effect = lambda _: events.append("read_flags") or {"1": True}

        class Client:
            def fetch_listing(self, *, sort, page, **kwargs):
                events.append("scan")
                values = ({"itemId": 1, "itemInfo": {"displayPrice": 100 if sort == RECOMMENDED else 200}},)
                return {"data": {"list": list(values) if page == 1 else []}}

        result = run_dry_run(repo, lambda: collect_prices(Client(), CATEGORIES[:2], Settings(), sleep=lambda _: None))
        self.assertEqual(events[0], "read_prices")
        self.assertEqual(events[-1], "read_flags")
        self.assertEqual(result["all_changes"], [{"product_code": "1", "old_price": 50, "new_price": 200}])

    def test_initial_read_failure_stops_before_scan(self):
        repo = self.repository([], {})
        repo.fetch_prices.side_effect = RepositoryError("database_read_failed")
        collect = Mock()
        with self.assertRaises(RepositoryError):
            run_dry_run(repo, collect)
        collect.assert_not_called()
        repo.fetch_recommendations.assert_not_called()


class FakeCursor:
    def __init__(self, connection, results, failure=None):
        self.connection = connection
        self.results = iter(results)
        self.failure = failure
        self.calls = []
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.closed = True

    def execute(self, sql, params=None):
        assert self.connection.read_only is True
        assert sql.startswith("SELECT "), "Only SELECT is authorized"
        self.calls.append((sql, params))
        if self.failure:
            raise self.failure

    def fetchall(self):
        return next(self.results)


class FakeConnection:
    def __init__(self, results, failure=None):
        self.read_only = False
        self.closed = False
        self.cur = FakeCursor(self, results, failure)

    def cursor(self):
        return self.cur

    def close(self):
        self.closed = True

    def commit(self):
        raise AssertionError("No commits allowed")


class RepositoryTests(unittest.TestCase):
    def repository(self, connection, batch_size=1000):
        factory = Mock(return_value=connection)
        repo = PostgresPriceRepository(DatabaseSettings(SECRET, recommendation_batch_size=batch_size), connect=factory)
        return repo, factory

    def test_initial_select_is_only_codes_prices_and_closes_without_commit(self):
        connection = FakeConnection([[(1, 100)]])
        repo, factory = self.repository(connection)
        self.assertEqual(repo.fetch_prices(), [(1, 100)])
        self.assertEqual(connection.cur.calls, [("SELECT product_code, price FROM products", None)])
        factory.assert_called_once_with(SECRET, connect_timeout=5, autocommit=False)
        self.assertTrue(connection.closed and connection.cur.closed)

    def test_recommendations_are_batched_parameterized_and_use_one_connection(self):
        connection = FakeConnection([[(2, False), (1, True)], [(3, True)]])
        repo, factory = self.repository(connection, batch_size=2)
        self.assertEqual(repo.fetch_recommendations(["3", "1", "2"]), {"2": False, "1": True, "3": True})
        self.assertEqual([params for _, params in connection.cur.calls], [([1, 2],), ([3],)])
        self.assertTrue(all("ANY(%s)" in sql for sql, _ in connection.cur.calls))
        factory.assert_called_once()
        self.assertTrue(connection.closed and connection.cur.closed)

    def test_empty_codes_do_not_open_connection(self):
        repo, factory = self.repository(FakeConnection([]))
        self.assertEqual(repo.fetch_recommendations([]), {})
        factory.assert_not_called()

    def test_missing_column_and_general_failures_close_and_sanitize(self):
        class MissingColumn(Exception):
            sqlstate = "42703"

        for failure, expected in ((MissingColumn(SECRET), "recommendation_column_missing"),
                                  (RuntimeError(SECRET), "database_read_failed")):
            with self.subTest(expected=expected):
                connection = FakeConnection([], failure)
                repo, _ = self.repository(connection)
                with self.assertRaises(RepositoryError) as caught:
                    repo.fetch_recommendations(["1"])
                self.assertEqual(caught.exception.code, expected)
                self.assertNotIn(SECRET, str(caught.exception))
                self.assertTrue(connection.closed and connection.cur.closed)

    def test_partial_batch_success_is_not_returned_after_later_failure(self):
        connection = FakeConnection([[(1, True)], []])
        repo, _ = self.repository(connection, batch_size=1)
        with self.assertRaises(RepositoryError) as caught:
            repo.fetch_recommendations(["1", "2"])
        self.assertEqual(caught.exception.code, "incomplete_recommendation_result")
        self.assertTrue(connection.closed)

    def test_connection_and_readonly_setup_failures_are_sanitized(self):
        factory = Mock(side_effect=RuntimeError(SECRET))
        repo = PostgresPriceRepository(DatabaseSettings(SECRET), connect=factory)
        with self.assertRaises(RepositoryError) as caught:
            repo.fetch_prices()
        self.assertNotIn(SECRET, str(caught.exception))

        class BadReadOnly:
            closed = False

            @property
            def read_only(self):
                return False

            @read_only.setter
            def read_only(self, value):
                raise RuntimeError(SECRET)

            def close(self):
                self.closed = True

        connection = BadReadOnly()
        repo, _ = self.repository(connection)
        with self.assertRaises(RepositoryError):
            repo.fetch_prices()
        self.assertTrue(connection.closed)

    def test_settings_and_bigint_bounds_reject_bad_input_without_exposing_url(self):
        with patch.dict(os.environ, {"DATABASE_URL": SECRET}, clear=True):
            self.assertEqual(DatabaseSettings.from_env().url, SECRET)
            self.assertNotIn(SECRET, repr(DatabaseSettings.from_env()))
        with patch.dict(os.environ, {"DATABASE_URL": "invalid://private-password"}, clear=True):
            with self.assertRaises(RepositoryError) as caught:
                DatabaseSettings.from_env()
            self.assertNotIn("private-password", str(caught.exception))
        repo, factory = self.repository(FakeConnection([]))
        for code in ("1 OR 1=1", "9223372036854775808"):
            with self.assertRaises(RepositoryError):
                repo.fetch_recommendations([code])
        factory.assert_not_called()


class DryRunCliTests(unittest.TestCase):
    def command(self, stored="stored_products.json"):
        return ["--dry-run", "--fixture", str(FIXTURES / "listing_response.json"),
                "--stored-products-fixture", str(FIXTURES / stored), "--category-limit", "1"]

    def test_offline_cli_never_opens_db_or_http_even_with_database_url_set(self):
        output = io.StringIO()
        with patch.dict(os.environ, {"DATABASE_URL": SECRET}, clear=True), \
                patch("price_refresh.repository.PostgresPriceRepository") as database, \
                patch("http.client.HTTPSConnection.connect", side_effect=AssertionError("No HTTP")), \
                redirect_stdout(output):
            code = main(self.command())
        database.assert_not_called()
        payload = json.loads(output.getvalue())
        self.assertEqual(code, 0)
        self.assertEqual(payload["all_change_count"], 2)
        self.assertEqual(payload["backend_target_count"], 1)
        self.assertEqual(payload["backend_request_preview"]["changed_count"], 1)
        self.assertNotIn(SECRET, output.getvalue())

    def test_missing_column_fixture_outputs_partial_result_and_nonzero_exit(self):
        output = io.StringIO()
        with patch.dict(os.environ, {}, clear=True), redirect_stdout(output), self.assertLogs(level="ERROR"):
            code = main(self.command("stored_products_missing_recommendation.json"))
        result = json.loads(output.getvalue())
        self.assertEqual(code, 1)
        self.assertEqual(result["all_change_count"], 1)
        self.assertEqual(result["recommendation_lookup"]["error"]["code"], "recommendation_column_missing")
        self.assertIsNone(result["backend_targets"])

    def test_live_db_adapter_path_runs_only_read_queries_with_fixture_scan(self):
        price_connection = FakeConnection([[(3263645, 40000)]])
        flag_connection = FakeConnection([[(3263645, True)]])
        factory = Mock(side_effect=[price_connection, flag_connection])
        repo = PostgresPriceRepository(DatabaseSettings(SECRET), connect=factory)
        output = io.StringIO()
        with patch.dict(os.environ, {"DATABASE_URL": SECRET}, clear=True), \
                patch("price_refresh.main.PostgresPriceRepository", return_value=repo), \
                patch("http.client.HTTPSConnection.connect", side_effect=AssertionError("No HTTP")), \
                redirect_stdout(output):
            code = main(["--dry-run", "--fixture", str(FIXTURES / "listing_response.json"), "--category-limit", "1"])
        self.assertEqual(code, 0)
        self.assertTrue(price_connection.closed and flag_connection.closed)
        calls = price_connection.cur.calls + flag_connection.cur.calls
        self.assertEqual(len(calls), 2)
        self.assertTrue(all(sql.startswith("SELECT ") for sql, _ in calls))

    def test_missing_database_config_does_not_scan_or_leak(self):
        with patch.dict(os.environ, {}, clear=True), patch("price_refresh.main.collect_prices") as scan, \
                self.assertLogs(level="ERROR") as logs:
            code = main(["--dry-run", "--fixture", str(FIXTURES / "listing_response.json")])
        self.assertEqual(code, 1)
        scan.assert_not_called()
        self.assertIn("database_url_missing", str(logs.output))

    def test_output_is_new_file_and_stable_across_runs(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {}, clear=True):
            results = []
            for name in ("a.json", "b.json"):
                output = Path(directory) / name
                self.assertEqual(main(self.command() + ["--output", str(output)]), 0)
                results.append(output.read_text())
            self.assertEqual(results[0], results[1])
            with self.assertLogs(level="ERROR"):
                self.assertEqual(main(self.command() + ["--output", str(output)]), 1)
            self.assertEqual(output.read_text(), results[1])

    def test_stored_fixture_requires_explicit_offline_dry_run(self):
        with redirect_stdout(io.StringIO()), patch("sys.stderr", new_callable=io.StringIO):
            with self.assertRaises(SystemExit):
                main(["--stored-products-fixture", str(FIXTURES / "stored_products.json")])


if __name__ == "__main__":
    unittest.main()
