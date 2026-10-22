"""Offline verification ONLY. Discovery never runs the manual live measurement."""

import http.client
import itertools
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

from manual_timing import (
    LISTING_HOST, LISTING_PATH, TimingAbort, block_backend_network, measure,
    new_stats, read_host_db_config,
    inspect_local_db,
)
from price_refresh.backend import BackendClient, BackendSettings
from price_refresh.scanner import collect_prices
from price_refresh.settings import Settings
from price_refresh.source.categories import CATEGORIES
from price_refresh.source.fixture_client import FixtureClient
from test_apply import FakeConnection, make_session


class ManualTimingOfflineTests(unittest.TestCase):
    def test_real_backend_and_unexpected_connections_blocked_before_network(self):
        stats = new_stats()
        with patch.object(http.client.HTTPSConnection, "connect") as transport, \
                block_backend_network(stats, allow_listing=True):
            with self.assertRaises(TimingAbort):
                http.client.HTTPSConnection("backend-disabled.invalid").connect()
            with self.assertRaises(TimingAbort):
                BackendClient(BackendSettings("https://backend-disabled.invalid", {"X-Test": "not-auth"})).send(
                    [{"product_code": "1", "old_price": 100, "new_price": 200}])
            transport.assert_not_called()
        self.assertEqual(stats["blocked_http_attempts"], 2)
        self.assertEqual(stats["actual_backend_http_requests"], 0)

    def test_only_exact_listing_host_post_path_is_allowed(self):
        stats = new_stats()
        with patch.object(http.client.HTTPSConnection, "connect") as connect, \
                patch.object(http.client.HTTPConnection, "request") as request, \
                patch.object(http.client.HTTPConnection, "getresponse", return_value=SimpleNamespace(status=200)), \
                block_backend_network(stats, allow_listing=True):
            connection = http.client.HTTPSConnection(LISTING_HOST)
            connection.connect()
            connection.request("POST", LISTING_PATH)
            self.assertEqual(connection.getresponse().status, 200)
            with self.assertRaises(TimingAbort):
                connection.request("PATCH", "/api/v1/internal/products")
            connect.assert_called_once()
            request.assert_called_once()
        self.assertEqual(stats["listing_http_request_attempts"], 1)
        self.assertEqual(stats["http_status_counts"], {"200": 1})

    def run_fixture(self, recommended):
        connection = FakeConnection({3263645: 1, 3263646: 1})
        connection.flags = {3263645: recommended, 3263646: False}
        session, _ = make_session(connection)
        waits = []
        clock = lambda: next(ticks) / 10
        ticks = itertools.count()
        stats = new_stats()
        with block_backend_network(stats):
            result = measure(session, lambda: collect_prices(
                FixtureClient(Path(__file__).parent / "fixtures" / "listing_response.json"),
                CATEGORIES[:1], Settings(), sleep=waits.append), stats=stats, clock=clock)
        self.assertEqual(waits, [1.0] * 4)
        self.assertTrue(connection.closed)
        self.assertEqual(result["local_price_commit_count"], 2)
        self.assertEqual(stats["blocked_http_attempts"], 0)
        self.assertEqual(stats["actual_backend_http_requests"], 0)
        return result

    def test_storage_boundary_with_recommended_outbox_and_no_backend(self):
        result = self.run_fixture(True)
        self.assertEqual(result["run_apply_status"], "completed")
        self.assertEqual(result["boundary"], "price_and_outbox_storage_complete")
        self.assertEqual(result["local_outbox_record_count"], 1)
        segments = result["segments_seconds"]
        elapsed = sum(segments[name] for name in (
            "lock_preflight_and_initial_db_read", "actual_collection", "comparison_price_and_outbox_storage"))
        self.assertAlmostEqual(elapsed, result["start_to_boundary_seconds"])
        self.assertGreater(result["program_run_apply_total_seconds"], elapsed)

    def test_not_recommended_changes_create_no_outbox_records(self):
        result = self.run_fixture(False)
        self.assertEqual(result["boundary"], "price_and_outbox_storage_complete")
        self.assertEqual(result["local_outbox_record_count"], 0)

    def test_lookup_abort_releases_lock_and_does_not_invent_boundary(self):
        connection = FakeConnection({1: 100})
        session, _ = make_session(connection)
        result = measure(session, Mock(side_effect=TimingAbort("stop_after_failed_lookup")))
        self.assertTrue(result["measurement_aborted"])
        self.assertEqual(result["boundary"], "not_reached")
        self.assertIsNone(result["start_to_boundary_seconds"])
        self.assertEqual(result["local_price_commit_count"], 0)
        self.assertTrue(connection.closed)
        self.assertFalse(connection.locked)

    def test_host_config_refuses_nonlocal_or_routing_override_before_socket_check(self):
        for url in ("postgresql://user:private-secret@remote.invalid/db",
                    "postgresql://user:private-secret@localhost/db?hostaddr=203.0.113.5"):
            with tempfile.TemporaryDirectory() as directory:
                filename = Path(directory) / "local.env"
                filename.write_text("DATABASE_URL=" + url)
                with patch("manual_timing.subprocess.run") as child:
                    with self.assertRaises(ValueError) as error:
                        read_host_db_config(filename)
                    self.assertNotIn("private-secret", str(error.exception))
                    child.assert_not_called()

    def test_missing_recommendation_column_is_readonly_and_blocks_live_prerequisite(self):
        connection = MagicMock()
        connection.__enter__.return_value = connection
        cursor = MagicMock()
        connection.cursor.return_value.__enter__.return_value = cursor
        cursor.fetchone.side_effect = [(123, 5432, "PostgreSQL test on test"), (True,), (50000, 50000)]
        cursor.fetchall.return_value = [("product_code", "bigint"), ("price", "integer")]

        def read_only_sql(sql, *_):
            self.assertTrue(sql.startswith("SELECT "))

        cursor.execute.side_effect = read_only_sql
        driver = SimpleNamespace(connect=Mock(return_value=connection))
        conninfo = SimpleNamespace(conninfo_to_dict=lambda _: {"host": "host.docker.internal", "port": "5432"})
        with patch.dict(sys.modules, {"psycopg": driver, "psycopg.conninfo": conninfo}), \
                patch.dict(os.environ, {"PRICE_REFRESH_TIMING_LOCAL_DB_CONFIRMED": "1"}):
            result = inspect_local_db("postgresql://test@host.docker.internal:5432/test")
        self.assertFalse(result["ready"])
        self.assertIsNone(result["required_columns"]["has_been_recommended"])
        self.assertEqual(result["unique_codes"], 50000)
        self.assertTrue(connection.read_only)
        connection.__exit__.assert_called_once()


if __name__ == "__main__":
    unittest.main()
