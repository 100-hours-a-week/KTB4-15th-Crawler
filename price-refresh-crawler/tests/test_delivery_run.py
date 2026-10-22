"""Outbox delivery settings, CLI wiring and transaction outcome classification, offline."""

import io
import unittest
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
from unittest.mock import Mock, patch

from price_refresh.backend import BackendClient
from price_refresh.delivery import DeliveryError, DeliveryRunReport, DeliverySettings, _Run
from price_refresh.main import main
from price_refresh.repository import DatabaseSettings
from price_refresh.writer import DeliverySession
from test_apply import FakeConnection, SqlError

NOW = datetime(2026, 10, 22, tzinfo=timezone.utc)
BACKEND_ENV = {"DATABASE_URL": "postgresql://unused/unused",
               "PRICE_REFRESH_BACKEND_BASE_URL": "https://backend.invalid",
               "PRICE_REFRESH_BACKEND_AUTH_HEADERS_JSON": '{"X-Internal-Api-Key":"test-only"}'}


class DeliverySettingsTests(unittest.TestCase):
    def test_defaults_and_backoff_cap(self):
        settings = DeliverySettings()
        self.assertEqual((settings.retry_delay_seconds, settings.retry_max_delay_seconds, settings.max_requests,
                          settings.max_batches), (300, 3600, 10, 20))
        delays = [settings.retry_at(NOW, attempt) - NOW for attempt in (0, 1, 2, 3, 4, 5, 50)]
        self.assertEqual([d.total_seconds() for d in delays], [300, 300, 600, 1200, 2400, 3600, 3600])

    def test_invalid_values(self):
        for kwargs in ({"retry_delay_seconds": 0}, {"retry_delay_seconds": True}, {"max_requests": 0},
                       {"max_requests": 1001}, {"retry_delay_seconds": 600, "retry_max_delay_seconds": 300},
                       {"retry_max_delay_seconds": 86401}, {"max_batches": 0}, {"max_batches": 10001}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                DeliverySettings(**kwargs)
        for env in ({"PRICE_REFRESH_DELIVERY_MAX_REQUESTS": "x"}, {"PRICE_REFRESH_DELIVERY_RETRY_DELAY_SECONDS": "1.5"}):
            with self.subTest(env=env), patch.dict("os.environ", env, clear=True), self.assertRaises(ValueError):
                DeliverySettings.from_env()


class TransactionOutcomeTests(unittest.TestCase):
    def run_for(self, connection):
        session = DeliverySession(DatabaseSettings("postgresql://unused/unused"), connect=Mock(return_value=connection))
        self.assertTrue(session.open())
        self.addCleanup(session.close)
        return _Run(session, Mock(), DeliverySettings(), lambda: NOW, DeliveryRunReport())

    def test_lost_commit_is_outcome_unknown_and_known_rejection_is_not(self):
        connection = FakeConnection({})
        connection.commit_errors = [SqlError("08006")]
        with self.assertRaises(DeliveryError) as raised:
            self.run_for(connection).write(lambda: None)
        self.assertEqual(raised.exception.code, "commit_outcome_unknown")
        connection = FakeConnection({})
        connection.commit_errors = [SqlError("23514")]
        with self.assertRaises(SqlError):
            self.run_for(connection).write(lambda: None)

    def test_http_requires_lock_and_no_open_transaction(self):
        connection = FakeConnection({})
        run = self.run_for(connection)
        run.assert_idle()
        connection.info.transaction_status = 2
        with self.assertRaises(Exception) as raised:
            run.assert_idle()
        self.assertEqual(raised.exception.code, "delivery_transaction_not_idle")


class DeliverCliTests(unittest.TestCase):
    def test_invalid_combinations_are_rejected_before_database(self):
        for args in (["--deliver", "--apply"], ["--deliver", "--dry-run"], ["--deliver", "--delivery-status"],
                     ["--deliver", "--resume-delivery"], ["--deliver", "--fixture", "x.json"],
                     ["--deliver", "--category-limit", "1"], ["--deliver", "--stored-products-fixture", "x.json"]):
            with self.subTest(args=args), patch("price_refresh.main.DatabaseSettings.from_env") as settings, \
                    patch("price_refresh.main.run_delivery") as run, redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as raised:
                    main(args)
                self.assertEqual(raised.exception.code, 2)
                settings.assert_not_called()
                run.assert_not_called()

    def test_configuration_errors_stop_before_database_attempts_or_http(self):
        broken = [{}, {k: v for k, v in BACKEND_ENV.items() if k != "PRICE_REFRESH_BACKEND_AUTH_HEADERS_JSON"},
                  {**BACKEND_ENV, "PRICE_REFRESH_BACKEND_BATCH_SIZE": "101"},
                  {**BACKEND_ENV, "PRICE_REFRESH_BACKEND_BASE_URL": "ftp://backend.invalid"},
                  {**BACKEND_ENV, "PRICE_REFRESH_DELIVERY_MAX_REQUESTS": "0"},
                  {k: v for k, v in BACKEND_ENV.items() if k != "DATABASE_URL"}]
        for env in broken:
            with self.subTest(keys=sorted(env)), patch.dict("os.environ", env, clear=True), \
                    patch("price_refresh.main.DeliverySession") as session, \
                    patch("price_refresh.main.run_delivery") as run, \
                    patch.object(BackendClient, "send", side_effect=AssertionError("no HTTP")), \
                    self.assertLogs(level="ERROR") as logs:
                self.assertEqual(main(["--deliver"]), 1)
                session.assert_not_called()
                run.assert_not_called()
                self.assertNotIn("test-only", str(logs.output))

    def test_exit_codes_and_no_29cm_settings(self):
        cases = (("completed", 0), ("skipped_concurrent", 0), ("auth_stopped", 3), ("completed_with_errors", 1),
                 ("auth_stop_not_saved", 1), ("failed", 1), ("outcome_unknown", 1), ("aborted", 1))
        for status, expected in cases:
            env = {**BACKEND_ENV, "PRICE_REFRESH_PAGE_SIZE": "not-a-number"}   # 29CM settings are never read
            with self.subTest(status=status), patch.dict("os.environ", env, clear=True), \
                    patch("price_refresh.main.collect_prices", side_effect=AssertionError("no 29CM")), \
                    patch("price_refresh.main.run_delivery", return_value=DeliveryRunReport(status=status)) as run, \
                    redirect_stdout(io.StringIO()):
                self.assertEqual(main(["--deliver"]), expected)
            session, client, settings = run.call_args.args
            self.assertIsInstance(session, DeliverySession)
            self.assertIsInstance(client, BackendClient)
            self.assertEqual(settings, DeliverySettings())


if __name__ == "__main__":
    unittest.main()
