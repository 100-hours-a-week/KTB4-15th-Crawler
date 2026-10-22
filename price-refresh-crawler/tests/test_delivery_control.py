"""Delivery stop/resume CLI and lock separation, offline."""

import io
import json
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest.mock import Mock, patch

from price_refresh.delivery_control import DeliveryControlError, ResumeReport, fetch_delivery_status, run_resume
from price_refresh.main import main
from price_refresh.outbox import DeliveryStop
from price_refresh.repository import DatabaseSettings
from price_refresh.writer import COLLECT_LOCK_KEY, DELIVERY_LOCK_KEY, DeliverySession, PostgresApplySession
from test_apply import FakeConnection

MAX_BIGINT = 9223372036854775807


def no_network():
    stack = [patch("http.client.HTTPConnection.connect", side_effect=AssertionError("no HTTP")),
             patch("http.client.HTTPSConnection.connect", side_effect=AssertionError("no HTTPS")),
             patch("price_refresh.backend.BackendSettings.from_env", side_effect=AssertionError("no backend")),
             patch("price_refresh.main.collect_prices", side_effect=AssertionError("no 29CM"))]
    return stack


class LockKeyTests(unittest.TestCase):
    def test_collection_and_delivery_keys_differ_and_are_bigint(self):
        self.assertNotEqual(COLLECT_LOCK_KEY, DELIVERY_LOCK_KEY)
        self.assertEqual(PostgresApplySession.lock_key, COLLECT_LOCK_KEY)
        self.assertEqual(DeliverySession.lock_key, DELIVERY_LOCK_KEY)
        self.assertTrue(0 < DELIVERY_LOCK_KEY <= MAX_BIGINT)
        self.assertFalse(hasattr(DeliverySession, "write_one"))

    def test_delivery_session_uses_its_own_key(self):
        connection = FakeConnection({})
        session = DeliverySession(DatabaseSettings("postgresql://unused/unused"), connect=Mock(return_value=connection))
        self.assertTrue(session.open())
        session.assert_lock()
        session.close()
        keys = [params for sql, params in connection.statements if "advisory" in sql and "pg_locks" not in sql]
        self.assertEqual(keys, [(DELIVERY_LOCK_KEY,), (DELIVERY_LOCK_KEY,)])
        lock_check = next(params for sql, params in connection.statements if "pg_locks" in sql)
        self.assertEqual(lock_check[1:], (DELIVERY_LOCK_KEY >> 32, DELIVERY_LOCK_KEY & 0xffffffff))


class ResumeOfflineTests(unittest.TestCase):
    def test_busy_delivery_lock_never_reads_or_writes(self):
        connection = FakeConnection({}, busy=True)
        session = DeliverySession(DatabaseSettings("postgresql://unused/unused"), connect=Mock(return_value=connection))
        emitted = []
        report = run_resume(session, emit=emitted.append)
        self.assertEqual(report.status, "skipped_concurrent")
        self.assertEqual(len(connection.statements), 1)
        self.assertTrue(connection.closed)
        self.assertEqual(emitted[0]["status"], "skipped_concurrent")

    def test_status_check_failure_never_reports_allowed(self):
        connect = Mock(side_effect=OSError("postgresql://user:private-secret@db/x"))
        with self.assertRaises(DeliveryControlError) as raised:
            fetch_delivery_status(DatabaseSettings("postgresql://unused/unused"), connect=connect)
        self.assertEqual(str(raised.exception), "delivery_status_check_failed")
        self.assertIsNone(raised.exception.__cause__)


class DeliveryCliTests(unittest.TestCase):
    def run_cli(self, args, env=None):
        output = io.StringIO()
        with patch.dict("os.environ", env if env is not None else {"DATABASE_URL": "postgresql://unused/unused"},
                        clear=True), redirect_stdout(output):
            patches = no_network()
            for item in patches:
                item.start()
            try:
                code = main(args)
            finally:
                for item in patches:
                    item.stop()
        return code, json.loads(output.getvalue()) if output.getvalue() else None

    def test_invalid_combinations_are_rejected_before_database(self):
        for args in (["--delivery-status", "--apply"], ["--resume-delivery", "--dry-run"],
                     ["--delivery-status", "--resume-delivery"], ["--delivery-status", "--fixture", "x.json"],
                     ["--resume-delivery", "--category-limit", "1"],
                     ["--resume-delivery", "--stored-products-fixture", "x.json"]):
            with self.subTest(args=args), patch("price_refresh.main.DatabaseSettings.from_env") as settings, \
                    patch("price_refresh.main.fetch_delivery_status") as status, \
                    patch("price_refresh.main.run_resume") as resume, redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as raised:
                    main(args)
                self.assertEqual(raised.exception.code, 2)
                settings.assert_not_called()
                status.assert_not_called()
                resume.assert_not_called()

    def test_status_exit_codes_distinguish_allowed_stopped_and_failed(self):
        cases = ((DeliveryStop(0, ()), 0, "allowed", True), (DeliveryStop(2, (401, 403)), 3, "stopped", False))
        for stop, expected, status, allowed in cases:
            with self.subTest(status=status), patch("price_refresh.main.fetch_delivery_status", return_value=stop):
                code, payload = self.run_cli(["--delivery-status"])
            self.assertEqual(code, expected)
            self.assertEqual((payload["status"], payload["delivery_allowed"], payload["auth_failed_count"]),
                             (status, allowed, stop.auth_failed_count))
        self.assertEqual(payload["http_statuses"], [401, 403])
        with patch("price_refresh.main.fetch_delivery_status",
                   side_effect=DeliveryControlError("delivery_status_check_failed")), \
                self.assertLogs(level="ERROR"):
            code, payload = self.run_cli(["--delivery-status"])
        self.assertEqual((code, payload["status"], payload["delivery_allowed"]), (1, "check_failed", None))

    def test_resume_exit_codes(self):
        for status, expected in (("resumed", 0), ("already_allowed", 0), ("skipped_concurrent", 1),
                                 ("failed", 1), ("commit_outcome_unknown", 1)):
            with self.subTest(status=status), \
                    patch("price_refresh.main.run_resume", return_value=ResumeReport(status=status)) as resume:
                code, _ = self.run_cli(["--resume-delivery"])
            self.assertEqual(code, expected)
            self.assertIsInstance(resume.call_args.args[0], DeliverySession)

    def test_missing_database_url_fails_without_backend_settings(self):
        with self.assertLogs(level="ERROR"):
            for args in (["--delivery-status"], ["--resume-delivery"]):
                self.assertEqual(self.run_cli(args, env={})[0], 1)


if __name__ == "__main__":
    unittest.main()
