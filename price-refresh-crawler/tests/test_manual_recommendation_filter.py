"""Offline checks for network blocking and safe local-test restoration."""

import http.client
import unittest
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

from manual_recommendation_filter import RestoringLocalSession, deny_real_http
from price_refresh.writer import PostgresApplySession


class RecommendationHarnessTests(unittest.TestCase):
    def test_actual_http_is_blocked_before_socket_use(self):
        attempts = []
        with patch("socket.create_connection") as network, deny_real_http(attempts):
            with self.assertRaisesRegex(AssertionError, "real_http_forbidden"):
                http.client.HTTPSConnection("backend-disabled.invalid").connect()
            with self.assertRaisesRegex(AssertionError, "real_http_forbidden"):
                http.client.HTTPConnection("backend-disabled.invalid").request("PATCH", "/")
            network.assert_not_called()
        self.assertEqual(len(attempts), 2)

    def session(self, *, current_price=101):
        session = RestoringLocalSession("postgresql://unused", Path("unused.json"))
        session.originals = [{"product_code": "1", "price": 100, "has_been_recommended": False}]
        session.assert_lock = Mock()
        session.connection = MagicMock()
        cursor = session.connection.cursor.return_value.__enter__.return_value
        cursor.fetchone.return_value = (current_price, True)
        cursor.fetchall.return_value = [(1, 100, False)]
        return session, cursor

    def test_prices_and_flags_are_restored_and_session_closes(self):
        session, cursor = self.session()
        with patch.object(PostgresApplySession, "close") as close:
            session.close()
            close.assert_called_once_with()
        self.assertTrue(session.restored)
        updates = [call for call in cursor.execute.call_args_list if call.args[0].startswith("UPDATE")]
        self.assertEqual(len(updates), 1)
        self.assertEqual(updates[0].args[1], (100, False, 1))

    def test_run_outbox_rows_are_captured_and_deleted_before_price_restore(self):
        session, cursor = self.session()
        session.scan_run_id = "00000000000040008000000000000001"
        with patch.object(PostgresApplySession, "close"):
            session.close()
        self.assertTrue(session.restored)
        statements = [call.args[0].split()[0] for call in cursor.execute.call_args_list]
        self.assertLess(statements.index("DELETE"), statements.index("UPDATE"))
        delete = next(call for call in cursor.execute.call_args_list if call.args[0].startswith("DELETE"))
        self.assertEqual(str(delete.args[1][0]), "00000000-0000-4000-8000-000000000001")

    def test_unexpected_price_is_not_overwritten_during_cleanup(self):
        session, cursor = self.session(current_price=999)
        with patch.object(PostgresApplySession, "close") as close:
            session.close()
            close.assert_called_once_with()
        self.assertFalse(session.restored)
        self.assertIsNotNone(session.restore_error)
        self.assertFalse(any(call.args[0].startswith("UPDATE") for call in cursor.execute.call_args_list))

    def test_failed_restore_commit_is_not_reported_as_restored(self):
        session, _ = self.session()
        session.connection.transaction.return_value.__exit__.side_effect = RuntimeError("commit interrupted")
        with patch.object(PostgresApplySession, "close") as close:
            session.close()
            close.assert_called_once_with()
        self.assertFalse(session.restored)
        self.assertIsNotNone(session.restore_error)


if __name__ == "__main__":
    unittest.main()
