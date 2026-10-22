"""Offline failure injection; no driver, API or real database required."""

import io
import json
import logging
import unittest
import uuid
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from datetime import datetime, timezone
from unittest.mock import Mock, patch
from types import SimpleNamespace

from price_refresh import outbox
from price_refresh.apply import run_apply
from price_refresh.logging_config import KstFormatter
from price_refresh.main import main
from price_refresh.readiness import ReadinessError, ReadinessIssue
from price_refresh.repository import DatabaseSettings
from price_refresh.scanner import ScanResult
from price_refresh.writer import PostgresApplySession


class SqlError(Exception):
    def __init__(self, state):
        self.sqlstate = state
        super().__init__("private driver detail must never be logged")


class FakeConnection:
    """Transaction snapshots model rollback; the session lock is independent."""

    def __init__(self, prices, *, busy=False):
        self.prices = dict(prices)
        self.busy = busy
        self.locked = False
        self.closed = self.broken = False
        self.pid = 123
        self.statements = []
        self.updates = []
        self.commits = 0
        self.rollbacks = 0
        self.failure_codes = {}
        self.commit_errors = []
        self.before_commit = None
        self.initial_rows = None
        self.flags = {}
        self.outbox = []
        self.outbox_errors = {}
        self.readiness = []
        self.readiness_calls = 0
        self.autocommit = True
        self.info = SimpleNamespace(transaction_status=0)

    @contextmanager
    def cursor(self):
        connection = self

        class Cursor:
            rowcount = 0
            value = None

            def execute(self, sql, params=None):
                connection.statements.append((sql, params))
                if connection.closed:
                    raise SqlError("08006")
                if "pg_try_advisory_lock" in sql:
                    connection.locked = not connection.busy
                    self.value = (connection.locked, connection.pid)
                elif "pg_advisory_unlock" in sql:
                    connection.locked = False
                    self.value = (True,)
                elif "pg_locks" in sql:
                    self.value = (connection.locked and params[0] == connection.pid,)
                elif sql == "SELECT product_code, price FROM products":
                    self.value = (connection.initial_rows if connection.initial_rows is not None
                                  else list(connection.prices.items()))
                elif sql.startswith("SELECT product_code, has_been_recommended"):
                    self.value = [(code, connection.flags.get(code, False)) for code in params[0]
                                  if code in connection.prices]
                elif "FOR UPDATE" in sql:
                    self.value = ((connection.prices[params[0]], connection.flags.get(params[0], False))
                                  if params[0] in connection.prices else None)
                elif sql.startswith("INSERT INTO public.price_change_outbox"):
                    run_id, code, original_old, old, new = params
                    if code in connection.outbox_errors:
                        raise connection.outbox_errors[code]
                    now = datetime(2026, 10, 22, tzinfo=timezone.utc)
                    values = dict(id=len(connection.outbox) + 1, scan_run_id=run_id, product_code=code,
                                  original_old_price=original_old, old_price=old, new_price=new,
                                  status="PENDING", attempt_count=0, next_attempt_at=now,
                                  last_attempt_at=None, last_backend_price=None, last_http_status=None,
                                  last_error_code=None, last_error_message=None, created_at=now,
                                  updated_at=now, completed_at=None)
                    connection.outbox.append(tuple(values[name] for name in outbox.COLUMNS))
                    self.value = connection.outbox[-1]
                elif sql.startswith("UPDATE products SET price"):
                    price, code = params
                    connection.updates.append(code)
                    if code in connection.failure_codes:
                        raise SqlError(connection.failure_codes[code])
                    connection.prices[code] = price
                    self.rowcount = 1
                else:
                    raise AssertionError(sql)

            def fetchone(self):
                return self.value

            def fetchall(self):
                return self.value

        yield Cursor()

    @contextmanager
    def transaction(self):
        snapshot, outbox_snapshot = self.prices.copy(), list(self.outbox)
        self.info.transaction_status = 2
        try:
            yield
        except BaseException:
            self.prices, self.outbox = snapshot, outbox_snapshot
            self.rollbacks += 1
            raise
        else:
            if self.before_commit:
                self.before_commit()
            error = self.commit_errors.pop(0) if self.commit_errors else None
            if error:
                # A network COMMIT failure intentionally leaves outcome ambiguous.
                if getattr(error, "sqlstate", None) not in {None, "08006", "40003"}:
                    self.prices, self.outbox = snapshot, outbox_snapshot
                raise error
            self.commits += 1
        finally:
            self.info.transaction_status = 0

    def rollback(self):
        self.rollbacks += 1

    def close(self):
        self.locked = False
        self.closed = True


def fake_readiness(connection):
    # Runs only on the locked session, before the initial product read.
    connection.readiness_calls += 1
    assert connection.locked and "pg_locks" in connection.statements[-1][0]
    assert not any(sql.startswith("SELECT product_code, price") for sql, _ in connection.statements)
    if isinstance(connection.readiness, Exception):
        raise connection.readiness
    return connection.readiness


def make_session(connection):
    connect = Mock(return_value=connection)
    session = PostgresApplySession(DatabaseSettings("postgresql://unused/unused"), connect=connect,
                                   readiness=fake_readiness)
    return session, connect


class ApplyTests(unittest.TestCase):
    def run_case(self, connection, observed, *, after_read=None, emit=None):
        session, connect = make_session(connection)

        def collect():
            self.assertTrue(connection.locked)
            self.assertEqual(connection.updates, [])  # finalize scanner first
            if after_read:
                after_read()
            return ScanResult(prices=observed)

        report = run_apply(session, collect, emit=emit)
        self.assertTrue(connection.closed)
        self.assertFalse(connection.locked)
        connect.assert_called_once_with("postgresql://unused/unused", connect_timeout=5, autocommit=True)
        return report

    def test_busy_lock_skips_initial_read_scan_and_writes(self):
        connection = FakeConnection({1: 100}, busy=True)
        session, _ = make_session(connection)
        collect = Mock(side_effect=AssertionError("must not scan"))
        report = run_apply(session, collect)
        self.assertEqual(report.status, "skipped_concurrent")
        self.assertIsNone(report.target_count)
        collect.assert_not_called()
        self.assertEqual(len(connection.statements), 1)
        self.assertTrue(connection.closed)

    def test_fresh_old_price_rise_fall_zero_and_false_flag_writes_price_only(self):
        connection = FakeConnection({1: 100, 2: 100, 3: 100})
        report = self.run_case(connection, {"1": 200, "2": 50, "3": 0},
                               after_read=lambda: connection.prices.update({1: 150}))
        self.assertEqual([(c.product_code, c.old_price, c.new_price) for c in report.saved_changes],
                         [("1", 150, 200), ("2", 100, 50), ("3", 100, 0)])
        self.assertEqual(connection.commits, 3)
        self.assertEqual((connection.outbox, report.outbox_records), ([], []))
        self.assertEqual(report.to_dict()["outbox_record_count"], 0)
        self.assertTrue(all(params is not None for sql, params in connection.statements
                            if sql.startswith("UPDATE")))

    def test_same_at_save_time_and_deleted_product_are_not_successes(self):
        connection = FakeConnection({1: 100, 2: 100})

        def race():
            connection.prices[1] = 200
            del connection.prices[2]

        report = self.run_case(connection, {"1": 200, "2": 200}, after_read=race)
        self.assertEqual(report.saved_changes, [])
        self.assertEqual((report.skipped_same_count, report.missing_count), (1, 1))
        self.assertEqual(connection.updates, [])

    def test_row_failure_rolls_back_and_next_product_commits(self):
        connection = FakeConnection({1: 100, 2: 100, 3: 100})
        connection.failure_codes[2] = "23514"
        report = self.run_case(connection, {"1": 200, "2": 200, "3": 200})
        self.assertEqual(report.status, "partial_failure")
        self.assertEqual([c.product_code for c in report.saved_changes], ["1", "3"])
        self.assertEqual(connection.prices, {1: 200, 2: 100, 3: 200})
        self.assertEqual(report.failed_products,
                         [{"product_code": "2", "error": "sqlstate_23514", "category": "database"}])

    def test_known_commit_rejection_is_failure_and_can_continue(self):
        connection = FakeConnection({1: 100, 2: 100})
        connection.commit_errors = [SqlError("23514"), None]
        report = self.run_case(connection, {"1": 200, "2": 200})
        self.assertEqual([c.product_code for c in report.saved_changes], ["2"])
        self.assertEqual(connection.prices, {1: 100, 2: 200})
        self.assertEqual(report.to_dict()["unknown_count"], 0)

    def test_commit_unknown_aborts_excludes_product_and_keeps_previous_success(self):
        for error in (SqlError("08006"), SqlError("40003"), OSError("private connection detail")):
            with self.subTest(error_type=type(error).__name__):
                connection = FakeConnection({1: 100, 2: 100, 3: 100})
                connection.commit_errors = [None, error]
                with self.assertLogs("price_refresh.apply", level="ERROR") as logs:
                    report = self.run_case(connection, {"1": 200, "2": 200, "3": 200})
                self.assertEqual(report.status, "aborted")
                self.assertEqual([c.product_code for c in report.saved_changes], ["1"])
                self.assertEqual(connection.updates, [1, 2])
                self.assertEqual(report.to_dict()["unknown_count"], 1)
                self.assertEqual(report.to_dict()["failed_count"], 0)
                self.assertNotIn("private", str(logs.output) + json.dumps(report.to_dict()))

    def test_lock_lost_during_scan_stops_all_writes(self):
        for loss in ("closed", "locked", "pid"):
            with self.subTest(loss=loss):
                connection = FakeConnection({1: 100})

                def lose():
                    setattr(connection, loss, {"closed": True, "locked": False, "pid": 456}[loss])

                report = self.run_case(connection, {"1": 200}, after_read=lose)
                self.assertEqual(report.status, "aborted")
                self.assertEqual(connection.updates, [])

    def test_lock_lost_after_one_commit_stops_later_updates(self):
        connection = FakeConnection({1: 100, 2: 100})
        connection.before_commit = lambda: setattr(connection, "closed", True)
        report = self.run_case(connection, {"1": 200, "2": 200})
        self.assertEqual(report.status, "aborted")
        self.assertEqual(connection.updates, [1])

    def test_failure_while_saving_lost_connection_aborts_without_reconnect(self):
        connection = FakeConnection({1: 100, 2: 100})
        connection.failure_codes[1] = "08006"
        report = self.run_case(connection, {"1": 200, "2": 200})
        self.assertEqual(report.status, "aborted")
        self.assertEqual(report.saved_changes, [])
        self.assertEqual(connection.updates, [1])
        self.assertEqual(report.to_dict()["unknown_count"], 0)  # COMMIT never attempted

    def test_success_list_is_empty_during_commit_and_summary_emit_holds_lock(self):
        connection = FakeConnection({1: 100, 2: 100})
        # Probe the report's append timing through the PriceChange list passed to asdict.
        appended = []
        from price_refresh.apply import ApplyReport
        report_object = ApplyReport(saved_changes=appended)
        connection.before_commit = lambda: self.assertEqual(len(appended), connection.commits)

        def emit(payload):
            self.assertTrue(connection.locked)
            self.assertFalse(connection.closed)
            self.assertEqual(payload["updated_count"], 2)
            self.assertEqual(payload["saved_changes"][0]["product_code"], "1")

        with patch("price_refresh.apply.ApplyReport", return_value=report_object):
            report = self.run_case(connection, {"1": 200, "2": 200}, emit=emit)
        self.assertEqual(len(report.saved_changes), 2)

    def test_target_counts_deduplicate_exclude_non_targets_and_invalid_observations(self):
        connection = FakeConnection({1: 100, 2: 100, 3: 100, 4: None})
        connection.initial_rows = [(1, 100), ("01", 100), (2, 100), (3, 100), (4, None), ("bad", 10)]
        report = self.run_case(connection, {"1": 200, "2": None, "4": 200, "999": 200})
        self.assertEqual((report.target_count, report.confirmed_count, report.unconfirmed_count), (4, 2, 2))
        self.assertEqual(report.to_dict()["updated_count"], 1)
        self.assertEqual(report.invalid_stored_row_count, 2)

    def test_invalid_prices_never_written_even_with_valid_code(self):
        for price in (None, -1, 1.1, 1.0, True, "100"):
            with self.subTest(price=price):
                connection = FakeConnection({1: 100})
                report = self.run_case(connection, {"1": price})
                self.assertEqual(connection.updates, [])
                self.assertEqual(report.confirmed_count, 0)

    def test_exception_during_scan_releases_lock_and_hides_exception_detail(self):
        connection = FakeConnection({1: 100})
        session, _ = make_session(connection)
        with self.assertLogs("price_refresh.apply", level="INFO") as logs:
            report = run_apply(session, Mock(side_effect=ValueError("private driver detail")))
        self.assertEqual(report.status, "aborted")
        self.assertTrue(connection.closed)
        self.assertNotIn("private", str(logs.output))

    def test_output_error_does_not_undo_committed_prices(self):
        connection = FakeConnection({1: 100})
        report = self.run_case(connection, {"1": 200}, emit=Mock(side_effect=OSError("private")))
        self.assertEqual(report.error, "result_output_failed")
        self.assertEqual(report.status, "aborted")
        self.assertEqual(connection.prices[1], 200)
        self.assertEqual(len(report.saved_changes), 1)

    def test_open_failure_closes_connection_and_does_not_scan(self):
        connection = FakeConnection({1: 100})
        connection.closed = True
        session, _ = make_session(connection)
        collect = Mock()
        report = run_apply(session, collect)
        self.assertEqual(report.error, "lock_connection_failed")
        collect.assert_not_called()
        self.assertTrue(connection.closed)


class PreflightAndOutboxTests(unittest.TestCase):
    run_case = ApplyTests.run_case

    def outbox_rows(self, connection):
        return [outbox.record_from_row(row) for row in connection.outbox]

    def test_preflight_runs_after_lock_and_before_read_and_scan(self):
        connection = FakeConnection({1: 100})
        report = self.run_case(connection, {"1": 200})
        self.assertEqual(connection.readiness_calls, 1)
        self.assertEqual(report.status, "completed")

    def test_preflight_problems_or_failure_stop_before_read_scan_and_writes(self):
        for readiness in ([ReadinessIssue("column_missing", "products.has_been_recommended")],
                          [ReadinessIssue("table_missing", "public.price_change_outbox")],
                          ReadinessError("readiness_check_failed")):
            with self.subTest(readiness=readiness):
                connection = FakeConnection({1: 100})
                connection.readiness = readiness
                session, _ = make_session(connection)
                collect = Mock(side_effect=AssertionError("must not scan"))
                emitted = []
                with self.assertLogs("price_refresh.apply", level="ERROR") as logs:
                    report = run_apply(session, collect, emit=emitted.append)
                self.assertEqual(report.status, "preflight_failed")
                collect.assert_not_called()
                self.assertFalse(any(sql.startswith(("SELECT product_code", "SELECT price", "UPDATE"))
                                     for sql, _ in connection.statements))
                self.assertTrue(connection.closed)
                self.assertFalse(connection.locked)
                self.assertEqual(emitted[0]["status"], "preflight_failed")
                self.assertNotIn("postgresql://", str(logs.output))
        self.assertEqual(report.error, "readiness_check_failed")

    def test_preflight_issues_are_reported_with_codes(self):
        connection = FakeConnection({1: 100})
        connection.readiness = [ReadinessIssue("privilege_missing", "INSERT price_change_outbox")]
        session, _ = make_session(connection)
        report = run_apply(session, Mock())
        self.assertEqual(report.error, "database_not_ready")
        self.assertEqual(report.readiness_issues,
                         [{"code": "privilege_missing", "target": "INSERT price_change_outbox"}])

    def test_recommended_change_commits_price_and_outbox_from_fresh_row_values(self):
        connection = FakeConnection({1: 100, 2: 100, 3: 100})
        connection.flags = {1: True, 2: False, 3: True}
        report = self.run_case(connection, {"1": 200, "2": 200, "3": 100},
                               after_read=lambda: connection.prices.update({1: 150}))
        self.assertEqual(connection.prices, {1: 200, 2: 200, 3: 100})
        records = self.outbox_rows(connection)
        self.assertEqual([(r.product_code, r.original_old_price, r.old_price, r.new_price, r.status)
                          for r in records], [("1", 150, 150, 200, "PENDING")])
        self.assertEqual(records[0].scan_run_id, uuid.UUID(report.run_id))
        self.assertEqual(report.outbox_records, [{"product_code": "1", "outbox_id": records[0].id}])
        payload = report.to_dict()
        self.assertEqual((payload["updated_count"], payload["outbox_record_count"]), (2, 1))
        self.assertEqual(connection.updates, [1, 2])
        self.assertEqual(connection.commits, 2)

    def test_same_price_never_writes_even_when_recommended(self):
        connection = FakeConnection({1: 100})
        connection.flags = {1: True}
        report = self.run_case(connection, {"1": 200}, after_read=lambda: connection.prices.update({1: 200}))
        self.assertEqual((connection.updates, connection.outbox), ([], []))
        self.assertEqual(report.skipped_same_count, 1)

    def test_one_run_id_for_all_records(self):
        connection = FakeConnection({1: 100, 2: 100})
        connection.flags = {1: True, 2: True}
        report = self.run_case(connection, {"1": 200, "2": 300})
        self.assertEqual({r.scan_run_id for r in self.outbox_rows(connection)}, {uuid.UUID(report.run_id)})

    def test_outbox_failure_rolls_back_price_and_next_product_continues(self):
        for error, code in ((SqlError("42501"), "sqlstate_42501"),
                            (outbox.OutboxError("outbox_record_conflict"), "outbox_record_conflict")):
            with self.subTest(code=code):
                connection = FakeConnection({1: 100, 2: 100})
                connection.flags = {1: True, 2: True}
                connection.outbox_errors[1] = error
                report = self.run_case(connection, {"1": 200, "2": 200})
                self.assertEqual(report.status, "partial_failure")
                self.assertEqual(connection.prices, {1: 100, 2: 200})
                self.assertEqual([r.product_code for r in self.outbox_rows(connection)], ["2"])
                self.assertEqual(report.failed_products,
                                 [{"product_code": "1", "error": code, "category": "outbox"}])
                self.assertEqual([c.product_code for c in report.saved_changes], ["2"])

    def test_null_or_wrong_type_flag_fails_product_without_assuming_false(self):
        for flag in (None, 1, "true"):
            with self.subTest(flag=flag):
                connection = FakeConnection({1: 100, 2: 100})
                connection.flags = {1: flag, 2: True}
                report = self.run_case(connection, {"1": 200, "2": 200})
                self.assertEqual(connection.prices, {1: 100, 2: 200})
                self.assertEqual(report.failed_products, [{"product_code": "1", "error": "invalid_recommendation_flag",
                                                           "category": "database"}])
                self.assertEqual(len(connection.outbox), 1)
                self.assertFalse(any(sql.startswith("UPDATE products SET has_been_recommended")
                                     for sql, _ in connection.statements))

    def test_connection_loss_during_outbox_insert_stops_later_writes(self):
        connection = FakeConnection({1: 100, 2: 100})
        connection.flags = {1: True, 2: True}
        connection.outbox_errors[1] = SqlError("08006")
        report = self.run_case(connection, {"1": 200, "2": 200})
        self.assertEqual(report.status, "aborted")
        self.assertEqual(report.error, "lock_session_lost")
        self.assertEqual(report.failed_products, [{"product_code": "1", "error": "lock_session_lost",
                                                   "category": "connection"}])
        self.assertEqual(connection.updates, [1])
        self.assertEqual((connection.prices, connection.outbox), ({1: 100, 2: 100}, []))

    def test_commit_unknown_with_outbox_counts_neither_price_nor_record(self):
        connection = FakeConnection({1: 100, 2: 100, 3: 100})
        connection.flags = {1: True, 2: True, 3: True}
        connection.commit_errors = [None, SqlError("08006")]
        report = self.run_case(connection, {"1": 200, "2": 200, "3": 200})
        self.assertEqual(report.status, "aborted")
        self.assertEqual([c.product_code for c in report.saved_changes], ["1"])
        self.assertEqual([r["product_code"] for r in report.outbox_records], ["1"])
        self.assertEqual(report.unknown_products, [{"product_code": "2", "error": "commit_outcome_unknown"}])
        self.assertEqual(connection.updates, [1, 2])


class ApplyCliTests(unittest.TestCase):
    def test_apply_rejects_fixture_and_conflicting_modes_before_any_database_or_scan(self):
        for arguments in (["--apply", "--fixture", "absent.json"],
                          ["--apply", "--stored-products-fixture", "absent.json"],
                          ["--apply", "--dry-run"]):
            with self.subTest(arguments=arguments), patch("price_refresh.main.PostgresApplySession") as session, \
                    patch("price_refresh.main.collect_prices") as collect, redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    main(arguments)
                self.assertEqual(error.exception.code, 2)
                session.assert_not_called()
                collect.assert_not_called()

    def test_default_and_offline_dry_run_never_open_apply_session(self):
        fixture = "tests/fixtures/listing_response.json"
        for args in (["--fixture", fixture, "--category-limit", "1"],
                     ["--dry-run", "--fixture", fixture, "--category-limit", "1",
                      "--stored-products-fixture", "tests/fixtures/stored_products.json"]):
            with self.subTest(args=args), patch("price_refresh.main.PostgresApplySession") as session, \
                    redirect_stdout(io.StringIO()) as output:
                self.assertEqual(main(args), 0)
                json.loads(output.getvalue())
                session.assert_not_called()

    def test_apply_cli_uses_locked_writer_without_backend_settings_or_http(self):
        connection = FakeConnection({1: 100})
        connection.flags = {1: True}
        session, _ = make_session(connection)
        with patch.dict("os.environ", {}, clear=True), \
                patch("price_refresh.main.PostgresApplySession", return_value=session), \
                patch("price_refresh.main.DatabaseSettings.from_env"), \
                patch("price_refresh.main.collect_prices", return_value=ScanResult(prices={"1": 200})), \
                patch("price_refresh.backend.BackendSettings.from_env", side_effect=AssertionError("no backend")), \
                patch("price_refresh.backend.BackendClient.send", side_effect=AssertionError("no backend HTTP")), \
                patch("price_refresh.delivery.BackendDelivery.deliver", side_effect=AssertionError("no delivery")), \
                patch("http.client.HTTPConnection.connect", side_effect=AssertionError("no HTTP")), \
                patch("http.client.HTTPSConnection.connect", side_effect=AssertionError("no HTTPS")), \
                redirect_stdout(io.StringIO()) as output:
            self.assertEqual(main(["--apply", "--category-limit", "1"]), 0)
        payload = json.loads(output.getvalue())
        self.assertEqual(payload["mode"], "apply")
        self.assertEqual((payload["updated_count"], payload["outbox_record_count"]), (1, 1))
        self.assertEqual(payload["saved_changes"],
                         [{"product_code": "1", "old_price": 100, "new_price": 200}])
        self.assertNotIn("backend", payload)
        self.assertTrue(connection.closed)

    def test_apply_cli_preflight_failure_exits_nonzero(self):
        connection = FakeConnection({1: 100})
        connection.readiness = [ReadinessIssue("table_missing", "public.price_change_outbox")]
        session, _ = make_session(connection)
        with patch("price_refresh.main.PostgresApplySession", return_value=session), \
                patch("price_refresh.main.DatabaseSettings.from_env"), \
                patch("price_refresh.main.collect_prices") as collect, \
                redirect_stdout(io.StringIO()) as output, self.assertLogs("price_refresh.apply", "ERROR"):
            self.assertEqual(main(["--apply"]), 1)
        collect.assert_not_called()
        self.assertEqual(json.loads(output.getvalue())["status"], "preflight_failed")

    def test_formatter_is_korean_time_even_when_container_uses_utc(self):
        record = logging.LogRecord("test", logging.INFO, "", 1, "hello", (), None)
        record.created = 0
        self.assertEqual(KstFormatter().formatTime(record), "1970-01-01T09:00:00.000+09:00")


if __name__ == "__main__":
    unittest.main()
