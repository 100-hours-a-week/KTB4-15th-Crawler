"""Backend contract, bounded delivery, and apply integration without external I/O."""

import io
import json
import unittest
from contextlib import redirect_stdout
from dataclasses import asdict
from unittest.mock import Mock, patch

from price_refresh.apply import run_apply
from price_refresh.backend import BackendError, BackendSettings, HttpResult, SUCCESS, PARTIAL, FAILED
from price_refresh.comparison import PriceChange
from price_refresh.delivery import BackendDelivery
from price_refresh.main import main
from price_refresh.repository import DatabaseSettings, RepositoryError
from price_refresh.scanner import ScanResult
from price_refresh.writer import SessionLost
from test_apply import FakeConnection, SqlError, make_session


def response(codes, *, failed=(), not_found=(), status=200, code=None):
    return HttpResult(status, {
        "code": code or (PARTIAL if failed or not_found else SUCCESS), "message": "test response",
        "data": {"received_count": len(codes), "failed_product_codes": list(failed),
                 "not_found_product_codes": list(not_found)},
    })


def changes(count):
    return [PriceChange(str(code), 100 + code, 200 + code) for code in range(1, count + 1)]


class FakeBackend:
    def __init__(self, scripted=(), *, size=100, on_send=None):
        self.settings = BackendSettings("https://backend.invalid", {"X-Test-Only": "test-value"}, size)
        self.scripted = list(scripted)
        self.calls = []
        self.on_send = on_send

    def send(self, products):
        self.calls.append(json.loads(json.dumps(products)))
        if self.on_send:
            self.on_send(products)
        outcome = self.scripted.pop(0) if self.scripted else response([p["product_code"] for p in products])
        if isinstance(outcome, Exception):
            raise outcome
        if callable(outcome):
            return outcome(products)
        return outcome


def fake_session(codes):
    session = Mock()
    session.assert_delivery_ready = Mock()
    session.fetch_recommendations.return_value = {code: True for code in codes}
    return session


class DeliveryTests(unittest.TestCase):
    def test_no_saved_changes_skips_lookup_and_http(self):
        session = fake_session([])
        client = FakeBackend()
        report = BackendDelivery(client).deliver(session, [])
        self.assertEqual(report.status, "skipped_no_saved_changes")
        session.fetch_recommendations.assert_not_called()
        self.assertEqual(client.calls, [])

    def test_true_only_and_false_is_not_a_failed_delivery(self):
        session = fake_session(["1", "2"])
        session.fetch_recommendations.return_value = {"1": True, "2": False}
        client = FakeBackend()
        report = BackendDelivery(client).deliver(session, changes(2))
        session.fetch_recommendations.assert_called_once_with(["1", "2"])
        self.assertEqual([p["product_code"] for p in client.calls[0]], ["1"])
        self.assertEqual(report.to_dict()["successful_count"], 1)
        self.assertEqual(report.target_count, 1)

    def test_recommendation_failure_incomplete_and_wrong_types_skip_http(self):
        for flags in (RepositoryError("recommendation_column_missing"), RuntimeError("private-secret"),
                      {}, {"1": None}, {"1": 1}, {"1": True, "2": True}):
            with self.subTest(flags_type=type(flags).__name__):
                session = fake_session(["1"])
                if isinstance(flags, Exception):
                    session.fetch_recommendations.side_effect = flags
                else:
                    session.fetch_recommendations.return_value = flags
                client = FakeBackend()
                with self.assertLogs("price_refresh.delivery", level="INFO") as logs:
                    report = BackendDelivery(client).deliver(session, changes(1))
                self.assertEqual(client.calls, [])
                self.assertIsNone(report.target_count)
                self.assertEqual(report.status, "recommendation_lookup_failed")
                self.assertNotIn("private-secret", str(logs.output) + json.dumps(report.to_dict()))

    def test_normal_zero_targets_differs_from_lookup_failure(self):
        session = fake_session(["1"])
        session.fetch_recommendations.return_value = {"1": False}
        client = FakeBackend()
        report = BackendDelivery(client).deliver(session, changes(1))
        self.assertEqual(report.recommendation_lookup, "success")
        self.assertEqual(report.status, "skipped_no_targets")
        self.assertEqual(report.target_count, 0)
        self.assertEqual(client.calls, [])

    def test_201_products_in_100_100_1_and_no_retry_on_success(self):
        saved = changes(201)
        client = FakeBackend()
        report = BackendDelivery(client).deliver(fake_session([c.product_code for c in saved]), saved)
        self.assertEqual([len(batch) for batch in client.calls], [100, 100, 1])
        self.assertEqual([p for batch in client.calls for p in batch], [asdict(c) for c in saved])
        self.assertEqual(report.to_dict()["initial_success_count"], 201)
        self.assertEqual(report.retry_request_count, 0)
        self.assertEqual(report.status, "completed")

    def test_backend_smaller_limit_and_duplicate_saved_entries(self):
        saved = changes(5)
        client = FakeBackend(size=2)
        report = BackendDelivery(client).deliver(fake_session([c.product_code for c in saved]), saved + saved)
        self.assertEqual([len(batch) for batch in client.calls], [2, 2, 1])
        self.assertEqual(report.target_count, 5)
        self.assertTrue(all(len({p["product_code"] for p in batch}) == len(batch) for batch in client.calls))

    def test_conflicting_saved_entries_fail_without_sending(self):
        client = FakeBackend()
        report = BackendDelivery(client).deliver(fake_session(["1"]),
            [PriceChange("1", 100, 200), PriceChange("1", 150, 200)])
        self.assertEqual(report.error, "invalid_saved_changes")
        self.assertEqual(client.calls, [])

    def test_partial_retry_only_explicit_failures_preserves_prices_and_counts(self):
        saved = changes(3)
        client = FakeBackend([response(["1", "2", "3"], failed=["2"], not_found=["3"])])
        report = BackendDelivery(client).deliver(fake_session(["1", "2", "3"]), saved)
        self.assertEqual(client.calls, [[asdict(c) for c in saved], [asdict(saved[1])]])
        result = report.to_dict()
        self.assertEqual((result["initial_success_count"], result["retry_product_count"],
                          result["retry_success_count"], result["successful_count"], result["not_found_count"]),
                         (1, 1, 1, 2, 1))
        self.assertEqual((result["failed_count"], result["unknown_count"]), (0, 0))

    def test_retry_failure_stays_failed_without_third_attempt(self):
        client = FakeBackend([response(["1"], failed=["1"]), response(["1"], failed=["1"])])
        report = BackendDelivery(client).deliver(fake_session(["1"]), changes(1))
        self.assertEqual(len(client.calls), 2)
        self.assertEqual(report.to_dict()["failed_count"], 1)
        self.assertEqual(report.to_dict()["unknown_count"], 0)

    def test_500_only_explicit_failed_retried_remainder_never_success(self):
        client = FakeBackend([response(["1", "2", "3"], failed=["2"], not_found=["3"],
                                       status=500, code=FAILED)])
        report = BackendDelivery(client).deliver(fake_session(["1", "2", "3"]), changes(3))
        self.assertEqual([p["product_code"] for p in client.calls[1]], ["2"])
        result = report.to_dict()
        self.assertEqual((result["initial_success_count"], result["retry_success_count"],
                          result["unknown_count"], result["not_found_count"]), (0, 1, 1, 1))
        self.assertEqual(result["unknown_codes"], ["1"])

    def test_timeout_connection_and_parse_errors_are_unknown_no_retry(self):
        for code in ("backend_timeout", "backend_connection_error", "backend_invalid_json"):
            with self.subTest(code=code):
                client = FakeBackend([BackendError(code, phase="response")])
                report = BackendDelivery(client).deliver(fake_session(["1", "2"]), changes(2))
                self.assertEqual(len(client.calls), 1)
                self.assertEqual((report.to_dict()["unknown_count"], report.to_dict()["failed_count"]), (2, 0))

    def test_retry_timeout_replaces_prior_known_failure_with_unknown(self):
        client = FakeBackend([response(["1"], failed=["1"]), BackendError("backend_timeout")])
        report = BackendDelivery(client).deliver(fake_session(["1"]), changes(1))
        self.assertEqual(len(client.calls), 2)
        self.assertEqual((report.to_dict()["failed_count"], report.to_dict()["unknown_count"]), (0, 1))

    def test_invalid_response_never_infers_success_or_retries(self):
        base = response(["1", "2"]).payload
        cases = [HttpResult(200, None), HttpResult(200, []), HttpResult(201, base),
                 HttpResult(500, base), response(["1", "2"], code=PARTIAL),
                 response(["1", "2"], failed=["1"], code=SUCCESS)]
        for field, values in (("failed_product_codes", (None, "1", [1], [True], ["01"], ["0"],
                                                       ["999"], ["1", "1"], [None], [{}])),
                              ("not_found_product_codes", (["2", "2"], ["bad"], 1)),
                              ("received_count", (1, True, "2", None))):
            for value in values:
                payload = json.loads(json.dumps(base))
                payload["data"][field] = value
                cases.append(HttpResult(200, payload))
        cases.append(response(["1", "2"], failed=["1"], not_found=["1"]))
        for missing in ("received_count", "failed_product_codes", "not_found_product_codes"):
            payload = json.loads(json.dumps(base))
            del payload["data"][missing]
            cases.append(HttpResult(200, payload))
        for result in cases:
            with self.subTest(status=result.status, payload=result.payload):
                client = FakeBackend([result])
                report = BackendDelivery(client).deliver(fake_session(["1", "2"]), changes(2))
                self.assertEqual(len(client.calls), 1)
                self.assertEqual(report.to_dict()["unknown_count"], 2)
                self.assertEqual(report.to_dict()["successful_count"], 0)

    def test_terminal_http_errors_no_retry_and_not_all_confirmed_failures(self):
        for status, code in ((400, "INVALID_REQUEST"), (401, "ANY_AUTH_ERROR"),
                             (403, "ANY_PERMISSION_ERROR"), (500, "INTERNAL_SERVER_ERROR"),
                             (502, FAILED), (302, SUCCESS)):
            with self.subTest(status=status):
                client = FakeBackend([response(["1"], failed=["1"], status=status, code=code)])
                report = BackendDelivery(client).deliver(fake_session(["1"]), changes(1))
                self.assertEqual(len(client.calls), 1)
                self.assertEqual(report.to_dict()["failed_count"], 0)
                self.assertEqual(report.to_dict()["unknown_count"], 1)

    def test_initial_batches_finish_before_retry_and_each_code_at_most_twice(self):
        def fail_all(products):
            codes = [p["product_code"] for p in products]
            return response(codes, failed=codes)
        saved = changes(201)
        client = FakeBackend([fail_all] * 6)
        report = BackendDelivery(client).deliver(fake_session([c.product_code for c in saved]), saved)
        self.assertEqual([len(batch) for batch in client.calls], [100, 100, 1, 100, 100, 1])
        self.assertEqual(client.calls[:3], client.calls[3:])
        self.assertEqual((report.retry_product_count, report.to_dict()["failed_count"]), (201, 201))

    def test_lock_loss_prevents_further_initial_batches_and_marks_unsent(self):
        session = fake_session([c.product_code for c in changes(201)])
        lost = False

        def guard():
            if lost:
                raise SessionLost("lock_session_lost")

        def on_send(_):
            nonlocal lost
            lost = True

        session.assert_delivery_ready.side_effect = guard
        client = FakeBackend(on_send=on_send)
        report = BackendDelivery(client).deliver(session, changes(201))
        self.assertEqual(report.status, "aborted")
        self.assertEqual(len(client.calls), 1)
        self.assertEqual((report.to_dict()["successful_count"], report.to_dict()["unsent_count"]), (100, 101))

    def test_lock_loss_during_retry_stops_remaining_retry_batches(self):
        session = fake_session(["1", "2", "3"])
        lost = False

        def guard():
            if lost:
                raise SessionLost("lock_session_lost")

        def on_send(_):
            nonlocal lost
            if len(client.calls) == 4:  # three first deliveries, then first retry
                lost = True

        def failed(products):
            codes = [p["product_code"] for p in products]
            return response(codes, failed=codes)

        session.assert_delivery_ready.side_effect = guard
        client = FakeBackend([failed] * 6, size=1, on_send=on_send)
        report = BackendDelivery(client).deliver(session, changes(3))
        self.assertEqual(report.status, "aborted")
        self.assertEqual(len(client.calls), 4)
        self.assertEqual((report.initial_request_count, report.retry_request_count, report.retry_product_count),
                         (3, 1, 1))
        self.assertEqual(report.to_dict()["failed_count"], 3)


class ApplyDeliveryTests(unittest.TestCase):
    def test_commit_unknown_never_enters_backend_phase(self):
        connection = FakeConnection({1: 100, 2: 100, 3: 100})
        connection.flags = {1: True, 2: True, 3: True}
        connection.commit_errors = [None, SqlError("08006")]
        session, _ = make_session(connection)
        client = FakeBackend()
        report = run_apply(session, lambda: ScanResult(prices={"1": 200, "2": 200, "3": 200}),
                           delivery=BackendDelivery(client))
        self.assertEqual(report.status, "aborted")
        self.assertEqual([c.product_code for c in report.saved_changes], ["1"])
        self.assertEqual(report.to_dict()["unknown_count"], 1)
        self.assertEqual(report.backend["skip_reason"], "skipped_apply_aborted")
        self.assertEqual(client.calls, [])
        self.assertFalse(any("has_been_recommended" in sql for sql, _ in connection.statements))

    def test_only_committed_items_transmitted_under_lock_after_transactions_end(self):
        connection = FakeConnection({1: 100, 2: 100, 3: 100, 4: 100})
        connection.flags = {1: True, 2: True, 3: True, 4: False}
        connection.failure_codes[2] = "23514"
        session, _ = make_session(connection)

        def during_http(_):
            self.assertTrue(connection.locked)
            self.assertEqual(connection.info.transaction_status, 0)
            self.assertEqual(connection.prices, {1: 200, 2: 100, 3: 100, 4: 200})

        client = FakeBackend([response(["1"], failed=["1"])], on_send=during_http)
        emitted = []

        def emit(payload):
            self.assertTrue(connection.locked)
            self.assertEqual(len(client.calls), 2)
            emitted.append(payload)

        report = run_apply(session, lambda: ScanResult(prices={"1": 200, "2": 200, "3": 100, "4": 200}),
                           delivery=BackendDelivery(client), emit=emit)
        self.assertEqual([c.product_code for c in report.saved_changes], ["1", "4"])
        self.assertEqual(client.calls, [[{"product_code": "1", "old_price": 100, "new_price": 200}]] * 2)
        self.assertTrue(connection.closed)
        self.assertEqual(emitted[0]["backend"]["successful_count"], 1)
        queries = [(sql, params) for sql, params in connection.statements if "has_been_recommended" in sql]
        self.assertEqual(queries[0][1], ([1, 4],))

    def test_delivery_failure_does_not_rollback_committed_prices(self):
        connection = FakeConnection({1: 100})
        connection.flags[1] = True
        session, _ = make_session(connection)
        client = FakeBackend([BackendError("backend_timeout")])
        report = run_apply(session, lambda: ScanResult(prices={"1": 200}), delivery=BackendDelivery(client))
        self.assertEqual(report.status, "partial_failure")
        self.assertEqual(connection.prices[1], 200)
        self.assertEqual(connection.rollbacks, 0)
        self.assertEqual(len(report.saved_changes), 1)
        self.assertEqual(report.backend["unknown_count"], 1)

    def test_lost_lock_after_initial_failure_prevents_retry_and_aborts_apply(self):
        connection = FakeConnection({1: 100})
        connection.flags[1] = True
        session, _ = make_session(connection)
        client = FakeBackend([response(["1"], failed=["1"])],
                             on_send=lambda _: setattr(connection, "locked", False))
        report = run_apply(session, lambda: ScanResult(prices={"1": 200}), delivery=BackendDelivery(client))
        self.assertEqual(len(client.calls), 1)
        self.assertEqual(report.status, "aborted")
        self.assertEqual(report.backend["failed_count"], 1)
        self.assertEqual(report.backend["retry_request_count"], 0)
        self.assertEqual(connection.prices[1], 200)

    def test_recommendation_query_is_batched_and_parameterized_on_lock_session(self):
        connection = FakeConnection({1: 100, 2: 100, 3: 100})
        connection.flags = {1: True, 2: False, 3: True}
        session, _ = make_session(connection)
        session.settings = DatabaseSettings("postgresql://unused/unused", recommendation_batch_size=2)
        self.assertTrue(session.open())
        try:
            self.assertEqual(session.fetch_recommendations(["3", "1", "2"]),
                             {"1": True, "2": False, "3": True})
            queries = [(sql, params) for sql, params in connection.statements if "has_been_recommended" in sql]
            self.assertEqual([params for _, params in queries], [([1, 2],), ([3],)])
            self.assertTrue(all("ANY(%s)" in sql for sql, _ in queries))
        finally:
            session.close()

    def test_open_transaction_prevents_delivery(self):
        for autocommit, transaction_status in ((True, 2), (False, 0)):
            connection = FakeConnection({1: 100})
            session, _ = make_session(connection)
            session.open()
            connection.autocommit = autocommit
            connection.info.transaction_status = transaction_status
            client = FakeBackend()
            try:
                report = BackendDelivery(client).deliver(session, changes(1))
                self.assertEqual(report.status, "aborted")
                self.assertEqual(client.calls, [])
            finally:
                session.close()

    def test_missing_or_bad_backend_config_blocks_db_read_scan_and_write(self):
        environments = [{}, {"PRICE_REFRESH_BACKEND_BASE_URL": "https://backend.invalid"},
            {"PRICE_REFRESH_BACKEND_BASE_URL": "https://backend.invalid",
             "PRICE_REFRESH_BACKEND_AUTH_HEADERS_JSON": "{}"},
            {"PRICE_REFRESH_BACKEND_BASE_URL": "https://backend.invalid",
             "PRICE_REFRESH_BACKEND_AUTH_HEADERS_JSON": '{"X-Test":"test"}',
             "PRICE_REFRESH_BACKEND_BATCH_SIZE": "101"}]
        for env in environments:
            with self.subTest(keys=sorted(env)), patch.dict("os.environ", env, clear=True), \
                    patch("price_refresh.main.PostgresApplySession") as session, \
                    patch("price_refresh.main.collect_prices") as collect:
                self.assertEqual(main(["--apply"]), 1)
                session.assert_not_called()
                collect.assert_not_called()

    def test_dry_run_ignores_backend_settings_and_never_calls_client(self):
        with patch("price_refresh.main.BackendSettings.from_env", side_effect=AssertionError("no preflight")), \
                patch("price_refresh.backend.BackendClient.send", side_effect=AssertionError("no HTTP")), \
                redirect_stdout(io.StringIO()) as stdout:
            result = main(["--dry-run", "--fixture", "tests/fixtures/listing_response.json",
                           "--stored-products-fixture", "tests/fixtures/stored_products.json", "--category-limit", "1"])
        self.assertEqual(result, 0)
        self.assertEqual(json.loads(stdout.getvalue())["mode"], "dry_run")


if __name__ == "__main__":
    unittest.main()
