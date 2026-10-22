"""New price_conflicts contract: request validation and response classification, offline."""

import copy
import json
import socket
import unittest
import uuid
from datetime import datetime, timezone
from unittest.mock import Mock

from price_refresh import price_sync
from price_refresh.backend import BackendClient, BackendError, BackendSettings, HttpResult
from price_refresh.outbox import OutboxRecord
from price_refresh.price_sync import (
    AUTH_REJECTED, COMPLETED, INVALID_RESPONSE, NOT_SENT, REQUEST_REJECTED, TEMPORARY_FAILURE,
    UNEXPECTED_STATUS, PriceConflict, RejectedItem, SyncItem, classify_send_error, parse_sync_response,
    prepare_batch,
)

SUCCESS = "PRODUCT_PRICE_UPDATE_SUCCESS"
PARTIAL = "PRODUCT_PRICE_UPDATE_PARTIAL_SUCCESS"
NOW = datetime(2026, 10, 22, tzinfo=timezone.utc)


def record(record_id, code, old, new):
    return {"id": record_id, "product_code": code, "old_price": old, "new_price": new}


def batch(*records):
    return prepare_batch(records or [record(1, "1234567", 89000, 79000), record(2, "2345678", 69000, 75000),
                                     record(3, "3456789", 10000, 12000)])


def body(code=None, *, received=..., not_found=(), conflicts=(), sent=3, **extra):
    data = {"received_count": sent if received is ... else received,
            "not_found_product_codes": list(not_found),
            "price_conflicts": [{"product_code": c, "current_price": p} for c, p in conflicts], **extra}
    return {"code": code or (PARTIAL if not_found or conflicts else SUCCESS), "message": "test", "data": data}


def codes(result):
    return result.succeeded, result.not_found, tuple(c.product_code for c in result.conflicts), result.unresolved


class PrepareBatchTests(unittest.TestCase):
    def test_valid_records_build_body_from_sent_products_only(self):
        prepared = batch()
        self.assertTrue(prepared.sendable)
        self.assertEqual(prepared.rejected, ())
        self.assertEqual(prepared.body(), {"changed_count": 3, "products": [
            {"product_code": "1234567", "old_price": 89000, "new_price": 79000},
            {"product_code": "2345678", "old_price": 69000, "new_price": 75000},
            {"product_code": "3456789", "old_price": 10000, "new_price": 12000}]})
        self.assertEqual(prepared.items[0], SyncItem(1, "1234567", 89000, 79000))

    def test_outbox_records_are_accepted(self):
        outbox_record = OutboxRecord(5, uuid.uuid4(), "7", 0, 100, 200, "PENDING", 0, NOW, None, None, None,
                                     None, None, NOW, NOW, None)
        self.assertEqual(prepare_batch([outbox_record]).items, (SyncItem(5, "7", 100, 200),))

    def test_invalid_prices_are_rejected_individually(self):
        for price in (0, -1, True, False, 1.0, 1.5, "100", None, 2 ** 31):
            with self.subTest(price=price):
                prepared = prepare_batch([record(1, "1", price, 200), record(2, "2", 100, price),
                                          record(3, "3", 100, 200)])
                self.assertEqual(prepared.rejected, (RejectedItem(1, "1", "invalid_price"),
                                                     RejectedItem(2, "2", "invalid_price")))
                self.assertEqual(prepared.body()["changed_count"], 1)
        self.assertEqual(prepare_batch([record(1, "1", 2 ** 31 - 1, 1)]).items[0].old_price, 2 ** 31 - 1)

    def test_same_price_missing_field_and_bad_code(self):
        prepared = prepare_batch([
            record(1, "1", 100, 100),
            {"id": 2, "product_code": "2", "old_price": 100},
            {"product_code": "3", "old_price": 100, "new_price": 200},
            record(4, "0123", 100, 200), record(5, 5, 100, 200), record(6, "1.5", 100, 200),
            record(7, "9223372036854775808", 100, 200), record(8, "", 100, 200),
            record(9, "9", 100, 200)])
        self.assertEqual([(r.record_id, r.reason) for r in prepared.rejected], [
            (1, "same_price"), (2, "missing_field"), (None, "missing_field"), (4, "invalid_product_code"),
            (5, "invalid_product_code"), (6, "invalid_product_code"), (7, "invalid_product_code"),
            (8, "invalid_product_code")])
        self.assertIsNone(prepared.rejected[4].product_code)
        self.assertEqual([item.record_id for item in prepared.items], [9])

    def test_duplicate_codes_reject_every_copy(self):
        prepared = prepare_batch([record(1, "1", 100, 200), record(2, "1", 100, 300), record(3, "2", 100, 200)])
        self.assertEqual([(r.record_id, r.reason) for r in prepared.rejected],
                         [(1, "duplicate_product_code"), (2, "duplicate_product_code")])
        self.assertEqual(prepared.body(), {"changed_count": 1, "products": [
            {"product_code": "2", "old_price": 100, "new_price": 200}]})

    def test_batch_size_and_nothing_sendable(self):
        self.assertEqual(prepare_batch([]).error, "empty_batch")
        full = [record(i, str(i), 100, 200) for i in range(1, 101)]
        self.assertEqual(prepare_batch(full).body()["changed_count"], 100)
        too_many = prepare_batch(full + [record(101, "101", 100, 200)])
        self.assertEqual((too_many.error, too_many.items, too_many.rejected), ("batch_too_large", (), ()))
        nothing = prepare_batch([record(1, "1", 0, 200)])
        self.assertEqual(nothing.error, "no_valid_products")
        self.assertEqual(len(nothing.rejected), 1)
        for prepared in (too_many, nothing, prepare_batch([])):
            with self.subTest(error=prepared.error), self.assertRaises(ValueError):
                prepared.body()

    def test_input_is_not_mutated(self):
        records = [record(1, "1", 100, 200), record(2, "1", 100, 300), record(3, "3", 0, 200)]
        before = copy.deepcopy(records)
        prepare_batch(records)
        self.assertEqual(records, before)


class CompletedResponseTests(unittest.TestCase):
    def parse(self, payload, prepared=None, status=200):
        prepared = prepared or batch()
        frozen = copy.deepcopy(payload)
        result = parse_sync_response(HttpResult(status, payload), prepared)
        self.assertEqual(payload, frozen)   # response is only read
        requested = [item.product_code for item in prepared.items]
        groups = [code for group in codes(result) for code in group]
        self.assertEqual(sorted(groups), sorted(requested))   # no overlap, nothing missing
        return result

    def test_full_success(self):
        result = self.parse(body())
        self.assertEqual((result.kind, result.http_status, result.error_code), (COMPLETED, 200, None))
        self.assertEqual(result.succeeded, ("1234567", "2345678", "3456789"))

    def test_partial_success_mixes_success_conflict_and_not_found(self):
        result = self.parse(body(not_found=["3456789"], conflicts=[("2345678", 65000)]))
        self.assertEqual(result.kind, COMPLETED)
        self.assertEqual(codes(result), (("1234567",), ("3456789",), ("2345678",), ()))
        self.assertEqual(result.conflicts, (PriceConflict("2345678", 65000),))

    def test_all_conflicts_all_not_found_and_failures_only(self):
        all_conflicts = self.parse(body(conflicts=[("3456789", 1), ("1234567", 2), ("2345678", 3)]))
        self.assertEqual([c.product_code for c in all_conflicts.conflicts], ["1234567", "2345678", "3456789"])
        self.assertEqual(all_conflicts.succeeded, ())
        all_missing = self.parse(body(not_found=["1234567", "2345678", "3456789"]))
        self.assertEqual((all_missing.kind, all_missing.succeeded), (COMPLETED, ()))
        mixed = self.parse(body(not_found=["1234567"], conflicts=[("2345678", 1), ("3456789", 2)]))
        self.assertEqual(codes(mixed), ((), ("1234567",), ("2345678", "3456789"), ()))

    def test_already_new_price_reported_as_success(self):
        prepared = batch(record(1, "1", 100, 200))
        self.assertEqual(self.parse(body(sent=1), prepared).succeeded, ("1",))

    def test_extra_legacy_field_is_ignored_not_used(self):
        result = self.parse(body(failed_product_codes=["1234567"]))
        self.assertEqual(result.succeeded, ("1234567", "2345678", "3456789"))


class InvalidResponseTests(unittest.TestCase):
    def assertInvalid(self, payload, error_code, prepared=None):
        prepared = prepared or batch()
        result = parse_sync_response(HttpResult(200, payload), prepared)
        self.assertEqual((result.kind, result.error_code), (INVALID_RESPONSE, error_code))
        self.assertEqual(result.unresolved, tuple(item.product_code for item in prepared.items))
        self.assertEqual((result.succeeded, result.not_found, result.conflicts), ((), (), ()))

    def test_top_level_and_required_arrays(self):
        for payload in (None, [], "ok", {"code": SUCCESS, "data": {}}, {"code": SUCCESS, "message": 1, "data": {}},
                        {"code": SUCCESS, "message": "m", "data": []}):
            with self.subTest(payload=payload):
                self.assertInvalid(payload, "INVALID_RESPONSE_FIELDS")
        for missing in ("not_found_product_codes", "price_conflicts"):
            payload = body()
            del payload["data"][missing]
            with self.subTest(missing=missing):
                self.assertInvalid(payload, "INVALID_RESPONSE_FIELDS")

    def test_legacy_response_cannot_replace_new_contract(self):
        legacy = {"code": PARTIAL, "message": "m", "data": {
            "received_count": 3, "failed_product_codes": ["2345678"], "not_found_product_codes": []}}
        self.assertInvalid(legacy, "INVALID_RESPONSE_FIELDS")

    def test_received_count(self):
        for received in (2, 4, True, "3", None, 3.0):
            with self.subTest(received=received):
                self.assertInvalid(body(received=received), "INVALID_RESPONSE_COUNT")

    def test_product_lists(self):
        cases = [
            body(not_found=["1234567", "1234567"]),
            body(conflicts=[("2345678", 1), ("2345678", 2)]),
            body(not_found=["2345678"], conflicts=[("2345678", 1)]),
            body(not_found=["9999999"]),
            body(conflicts=[("9999999", 1)]),
            body(not_found=[1234567]),
            body(not_found=["01234567"]),
            body(not_found=[{"code": "x"}]),
            body(not_found="1234567"),
            body(price_conflicts={"product_code": "2345678", "current_price": 1}),
        ]
        for payload in cases:
            with self.subTest(data=payload["data"]):
                self.assertInvalid(payload, "INVALID_RESPONSE_PRODUCTS")

    def test_conflict_entries_and_prices(self):
        for entry in ({"product_code": "2345678"}, {"current_price": 1}, "2345678", None):
            with self.subTest(entry=entry):
                self.assertInvalid(body(price_conflicts=[entry]), "INVALID_RESPONSE_FIELDS")
        for price in (0, -1, True, 1.5, "65000", None, 2 ** 31, 69000, 75000):
            with self.subTest(price=price):
                self.assertInvalid(body(conflicts=[("2345678", price)]), "INVALID_RESPONSE_PRICE")

    def test_code_must_match_lists(self):
        for payload in (body(SUCCESS, not_found=["1234567"]), body(SUCCESS, conflicts=[("2345678", 1)]),
                        body(PARTIAL), body("PRODUCT_PRICE_UPDATE_FAILED"), body("OK")):
            with self.subTest(code=payload["code"]):
                self.assertInvalid(payload, "INVALID_RESPONSE_CODE")


class ErrorClassificationTests(unittest.TestCase):
    def assertUnresolved(self, result, kind, status, error_code, prepared):
        self.assertEqual((result.kind, result.http_status, result.error_code), (kind, status, error_code))
        self.assertEqual(result.unresolved, tuple(item.product_code for item in prepared.items))
        self.assertEqual((result.succeeded, result.not_found, result.conflicts), ((), (), ()))

    def test_auth_with_json_body(self):
        prepared = batch()
        for status in (401, 403):
            result = parse_sync_response(HttpResult(status, {"code": "UNAUTHORIZED", "message": "m", "data": None}),
                                         prepared)
            self.assertUnresolved(result, AUTH_REJECTED, status, "AUTH_FAILED", prepared)

    def test_request_errors(self):
        prepared = batch()
        for code, expected in (("INVALID_REQUEST", "INVALID_REQUEST"),
                               ("INVALID_JSON_FORMAT", "INVALID_JSON_FORMAT"), ("OTHER", "BAD_REQUEST")):
            result = parse_sync_response(HttpResult(400, {"code": code, "message": "m", "data": None}), prepared)
            self.assertUnresolved(result, REQUEST_REJECTED, 400, expected, prepared)
        self.assertUnresolved(parse_sync_response(HttpResult(400, ["x"]), prepared),
                              REQUEST_REJECTED, 400, "BAD_REQUEST", prepared)

    def test_server_errors_never_yield_successes(self):
        prepared = batch()
        for payload in ({"code": "INTERNAL_SERVER_ERROR", "message": "m", "data": None}, body(),
                        body("PRODUCT_PRICE_UPDATE_FAILED", not_found=["1234567"]), None):
            for status in (500, 502, 503):
                with self.subTest(status=status, payload=payload):
                    self.assertUnresolved(parse_sync_response(HttpResult(status, payload), prepared),
                                          TEMPORARY_FAILURE, status, "SERVER_ERROR", prepared)

    def test_unexpected_status_is_never_success(self):
        prepared = batch()
        for status in (201, 204, 302, 404, 409, 429):
            with self.subTest(status=status):
                self.assertUnresolved(parse_sync_response(HttpResult(status, body()), prepared),
                                      UNEXPECTED_STATUS, status, "UNEXPECTED_STATUS", prepared)

    def test_unsendable_batch_is_a_caller_bug(self):
        with self.assertRaises(ValueError):
            parse_sync_response(HttpResult(200, body()), prepare_batch([]))
        with self.assertRaises(ValueError):
            classify_send_error(BackendError("backend_timeout"), prepare_batch([]))


class RawFakeResponse:
    def __init__(self, status, raw):
        self.status, self.raw = status, raw

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass

    def read(self):
        return self.raw


class RealClientErrorTests(unittest.TestCase):
    """The real BackendClient parses bytes; only the socket-level connection is fake."""

    def send(self, prepared, *, status=200, raw=b"", error=None):
        connection = Mock()
        if error:
            connection.getresponse.side_effect = error
        else:
            connection.getresponse.return_value = RawFakeResponse(status, raw)
        client = BackendClient(BackendSettings("https://backend.invalid", {"X-Test-Only": "test-value"}),
                               connection_factory=Mock(return_value=connection))
        try:
            return parse_sync_response(client.send(prepared.products()), prepared)
        except BackendError as caught:
            self.assertEqual(caught.args, (caught.code,))   # no remote body in the exception
            return classify_send_error(caught, prepared)

    def test_auth_with_html_empty_or_bad_json_body(self):
        prepared = batch()
        for status in (401, 403):
            for raw in (b"<html>private-secret</html>", b"", b"{bad", b'{"code":"A","code":"B"}'):
                with self.subTest(status=status, raw=raw):
                    result = self.send(prepared, status=status, raw=raw)
                    self.assertEqual((result.kind, result.http_status, result.error_code),
                                     (AUTH_REJECTED, status, "AUTH_FAILED"))
                    self.assertNotIn("private-secret", repr(result))

    def test_bad_json_on_200_and_5xx_and_400(self):
        prepared = batch()
        for status, raw, kind, code in (
                (200, b"{bad", INVALID_RESPONSE, "INVALID_RESPONSE_JSON"),
                (200, b'{"code":"PRODUCT_PRICE_UPDATE_SUCCESS","code":"PRODUCT_PRICE_UPDATE_SUCCESS"}',
                 INVALID_RESPONSE, "INVALID_RESPONSE_JSON"),
                (500, b"<html>", TEMPORARY_FAILURE, "SERVER_ERROR"),
                (400, b"", REQUEST_REJECTED, "BAD_REQUEST"),
                (404, b"<html>", UNEXPECTED_STATUS, "UNEXPECTED_STATUS")):
            with self.subTest(status=status, raw=raw):
                result = self.send(prepared, status=status, raw=raw)
                self.assertEqual((result.kind, result.http_status, result.error_code), (kind, status, code))
                self.assertEqual(result.succeeded, ())

    def test_valid_body_through_real_client(self):
        prepared = batch()
        raw = json.dumps(body(not_found=["3456789"], conflicts=[("2345678", 65000)])).encode()
        result = self.send(prepared, raw=raw)
        self.assertEqual(codes(result), (("1234567",), ("3456789",), ("2345678",), ()))

    def test_timeout_and_connection_errors(self):
        prepared = batch()
        for error, code in ((socket.timeout("private"), "BACKEND_TIMEOUT"), (TimeoutError(), "BACKEND_TIMEOUT"),
                            (ConnectionResetError("private"), "BACKEND_CONNECTION_ERROR")):
            with self.subTest(code=code):
                result = self.send(prepared, error=error)
                self.assertEqual((result.kind, result.http_status, result.error_code),
                                 (TEMPORARY_FAILURE, None, code))
                self.assertEqual(result.unresolved, ("1234567", "2345678", "3456789"))

    def test_other_errors(self):
        prepared = batch()
        self.assertEqual(classify_send_error(BackendError("invalid_backend_request_batch", phase="prepare"),
                                             prepared).kind, NOT_SENT)
        unknown = classify_send_error(RuntimeError("private"), prepared)
        self.assertEqual((unknown.kind, unknown.error_code), (TEMPORARY_FAILURE, "BACKEND_TRANSPORT_FAILED"))
        auth = classify_send_error(BackendError("backend_connection_error", phase="json", http_status=403), prepared)
        self.assertEqual(auth.kind, AUTH_REJECTED)


class ErrorCodeCompatibilityTests(unittest.TestCase):
    def test_error_codes_fit_outbox_error_column_rules(self):
        from price_refresh.outbox import _error
        for code in ("AUTH_FAILED", "INVALID_REQUEST", "INVALID_JSON_FORMAT", "BAD_REQUEST", "SERVER_ERROR",
                     "UNEXPECTED_STATUS", "BACKEND_TIMEOUT", "BACKEND_CONNECTION_ERROR",
                     "BACKEND_TRANSPORT_FAILED", "REQUEST_NOT_SENT", "INVALID_RESPONSE_JSON",
                     "INVALID_RESPONSE_FIELDS", "INVALID_RESPONSE_CODE", "INVALID_RESPONSE_COUNT",
                     "INVALID_RESPONSE_PRODUCTS", "INVALID_RESPONSE_PRICE"):
            with self.subTest(code=code):
                _error(code, None, None)
        self.assertEqual(price_sync.MAX_PRODUCTS, 100)


if __name__ == "__main__":
    unittest.main()
