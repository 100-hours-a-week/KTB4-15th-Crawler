"""HTTP wire contract, configuration, and local-only HTTP server tests."""

import http.client
import json
import socket
import threading
import unittest
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, HTTPServer
from unittest.mock import Mock, patch

from price_refresh.backend import BackendClient, BackendError, BackendSettings, ENDPOINT
from price_refresh.delivery import BackendDelivery
from test_delivery import changes, fake_session, response


class BackendSettingsTests(unittest.TestCase):
    def test_explicit_auth_injection_and_defaults(self):
        env = {"PRICE_REFRESH_BACKEND_BASE_URL": "https://backend.invalid",
               "PRICE_REFRESH_BACKEND_AUTH_HEADERS_JSON": '{"X-Confirmed-By-Operator":"test-only"}'}
        with patch.dict("os.environ", env, clear=True):
            settings = BackendSettings.from_env()
        self.assertEqual((settings.batch_size, settings.connect_timeout, settings.read_timeout), (100, 5, 20))
        self.assertEqual(settings.auth_headers, {"X-Confirmed-By-Operator": "test-only"})
        self.assertNotIn("test-only", repr(settings))
        self.assertNotIn("backend.invalid", repr(settings))

    def test_invalid_url_batch_timeouts_and_auth_rejected_without_secret_messages(self):
        base = {"base_url": "https://backend.invalid", "auth_headers": {"X-Test": "private-secret"}}
        cases = []
        for url in ("", "file:///tmp/sock", "https://user:private-secret@backend.invalid", "https://host/path",
                    "https://host?secret=private-secret", "https://host/#secret", "https://host:bad",
                    "https://host:0", "https://host\nother", None):
            cases.append({**base, "base_url": url})
        for size in (0, 101, True, 1.5, "100"):
            cases.append({**base, "batch_size": size})
        for value in (0, -1, float("nan"), float("inf"), True, "5"):
            cases.extend([{**base, "connect_timeout": value}, {**base, "read_timeout": value}])
        for headers in ({}, None, [], {"Content-Length": "9"}, {"X-Test": "bad\r\nInjected: true"},
                        {"Bad Header": "value"}, {"X-Test": ""}, {"X-Test": None},
                        {"X-Test": "test", "x-test": "duplicate"}, {"X-Test": "한글"}):
            cases.append({**base, "auth_headers": headers})
        for case in cases:
            with self.subTest(setting_names=sorted(case)):
                with self.assertRaises(BackendError) as error:
                    BackendSettings(**case)
                self.assertNotIn("private-secret", str(error.exception))

    def test_environment_invalid_json_numeric_values_and_lower_backend_cap(self):
        env = {"PRICE_REFRESH_BACKEND_BASE_URL": "https://backend.invalid",
               "PRICE_REFRESH_BACKEND_AUTH_HEADERS_JSON": '{"X-Test":"test"}'}
        for key, value in (("PRICE_REFRESH_BACKEND_AUTH_HEADERS_JSON", "broken-json"),
                           ("PRICE_REFRESH_BACKEND_AUTH_HEADERS_JSON", '{"X-Test":"a","X-Test":"b"}'),
                           ("PRICE_REFRESH_BACKEND_BATCH_SIZE", "1.5"),
                           ("PRICE_REFRESH_BACKEND_READ_TIMEOUT", "nan")):
            with self.subTest(key=key), patch.dict("os.environ", {**env, key: value}, clear=True):
                with self.assertRaises(BackendError):
                    BackendSettings.from_env()
        with patch.dict("os.environ", {**env, "PRICE_REFRESH_BACKEND_BATCH_SIZE": "25"}, clear=True):
            self.assertEqual(BackendSettings.from_env().batch_size, 25)


class FakeResponse:
    status = 200

    def __init__(self, body=None):
        self.body = body or json.dumps(response(["1"]).payload).encode()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass

    def read(self):
        return self.body


class BackendClientTests(unittest.TestCase):
    def connection(self):
        connection = Mock()
        connection.getresponse.return_value = FakeResponse()
        return connection

    def test_patch_body_header_injection_timeouts_order_close_and_one_request(self):
        connection = self.connection()
        factory = Mock(return_value=connection)
        settings = BackendSettings("https://backend.invalid", {"X-Test-Auth": "test-only"})
        result = BackendClient(settings, connection_factory=factory).send([
            {"product_code": "1", "old_price": 100, "new_price": 0}])
        self.assertEqual(result.status, 200)
        factory.assert_called_once_with("backend.invalid", port=None, timeout=5)
        calls = [entry[0] for entry in connection.mock_calls]
        self.assertEqual(calls, ["connect", "sock.settimeout", "request", "getresponse", "close"])
        connection.sock.settimeout.assert_called_once_with(20)
        args, kwargs = connection.request.call_args
        self.assertEqual(args, ("PATCH", ENDPOINT))
        self.assertEqual(json.loads(kwargs["body"]), {"changed_count": 1, "products": [
            {"product_code": "1", "old_price": 100, "new_price": 0}]})
        self.assertEqual(kwargs["headers"]["X-Test-Auth"], "test-only")
        self.assertNotIn("Authorization", kwargs["headers"])

    def test_transport_timeout_and_json_failure_close_without_retry(self):
        for phase, error in (("connect", socket.timeout("private-secret")),
                             ("request", BrokenPipeError("private-secret")),
                             ("getresponse", http.client.RemoteDisconnected("private-secret"))):
            with self.subTest(phase=phase):
                connection = self.connection()
                getattr(connection, phase).side_effect = error
                factory = Mock(return_value=connection)
                with self.assertRaises(BackendError) as caught:
                    BackendClient(BackendSettings("https://backend.invalid", {"X-Test": "test"}),
                                  connection_factory=factory).send([{"product_code": "1"}])
                self.assertNotIn("private-secret", str(caught.exception))
                factory.assert_called_once()
                connection.close.assert_called_once()
                self.assertLessEqual(connection.request.call_count, 1)
        for body in (b"private-secret html", b'{"code":"first","code":"second"}'):
            connection = self.connection()
            connection.getresponse.return_value = FakeResponse(body)
            with self.assertRaises(BackendError) as caught:
                BackendClient(BackendSettings("https://backend.invalid", {"X-Test": "test"}),
                              connection_factory=Mock(return_value=connection)).send([{"product_code": "1"}])
            self.assertEqual(caught.exception.code, "backend_invalid_json")
            self.assertNotIn("private-secret", str(caught.exception))
            connection.request.assert_called_once()

    def test_batch_cap_and_duplicates_rejected_before_connection(self):
        factory = Mock()
        client = BackendClient(BackendSettings("https://backend.invalid", {"X-Test": "test"}, batch_size=1),
                               connection_factory=factory)
        for products in ([], [{"product_code": "1"}] * 2):
            with self.assertRaises(BackendError):
                client.send(products)
        factory.assert_not_called()
        client.settings = BackendSettings("https://backend.invalid", {"X-Test": "test"})
        with self.assertRaises(BackendError):
            client.send([{"product_code": "1"}] * 2)
        factory.assert_not_called()


@contextmanager
def local_backend(*, redirect=False):
    """Loopback only, ephemeral port, records test data; never logs headers/bodies."""
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_PATCH(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append((self.path, body, self.headers.get("X-Test-Only")))
            codes = [product["product_code"] for product in body["products"]]
            payload = response(codes, failed=["2"] if len(requests) == 1 and "2" in codes else []).payload
            raw = json.dumps(payload).encode()
            self.send_response(302 if redirect else 200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            if redirect:
                self.send_header("Location", "/must-not-follow")
            self.end_headers()
            self.wfile.write(raw)

        def log_message(self, *_):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server.server_address[1], requests
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


class LocalHttpTests(unittest.TestCase):
    def test_real_loopback_patch_partial_retry_and_exact_request_counts(self):
        with local_backend() as (port, requests):
            client = BackendClient(BackendSettings(f"http://127.0.0.1:{port}", {"X-Test-Only": "test-value"}))
            report = BackendDelivery(client).deliver(fake_session(["1", "2", "3"]), changes(3))
        self.assertEqual(len(requests), 2)
        self.assertEqual([len(body["products"]) for _, body, _ in requests], [3, 1])
        for path, body, auth in requests:
            self.assertEqual(path, ENDPOINT)
            self.assertEqual(body["changed_count"], len(body["products"]))
            self.assertTrue(all(type(p["product_code"]) is str for p in body["products"]))
            self.assertEqual(auth, "test-value")
        self.assertEqual(requests[1][1]["products"], [requests[0][1]["products"][1]])
        self.assertEqual(report.to_dict()["successful_count"], 3)

    def test_real_loopback_redirect_is_not_followed_or_retried(self):
        with local_backend(redirect=True) as (port, requests):
            client = BackendClient(BackendSettings(f"http://127.0.0.1:{port}", {"X-Test-Only": "test-value"}))
            report = BackendDelivery(client).deliver(fake_session(["1"]), changes(1))
        self.assertEqual(len(requests), 1)
        self.assertEqual(report.to_dict()["unknown_count"], 1)


if __name__ == "__main__":
    unittest.main()
