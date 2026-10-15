import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import Mock, patch

from price_refresh.main import main
from price_refresh.parser import (
    ListingResponseError, display_price, listing_items, product_code,
)
from price_refresh.scanner import collect_prices
from price_refresh.settings import Settings
from price_refresh.source.categories import CATEGORIES
from price_refresh.source.client import (
    DEFAULT_HEADERS, LISTING_ENDPOINT, MOST_REVIEWED, RECOMMENDED,
    ListingRequestError, TwentyNineCmClient,
)

FIXTURE = Path(__file__).parent / "fixtures" / "listing_response.json"
R, V = RECOMMENDED, MOST_REVIEWED


def item(code, price=100):
    return {"itemId": code, "itemInfo": {"displayPrice": price}}


def listing(*items):
    return {"data": {"list": list(items)}}


class FakeClient:
    def __init__(self, responses=None, handler=None):
        self.responses = responses or {}
        self.handler = handler
        self.calls = []

    def fetch_listing(self, **request):
        key = (request["smallId"], request["sort"], request["page"])
        self.calls.append(request)
        value = self.handler(request) if self.handler else self.responses.get(key, listing())
        if isinstance(value, Exception):
            raise value
        return value


class ScannerTests(unittest.TestCase):
    def scan(self, client, categories=None, settings=None):
        self.waits = []
        return collect_prices(client, categories or CATEGORIES[:1], settings or Settings(),
                              sleep=self.waits.append)

    def client(self, responses):
        return FakeClient({(CATEGORIES[0]["smallId"], sort, page): response
                           for (sort, page), response in responses.items()})

    def order(self, client):
        return [(call["sort"], call["page"]) for call in client.calls]

    def test_alternating_pages_start_at_one_and_short_pages_continue(self):
        client = self.client({(sort, page): listing(item(page))
                              for sort in (R, V) for page in (1, 2)})
        result = self.scan(client)
        self.assertEqual(self.order(client), [(R, 1), (V, 1), (R, 2), (V, 2), (R, 3), (V, 3)])
        self.assertTrue(all(call["size"] == 50 for call in client.calls))
        self.assertEqual(self.waits, [1.0] * 6)
        self.assertEqual(result.categories[0].stop_reason, "lists_exhausted")

    def test_one_sort_exhaustion_keeps_other_sort_running(self):
        client = self.client({(V, page): listing(item(page)) for page in (1, 2, 3)})
        result = self.scan(client)
        self.assertEqual(self.order(client), [(R, 1), (V, 1), (V, 2), (V, 3), (V, 4)])
        self.assertEqual(result.prices, {"1": 100, "2": 100, "3": 100})
        self.assertEqual(result.categories[0].sort_status, {R: "exhausted", V: "exhausted"})

    def test_unique_500_counts_duplicates_once_and_reuses_duplicate_price(self):
        count = 0

        def response(_):
            nonlocal count
            count += 1
            start = 2 + (count - 1) * 49
            return listing(item(1, count), *(item(i) for i in range(start, start + 49)))

        client = FakeClient(handler=response)
        result = self.scan(client)
        self.assertEqual(len(client.calls), 11)
        self.assertEqual(result.categories[0].seen_unique_count, 500)
        self.assertEqual(result.categories[0].stop_reason, "unique_limit")
        self.assertEqual(len(result.prices), 500)
        self.assertEqual(result.prices["1"], 11)
        self.assertNotIn("501", result.prices)

    def test_combined_100_request_cap_on_repeated_code(self):
        client = FakeClient()
        client.handler = lambda _: listing(item(1, len(client.calls)))
        result = self.scan(client)
        self.assertEqual(len(client.calls), 100)
        self.assertEqual(self.order(client)[-2:], [(R, 50), (V, 50)])
        self.assertEqual(result.categories[0].page_requests, 100)
        self.assertEqual(result.categories[0].seen_unique_count, 1)
        self.assertEqual(result.categories[0].stop_reason, "page_limit")
        self.assertEqual(result.prices, {"1": 100})

    def test_request_cap_is_not_reset_when_one_sort_ends(self):
        client = FakeClient(handler=lambda req: listing() if req["sort"] == R else listing(item(1)))
        result = self.scan(client)
        self.assertEqual(len(client.calls), 100)
        self.assertEqual(self.order(client)[-1], (V, 99))
        self.assertEqual(result.categories[0].stop_reason, "page_limit")

    def test_stop_at_500th_unique_inside_page_in_response_order(self):
        payload = listing(*(item(i) for i in range(1, 500)), item(1, 200),
                          item(500, 0), item(1, 999), item(501, 999))
        client = self.client({(R, 1): payload})
        result = self.scan(client)
        self.assertEqual(self.order(client), [(R, 1)])
        self.assertEqual(result.prices["1"], 200)
        self.assertEqual(result.prices["500"], 0)
        self.assertNotIn("501", result.prices)
        self.assertEqual(len(result.prices), 500)

    def test_invalid_prices_still_count_toward_unique_limit(self):
        client = self.client({(R, 1): listing(*(item(i, None) for i in range(1, 501)), item(501))})
        with self.assertLogs("price_refresh.scanner", level="WARNING"):
            result = self.scan(client)
        self.assertEqual(len(client.calls), 1)
        self.assertEqual(result.categories[0].seen_unique_count, 500)
        self.assertEqual(result.prices, {})

    def test_global_maxima_across_sorts_and_categories_local_counting(self):
        first, second = CATEGORIES[:2]
        client = FakeClient({
            (first["smallId"], R, 1): listing(item(1, 100), item(2, 200)),
            (first["smallId"], V, 1): listing(item("001", 300)),
            (second["smallId"], R, 1): listing(item(1, 250), item(2, 400)),
            (second["smallId"], V, 1): listing(item("1", 500), item(3, 0)),
        })
        result = self.scan(client, CATEGORIES[:2])
        self.assertEqual(result.prices, {"1": 500, "2": 400, "3": 0})
        self.assertEqual([report.seen_unique_count for report in result.categories], [2, 3])
        self.assertEqual([report.small_id for report in result.categories],
                         [category["smallId"] for category in CATEGORIES[:2]])

    def test_each_category_has_its_own_request_cap(self):
        client = FakeClient(handler=lambda _: listing(item(1)))
        result = self.scan(client, CATEGORIES[:2], Settings(max_page_requests=3))
        self.assertEqual([report.page_requests for report in result.categories], [3, 3])
        self.assertEqual(self.order(client), [(R, 1), (V, 1), (R, 2)] * 2)

    def test_bad_codes_do_not_consume_unique_limit_and_zero_price_is_allowed(self):
        client = self.client({(R, 1): listing(item(True), item(-1), item("abc"),
                                                   item(1, 0), item("001", None), item(2, -1))})
        with self.assertLogs("price_refresh.scanner", level="WARNING"):
            result = self.scan(client)
        self.assertEqual(result.prices, {"1": 0})
        self.assertEqual(result.categories[0].seen_unique_count, 2)

    def test_malformed_response_is_error_other_sort_and_categories_continue(self):
        first, second = CATEGORIES[:2]
        client = FakeClient({
            (first["smallId"], R, 1): {"data": {}},
            (first["smallId"], V, 1): listing(item(1, 123)),
            (second["smallId"], R, 1): listing(item(2, 234)),
        })
        with self.assertLogs("price_refresh.scanner", level="INFO") as logs:
            result = self.scan(client, CATEGORIES[:2])
        self.assertEqual(result.prices, {"1": 123, "2": 234})
        self.assertEqual(result.categories[0].stop_reason, "lookup_error")
        self.assertEqual(result.categories[0].sort_status, {R: "error", V: "exhausted"})
        self.assertEqual(result.categories[1].stop_reason, "lists_exhausted")
        self.assertTrue(any("lookup_error" in line and "page=1" in line for line in logs.output))
        self.assertTrue(any("category_end" in line for line in logs.output))

    def test_transport_failures_are_not_zero_or_empty_and_not_retried(self):
        for error in (ListingRequestError("HTTP 429"), TimeoutError("read timeout")):
            with self.subTest(error=error):
                client = FakeClient(handler=lambda _: error)
                with self.assertLogs("price_refresh.scanner", level="ERROR"):
                    result = self.scan(client)
                self.assertEqual(self.order(client), [(R, 1), (V, 1)])
                self.assertEqual(self.waits, [1, 1])
                self.assertEqual(result.prices, {})
                self.assertEqual(len(result.categories[0].errors), 2)
                self.assertEqual(result.categories[0].stop_reason, "lookup_error")

    def test_response_wait_occurs_before_next_request_including_empty_page(self):
        events = []
        client = FakeClient(handler=lambda req: events.append(("request", req["sort"])) or listing())
        collect_prices(client, CATEGORIES[:1], Settings(request_delay=1.25),
                       sleep=lambda seconds: events.append(("sleep", seconds)))
        self.assertEqual(events, [("request", R), ("sleep", 1.25),
                                  ("request", V), ("sleep", 1.25)])


class ParserTests(unittest.TestCase):
    def test_original_fixture_primary_price_field(self):
        payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
        first = listing_items(payload)[0]
        self.assertEqual(product_code(first), "3263645")
        self.assertEqual(display_price(first), 31430)  # not itemEvent's 99999

    def test_codes_normalize_to_positive_ascii_digit_strings(self):
        for value, expected in ((3263645, "3263645"), ("003263645", "3263645")):
            self.assertEqual(product_code(item(value)), expected)
        for value in (None, True, False, -1, 0, 1.5, 1.0, "", "0", "000", "-1",
                      "+1", " 1", "1 ", "1.0", "١", "１２", [], {}):
            with self.subTest(value=value):
                self.assertIsNone(product_code(item(value)))
        for value in ({}, None, [], 1):
            self.assertIsNone(product_code(value))

    def test_only_nonnegative_integer_prices_and_no_extra_fields_required(self):
        self.assertEqual(display_price(item(1, 0)), 0)
        self.assertEqual(display_price(item(1, 79000)), 79000)
        for value in (None, True, False, -1, 1.5, 100.0, "100", "0", [], {},
                      float("nan"), float("inf")):
            with self.subTest(value=value):
                self.assertIsNone(display_price(item(1, value)))
        for value in (None, [], {}, {"itemId": 1}, {"itemInfo": None},
                      {"itemInfo": []}, {"itemInfo": {}}):
            self.assertIsNone(display_price(value))

    def test_empty_list_is_valid_but_missing_or_wrong_shape_is_error(self):
        self.assertEqual(listing_items(listing()), [])
        for payload in (None, [], {}, {"data": None}, {"data": []}, {"data": {}},
                        {"data": {"list": None}}, {"data": {"list": {}}},
                        {"error": "failure"}):
            with self.subTest(payload=payload):
                with self.assertRaises(ListingResponseError):
                    listing_items(payload)


class ClientTests(unittest.TestCase):
    def setup_connection(self, raw=None, status=200):
        self.events = []
        response = Mock(status=status)
        response.read.side_effect = lambda: self.events.append("read") or (
            raw if raw is not None else FIXTURE.read_bytes()
        )
        response.__enter__ = Mock(return_value=response)
        response.__exit__ = Mock(return_value=False)
        connection = Mock()
        connection.connect.side_effect = lambda: self.events.append("connect")
        connection.sock.settimeout.side_effect = lambda timeout: self.events.append(("timeout", timeout))
        connection.request.side_effect = lambda *a, **kw: self.events.append("request")
        connection.getresponse.side_effect = lambda: self.events.append("getresponse") or response
        factory = Mock(return_value=connection)
        return connection, factory

    def fetch(self, client):
        return client.fetch_listing(**{key: CATEGORIES[0][key] for key in ("largeId", "middleId", "smallId")},
                                    sort=R)

    def test_request_contract_and_distinct_timeout_application(self):
        connection, factory = self.setup_connection()
        result = self.fetch(TwentyNineCmClient(Settings(), connection_factory=factory))
        factory.assert_called_once_with("display-bff-api.29cm.co.kr", timeout=5.0)
        self.assertEqual(self.events, ["connect", ("timeout", 20.0), "request", "getresponse", "read"])
        args, kwargs = connection.request.call_args
        self.assertEqual(args, ("POST", "/api/v1/listing/items?colorchipVariant=treatment"))
        self.assertEqual(kwargs["headers"], DEFAULT_HEADERS)
        self.assertEqual(json.loads(kwargs["body"]), {
            "pageType": "CATEGORY_PLP", "sortType": R,
            "facets": {"categoryFacetInputs": [{"largeId": 272100100,
                        "middleId": 272103100, "smallId": 272103105}]},
            "pageRequest": {"page": 1, "size": 50},
        })
        self.assertEqual(result, json.loads(FIXTURE.read_bytes()))
        connection.close.assert_called_once()

    def test_custom_timeouts_and_page_settings(self):
        connection, factory = self.setup_connection()
        client = TwentyNineCmClient(Settings(connect_timeout=2, read_timeout=7), connection_factory=factory)
        client.fetch_listing(largeId="1", middleId="2", smallId="3", sort=V, page=4, size=20)
        factory.assert_called_once_with("display-bff-api.29cm.co.kr", timeout=2)
        connection.sock.settimeout.assert_called_once_with(7)
        body = json.loads(connection.request.call_args.kwargs["body"])
        self.assertEqual(body["sortType"], V)
        self.assertEqual(body["pageRequest"], {"page": 4, "size": 20})

    def test_http_and_invalid_json_errors_close_connection_without_retry(self):
        for raw, status in ((b"{}", 429), (b"{}", 500), (b"", 204),
                            (b"<html>blocked</html>", 200), (b"\xff", 200), (b"[]", 200)):
            with self.subTest(raw=raw, status=status):
                connection, factory = self.setup_connection(raw, status)
                with self.assertRaises(ListingRequestError):
                    self.fetch(TwentyNineCmClient(Settings(), connection_factory=factory))
                connection.request.assert_called_once()
                connection.close.assert_called_once()

    def test_connect_and_read_timeout_failures_close_without_retry(self):
        for operation in ("connect", "getresponse"):
            with self.subTest(operation=operation):
                connection, factory = self.setup_connection()
                getattr(connection, operation).side_effect = TimeoutError("timed out")
                with self.assertRaises(ListingRequestError):
                    self.fetch(TwentyNineCmClient(Settings(), connection_factory=factory))
                factory.assert_called_once()
                connection.close.assert_called_once()


class SettingsAndCliTests(unittest.TestCase):
    def test_default_and_environment_settings(self):
        self.assertEqual(Settings.from_env({}), Settings())
        settings = Settings.from_env({"PRICE_REFRESH_MAX_PAGE_REQUESTS": "2",
                                      "PRICE_REFRESH_CONNECT_TIMEOUT": "3.5"})
        self.assertEqual(settings.max_page_requests, 2)
        self.assertEqual(settings.connect_timeout, 3.5)

    def test_invalid_settings_fail_before_request(self):
        for key, value in (("CONCURRENCY", "2"), ("PAGE_SIZE", "51"), ("MAX_UNIQUE", "501"),
                           ("MAX_PAGE_REQUESTS", "101"), ("MAX_PAGE_REQUESTS", "0"),
                           ("READ_TIMEOUT", "0"), ("REQUEST_DELAY", "-1"),
                           ("REQUEST_DELAY", "nan"), ("CONNECT_TIMEOUT", "inf"),
                           ("MAX_UNIQUE", "2.5"), ("READ_TIMEOUT", "oops")):
            with self.subTest(key=key, value=value):
                with self.assertRaises(ValueError):
                    Settings.from_env({f"PRICE_REFRESH_{key}": value})
        for values in ({"max_unique": True}, {"max_page_requests": 1.5}, {"request_delay": True}):
            with self.assertRaises(ValueError):
                Settings(**values)

    def test_cli_fixture_is_offline_fast_and_outputs_raw_price_mapping(self):
        output = io.StringIO()
        with patch.dict(os.environ, {}, clear=True), patch("time.sleep") as sleep, \
                patch("http.client.HTTPSConnection.connect") as connect, redirect_stdout(output):
            code = main(["--fixture", str(FIXTURE), "--category-limit", "1"])
        self.assertEqual(code, 0)
        sleep.assert_not_called()
        connect.assert_not_called()
        payload = json.loads(output.getvalue())
        self.assertEqual(payload["prices"]["3263645"], 31430)
        self.assertEqual(len(payload["prices"]), 5)
        self.assertEqual(payload["categories"][0]["page_requests"], 4)
        self.assertEqual(set(payload), {"prices", "categories"})

    def test_cli_returns_failure_with_partial_results(self):
        client = FakeClient(handler=lambda _: {"data": {}})
        output = io.StringIO()
        with patch.dict(os.environ, {}, clear=True), \
                patch("price_refresh.main.FixtureClient", return_value=client), \
                redirect_stdout(output), self.assertLogs(level="ERROR"):
            code = main(["--fixture", str(FIXTURE), "--category-limit", "1"])
        self.assertEqual(code, 1)
        self.assertEqual(len(json.loads(output.getvalue())["categories"][0]["errors"]), 2)

    def test_cli_does_not_overwrite_existing_output(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            path.write_text("keep", encoding="utf-8")
            with patch.dict(os.environ, {}, clear=True), self.assertLogs(level="ERROR"):
                code = main(["--fixture", str(FIXTURE), "--category-limit", "1", "--output", str(path)])
            self.assertEqual(code, 1)
            self.assertEqual(path.read_text(), "keep")

    def test_cli_runs_from_new_folder_without_parent_imports(self):
        completed = subprocess.run(
            [sys.executable, "-I", "-c",
             "import sys, runpy; sys.path.insert(0, '.'); runpy.run_module('price_refresh', run_name='__main__')",
             "--fixture", str(FIXTURE.resolve()), "--category-limit", "1"],
            cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True,
            env={key: value for key, value in os.environ.items() if not key.startswith("PRICE_REFRESH_")},
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(len(json.loads(completed.stdout)["prices"]), 5)


if __name__ == "__main__":
    unittest.main()
