"""At most three real listing calls, using unchanged legacy/new clients.

The legacy client is loaded from a single read-only mounted file. Observation
wrappers preserve default headers, TLS verification and native SSL contexts.
Redirects are disabled only in the legacy diagnostic opener to bound requests.
"""

import argparse
import hashlib
import http.client
import importlib.util
import json
import time
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import Mock

from price_refresh.parser import display_price, listing_items, product_code
from price_refresh.settings import Settings
from price_refresh.source.categories import CATEGORIES
from price_refresh.source.client import (
    DEFAULT_HEADERS, LISTING_ENDPOINT, MOST_REVIEWED, RECOMMENDED, TwentyNineCmClient,
)
from .runtime import runtime_info


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


def load_legacy(path):
    spec = importlib.util.spec_from_file_location("_legacy_price_client", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def assert_request_contract(legacy, codes):
    """Compare both actual request builders using fake transports, zero HTTP."""
    requests = []
    response = Mock()
    response.status = 200
    response.headers = {"Content-Type": "application/json"}
    response.read.return_value = b'{"data":{"list":[]}}'
    response.__enter__ = Mock(return_value=response)
    response.__exit__ = Mock(return_value=False)

    def opener(request, timeout):
        requests.append((request.full_url, request.method, request.data,
                         {key.lower(): value for key, value in request.headers.items()}))
        return response

    legacy.TwentyNineCmClient(opener=opener).fetch_listing(
        **codes, sort=MOST_REVIEWED, page=1, size=50,
    )
    connection = Mock()
    connection.getresponse.return_value = response

    def request(method, path, body, headers):
        requests.append(("https://display-bff-api.29cm.co.kr" + path, method, body,
                         {key.lower(): value for key, value in headers.items()}))

    connection.request.side_effect = request
    TwentyNineCmClient(Settings(), connection_factory=lambda *a, **kw: connection).fetch_listing(
        **codes, sort=MOST_REVIEWED, page=1, size=50,
    )
    if (legacy.LISTING_ENDPOINT != LISTING_ENDPOINT or legacy.DEFAULT_HEADERS != DEFAULT_HEADERS
            or len(requests) != 2 or requests[0] != requests[1]):
        raise ValueError("Legacy/new request contracts differ; no live calls allowed")
    return {"url": requests[0][0], "body_sha256": hashlib.sha256(requests[0][2]).hexdigest(),
            "identical_url_method_headers_body": True}


def perform_probe(kind, sort, legacy, codes):
    record = {"client": kind, "sort": sort, "page": 1, "size": 50,
              "started_at_seoul": datetime.now(timezone(timedelta(hours=9))).isoformat(),
              "http_status": None, "content_type": None, "json_response": None,
              "data_list_present": None, "product_count": None, "samples": [],
              "error": None, "ok": False, "stage": "setup", "http_attempts": 0}
    captured = {}

    class ObservedConnection(http.client.HTTPSConnection):
        def connect(self):
            record["stage"] = "DNS/TCP/TLS connection"
            record["observed_connect_timeout_seconds"] = self.timeout
            super().connect()
            record["negotiated_tls"] = self.sock.version()
            record["negotiated_alpn"] = self.sock.selected_alpn_protocol()

        def request(self, *args, **kwargs):
            record["http_attempts"] += 1
            if record["http_attempts"] > 1:
                raise RuntimeError("Additional request prohibited")
            record["stage"] = "send request"
            super().request(*args, **kwargs)

        def getresponse(self):
            record["stage"] = "receive response headers"
            record["observed_socket_timeout_seconds"] = self.sock.gettimeout()
            response = super().getresponse()
            record["http_status"] = response.status
            record["content_type"] = response.getheader("Content-Type")
            record["response_headers"] = {
                name: response.getheader(name) for name in ("Server", "cf-mitigated", "cf-ray")
            }
            original_read = response.read

            def read(*args, **kwargs):
                record["stage"] = "read response body"
                raw = original_read(*args, **kwargs)
                captured["raw"] = raw
                record["stage"] = (
                    "HTTP status handling" if not 200 <= response.status < 300 else "JSON parsing"
                )
                return raw

            response.read = read
            return response

    class ObservedHTTPSHandler(urllib.request.HTTPSHandler):
        def do_open(self, http_class, request, **kwargs):
            return super().do_open(ObservedConnection, request, **kwargs)

    # The legacy logger otherwise prints HTML prefixes. Restore its state after use.
    logger_disabled = legacy.logger.disabled
    started = time.monotonic()
    try:
        if kind == "legacy":
            legacy.logger.disabled = True
            opener = urllib.request.build_opener(NoRedirect(), ObservedHTTPSHandler())
            client = legacy.TwentyNineCmClient(timeout=10, opener=opener.open)
            record["timeout"] = {"single_io_seconds": 10}
        else:
            client = TwentyNineCmClient(Settings(), connection_factory=ObservedConnection)
            record["timeout"] = {"connect_seconds": 5, "socket_io_seconds": 20}
        payload = client.fetch_listing(**codes, sort=sort, page=1, size=50)
        record["stage"] = "listing shape validation"
        items = listing_items(payload)
        record["product_count"] = len(items)
        for item in items[:3]:
            info = item.get("itemInfo") if isinstance(item, dict) else None
            raw_code = item.get("itemId") if isinstance(item, dict) else None
            raw_price = info.get("displayPrice") if isinstance(info, dict) else None
            code, price = product_code(item), display_price(item)
            record["samples"].append({"itemId": raw_code, "displayPrice": raw_price,
                                      "parsed_code": code, "parsed_price": price,
                                      "matches": code is not None and price is not None
                                      and code == str(raw_code) and price == raw_price})
        record["ok"] = all(sample["matches"] for sample in record["samples"])
        record["stage"] = "completed"
    except Exception as error:
        chain = []
        current = error
        while current is not None:
            chain.append(type(current).__name__)
            current = current.__cause__
        record["error"] = {"exception_chain": chain, "failure_stage": record["stage"]}
    finally:
        legacy.logger.disabled = logger_disabled
        record["elapsed_seconds"] = round(time.monotonic() - started, 6)
    raw = captured.get("raw")
    if raw is not None:
        record["cloudflare_marker"] = b"cloudflare" in raw.lower()
        record["captcha_marker"] = b"captcha" in raw.lower()
        try:
            observed = json.loads(raw.decode("utf-8"))
            record["json_response"] = True
            data = observed.get("data") if isinstance(observed, dict) else None
            record["data_list_present"] = isinstance(data, dict) and "list" in data
        except (ValueError, UnicodeError):
            record["json_response"] = False
            record["data_list_present"] = False
    return record


def compare(legacy, codes, *, probe=perform_probe, sleep=time.sleep):
    records = []
    for kind, sort in (("legacy", MOST_REVIEWED), ("new", MOST_REVIEWED), ("new", RECOMMENDED)):
        if len(records) == 2 and not all(record["ok"] for record in records):
            break
        try:
            records.append(probe(kind, sort, legacy, codes))
        finally:
            sleep(1.0)  # also after the final response/failure, never retry
    return records


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-client", type=Path, required=True,
                        help="Unmodified legacy client.py, mounted read-only")
    parser.add_argument("--live", action="store_true", help="Allow at most three API calls")
    args = parser.parse_args(argv)
    legacy = load_legacy(args.legacy_client)
    category = CATEGORIES[0]
    codes = {name: category[name] for name in ("largeId", "middleId", "smallId")}
    contract = assert_request_contract(legacy, codes)
    proxy_settings = urllib.request.getproxies()
    proxy_configured = any(value for key, value in proxy_settings.items() if key != "no")
    # Do not silently change proxy conditions between urllib and direct HTTPS.
    if args.live and proxy_configured:
        parser.error("Proxy configured; live comparison stopped without changing settings")
    records = compare(legacy, codes) if args.live else []
    print(json.dumps({"runtime": runtime_info(), "category": category,
                      "request_contract": contract, "proxy_configured": bool(proxy_configured),
                      "live": args.live, "requests": records,
                      "attempted_requests": len(records)}, ensure_ascii=False, indent=2))
    return 0 if not args.live or all(record["ok"] for record in records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
