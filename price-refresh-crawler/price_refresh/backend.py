"""Explicit backend configuration, one PATCH attempt, and strict response parsing."""

import http.client
import json
import math
import os
import re
import socket
from dataclasses import dataclass, field
from urllib.parse import urlsplit

from .comparison import normalize_code

ENDPOINT = "/api/v1/internal/products"
SUCCESS = "PRODUCT_PRICE_UPDATE_SUCCESS"
PARTIAL = "PRODUCT_PRICE_UPDATE_PARTIAL_SUCCESS"
FAILED = "PRODUCT_PRICE_UPDATE_FAILED"


def unique_json_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate_json_field")
        result[key] = value
    return result


class BackendError(RuntimeError):
    """Internal constant codes only; never include remote bodies or credentials."""

    def __init__(self, code, *, phase=None, http_status=None):
        self.code = code
        self.phase = phase
        self.http_status = http_status
        super().__init__(code)


@dataclass(frozen=True)
class BackendSettings:
    base_url: str = field(repr=False)
    auth_headers: dict = field(repr=False)
    batch_size: int = 100
    connect_timeout: float = 5
    read_timeout: float = 20

    def __post_init__(self):
        try:
            url = urlsplit(self.base_url)
            if (not isinstance(self.base_url, str)
                    or any(c.isspace() or ord(c) < 32 or ord(c) == 127 for c in self.base_url)
                    or url.scheme not in {"http", "https"} or not url.hostname
                    or url.username is not None or url.password is not None
                    or url.path not in {"", "/"} or url.query or url.fragment
                    or (url.port is not None and url.port < 1)):
                raise ValueError
            # http.client must be able to encode the host/path without modifying it.
            self.base_url.encode("ascii")
        except (ValueError, TypeError, AttributeError, UnicodeError):
            raise BackendError("invalid_backend_url") from None
        if type(self.batch_size) is not int or not 1 <= self.batch_size <= 100:
            raise BackendError("invalid_backend_batch_size")
        for value in (self.connect_timeout, self.read_timeout):
            if type(value) not in {int, float} or not math.isfinite(value) or value <= 0:
                raise BackendError("invalid_backend_timeout")
        if not isinstance(self.auth_headers, dict) or not self.auth_headers:
            raise BackendError("backend_auth_configuration_missing")
        reserved = {"host", "content-type", "accept", "content-length", "transfer-encoding",
                    "connection", "upgrade", "trailer", "te", "accept-encoding"}
        seen = set()
        for name, value in self.auth_headers.items():
            if (not isinstance(name, str) or not re.fullmatch(r"[!#$%&'*+.^_`|~0-9A-Za-z-]+", name)
                    or name.lower() in reserved or name.lower() in seen
                    or not isinstance(value, str) or not value.strip()
                    or any(ord(c) < 32 or ord(c) == 127 for c in value)):
                raise BackendError("invalid_backend_auth_configuration")
            try:
                value.encode("latin-1")
            except UnicodeError:
                raise BackendError("invalid_backend_auth_configuration") from None
            seen.add(name.lower())
        # Detach from the caller's mutable configuration.
        object.__setattr__(self, "auth_headers", dict(self.auth_headers))

    @classmethod
    def from_env(cls):
        url = os.getenv("PRICE_REFRESH_BACKEND_BASE_URL", "")
        if not url:
            raise BackendError("backend_url_missing")
        raw_headers = os.getenv("PRICE_REFRESH_BACKEND_AUTH_HEADERS_JSON", "")
        if not raw_headers:
            raise BackendError("backend_auth_configuration_missing")
        try:
            headers = json.loads(raw_headers, object_pairs_hook=unique_json_object)
            return cls(url, headers, int(os.getenv("PRICE_REFRESH_BACKEND_BATCH_SIZE", "100")),
                       float(os.getenv("PRICE_REFRESH_BACKEND_CONNECT_TIMEOUT", "5")),
                       float(os.getenv("PRICE_REFRESH_BACKEND_READ_TIMEOUT", "20")))
        except (ValueError, TypeError):
            raise BackendError("invalid_backend_configuration") from None


@dataclass
class HttpResult:
    status: int
    payload: object


class BackendClient:
    def __init__(self, settings, *, connection_factory=None):
        self.settings = settings
        self._connection_factory = connection_factory

    def send(self, products):
        """Exactly one connection/request; no redirect following or HTTP retries."""
        if not products or len(products) > self.settings.batch_size:
            raise BackendError("invalid_backend_request_batch", phase="prepare")
        codes = [product["product_code"] for product in products]
        if len(set(codes)) != len(codes):
            raise BackendError("duplicate_backend_request_codes", phase="prepare")
        raw = json.dumps({"changed_count": len(products), "products": products}).encode("utf-8")
        url = urlsplit(self.settings.base_url)
        factory = self._connection_factory or (http.client.HTTPSConnection if url.scheme == "https"
                                                else http.client.HTTPConnection)
        connection, phase, status = None, "connect", None
        try:
            connection = factory(url.hostname, port=url.port, timeout=self.settings.connect_timeout)
            connection.connect()
            phase = "socket_setup"
            connection.sock.settimeout(self.settings.read_timeout)
            phase = "request"
            connection.request("PATCH", ENDPOINT, body=raw, headers={
                **self.settings.auth_headers, "Content-Type": "application/json", "Accept": "application/json",
            })
            phase = "response"
            with connection.getresponse() as response:
                status = response.status
                body = response.read()
            phase = "json"
            try:
                payload = json.loads(body.decode("utf-8"), object_pairs_hook=unique_json_object)
            except (ValueError, UnicodeError):
                raise BackendError("backend_invalid_json", phase=phase, http_status=status) from None
            return HttpResult(status, payload)
        except BackendError:
            raise
        except (socket.timeout, TimeoutError):
            raise BackendError("backend_timeout", phase=phase, http_status=status) from None
        except (OSError, http.client.HTTPException):
            raise BackendError("backend_connection_error", phase=phase, http_status=status) from None
        finally:
            if connection is not None:
                connection.close()


@dataclass
class BatchOutcome:
    successful: set = field(default_factory=set)
    failed: set = field(default_factory=set)
    not_found: set = field(default_factory=set)
    unknown: set = field(default_factory=set)
    error: object = None


def classify_response(response, requested):
    """Only HTTP 200 can establish success by complement of the two arrays."""
    requested = set(requested)
    payload = response.payload
    if not isinstance(payload, dict) or not isinstance(payload.get("code"), str):
        return BatchOutcome(unknown=requested, error="invalid_backend_response")
    code = payload["code"]
    # These errors are terminal for this attempt, but no per-product persistence
    # contract was supplied. Do not label every product as a confirmed DB failure.
    if response.status == 400 and code == "INVALID_REQUEST":
        return BatchOutcome(unknown=requested, error="backend_invalid_request")
    if response.status in {401, 403}:
        return BatchOutcome(unknown=requested, error="backend_auth_rejected")
    if response.status == 500 and code == "INTERNAL_SERVER_ERROR":
        return BatchOutcome(unknown=requested, error="backend_internal_server_error")
    if not ((response.status == 200 and code in {SUCCESS, PARTIAL})
            or (response.status == 500 and code == FAILED)):
        return BatchOutcome(unknown=requested, error="backend_unclassified_response")
    data = payload.get("data")
    if (not isinstance(payload.get("message"), str) or not isinstance(data, dict)
            or type(data.get("received_count")) is not int
            or data["received_count"] != len(requested)):
        return BatchOutcome(unknown=requested, error="invalid_backend_response")
    arrays = []
    for name in ("failed_product_codes", "not_found_product_codes"):
        values = data.get(name)
        if (not isinstance(values, list)
                or any(not isinstance(code, str) or normalize_code(code) != code
                       or code not in requested for code in values)
                or len(set(values)) != len(values)):
            return BatchOutcome(unknown=requested, error="invalid_backend_response")
        arrays.append(set(values))
    failed, not_found = arrays
    if failed & not_found or (response.status == 200 and (
            (code == SUCCESS and bool(failed or not_found))
            or (code == PARTIAL and not (failed or not_found)))):
        return BatchOutcome(unknown=requested, error="invalid_backend_response")
    remainder = requested - failed - not_found
    return BatchOutcome(successful=remainder if response.status == 200 else set(),
                        failed=failed, not_found=not_found,
                        unknown=remainder if response.status != 200 else set())
