"""Price sync contract (price_conflicts): request validation and response classification.

Pure functions: no DB, no HTTP. The legacy failed_product_codes contract stays in backend.py.
"""

from collections.abc import Mapping
from dataclasses import dataclass

from .backend import BackendError, PARTIAL, SUCCESS
from .comparison import normalize_code
from .outbox import AUTH_FAILED, MAX_BIGINT, MAX_INTEGER

MAX_PRODUCTS = 100
FIELDS = ("id", "product_code", "old_price", "new_price")

# Result kinds.
COMPLETED = "completed"                  # validated HTTP 200; per-product results are final
AUTH_REJECTED = "auth_rejected"          # 401/403: stop all sending (error_code AUTH_FAILED)
REQUEST_REJECTED = "request_rejected"    # 400: nothing confirmed, offending product unknown
TEMPORARY_FAILURE = "temporary_failure"  # 5xx, timeout, connection: resend the same values
INVALID_RESPONSE = "invalid_response"    # HTTP 200 breaking the contract: nothing confirmed
UNEXPECTED_STATUS = "unexpected_status"  # any other status: nothing confirmed
NOT_SENT = "not_sent"                    # the client refused the batch before connecting


@dataclass(frozen=True)
class SyncItem:
    record_id: object
    product_code: str
    old_price: int
    new_price: int

    def payload(self):
        return {"product_code": self.product_code, "old_price": self.old_price, "new_price": self.new_price}


@dataclass(frozen=True)
class RejectedItem:
    record_id: object
    product_code: object
    reason: str


@dataclass(frozen=True)
class PreparedBatch:
    items: tuple = ()
    rejected: tuple = ()
    error: str = None   # batch-level: empty_batch, batch_too_large, no_valid_products

    @property
    def sendable(self):
        return self.error is None

    def products(self):
        """The list BackendClient.send expects; changed_count is len() of exactly this."""
        if not self.sendable:
            raise ValueError("batch_not_sendable")
        return [item.payload() for item in self.items]

    def body(self):
        products = self.products()
        return {"changed_count": len(products), "products": products}


@dataclass(frozen=True)
class PriceConflict:
    product_code: str
    current_price: int


@dataclass(frozen=True)
class SyncResult:
    """Every requested product_code is in exactly one of the four code groups."""
    kind: str
    http_status: int = None
    error_code: str = None
    succeeded: tuple = ()
    not_found: tuple = ()
    conflicts: tuple = ()
    unresolved: tuple = ()


def _positive_price(value):
    return type(value) is int and 0 < value <= MAX_INTEGER


def _valid_code(value):
    return (isinstance(value, str) and normalize_code(value) == value and len(value) <= 19
            and int(value) <= MAX_BIGINT)


def _field(record, name):
    if isinstance(record, Mapping):
        return record[name] if name in record else _MISSING
    return getattr(record, name, _MISSING)


_MISSING = object()


def prepare_batch(records):
    """Split records (OutboxRecord or mappings with id/product_code/old_price/new_price).

    Never mutates, converts or truncates input. Every record sharing a duplicated
    product_code is rejected; none is picked. A batch over 100 is not split.
    """
    records = list(records)
    if not records:
        return PreparedBatch(error="empty_batch")
    if len(records) > MAX_PRODUCTS:
        return PreparedBatch(error="batch_too_large")
    values = [{name: _field(record, name) for name in FIELDS} for record in records]
    counts = {}
    for value in values:
        if _valid_code(value["product_code"]):
            counts[value["product_code"]] = counts.get(value["product_code"], 0) + 1
    items, rejected = [], []
    for value in values:
        record_id = None if value["id"] is _MISSING else value["id"]
        code = value["product_code"]
        if any(value[name] is _MISSING for name in FIELDS):
            reason = "missing_field"
        elif not _valid_code(code):
            reason = "invalid_product_code"
        elif counts[code] > 1:
            reason = "duplicate_product_code"
        elif not (_positive_price(value["old_price"]) and _positive_price(value["new_price"])):
            reason = "invalid_price"
        elif value["old_price"] == value["new_price"]:
            reason = "same_price"
        else:
            items.append(SyncItem(record_id, code, value["old_price"], value["new_price"]))
            continue
        rejected.append(RejectedItem(record_id, code if isinstance(code, str) else None, reason))
    return PreparedBatch(tuple(items), tuple(rejected), None if items else "no_valid_products")


def _unresolved(batch, kind, http_status, error_code):
    return SyncResult(kind, http_status, error_code,
                      unresolved=tuple(item.product_code for item in batch.items))


def _codes(values, requested):
    # Type checks first: an object/array entry must not reach the set/dict lookups.
    if (not isinstance(values, list) or not all(_valid_code(code) and code in requested for code in values)
            or len(set(values)) != len(values)):
        return None
    return values


def _completed(payload, batch):
    """Return a SyncResult for a contract-valid HTTP 200 body, or an INVALID_RESPONSE code."""
    if (not isinstance(payload, dict) or not isinstance(payload.get("message"), str)
            or not isinstance(payload.get("data"), dict)):
        return "INVALID_RESPONSE_FIELDS"
    code, data = payload.get("code"), payload["data"]
    if code not in (SUCCESS, PARTIAL):
        return "INVALID_RESPONSE_CODE"
    if "not_found_product_codes" not in data or "price_conflicts" not in data:
        return "INVALID_RESPONSE_FIELDS"
    count = data.get("received_count")
    if type(count) is not int or count != len(batch.items):
        return "INVALID_RESPONSE_COUNT"
    sent = {item.product_code: item for item in batch.items}
    not_found = _codes(data["not_found_product_codes"], sent)
    conflicts = data["price_conflicts"]
    if not_found is None or not isinstance(conflicts, list):
        return "INVALID_RESPONSE_PRODUCTS"
    if any(not isinstance(entry, dict) or "product_code" not in entry or "current_price" not in entry
           for entry in conflicts):
        return "INVALID_RESPONSE_FIELDS"
    conflict_codes = _codes([entry["product_code"] for entry in conflicts], sent)
    if conflict_codes is None or set(conflict_codes) & set(not_found):
        return "INVALID_RESPONSE_PRODUCTS"
    for entry in conflicts:
        item, price = sent[entry["product_code"]], entry["current_price"]
        if not _positive_price(price) or price in (item.old_price, item.new_price):
            return "INVALID_RESPONSE_PRICE"
    if (code == SUCCESS) != (not not_found and not conflicts):
        return "INVALID_RESPONSE_CODE"
    missing, prices = set(not_found), {entry["product_code"]: entry["current_price"] for entry in conflicts}
    # Request order everywhere; the payload itself is only read.
    return SyncResult(
        COMPLETED, 200, None,
        succeeded=tuple(code for code in sent if code not in missing and code not in prices),
        not_found=tuple(code for code in sent if code in missing),
        conflicts=tuple(PriceConflict(code, prices[code]) for code in sent if code in prices),
    )


def parse_sync_response(response, batch):
    """Classify an HttpResult(status, payload) for the PreparedBatch that was actually sent.

    Per-product results exist only for a fully validated HTTP 200. 401/403 win over
    any body. A 5xx never yields successes, whatever its body says.
    """
    if not batch.sendable:
        raise ValueError("batch_not_sendable")
    status, payload = response.status, response.payload
    if status in (401, 403):
        return _unresolved(batch, AUTH_REJECTED, status, AUTH_FAILED)
    if status == 400:
        code = payload.get("code") if isinstance(payload, dict) else None
        return _unresolved(batch, REQUEST_REJECTED, status,
                           code if code in ("INVALID_REQUEST", "INVALID_JSON_FORMAT") else "BAD_REQUEST")
    if type(status) is int and 500 <= status <= 599:
        return _unresolved(batch, TEMPORARY_FAILURE, status, "SERVER_ERROR")
    if status != 200:
        return _unresolved(batch, UNEXPECTED_STATUS, status if type(status) is int else None,
                           "UNEXPECTED_STATUS")
    result = _completed(payload, batch)
    return result if isinstance(result, SyncResult) else _unresolved(batch, INVALID_RESPONSE, 200, result)


def classify_send_error(error, batch):
    """Classify an exception raised by BackendClient.send for the batch that was being sent.

    BackendError keeps http_status when the body was not JSON, so an HTML/empty
    401/403 is still an authentication failure rather than a retryable error.
    """
    if not batch.sendable:
        raise ValueError("batch_not_sendable")
    status = getattr(error, "http_status", None) if isinstance(error, BackendError) else None
    if status in (401, 403):
        return _unresolved(batch, AUTH_REJECTED, status, AUTH_FAILED)
    if not isinstance(error, BackendError):
        return _unresolved(batch, TEMPORARY_FAILURE, None, "BACKEND_TRANSPORT_FAILED")
    if error.phase == "prepare":
        return _unresolved(batch, NOT_SENT, None, "REQUEST_NOT_SENT")
    if status == 400:
        return _unresolved(batch, REQUEST_REJECTED, status, "BAD_REQUEST")
    if type(status) is int and 500 <= status <= 599:
        return _unresolved(batch, TEMPORARY_FAILURE, status, "SERVER_ERROR")
    if error.code == "backend_invalid_json":
        if status == 200:
            return _unresolved(batch, INVALID_RESPONSE, status, "INVALID_RESPONSE_JSON")
        return _unresolved(batch, UNEXPECTED_STATUS, status, "UNEXPECTED_STATUS")
    if error.code == "backend_timeout":
        return _unresolved(batch, TEMPORARY_FAILURE, status, "BACKEND_TIMEOUT")
    return _unresolved(batch, TEMPORARY_FAILURE, status, "BACKEND_CONNECTION_ERROR")
