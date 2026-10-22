import uuid
from dataclasses import dataclass, fields
from datetime import datetime

from .comparison import normalize_code, valid_price

PENDING = "PENDING"
RETRY_WAIT = "RETRY_WAIT"
DONE = "DONE"
NOT_FOUND = "NOT_FOUND"
BLOCKED = "BLOCKED"
STATUSES = (PENDING, RETRY_WAIT, DONE, NOT_FOUND, BLOCKED)
AUTH_FAILED = "AUTH_FAILED"
MAX_BIGINT = 9223372036854775807
MAX_INTEGER = 2147483647


class OutboxError(RuntimeError):
    """Static internal codes only; never carries driver messages or row values."""

    def __init__(self, code):
        self.code = code
        super().__init__(code)


@dataclass(frozen=True)
class OutboxRecord:
    id: int
    scan_run_id: uuid.UUID
    product_code: str
    original_old_price: int
    old_price: int
    new_price: int
    status: str
    attempt_count: int
    next_attempt_at: datetime | None
    last_attempt_at: datetime | None
    last_backend_price: int | None
    last_http_status: int | None
    last_error_code: str | None
    last_error_message: str | None
    created_at: datetime
    updated_at: datetime
    completed_at: datetime | None


COLUMNS = tuple(field.name for field in fields(OutboxRecord))
_SELECT_LIST = ", ".join(COLUMNS)


def _code(value):
    code = normalize_code(value)
    if code is None or len(code) > 19 or int(code) > MAX_BIGINT:
        return None
    return code


def _price(value):
    price = valid_price(value)
    return price if price is not None and price <= MAX_INTEGER else None


def record_from_row(row):
    if row is None or len(row) != len(COLUMNS):
        raise OutboxError("invalid_outbox_row")
    values = dict(zip(COLUMNS, row))
    code = _code(values["product_code"]) if type(values["product_code"]) is int else None
    prices = [_price(values[name]) for name in ("original_old_price", "old_price", "new_price")]
    if (type(values["id"]) is not int or not isinstance(values["scan_run_id"], uuid.UUID)
            or code is None or None in prices or values["status"] not in STATUSES
            or type(values["attempt_count"]) is not int or values["attempt_count"] < 0
            or not isinstance(values["created_at"], datetime)
            or not isinstance(values["updated_at"], datetime)
            or (values["status"] == DONE) != (values["completed_at"] is not None)):
        raise OutboxError("invalid_outbox_row")
    values["product_code"] = code
    return OutboxRecord(**values)


def insert_pending(connection, scan_run_id, product_code, old_price, new_price):
    try:
        run_id = scan_run_id if isinstance(scan_run_id, uuid.UUID) else uuid.UUID(scan_run_id)
    except (TypeError, ValueError, AttributeError):
        raise OutboxError("invalid_outbox_record") from None
    code = _code(product_code) if isinstance(product_code, str) else None
    old, new = _price(old_price), _price(new_price)
    if code != product_code or old is None or new is None or old == new:
        raise OutboxError("invalid_outbox_record")
    with connection.cursor() as cursor:
        cursor.execute(
            "INSERT INTO public.price_change_outbox "
            "(scan_run_id, product_code, original_old_price, old_price, new_price) "
            "VALUES (%s, %s, %s, %s, %s) "
            "ON CONFLICT (scan_run_id, product_code) DO NOTHING "
            f"RETURNING {_SELECT_LIST}",
            (run_id, int(code), old, old, new),
        )
        row = cursor.fetchone()
        if row is None:
            cursor.execute(
                f"SELECT {_SELECT_LIST} FROM public.price_change_outbox "
                "WHERE scan_run_id = %s AND product_code = %s",
                (run_id, int(code)),
            )
            row = cursor.fetchone()
    record = record_from_row(row)
    # old_price may be adjusted later; the original change is what identifies a duplicate.
    if (record.original_old_price, record.new_price) != (old, new):
        raise OutboxError("outbox_record_conflict")
    return record


ATTEMPT_UNCONFIRMED = "ATTEMPT_UNCONFIRMED"
MAX_BATCH = 100
_ERROR_CODE_CHARS = frozenset("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_.-")
_MAX_ERROR_MESSAGE = 500
# Only an attempt marked by mark_attempts can receive a result.
_ATTEMPTED = f"status = '{RETRY_WAIT}' AND last_error_code = '{ATTEMPT_UNCONFIRMED}'"


def _time(value):
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise OutboxError("invalid_outbox_time")
    return value


def _schedule(now, next_attempt_at):
    if _time(next_attempt_at) <= _time(now):
        raise OutboxError("invalid_outbox_time")
    return now, next_attempt_at


def _record_id(value):
    if type(value) is not int or not 0 < value <= MAX_BIGINT:
        raise OutboxError("invalid_outbox_id")
    return value


def _http_status(value):
    if value is not None and (type(value) is not int or not 100 <= value <= 599):
        raise OutboxError("invalid_outbox_error")
    return value


def _error(code, message, http_status):
    if (not isinstance(code, str) or not 0 < len(code) <= 64 or not set(code) <= _ERROR_CODE_CHARS
            or code == ATTEMPT_UNCONFIRMED):
        raise OutboxError("invalid_outbox_error")
    if message is not None and (not isinstance(message, str) or len(message) > _MAX_ERROR_MESSAGE
                                or any(ord(c) < 32 or ord(c) == 127 for c in message)):
        raise OutboxError("invalid_outbox_error")
    return code, message, _http_status(http_status)


def _update(connection, sql, params):
    with connection.cursor() as cursor:
        cursor.execute(sql + f" RETURNING {_SELECT_LIST}", params)
        rows = cursor.fetchall()
    return [record_from_row(row) for row in rows]


def select_due(connection, *, now, limit=MAX_BATCH):
    """Due records, at most one per product: the oldest unfinished record of that product.

    DONE and NOT_FOUND never hold back later records; PENDING, RETRY_WAIT (even when
    scheduled in the future) and BLOCKED do. Ordered by next_attempt_at, id.
    Selection alone does not prevent concurrent senders; the caller holds a lock.
    """
    if type(limit) is not int or not 1 <= limit <= MAX_BATCH:
        raise OutboxError("invalid_outbox_limit")
    # Status literals (not parameters) let the planner match the partial indexes.
    with connection.cursor() as cursor:
        cursor.execute(
            f"SELECT {_SELECT_LIST} FROM public.price_change_outbox AS item "
            f"WHERE item.status IN ('{PENDING}', '{RETRY_WAIT}') AND item.next_attempt_at <= %s "
            "AND NOT EXISTS (SELECT 1 FROM public.price_change_outbox AS earlier "
            "WHERE earlier.product_code = item.product_code AND earlier.id < item.id "
            f"AND earlier.status NOT IN ('{DONE}', '{NOT_FOUND}')) "
            "ORDER BY item.next_attempt_at, item.id LIMIT %s",
            (_time(now), limit),
        )
        records = [record_from_row(row) for row in cursor.fetchall()]
    if len({record.product_code for record in records}) != len(records):
        raise OutboxError("duplicate_outbox_product")
    return records


def mark_attempts(connection, record_ids, *, now, next_attempt_at):
    """Record a send attempt BEFORE the HTTP request; returns the records actually marked.

    The record stays RETRY_WAIT until a result is stored, so a crash after sending
    makes it due again at next_attempt_at. The only place attempt_count increases.
    """
    now, next_attempt_at = _schedule(now, next_attempt_at)
    ids = [_record_id(value) for value in record_ids]
    if not ids or len(ids) > MAX_BATCH or len(set(ids)) != len(ids):
        raise OutboxError("invalid_outbox_id")
    return sorted(_update(connection,
        "UPDATE public.price_change_outbox SET status = %s, attempt_count = attempt_count + 1, "
        "last_attempt_at = %s, next_attempt_at = %s, last_error_code = %s, "
        "last_error_message = NULL, last_http_status = NULL, updated_at = %s "
        "WHERE id = ANY(%s) AND status IN (%s, %s)",
        (RETRY_WAIT, now, next_attempt_at, ATTEMPT_UNCONFIRMED, now, ids, PENDING, RETRY_WAIT)),
        key=lambda record: record.id)


def _one(records):
    return records[0] if records else None


def mark_done(connection, record_id, *, now, http_status=None):
    return _one(_update(connection,
        "UPDATE public.price_change_outbox SET status = %s, completed_at = %s, next_attempt_at = NULL, "
        "last_error_code = NULL, last_error_message = NULL, last_http_status = %s, updated_at = %s "
        f"WHERE id = %s AND {_ATTEMPTED}",
        (DONE, _time(now), _http_status(http_status), now, _record_id(record_id))))


def mark_retry(connection, record_id, *, now, next_attempt_at, error_code,
               error_message=None, http_status=None):
    """Temporary server/transport failure or unknown outcome; request prices are kept."""
    now, next_attempt_at = _schedule(now, next_attempt_at)
    code, message, status = _error(error_code, error_message, http_status)
    return _one(_update(connection,
        "UPDATE public.price_change_outbox SET next_attempt_at = %s, last_error_code = %s, "
        "last_error_message = %s, last_http_status = %s, updated_at = %s "
        f"WHERE id = %s AND {_ATTEMPTED}",
        (next_attempt_at, code, message, status, now, _record_id(record_id))))


def mark_price_conflict(connection, record_id, *, now, next_attempt_at, backend_price,
                        error_code="PRICE_CONFLICT", http_status=None):
    """Next request uses the validated backend price as old_price; products.price is untouched.

    Not applied when backend_price equals new_price: such a record could never be sent.
    """
    now, next_attempt_at = _schedule(now, next_attempt_at)
    code, _, status = _error(error_code, None, http_status)
    price = _price(backend_price)
    if price is None:
        raise OutboxError("invalid_backend_price")
    return _one(_update(connection,
        "UPDATE public.price_change_outbox SET old_price = %s, last_backend_price = %s, "
        "next_attempt_at = %s, last_error_code = %s, last_error_message = NULL, "
        "last_http_status = %s, updated_at = %s "
        f"WHERE id = %s AND {_ATTEMPTED} AND new_price <> %s",
        (price, price, next_attempt_at, code, status, now, _record_id(record_id), price)))


def mark_not_found(connection, record_id, *, now, http_status=None):
    """Excluded from retries; later records of the same product stay selectable."""
    return _one(_update(connection,
        "UPDATE public.price_change_outbox SET status = %s, next_attempt_at = NULL, "
        "last_error_code = %s, last_error_message = NULL, last_http_status = %s, updated_at = %s "
        f"WHERE id = %s AND {_ATTEMPTED}",
        (NOT_FOUND, NOT_FOUND, _http_status(http_status), _time(now), _record_id(record_id))))


def mark_blocked(connection, record_id, *, now, error_code, error_message=None, http_status=None):
    """Hold one record and every later record of its product until manual action.

    Allowed before an attempt (local validation) or after one. Authentication
    failures are not per-record and are rejected here.
    """
    code, message, status = _error(error_code, error_message, http_status)
    if code == AUTH_FAILED or status in {401, 403}:
        raise OutboxError("invalid_outbox_error")
    return _one(_update(connection,
        "UPDATE public.price_change_outbox SET status = %s, next_attempt_at = NULL, "
        "last_error_code = %s, last_error_message = %s, last_http_status = %s, updated_at = %s "
        "WHERE id = %s AND status IN (%s, %s)",
        (BLOCKED, code, message, status, _time(now), _record_id(record_id), PENDING, RETRY_WAIT)))
