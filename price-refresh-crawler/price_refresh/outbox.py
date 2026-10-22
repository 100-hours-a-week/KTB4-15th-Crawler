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
