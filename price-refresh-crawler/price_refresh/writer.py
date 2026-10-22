"""One PostgreSQL session owns the collection lock and every price+outbox transaction."""

import logging
import re
from dataclasses import dataclass

from .comparison import PriceChange, normalize_code, valid_price
from .outbox import OutboxError, insert_pending
from .readiness import check_readiness

logger = logging.getLogger(__name__)
# One fixed bigint key for ALL collection (--apply) runs (ASCII PRICEREF).
COLLECT_LOCK_KEY = 0x5052494345524546
# Outbox delivery and its manual resume (ASCII PRICEDLV); never blocks or is blocked by collection.
DELIVERY_LOCK_KEY = 0x5052494345444C56


class ApplyError(RuntimeError):
    """Static internal codes only: driver messages can contain credentials."""

    def __init__(self, code):
        self.code = code
        super().__init__(code)


class SessionLost(ApplyError):
    pass


class CommitUnknown(SessionLost):
    pass


@dataclass
class WriteResult:
    status: str
    change: object = None
    error: str = None
    # "database" or "outbox": which step of the product transaction failed.
    error_category: str = None
    outbox_id: int = None


def sqlstate(error):
    value = getattr(error, "sqlstate", None)
    return value if isinstance(value, str) and re.fullmatch(r"[0-9A-Z]{5}", value) else None


def connection_failure(connection, error):
    state = sqlstate(error)
    return (isinstance(error, SessionLost) or connection.closed or connection.broken
            or state is None or state.startswith("08")
            or state in {"57P01", "57P02", "57P03", "40003"})


class LockedSession:
    lock_key = None

    def __init__(self, settings, *, connect=None):
        self.settings = settings
        self._connect = connect
        self.connection = None
        self.acquired = False
        self.backend_pid = None

    def open(self):
        try:
            connect = self._connect
            if connect is None:
                import psycopg
                connect = psycopg.connect
            self.connection = connect(self.settings.url,
                                      connect_timeout=self.settings.connect_timeout,
                                      autocommit=True)
            self.connection.read_only = False
            with self.connection.cursor() as cursor:
                cursor.execute("SELECT pg_try_advisory_lock(%s), pg_backend_pid()", (self.lock_key,))
                self.acquired, self.backend_pid = cursor.fetchone()
            return self.acquired
        except Exception:
            raise SessionLost("lock_connection_failed") from None

    def assert_lock(self):
        try:
            if not self.acquired or self.connection.closed or self.connection.broken:
                raise SessionLost("lock_session_lost")
            with self.connection.cursor() as cursor:
                # Checking ownership does not re-enter/increment the advisory lock.
                # PID also detects an incompatible transaction-pooling connection.
                cursor.execute(
                    "SELECT pg_backend_pid() = %s AND EXISTS ("
                    "SELECT 1 FROM pg_locks WHERE locktype = 'advisory' "
                    "AND pid = pg_backend_pid() AND classid::bigint = %s "
                    "AND objid::bigint = %s AND objsubid = 1 AND granted)",
                    (self.backend_pid, self.lock_key >> 32, self.lock_key & 0xffffffff),
                )
                if cursor.fetchone()[0] is not True:
                    raise SessionLost("lock_session_lost")
        except SessionLost:
            raise
        except Exception:
            raise SessionLost("lock_session_lost") from None

    def close(self):
        if self.connection is None:
            return
        try:
            if self.acquired and not self.connection.closed and not self.connection.broken:
                with self.connection.cursor() as cursor:
                    cursor.execute("SELECT pg_advisory_unlock(%s)", (self.lock_key,))
        except Exception:
            logger.error("apply_cleanup_failed code=lock_release_failed")
        finally:
            try:
                self.connection.close()
            except Exception:
                logger.error("apply_cleanup_failed code=session_close_failed")
            self.acquired = False


class PostgresApplySession(LockedSession):
    lock_key = COLLECT_LOCK_KEY

    def __init__(self, settings, *, connect=None, readiness=check_readiness):
        super().__init__(settings, connect=connect)
        self._readiness = readiness

    def check_readiness(self):
        """Catalog-only preflight on the autocommit lock session; ReadinessError propagates."""
        self.assert_lock()
        return self._readiness(self.connection)

    def fetch_prices(self):
        self.assert_lock()
        try:
            with self.connection.cursor() as cursor:
                cursor.execute("SELECT product_code, price FROM products")
                return cursor.fetchall()
        except Exception:
            raise ApplyError("initial_price_read_failed") from None

    def write_one(self, code, new_price, scan_run_id):
        """Commit one price change, plus its PENDING outbox record when recommended.

        The flag is read from the locked row in the same transaction, so a price
        UPDATE can never commit without the outbox INSERT it requires.
        """
        if (not isinstance(code, str) or normalize_code(code) != code or len(code) > 19
                or int(code) > 9223372036854775807 or valid_price(new_price) is None):
            return WriteResult("failed", error="invalid_price_candidate", error_category="database")
        phase, step, wrote = "transaction", "database", False
        try:
            self.assert_lock()
            # Explicit short transaction on the SAME autocommit session.
            # Session-level advisory locks survive its COMMIT/ROLLBACK.
            with self.connection.transaction():
                self.assert_lock()
                with self.connection.cursor() as cursor:
                    cursor.execute("SELECT price, has_been_recommended FROM products "
                                   "WHERE product_code = %s FOR UPDATE", (int(code),))
                    row = cursor.fetchone()
                    if row is None:
                        result = WriteResult("missing")
                    elif valid_price(row[0]) is None:
                        raise ApplyError("invalid_current_database_price")
                    elif row[0] == new_price:
                        result = WriteResult("same")
                    elif type(row[1]) is not bool:
                        # NULL/unexpected is never treated as FALSE.
                        raise ApplyError("invalid_recommendation_flag")
                    else:
                        old_price, recommended = row
                        cursor.execute("UPDATE products SET price = %s WHERE product_code = %s",
                                       (new_price, int(code)))
                        if cursor.rowcount != 1:
                            raise ApplyError("unexpected_update_count")
                        wrote = True
                        record = None
                        if recommended:
                            step = "outbox"
                            record = insert_pending(self.connection, scan_run_id, code, old_price, new_price)
                        result = WriteResult("updated", PriceChange(code, old_price, new_price),
                                             outbox_id=record.id if record else None)
                self.assert_lock()
                phase = "commit"
            # Only reachable after the context manager confirms COMMIT.
            return result
        except Exception as error:
            state = sqlstate(error)
            known = isinstance(error, (ApplyError, OutboxError))
            unsafe = connection_failure(self.connection, error)
            if phase == "commit" and wrote and unsafe:
                raise CommitUnknown("commit_outcome_unknown") from None
            if isinstance(error, SessionLost) or (unsafe and not known):
                raise SessionLost("lock_session_lost") from None
            # The transaction context rolls back statement/known COMMIT failures,
            # including the price UPDATE when the outbox INSERT failed.
            # Validate that rollback completed and the session still owns its lock.
            try:
                self.connection.rollback()
                self.assert_lock()
            except Exception:
                raise SessionLost("rollback_or_lock_failed") from None
            return WriteResult("failed", error=(error.code if known else "sqlstate_" + state),
                               error_category=("outbox" if isinstance(error, OutboxError)
                                               or (step == "outbox" and phase != "commit")
                                               else "database"))


class DeliverySession(LockedSession):
    """Lock for the outbox delivery job and manual resume; separate from collection."""

    lock_key = DELIVERY_LOCK_KEY
