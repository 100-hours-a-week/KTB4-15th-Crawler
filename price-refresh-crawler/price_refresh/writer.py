"""One PostgreSQL session owns the run lock and every price transaction."""

import logging
import re
from dataclasses import dataclass

from .comparison import PriceChange, normalize_code, valid_price

logger = logging.getLogger(__name__)
# One fixed bigint key for ALL apply runs (ASCII PRICEREF).
LOCK_KEY = 0x5052494345524546


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


def sqlstate(error):
    value = getattr(error, "sqlstate", None)
    return value if isinstance(value, str) and re.fullmatch(r"[0-9A-Z]{5}", value) else None


def connection_failure(connection, error):
    state = sqlstate(error)
    return (isinstance(error, SessionLost) or connection.closed or connection.broken
            or state is None or state.startswith("08")
            or state in {"57P01", "57P02", "57P03", "40003"})


class PostgresApplySession:
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
                cursor.execute("SELECT pg_try_advisory_lock(%s), pg_backend_pid()", (LOCK_KEY,))
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
                    (self.backend_pid, LOCK_KEY >> 32, LOCK_KEY & 0xffffffff),
                )
                if cursor.fetchone()[0] is not True:
                    raise SessionLost("lock_session_lost")
        except SessionLost:
            raise
        except Exception:
            raise SessionLost("lock_session_lost") from None

    def fetch_prices(self):
        self.assert_lock()
        try:
            with self.connection.cursor() as cursor:
                cursor.execute("SELECT product_code, price FROM products")
                return cursor.fetchall()
        except Exception:
            raise ApplyError("initial_price_read_failed") from None

    def write_one(self, code, new_price):
        if (not isinstance(code, str) or normalize_code(code) != code or len(code) > 19
                or int(code) > 9223372036854775807 or valid_price(new_price) is None):
            return WriteResult("failed", error="invalid_price_candidate")
        phase, wrote = "transaction", False
        try:
            self.assert_lock()
            # Explicit short transaction on the SAME autocommit session.
            # Session-level advisory locks survive its COMMIT/ROLLBACK.
            with self.connection.transaction():
                self.assert_lock()
                with self.connection.cursor() as cursor:
                    cursor.execute("SELECT price FROM products WHERE product_code = %s FOR UPDATE",
                                   (int(code),))
                    row = cursor.fetchone()
                    if row is None:
                        result = WriteResult("missing")
                    elif valid_price(row[0]) is None:
                        raise ApplyError("invalid_current_database_price")
                    elif row[0] == new_price:
                        result = WriteResult("same")
                    else:
                        old_price = row[0]
                        cursor.execute("UPDATE products SET price = %s WHERE product_code = %s",
                                       (new_price, int(code)))
                        if cursor.rowcount != 1:
                            raise ApplyError("unexpected_update_count")
                        wrote = True
                        result = WriteResult("updated", PriceChange(code, old_price, new_price))
                self.assert_lock()
                phase = "commit"
            # Only reachable after the context manager confirms COMMIT.
            return result
        except Exception as error:
            state = sqlstate(error)
            unsafe = connection_failure(self.connection, error)
            if phase == "commit" and wrote and unsafe:
                raise CommitUnknown("commit_outcome_unknown") from None
            if isinstance(error, SessionLost) or (unsafe and not isinstance(error, ApplyError)):
                raise SessionLost("lock_session_lost") from None
            # The transaction context rolls back statement/known COMMIT failures.
            # Validate that rollback completed and the session still owns its lock.
            try:
                self.connection.rollback()
                self.assert_lock()
            except Exception:
                raise SessionLost("rollback_or_lock_failed") from None
            return WriteResult("failed", error=(error.code if isinstance(error, ApplyError)
                                                else "sqlstate_" + state))

    def close(self):
        if self.connection is None:
            return
        try:
            if self.acquired and not self.connection.closed and not self.connection.broken:
                with self.connection.cursor() as cursor:
                    cursor.execute("SELECT pg_advisory_unlock(%s)", (LOCK_KEY,))
        except Exception:
            logger.error("apply_cleanup_failed code=lock_release_failed")
        finally:
            try:
                self.connection.close()
            except Exception:
                logger.error("apply_cleanup_failed code=session_close_failed")
            self.acquired = False
