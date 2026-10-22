import logging
from contextlib import closing
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

from .outbox import delivery_stop_status, release_auth_stop
from .writer import SessionLost, connection_failure, sqlstate

logger = logging.getLogger(__name__)


class DeliveryControlError(RuntimeError):
    """Static internal codes only; driver messages can contain the DSN."""

    def __init__(self, code):
        self.code = code
        super().__init__(code)


def fetch_delivery_status(settings, *, connect=None):
    """Read-only check on its own connection; any failure raises instead of reporting "allowed"."""
    try:
        if connect is None:
            import psycopg
            connect = psycopg.connect
        with closing(connect(settings.url, connect_timeout=settings.connect_timeout,
                             autocommit=False)) as connection:
            connection.read_only = True
            return delivery_stop_status(connection)
    except Exception:
        raise DeliveryControlError("delivery_status_check_failed") from None


@dataclass
class ResumeReport:
    status: str = "aborted"
    auth_failed_count: int = None
    http_statuses: list = field(default_factory=list)
    resumed_ids: list = field(default_factory=list)
    resumed_at: str = None
    error: str = None

    def to_dict(self):
        result = asdict(self)
        result.update(mode="resume_delivery", resumed_count=len(self.resumed_ids))
        return result


def run_resume(session, *, now=None, emit=None):
    """Under the delivery lock, clear every AUTH_FAILED stop in one short transaction.

    status: resumed, already_allowed, skipped_concurrent (another delivery/resume
    holds the lock), failed (rolled back; the stop remains) or
    commit_outcome_unknown (re-check with --delivery-status).
    """
    report = ResumeReport()
    now = now or datetime.now(timezone.utc)
    committing = False
    try:
        if not session.open():
            report.status = "skipped_concurrent"
            logger.info("delivery_resume_skipped reason=lock_busy")
            return report
        session.assert_lock()
        stop = delivery_stop_status(session.connection)
        report.auth_failed_count, report.http_statuses = stop.auth_failed_count, list(stop.http_statuses)
        if not stop.stopped:
            report.status = "already_allowed"
            logger.info("delivery_resume_not_needed auth_failed=0")
            return report
        logger.warning("delivery_stopped_by_auth_failure auth_failed=%s http_statuses=%s",
                       stop.auth_failed_count, list(stop.http_statuses))
        with session.connection.transaction():
            records = release_auth_stop(session.connection, now=now)
            session.assert_lock()
            committing = True
        report.resumed_ids = [record.id for record in records]
        report.resumed_at = now.isoformat()
        report.status = "resumed"
        logger.warning("delivery_resumed manually released=%s at=%s", len(records), report.resumed_at)
    except Exception as error:
        if committing and connection_failure(session.connection, error):
            report.status, report.error = "commit_outcome_unknown", "commit_outcome_unknown"
        else:
            report.status = "failed"
            report.error = (error.code if isinstance(error, SessionLost)
                            else "sqlstate_" + sqlstate(error) if sqlstate(error) else "resume_failed")
        logger.error("delivery_resume_failed status=%s code=%s", report.status, report.error)
    finally:
        try:
            if emit is not None:
                emit(report.to_dict())
        finally:
            session.close()
    return report
