import logging
import os
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone

from .outbox import (
    OutboxError, delivery_stop_status, mark_attempts, mark_auth_failed, mark_blocked, mark_done,
    mark_not_found, mark_price_conflict, mark_retry, select_due,
)
from .price_sync import (
    AUTH_REJECTED, COMPLETED, NOT_SENT, REQUEST_REJECTED, classify_send_error, parse_sync_response,
    prepare_batch,
)
from .writer import SessionLost, connection_failure, sqlstate

logger = logging.getLogger(__name__)
REQUEST_REJECTED_MESSAGE = "Backend rejected the whole request without naming a product; check before resuming"


class DeliveryError(RuntimeError):
    """Static internal codes only."""

    def __init__(self, code):
        self.code = code
        super().__init__(code)


@dataclass(frozen=True)
class DeliverySettings:
    """Per-record retry backoff (base doubling up to max) and per-run request/batch caps.

    max_batches also bounds batches that send nothing (all records invalid).
    """
    retry_delay_seconds: int = 300
    retry_max_delay_seconds: int = 3600
    max_requests: int = 10
    max_batches: int = 20

    def __post_init__(self):
        for value, upper in ((self.retry_delay_seconds, 86400), (self.retry_max_delay_seconds, 86400),
                             (self.max_requests, 1000), (self.max_batches, 10000)):
            if type(value) is not int or not 1 <= value <= upper:
                raise ValueError("invalid_delivery_settings")
        if self.retry_max_delay_seconds < self.retry_delay_seconds:
            raise ValueError("invalid_delivery_settings")

    @classmethod
    def from_env(cls):
        try:
            return cls(int(os.getenv("PRICE_REFRESH_DELIVERY_RETRY_DELAY_SECONDS", "300")),
                       int(os.getenv("PRICE_REFRESH_DELIVERY_RETRY_MAX_DELAY_SECONDS", "3600")),
                       int(os.getenv("PRICE_REFRESH_DELIVERY_MAX_REQUESTS", "10")),
                       int(os.getenv("PRICE_REFRESH_DELIVERY_MAX_BATCHES", "20")))
        except (TypeError, ValueError):
            raise ValueError("invalid_delivery_settings") from None

    def retry_at(self, now, attempt_count):
        exponent = min(max(attempt_count, 1) - 1, 20)
        return now + timedelta(seconds=min(self.retry_delay_seconds * 2 ** exponent, self.retry_max_delay_seconds))


@dataclass
class DeliveryRunReport:
    status: str = "aborted"
    batch_count: int = 0
    selected_count: int = 0
    invalid_blocked_count: int = 0
    attempted_count: int = 0
    http_request_count: int = 0
    succeeded_count: int = 0
    conflict_retry_count: int = 0
    error_retry_count: int = 0
    not_found_count: int = 0
    request_rejected_blocked_count: int = 0
    auth_stopped: bool = False
    auth_stop_saved: object = None
    result_save_failed: bool = False
    outcome_unknown: bool = False
    batch_limit_reached: bool = False
    limit_reason: str = None   # max_requests or max_batches; due records remain for the next run
    stop_reason: str = None
    error: str = None

    def to_dict(self):
        return dict(asdict(self), mode="deliver")


EXIT_CODES = {"completed": 0, "skipped_concurrent": 0, "auth_stopped": 3}


def _utc_now():
    return datetime.now(timezone.utc)


class _Run:
    def __init__(self, session, client, settings, clock, report):
        self.session, self.client, self.settings, self.clock = session, client, settings, clock
        self.connection = session.connection
        self.report = report
        # Only records due at run start: anything rescheduled or inserted during this run
        # waits for a later run, so the run cannot keep feeding itself.
        self.cutoff = clock()

    def write(self, operation):
        """One short transaction; any unapplied change raises and rolls everything back."""
        self.session.assert_lock()
        committing = False
        try:
            with self.connection.transaction():
                operation()
                self.session.assert_lock()
                committing = True
        except Exception as error:
            if committing and connection_failure(self.connection, error):
                raise DeliveryError("commit_outcome_unknown") from None
            raise

    def assert_idle(self):
        """Lock still held and no transaction open across the HTTP request."""
        self.session.assert_lock()
        if not self.connection.autocommit or self.connection.info.transaction_status != 0:
            raise SessionLost("delivery_transaction_not_idle")

    def batch(self):
        """Process one due batch; returns False when the run must end."""
        now = self.clock()
        self.session.assert_lock()
        records = select_due(self.connection, now=self.cutoff, limit=self.client.settings.batch_size)
        if not records:
            return False
        report = self.report
        for limit in ("max_requests", "max_batches"):
            used = report.http_request_count if limit == "max_requests" else report.batch_count
            if used >= getattr(self.settings, limit):
                report.batch_limit_reached, report.limit_reason = True, limit
                return False
        report.batch_count += 1
        report.selected_count += len(records)
        batch = prepare_batch(records)
        if batch.error == "batch_too_large":
            raise DeliveryError("batch_too_large")
        if batch.rejected:
            def block():
                for item in batch.rejected:
                    _applied(mark_blocked(self.connection, item.record_id, now=now,
                                          error_code="LOCAL_" + item.reason.upper()))
            self.write(block)
            report.invalid_blocked_count += len(batch.rejected)
            logger.warning("delivery_invalid_records_blocked count=%s", len(batch.rejected))
        if not batch.sendable:
            return True
        attempts = {}

        def attempt():
            marked = mark_attempts(self.connection, [item.record_id for item in batch.items], now=now,
                                   next_attempt_at=self.settings.retry_at(now, 1))
            attempts.update((record.id, record) for record in marked)
            # The exact prices being sent must be the ones just marked as attempted.
            if ({record.id: (record.product_code, record.old_price, record.new_price) for record in marked}
                    != {item.record_id: (item.product_code, item.old_price, item.new_price) for item in batch.items}):
                raise DeliveryError("attempt_not_recorded")
        self.write(attempt)
        report.attempted_count += len(attempts)
        self.assert_idle()
        try:
            result = parse_sync_response(self.client.send(batch.products()), batch)
        except Exception as error:
            result = classify_send_error(error, batch)
        if result.kind != NOT_SENT:
            report.http_request_count += 1
        logger.info("delivery_batch_end kind=%s http_status=%s code=%s products=%s succeeded=%s "
                    "conflicts=%s not_found=%s unresolved=%s", result.kind, result.http_status,
                    result.error_code, len(batch.items), len(result.succeeded), len(result.conflicts),
                    len(result.not_found), len(result.unresolved))
        try:
            return self.store(batch, result, attempts, self.clock())
        except Exception:
            # The backend may already have applied the request; records stay ATTEMPT_UNCONFIRMED.
            report.result_save_failed = True
            raise

    def store(self, batch, result, attempts, now):
        ids = {item.product_code: item.record_id for item in batch.items}
        report, connection = self.report, self.connection
        if result.kind == AUTH_REJECTED:
            report.auth_stopped, report.stop_reason = True, "auth_rejected"
            try:
                self.write(lambda: mark_auth_failed(connection, list(ids.values()), now=now,
                                                    next_attempt_at=self.settings.retry_at(now, 1),
                                                    http_status=result.http_status))
                report.auth_stop_saved = True
            except Exception as error:
                report.auth_stop_saved = False
                report.error = getattr(error, "code", None) or "auth_stop_not_saved"
                logger.error("delivery_auth_stop_not_saved code=%s", report.error)
            logger.error("delivery_stopped_by_auth http_status=%s saved=%s", result.http_status,
                         report.auth_stop_saved)
            return False
        if result.kind == COMPLETED:
            def save():
                for code in result.succeeded:
                    _applied(mark_done(connection, ids[code], now=now, http_status=200))
                for conflict in result.conflicts:
                    record_id = ids[conflict.product_code]
                    _applied(mark_price_conflict(
                        connection, record_id, now=now, backend_price=conflict.current_price, http_status=200,
                        next_attempt_at=self.settings.retry_at(now, attempts[record_id].attempt_count)))
                for code in result.not_found:
                    _applied(mark_not_found(connection, ids[code], now=now, http_status=200))
            self.write(save)
            report.succeeded_count += len(result.succeeded)
            report.conflict_retry_count += len(result.conflicts)
            report.not_found_count += len(result.not_found)
            return True
        if result.kind == REQUEST_REJECTED:
            def block():
                for record_id in ids.values():
                    _applied(mark_blocked(connection, record_id, now=now, error_code=result.error_code,
                                          error_message=REQUEST_REJECTED_MESSAGE, http_status=result.http_status))
            self.write(block)
            report.request_rejected_blocked_count += len(ids)
            logger.error("delivery_request_rejected code=%s blocked=%s", result.error_code, len(ids))
            return True

        # temporary_failure, invalid_response, unexpected_status, not_sent: keep request values.
        def retry():
            for record_id in ids.values():
                _applied(mark_retry(connection, record_id, now=now, error_code=result.error_code,
                                    http_status=result.http_status,
                                    next_attempt_at=self.settings.retry_at(now, attempts[record_id].attempt_count)))
        self.write(retry)
        report.error_retry_count += len(ids)
        report.stop_reason = result.kind
        return False


def _applied(record):
    if record is None:
        raise DeliveryError("result_not_saved")
    return record


def run_delivery(session, client, settings, *, clock=_utc_now, emit=None):
    """Settings and client must be validated by the caller BEFORE this opens the database."""
    report = DeliveryRunReport()
    started = time.monotonic()
    logger.info("delivery_start")
    try:
        if not session.open():
            report.status = "skipped_concurrent"
            logger.info("delivery_skipped reason=lock_busy")
            return report
        session.assert_lock()
        try:
            stop = delivery_stop_status(session.connection)
        except Exception:
            raise DeliveryError("stop_check_failed") from None
        if stop.stopped:
            report.status, report.auth_stopped, report.stop_reason = "auth_stopped", True, "auth_failed_pending"
            logger.warning("delivery_skipped reason=auth_stop auth_failed=%s http_statuses=%s",
                           stop.auth_failed_count, list(stop.http_statuses))
            return report
        run = _Run(session, client, settings, clock, report)
        while run.batch():
            pass
        if report.auth_stopped:
            report.status = "auth_stopped" if report.auth_stop_saved else "auth_stop_not_saved"
        elif (report.stop_reason or report.invalid_blocked_count or report.request_rejected_blocked_count):
            report.status = "completed_with_errors"
        else:
            report.status = "completed"
    except SessionLost as error:
        report.error = error.code
        logger.error("delivery_aborted code=%s", error.code)
    except (DeliveryError, OutboxError) as error:
        report.outcome_unknown = error.code == "commit_outcome_unknown"
        report.status = "outcome_unknown" if report.outcome_unknown else "failed"
        report.error = error.code
        logger.error("delivery_aborted code=%s", error.code)
    except Exception as error:
        report.status = "failed"
        report.error = "sqlstate_" + sqlstate(error) if sqlstate(error) else "delivery_failed"
        logger.error("delivery_aborted code=%s", report.error)
    finally:
        try:
            if emit is not None:
                emit(report.to_dict())
            logger.info(
                "delivery_end status=%s duration_seconds=%.3f selected=%s invalid_blocked=%s attempted=%s "
                "http_requests=%s succeeded=%s conflict_retry=%s error_retry=%s not_found=%s "
                "request_rejected_blocked=%s auth_stopped=%s auth_stop_saved=%s result_save_failed=%s "
                "outcome_unknown=%s batch_limit_reached=%s limit_reason=%s batches=%s stop_reason=%s error=%s",
                report.status, time.monotonic() - started, report.selected_count, report.invalid_blocked_count,
                report.attempted_count, report.http_request_count, report.succeeded_count,
                report.conflict_retry_count, report.error_retry_count, report.not_found_count,
                report.request_rejected_blocked_count, report.auth_stopped, report.auth_stop_saved,
                report.result_save_failed, report.outcome_unknown, report.batch_limit_reached,
                report.limit_reason, report.batch_count,
                report.stop_reason, report.error)
        finally:
            session.close()
    return report
