"""Apply orchestration. No recommendation gate, backend calls or persistent queue."""

import logging
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone

from .comparison import compare_prices, normalize_code, valid_price
from .writer import ApplyError, CommitUnknown, SessionLost

logger = logging.getLogger(__name__)
KST = timezone(timedelta(hours=9), "KST")


def now_kst():
    return datetime.now(KST).isoformat(timespec="seconds")


@dataclass
class ApplyReport:
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    status: str = "aborted"
    target_count: object = None
    confirmed_count: object = None
    unconfirmed_count: object = None
    saved_changes: list = field(default_factory=list)
    failed_products: list = field(default_factory=list)
    unknown_products: list = field(default_factory=list)
    skipped_same_count: int = 0
    missing_count: int = 0
    scan_error_count: int = 0
    invalid_stored_row_count: int = 0
    error: object = None

    def to_dict(self):
        result = asdict(self)
        result.update(mode="apply", updated_count=len(self.saved_changes),
                      failed_count=len(self.failed_products), unknown_count=len(self.unknown_products))
        return result


def run_apply(session, collect, *, emit=None):
    """Emit results and log summary BEFORE releasing the session lock.

    collect must finish finalizing all observed maxima before any write starts.
    Tests inject it; the CLI rejects every combination of --apply and fixtures.
    """
    report = ApplyReport()
    started = time.monotonic()
    logger.info("apply_start run_id=%s time=%s", report.run_id, now_kst())
    active_code = None
    try:
        if not session.open():
            report.status = "skipped_concurrent"
            logger.info("apply_skipped run_id=%s reason=lock_busy", report.run_id)
        else:
            stored = session.fetch_prices()
            targets = {normalize_code(code) for code, _ in stored} - {None}
            report.target_count = len(targets)
            scan = collect()
            # Counts are restricted to AI targets, never all listing IDs.
            confirmed = {normalize_code(code) for code, price in scan.prices.items()
                         if valid_price(price) is not None} & targets
            report.confirmed_count = len(confirmed)
            report.unconfirmed_count = len(targets - confirmed)
            report.scan_error_count = sum(len(category.errors) for category in scan.categories)
            comparison = compare_prices(stored, scan.prices)
            report.invalid_stored_row_count = comparison.invalid_stored_rows
            if comparison.invalid_stored_rows:
                logger.warning("apply_invalid_stored_rows count=%s", comparison.invalid_stored_rows)
            session.assert_lock()
            for candidate in comparison.changes:
                active_code = candidate.product_code
                result = session.write_one(active_code, candidate.new_price)
                if result.status == "updated":
                    report.saved_changes.append(result.change)
                elif result.status == "same":
                    report.skipped_same_count += 1
                    logger.info("apply_product_skipped code=%s reason=same_price", active_code)
                elif result.status == "missing":
                    report.missing_count += 1
                    logger.warning("apply_product_skipped code=%s reason=product_missing", active_code)
                else:
                    report.failed_products.append({"product_code": active_code, "error": result.error})
                    logger.error("apply_product_failed code=%s error=%s", active_code, result.error)
                active_code = None
            # Stage 5 can use report.saved_changes HERE, under this same lock.
            # No recommendation lookup or backend delivery is performed in stage 4.
            session.assert_lock()
            report.status = ("partial_failure" if report.failed_products or report.scan_error_count
                             or report.missing_count else "completed")
    except CommitUnknown as error:
        report.error = error.code
        report.unknown_products.append({"product_code": active_code, "error": error.code})
        logger.error("apply_product_unknown code=%s error=%s", active_code, error.code)
    except SessionLost as error:
        report.error = error.code
        if active_code is not None:
            report.failed_products.append({"product_code": active_code, "error": error.code})
        logger.error("apply_aborted code=%s product_code=%s", error.code, active_code)
    except Exception as error:
        report.error = error.code if isinstance(error, ApplyError) else "execution_failed"
        logger.error("apply_aborted code=%s", report.error)
    finally:
        try:
            if emit is not None:
                try:
                    emit(report.to_dict())
                except Exception:
                    report.status = "aborted"
                    report.error = "result_output_failed"
                    logger.error("apply_output_failed code=result_output_failed")
            logger.info(
                "apply_end run_id=%s time=%s duration_seconds=%.3f status=%s "
                "대상=%s 확인=%s 변경=%s 미확인=%s failed=%s unknown=%s skipped_same=%s "
                "missing=%s scan_errors=%s error=%s",
                report.run_id, now_kst(), time.monotonic() - started, report.status,
                report.target_count, report.confirmed_count, len(report.saved_changes),
                report.unconfirmed_count, len(report.failed_products), len(report.unknown_products),
                report.skipped_same_count, report.missing_count, report.scan_error_count, report.error,
            )
        finally:
            session.close()
    return report
