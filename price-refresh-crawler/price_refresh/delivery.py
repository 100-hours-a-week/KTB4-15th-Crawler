"""Sequential first delivery + ONE retry of explicitly retryable product codes."""

import logging
from dataclasses import asdict, dataclass, field

from .backend import BackendError, classify_response
from .comparison import code_order, normalize_code, valid_price
from .repository import RepositoryError, recommendation_rows
from .writer import SessionLost

logger = logging.getLogger(__name__)


@dataclass
class DeliveryReport:
    status: str = "not_started"
    recommendation_lookup: str = "not_started"
    skip_reason: object = None
    error: object = None
    target_count: object = None
    initial_request_count: int = 0
    retry_request_count: int = 0
    retry_product_count: int = 0
    initial_success_codes: list = field(default_factory=list)
    retry_success_codes: list = field(default_factory=list)
    not_found_codes: list = field(default_factory=list)
    failed_codes: list = field(default_factory=list)
    unknown_codes: list = field(default_factory=list)
    unsent_codes: list = field(default_factory=list)
    request_errors: list = field(default_factory=list)

    def to_dict(self):
        result = asdict(self)
        result.update(initial_success_count=len(self.initial_success_codes),
                      retry_success_count=len(self.retry_success_codes),
                      successful_count=len(self.initial_success_codes) + len(self.retry_success_codes),
                      not_found_count=len(self.not_found_codes), failed_count=len(self.failed_codes),
                      unknown_count=len(self.unknown_codes), unsent_count=len(self.unsent_codes))
        return result


class BackendDelivery:
    def __init__(self, client):
        self.client = client

    def deliver(self, session, saved_changes):
        report = DeliveryReport()
        states, initial_success, retry_success = {}, set(), set()
        try:
            changes = {}
            for change in saved_changes:
                code = change.product_code
                if (not isinstance(code, str) or normalize_code(code) != code
                        or valid_price(change.old_price) is None or valid_price(change.new_price) is None
                        or (code in changes and changes[code] != change)):
                    raise BackendError("invalid_saved_changes")
                changes[code] = change
            if not changes:
                report.status = report.skip_reason = "skipped_no_saved_changes"
                report.recommendation_lookup = "skipped_no_saved_changes"
                report.target_count = 0
                return report
            codes = sorted(changes, key=code_order)
            session.assert_delivery_ready()
            try:
                flags = recommendation_rows(session.fetch_recommendations(codes).items(), codes)
                report.recommendation_lookup = "success"
            except SessionLost:
                raise
            except Exception as error:
                report.recommendation_lookup = "failed"
                report.status = "recommendation_lookup_failed"
                report.skip_reason = "recommendation_lookup_failed"
                report.error = (error.code if isinstance(error, RepositoryError)
                                else "recommendation_lookup_failed")
                logger.error("backend_recommendation_failed code=%s", report.error)
                return report
            targets = [code for code in codes if flags[code] is True]
            report.target_count = len(targets)
            states = {code: "unsent" for code in targets}
            if not targets:
                report.status = report.skip_reason = "skipped_no_targets"
                return report
            retryable = set()

            def send_batch(batch, *, retry):
                # Every transaction has ended, but the session lock must still exist.
                session.assert_delivery_ready()
                if retry:
                    report.retry_request_count += 1
                    report.retry_product_count += len(batch)
                else:
                    report.initial_request_count += 1
                # Once a call starts, its outcome is unknown until validated.
                for code in batch:
                    states[code] = "unknown"
                status, phase, error_code = None, None, None
                try:
                    response = self.client.send([asdict(changes[code]) for code in batch])
                    status = response.status
                    outcome = classify_response(response, batch)
                    for state, values in (("successful", outcome.successful), ("failed", outcome.failed),
                                          ("not_found", outcome.not_found), ("unknown", outcome.unknown)):
                        for code in values:
                            states[code] = state
                    (retry_success if retry else initial_success).update(outcome.successful)
                    if not retry:
                        retryable.update(outcome.failed)
                    error_code = outcome.error
                except SessionLost:
                    raise
                except Exception as error:
                    error_code = error.code if isinstance(error, BackendError) else "backend_transport_failed"
                    if isinstance(error, BackendError):
                        status, phase = error.http_status, error.phase
                if error_code:
                    report.request_errors.append({"attempt": "retry" if retry else "initial",
                                                  "http_status": status, "phase": phase,
                                                  "error": error_code, "product_codes": list(batch)})
                    logger.error("backend_request_failed attempt=%s http_status=%s phase=%s code=%s products=%s",
                                 "retry" if retry else "initial", status, phase, error_code, len(batch))
                logger.info("backend_batch_end attempt=%s http_status=%s products=%s successful=%s "
                            "failed=%s not_found=%s unknown=%s",
                            "retry" if retry else "initial", status, len(batch),
                            sum(states[code] == "successful" for code in batch),
                            sum(states[code] == "failed" for code in batch),
                            sum(states[code] == "not_found" for code in batch),
                            sum(states[code] == "unknown" for code in batch))
                session.assert_delivery_ready()

            size = self.client.settings.batch_size
            for start in range(0, len(targets), size):
                send_batch(targets[start:start + size], retry=False)
            # Only one retry phase. Original committed old/new values are reused.
            retry_codes = sorted(retryable, key=code_order)
            for start in range(0, len(retry_codes), size):
                send_batch(retry_codes[start:start + size], retry=True)
            report.status = ("partial_failure" if any(state != "successful" for state in states.values())
                             else "completed")
        except SessionLost as error:
            report.status = "aborted"
            report.error = error.code
            logger.error("backend_aborted code=%s", error.code)
        except Exception as error:
            report.status = "partial_failure"
            report.error = error.code if isinstance(error, BackendError) else "backend_delivery_failed"
            logger.error("backend_delivery_failed code=%s", report.error)
        finally:
            report.initial_success_codes = sorted(initial_success, key=code_order)
            report.retry_success_codes = sorted(retry_success, key=code_order)
            for state, attribute in (("failed", "failed_codes"), ("not_found", "not_found_codes"),
                                     ("unknown", "unknown_codes"), ("unsent", "unsent_codes")):
                setattr(report, attribute, sorted((code for code, value in states.items() if value == state),
                                                 key=code_order))
            counts = report.to_dict()
            logger.info("backend_end status=%s targets=%s initial_requests=%s initial_success=%s "
                        "retry_requests=%s retry_products=%s retry_success=%s successful=%s "
                        "not_found=%s failed=%s unknown=%s unsent=%s recommendation_lookup=%s "
                        "skip_reason=%s error=%s", report.status, report.target_count,
                        report.initial_request_count, counts["initial_success_count"], report.retry_request_count,
                        report.retry_product_count, counts["retry_success_count"], counts["successful_count"],
                        counts["not_found_count"], counts["failed_count"], counts["unknown_count"],
                        counts["unsent_count"], report.recommendation_lookup, report.skip_reason, report.error)
        return report
