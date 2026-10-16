"""Read, scan, compare, filter and preview; no writes or backend requests."""

import logging
from dataclasses import asdict

from .comparison import compare_prices
from .repository import RepositoryError, recommendation_rows

logger = logging.getLogger(__name__)


def run_dry_run(repository, collect):
    # Finish/close initial DB read before making any listing requests.
    stored = repository.fetch_prices()
    scan = collect()
    comparison = compare_prices(stored, scan.prices)
    changes = [asdict(change) for change in comparison.changes]
    if comparison.invalid_stored_rows:
        logger.warning("invalid_stored_rows count=%s", comparison.invalid_stored_rows)
    lookup = {"status": "skipped_no_changes", "error": None}
    targets = []
    if changes:
        codes = [change["product_code"] for change in changes]
        try:
            raw = repository.fetch_recommendations(codes)
            # Validate fake/adapted repositories too: missing/null is never FALSE.
            flags = recommendation_rows(raw.items(), codes)
            targets = [change for change in changes if flags[change["product_code"]] is True]
            lookup["status"] = "success"
        except Exception as error:
            code = error.code if isinstance(error, RepositoryError) else "recommendation_lookup_failed"
            lookup = {"status": "failed", "error": {"code": code}}
            targets = None
            logger.error("recommendation_lookup_failed code=%s", code)
    return {
        "mode": "dry_run",
        "all_changes": changes,
        "all_change_count": len(changes),
        "recommendation_lookup": lookup,
        "backend_targets": targets,
        "backend_target_count": len(targets) if targets is not None else None,
        "backend_request_preview": ({"changed_count": len(targets), "products": targets}
                                    if targets is not None else None),
        "invalid_stored_row_count": comparison.invalid_stored_rows,
        "scan": {"categories": [asdict(category) for category in scan.categories],
                 "has_errors": any(category.errors for category in scan.categories)},
    }
