"""Sequential alternating scans with category-local bounds and global maxima."""

import logging
import time
from dataclasses import dataclass, field
from typing import Callable, Iterable, Mapping, Protocol

from .parser import ListingResponseError, display_price, listing_items, product_code
from .settings import Settings
from .source.client import ListingRequestError, MOST_REVIEWED, RECOMMENDED

logger = logging.getLogger(__name__)
SORTS = (RECOMMENDED, MOST_REVIEWED)


class ListingClient(Protocol):
    def fetch_listing(self, *, largeId: str, middleId: str, smallId: str,
                      sort: str, page: int, size: int) -> Mapping: ...


@dataclass
class CategoryResult:
    main_category: str
    sub_category: str
    small_id: str
    seen_unique_count: int = 0
    page_requests: int = 0
    stop_reason: str = ""
    sort_status: dict = field(default_factory=dict)
    errors: list = field(default_factory=list)


@dataclass
class ScanResult:
    prices: dict[str, int] = field(default_factory=dict)
    categories: list[CategoryResult] = field(default_factory=list)


def collect_prices(client: ListingClient, categories: Iterable[Mapping],
                   settings: Settings, *, sleep: Callable[[float], None] = time.sleep
                   ) -> ScanResult:
    result = ScanResult()
    for category in categories:
        report = CategoryResult(category["main_category"], category["sub_category"],
                                category["smallId"])
        result.categories.append(report)
        seen: set[str] = set()
        pages = {sort: 1 for sort in SORTS}
        report.sort_status = {sort: "active" for sort in SORTS}
        logger.info("category_start small_id=%s category=%s/%s", report.small_id,
                    report.main_category, report.sub_category)
        while (len(seen) < settings.max_unique
               and report.page_requests < settings.max_page_requests
               and "active" in report.sort_status.values()):
            for sort in SORTS:
                if (len(seen) >= settings.max_unique
                        or report.page_requests >= settings.max_page_requests):
                    break
                if report.sort_status[sort] != "active":
                    continue
                page = pages[sort]
                report.page_requests += 1  # failed requests also consume the cap
                try:
                    try:
                        payload = client.fetch_listing(
                            largeId=category["largeId"], middleId=category["middleId"],
                            smallId=category["smallId"], sort=sort, page=page,
                            size=settings.page_size,
                        )
                    finally:
                        # Also pace HTTP/transport failures; never retry automatically.
                        sleep(settings.request_delay)
                    items = listing_items(payload)
                except (ListingRequestError, ListingResponseError, OSError) as error:
                    report.sort_status[sort] = "error"
                    report.errors.append({"sort": sort, "page": page, "message": str(error)})
                    logger.error("lookup_error small_id=%s sort=%s page=%s error=%s",
                                 report.small_id, sort, page, error)
                    continue
                pages[sort] += 1
                if not items:
                    report.sort_status[sort] = "exhausted"
                    logger.info("sort_exhausted small_id=%s sort=%s page=%s",
                                report.small_id, sort, page)
                    continue
                invalid = 0
                for item in items:
                    code = product_code(item)
                    if code is None:
                        invalid += 1
                        continue
                    seen.add(code)  # valid identifiers count even without valid prices
                    price = display_price(item)
                    if price is None:
                        invalid += 1
                    else:
                        result.prices[code] = max(result.prices.get(code, price), price)
                    # Include the 500th code's price, then stop before any next item.
                    if len(seen) >= settings.max_unique:
                        break
                if invalid:
                    logger.warning("invalid_candidates small_id=%s sort=%s page=%s count=%s",
                                   report.small_id, sort, page, invalid)
        report.seen_unique_count = len(seen)
        if len(seen) >= settings.max_unique:
            report.stop_reason = "unique_limit"
        elif "active" not in report.sort_status.values():
            report.stop_reason = "lookup_error" if report.errors else "lists_exhausted"
        else:
            report.stop_reason = "page_limit"
        logger.info("category_end small_id=%s reason=%s unique=%s page_requests=%s errors=%s",
                    report.small_id, report.stop_reason, report.seen_unique_count,
                    report.page_requests, len(report.errors))
    return result
