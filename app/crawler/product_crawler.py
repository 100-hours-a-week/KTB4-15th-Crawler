"""카테고리별 상품 수집 순서를 조정한다."""

import logging
import json
import time
from collections.abc import Iterable, Mapping
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, NamedTuple, Optional, Union

from app.config.categories import STORAGE_MAIN_CATEGORY_MAP
from app.crawler.client import MOST_REVIEWED, RECOMMENDED, TwentyNineCmClient
from app.crawler.parser import parse_products
from app.crawler.product_filters import is_multi_color_product
from app.models.product import Product


logger = logging.getLogger(__name__)

MALE_LARGE_ID = "272100100"
MAIN_CATEGORY_MIDDLE_IDS = {
    "상의": "272103100",
    "하의": "272104100",
    "아우터": "272102100",
    "니트웨어": "272110100",
}


def _category_codes(category: Mapping[str, Any]) -> dict[str, str]:
    """검증된 카테고리 설정을 API body 필드 그대로 추출한다."""
    try:
        main_category = str(category["main_category"])
        large_id = str(category["largeId"])
        middle_id = str(category["middleId"])
        small_id = str(category["smallId"])
        expected_middle_id = MAIN_CATEGORY_MIDDLE_IDS.get(main_category)
        if large_id != MALE_LARGE_ID:
            raise ValueError(
                f"남성 카테고리의 largeId는 {MALE_LARGE_ID}이어야 합니다: {large_id}"
            )
        if expected_middle_id is None:
            raise ValueError(f"지원하지 않는 main_category입니다: {main_category}")
        if middle_id != expected_middle_id:
            raise ValueError(
                f"{main_category}의 middleId는 {expected_middle_id}이어야 합니다: {middle_id}"
            )
        return {
            "largeId": large_id,
            "middleId": middle_id,
            "smallId": small_id,
        }
    except (KeyError, TypeError) as error:
        raise ValueError(
            "카테고리에 main_category, largeId, middleId, smallId가 필요합니다"
        ) from error


def _find_item(response: Mapping[str, Any], product_code: int) -> Mapping[str, Any]:
    """응답에서 상품 원본을 찾아 category metadata 검증에 사용한다."""
    data = response.get("data")
    items = data.get("list") if isinstance(data, Mapping) else None
    if isinstance(items, list):
        for item in items:
            if isinstance(item, Mapping) and item.get("itemId") == product_code:
                return item
    return {}


def _log_category_mismatch(
    item: Mapping[str, Any],
    product_code: int,
    requested_main_category: str,
    requested_sub_category: str,
    sort: str,
) -> None:
    """응답 category metadata는 검증/로그에만 사용하고 상품은 유지한다."""
    event = item.get("itemEvent") if isinstance(item, Mapping) else None
    properties = event.get("eventProperties") if isinstance(event, Mapping) else None
    properties = properties if isinstance(properties, Mapping) else {}
    response_main = properties.get("middleCategoryName")
    response_sub = properties.get("smallCategoryName")
    main_mismatch = response_main != requested_main_category
    sub_mismatch = response_sub != requested_sub_category
    if main_mismatch or sub_mismatch:
        logger.warning(
            "29CM category metadata mismatch: product_code=%s sort=%s "
            "requested_main=%r requested_sub=%r response_main=%r "
            "response_sub=%r main_mismatch=%s sub_mismatch=%s",
            product_code,
            sort,
            requested_main_category,
            requested_sub_category,
            response_main,
            response_sub,
            main_mismatch,
            sub_mismatch,
        )


class QuotaCrawlResult(NamedTuple):
    """quota 기반 수집 결과.

    ``main_category_counts``는 저장용으로 정규화되기 전, 요청 기준
    main_category별 현재까지 채택된 고유 상품 수를 담는다.
    """

    products: list[Product]
    main_category_counts: dict[str, int]


def _fresh_sub_category_state() -> dict[str, Any]:
    return {
        "collected": 0,
        "sorts": [{"page": 1, "exhausted": False}, {"page": 1, "exhausted": False}],
    }


def _sub_category_fully_exhausted(state: Mapping[str, Any]) -> bool:
    """두 정렬 모두 더 이상 반환할 상품이 없는지 확인한다."""
    return all(sort_state["exhausted"] for sort_state in state["sorts"])

def _load_quota_checkpoint(
    path: Optional[Path],
    category_count: int,
    main_category_quotas: Mapping[str, int],
) -> tuple[dict[str, int], dict[int, dict[str, Any]]]:
    if path is None or not path.exists():
        main_counts = {main: 0 for main in main_category_quotas}
        sub_states = {
            index: _fresh_sub_category_state() for index in range(category_count)
        }
        return main_counts, sub_states
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        main_counts = {
            main: int(raw["main_counts"].get(main, 0)) for main in main_category_quotas
        }
        sub_states: dict[int, dict[str, Any]] = {}
        for index_str, state in raw.get("sub_states", {}).items():
            sub_states[int(index_str)] = {
                "collected": int(state["collected"]),
                "sorts": [
                    {
                        "page": int(sort_state["page"]),
                        "exhausted": bool(sort_state["exhausted"]),
                    }
                    for sort_state in state["sorts"]
                ],
            }
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as error:
        raise ValueError(f"quota 체크포인트를 읽을 수 없습니다: {path}") from error
    for index in range(category_count):
        sub_states.setdefault(index, _fresh_sub_category_state())
    return main_counts, sub_states


def _save_quota_checkpoint(
    path: Path,
    main_counts: Mapping[str, int],
    sub_states: Mapping[int, dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "main_counts": dict(main_counts),
        "sub_states": {str(index): state for index, state in sub_states.items()},
    }
    path.write_text(
        json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def crawl_products_with_quota(
    categories: Iterable[Mapping[str, Any]],
    *,
    client: TwentyNineCmClient,
    main_category_quotas: Mapping[str, int],
    sub_category_base_quotas: Mapping[str, int],
    crawled_at: Optional[datetime] = None,
    page_size: int = 50,
    request_delay: float = 0.0,
    checkpoint_path: Optional[Union[str, Path]] = None,
    on_products_collected: Optional[Callable[[list[Product]], None]] = None,
) -> QuotaCrawlResult:
    """카테고리별 quota를 준수하며 상품을 수집한다."""
    if page_size < 1:
        raise ValueError("page_size는 1 이상이어야 합니다")
    if request_delay < 0:
        raise ValueError("request_delay는 0 이상이어야 합니다")
    for main, quota in main_category_quotas.items():
        if quota < 1:
            raise ValueError(f"{main}의 quota는 1 이상이어야 합니다")

    timestamp = crawled_at if crawled_at is not None else datetime.now().astimezone()
    if timestamp.utcoffset() is None:
        raise ValueError("crawled_at은 시간대가 포함된 datetime이어야 합니다")

    category_list = list(categories)
    checkpoint = Path(checkpoint_path) if checkpoint_path is not None else None
    main_counts, sub_states = _load_quota_checkpoint(
        checkpoint, len(category_list), main_category_quotas
    )

    unique_products: dict[int, Product] = {}
    excluded_multi_color = 0

    subs_by_main: dict[str, list[int]] = {}
    for index, category in enumerate(category_list):
        subs_by_main.setdefault(str(category["main_category"]), []).append(index)

    def fill(index: int, target: int) -> None:
        nonlocal excluded_multi_color
        category = category_list[index]
        requested_main = str(category["main_category"])
        requested_sub = str(category["sub_category"])
        main_quota = main_category_quotas[requested_main]
        codes = _category_codes(category)
        state = sub_states[index]
        for sort_index, sort in enumerate((RECOMMENDED, MOST_REVIEWED)):
            if state["collected"] >= target or main_counts[requested_main] >= main_quota:
                break
            sort_state = state["sorts"][sort_index]
            while not sort_state["exhausted"]:
                if (
                    state["collected"] >= target
                    or main_counts[requested_main] >= main_quota
                ):
                    break
                response = client.fetch_listing(
                    **codes, sort=sort, page=sort_state["page"], size=page_size,
                )
                if request_delay:
                    time.sleep(request_delay)
                items = response.get("data", {}).get("list", [])
                if not items:
                    sort_state["exhausted"] = True
                else:
                    products = parse_products(response, crawled_at=timestamp)
                    newly_collected: list[Product] = []
                    for product in products:
                        if (
                            state["collected"] >= target
                            or main_counts[requested_main] >= main_quota
                        ):
                            break
                        if is_multi_color_product(product.product_name):
                            excluded_multi_color += 1
                            continue
                        if product.product_code in unique_products:
                            continue
                        raw_item = _find_item(response, product.product_code)
                        _log_category_mismatch(
                            raw_item,
                            product.product_code,
                            requested_main,
                            requested_sub,
                            sort,
                        )
                        stored_main = STORAGE_MAIN_CATEGORY_MAP[requested_main]
                        stored_product = replace(
                            product,
                            main_category=stored_main,
                            sub_category=requested_sub,
                        )
                        unique_products[product.product_code] = stored_product
                        newly_collected.append(stored_product)
                        state["collected"] += 1
                        main_counts[requested_main] += 1
                    sort_state["page"] += 1
                    # 체크포인트가 이 페이지를 처리했다고 기록하기 전에
                    # 새로 채택된 상품을 먼저 영속화해서 크래시로 인한
                    # 데이터 유실을 막는다.
                    if on_products_collected is not None and newly_collected:
                        on_products_collected(newly_collected)
                if checkpoint is not None:
                    _save_quota_checkpoint(checkpoint, main_counts, sub_states)

    for main, indices in subs_by_main.items():
        if main not in main_category_quotas:
            continue
        base_quota = sub_category_base_quotas.get(main, main_category_quotas[main])
        main_quota = main_category_quotas[main]

        # 1차: sub_category별 기본 quota까지 채운다.
        for index in indices:
            if main_counts[main] >= main_quota:
                break
            fill(index, target=base_quota)

        # 2차: main_category quota가 남으면 아직 소진되지 않은 sub_category에 부족분을 재분배한다.
        for index in indices:
            if main_counts[main] >= main_quota:
                break
            if _sub_category_fully_exhausted(sub_states[index]):
                continue
            remaining_target = (
                main_quota - main_counts[main] + sub_states[index]["collected"]
            )
            fill(index, target=remaining_target)

    fully_settled = all(
        main_counts[main] >= main_category_quotas[main]
        or all(_sub_category_fully_exhausted(sub_states[index]) for index in indices)
        for main, indices in subs_by_main.items()
        if main in main_category_quotas
    )
    if checkpoint is not None and fully_settled:
        checkpoint.unlink(missing_ok=True)

    if excluded_multi_color:
        logger.info("다중 색상 상품 제외(quota 수집): count=%s", excluded_multi_color)

    return QuotaCrawlResult(
        products=list(unique_products.values()),
        main_category_counts=dict(main_counts),
    )
