"""29CM data.list 응답 파싱. 네트워크 요청, 후보 선정, 중복 제거, 저장은 하지 않는다."""

from collections.abc import Mapping
from datetime import datetime
from typing import Any, Optional
from app.models.product import Product


def _required(item: Mapping[str, Any], path: str, expected_type: type) -> Any:
    value: Any = item
    for key in path.split("."):
        if not isinstance(value, Mapping) or key not in value:
            raise ValueError(f"필수 필드 누락: {path}")
        value = value[key]
    if type(value) is not expected_type:
        raise ValueError(f"필드 타입 오류: {path} (필요한 타입: {expected_type.__name__})")
    if expected_type is str and not value.strip():
        raise ValueError(f"필수 필드가 빈 문자열: {path}")
    return value


def _crawl_timestamp(crawled_at: Optional[datetime]) -> datetime:
    timestamp = crawled_at if crawled_at is not None else datetime.now().astimezone()
    if not isinstance(timestamp, datetime) or timestamp.utcoffset() is None:
        raise ValueError("crawled_at은 시간대가 포함된 datetime이어야 합니다")
    return timestamp


def parse_product(
    item: Mapping[str, Any],
    *,
    crawled_at: Optional[datetime] = None,
    main_category: str = "",
    sub_category: str = "",
) -> Product:
    """상품 하나를 변환한다. 카테고리는 호출자가 확정한다.

    색상은 컬러칩/상품명에서 추출하지 않고 항상 None으로 둔다 — 색상은
    추후 별도의 이미지 특징 키워드 추출 단계에서 채운다.
    """
    if not isinstance(item, Mapping):
        raise ValueError("상품은 JSON 객체여야 합니다")
    return Product(
        product_code=_required(item, "itemId", int),
        product_name=_required(item, "itemInfo.productName", str),
        detail_url=_required(item, "itemUrl.webLink", str),
        image_url=_required(item, "itemInfo.thumbnailUrl", str),
        price=_required(item, "itemInfo.displayPrice", int),
        is_sold_out=_required(item, "itemInfo.isSoldOut", bool),
        main_category=main_category,
        sub_category=sub_category,
        color=None,
        crawled_at=_crawl_timestamp(crawled_at),
    )


def parse_products(
    response: Mapping[str, Any],
    *,
    crawled_at: Optional[datetime] = None,
    main_category: str = "",
    sub_category: str = "",
) -> list[Product]:
    """data.list를 순서대로 변환한다. 한 응답의 모든 상품에 같은 시각을 사용한다."""
    if not isinstance(response, Mapping):
        raise ValueError("응답은 JSON 객체여야 합니다")
    items = _required(response, "data.list", list)
    timestamp = _crawl_timestamp(crawled_at)
    products = []
    for index, item in enumerate(items):
        try:
            products.append(
                parse_product(
                    item,
                    crawled_at=timestamp,
                    main_category=main_category,
                    sub_category=sub_category,
                )
            )
        except ValueError as error:
            raise ValueError(f"data.list[{index}]: {error}") from error
    return products
