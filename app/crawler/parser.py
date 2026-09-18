"""29CM data.list 응답 파싱. 네트워크 요청, 후보 선정, 중복 제거, 저장은 하지 않는다."""

import re
from collections.abc import Mapping
from datetime import datetime
from typing import Any, Optional

from app.models.product import Product


# 상품명에서 독립된 색상 토큰만 인정한다. 영문은 한국어 서비스 값으로 변환한다.
_COLOR_NAMES = {
    "off white": "오프화이트",
    "미드블루": "미드블루",
    "mid blue": "미드블루",
    "라이트블루": "라이트블루",
    "light blue": "라이트블루",
    "다크블루": "다크블루",
    "dark blue": "다크블루",
    "딥네이비": "딥네이비",
    "deep navy": "딥네이비",
    "다크네이비": "다크네이비",
    "dark navy": "다크네이비",
    "멜란지그레이": "멜란지그레이",
    "melange gray": "멜란지그레이",
    "melange grey": "멜란지그레이",
    "인디고": "인디고",
    "indigo": "인디고",
    "나이트인디고": "나이트인디고",
    "night indigo": "나이트인디고",
    "라이트인디고": "라이트인디고",
    "light indigo": "라이트인디고",
    "클리어스카이": "클리어스카이",
    "연청": "연청",
    "중청": "중청",
    "빈티지블루": "빈티지블루",
    "vintage blue": "빈티지블루",
    "charcoal": "차콜",
    "bluish charcoal": "블루시 차콜",
    "chocolate brown": "초콜릿 브라운",
    "white": "화이트",
    "black": "블랙",
    "navy": "네이비",
    "ivory": "아이보리",
    "beige": "베이지",
    "gray": "그레이",
    "grey": "그레이",
    "brown": "브라운",
    "blue": "블루",
    "green": "그린",
    "red": "레드",
    "pink": "핑크",
    "yellow": "옐로우",
    "purple": "퍼플",
    "khaki": "카키",
}
_COLOR_NAMES.update({name: name for name in tuple(_COLOR_NAMES.values())})
_COLOR_TOKEN_PATTERN = re.compile(
    r"(?<![가-힣A-Za-z])("
    + "|".join(
        re.escape(name)
        for name in sorted(_COLOR_NAMES, key=len, reverse=True)
    )
    + r")(?![가-힣A-Za-z])",
    re.IGNORECASE,
)
_COLOR_SUFFIX_PATTERN = re.compile(
    r"(?:^|[^가-힣A-Za-z])("
    + "|".join(
        re.escape(name)
        for name in sorted(_COLOR_NAMES, key=len, reverse=True)
    )
    + r")(?=[^가-힣A-Za-z]*$)",
    re.IGNORECASE,
)


def _required(item: Mapping[str, Any], path: str, expected_type: type) -> Any:
    value: Any = item
    for key in path.split("."):
        if not isinstance(value, Mapping) or key not in value:
            raise ValueError(f"필수 필드 누락: {path}")
        value = value[key]
    # bool은 int의 하위 타입이므로 타입을 정확하게 검사한다.
    if type(value) is not expected_type:
        raise ValueError(f"필드 타입 오류: {path} (필요한 타입: {expected_type.__name__})")
    if expected_type is str and not value.strip():
        raise ValueError(f"필수 필드가 빈 문자열: {path}")
    return value


def _optional_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _extract_suffix_color(product_name: str) -> Optional[str]:
    match = _COLOR_SUFFIX_PATTERN.search(product_name)
    if not match:
        return None

    prefix = product_name[:match.start()].rstrip(" \t-_([/")
    previous = list(_COLOR_TOKEN_PATTERN.finditer(prefix))
    if previous and previous[-1].end() == len(prefix):
        previous_value = _COLOR_NAMES[previous[-1].group(1).lower()]
        current_value = _COLOR_NAMES[match.group(1).lower()]
        if previous_value != current_value:
            return None
    return _COLOR_NAMES[match.group(1).lower()]


def extract_color(item: Mapping[str, Any]) -> Optional[str]:
    """색상칩을 우선하고, 없으면 상품명 전체에서 보수적으로 색상을 찾는다."""
    info = _optional_mapping(item.get("itemInfo"))
    event = _optional_mapping(item.get("itemEvent"))
    properties = _optional_mapping(event.get("eventProperties"))
    if properties.get("isColorchip") is True:
        group = _optional_mapping(info.get("itemGroup"))
        colors = group.get("colors")
        if isinstance(colors, list) and colors:
            name = _optional_mapping(colors[0]).get("name")
            if isinstance(name, str) and name.strip():
                return name.strip()

    product_name = info.get("productName")
    if isinstance(product_name, str):
        suffix_color = _extract_suffix_color(product_name)
        if suffix_color is not None:
            return suffix_color

        matches = list(_COLOR_TOKEN_PATTERN.finditer(product_name))
        candidates = []
        for match in matches:
            key = match.group(1).lower()
            value = _COLOR_NAMES[key]
            if not any(
                value == existing
                or (
                    match.start() >= other.start()
                    and match.end() <= other.end()
                )
                for existing, other in candidates
            ):
                candidates.append((value, match))
        # A composite match such as 미드블루 subsumes its basic 색상 match.
        candidates = [
            (value, match)
            for value, match in candidates
            if not any(
                value != other_value
                and match.start() >= other_match.start()
                and match.end() <= other_match.end()
                for other_value, other_match in candidates
            )
        ]
        distinct = {value for value, _ in candidates}
        if len(distinct) == 1:
            return next(iter(distinct))
    return None


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
    """상품 하나를 변환한다. 카테고리는 호출자가 확정한다."""
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
        color=extract_color(item),
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
