import re

_MULTI_COLOR_PATTERN = re.compile(
    r"(?<![0-9A-Za-z가-힣])(\d+)\s*(?:colors?|컬러)(?![0-9A-Za-z가-힣])",
    re.IGNORECASE,
)
_MULTI_COLOR_MIN_COUNT = 2


def is_multi_color_product(product_name: str) -> bool:
    """상품명이 2가지 이상 색상 옵션을 나타내는지 반환한다."""
    if not product_name:
        return False
    return any(
        int(match.group(1)) >= _MULTI_COLOR_MIN_COUNT
        for match in _MULTI_COLOR_PATTERN.finditer(product_name)
    )
