import os


def _positive_int(name: str, default: str) -> int:
    raw = os.getenv(name, default)
    try:
        value = int(raw)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name}은 정수여야 합니다: {raw!r}") from error
    if value < 1:
        raise ValueError(f"{name}은 1 이상이어야 합니다: {value}")
    return value


def _non_negative_float(name: str, default: str) -> float:
    raw = os.getenv(name, default)
    try:
        value = float(raw)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name}은 숫자여야 합니다: {raw!r}") from error
    if value < 0:
        raise ValueError(f"{name}은 0 이상이어야 합니다: {value}")
    return value


CRAWL_MAX_PAGE = _positive_int("CRAWL_MAX_PAGE", "2")
CRAWL_PAGE_SIZE = _positive_int("CRAWL_PAGE_SIZE", "50")
CRAWL_TIMEOUT = _positive_int("CRAWL_TIMEOUT", "15")
CRAWL_REQUEST_DELAY = _non_negative_float("CRAWL_REQUEST_DELAY", "0.5")
OUTPUT_PATH = os.getenv("OUTPUT_PATH", "data/products.json")
CRAWL_CHECKPOINT_PATH = os.getenv("CRAWL_CHECKPOINT_PATH") or None
# 기준 데이터셋(products.json)의 최대 고유 product_code 개수.
MAX_PRODUCTS = _positive_int("MAX_PRODUCTS", "50000")
