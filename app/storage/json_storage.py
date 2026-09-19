"""상품을 JSON으로 저장한다."""

import json
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, Union

from app.crawler.product_filters import is_multi_color_product
from app.models.product import Product


DEFAULT_MAX_PRODUCTS = 50_000


def product_to_dict(product: Product) -> dict:
    """Convert a Product to the representation used in the JSON file."""
    data = asdict(product)
    data["crawled_at"] = product.crawled_at.isoformat()
    return data


def save_products(
    products: Iterable[Product],
    path: Union[str, Path],
    *,
    max_products: int = DEFAULT_MAX_PRODUCTS,
) -> None:

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    stored: dict[int, dict] = {}
    if output.exists():
        try:
            existing = json.loads(output.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"기존 JSON을 읽을 수 없습니다: {output}") from error
        if not isinstance(existing, list):
            raise ValueError("기존 상품 JSON은 배열이어야 합니다")
        for data in existing:
            if not isinstance(data, dict):
                raise ValueError("기존 상품 JSON 상품이 객체가 아닙니다")
            product_code = data.get("product_code", data.get("source_product_id"))
            if type(product_code) is not int:
                raise ValueError(
                    "기존 상품 JSON의 product_code/source_product_id가 올바르지 않습니다"
                )
            if "product_code" not in data:
                data = dict(data)
                data["product_code"] = product_code
                data.pop("source_product_id", None)
            stored.setdefault(product_code, data)
    for product in products:
        if len(stored) >= max_products:
            break
        if is_multi_color_product(product.product_name):
            continue
        if product.product_code not in stored:
            stored[product.product_code] = product_to_dict(product)
    output.write_text(
        json.dumps(
            list(stored.values()),
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
