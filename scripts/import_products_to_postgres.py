import json
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.models.product import Product  # noqa: E402
from app.storage.postgres_storage import get_connection, save_products  # noqa: E402

DEFAULT_JSON_PATH = PROJECT_ROOT / "data" / "products.json"
DEFAULT_BATCH_SIZE = 1000

_REQUIRED_STR_FIELDS = (
    "product_name",
    "detail_url",
    "image_url",
    "main_category",
    "sub_category",
)


def _row_to_product(row: dict, *, index: int) -> Product:
    """JSON 한 row를 Product로 변환한다. 문제가 있으면 위치를 포함해 실패한다."""
    if not isinstance(row, dict):
        raise ValueError(f"row[{index}]: 객체가 아닙니다")

    product_code = row.get("product_code")
    if type(product_code) is not int or product_code <= 0:
        raise ValueError(
            f"row[{index}]: product_code가 올바르지 않습니다: {product_code!r}"
        )

    for field in _REQUIRED_STR_FIELDS:
        value = row.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                f"row[{index}] product_code={product_code}: "
                f"{field}가 올바르지 않습니다: {value!r}"
            )

    price = row.get("price")
    if type(price) is not int or price < 0:
        raise ValueError(
            f"row[{index}] product_code={product_code}: "
            f"price가 올바르지 않습니다: {price!r}"
        )

    is_sold_out = row.get("is_sold_out")
    if type(is_sold_out) is not bool:
        raise ValueError(
            f"row[{index}] product_code={product_code}: "
            f"is_sold_out이 올바르지 않습니다: {is_sold_out!r}"
        )

    color = row.get("color")
    if color is not None and not isinstance(color, str):
        raise ValueError(
            f"row[{index}] product_code={product_code}: "
            f"color가 올바르지 않습니다: {color!r}"
        )

    crawled_at_raw = row.get("crawled_at")
    if not isinstance(crawled_at_raw, str):
        raise ValueError(
            f"row[{index}] product_code={product_code}: "
            f"crawled_at이 올바르지 않습니다: {crawled_at_raw!r}"
        )
    try:
        crawled_at = datetime.fromisoformat(crawled_at_raw)
    except ValueError as error:
        raise ValueError(
            f"row[{index}] product_code={product_code}: "
            f"crawled_at 파싱 실패: {crawled_at_raw!r}"
        ) from error
    if crawled_at.utcoffset() is None:
        raise ValueError(
            f"row[{index}] product_code={product_code}: "
            "crawled_at에 시간대 정보가 없습니다"
        )

    return Product(
        product_code=product_code,
        product_name=row["product_name"],
        detail_url=row["detail_url"],
        image_url=row["image_url"],
        price=price,
        is_sold_out=is_sold_out,
        main_category=row["main_category"],
        sub_category=row["sub_category"],
        color=color,
        crawled_at=crawled_at,
    )


def load_products(path: Path) -> list[Product]:
    """JSON 배열을 읽어 검증된 Product 목록으로 변환한다."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"{path}는 JSON 배열이어야 합니다")
    return [_row_to_product(row, index=index) for index, row in enumerate(raw)]


def _count_products(connection) -> int:
    with connection.cursor() as cursor:
        cursor.execute("SELECT COUNT(*) FROM products")
        return cursor.fetchone()[0]

def main(
    json_path: Path = DEFAULT_JSON_PATH, batch_size: int = DEFAULT_BATCH_SIZE
) -> None:
    print(f"JSON 경로: {json_path}")
    products = load_products(json_path)
    unique_codes = {product.product_code for product in products}
    print(f"JSON 전체 row: {len(products)}")
    print(f"JSON unique product_code: {len(unique_codes)}")
    if len(unique_codes) != len(products):
        print(
            "경고: JSON 안에 중복 product_code가 있습니다 "
            "(먼저 나온 row만 반영될 수 있습니다)"
        )

    batch_count = -(-len(products) // batch_size) if products else 0
    print(f"batch_size={batch_size}, 예상 batch 수={batch_count}")

    connection = get_connection()
    try:
        before_count = _count_products(connection)
        print(f"적재 전 DB row 수: {before_count}")

        inserted = save_products(products, batch_size=batch_size, connection=connection)

        after_count = _count_products(connection)
    finally:
        connection.close()

    skipped = len(products) - inserted
    print(f"처리한 row: {len(products)}")
    print(f"실제 insert된 row: {inserted}")
    print(f"conflict/skipped row: {skipped}")
    print(f"최종 DB row 수: {after_count}")

if __name__ == "__main__":
    main()
