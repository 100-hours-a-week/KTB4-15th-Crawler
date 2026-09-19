from typing import Any, Iterator, Optional, Sequence
from app.config.database import get_database_url
from app.models.product import Product

_INSERT_COLUMNS = (
    "product_code",
    "product_name",
    "price",
    "color",
    "detail_url",
    "image_url",
    "main_category",
    "sub_category",
    "is_sold_out",
    "crawled_at",
)

def get_connection() -> Any:
    import psycopg
    return psycopg.connect(get_database_url())


def product_to_row(product: Product) -> tuple:
    return (
        product.product_code,
        product.product_name,
        product.price,
        product.color,
        product.detail_url,
        product.image_url,
        product.main_category,
        product.sub_category,
        product.is_sold_out,
        product.crawled_at,
    )

def _batches(
    products: Sequence[Product], batch_size: int
) -> Iterator[Sequence[Product]]:
    for start in range(0, len(products), batch_size):
        yield products[start : start + batch_size]


def build_insert_sql(row_count: int) -> str:
    columns = ", ".join(_INSERT_COLUMNS)
    placeholder_row = "(" + ", ".join(["%s"] * len(_INSERT_COLUMNS)) + ")"
    values_clause = ", ".join([placeholder_row] * row_count)
    return (
        f"INSERT INTO products ({columns}) VALUES {values_clause} "
        "ON CONFLICT (product_code) DO NOTHING"
    )


def _insert_batch(cursor: Any, batch: Sequence[Product]) -> int:
    if not batch:
        return 0
    sql = build_insert_sql(len(batch))
    params = [value for product in batch for value in product_to_row(product)]
    cursor.execute(sql, params)
    return cursor.rowcount


def save_products(
    products: Sequence[Product],
    *,
    batch_size: int = 1000,
    connection: Optional[Any] = None,
) -> int:

    if batch_size < 1:
        raise ValueError("batch_size는 1 이상이어야 합니다")

    products = list(products)
    if not products:
        return 0

    owns_connection = connection is None
    conn = connection if connection is not None else get_connection()
    try:
        inserted_total = 0
        with conn.cursor() as cursor:
            for batch in _batches(products, batch_size):
                inserted_total += _insert_batch(cursor, batch)
        conn.commit()
        return inserted_total
    finally:
        if owns_connection:
            conn.close()
