"""상품 이미지 검증 전용 PostgreSQL 접근.

`app/storage/postgres_storage.py`(크롤러 적재 경로, Python 3.9 호환 스타일)를 건드리지
않기 위해 별도 모듈로 뒀다. `get_connection()`은 그대로 재사용한다.

이 모듈의 함수는 모두 이미 열려 있는 connection 을 인자로 받는다. connection 의
생성/종료는 `scripts/validate_product_images.py`(호출하는 쪽)가 관리한다.
"""

from collections.abc import Sequence
from typing import Any

from app.models.product import Product

_PRODUCT_COLUMNS = (
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


def _row_to_product(row: Sequence[Any]) -> Product:
    (
        product_code,
        product_name,
        price,
        color,
        detail_url,
        image_url,
        main_category,
        sub_category,
        is_sold_out,
        crawled_at,
    ) = row
    return Product(
        product_code=product_code,
        product_name=product_name,
        detail_url=detail_url,
        image_url=image_url,
        price=price,
        is_sold_out=is_sold_out,
        main_category=main_category,
        sub_category=sub_category,
        color=color,
        crawled_at=crawled_at,
    )


def get_pending_products(connection: Any, *, limit: int | None = None) -> list[Product]:
    """validation_status = 'PENDING' 인 상품을 product_code 순으로 반환한다."""
    columns = ", ".join(_PRODUCT_COLUMNS)
    sql = (
        f"SELECT {columns} FROM products "
        "WHERE validation_status = 'PENDING' ORDER BY product_code"
    )
    params: tuple[Any, ...] = ()
    if limit is not None:
        sql += " LIMIT %s"
        params = (limit,)
    with connection.cursor() as cursor:
        cursor.execute(sql, params)
        rows = cursor.fetchall()
    return [_row_to_product(row) for row in rows]


def update_validation_result(
    connection: Any, product_code: int, status: str, reason: str | None
) -> None:
    """검증 결과 한 건을 기록하고 즉시 commit 한다.

    상품 수만 건을 순회하는 동안 하나의 트랜잭션에 묶어두면, 중간에 프로그램이
    죽었을 때 그동안의 진행이 전부 사라져 PENDING 이어서 재개하는 의미가
    없어진다. 그래서 한 건마다 commit 한다.
    """
    with connection.cursor() as cursor:
        cursor.execute(
            "UPDATE products SET validation_status = %s, validation_reason = %s "
            "WHERE product_code = %s",
            (status, reason, product_code),
        )
    connection.commit()


def get_validated_products(connection: Any) -> list[tuple[Product, str, str | None]]:
    """PASS/FAIL 로 판정된 상품 전체를 product_code 순으로 반환한다. CSV 재생성용이다."""
    columns = ", ".join(_PRODUCT_COLUMNS)
    sql = (
        f"SELECT {columns}, validation_status, validation_reason FROM products "
        "WHERE validation_status IN ('PASS', 'FAIL') ORDER BY product_code"
    )
    with connection.cursor() as cursor:
        cursor.execute(sql)
        rows = cursor.fetchall()
    return [(_row_to_product(row[:-2]), row[-2], row[-1]) for row in rows]


def count_by_validation_status(connection: Any) -> dict[str, int]:
    """PENDING/PASS/FAIL 상품 수를 반환한다. 없는 상태는 0으로 채운다."""
    with connection.cursor() as cursor:
        cursor.execute(
            "SELECT validation_status, COUNT(*) FROM products GROUP BY validation_status"
        )
        rows = cursor.fetchall()
    counts = {"PENDING": 0, "PASS": 0, "FAIL": 0}
    counts.update({status: count for status, count in rows})
    return counts


def delete_failed_products(connection: Any) -> int:
    """validation_status = 'FAIL' 인 상품을 삭제하고 삭제된 행 수를 반환한다."""
    with connection.cursor() as cursor:
        cursor.execute("DELETE FROM products WHERE validation_status = 'FAIL'")
        deleted = cursor.rowcount
    connection.commit()
    return deleted
