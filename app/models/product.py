"""크롤링 설계 문서의 최종 상품 데이터 구조."""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass(frozen=True)
class Product:
    product_code: int
    product_name: str
    detail_url: str
    image_url: str
    price: int
    is_sold_out: bool
    main_category: str
    sub_category: str
    color: Optional[str]
    crawled_at: datetime
