from app.config.categories import CATEGORIES
from app.config.quota import MAIN_CATEGORY_QUOTAS, SUB_CATEGORY_BASE_QUOTAS
from app.config.settings import (
    CRAWL_PAGE_SIZE,
    CRAWL_REQUEST_DELAY,
    CRAWL_CHECKPOINT_PATH,
    CRAWL_TIMEOUT,
    MAX_PRODUCTS,
    OUTPUT_PATH,
)
from app.crawler.client import TwentyNineCmClient
from app.crawler.product_crawler import crawl_products_with_quota
from app.storage.json_storage import save_products


def main() -> None:
    print("Crawl started")
    print(f"categories={len(CATEGORIES)}")
    print(f"main_category_quotas={MAIN_CATEGORY_QUOTAS}")
    print(f"sub_category_base_quotas={SUB_CATEGORY_BASE_QUOTAS}")
    print(f"total_quota={sum(MAIN_CATEGORY_QUOTAS.values())}")
    print(f"page_size={CRAWL_PAGE_SIZE}")

    client = TwentyNineCmClient(timeout=CRAWL_TIMEOUT)

    def _persist(batch) -> None:
        # 체크포인트가 페이지를 "처리 완료"로 기록하기 전에 즉시 저장해서,
        # 장시간 수집 도중 중단돼도 이미 모은 상품이 유실되지 않게 한다.
        save_products(batch, OUTPUT_PATH, max_products=MAX_PRODUCTS)

    result = crawl_products_with_quota(
        CATEGORIES,
        client=client,
        main_category_quotas=MAIN_CATEGORY_QUOTAS,
        sub_category_base_quotas=SUB_CATEGORY_BASE_QUOTAS,
        page_size=CRAWL_PAGE_SIZE,
        request_delay=CRAWL_REQUEST_DELAY,
        checkpoint_path=CRAWL_CHECKPOINT_PATH,
        on_products_collected=_persist,
    )

    print(f"Collected unique products: {len(result.products)}")
    print(f"main_category_counts={result.main_category_counts}")
    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
