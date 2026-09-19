"""요청 main_category 기준 quota 수집(crawl_products_with_quota) 테스트."""

import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from app.crawler.client import MOST_REVIEWED, RECOMMENDED
from app.crawler.product_crawler import crawl_products_with_quota


CRAWLED_AT = datetime(2026, 9, 18, 12, 0, tzinfo=timezone(timedelta(hours=9)))


def _make_item(code: int, *, product_name: str = None) -> dict:
    return {
        "itemId": code,
        "itemInfo": {
            "productName": product_name or f"Item {code}",
            "thumbnailUrl": "https://img.29cm.co.kr/item/example.jpg",
            "displayPrice": 10000,
            "isSoldOut": False,
        },
        "itemUrl": {"webLink": f"https://product.29cm.co.kr/catalog/{code}"},
    }


class QuotaFakeClient:
    """smallId별로 정해진 개수만큼만 상품을 내어주는 가짜 client.

    정렬(sort)과 무관하게 같은 smallId는 같은 상품 집합을 반환해서
    실제 API처럼 두 정렬이 같은 카탈로그를 다른 순서로 보여주는 상황을
    흉내 낸다. product_code는 smallId마다 겹치지 않게 만든다.
    """

    def __init__(self, capacities: dict, *, product_name_by_index=None):
        self.capacities = capacities
        self.product_name_by_index = product_name_by_index or {}
        self.calls = []

    def fetch_listing(self, *, largeId, middleId, smallId, sort, page, size):
        self.calls.append((smallId, sort, page))
        capacity = self.capacities.get(smallId, 0)
        start = (page - 1) * size
        if start >= capacity:
            return {"data": {"list": []}}
        end = min(capacity, start + size)
        base_code = int(smallId) * 1_000_000
        items = [
            _make_item(
                base_code + i,
                product_name=self.product_name_by_index.get((smallId, i)),
            )
            for i in range(start, end)
        ]
        return {"data": {"list": items}}


def _outer_category(sub_category: str, small_id: str) -> dict:
    return {
        "main_category": "아우터",
        "sub_category": sub_category,
        "largeId": "272100100",
        "middleId": "272102100",
        "smallId": small_id,
    }


class QuotaCrawlerTests(unittest.TestCase):
    def test_base_quota_then_redistribution_fills_main_quota(self):
        categories = [
            _outer_category("서브A", "900001"),
            _outer_category("서브B", "900002"),
            _outer_category("서브C", "900003"),
        ]
        capacities = {"900001": 1, "900002": 1, "900003": 20}
        client = QuotaFakeClient(capacities)

        result = crawl_products_with_quota(
            categories,
            client=client,
            main_category_quotas={"아우터": 10},
            sub_category_base_quotas={"아우터": 2},
            crawled_at=CRAWLED_AT,
            page_size=2,
        )

        self.assertEqual(result.main_category_counts, {"아우터": 10})
        self.assertEqual(len(result.products), 10)

        by_sub = {}
        for product in result.products:
            by_sub.setdefault(product.sub_category, 0)
            by_sub[product.sub_category] += 1
        self.assertEqual(by_sub, {"서브A": 1, "서브B": 1, "서브C": 8})

        # 저장값은 상의로 정규화되고 색상은 항상 None이어야 한다.
        self.assertTrue(all(p.main_category == "상의" for p in result.products))
        self.assertTrue(all(p.color is None for p in result.products))

    def test_exhausted_sub_category_does_not_block_others_or_hang(self):
        categories = [
            _outer_category("서브A", "900001"),
            _outer_category("서브B", "900002"),
        ]
        # 두 sub 모두 base quota보다 재고가 적어 금방 소진된다.
        capacities = {"900001": 1, "900002": 1}
        client = QuotaFakeClient(capacities)

        result = crawl_products_with_quota(
            categories,
            client=client,
            main_category_quotas={"아우터": 100},
            sub_category_base_quotas={"아우터": 5},
            crawled_at=CRAWLED_AT,
            page_size=2,
        )

        # 재고가 없으므로 main quota(100)를 못 채워도 무한 요청 없이 끝나야 한다.
        self.assertEqual(result.main_category_counts, {"아우터": 2})
        self.assertEqual(len(result.products), 2)

    def test_duplicate_product_code_across_sub_categories_counted_once(self):
        categories = [
            _outer_category("서브A", "900001"),
            _outer_category("서브B", "900001"),
        ]
        capacities = {"900001": 3}
        client = QuotaFakeClient(capacities)

        result = crawl_products_with_quota(
            categories,
            client=client,
            main_category_quotas={"아우터": 100},
            sub_category_base_quotas={"아우터": 3},
            crawled_at=CRAWLED_AT,
            page_size=5,
        )

        self.assertEqual(result.main_category_counts, {"아우터": 3})
        self.assertEqual(len(result.products), 3)

    def test_multi_color_products_are_excluded_from_quota(self):
        categories = [_outer_category("서브A", "900001")]
        client = QuotaFakeClient(
            {"900001": 3},
            product_name_by_index={("900001", 1): "2colors 자켓"},
        )

        result = crawl_products_with_quota(
            categories,
            client=client,
            main_category_quotas={"아우터": 100},
            sub_category_base_quotas={"아우터": 5},
            crawled_at=CRAWLED_AT,
            page_size=5,
        )

        self.assertEqual(result.main_category_counts, {"아우터": 2})
        self.assertEqual(len(result.products), 2)

    def test_multiple_main_categories_are_tracked_independently(self):
        categories = [
            _outer_category("서브A", "900001"),
            {
                "main_category": "니트웨어",
                "sub_category": "카디건",
                "largeId": "272100100",
                "middleId": "272110100",
                "smallId": "900101",
            },
        ]
        client = QuotaFakeClient({"900001": 10, "900101": 10})

        result = crawl_products_with_quota(
            categories,
            client=client,
            main_category_quotas={"아우터": 3, "니트웨어": 2},
            sub_category_base_quotas={"아우터": 3, "니트웨어": 2},
            crawled_at=CRAWLED_AT,
            page_size=5,
        )

        self.assertEqual(result.main_category_counts, {"아우터": 3, "니트웨어": 2})
        self.assertEqual(len(result.products), 5)
        self.assertTrue(all(p.main_category == "상의" for p in result.products))

    def test_checkpoint_resumes_quota_progress_after_failure(self):
        categories = [
            _outer_category("서브A", "900001"),
            _outer_category("서브B", "900002"),
        ]
        capacities = {"900001": 10, "900002": 10}

        class FailingClient(QuotaFakeClient):
            def __init__(self, capacities):
                super().__init__(capacities)
                self.failed = False

            def fetch_listing(self, **kwargs):
                if not self.failed and len(self.calls) == 2:
                    self.failed = True
                    raise RuntimeError("temporary failure")
                return super().fetch_listing(**kwargs)

        checkpoint = Path("quota-checkpoint-test.json")
        checkpoint.unlink(missing_ok=True)
        persisted: list = []

        def _persist(batch):
            persisted.extend(batch)

        try:
            with self.assertRaises(RuntimeError):
                crawl_products_with_quota(
                    categories,
                    client=FailingClient(capacities),
                    main_category_quotas={"아우터": 6},
                    sub_category_base_quotas={"아우터": 3},
                    crawled_at=CRAWLED_AT,
                    page_size=2,
                    checkpoint_path=checkpoint,
                    on_products_collected=_persist,
                )
            self.assertTrue(checkpoint.exists())
            saved_state = json.loads(checkpoint.read_text(encoding="utf-8"))
            self.assertIn("main_counts", saved_state)
            self.assertIn("sub_states", saved_state)
            # 실패 전에 완료된 sub_category(서브A)만큼은 이미 콜백으로
            # 저장돼 있어야 한다 — 체크포인트만 믿고 있으면 유실된다.
            self.assertEqual(len(persisted), 3)

            result = crawl_products_with_quota(
                categories,
                client=QuotaFakeClient(capacities),
                main_category_quotas={"아우터": 6},
                sub_category_base_quotas={"아우터": 3},
                crawled_at=CRAWLED_AT,
                page_size=2,
                checkpoint_path=checkpoint,
                on_products_collected=_persist,
            )
            self.assertEqual(result.main_category_counts, {"아우터": 6})
            # 재실행 호출의 반환값에는 서브A가 이미 완료된 상태라 skip돼서
            # 새로 모은 서브B 몫(3개)만 담긴다.
            self.assertEqual(len(result.products), 3)
            # 콜백을 통해 누적된 전체(서브A+서브B)는 6개, 중복 없이 모두 저장된다.
            self.assertEqual(len(persisted), 6)
            self.assertEqual(len({p.product_code for p in persisted}), 6)
            self.assertFalse(checkpoint.exists())
        finally:
            checkpoint.unlink(missing_ok=True)

    def test_invalid_quota_is_rejected(self):
        with self.assertRaises(ValueError):
            crawl_products_with_quota(
                [_outer_category("서브A", "900001")],
                client=QuotaFakeClient({"900001": 1}),
                main_category_quotas={"아우터": 0},
                sub_category_base_quotas={"아우터": 1},
            )

    def test_real_quota_config_sums_to_fifty_thousand(self):
        from app.config.quota import MAIN_CATEGORY_QUOTAS, TOTAL_QUOTA

        self.assertEqual(TOTAL_QUOTA, 50_000)
        self.assertEqual(sum(MAIN_CATEGORY_QUOTAS.values()), 50_000)
        self.assertEqual(
            set(MAIN_CATEGORY_QUOTAS), {"상의", "하의", "아우터", "니트웨어"}
        )


if __name__ == "__main__":
    unittest.main()
