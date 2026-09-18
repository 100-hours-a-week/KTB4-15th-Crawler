"""Step 5~7의 HTTP 요청, 수집 조정, JSON 병합 테스트."""

import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from app.crawler.client import MOST_REVIEWED, RECOMMENDED, TwentyNineCmClient
from app.crawler.product_crawler import _category_codes, crawl_products
from app.crawler.parser import parse_product
from test_parser import CRAWLED_AT, FIXTURE_PATH


class _Response:
    def __init__(self, payload, status=200, content_type="application/json"):
        self.payload = json.dumps(payload).encode("utf-8")
        self.status = status
        self.headers = {"Content-Type": content_type}

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return None

    def read(self):
        return self.payload

    def getcode(self):
        return self.status


class Step5To7Tests(unittest.TestCase):
    def setUp(self):
        self.response = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))

    def test_client_builds_post_json_request(self):
        requests = []

        def opener(request, timeout):
            requests.append((request, timeout))
            return _Response(self.response)

        client = TwentyNineCmClient(
            endpoint="https://example.test/api/items",
            query_params={"extra": "value"},
            headers={"X-Test": "test"},
            timeout=3,
            opener=opener,
        )
        result = client.fetch_listing(
            largeId="272100100",
            middleId="272103100",
            smallId="272103108",
            sort=RECOMMENDED,
        )

        request = requests[0][0]
        query = parse_qs(urlparse(request.full_url).query)
        body = json.loads(request.data.decode("utf-8"))
        self.assertEqual(query["colorchipVariant"], ["treatment"])
        self.assertEqual(query["extra"], ["value"])
        self.assertEqual(request.method, "POST")
        self.assertEqual(body["sortType"], RECOMMENDED)
        self.assertEqual(body["pageRequest"], {"page": 1, "size": 50})
        self.assertEqual(
            body["facets"]["categoryFacetInputs"][0],
            {"largeId": 272100100, "middleId": 272103100, "smallId": 272103108},
        )
        self.assertEqual(request.headers["X-test"], "test")
        self.assertEqual(requests[0][1], 3)
        self.assertEqual(result, self.response)

    def test_crawler_requests_both_sorts_and_deduplicates_candidates(self):
        calls = []
        response = self.response

        class FakeClient:
            def fetch_listing(self, **kwargs):
                calls.append(kwargs["sort"])
                return response

        products = crawl_products(
            [{
                "main_category": "상의",
                "sub_category": "반소매 셔츠",
                "largeId": "272100100",
                "middleId": "272103100",
                "smallId": "272103105",
            }],
            client=FakeClient(),
            crawled_at=CRAWLED_AT,
            max_page=1,
        )

        self.assertEqual(calls, [RECOMMENDED, MOST_REVIEWED])
        self.assertEqual(len(products), 5)
        self.assertEqual({product.crawled_at for product in products}, {CRAWLED_AT})
        self.assertEqual({product.main_category for product in products}, {"상의"})
        self.assertEqual({product.sub_category for product in products}, {"반소매 셔츠"})

    def test_crawler_fetches_two_pages_for_each_sort_and_uses_all_items(self):
        calls = []
        first = self.response
        second = deepcopy(self.response)
        second["data"]["list"] = second["data"]["list"][:2]

        class FakeClient:
            def fetch_listing(self, **kwargs):
                calls.append((kwargs["sort"], kwargs["page"], kwargs["size"]))
                return first if kwargs["page"] == 1 else second

        products = crawl_products(
            [{
                "main_category": "상의",
                "sub_category": "반소매 셔츠",
                "largeId": "272100100",
                "middleId": "272103100",
                "smallId": "272103105",
            }],
            client=FakeClient(),
            crawled_at=CRAWLED_AT,
            max_page=2,
            page_size=50,
            request_delay=0,
        )

        self.assertEqual(
            calls,
            [
                (RECOMMENDED, 1, 50),
                (RECOMMENDED, 2, 50),
                (MOST_REVIEWED, 1, 50),
                (MOST_REVIEWED, 2, 50),
            ],
        )
        self.assertEqual(len(products), 5)

    def test_crawler_resumes_from_checkpoint_after_request_failure(self):
        calls = []
        checkpoint = Path("checkpoint-resume-test.json")
        checkpoint.unlink(missing_ok=True)
        response = self.response

        class FailingClient:
            def fetch_listing(self, **kwargs):
                calls.append((kwargs["sort"], kwargs["page"]))
                if len(calls) == 2:
                    raise RuntimeError("temporary failure")
                return response

        category = [{
            "main_category": "상의",
            "sub_category": "반소매 셔츠",
            "largeId": "272100100",
            "middleId": "272103100",
            "smallId": "272103105",
        }]
        try:
            with self.assertRaises(RuntimeError):
                crawl_products(
                    category,
                    client=FailingClient(),
                    crawled_at=CRAWLED_AT,
                    max_page=2,
                    checkpoint_path=checkpoint,
                )
            saved_state = json.loads(checkpoint.read_text())
            self.assertEqual(
                {key: saved_state[key] for key in (
                    "category_index", "sort_index", "page"
                )},
                {"category_index": 0, "sort_index": 0, "page": 2},
            )
            class ResumingClient:
                def fetch_listing(self, **kwargs):
                    calls.append((kwargs["sort"], kwargs["page"]))
                    return response

            crawl_products(
                category,
                client=ResumingClient(),
                crawled_at=CRAWLED_AT,
                max_page=2,
                checkpoint_path=checkpoint,
            )
            self.assertEqual(calls[2:], [
                (RECOMMENDED, 2),
                (MOST_REVIEWED, 1),
                (MOST_REVIEWED, 2),
            ])
            self.assertFalse(checkpoint.exists())
        finally:
            checkpoint.unlink(missing_ok=True)

    def test_crawler_stops_a_sort_when_page_is_empty(self):
        calls = []
        response = self.response

        class FakeClient:
            def fetch_listing(self, **kwargs):
                calls.append((kwargs["sort"], kwargs["page"]))
                if kwargs["page"] == 2:
                    return {"data": {"list": []}}
                return response

        products = crawl_products(
            [{
                "main_category": "상의",
                "sub_category": "반소매 셔츠",
                "largeId": "272100100",
                "middleId": "272103100",
                "smallId": "272103105",
            }],
            client=FakeClient(),
            crawled_at=CRAWLED_AT,
            max_page=3,
            request_delay=0,
        )

        self.assertEqual(
            calls,
            [
                (RECOMMENDED, 1),
                (RECOMMENDED, 2),
                (MOST_REVIEWED, 1),
                (MOST_REVIEWED, 2),
            ],
        )
        self.assertEqual(len(products), 5)

    def test_candidate_limit_restricts_each_page_for_tests(self):
        response = self.response

        class FakeClient:
            def fetch_listing(self, **kwargs):
                return response

        products = crawl_products(
            [{
                "main_category": "상의",
                "sub_category": "반소매 셔츠",
                "largeId": "272100100",
                "middleId": "272103100",
                "smallId": "272103105",
            }],
            client=FakeClient(),
            crawled_at=CRAWLED_AT,
            max_page=1,
            candidate_limit=2,
        )

        self.assertEqual(len(products), 2)

    def test_crawler_excludes_multi_color_products(self):
        response = deepcopy(self.response)
        response["data"]["list"][0]["itemInfo"]["productName"] = "3 colors 데님 팬츠"

        class FakeClient:
            def fetch_listing(self, **kwargs):
                return response

        products = crawl_products(
            [{
                "main_category": "상의",
                "sub_category": "반소매 셔츠",
                "largeId": "272100100",
                "middleId": "272103100",
                "smallId": "272103105",
            }],
            client=FakeClient(),
            crawled_at=CRAWLED_AT,
            max_page=1,
        )

        self.assertNotIn(3263645, {product.product_code for product in products})

    def test_crawler_keeps_mismatch_and_uses_requested_category(self):
        response = deepcopy(self.response)
        properties = response["data"]["list"][0]["itemEvent"]["eventProperties"]
        properties["middleCategoryName"] = "여성의류"
        properties["smallCategoryName"] = "점프수트"

        class FakeClient:
            def fetch_listing(self, **kwargs):
                return response

        products = crawl_products(
            [{
                "main_category": "아우터",
                "sub_category": "플리스",
                "largeId": "272100100",
                "middleId": "272102100",
                "smallId": "272102101",
            }],
            client=FakeClient(),
            crawled_at=CRAWLED_AT,
        )

        product = next(
            product for product in products if product.product_code == 3263645
        )
        self.assertEqual(product.main_category, "상의")
        self.assertEqual(product.sub_category, "플리스")

    def test_crawler_keeps_item_without_category_metadata(self):
        response = deepcopy(self.response)
        del response["data"]["list"][0]["itemEvent"]

        class FakeClient:
            def fetch_listing(self, **kwargs):
                return response

        products = crawl_products(
            [{
                "main_category": "아우터",
                "sub_category": "플리스",
                "largeId": "272100100",
                "middleId": "272102100",
                "smallId": "272102101",
            }],
            client=FakeClient(),
            crawled_at=CRAWLED_AT,
        )
        product = next(
            product for product in products if product.product_code == 3263645
        )
        self.assertEqual((product.main_category, product.sub_category),
                         ("상의", "플리스"))

    def test_category_codes_match_male_category_hierarchy(self):
        self.assertEqual(
            _category_codes({
                "main_category": "니트웨어",
                "sub_category": "기타 니트",
                "largeId": "272100100",
                "middleId": "272110100",
                "smallId": "272110109",
            }),
            {
                "largeId": "272100100",
                "middleId": "272110100",
                "smallId": "272110109",
            },
        )

    def test_category_codes_reject_mismatched_main_category_id(self):
        with self.assertRaises(ValueError):
            _category_codes({
                "main_category": "니트웨어",
                "sub_category": "기타 니트",
                "largeId": "272100100",
                "middleId": "272102100",
                "smallId": "272110109",
            })

    def test_stored_main_category_is_normalized_per_requested_category(self):
        response = self.response

        class FakeClient:
            def fetch_listing(self, **kwargs):
                return response

        cases = [
            ("상의", "반소매 티셔츠", "272103100", "272103101", "상의"),
            ("하의", "데님 팬츠", "272104100", "272104104", "하의"),
            ("아우터", "블루종", "272102100", "272102123", "상의"),
            ("니트웨어", "가디건", "272110100", "272110104", "상의"),
        ]
        for requested_main, sub_category, middle_id, small_id, expected_main in cases:
            with self.subTest(requested_main=requested_main):
                products = crawl_products(
                    [{
                        "main_category": requested_main,
                        "sub_category": sub_category,
                        "largeId": "272100100",
                        "middleId": middle_id,
                        "smallId": small_id,
                    }],
                    client=FakeClient(),
                    crawled_at=CRAWLED_AT,
                    max_page=1,
                )
                self.assertTrue(products)
                for product in products:
                    self.assertEqual(product.main_category, expected_main)
                    self.assertEqual(product.sub_category, sub_category)

    def test_stored_color_is_always_none_even_when_parser_extracts_one(self):
        response = self.response

        class FakeClient:
            def fetch_listing(self, **kwargs):
                return response

        products = crawl_products(
            [{
                "main_category": "상의",
                "sub_category": "반소매 티셔츠",
                "largeId": "272100100",
                "middleId": "272103100",
                "smallId": "272103101",
            }],
            client=FakeClient(),
            crawled_at=CRAWLED_AT,
            max_page=1,
        )

        self.assertTrue(products)
        self.assertTrue(all(product.color is None for product in products))

    def test_categories_config_still_requests_four_main_categories(self):
        from app.config.categories import CATEGORIES

        main_categories = {category["main_category"] for category in CATEGORIES}
        self.assertEqual(main_categories, {"상의", "하의", "아우터", "니트웨어"})

    def test_storage_main_category_map_normalizes_outer_and_knit_to_top(self):
        from app.config.categories import STORAGE_MAIN_CATEGORY_MAP

        self.assertEqual(STORAGE_MAIN_CATEGORY_MAP, {
            "상의": "상의",
            "하의": "하의",
            "아우터": "상의",
            "니트웨어": "상의",
        })

    def test_json_storage_merges_by_product_code(self):
        from app.storage.json_storage import product_to_dict, save_products

        products = [
            parse_product(item, crawled_at=CRAWLED_AT)
            for item in self.response["data"]["list"][:2]
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "products.json"
            existing = product_to_dict(products[0])
            existing["product_name"] = "old"
            path.write_text(json.dumps([existing]), encoding="utf-8")
            save_products(products, path)
            stored = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(len(stored), 2)
        self.assertEqual(stored[0], existing)
        self.assertEqual(stored[0]["crawled_at"], CRAWLED_AT.isoformat())

    def test_json_storage_recognizes_legacy_source_product_id(self):
        from app.storage.json_storage import save_products

        product = parse_product(self.response["data"]["list"][0], crawled_at=CRAWLED_AT)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "products.json"
            path.write_text(json.dumps([{
                "source_product_id": product.product_code,
                "product_name": "old",
            }]), encoding="utf-8")
            save_products([product], path)
            stored = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(len(stored), 1)
        self.assertEqual(stored[0]["product_code"], product.product_code)
