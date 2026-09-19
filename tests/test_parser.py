"""고정 Fixture를 이용한 네트워크 없는 파서 테스트."""

import json
import unittest
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from app.crawler.parser import parse_product, parse_products
from app.models.product import Product

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "listing_response.json"
CRAWLED_AT = datetime(2026, 9, 16, 15, 30, tzinfo=timezone(timedelta(hours=9)))

class ParserTests(unittest.TestCase):
    def setUp(self):
        self.response = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
        self.item = self.response["data"]["list"][0]

    def test_product_fields_and_primary_source(self):
        product = parse_product(self.item, crawled_at=CRAWLED_AT)
        self.assertIsInstance(product, Product)
        self.assertEqual(asdict(product), {
            "product_code": 3263645,
            "product_name": "Jacquard Stripe Roll Up Half Shirts White",
            "detail_url": "https://product.29cm.co.kr/catalog/3263645",
            "image_url": "https://img.29cm.co.kr/item/example.jpg",
            "price": 31430,
            "is_sold_out": False,
            "main_category": "",
            "sub_category": "",
            "color": None,
            "crawled_at": CRAWLED_AT,
        })

    def test_color_is_always_none_regardless_of_colorchip_or_product_name(self):
        products = parse_products(self.response, crawled_at=CRAWLED_AT)
        self.assertTrue(all(p.color is None for p in products))
        self.assertTrue(products[2].is_sold_out)

    def test_each_required_field_missing(self):
        paths = (
            "itemId", "itemInfo.productName", "itemUrl.webLink",
            "itemInfo.thumbnailUrl", "itemInfo.displayPrice", "itemInfo.isSoldOut",
        )
        for path in paths:
            with self.subTest(path=path):
                item = deepcopy(self.item)
                parent = item
                parts = path.split(".")
                for key in parts[:-1]:
                    parent = parent[key]
                del parent[parts[-1]]
                with self.assertRaises(ValueError) as error:
                    parse_product(item)
                self.assertIn(path, str(error.exception))

    def test_invalid_required_types_are_rejected(self):
        for path, value in (
            ("itemId", True), ("itemId", "3263645"), ("itemId", None),
            ("itemInfo.displayPrice", "31430"), ("itemInfo.displayPrice", 31.43),
            ("itemInfo.isSoldOut", "false"), ("itemInfo.isSoldOut", 0),
            ("itemInfo.productName", None), ("itemInfo.productName", " "),
        ):
            with self.subTest(path=path, value=value):
                item = deepcopy(self.item)
                parent = item
                parts = path.split(".")
                for key in parts[:-1]:
                    parent = parent[key]
                parent[parts[-1]] = value
                with self.assertRaises(ValueError):
                    parse_product(item)

    def test_missing_required_parent_objects(self):
        for key in ("itemInfo", "itemUrl"):
            with self.subTest(key=key):
                item = deepcopy(self.item)
                item[key] = None
                with self.assertRaises(ValueError):
                    parse_product(item)

    def test_category_metadata_is_optional(self):
        item = deepcopy(self.item)
        del item["itemEvent"]
        product = parse_product(item)
        self.assertEqual(product.main_category, "")
        self.assertEqual(product.sub_category, "")

    def test_missing_small_category_does_not_fail(self):
        item = deepcopy(self.item)
        del item["itemEvent"]["eventProperties"]["smallCategoryName"]
        product = parse_product(item)
        self.assertEqual(product.product_code, 3263645)

    def test_zero_price_and_false_sold_out_are_preserved(self):
        self.item["itemInfo"]["displayPrice"] = 0
        product = parse_product(self.item)
        self.assertEqual(product.price, 0)
        self.assertIs(product.is_sold_out, False)

    def test_empty_listing(self):
        self.assertEqual(parse_products({"data": {"list": []}}), [])

    def test_invalid_response_shape(self):
        for response in (None, [], {}, {"data": None}, {"data": {}},
                         {"data": {"list": None}}, {"data": {"list": {}}}):
            with self.subTest(response=response):
                with self.assertRaises(ValueError):
                    parse_products(response)

    def test_invalid_item_has_list_index(self):
        for item in (None, [], {}):
            with self.subTest(item=item):
                with self.assertRaises(ValueError) as error:
                    parse_products({"data": {"list": [self.item, item]}})
                self.assertIn("data.list[1]", str(error.exception))

    def test_order_and_duplicates_are_preserved(self):
        items = self.response["data"]["list"]
        self.response["data"]["list"] = [items[2], items[0], items[2]]
        products = parse_products(self.response)
        self.assertEqual([p.product_code for p in products],
                         [3263647, 3263645, 3263647])

    def test_input_is_not_modified(self):
        original = deepcopy(self.response)
        parse_products(self.response)
        self.assertEqual(self.response, original)

    def test_generated_crawled_at_is_current_and_timezone_aware(self):
        before = datetime.now().astimezone()
        product = parse_product(self.item)
        after = datetime.now().astimezone()
        actual = product.crawled_at
        self.assertIsNotNone(actual.utcoffset())
        self.assertLessEqual(before, actual)
        self.assertLessEqual(actual, after)

    def test_listing_shares_one_crawled_at(self):
        for timestamp in (None, CRAWLED_AT):
            with self.subTest(timestamp=timestamp):
                products = parse_products(self.response, crawled_at=timestamp)
                self.assertEqual(len({p.crawled_at for p in products}), 1)
                if timestamp is not None:
                    self.assertEqual(products[0].crawled_at, timestamp)

    def test_naive_crawled_at_is_rejected(self):
        for parser, value in ((parse_product, self.item), (parse_products, self.response)):
            with self.subTest(parser=parser.__name__):
                with self.assertRaises(ValueError):
                    parser(value, crawled_at=datetime(2026, 9, 16, 15, 30))


if __name__ == "__main__":
    unittest.main()
