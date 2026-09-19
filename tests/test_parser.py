"""고정 Fixture를 이용한 네트워크 없는 파서 테스트."""

import json
import unittest
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

from app.crawler.parser import extract_color, parse_product, parse_products
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
            "color": "아이보리",
            "crawled_at": CRAWLED_AT,
        })

    def test_fixture_colors(self):
        products = parse_products(self.response, crawled_at=CRAWLED_AT)
        self.assertEqual([p.color for p in products],
                         ["아이보리", "화이트", "블랙", None, "네이비"])
        self.assertTrue(products[2].is_sold_out)

    def test_colorchip_uses_first_color_over_product_name(self):
        self.assertEqual(extract_color(self.item), "아이보리")
        self.item["itemInfo"]["itemGroup"]["colors"].reverse()
        self.assertEqual(extract_color(self.item), "블랙")

    def test_colorchip_wins_over_a_different_name_color(self):
        self.item["itemEvent"]["eventProperties"]["isColorchip"] = True
        self.item["itemInfo"]["itemGroup"]["colors"] = [{"name": "화이트"}]
        self.item["itemInfo"]["productName"] = "에센셜 블랙 데님"
        self.assertEqual(extract_color(self.item), "화이트")

    def test_no_colorchip_ignores_colors(self):
        self.item["itemEvent"]["eventProperties"]["isColorchip"] = False
        self.item["itemInfo"]["productName"] = "Half Shirts"
        self.assertIsNone(extract_color(self.item))

    def test_missing_colors_keep_null_without_name_suffix(self):
        self.item["itemInfo"]["productName"] = "Half Shirts"
        for group in (None, {}, {"colors": None}, {"colors": []},
                      {"colors": [{}]}, {"colors": [{"name": None}]},
                      {"colors": [{"name": " "}]}, {"colors": [None]},
                      {"colors": {"name": "블랙"}}):
            with self.subTest(group=group):
                self.item["itemInfo"]["itemGroup"] = group
                self.assertIsNone(extract_color(self.item))
        del self.item["itemInfo"]["itemGroup"]
        self.assertIsNone(extract_color(self.item))

    def test_missing_colors_fall_back_to_navy(self):
        self.item["itemInfo"]["productName"] = "Half Shirts Navy"
        del self.item["itemInfo"]["itemGroup"]
        self.assertEqual(extract_color(self.item), "네이비")

    def test_only_first_chip_is_used_before_name_fallback(self):
        self.item["itemInfo"]["itemGroup"]["colors"][0] = {}
        self.assertEqual(extract_color(self.item), "화이트")

    def test_clear_color_suffixes(self):
        self.item["itemEvent"]["eventProperties"]["isColorchip"] = False
        for suffix, expected in (
            ("White", "화이트"), ("Black", "블랙"), ("Navy", "네이비"),
            ("white", "화이트"), ("BLACK", "블랙"), ("[Navy]", "네이비"),
            ("(White)", "화이트"), ("블랙", "블랙"), ("Off White", "오프화이트"),
        ):
            with self.subTest(suffix=suffix):
                self.item["itemInfo"]["productName"] = f"Half Shirts {suffix} "
                self.assertEqual(extract_color(self.item), expected)

    def test_color_tokens_are_found_in_the_full_product_name(self):
        self.item["itemEvent"]["eventProperties"]["isColorchip"] = False
        for name, expected in (
            ("[CK] 슬림 스트레이트핏 미드블루 스트레치 데님 4RB902G R81", "미드블루"),
            ("[CK] 슬림핏 에센셜 블랙 데님 4RB738G 846", "블랙"),
            ("에센셜 3S 라이프스타일 우븐 쇼츠 - 블랙 / JE1309", "블랙"),
            ("ESSENTIAL NAVY PANTS", "네이비"),
            ("Half Shirts Black", "블랙"),
        ):
            with self.subTest(name=name):
                self.item["itemInfo"]["productName"] = name
                self.assertEqual(extract_color(self.item), expected)

    def test_color_name_false_positive_and_multiple_colors_are_rejected(self):
        self.item["itemEvent"]["eventProperties"]["isColorchip"] = False
        for name in ("블랙야크 남성 자켓", "블랙 화이트 배색 티셔츠",
                     "에센셜 스트레이트 데님 팬츠"):
            with self.subTest(name=name):
                self.item["itemInfo"]["productName"] = name
                self.assertIsNone(extract_color(self.item))

        self.item["itemInfo"]["productName"] = "2001 SLOW WORKER DENIM WASH JACKET [BLACK INDIGO]"
        self.assertIsNone(extract_color(self.item))

    def test_product_name_suffix_wins_over_another_middle_color(self):
        self.item["itemEvent"]["eventProperties"]["isColorchip"] = False
        self.item["itemInfo"]["productName"] = (
            "TG3-SH2101 인디고 버튼다운 셔츠 - 중청"
        )
        self.assertEqual(extract_color(self.item), "중청")

    def test_missing_optional_colorchip_flag(self):
        del self.item["itemEvent"]["eventProperties"]["isColorchip"]
        self.assertEqual(parse_product(self.item).color, "화이트")

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
