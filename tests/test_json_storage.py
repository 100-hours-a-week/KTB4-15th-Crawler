"""JSON 저장 경계에서 datetime을 문자열로 변환하는지 검증한다."""

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from app.crawler.parser import parse_product
from app.storage.json_storage import product_to_dict, save_products

from test_parser import CRAWLED_AT, FIXTURE_PATH


class JsonStorageTests(unittest.TestCase):
    def test_append_only_preserves_existing_and_first_new_product(self):
        item = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))["data"]["list"][0]
        existing = parse_product(item, crawled_at=CRAWLED_AT)
        changed = replace(existing, price=existing.price + 100, product_name="changed")
        new = replace(existing, product_code=existing.product_code + 1)
        changed_new = replace(new, price=new.price + 200)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "products.json"
            save_products([existing], path)
            save_products([changed, new, changed_new], path)
            expected = [product_to_dict(existing), product_to_dict(new)]
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), expected)
            save_products([changed, new], path)
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), expected)

    def test_storage_excludes_multi_color_new_products(self):
        item = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))["data"]["list"][0]
        product = parse_product(item, crawled_at=CRAWLED_AT)
        names = ("3colors", "3 colors", "3color", "3 color", "3컬러", "3 컬러", "4 COLORS")
        excluded = [replace(product, product_code=product.product_code + i + 1,
                            product_name=f"티셔츠 {name}")
                    for i, name in enumerate(names)]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "products.json"
            save_products([product], path)
            save_products(excluded, path)
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")),
                             [product_to_dict(product)])

    def test_crawled_at_is_iso_string_in_json(self):
        item = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))["data"]["list"][0]
        product = parse_product(item, crawled_at=CRAWLED_AT)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "products.json"
            save_products([product], path)
            stored = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(stored[0]["crawled_at"], CRAWLED_AT.isoformat())
        self.assertIsInstance(stored[0]["crawled_at"], str)

    def test_max_products_caps_total_unique_products_across_runs(self):
        item = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))["data"]["list"][0]
        base = parse_product(item, crawled_at=CRAWLED_AT)
        existing = [replace(base, product_code=base.product_code + i) for i in range(2)]
        new = [replace(base, product_code=base.product_code + 100 + i) for i in range(2)]

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "products.json"
            save_products(existing, path, max_products=3)
            save_products(new, path, max_products=3)
            stored = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(len(stored), 3)

    def test_max_products_blocks_new_products_once_cap_is_reached(self):
        item = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))["data"]["list"][0]
        base = parse_product(item, crawled_at=CRAWLED_AT)
        existing = [replace(base, product_code=base.product_code + i) for i in range(3)]
        new = [replace(base, product_code=base.product_code + 100)]

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "products.json"
            save_products(existing, path, max_products=3)
            save_products(new, path, max_products=3)
            stored = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(len(stored), 3)
        self.assertEqual(
            {row["product_code"] for row in stored},
            {base.product_code, base.product_code + 1, base.product_code + 2},
        )

    def test_creates_parent_directory_for_output(self):
        item = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))["data"]["list"][0]
        product = parse_product(item, crawled_at=CRAWLED_AT)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "products.json"
            save_products([product], path)
            self.assertTrue(path.exists())
