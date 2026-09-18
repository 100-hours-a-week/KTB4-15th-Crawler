"""크롤러 실행 설정 테스트."""

import os
import unittest
from importlib import reload

from app.config import settings


class SettingsTests(unittest.TestCase):
    def tearDown(self):
        for name in (
            "CRAWL_MAX_PAGE",
            "CRAWL_PAGE_SIZE",
            "CRAWL_TIMEOUT",
            "CRAWL_REQUEST_DELAY",
            "OUTPUT_PATH",
            "CRAWL_CHECKPOINT_PATH",
            "MAX_PRODUCTS",
        ):
            os.environ.pop(name, None)
        reload(settings)

    def test_default_settings(self):
        reload(settings)
        self.assertEqual(settings.CRAWL_MAX_PAGE, 2)
        self.assertEqual(settings.CRAWL_PAGE_SIZE, 50)
        self.assertEqual(settings.CRAWL_TIMEOUT, 15)
        self.assertEqual(settings.CRAWL_REQUEST_DELAY, 0.5)
        self.assertEqual(settings.OUTPUT_PATH, "data/products.json")
        self.assertIsNone(settings.CRAWL_CHECKPOINT_PATH)
        self.assertEqual(settings.MAX_PRODUCTS, 50000)

    def test_environment_overrides(self):
        os.environ.update({
            "CRAWL_MAX_PAGE": "3",
            "CRAWL_PAGE_SIZE": "25",
            "CRAWL_TIMEOUT": "8",
            "CRAWL_REQUEST_DELAY": "0",
            "OUTPUT_PATH": "tmp/products.json",
            "CRAWL_CHECKPOINT_PATH": "tmp/crawl-checkpoint.json",
            "MAX_PRODUCTS": "1000",
        })
        reload(settings)
        self.assertEqual(settings.CRAWL_MAX_PAGE, 3)
        self.assertEqual(settings.CRAWL_PAGE_SIZE, 25)
        self.assertEqual(settings.CRAWL_TIMEOUT, 8)
        self.assertEqual(settings.CRAWL_REQUEST_DELAY, 0)
        self.assertEqual(settings.OUTPUT_PATH, "tmp/products.json")
        self.assertEqual(
            settings.CRAWL_CHECKPOINT_PATH, "tmp/crawl-checkpoint.json"
        )
        self.assertEqual(settings.MAX_PRODUCTS, 1000)

    def test_invalid_settings_fail_early(self):
        os.environ["CRAWL_MAX_PAGE"] = "zero"
        with self.assertRaises(ValueError):
            reload(settings)
        os.environ.pop("CRAWL_MAX_PAGE")
