"""app/config/categories.py의 요청/저장 카테고리 설정 검증."""

import unittest
from app.config.categories import CATEGORIES, STORAGE_MAIN_CATEGORY_MAP

class CategoriesConfigTests(unittest.TestCase):
    def test_still_requests_four_main_categories(self):
        main_categories = {category["main_category"] for category in CATEGORIES}
        self.assertEqual(main_categories, {"상의", "하의", "아우터", "니트웨어"})

    def test_storage_main_category_map_normalizes_outer_and_knit_to_top(self):
        self.assertEqual(STORAGE_MAIN_CATEGORY_MAP, {
            "상의": "상의",
            "하의": "하의",
            "아우터": "상의",
            "니트웨어": "상의",
        })


if __name__ == "__main__":
    unittest.main()
