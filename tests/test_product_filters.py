import unittest
from app.crawler.product_filters import is_multi_color_product


class ProductFilterTests(unittest.TestCase):
    def test_multi_color_names_are_excluded(self):
        for name in (
            "2colors 티셔츠",
            "2 colors 티셔츠",
            "2color 티셔츠",
            "2 color 티셔츠",
            "2컬러 티셔츠",
            "2 컬러 티셔츠",
            "3colors 니트",
            "3 colors 데님 팬츠",
            "3color 데님 팬츠",
            "3 color 데님 팬츠",
            "3컬러 데님 팬츠",
            "3 컬러 데님 팬츠",
            "4 COLORS 셔츠",
            "10 colors 팬츠",
            "12컬러 자켓",
        ):
            with self.subTest(name=name):
                self.assertTrue(is_multi_color_product(name))

    def test_numeric_names_without_color_marker_are_kept(self):
        for name in (
            "501 데님 팬츠",
            "4포켓 팬츠",
            "3버튼 코트",
            "2WAY 블루종",
            "COLOR BLOCK 티셔츠",
        ):
            with self.subTest(name=name):
                self.assertFalse(is_multi_color_product(name))

    def test_single_color_count_is_kept(self):
        for name in ("1color 티셔츠", "1 컬러 티셔츠"):
            with self.subTest(name=name):
                self.assertFalse(is_multi_color_product(name))

    def test_underscore_or_bracket_separated_multi_color_is_excluded(self):
        for name in (
            "LUN DYEING SWEATSHIRT_2COLORS",
            "Knoll sleeveless (Unisex)_dyed_2 Colors",
            "Soft Layer Leggings_4color-VN26S004",
            "[기본핏 OR +5CM 선택] 크리스피 나일론 팬츠 쇼츠_11COLOR",
            "AUTUMN COLOR CURVED PANTS_4COLORS",
            "TOLLO CURVED PANTS_2COLORS",
            "Baka string pants (Unisex)_2 Colors",
            "셀비 포켓 데님 팬츠_9COLORS",
            "[BEST] Diagonal jogger pants (Unisex)_18 Colors",
            "남녀공용 보더라인 세미 와이드 루즈 원턱 트랙팬츠_5colors",
            "뮤렌 슬랙스_2COLORS",
            "워시드 다잉 코튼 팬츠_5Color",
            "[PF]레거시 플리스 하프집업_FS3FR02U_3COLOR",
            "에센셜 무드 헨리넥 니트_8Color",
            "[PF]웨이파인더 코위찬 베스트 집업_FS4NV01U_2COLOR",
        ):
            with self.subTest(name=name):
                self.assertTrue(is_multi_color_product(name))

    def test_digit_embedded_in_model_code_is_not_excluded(self):
        for name in (
            "TG3-SH2101colors 버튼다운 셔츠",
            "ABC12컬러디자인",
        ):
            with self.subTest(name=name):
                self.assertFalse(is_multi_color_product(name))
