"""category → detection label 매핑과 동의어 bounding box 중복 제거를 검증한다.

Grounding DINO 모델 자체는 로드하지 않는다(torch/transformers 없이도 통과한다).
"""

import unittest

from app.config.categories import CATEGORIES
from app.validation.detector import (
    MAIN_CATEGORY_DETECTION_LABELS,
    SUB_CATEGORY_DETECTION_LABELS,
    Detection,
    box_area,
    check_cropped_lower_body,
    deduplicate_by_iou,
    get_detection_labels,
    intersection_area,
)


class SubCategoryMappingCoverageTests(unittest.TestCase):
    """app/config/categories.py 에 실제로 존재하는 sub_category 만 기준으로 한다."""

    def test_every_configured_sub_category_has_detection_labels(self):
        configured = {category["sub_category"] for category in CATEGORIES}
        missing = configured - SUB_CATEGORY_DETECTION_LABELS.keys()
        self.assertEqual(missing, set())

    def test_every_configured_main_category_has_a_fallback(self):
        # main_category 는 저장 시 STORAGE_MAIN_CATEGORY_MAP 을 거쳐 "상의"/"하의" 로만
        # 저장된다.
        self.assertEqual(set(MAIN_CATEGORY_DETECTION_LABELS), {"상의", "하의"})

    def test_labels_are_non_empty_lists_of_strings(self):
        for sub_category, labels in SUB_CATEGORY_DETECTION_LABELS.items():
            with self.subTest(sub_category=sub_category):
                self.assertTrue(labels)
                self.assertTrue(all(isinstance(label, str) and label for label in labels))


class GetDetectionLabelsTests(unittest.TestCase):
    def test_uses_sub_category_mapping_when_present(self):
        labels = get_detection_labels("상의", "스웨트셔츠")
        self.assertEqual(labels, SUB_CATEGORY_DETECTION_LABELS["스웨트셔츠"])

    def test_falls_back_to_main_category_when_sub_category_is_unmapped(self):
        labels = get_detection_labels("상의", "존재하지 않는 서브카테고리")
        self.assertEqual(labels, MAIN_CATEGORY_DETECTION_LABELS["상의"])

    def test_returns_none_when_neither_is_mapped(self):
        labels = get_detection_labels("존재하지 않는 메인카테고리", "존재하지 않는 서브카테고리")
        self.assertIsNone(labels)


class DeduplicateByIouTests(unittest.TestCase):
    def test_same_box_detected_under_two_synonym_labels_is_merged_into_one(self):
        # 예시: 같은 셔츠가 shirt/blouse 로 중복 탐지된 경우.
        shirt = Detection(label="shirt", score=0.92, box=(10, 10, 100, 200))
        blouse = Detection(label="blouse", score=0.84, box=(12, 11, 101, 199))

        kept = deduplicate_by_iou([shirt, blouse], iou_threshold=0.5)

        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0], shirt)  # 더 높은 confidence 를 남긴다.

    def test_non_overlapping_boxes_are_both_kept(self):
        left = Detection(label="shirt", score=0.9, box=(0, 0, 50, 50))
        right = Detection(label="shirt", score=0.9, box=(200, 200, 250, 250))

        kept = deduplicate_by_iou([left, right], iou_threshold=0.5)

        self.assertEqual(len(kept), 2)

    def test_empty_input_returns_empty_list(self):
        self.assertEqual(deduplicate_by_iou([]), [])


class BoxAreaAndIntersectionAreaTests(unittest.TestCase):
    def test_box_area(self):
        self.assertAlmostEqual(box_area((10, 20, 30, 50)), 20 * 30)

    def test_intersection_area_of_overlapping_boxes(self):
        area = intersection_area((0, 0, 10, 10), (5, 5, 15, 15))
        self.assertAlmostEqual(area, 5 * 5)

    def test_intersection_area_of_non_overlapping_boxes_is_zero(self):
        self.assertEqual(intersection_area((0, 0, 10, 10), (20, 20, 30, 30)), 0.0)


IMAGE_HEIGHT = 1000.0
# 아래 값들은 person_top_ratio<=0.01, garment_top_ratio<=0.30,
# garment_person_area_ratio>=0.42, garment_in_person_ratio>=0.90 을 모두 만족한다.
_ALL_CONDITIONS_MET_PERSON_BOX = (100.0, 0.0, 400.0, 990.0)
_ALL_CONDITIONS_MET_GARMENT_BOX = (120.0, 200.0, 380.0, 950.0)
_PERSON_BOX_AREA = 300.0 * 990.0  # _ALL_CONDITIONS_MET_PERSON_BOX 의 면적(297000).


class CheckCroppedLowerBodyTests(unittest.TestCase):
    def test_all_conditions_met_is_cropped_lower_body(self):
        result = check_cropped_lower_body(
            person_box=_ALL_CONDITIONS_MET_PERSON_BOX,
            garment_box=_ALL_CONDITIONS_MET_GARMENT_BOX,
            image_height=IMAGE_HEIGHT,
        )

        self.assertTrue(result.is_cropped_lower_body)
        self.assertAlmostEqual(result.person_top_ratio, 0.0)
        self.assertAlmostEqual(result.garment_top_ratio, 0.2)
        self.assertAlmostEqual(result.garment_person_area_ratio, 195000 / 297000)
        self.assertAlmostEqual(result.garment_in_person_ratio, 1.0)

    def test_person_top_ratio_exceeded_is_not_cropped_lower_body(self):
        # 사람 박스가 화면 위에서 5% 지점부터 시작 (> 0.01), 나머지는 그대로 통과.
        result = check_cropped_lower_body(
            person_box=(100.0, 50.0, 400.0, 990.0),
            garment_box=_ALL_CONDITIONS_MET_GARMENT_BOX,
            image_height=IMAGE_HEIGHT,
        )

        self.assertGreater(result.person_top_ratio, 0.01)
        self.assertLessEqual(result.garment_top_ratio, 0.30)
        self.assertGreaterEqual(result.garment_person_area_ratio, 0.42)
        self.assertGreaterEqual(result.garment_in_person_ratio, 0.90)
        self.assertFalse(result.is_cropped_lower_body)

    def test_garment_top_ratio_exceeded_is_not_cropped_lower_body(self):
        # 하의 박스가 화면 위에서 35% 지점부터 시작 (> 0.30), 나머지는 그대로 통과.
        result = check_cropped_lower_body(
            person_box=_ALL_CONDITIONS_MET_PERSON_BOX,
            garment_box=(120.0, 350.0, 380.0, 950.0),
            image_height=IMAGE_HEIGHT,
        )

        self.assertLessEqual(result.person_top_ratio, 0.01)
        self.assertGreater(result.garment_top_ratio, 0.30)
        self.assertGreaterEqual(result.garment_person_area_ratio, 0.42)
        self.assertGreaterEqual(result.garment_in_person_ratio, 0.90)
        self.assertFalse(result.is_cropped_lower_body)

    def test_garment_person_area_ratio_below_minimum_is_not_cropped_lower_body(self):
        # 하의 박스가 사람 박스에 비해 너무 작음(면적비 < 0.42), 나머지는 그대로 통과.
        result = check_cropped_lower_body(
            person_box=_ALL_CONDITIONS_MET_PERSON_BOX,
            garment_box=(150.0, 200.0, 300.0, 700.0),
            image_height=IMAGE_HEIGHT,
        )

        self.assertLessEqual(result.person_top_ratio, 0.01)
        self.assertLessEqual(result.garment_top_ratio, 0.30)
        self.assertLess(result.garment_person_area_ratio, 0.42)
        self.assertGreaterEqual(result.garment_in_person_ratio, 0.90)
        self.assertFalse(result.is_cropped_lower_body)

    def test_garment_person_area_ratio_just_below_threshold_is_not_cropped_lower_body(self):
        # 면적비를 threshold 바로 아래(0.41)로 두고 나머지 조건은 전부 만족시켜서,
        # area ratio 하나 때문에 fallback 이 꺼지는 경계 케이스를 확인한다.
        garment_box = (100.0, 200.0, 400.0, 605.9)  # area = 121770 = 0.41 * 297000

        result = check_cropped_lower_body(
            person_box=_ALL_CONDITIONS_MET_PERSON_BOX,
            garment_box=garment_box,
            image_height=IMAGE_HEIGHT,
        )

        self.assertAlmostEqual(result.garment_person_area_ratio, 0.41)
        self.assertLessEqual(result.person_top_ratio, 0.01)
        self.assertLessEqual(result.garment_top_ratio, 0.30)
        self.assertGreaterEqual(result.garment_in_person_ratio, 0.90)
        self.assertFalse(result.is_cropped_lower_body)

    def test_garment_person_area_ratio_at_exactly_the_threshold_is_cropped_lower_body(self):
        # 0.42 는 경계값이고 >= 이므로 정확히 0.42 여도 True 여야 한다. (110~290, 200~893 은
        # 부동소수점 오차 없이 area=124740=0.42*297000 이 정확히 나오도록 고른 값이다.)
        garment_box = (110.0, 200.0, 290.0, 893.0)  # area = 124740 = 0.42 * 297000

        result = check_cropped_lower_body(
            person_box=_ALL_CONDITIONS_MET_PERSON_BOX,
            garment_box=garment_box,
            image_height=IMAGE_HEIGHT,
        )

        self.assertAlmostEqual(result.garment_person_area_ratio, 0.42)
        self.assertTrue(result.is_cropped_lower_body)

    def test_regression_area_ratio_0_4517_is_now_cropped_lower_body(self):
        # 실제 상품(171263/182271/243204)에서 garment_person_area_ratio 가 약 0.45~0.49
        # 였는데 threshold=0.50 이라 잘못 FAIL 되던 케이스. threshold 를 0.42 로 낮춘
        # 뒤에는 True 가 나와야 한다(0.42 <= 0.4517 < 옛 threshold 0.50).
        garment_box = (100.0, 200.0, 400.0, 647.1833333333333)  # area = 134155 ≈ 0.4517 * 297000

        result = check_cropped_lower_body(
            person_box=_ALL_CONDITIONS_MET_PERSON_BOX,
            garment_box=garment_box,
            image_height=IMAGE_HEIGHT,
        )

        self.assertAlmostEqual(result.garment_person_area_ratio, 134155 / _PERSON_BOX_AREA)
        self.assertAlmostEqual(result.garment_person_area_ratio, 0.4517, places=3)
        self.assertLess(result.garment_person_area_ratio, 0.50)  # 옛 threshold 라면 FAIL 이었다.
        self.assertTrue(result.is_cropped_lower_body)

    def test_garment_in_person_ratio_below_minimum_is_not_cropped_lower_body(self):
        # 하의 박스 대부분이 사람 박스 밖으로 나가 있음(포함비 < 0.90), 나머지는 통과.
        result = check_cropped_lower_body(
            person_box=_ALL_CONDITIONS_MET_PERSON_BOX,
            garment_box=(350.0, 200.0, 650.0, 950.0),
            image_height=IMAGE_HEIGHT,
        )

        self.assertLessEqual(result.person_top_ratio, 0.01)
        self.assertLessEqual(result.garment_top_ratio, 0.30)
        self.assertGreaterEqual(result.garment_person_area_ratio, 0.42)
        self.assertLess(result.garment_in_person_ratio, 0.90)
        self.assertFalse(result.is_cropped_lower_body)

    def test_zero_image_height_is_not_cropped_lower_body(self):
        result = check_cropped_lower_body(
            person_box=_ALL_CONDITIONS_MET_PERSON_BOX,
            garment_box=_ALL_CONDITIONS_MET_GARMENT_BOX,
            image_height=0.0,
        )

        self.assertFalse(result.is_cropped_lower_body)

    def test_zero_area_person_box_is_not_cropped_lower_body(self):
        result = check_cropped_lower_body(
            person_box=(100.0, 100.0, 100.0, 100.0),
            garment_box=_ALL_CONDITIONS_MET_GARMENT_BOX,
            image_height=IMAGE_HEIGHT,
        )

        self.assertFalse(result.is_cropped_lower_body)


if __name__ == "__main__":
    unittest.main()
