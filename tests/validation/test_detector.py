"""category → detection label 매핑과 동의어 bounding box 중복 제거를 검증한다.

Grounding DINO 모델 자체는 로드하지 않는다(torch/transformers 없이도 통과한다).
"""

import unittest

from app.config.categories import CATEGORIES
from app.validation.detector import (
    MAIN_CATEGORY_DETECTION_LABELS,
    SUB_CATEGORY_DETECTION_LABELS,
    Detection,
    deduplicate_by_iou,
    get_detection_labels,
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


if __name__ == "__main__":
    unittest.main()
