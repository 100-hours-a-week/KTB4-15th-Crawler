"""한 상품 검증의 전체 흐름(orchestration)을 fake detector/pose/image_loader 로 검증한다.

실제 Grounding DINO/MediaPipe 모델은 로드하지 않는다. 이 파일은 product_image_validator.py
가 최상단에서 쓰는 OpenCV/numpy 는 필요하다(실제 모델은 아니다).
"""

import unittest
from datetime import UTC, datetime

import numpy as np

from app.models.product import Product
from app.validation.models import (
    Detection,
    DetectionResult,
    PoseDirection,
    ValidationReason,
    ValidationStatus,
)
from app.validation.product_image_validator import validate_product_image

_CRAWLED_AT = datetime(2026, 9, 22, tzinfo=UTC)
_FAKE_IMAGE = np.zeros((10, 10, 3), dtype=np.uint8)


def _product(*, sub_category="스웨트셔츠", main_category="상의") -> Product:
    return Product(
        product_code=1,
        product_name="테스트 상품",
        detail_url="https://product.29cm.co.kr/catalog/1",
        image_url="https://img.29cm.co.kr/item/example.jpg",
        price=10000,
        is_sold_out=False,
        main_category=main_category,
        sub_category=sub_category,
        color=None,
        crawled_at=_CRAWLED_AT,
    )


def _detection_result(persons=0, garments=1) -> DetectionResult:
    box = (0.0, 0.0, 5.0, 5.0)
    return DetectionResult(
        persons=[Detection("person", 0.9, box) for _ in range(persons)],
        garments=[Detection("shirt", 0.9, box) for _ in range(garments)],
    )


class FakeDetector:
    def __init__(self, result: DetectionResult):
        self._result = result
        self.calls: list[tuple] = []

    def detect(self, image_rgb, garment_labels):
        self.calls.append((image_rgb, tuple(garment_labels)))
        return self._result


class FakePoseEstimator:
    def __init__(self, direction: PoseDirection):
        self._direction = direction
        self.call_count = 0
        self.calls: list[bool] = []  # 매 호출의 allow_lower_body_fallback 값

    def estimate_direction(self, image_rgb, *, allow_lower_body_fallback=False):
        self.call_count += 1
        self.calls.append(allow_lower_body_fallback)
        return self._direction


def _fake_image_loader(url: str):
    return _FAKE_IMAGE


def _failing_image_loader(url: str):
    raise ValueError("boom")


class ValidateProductImageTests(unittest.TestCase):
    def test_category_mapping_not_found_short_circuits_before_download(self):
        loaded = []

        def loader(url):
            loaded.append(url)
            return _FAKE_IMAGE

        product = _product(main_category="없는카테고리", sub_category="없는서브카테고리")
        detector = FakeDetector(_detection_result())

        outcome = validate_product_image(
            product,
            detector=detector,
            pose_estimator=FakePoseEstimator(PoseDirection.FRONT),
            image_loader=loader,
        )

        self.assertEqual(outcome.status, ValidationStatus.FAIL)
        self.assertEqual(outcome.reason, ValidationReason.CATEGORY_MAPPING_NOT_FOUND)
        self.assertEqual(loaded, [])  # 이미지를 내려받지 않는다.
        self.assertEqual(detector.calls, [])

    def test_image_load_failure_becomes_image_load_failed(self):
        outcome = validate_product_image(
            _product(),
            detector=FakeDetector(_detection_result()),
            pose_estimator=FakePoseEstimator(PoseDirection.FRONT),
            image_loader=_failing_image_loader,
        )

        self.assertEqual(outcome.status, ValidationStatus.FAIL)
        self.assertEqual(outcome.reason, ValidationReason.IMAGE_LOAD_FAILED)

    def test_sub_category_mapping_is_used_for_detection_labels(self):
        detector = FakeDetector(_detection_result(persons=0, garments=1))

        validate_product_image(
            _product(sub_category="스웨트셔츠"),
            detector=detector,
            pose_estimator=FakePoseEstimator(PoseDirection.FRONT),
            image_loader=_fake_image_loader,
        )

        [(_, labels)] = detector.calls
        self.assertIn("sweatshirt", labels)

    def test_main_category_fallback_is_used_when_sub_category_is_unmapped(self):
        detector = FakeDetector(_detection_result(persons=0, garments=1))

        validate_product_image(
            _product(main_category="상의", sub_category="없는서브카테고리"),
            detector=detector,
            pose_estimator=FakePoseEstimator(PoseDirection.FRONT),
            image_loader=_fake_image_loader,
        )

        [(_, labels)] = detector.calls
        self.assertIn("shirt", labels)

    def test_pose_estimator_is_only_called_when_exactly_one_person_is_found(self):
        pose_estimator = FakePoseEstimator(PoseDirection.FRONT)

        validate_product_image(
            _product(),
            detector=FakeDetector(_detection_result(persons=0, garments=1)),
            pose_estimator=pose_estimator,
            image_loader=_fake_image_loader,
        )

        self.assertEqual(pose_estimator.call_count, 0)

    def test_single_person_front_passes(self):
        outcome = validate_product_image(
            _product(),
            detector=FakeDetector(_detection_result(persons=1, garments=1)),
            pose_estimator=FakePoseEstimator(PoseDirection.FRONT),
            image_loader=_fake_image_loader,
        )

        self.assertEqual(outcome.status, ValidationStatus.PASS)

    def test_single_person_back_fails(self):
        outcome = validate_product_image(
            _product(),
            detector=FakeDetector(_detection_result(persons=1, garments=1)),
            pose_estimator=FakePoseEstimator(PoseDirection.BACK),
            image_loader=_fake_image_loader,
        )

        self.assertEqual(outcome.status, ValidationStatus.FAIL)
        self.assertEqual(outcome.reason, ValidationReason.BACK_VIEW)

    def test_multiple_persons_fails_without_calling_pose_estimator(self):
        pose_estimator = FakePoseEstimator(PoseDirection.FRONT)

        outcome = validate_product_image(
            _product(),
            detector=FakeDetector(_detection_result(persons=2, garments=1)),
            pose_estimator=pose_estimator,
            image_loader=_fake_image_loader,
        )

        self.assertEqual(outcome.status, ValidationStatus.FAIL)
        self.assertEqual(outcome.reason, ValidationReason.MULTIPLE_PERSONS)
        self.assertEqual(pose_estimator.call_count, 0)

    def test_bottom_product_with_one_garment_enables_lower_body_fallback(self):
        pose_estimator = FakePoseEstimator(PoseDirection.FRONT)

        validate_product_image(
            _product(main_category="하의", sub_category="슬림 팬츠"),
            detector=FakeDetector(_detection_result(persons=1, garments=1)),
            pose_estimator=pose_estimator,
            image_loader=_fake_image_loader,
        )

        self.assertEqual(pose_estimator.calls, [True])

    def test_top_product_does_not_enable_lower_body_fallback(self):
        pose_estimator = FakePoseEstimator(PoseDirection.FRONT)

        validate_product_image(
            _product(main_category="상의", sub_category="스웨트셔츠"),
            detector=FakeDetector(_detection_result(persons=1, garments=1)),
            pose_estimator=pose_estimator,
            image_loader=_fake_image_loader,
        )

        self.assertEqual(pose_estimator.calls, [False])

    def test_bottom_product_with_multiple_garments_does_not_enable_fallback(self):
        pose_estimator = FakePoseEstimator(PoseDirection.FRONT)

        validate_product_image(
            _product(main_category="하의", sub_category="슬림 팬츠"),
            detector=FakeDetector(_detection_result(persons=1, garments=2)),
            pose_estimator=pose_estimator,
            image_loader=_fake_image_loader,
        )

        self.assertEqual(pose_estimator.calls, [False])


if __name__ == "__main__":
    unittest.main()
