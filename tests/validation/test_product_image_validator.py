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
    ValidationReason,
    ValidationStatus,
)
from app.validation.pose import BodyLandmarks
from app.validation.product_image_validator import validate_product_image

_CRAWLED_AT = datetime(2026, 9, 22, tzinfo=UTC)
_FAKE_IMAGE = np.zeros((10, 10, 3), dtype=np.uint8)

# 하의 크롭샷 bbox fallback 테스트 전용. tests/validation/test_detector.py 의
# CheckCroppedLowerBodyTests 와 같은 값이다(그쪽은 check_cropped_lower_body() 자체를,
# 여기서는 validate_product_image() 를 통한 최종 판정을 검증한다).
_CROP_IMAGE_HEIGHT = 1000
_CROP_IMAGE = np.zeros((_CROP_IMAGE_HEIGHT, 500, 3), dtype=np.uint8)
_CROP_PERSON_BOX_OK = (100.0, 0.0, 400.0, 990.0)
_CROP_GARMENT_BOX_OK = (120.0, 200.0, 380.0, 950.0)

# classify_direction() 을 실제로 거치므로(더 이상 estimate_direction() 을 페이크로
# 가짜 PoseDirection 을 바로 돌려주지 않는다), 원하는 판정 결과가 나오는 실제
# BodyLandmarks 를 만든다. 값은 tests/validation/test_pose.py 의 _landmarks() 와 같은
# 구성이다.
_VISIBLE = 0.95
_HIDDEN = 0.1
_TORSO_Y = 0.3


def _landmarks(shoulder_dx: float = 0.0, *, nose_visibility: float = _VISIBLE) -> BodyLandmarks:
    right_x = 0.5
    return BodyLandmarks(
        nose_visibility=nose_visibility,
        left_shoulder_x=right_x + shoulder_dx,
        left_shoulder_visibility=_VISIBLE,
        right_shoulder_x=right_x,
        right_shoulder_visibility=_VISIBLE,
        shoulder_mid_y=0.0,
        hip_mid_y=_TORSO_Y,
        left_hip_visibility=_VISIBLE,
        right_hip_visibility=_VISIBLE,
        left_knee_visibility=_VISIBLE,
        right_knee_visibility=_VISIBLE,
        left_ankle_visibility=_VISIBLE,
        right_ankle_visibility=_VISIBLE,
    )


_FRONT_LANDMARKS = _landmarks(shoulder_dx=0.15)  # rotation_ratio = 0.5 -> FRONT
_BACK_LANDMARKS = _landmarks(shoulder_dx=-0.15, nose_visibility=_HIDDEN)  # -> BACK
# 어깨/골반 모두 안 보이므로 allow_lower_body_fallback 값과 무관하게 항상 UNCERTAIN 이다.
_UNCERTAIN_LANDMARKS = BodyLandmarks(
    nose_visibility=_HIDDEN,
    left_shoulder_x=0.5,
    left_shoulder_visibility=_HIDDEN,
    right_shoulder_x=0.5,
    right_shoulder_visibility=_HIDDEN,
    shoulder_mid_y=0.0,
    hip_mid_y=_TORSO_Y,
    left_hip_visibility=_HIDDEN,
    right_hip_visibility=_HIDDEN,
    left_knee_visibility=_HIDDEN,
    right_knee_visibility=_HIDDEN,
    left_ankle_visibility=_HIDDEN,
    right_ankle_visibility=_HIDDEN,
)
# 상반신(어깨/코)이 잘려 안 보이지만 하반신(골반/무릎/발목)은 보이는 landmarks.
# allow_lower_body_fallback=True 일 때만 FRONT, False 면 UNCERTAIN 이 된다
# (tests/validation/test_pose.py 의 BottomWearLowerBodyFallbackTests 와 같은 구성).
_CROPPED_UPPER_BODY_LANDMARKS = BodyLandmarks(
    nose_visibility=_HIDDEN,
    left_shoulder_x=0.5,
    left_shoulder_visibility=_HIDDEN,
    right_shoulder_x=0.5,
    right_shoulder_visibility=_HIDDEN,
    shoulder_mid_y=0.0,
    hip_mid_y=_TORSO_Y,
    left_hip_visibility=_VISIBLE,
    right_hip_visibility=_VISIBLE,
    left_knee_visibility=_VISIBLE,
    right_knee_visibility=_VISIBLE,
    left_ankle_visibility=_VISIBLE,
    right_ankle_visibility=_VISIBLE,
)


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
    """get_landmarks() 만 구현한다. validate_product_image() 는 이제 이 메서드만 쓴다
    (MediaPipe 를 한 번만 돌리기 위해 estimate_direction() 은 이 경로에서 쓰지 않는다).
    """

    def __init__(self, landmarks: BodyLandmarks | None = _FRONT_LANDMARKS):
        self._landmarks = landmarks
        self.get_landmarks_calls = 0

    def get_landmarks(self, image_rgb):
        self.get_landmarks_calls += 1
        return self._landmarks


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
            pose_estimator=FakePoseEstimator(),
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
            pose_estimator=FakePoseEstimator(),
            image_loader=_failing_image_loader,
        )

        self.assertEqual(outcome.status, ValidationStatus.FAIL)
        self.assertEqual(outcome.reason, ValidationReason.IMAGE_LOAD_FAILED)

    def test_sub_category_mapping_is_used_for_detection_labels(self):
        detector = FakeDetector(_detection_result(persons=0, garments=1))

        validate_product_image(
            _product(sub_category="스웨트셔츠"),
            detector=detector,
            pose_estimator=FakePoseEstimator(),
            image_loader=_fake_image_loader,
        )

        [(_, labels)] = detector.calls
        self.assertIn("sweatshirt", labels)

    def test_main_category_fallback_is_used_when_sub_category_is_unmapped(self):
        detector = FakeDetector(_detection_result(persons=0, garments=1))

        validate_product_image(
            _product(main_category="상의", sub_category="없는서브카테고리"),
            detector=detector,
            pose_estimator=FakePoseEstimator(),
            image_loader=_fake_image_loader,
        )

        [(_, labels)] = detector.calls
        self.assertIn("shirt", labels)

    def test_pose_estimator_is_only_called_when_exactly_one_person_is_found(self):
        pose_estimator = FakePoseEstimator()

        validate_product_image(
            _product(),
            detector=FakeDetector(_detection_result(persons=0, garments=1)),
            pose_estimator=pose_estimator,
            image_loader=_fake_image_loader,
        )

        self.assertEqual(pose_estimator.get_landmarks_calls, 0)

    def test_single_person_front_passes(self):
        outcome = validate_product_image(
            _product(),
            detector=FakeDetector(_detection_result(persons=1, garments=1)),
            pose_estimator=FakePoseEstimator(landmarks=_FRONT_LANDMARKS),
            image_loader=_fake_image_loader,
        )

        self.assertEqual(outcome.status, ValidationStatus.PASS)

    def test_single_person_back_fails(self):
        outcome = validate_product_image(
            _product(),
            detector=FakeDetector(_detection_result(persons=1, garments=1)),
            pose_estimator=FakePoseEstimator(landmarks=_BACK_LANDMARKS),
            image_loader=_fake_image_loader,
        )

        self.assertEqual(outcome.status, ValidationStatus.FAIL)
        self.assertEqual(outcome.reason, ValidationReason.BACK_VIEW)

    def test_multiple_persons_fails_without_calling_pose_estimator(self):
        pose_estimator = FakePoseEstimator()

        outcome = validate_product_image(
            _product(),
            detector=FakeDetector(_detection_result(persons=2, garments=1)),
            pose_estimator=pose_estimator,
            image_loader=_fake_image_loader,
        )

        self.assertEqual(outcome.status, ValidationStatus.FAIL)
        self.assertEqual(outcome.reason, ValidationReason.MULTIPLE_PERSONS)
        self.assertEqual(pose_estimator.get_landmarks_calls, 0)

    def test_get_landmarks_is_called_exactly_once_when_one_person_is_found(self):
        # MediaPipe(get_landmarks) 는 person_count == 1 일 때 정확히 한 번만 호출돼야
        # 한다 — estimate_direction() 을 거쳐 추가로 호출되지 않는다.
        pose_estimator = FakePoseEstimator(landmarks=_FRONT_LANDMARKS)

        validate_product_image(
            _product(),
            detector=FakeDetector(_detection_result(persons=1, garments=1)),
            pose_estimator=pose_estimator,
            image_loader=_fake_image_loader,
        )

        self.assertEqual(pose_estimator.get_landmarks_calls, 1)

    def test_bottom_product_with_one_garment_enables_lower_body_fallback(self):
        # 상체 landmark 는 부족하지만 하체 landmark 는 충분한 경우, 하의 + garment
        # 1개 조합에서만 allow_lower_body_fallback=True 로 classify_direction 이
        # FRONT(=PASS)를 내는지로 간접 확인한다(pose_estimator 는 이제 이 값을
        # 직접 넘겨받지 않는다 — product_image_validator 가 classify_direction 을
        # 직접 호출하기 때문이다).
        pose_estimator = FakePoseEstimator(landmarks=_CROPPED_UPPER_BODY_LANDMARKS)

        outcome = validate_product_image(
            _product(main_category="하의", sub_category="슬림 팬츠"),
            detector=FakeDetector(_detection_result(persons=1, garments=1)),
            pose_estimator=pose_estimator,
            image_loader=_fake_image_loader,
        )

        self.assertEqual(outcome.status, ValidationStatus.PASS)
        self.assertEqual(pose_estimator.get_landmarks_calls, 1)

    def test_top_product_does_not_enable_lower_body_fallback(self):
        pose_estimator = FakePoseEstimator(landmarks=_CROPPED_UPPER_BODY_LANDMARKS)

        outcome = validate_product_image(
            _product(main_category="상의", sub_category="스웨트셔츠"),
            detector=FakeDetector(_detection_result(persons=1, garments=1)),
            pose_estimator=pose_estimator,
            image_loader=_fake_image_loader,
        )

        self.assertEqual(outcome.status, ValidationStatus.FAIL)
        self.assertEqual(outcome.reason, ValidationReason.UNCERTAIN)

    def test_bottom_product_with_multiple_garments_does_not_enable_fallback(self):
        pose_estimator = FakePoseEstimator(landmarks=_CROPPED_UPPER_BODY_LANDMARKS)

        outcome = validate_product_image(
            _product(main_category="하의", sub_category="슬림 팬츠"),
            detector=FakeDetector(_detection_result(persons=1, garments=2)),
            pose_estimator=pose_estimator,
            image_loader=_fake_image_loader,
        )

        self.assertEqual(outcome.status, ValidationStatus.FAIL)
        self.assertEqual(outcome.reason, ValidationReason.UNCERTAIN)


class BottomCropFallbackTests(unittest.TestCase):
    """하의 + person1 + garment1 + MediaPipe landmarks=None 전용 bbox fallback."""

    @staticmethod
    def _detection(person_box, garment_box) -> DetectionResult:
        return DetectionResult(
            persons=[Detection("person", 0.9, person_box)],
            garments=[Detection("slim pants", 0.9, garment_box)],
        )

    @staticmethod
    def _bottom_product() -> Product:
        return _product(main_category="하의", sub_category="슬림 팬츠")

    def _crop_image_loader(self, url: str):
        return _CROP_IMAGE

    def test_all_bbox_conditions_met_passes(self):
        pose_estimator = FakePoseEstimator(landmarks=None)

        outcome = validate_product_image(
            self._bottom_product(),
            detector=FakeDetector(self._detection(_CROP_PERSON_BOX_OK, _CROP_GARMENT_BOX_OK)),
            pose_estimator=pose_estimator,
            image_loader=self._crop_image_loader,
        )

        self.assertEqual(outcome.status, ValidationStatus.PASS)
        self.assertEqual(outcome.reason, ValidationReason.VALID)
        self.assertEqual(pose_estimator.get_landmarks_calls, 1)

    def test_person_top_ratio_exceeded_stays_uncertain(self):
        person_box = (100.0, 50.0, 400.0, 990.0)  # person_top_ratio = 0.05 > 0.01

        outcome = validate_product_image(
            self._bottom_product(),
            detector=FakeDetector(self._detection(person_box, _CROP_GARMENT_BOX_OK)),
            pose_estimator=FakePoseEstimator(landmarks=None),
            image_loader=self._crop_image_loader,
        )

        self.assertEqual(outcome.status, ValidationStatus.FAIL)
        self.assertEqual(outcome.reason, ValidationReason.UNCERTAIN)

    def test_garment_top_ratio_exceeded_stays_uncertain(self):
        garment_box = (120.0, 350.0, 380.0, 950.0)  # garment_top_ratio = 0.35 > 0.30

        outcome = validate_product_image(
            self._bottom_product(),
            detector=FakeDetector(self._detection(_CROP_PERSON_BOX_OK, garment_box)),
            pose_estimator=FakePoseEstimator(landmarks=None),
            image_loader=self._crop_image_loader,
        )

        self.assertEqual(outcome.status, ValidationStatus.FAIL)
        self.assertEqual(outcome.reason, ValidationReason.UNCERTAIN)

    def test_garment_person_area_ratio_below_minimum_stays_uncertain(self):
        garment_box = (150.0, 200.0, 300.0, 700.0)  # 면적비 < 0.42

        outcome = validate_product_image(
            self._bottom_product(),
            detector=FakeDetector(self._detection(_CROP_PERSON_BOX_OK, garment_box)),
            pose_estimator=FakePoseEstimator(landmarks=None),
            image_loader=self._crop_image_loader,
        )

        self.assertEqual(outcome.status, ValidationStatus.FAIL)
        self.assertEqual(outcome.reason, ValidationReason.UNCERTAIN)

    def test_regression_area_ratio_0_4517_now_passes_end_to_end(self):
        # 실제 상품(171263/182271/243204)에서 garment_person_area_ratio 가 약 0.45~0.49 라서
        # threshold=0.50 일 때는 잘못 FAIL/UNCERTAIN 이었다. 0.42 로 낮춘 뒤에는
        # validate_product_image() 전체 흐름에서도 PASS 가 나와야 한다.
        garment_box = (100.0, 200.0, 400.0, 647.1833333333333)  # area ratio ≈ 0.4517

        outcome = validate_product_image(
            self._bottom_product(),
            detector=FakeDetector(self._detection(_CROP_PERSON_BOX_OK, garment_box)),
            pose_estimator=FakePoseEstimator(landmarks=None),
            image_loader=self._crop_image_loader,
        )

        self.assertEqual(outcome.status, ValidationStatus.PASS)
        self.assertEqual(outcome.reason, ValidationReason.VALID)

    def test_garment_in_person_ratio_below_minimum_stays_uncertain(self):
        garment_box = (350.0, 200.0, 650.0, 950.0)  # 포함비 < 0.90

        outcome = validate_product_image(
            self._bottom_product(),
            detector=FakeDetector(self._detection(_CROP_PERSON_BOX_OK, garment_box)),
            pose_estimator=FakePoseEstimator(landmarks=None),
            image_loader=self._crop_image_loader,
        )

        self.assertEqual(outcome.status, ValidationStatus.FAIL)
        self.assertEqual(outcome.reason, ValidationReason.UNCERTAIN)

    def test_top_product_does_not_get_bbox_fallback_even_with_the_same_boxes(self):
        top_product = _product(main_category="상의", sub_category="스웨트셔츠")

        outcome = validate_product_image(
            top_product,
            detector=FakeDetector(self._detection(_CROP_PERSON_BOX_OK, _CROP_GARMENT_BOX_OK)),
            pose_estimator=FakePoseEstimator(landmarks=None),
            image_loader=self._crop_image_loader,
        )

        self.assertEqual(outcome.status, ValidationStatus.FAIL)
        self.assertEqual(outcome.reason, ValidationReason.UNCERTAIN)

    def test_landmarks_present_uses_existing_pose_result_not_bbox(self):
        # landmarks 가 있으면(None 이 아니면) bbox 를 아예 계산하지 않고 classify_direction
        # 결과(여기서는 UNCERTAIN)를 그대로 쓴다 — bbox 조건은 전부 만족하는 box 를 줘도.
        pose_estimator = FakePoseEstimator(landmarks=_UNCERTAIN_LANDMARKS)

        outcome = validate_product_image(
            self._bottom_product(),
            detector=FakeDetector(self._detection(_CROP_PERSON_BOX_OK, _CROP_GARMENT_BOX_OK)),
            pose_estimator=pose_estimator,
            image_loader=self._crop_image_loader,
        )

        self.assertEqual(outcome.status, ValidationStatus.FAIL)
        self.assertEqual(outcome.reason, ValidationReason.UNCERTAIN)
        self.assertEqual(pose_estimator.get_landmarks_calls, 1)

    def test_get_landmarks_is_called_exactly_once_even_when_landmarks_are_none(self):
        # 회귀 테스트: get_landmarks() 가 None 을 반환해서 bbox fallback 으로 넘어가는
        # 경로에서도 MediaPipe(get_landmarks) 는 딱 한 번만 호출돼야 한다. (수정 전에는
        # estimate_direction() 내부 호출 1회 + UNCERTAIN 판정 후 재확인 1회, 총 2회
        # 호출되는 버그가 있었다.)
        pose_estimator = FakePoseEstimator(landmarks=None)

        validate_product_image(
            self._bottom_product(),
            detector=FakeDetector(self._detection(_CROP_PERSON_BOX_OK, _CROP_GARMENT_BOX_OK)),
            pose_estimator=pose_estimator,
            image_loader=self._crop_image_loader,
        )

        self.assertEqual(pose_estimator.get_landmarks_calls, 1)


if __name__ == "__main__":
    unittest.main()
