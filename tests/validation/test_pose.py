"""landmark 좌표만으로 방향 판정 로직을 검증한다. MediaPipe 모델은 로드하지 않는다."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

from app.validation.models import PoseDirection
from app.validation.pose import (
    BodyLandmarks,
    PoseEstimator,
    classify_direction,
    extract_landmarks,
    has_sufficient_lower_body_landmarks,
    has_sufficient_upper_body_landmarks,
)

_VISIBLE = 0.95
_HIDDEN = 0.1  # MIN_LANDMARK_VISIBILITY(0.5) 아래.
_TORSO_Y = 0.3  # shoulder_mid_y, hip_mid_y 가 이만큼 떨어져 있다고 가정한다.


def _landmarks(
    shoulder_dx: float = 0.0,
    *,
    nose_visibility: float = _VISIBLE,
    shoulder_visibility: float = _VISIBLE,
    hip_visibility: float = _VISIBLE,
    knee_visibility: float = _VISIBLE,
    ankle_visibility: float = _VISIBLE,
) -> BodyLandmarks:
    """shoulder_dx = left_shoulder_x - right_shoulder_x."""
    right_x = 0.5
    return BodyLandmarks(
        nose_visibility=nose_visibility,
        left_shoulder_x=right_x + shoulder_dx,
        left_shoulder_visibility=shoulder_visibility,
        right_shoulder_x=right_x,
        right_shoulder_visibility=shoulder_visibility,
        shoulder_mid_y=0.0,
        hip_mid_y=_TORSO_Y,
        left_hip_visibility=hip_visibility,
        right_hip_visibility=hip_visibility,
        left_knee_visibility=knee_visibility,
        right_knee_visibility=knee_visibility,
        left_ankle_visibility=ankle_visibility,
        right_ankle_visibility=ankle_visibility,
    )


class ClassifyDirectionTests(unittest.TestCase):
    def test_no_landmarks_is_uncertain(self):
        self.assertEqual(classify_direction(None), PoseDirection.UNCERTAIN)

    def test_clearly_mirrored_shoulders_is_front(self):
        # rotation_ratio = shoulder_dx / torso_scale = 0.15 / 0.3 = 0.5 (>= 0.35)
        landmarks = _landmarks(shoulder_dx=0.15)
        self.assertEqual(classify_direction(landmarks), PoseDirection.FRONT)

    def test_partially_turned_toward_front_is_side_allowed(self):
        # rotation_ratio = 0.06 / 0.3 = 0.2 (0.10 <= x < 0.35)
        landmarks = _landmarks(shoulder_dx=0.06)
        self.assertEqual(classify_direction(landmarks), PoseDirection.SIDE_ALLOWED)

    def test_clearly_non_mirrored_shoulders_is_back(self):
        # rotation_ratio = -0.15 / 0.3 = -0.5 (<= -0.35)
        landmarks = _landmarks(shoulder_dx=-0.15, nose_visibility=0.1)
        self.assertEqual(classify_direction(landmarks), PoseDirection.BACK)

    def test_near_zero_rotation_is_side_angle_too_large(self):
        # rotation_ratio = 0.0 / 0.3 = 0.0 (거의 정측면)
        landmarks = _landmarks(shoulder_dx=0.0, nose_visibility=0.1)
        self.assertEqual(classify_direction(landmarks), PoseDirection.SIDE_ANGLE_TOO_LARGE)

    def test_low_shoulder_visibility_is_uncertain(self):
        landmarks = BodyLandmarks(
            nose_visibility=_VISIBLE,
            left_shoulder_x=0.6,
            left_shoulder_visibility=0.1,
            right_shoulder_x=0.5,
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
        self.assertEqual(classify_direction(landmarks), PoseDirection.UNCERTAIN)

    def test_degenerate_torso_scale_is_uncertain(self):
        landmarks = _landmarks(shoulder_dx=0.15)
        landmarks = BodyLandmarks(**{**landmarks.__dict__, "hip_mid_y": 0.0})
        self.assertEqual(classify_direction(landmarks), PoseDirection.UNCERTAIN)

    def test_front_leaning_result_with_hidden_face_is_uncertain(self):
        # 어깨 좌표로는 FRONT 지만 얼굴이 거의 안 보이면 모순이므로 신뢰하지 않는다.
        landmarks = _landmarks(shoulder_dx=0.15, nose_visibility=0.1)
        self.assertEqual(classify_direction(landmarks), PoseDirection.UNCERTAIN)


class _Landmark:
    def __init__(self, x, y, visibility):
        self.x, self.y, self.visibility = x, y, visibility


class _FakePoseLandmarkerResult:
    def __init__(self, pose_landmarks):
        self.pose_landmarks = pose_landmarks


def _fake_33_landmarks(overrides: dict[int, _Landmark]) -> list[_Landmark]:
    points = [_Landmark(0.5, 0.5, _VISIBLE) for _ in range(33)]
    for index, landmark in overrides.items():
        points[index] = landmark
    return points


class ExtractLandmarksTests(unittest.TestCase):
    def test_no_detected_pose_returns_none(self):
        result = _FakePoseLandmarkerResult(pose_landmarks=[])
        self.assertIsNone(extract_landmarks(result))

    def test_extracts_the_first_detected_pose(self):
        points = _fake_33_landmarks(
            {
                0: _Landmark(0.5, 0.2, 0.9),  # nose
                11: _Landmark(0.6, 0.4, 0.8),  # left shoulder
                12: _Landmark(0.4, 0.4, 0.8),  # right shoulder
                23: _Landmark(0.55, 0.7, 0.8),  # left hip
                24: _Landmark(0.45, 0.7, 0.8),  # right hip
                25: _Landmark(0.55, 0.85, 0.7),  # left knee
                26: _Landmark(0.45, 0.85, 0.7),  # right knee
                27: _Landmark(0.55, 1.0, 0.6),  # left ankle
                28: _Landmark(0.45, 1.0, 0.6),  # right ankle
            }
        )
        result = _FakePoseLandmarkerResult(pose_landmarks=[points])

        landmarks = extract_landmarks(result)

        self.assertIsNotNone(landmarks)
        self.assertAlmostEqual(landmarks.left_shoulder_x, 0.6)
        self.assertAlmostEqual(landmarks.right_shoulder_x, 0.4)
        self.assertAlmostEqual(landmarks.nose_visibility, 0.9)
        self.assertAlmostEqual(landmarks.shoulder_mid_y, 0.4)
        self.assertAlmostEqual(landmarks.hip_mid_y, 0.7)
        self.assertAlmostEqual(landmarks.left_hip_visibility, 0.8)
        self.assertAlmostEqual(landmarks.right_hip_visibility, 0.8)
        self.assertAlmostEqual(landmarks.left_knee_visibility, 0.7)
        self.assertAlmostEqual(landmarks.right_ankle_visibility, 0.6)


class HasSufficientUpperBodyLandmarksTests(unittest.TestCase):
    def test_true_when_both_shoulders_are_visible(self):
        self.assertTrue(has_sufficient_upper_body_landmarks(_landmarks()))

    def test_false_when_either_shoulder_is_not_visible(self):
        self.assertFalse(
            has_sufficient_upper_body_landmarks(_landmarks(shoulder_visibility=_HIDDEN))
        )


class HasSufficientLowerBodyLandmarksTests(unittest.TestCase):
    def test_true_when_hips_and_a_knee_are_visible(self):
        landmarks = _landmarks(
            hip_visibility=_VISIBLE, knee_visibility=_VISIBLE, ankle_visibility=_HIDDEN
        )
        self.assertTrue(has_sufficient_lower_body_landmarks(landmarks))

    def test_true_when_hips_and_an_ankle_are_visible(self):
        landmarks = _landmarks(
            hip_visibility=_VISIBLE, knee_visibility=_HIDDEN, ankle_visibility=_VISIBLE
        )
        self.assertTrue(has_sufficient_lower_body_landmarks(landmarks))

    def test_false_when_hips_are_not_visible(self):
        landmarks = _landmarks(
            hip_visibility=_HIDDEN, knee_visibility=_VISIBLE, ankle_visibility=_VISIBLE
        )
        self.assertFalse(has_sufficient_lower_body_landmarks(landmarks))

    def test_false_when_neither_knee_nor_ankle_is_visible(self):
        landmarks = _landmarks(
            hip_visibility=_VISIBLE, knee_visibility=_HIDDEN, ankle_visibility=_HIDDEN
        )
        self.assertFalse(has_sufficient_lower_body_landmarks(landmarks))


class BottomWearLowerBodyFallbackTests(unittest.TestCase):
    """하의 상품에서 상반신이 잘린 경우의 fallback(요구사항: person 1 + 하의 1개)."""

    def _cropped_upper_body(self, **lower_body_overrides) -> BodyLandmarks:
        # 어깨/코가 안 보이는, 상반신이 잘린 사진을 흉내낸다.
        return _landmarks(
            shoulder_visibility=_HIDDEN, nose_visibility=_HIDDEN, **lower_body_overrides
        )

    def test_sufficient_lower_body_passes_as_front(self):
        landmarks = self._cropped_upper_body(hip_visibility=_VISIBLE, knee_visibility=_VISIBLE)

        direction = classify_direction(landmarks, allow_lower_body_fallback=True)

        self.assertEqual(direction, PoseDirection.FRONT)

    def test_insufficient_lower_body_is_uncertain(self):
        landmarks = self._cropped_upper_body(
            hip_visibility=_HIDDEN, knee_visibility=_HIDDEN, ankle_visibility=_HIDDEN
        )

        direction = classify_direction(landmarks, allow_lower_body_fallback=True)

        self.assertEqual(direction, PoseDirection.UNCERTAIN)

    def test_fallback_disabled_by_default_even_with_sufficient_lower_body(self):
        # allow_lower_body_fallback 이 False(기본값)면 하체가 충분히 보여도 동작이
        # 바뀌면 안 된다. 상의/아우터 등 기존 흐름을 그대로 보존하기 위한 회귀 테스트다.
        landmarks = self._cropped_upper_body(hip_visibility=_VISIBLE, knee_visibility=_VISIBLE)

        self.assertEqual(classify_direction(landmarks), PoseDirection.UNCERTAIN)
        self.assertEqual(
            classify_direction(landmarks, allow_lower_body_fallback=False),
            PoseDirection.UNCERTAIN,
        )

    def test_fallback_is_ignored_when_upper_body_is_already_sufficient(self):
        # 상체가 충분히 보이면 하체 상태와 무관하게 기존 rotation_ratio 판정을 그대로 쓴다.
        landmarks = _landmarks(shoulder_dx=0.15)  # 기존 FRONT 케이스

        with_fallback = classify_direction(landmarks, allow_lower_body_fallback=True)
        without_fallback = classify_direction(landmarks, allow_lower_body_fallback=False)

        self.assertEqual(with_fallback, without_fallback)
        self.assertEqual(with_fallback, PoseDirection.FRONT)


class PoseEstimatorGetLandmarksTests(unittest.TestCase):
    """get_landmarks()/estimate_direction() 의 동작이 리팩터링 전과 같은지 확인한다."""

    def _build_estimator(self, pose_landmarks: list) -> PoseEstimator:
        cpu_delegate = object()
        image_mode = object()

        class FakeBaseOptions:
            Delegate = SimpleNamespace(CPU=cpu_delegate)

            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)

        class FakePoseLandmarkerOptions:
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)

        class FakeLandmarker:
            @staticmethod
            def create_from_options(options):
                return FakeLandmarker()

            def detect(self, mp_image):
                return _FakePoseLandmarkerResult(pose_landmarks=pose_landmarks)

            def close(self):
                pass

        class FakeImage:
            def __init__(self, *, image_format, data):
                self.image_format = image_format
                self.data = data

        fake_mediapipe = ModuleType("mediapipe")
        fake_mediapipe.ImageFormat = SimpleNamespace(SRGB=object())
        fake_mediapipe.Image = FakeImage
        fake_mediapipe.tasks = SimpleNamespace(
            BaseOptions=FakeBaseOptions,
            vision=SimpleNamespace(
                PoseLandmarkerOptions=FakePoseLandmarkerOptions,
                RunningMode=SimpleNamespace(IMAGE=image_mode),
                PoseLandmarker=FakeLandmarker,
            ),
        )

        with TemporaryDirectory() as temporary_directory:
            model_path = Path(temporary_directory) / "pose_landmarker.task"
            model_path.touch()
            with patch.dict("sys.modules", {"mediapipe": fake_mediapipe}):
                return PoseEstimator(model_path=str(model_path))

    def test_get_landmarks_extracts_the_detected_pose(self):
        points = _fake_33_landmarks(
            {
                11: _Landmark(0.6, 0.4, 0.9),  # left shoulder
                12: _Landmark(0.4, 0.4, 0.9),  # right shoulder
                23: _Landmark(0.55, 0.7, 0.9),  # left hip
                24: _Landmark(0.45, 0.7, 0.9),  # right hip
            }
        )
        estimator = self._build_estimator(pose_landmarks=[points])

        landmarks = estimator.get_landmarks("fake-image")

        self.assertIsNotNone(landmarks)
        self.assertAlmostEqual(landmarks.left_shoulder_x, 0.6)
        estimator.close()

    def test_get_landmarks_returns_none_when_nobody_is_detected(self):
        estimator = self._build_estimator(pose_landmarks=[])

        self.assertIsNone(estimator.get_landmarks("fake-image"))
        estimator.close()

    def test_estimate_direction_still_works_through_get_landmarks(self):
        # 어깨가 뚜렷하게 반전된, 정면으로 보는 사람 케이스.
        points = _fake_33_landmarks(
            {
                0: _Landmark(0.5, 0.2, 0.9),  # nose
                11: _Landmark(0.65, 0.4, 0.9),  # left shoulder
                12: _Landmark(0.35, 0.4, 0.9),  # right shoulder
                23: _Landmark(0.55, 0.7, 0.9),  # left hip
                24: _Landmark(0.45, 0.7, 0.9),  # right hip
            }
        )
        estimator = self._build_estimator(pose_landmarks=[points])

        direction = estimator.estimate_direction("fake-image")

        self.assertEqual(direction, PoseDirection.FRONT)
        estimator.close()


class PoseEstimatorOptionsTests(unittest.TestCase):
    def test_initializes_image_mode_cpu_without_segmentation_masks(self):
        cpu_delegate = object()
        image_mode = object()
        captured = {}

        class FakeBaseOptions:
            Delegate = SimpleNamespace(CPU=cpu_delegate)

            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)

        class FakePoseLandmarkerOptions:
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)

        class FakeLandmarker:
            @staticmethod
            def create_from_options(options):
                captured["options"] = options
                return SimpleNamespace(close=lambda: None)

        fake_mediapipe = ModuleType("mediapipe")
        fake_mediapipe.tasks = SimpleNamespace(
            BaseOptions=FakeBaseOptions,
            vision=SimpleNamespace(
                PoseLandmarkerOptions=FakePoseLandmarkerOptions,
                RunningMode=SimpleNamespace(IMAGE=image_mode),
                PoseLandmarker=FakeLandmarker,
            ),
        )

        with TemporaryDirectory() as temporary_directory:
            model_path = Path(temporary_directory) / "pose_landmarker.task"
            model_path.touch()
            with patch.dict("sys.modules", {"mediapipe": fake_mediapipe}):
                estimator = PoseEstimator(model_path=str(model_path))

        options = captured["options"]
        self.assertEqual(options.base_options.model_asset_path, str(model_path))
        self.assertIs(options.base_options.delegate, cpu_delegate)
        self.assertIs(options.running_mode, image_mode)
        self.assertFalse(options.output_segmentation_masks)
        estimator.close()


if __name__ == "__main__":
    unittest.main()
