"""MediaPipe Pose Landmarker 로 사람의 앞/옆/뒤 방향을 판정한다.

https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker/python
"""

from dataclasses import dataclass
from typing import Self

from app.validation.models import PoseDirection

# 표준 BlazePose 33-point 토폴로지의 landmark 인덱스.
_NOSE = 0
_LEFT_SHOULDER = 11
_RIGHT_SHOULDER = 12
_LEFT_HIP = 23
_RIGHT_HIP = 24
_LEFT_KNEE = 25
_RIGHT_KNEE = 26
_LEFT_ANKLE = 27
_RIGHT_ANKLE = 28

# 실제 이미지로 조정이 필요한 값들(요구사항 §5, §21).
MIN_LANDMARK_VISIBILITY = 0.5
FRONT_ROTATION_RATIO = 0.35
SIDE_ROTATION_RATIO = 0.10
MIN_TORSO_SCALE = 0.02  # 정규화 좌표 기준. 이보다 작으면 landmark 가 뭉쳐 있다고 본다.

# https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task
DEFAULT_MODEL_PATH = "models/pose_landmarker_full.task"


class PoseModelNotFoundError(RuntimeError):
    """pose_landmarker.task 모델 파일이 없을 때 발생한다."""


@dataclass(frozen=True)
class BodyLandmarks:
    """방향 판정에 필요한 landmark 좌표/visibility 만 담는다."""

    nose_visibility: float
    left_shoulder_x: float
    left_shoulder_visibility: float
    right_shoulder_x: float
    right_shoulder_visibility: float
    shoulder_mid_y: float
    hip_mid_y: float
    # 하의 상품에서 상체가 잘린 경우의 fallback 에만 쓴다(§ has_sufficient_lower_body_landmarks).
    left_hip_visibility: float
    right_hip_visibility: float
    left_knee_visibility: float
    right_knee_visibility: float
    left_ankle_visibility: float
    right_ankle_visibility: float


def extract_landmarks(pose_landmarker_result) -> BodyLandmarks | None:
    """MediaPipe PoseLandmarkerResult 에서 방향 판정에 필요한 값만 뽑는다.

    사람이 감지되지 않았으면 None 을 반환한다. 여러 명이 감지돼도 첫 번째만
    쓴다(여러 명인 경우는 Grounding DINO person 개수로 이미 걸러진다).
    """
    if not pose_landmarker_result.pose_landmarks:
        return None
    landmarks = pose_landmarker_result.pose_landmarks[0]
    nose = landmarks[_NOSE]
    left_shoulder = landmarks[_LEFT_SHOULDER]
    right_shoulder = landmarks[_RIGHT_SHOULDER]
    left_hip = landmarks[_LEFT_HIP]
    right_hip = landmarks[_RIGHT_HIP]
    left_knee = landmarks[_LEFT_KNEE]
    right_knee = landmarks[_RIGHT_KNEE]
    left_ankle = landmarks[_LEFT_ANKLE]
    right_ankle = landmarks[_RIGHT_ANKLE]
    return BodyLandmarks(
        nose_visibility=nose.visibility,
        left_shoulder_x=left_shoulder.x,
        left_shoulder_visibility=left_shoulder.visibility,
        right_shoulder_x=right_shoulder.x,
        right_shoulder_visibility=right_shoulder.visibility,
        shoulder_mid_y=(left_shoulder.y + right_shoulder.y) / 2,
        hip_mid_y=(left_hip.y + right_hip.y) / 2,
        left_hip_visibility=left_hip.visibility,
        right_hip_visibility=right_hip.visibility,
        left_knee_visibility=left_knee.visibility,
        right_knee_visibility=right_knee.visibility,
        left_ankle_visibility=left_ankle.visibility,
        right_ankle_visibility=right_ankle.visibility,
    )


def has_sufficient_upper_body_landmarks(landmarks: BodyLandmarks) -> bool:
    """어깨가 둘 다 보여야 기존 rotation_ratio 판정을 신뢰할 수 있다고 본다."""
    return (
        landmarks.left_shoulder_visibility >= MIN_LANDMARK_VISIBILITY
        and landmarks.right_shoulder_visibility >= MIN_LANDMARK_VISIBILITY
    )


def has_sufficient_lower_body_landmarks(landmarks: BodyLandmarks) -> bool:
    """골반이 둘 다 보이고, 무릎이나 발목 중 하나라도 보이면 하체가 찍힌 것으로 본다."""
    hips_visible = (
        landmarks.left_hip_visibility >= MIN_LANDMARK_VISIBILITY
        and landmarks.right_hip_visibility >= MIN_LANDMARK_VISIBILITY
    )
    legs_visible = (
        landmarks.left_knee_visibility >= MIN_LANDMARK_VISIBILITY
        or landmarks.right_knee_visibility >= MIN_LANDMARK_VISIBILITY
        or landmarks.left_ankle_visibility >= MIN_LANDMARK_VISIBILITY
        or landmarks.right_ankle_visibility >= MIN_LANDMARK_VISIBILITY
    )
    return hips_visible and legs_visible


def classify_direction(
    landmarks: BodyLandmarks | None, *, allow_lower_body_fallback: bool = False
) -> PoseDirection:
    """어깨/골반/코 landmark 로 방향을 판정한다.

    원리: MediaPipe 는 landmark 이름을 사람 기준 좌우로 붙인다. 사람이 카메라를
    정면으로 보면 자신의 왼쪽 어깨가 화면 오른쪽에 찍히고, 뒷모습이면 반대로
    화면 왼쪽에 찍힌다. 몸을 옆으로 돌릴수록 두 어깨의 화면 x 좌표 차이가
    몸통 길이(어깨~골반) 대비 0 에 가까워진다. 이 부호와 크기로 방향을
    추정한다. threshold 는 실제 이미지로 조정해야 한다(요구사항 §21).

    allow_lower_body_fallback: 하의 상품처럼 상반신이 잘려 있어도 정상일 수 있는
    경우에만 True 로 준다. 어깨가 충분히 보이면 이 값과 무관하게 항상 위 로직을
    그대로 쓴다. 어깨가 부족할 때만, 이 값이 True 이고 골반/다리가 충분히 보이면
    FRONT(=PASS)로, 그것도 부족하면 UNCERTAIN 으로 판정한다.
    """
    if landmarks is None:
        return PoseDirection.UNCERTAIN

    if not has_sufficient_upper_body_landmarks(landmarks):
        if allow_lower_body_fallback and has_sufficient_lower_body_landmarks(landmarks):
            return PoseDirection.FRONT
        return PoseDirection.UNCERTAIN

    torso_scale = abs(landmarks.hip_mid_y - landmarks.shoulder_mid_y)
    if torso_scale < MIN_TORSO_SCALE:
        return PoseDirection.UNCERTAIN

    rotation_ratio = (landmarks.left_shoulder_x - landmarks.right_shoulder_x) / torso_scale

    if rotation_ratio >= FRONT_ROTATION_RATIO:
        direction = PoseDirection.FRONT
    elif rotation_ratio >= SIDE_ROTATION_RATIO:
        direction = PoseDirection.SIDE_ALLOWED
    elif rotation_ratio <= -FRONT_ROTATION_RATIO:
        direction = PoseDirection.BACK
    else:
        direction = PoseDirection.SIDE_ANGLE_TOO_LARGE

    # 정면/허용 측면으로 판정됐는데 얼굴이 거의 보이지 않으면 모순이므로 신뢰하지 않는다.
    if (
        direction in (PoseDirection.FRONT, PoseDirection.SIDE_ALLOWED)
        and landmarks.nose_visibility < MIN_LANDMARK_VISIBILITY
    ):
        return PoseDirection.UNCERTAIN

    return direction


class PoseEstimator:
    """MediaPipe Pose Landmarker 를 한 번만 초기화하고 여러 이미지에 재사용한다."""

    def __init__(self, *, model_path: str = DEFAULT_MODEL_PATH) -> None:
        from pathlib import Path

        if not Path(model_path).exists():
            raise PoseModelNotFoundError(
                f"pose landmarker 모델 파일을 찾을 수 없습니다: {model_path}. "
                "https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker#models "
                "에서 내려받아 해당 경로에 두세요."
            )

        # 무거운 의존성은 실제 사용 시점에만 필요하므로 여기서 import 한다.
        import mediapipe as mp

        self._mp = mp
        base_options = mp.tasks.BaseOptions(
            model_asset_path=model_path,
            delegate=mp.tasks.BaseOptions.Delegate.CPU,
        )
        options = mp.tasks.vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            output_segmentation_masks=False,
        )
        self._landmarker = mp.tasks.vision.PoseLandmarker.create_from_options(options)

    def get_landmarks(self, image_rgb) -> BodyLandmarks | None:
        """classify_direction 을 거치기 전의 원본 landmark 값이다. 디버깅 출력에 쓴다.

        estimate_direction() 도 내부적으로 이 메서드를 쓴다. 동작은 그대로이고,
        추출 단계를 별도로 호출할 수 있게 이름만 붙인 것이다.
        """
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=image_rgb)
        result = self._landmarker.detect(mp_image)
        return extract_landmarks(result)

    def estimate_direction(
        self, image_rgb, *, allow_lower_body_fallback: bool = False
    ) -> PoseDirection:
        """image_rgb 는 HxWx3 RGB numpy 배열이다."""
        return classify_direction(
            self.get_landmarks(image_rgb), allow_lower_body_fallback=allow_lower_body_fallback
        )

    def close(self) -> None:
        self._landmarker.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
