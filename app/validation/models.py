"""상품 이미지 검증에서 쓰는 데이터 구조."""

from dataclasses import dataclass
from enum import Enum


class PoseDirection(Enum):
    """MediaPipe landmark 로 판정한 사람의 몸 방향."""

    FRONT = "FRONT"
    SIDE_ALLOWED = "SIDE_ALLOWED"
    SIDE_ANGLE_TOO_LARGE = "SIDE_ANGLE_TOO_LARGE"
    BACK = "BACK"
    UNCERTAIN = "UNCERTAIN"


class ValidationStatus(Enum):
    PENDING = "PENDING"
    PASS = "PASS"
    FAIL = "FAIL"


class ValidationReason(Enum):
    VALID = "VALID"
    GARMENT_NOT_FOUND = "GARMENT_NOT_FOUND"
    MULTIPLE_GARMENTS = "MULTIPLE_GARMENTS"
    MULTIPLE_PERSONS = "MULTIPLE_PERSONS"
    BACK_VIEW = "BACK_VIEW"
    SIDE_ANGLE_TOO_LARGE = "SIDE_ANGLE_TOO_LARGE"
    UNCERTAIN = "UNCERTAIN"
    IMAGE_LOAD_FAILED = "IMAGE_LOAD_FAILED"
    CATEGORY_MAPPING_NOT_FOUND = "CATEGORY_MAPPING_NOT_FOUND"


@dataclass(frozen=True)
class Detection:
    """탐지된 객체 하나. box 는 (x0, y0, x1, y1) 픽셀 좌표다."""

    label: str
    score: float
    box: tuple[float, float, float, float]


@dataclass(frozen=True)
class DetectionResult:
    """Grounding DINO 탐지 결과. 동의어 label 중복 제거까지 끝난 상태다."""

    persons: list[Detection]
    garments: list[Detection]

    @property
    def person_count(self) -> int:
        return len(self.persons)

    @property
    def garment_count(self) -> int:
        return len(self.garments)


@dataclass(frozen=True)
class ValidationResult:
    status: ValidationStatus
    reason: ValidationReason

    @property
    def passed(self) -> bool:
        return self.status is ValidationStatus.PASS
