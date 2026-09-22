"""Grounding DINO / MediaPipe 결과를 받아 최종 PASS/FAIL 을 판단한다."""

from app.validation.models import (
    DetectionResult,
    PoseDirection,
    ValidationReason,
    ValidationResult,
    ValidationStatus,
)

_PASS = ValidationResult(ValidationStatus.PASS, ValidationReason.VALID)

_POSE_REASON = {
    PoseDirection.FRONT: None,
    PoseDirection.SIDE_ALLOWED: None,
    PoseDirection.BACK: ValidationReason.BACK_VIEW,
    PoseDirection.SIDE_ANGLE_TOO_LARGE: ValidationReason.SIDE_ANGLE_TOO_LARGE,
    PoseDirection.UNCERTAIN: ValidationReason.UNCERTAIN,
}


def decide(
    detection: DetectionResult, pose_direction: PoseDirection | None = None
) -> ValidationResult:
    """요구사항 §2 의 판정 기준을 그대로 옮긴 것이다.

    person_count == 1 일 때만 pose_direction 을 사용한다. 그 외에는 무시한다.
    """
    if detection.person_count == 0:
        if detection.garment_count == 1:
            return _PASS
        if detection.garment_count == 0:
            return ValidationResult(ValidationStatus.FAIL, ValidationReason.GARMENT_NOT_FOUND)
        return ValidationResult(ValidationStatus.FAIL, ValidationReason.MULTIPLE_GARMENTS)

    if detection.person_count >= 2:
        return ValidationResult(ValidationStatus.FAIL, ValidationReason.MULTIPLE_PERSONS)

    # person_count == 1
    if pose_direction is None:
        pose_direction = PoseDirection.UNCERTAIN
    reason = _POSE_REASON[pose_direction]
    if reason is None:
        return _PASS
    return ValidationResult(ValidationStatus.FAIL, reason)
