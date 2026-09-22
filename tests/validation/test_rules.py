"""요구사항 §2 의 PASS/FAIL 판정 기준을 검증한다. 실제 모델은 쓰지 않는다."""

import unittest

from app.validation.models import Detection, DetectionResult, PoseDirection, ValidationStatus
from app.validation.rules import decide

_BOX = (0.0, 0.0, 10.0, 10.0)


def _detection(label: str, score: float = 0.9) -> Detection:
    return Detection(label=label, score=score, box=_BOX)


def _result(persons: int, garments: int) -> DetectionResult:
    return DetectionResult(
        persons=[_detection("person") for _ in range(persons)],
        garments=[_detection("shirt") for _ in range(garments)],
    )


class NoPersonTests(unittest.TestCase):
    def test_exactly_one_garment_passes(self):
        outcome = decide(_result(persons=0, garments=1))
        self.assertEqual(outcome.status, ValidationStatus.PASS)

    def test_no_garment_fails_with_garment_not_found(self):
        outcome = decide(_result(persons=0, garments=0))
        self.assertEqual(outcome.status, ValidationStatus.FAIL)
        self.assertEqual(outcome.reason.value, "GARMENT_NOT_FOUND")

    def test_multiple_garments_fail(self):
        outcome = decide(_result(persons=0, garments=2))
        self.assertEqual(outcome.status, ValidationStatus.FAIL)
        self.assertEqual(outcome.reason.value, "MULTIPLE_GARMENTS")


class MultiplePersonsTests(unittest.TestCase):
    def test_two_or_more_persons_fail_regardless_of_garments(self):
        outcome = decide(_result(persons=2, garments=1))
        self.assertEqual(outcome.status, ValidationStatus.FAIL)
        self.assertEqual(outcome.reason.value, "MULTIPLE_PERSONS")


class SinglePersonPoseTests(unittest.TestCase):
    def test_front_passes(self):
        outcome = decide(_result(persons=1, garments=1), PoseDirection.FRONT)
        self.assertEqual(outcome.status, ValidationStatus.PASS)

    def test_side_allowed_passes(self):
        outcome = decide(_result(persons=1, garments=1), PoseDirection.SIDE_ALLOWED)
        self.assertEqual(outcome.status, ValidationStatus.PASS)

    def test_back_fails_with_back_view(self):
        outcome = decide(_result(persons=1, garments=1), PoseDirection.BACK)
        self.assertEqual(outcome.status, ValidationStatus.FAIL)
        self.assertEqual(outcome.reason.value, "BACK_VIEW")

    def test_side_angle_too_large_fails(self):
        outcome = decide(_result(persons=1, garments=1), PoseDirection.SIDE_ANGLE_TOO_LARGE)
        self.assertEqual(outcome.status, ValidationStatus.FAIL)
        self.assertEqual(outcome.reason.value, "SIDE_ANGLE_TOO_LARGE")

    def test_uncertain_fails_not_passes(self):
        outcome = decide(_result(persons=1, garments=1), PoseDirection.UNCERTAIN)
        self.assertEqual(outcome.status, ValidationStatus.FAIL)
        self.assertEqual(outcome.reason.value, "UNCERTAIN")

    def test_missing_pose_direction_is_treated_as_uncertain(self):
        outcome = decide(_result(persons=1, garments=1), None)
        self.assertEqual(outcome.status, ValidationStatus.FAIL)
        self.assertEqual(outcome.reason.value, "UNCERTAIN")


if __name__ == "__main__":
    unittest.main()
