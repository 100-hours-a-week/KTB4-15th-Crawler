import csv
import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import scripts.test_landmark_quality_gate as quality_script
from app.validation.models import Detection, DetectionResult
from app.validation.pose import BodyLandmarks


def _candidate(code: int, category: str = "상의") -> quality_script.Candidate:
    return quality_script.Candidate(
        product_code=code,
        product_name=f"상품 {code}",
        main_category=category,
        sub_category="스웨트셔츠" if category == "상의" else "슬림 팬츠",
        image_url=f"https://example.com/{code}.jpg",
    )


def _landmarks(*, hip_visibility: float = 0.9) -> BodyLandmarks:
    return BodyLandmarks(
        nose_visibility=0.9,
        left_shoulder_x=0.65,
        left_shoulder_visibility=0.9,
        right_shoulder_x=0.35,
        right_shoulder_visibility=0.9,
        shoulder_mid_y=0.3,
        hip_mid_y=0.7,
        left_hip_visibility=hip_visibility,
        right_hip_visibility=hip_visibility,
        left_knee_visibility=0.9,
        right_knee_visibility=0.9,
        left_ankle_visibility=0.9,
        right_ankle_visibility=0.9,
    )


def _detection(persons: int, garments: int) -> DetectionResult:
    box = (0.0, 0.0, 10.0, 10.0)
    return DetectionResult(
        persons=[Detection("person", 0.9, box) for _ in range(persons)],
        garments=[Detection("shirt", 0.9, box) for _ in range(garments)],
    )


class FakeDetector:
    def __init__(self, results):
        self.results = iter(results)

    def detect(self, image, labels):
        return next(self.results)


class FakePoseEstimator:
    def __init__(self, results):
        self.results = iter(results)
        self.calls = []

    def get_landmarks(self, image):
        self.calls.append(image)
        return next(self.results)


class CandidateSamplingTests(unittest.TestCase):
    def test_randomized_order_is_reproducible_and_seed_sensitive(self):
        candidates = [_candidate(code) for code in range(20)]

        first = quality_script._randomized_candidates(
            candidates, seed=42, main_category="상의"
        )
        repeated = quality_script._randomized_candidates(
            candidates, seed=42, main_category="상의"
        )
        different = quality_script._randomized_candidates(
            candidates, seed=43, main_category="상의"
        )

        self.assertEqual(first, repeated)
        self.assertNotEqual(first, different)

    def test_selection_uses_counts_and_landmark_presence_not_gate_result(self):
        candidates = [_candidate(code) for code in (1, 2, 3, 4)]
        detector = FakeDetector(
            [
                _detection(persons=0, garments=1),
                _detection(persons=1, garments=1),
                _detection(persons=1, garments=1),
                _detection(persons=1, garments=1),
            ]
        )
        bad_landmarks = _landmarks(hip_visibility=0.1)
        pose_estimator = FakePoseEstimator([None, bad_landmarks, _landmarks()])

        rows = quality_script._collect_category(
            candidates,
            target=2,
            detector=detector,
            pose_estimator=pose_estimator,
            image_loader=lambda url: url,
        )

        self.assertEqual([row["product_code"] for row in rows], [3, 4])
        self.assertEqual([row["landmark_reliable"] for row in rows], [False, True])
        self.assertEqual(
            pose_estimator.calls,
            ["https://example.com/2.jpg", "https://example.com/3.jpg", "https://example.com/4.jpg"],
        )


class CsvTests(unittest.TestCase):
    def test_write_collection_has_required_columns_and_blank_human_fields(self):
        row = quality_script._csv_row(_candidate(1), _landmarks())
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = quality_script._write_collection(
                [row], collect=1, seed=42, results_dir=Path(temporary_directory)
            )
            with path.open(newline="", encoding="utf-8-sig") as csv_file:
                reader = csv.DictReader(csv_file)
                written = list(reader)

        self.assertEqual(tuple(reader.fieldnames), quality_script.CSV_COLUMNS)
        self.assertEqual(written[0]["human_expected_reliable"], "")
        self.assertEqual(written[0]["human_note"], "")


class AnalyzeTests(unittest.TestCase):
    def _write_csv(self, directory: str, labels) -> Path:
        path = Path(directory) / "quality.csv"
        with path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=quality_script.CSV_COLUMNS)
            writer.writeheader()
            for code, human, gate in labels:
                row = quality_script._csv_row(_candidate(code), _landmarks())
                row["human_expected_reliable"] = human
                row["landmark_reliable"] = gate
                writer.writerow(row)
        return path

    def test_analyze_calculates_confusion_matrix_metrics_and_mismatches(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = self._write_csv(
                temporary_directory,
                [(1, "YES", True), (2, "NO", False), (3, "NO", True), (4, "YES", False)],
            )

            result = quality_script.analyze(path)

        self.assertEqual((result.tp, result.tn, result.fp, result.fn), (1, 1, 1, 1))
        self.assertEqual(
            (result.accuracy, result.precision, result.recall, result.specificity),
            (0.5, 0.5, 0.5, 0.5),
        )
        self.assertEqual(
            [(row["outcome"], row["product_code"]) for row in result.mismatches],
            [("FP", "3"), ("FN", "4")],
        )

        output = io.StringIO()
        with redirect_stdout(output):
            quality_script._print_analysis(result)
        self.assertIn("FP product_code=3", output.getvalue())
        self.assertIn("torso_scale=", output.getvalue())

    def test_analyze_rejects_unlabeled_rows(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = self._write_csv(temporary_directory, [(1, "", True)])

            with self.assertRaisesRegex(ValueError, "human_expected_reliable"):
                quality_script.analyze(path)


if __name__ == "__main__":
    unittest.main()
