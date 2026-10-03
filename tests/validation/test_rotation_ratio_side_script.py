import csv
import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import numpy as np

import scripts.test_rotation_ratio_side as rotation_script
from app.validation.models import Detection, DetectionResult, PoseDirection
from app.validation.pose import BodyLandmarks, is_direction_landmark_reliable


def _candidate(code: int) -> rotation_script.Candidate:
    return rotation_script.Candidate(
        product_code=code,
        product_name=f"상품 {code}",
        main_category="상의",
        sub_category="스웨트셔츠",
        image_url=f"https://example.com/{code}.jpg",
    )


def _landmarks(
    *, shoulder_dx: float = 0.15, hip_visibility: float = 0.9
) -> BodyLandmarks:
    return BodyLandmarks(
        nose_visibility=0.9,
        left_shoulder_x=0.5 + shoulder_dx,
        left_shoulder_visibility=0.9,
        right_shoulder_x=0.5,
        right_shoulder_visibility=0.9,
        shoulder_mid_y=0.3,
        hip_mid_y=0.6,
        left_hip_visibility=hip_visibility,
        right_hip_visibility=hip_visibility,
        left_knee_visibility=0.9,
        right_knee_visibility=0.9,
        left_ankle_visibility=0.9,
        right_ankle_visibility=0.9,
    )


def _detection(persons: int = 1, garments: int = 1) -> DetectionResult:
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


class FakeInputQueue:
    """사람의 응답을 미리 정해둔 순서대로 돌려준다."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0

    def __call__(self, prompt=""):
        self.calls += 1
        if not self.responses:
            raise AssertionError("입력이 더 필요한데 준비된 응답이 없습니다")
        return self.responses.pop(0)


def _forbidden_input(prompt=""):
    raise AssertionError("Quality Gate를 통과하지 못한 후보는 사람에게 물어보면 안 된다")


class _RaisingInput:
    """첫 호출에서 바로 Ctrl-C(KeyboardInterrupt)를 흉내낸다."""

    def __call__(self, prompt=""):
        raise KeyboardInterrupt()


class CollectionTests(unittest.TestCase):
    def test_candidate_order_is_seeded_and_reproducible(self):
        candidates = [_candidate(code) for code in range(20)]

        first = rotation_script._randomized_candidates(candidates, seed=42)
        repeated = rotation_script._randomized_candidates(candidates, seed=42)
        different = rotation_script._randomized_candidates(candidates, seed=43)

        self.assertEqual(first, repeated)
        self.assertNotEqual(first, different)

    def test_uses_the_production_quality_gate_function(self):
        self.assertIs(
            rotation_script.is_direction_landmark_reliable,
            is_direction_landmark_reliable,
        )

    def test_quality_true_is_collected_with_one_mediapipe_call(self):
        pose_estimator = FakePoseEstimator([_landmarks()])

        rows, summary = rotation_script._collect_rows(
            [_candidate(1)],
            target=1,
            detector=FakeDetector([_detection()]),
            pose_estimator=pose_estimator,
            image_loader=lambda url: url,
        )

        self.assertEqual([row["product_code"] for row in rows], [1])
        self.assertEqual(rows[0]["landmark_reliable"], True)
        self.assertEqual(pose_estimator.calls, ["https://example.com/1.jpg"])
        self.assertEqual(summary["landmark_quality_bad"], 0)

    def test_quality_false_is_excluded_and_counted_without_classification(self):
        bad_landmarks = _landmarks(hip_visibility=0.1)
        good_landmarks = _landmarks(shoulder_dx=-0.15)
        pose_estimator = FakePoseEstimator([bad_landmarks, good_landmarks])

        with patch.object(
            rotation_script,
            "classify_direction",
            return_value=PoseDirection.BACK,
        ) as classify:
            rows, summary = rotation_script._collect_rows(
                [_candidate(1), _candidate(2)],
                target=1,
                detector=FakeDetector([_detection(), _detection()]),
                pose_estimator=pose_estimator,
                image_loader=lambda url: url,
            )

        self.assertEqual([row["product_code"] for row in rows], [2])
        self.assertEqual(summary["landmark_quality_bad"], 1)
        self.assertEqual(len(pose_estimator.calls), 2)
        classify.assert_called_once_with(good_landmarks)

    def test_sampling_does_not_filter_by_direction_or_rotation_ratio(self):
        zero_ratio_landmarks = _landmarks(shoulder_dx=0.0)

        with patch.object(
            rotation_script,
            "classify_direction",
            return_value=PoseDirection.SIDE_ANGLE_TOO_LARGE,
        ):
            rows, _ = rotation_script._collect_rows(
                [_candidate(1)],
                target=1,
                detector=FakeDetector([_detection()]),
                pose_estimator=FakePoseEstimator([zero_ratio_landmarks]),
                image_loader=lambda url: url,
            )

        self.assertEqual(len(rows), 1)
        self.assertEqual(float(rows[0]["rotation_ratio"]), 0.0)
        self.assertEqual(
            rows[0]["current_mediapipe_direction"], "SIDE_ANGLE_TOO_LARGE"
        )


class ContactSheetTests(unittest.TestCase):
    def test_only_quality_gate_true_reaches_contact_sheet_callback(self):
        accepted_codes = []
        bad_landmarks = _landmarks(hip_visibility=0.1)
        good_landmarks = _landmarks()
        pose_estimator = FakePoseEstimator([bad_landmarks, good_landmarks])

        rows, summary = rotation_script._collect_rows(
            [_candidate(1), _candidate(2)],
            target=1,
            detector=FakeDetector([_detection(), _detection()]),
            pose_estimator=pose_estimator,
            image_loader=lambda url: np.zeros((20, 10, 3), dtype=np.uint8),
            on_accept=lambda candidate, image, landmarks, row: accepted_codes.append(
                candidate.product_code
            ),
        )

        self.assertEqual(accepted_codes, [2])
        self.assertEqual([row["product_code"] for row in rows], [2])
        self.assertEqual(summary["landmark_quality_bad"], 1)
        self.assertEqual(len(pose_estimator.calls), 2)

    def test_contact_cell_labels_do_not_expose_model_results(self):
        candidate = _candidate(123456)
        image = np.zeros((40, 20, 3), dtype=np.uint8)

        with patch.object(
            rotation_script.cv2,
            "putText",
            wraps=rotation_script.cv2.putText,
        ) as put_text:
            cell = rotation_script._build_contact_cell(candidate, image)

        displayed_text = [call.args[1] for call in put_text.call_args_list]
        self.assertEqual(displayed_text, ["123456", "TOP"])
        self.assertEqual(
            cell.shape,
            (
                rotation_script.CONTACT_CELL_HEIGHT,
                rotation_script.CONTACT_CELL_WIDTH,
                3,
            ),
        )

    def test_one_hundred_items_with_25_per_sheet_writes_four_sheets_and_mapping_csv(self):
        cell = np.zeros(
            (
                rotation_script.CONTACT_CELL_HEIGHT,
                rotation_script.CONTACT_CELL_WIDTH,
                3,
            ),
            dtype=np.uint8,
        )
        items = []
        for code in range(100):
            row = rotation_script._csv_row(_candidate(code), _landmarks())
            items.append(rotation_script.ContactSheetItem(row=row, cell_rgb=cell))

        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory)
            sheet_paths = rotation_script._write_contact_sheets(
                items,
                per_sheet=25,
                output_dir=output_dir / "sheets",
            )
            csv_path = rotation_script._write_contact_mapping(
                items,
                per_sheet=25,
                seed=44,
                sample_count=100,
                output_dir=output_dir,
            )
            with csv_path.open(newline="", encoding="utf-8-sig") as csv_file:
                mapping_rows = list(csv.DictReader(csv_file))

            self.assertEqual(len(sheet_paths), 4)
            self.assertTrue(all(path.exists() for path in sheet_paths))
            self.assertEqual([path.name for path in sheet_paths], [
                "sheet_01.jpg",
                "sheet_02.jpg",
                "sheet_03.jpg",
                "sheet_04.jpg",
            ])
            self.assertEqual(len(mapping_rows), 100)
            self.assertEqual(mapping_rows[0]["sheet"], "sheet_01.jpg")
            self.assertEqual(mapping_rows[0]["position"], "1")
            self.assertEqual(mapping_rows[25]["sheet"], "sheet_02.jpg")
            self.assertEqual(mapping_rows[25]["position"], "1")
            self.assertEqual(
                tuple(mapping_rows[0]), rotation_script.CONTACT_CSV_COLUMNS
            )

    def test_per_sheet_must_be_between_20_and_25(self):
        with self.assertRaises(ValueError):
            rotation_script.create_contact_sheets(100, seed=44, per_sheet=19)
        with self.assertRaises(ValueError):
            rotation_script.create_contact_sheets(100, seed=44, per_sheet=26)


class SideTooLargeCollectionTests(unittest.TestCase):
    def test_side_too_large_label_is_collected_and_stops_at_target(self):
        candidates = [_candidate(1), _candidate(2)]
        pose_estimator = FakePoseEstimator([_landmarks(), _landmarks()])
        input_fn = FakeInputQueue(["SIDE_TOO_LARGE"])

        rows, summary = rotation_script._collect_side_too_large_rows(
            candidates,
            target=1,
            detector=FakeDetector([_detection()]),
            pose_estimator=pose_estimator,
            image_loader=lambda url: url,
            input_fn=input_fn,
        )

        self.assertEqual([row["product_code"] for row in rows], [1])
        self.assertEqual(rows[0]["human_side_group"], "SIDE_TOO_LARGE")
        self.assertEqual(summary["side_too_large_collected"], 1)
        self.assertEqual(summary["quality_gate_passed"], 1)
        self.assertEqual(summary["scanned"], 1)  # 목표 도달 후 두 번째 후보는 건드리지 않는다.
        self.assertEqual(len(pose_estimator.calls), 1)

    def test_skip_label_is_not_collected_but_counted(self):
        candidates = [_candidate(1), _candidate(2)]
        pose_estimator = FakePoseEstimator([_landmarks(), _landmarks()])
        input_fn = FakeInputQueue(["SKIP", "SIDE_TOO_LARGE"])

        rows, summary = rotation_script._collect_side_too_large_rows(
            candidates,
            target=1,
            detector=FakeDetector([_detection(), _detection()]),
            pose_estimator=pose_estimator,
            image_loader=lambda url: url,
            input_fn=input_fn,
        )

        self.assertEqual([row["product_code"] for row in rows], [2])
        self.assertEqual(summary["skipped"], 1)
        self.assertEqual(summary["side_too_large_collected"], 1)
        self.assertEqual(summary["scanned"], 2)

    def test_exclude_label_is_not_collected_but_counted(self):
        candidates = [_candidate(1), _candidate(2)]
        pose_estimator = FakePoseEstimator([_landmarks(), _landmarks()])
        input_fn = FakeInputQueue(["EXCLUDE", "SIDE_TOO_LARGE"])

        rows, summary = rotation_script._collect_side_too_large_rows(
            candidates,
            target=1,
            detector=FakeDetector([_detection(), _detection()]),
            pose_estimator=pose_estimator,
            image_loader=lambda url: url,
            input_fn=input_fn,
        )

        self.assertEqual([row["product_code"] for row in rows], [2])
        self.assertEqual(summary["excluded"], 1)
        self.assertEqual(summary["side_too_large_collected"], 1)

    def test_invalid_input_is_reprompted(self):
        pose_estimator = FakePoseEstimator([_landmarks()])
        input_fn = FakeInputQueue(["front", "  side_too_large  "])

        rows, _summary = rotation_script._collect_side_too_large_rows(
            [_candidate(1)],
            target=1,
            detector=FakeDetector([_detection()]),
            pose_estimator=pose_estimator,
            image_loader=lambda url: url,
            input_fn=input_fn,
        )

        self.assertEqual(len(rows), 1)
        self.assertEqual(input_fn.calls, 2)

    def test_person_count_not_one_never_prompts_human(self):
        rows, summary = rotation_script._collect_side_too_large_rows(
            [_candidate(1)],
            target=1,
            detector=FakeDetector([_detection(persons=0)]),
            pose_estimator=FakePoseEstimator([]),
            image_loader=lambda url: url,
            input_fn=_forbidden_input,
        )

        self.assertEqual(rows, [])
        self.assertEqual(summary["person_count_not_one"], 1)
        self.assertEqual(summary["quality_gate_passed"], 0)

    def test_landmark_quality_bad_never_prompts_human(self):
        bad_landmarks = _landmarks(hip_visibility=0.1)

        rows, summary = rotation_script._collect_side_too_large_rows(
            [_candidate(1)],
            target=1,
            detector=FakeDetector([_detection()]),
            pose_estimator=FakePoseEstimator([bad_landmarks]),
            image_loader=lambda url: url,
            input_fn=_forbidden_input,
        )

        self.assertEqual(rows, [])
        self.assertEqual(summary["landmark_quality_bad"], 1)
        self.assertEqual(summary["quality_gate_passed"], 0)

    def test_does_not_use_rotation_ratio_or_direction_for_gating(self):
        # shoulder_dx=0.0 이면 rotation_ratio=0 이라 SIDE_ANGLE_TOO_LARGE 로 분류될
        # 값이지만, 게이트/사람 응답과는 무관하게 그대로 기록만 돼야 한다.
        zero_ratio_landmarks = _landmarks(shoulder_dx=0.0)
        input_fn = FakeInputQueue(["SIDE_TOO_LARGE"])

        with patch.object(
            rotation_script,
            "classify_direction",
            return_value=PoseDirection.FRONT,
        ) as classify:
            rows, _summary = rotation_script._collect_side_too_large_rows(
                [_candidate(1)],
                target=1,
                detector=FakeDetector([_detection()]),
                pose_estimator=FakePoseEstimator([zero_ratio_landmarks]),
                image_loader=lambda url: url,
                input_fn=input_fn,
            )

        self.assertEqual(len(rows), 1)
        self.assertEqual(float(rows[0]["rotation_ratio"]), 0.0)
        self.assertEqual(rows[0]["current_mediapipe_direction"], "FRONT")
        self.assertEqual(rows[0]["human_side_group"], "SIDE_TOO_LARGE")
        classify.assert_called_once_with(zero_ratio_landmarks)

    def test_keyboard_interrupt_saves_whatever_was_collected_so_far(self):
        candidates = [_candidate(1), _candidate(2)]
        pose_estimator = FakePoseEstimator([_landmarks(), _landmarks()])
        input_fn = FakeInputQueue(["SIDE_TOO_LARGE"])
        responses = iter([input_fn, _RaisingInput()])

        def dispatching_input(prompt=""):
            return next(responses)(prompt)

        rows, summary = rotation_script._collect_side_too_large_rows(
            candidates,
            target=5,
            detector=FakeDetector([_detection(), _detection()]),
            pose_estimator=pose_estimator,
            image_loader=lambda url: url,
            input_fn=dispatching_input,
        )

        self.assertEqual(len(rows), 1)  # 두 번째 후보는 Ctrl-C 로 끊겼다.
        self.assertEqual(summary["side_too_large_collected"], 1)

    def test_running_out_of_candidates_does_not_raise(self):
        pose_estimator = FakePoseEstimator([_landmarks()])
        input_fn = FakeInputQueue(["SKIP"])

        rows, summary = rotation_script._collect_side_too_large_rows(
            [_candidate(1)],
            target=5,
            detector=FakeDetector([_detection()]),
            pose_estimator=pose_estimator,
            image_loader=lambda url: url,
            input_fn=input_fn,
        )

        self.assertEqual(rows, [])
        self.assertEqual(summary["side_too_large_collected"], 0)
        self.assertEqual(summary["skipped"], 1)

    def test_write_side_too_large_collection_uses_the_expected_filename(self):
        row = {column: "" for column in rotation_script.CSV_COLUMNS}
        row["product_code"] = 1
        with tempfile.TemporaryDirectory() as temporary_directory:
            results_dir = Path(temporary_directory)

            output_path = rotation_script._write_side_too_large_collection(
                [row], target=15, seed=42, results_dir=results_dir
            )

            self.assertEqual(output_path, results_dir / "side_too_large_seed42_n15.csv")
            with output_path.open(newline="", encoding="utf-8-sig") as csv_file:
                written_rows = list(csv.DictReader(csv_file))
        self.assertEqual(written_rows[0]["product_code"], "1")

    def test_collect_side_too_large_raises_for_non_positive_target(self):
        with self.assertRaises(ValueError):
            rotation_script.collect_side_too_large(0, seed=42)


class ParseArgsAndMainTests(unittest.TestCase):
    def test_contact_sheet_arguments_are_parsed(self):
        args = rotation_script._parse_args(
            ["--contact-sheet", "100", "--seed", "44", "--per-sheet", "25"]
        )

        self.assertEqual(args.contact_sheet, 100)
        self.assertEqual(args.seed, 44)
        self.assertEqual(args.per_sheet, 25)

    def test_collect_side_too_large_is_mutually_exclusive_with_collect(self):
        with self.assertRaises(SystemExit):
            rotation_script._parse_args(
                ["--collect", "1", "--collect-side-too-large", "1"]
            )

    def test_collect_side_too_large_is_mutually_exclusive_with_analyze(self):
        with self.assertRaises(SystemExit):
            rotation_script._parse_args(
                ["--analyze", "foo.csv", "--collect-side-too-large", "1"]
            )

    def test_collect_side_too_large_parses_as_int(self):
        args = rotation_script._parse_args(["--collect-side-too-large", "15", "--seed", "7"])
        self.assertEqual(args.collect_side_too_large, 15)
        self.assertEqual(args.seed, 7)
        self.assertIsNone(args.collect)
        self.assertIsNone(args.analyze)

    def test_main_dispatches_to_collect_side_too_large(self):
        fake_result = rotation_script.CollectionResult(
            csv_path=Path("fake.csv"), summary={"scanned": 0}
        )
        with (
            patch.object(
                rotation_script, "collect_side_too_large", return_value=fake_result
            ) as collect_mock,
            patch.object(rotation_script, "_print_side_too_large_summary"),
        ):
            rotation_script.main(["--collect-side-too-large", "15", "--seed", "42"])

        collect_mock.assert_called_once_with(15, seed=42)


class AnalyzeTests(unittest.TestCase):
    def _write_csv(self, directory: str, rows: list[dict[str, str]]) -> Path:
        path = Path(directory) / "rotation.csv"
        columns = ("abs_rotation_ratio", "human_side_group", "human_sample_valid")
        with path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
        return path

    def test_analyze_reports_group_stats_order_iqr_and_threshold_sweep(self):
        rows = [
            {"abs_rotation_ratio": "0.8", "human_side_group": "FRONT", "human_sample_valid": "YES"},
            {"abs_rotation_ratio": "0.6", "human_side_group": "BACK", "human_sample_valid": "YES"},
            {"abs_rotation_ratio": "0.3", "human_side_group": "SIDE_ALLOWED", "human_sample_valid": "YES"},
            {"abs_rotation_ratio": "0.2", "human_side_group": "SIDE_ALLOWED", "human_sample_valid": "YES"},
            {"abs_rotation_ratio": "0.08", "human_side_group": "SIDE_TOO_LARGE", "human_sample_valid": "YES"},
            {"abs_rotation_ratio": "0.02", "human_side_group": "SIDE_TOO_LARGE", "human_sample_valid": "YES"},
            {"abs_rotation_ratio": "9.0", "human_side_group": "", "human_sample_valid": ""},
            {"abs_rotation_ratio": "9.0", "human_side_group": "FRONT", "human_sample_valid": "NO"},
        ]
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = self._write_csv(temporary_directory, rows)

            result = rotation_script.analyze(path)

        self.assertEqual(result.labeled_count, 6)
        self.assertEqual(result.unlabeled_count, 1)
        self.assertEqual(result.excluded_count, 1)
        self.assertEqual(result.groups["FRONT_OR_BACK"].count, 2)
        self.assertAlmostEqual(result.groups["FRONT_OR_BACK"].median, 0.7)
        self.assertTrue(result.median_order_holds)
        self.assertEqual(len(result.iqr_overlaps), 2)
        self.assertTrue(result.threshold_sweep)

        output = io.StringIO()
        with redirect_stdout(output):
            rotation_script._print_analysis(result)
        self.assertIn("median_order_holds: True", output.getvalue())
        self.assertIn("IQR overlaps:", output.getvalue())
        self.assertIn("exploratory_threshold_sweep_top:", output.getvalue())

    def test_new_front_or_back_label_is_supported(self):
        rows = [
            {
                "abs_rotation_ratio": "0.5",
                "human_side_group": "FRONT_OR_BACK",
                "human_sample_valid": "",
            }
        ]
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = self._write_csv(temporary_directory, rows)

            result = rotation_script.analyze(path)

        self.assertEqual(result.groups["FRONT_OR_BACK"].count, 1)
        self.assertIsNone(result.median_order_holds)


if __name__ == "__main__":
    unittest.main()
