"""scripts/validate_product_images.py 의 orchestration 만 검증한다.

Grounding DINO/MediaPipe 파이프라인 자체(app.validation.product_image_validator)는
다른 테스트에서 이미 다루므로, 여기서는 validate_product_image 를 fake 로 바꿔서
"검증 → (dry-run 이면 아무것도 안 함 / --commit 이면 DB 반영) → CSV" 흐름과
--limit/--delete-failed 조합만 확인한다. 실제 DB, 모델, 네트워크는 쓰지 않는다.
"""

import contextlib
import csv
import io
import tempfile
import unittest
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

import scripts.validate_product_images as validate_script
from app.models.product import Product
from app.validation.models import (
    Detection,
    DetectionResult,
    ValidationReason,
    ValidationResult,
    ValidationStatus,
)
from app.validation.pose import BodyLandmarks

CRAWLED_AT = datetime(2026, 9, 22, 12, tzinfo=UTC)


def _product(product_code: int, *, main_category="상의", sub_category="스웨트셔츠") -> Product:
    return Product(
        product_code=product_code,
        product_name=f"상품 {product_code}",
        detail_url=f"https://product.29cm.co.kr/catalog/{product_code}",
        image_url="https://img.29cm.co.kr/item/example.jpg",
        price=10000,
        is_sold_out=False,
        main_category=main_category,
        sub_category=sub_category,
        color=None,
        crawled_at=CRAWLED_AT,
    )


def _pending_row(product_code: int, **product_kwargs) -> tuple:
    """app/validation/storage.py 의 _PRODUCT_COLUMNS 순서와 같은 raw DB row."""
    product = _product(product_code, **product_kwargs)
    return (
        product.product_code,
        product.product_name,
        product.price,
        product.color,
        product.detail_url,
        product.image_url,
        product.main_category,
        product.sub_category,
        product.is_sold_out,
        product.crawled_at,
    )


# --- app/validation/storage.py 가 실제로 실행하는 SQL 패턴만 흉내내는 fake ---


class FakeCursor:
    def __init__(self, connection: "FakeConnection"):
        self._connection = connection
        self._last_sql = ""

    def execute(self, sql, params=()):
        self._last_sql = sql
        self._connection.executed.append((sql, tuple(params) if params else ()))

    def fetchall(self):
        if "validation_status = 'PENDING'" in self._last_sql:
            return self._connection.pending_rows
        if "GROUP BY validation_status" in self._last_sql:
            return self._connection.status_count_rows
        if self._last_sql.startswith("SELECT") and "WHERE product_code = %s" in self._last_sql:
            return self._connection.product_rows
        raise AssertionError(f"이 fake 가 지원하지 않는 조회입니다: {self._last_sql}")

    @property
    def rowcount(self):
        return self._connection.delete_rowcount

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class FakeConnection:
    def __init__(
        self, *, pending_rows=(), status_count_rows=(), delete_rowcount=0, product_rows=()
    ):
        self.pending_rows = list(pending_rows)
        self.status_count_rows = list(status_count_rows)
        self.delete_rowcount = delete_rowcount
        self.product_rows = list(product_rows)
        self.executed: list[tuple[str, tuple]] = []
        self.commit_count = 0
        self.closed = False

    def cursor(self):
        return FakeCursor(self)

    def commit(self):
        self.commit_count += 1

    def close(self):
        self.closed = True


class WriteCsvResultsTests(unittest.TestCase):
    def test_writes_pass_and_fail_csv_from_this_runs_results(self):
        results = [
            (_product(1001), ValidationResult(ValidationStatus.PASS, ValidationReason.VALID)),
            (
                _product(1002),
                ValidationResult(ValidationStatus.FAIL, ValidationReason.MULTIPLE_GARMENTS),
            ),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            results_dir = Path(tmp)

            pass_count, fail_count = validate_script._write_csv_results(results, results_dir)

            self.assertEqual((pass_count, fail_count), (1, 1))
            pass_rows = list(csv.reader((results_dir / "pass_products.csv").open()))
            fail_rows = list(csv.reader((results_dir / "fail_products.csv").open()))

            self.assertEqual(pass_rows[0], list(validate_script.CSV_COLUMNS))
            self.assertEqual(pass_rows[1][0], "1001")
            self.assertEqual(pass_rows[1][-2:], ["PASS", ""])

            self.assertEqual(fail_rows[1][0], "1002")
            self.assertEqual(fail_rows[1][-2:], ["FAIL", "MULTIPLE_GARMENTS"])

    def test_rerunning_overwrites_instead_of_appending(self):
        results = [
            (_product(1001), ValidationResult(ValidationStatus.PASS, ValidationReason.VALID))
        ]
        with tempfile.TemporaryDirectory() as tmp:
            results_dir = Path(tmp)

            validate_script._write_csv_results(results, results_dir)
            validate_script._write_csv_results(results, results_dir)

            pass_rows = list(csv.reader((results_dir / "pass_products.csv").open()))
            # header + 상품 1건. 두 번 실행해도 누적되지 않는다.
            self.assertEqual(len(pass_rows), 2)

    def test_empty_results_still_writes_header_only_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            results_dir = Path(tmp)

            pass_count, fail_count = validate_script._write_csv_results([], results_dir)

            self.assertEqual((pass_count, fail_count), (0, 0))
            pass_rows = list(csv.reader((results_dir / "pass_products.csv").open()))
            self.assertEqual(pass_rows, [list(validate_script.CSV_COLUMNS)])


class CommitResultsTests(unittest.TestCase):
    def test_calls_update_validation_result_for_each_product_and_commits(self):
        connection = FakeConnection()
        results = [
            (_product(1001), ValidationResult(ValidationStatus.PASS, ValidationReason.VALID)),
            (_product(1002), ValidationResult(ValidationStatus.FAIL, ValidationReason.BACK_VIEW)),
        ]

        validate_script._commit_results(connection, results)

        updates = [entry for entry in connection.executed if entry[0].startswith("UPDATE")]
        self.assertEqual(len(updates), 2)
        self.assertEqual(updates[0][1], ("PASS", None, 1001))
        self.assertEqual(updates[1][1], ("FAIL", "BACK_VIEW", 1002))
        self.assertEqual(connection.commit_count, 2)  # update_validation_result 가 건마다 commit


class PrintDetectionsTests(unittest.TestCase):
    def test_prints_label_score_and_box_per_detection_with_index(self):
        detections = [
            Detection("person", 0.912, (1.0, 2.0, 3.0, 4.0)),
            Detection("person", 0.5, (5.0, 6.0, 7.0, 8.0)),
        ]
        stdout = io.StringIO()

        with contextlib.redirect_stdout(stdout):
            validate_script._print_detections("person", detections)

        output = stdout.getvalue()
        self.assertIn("person[0] label = person", output)
        self.assertIn("person[0] score = 0.912", output)
        self.assertIn("person[0] box = [1.0, 2.0, 3.0, 4.0]", output)
        self.assertIn("person[1] label = person", output)
        self.assertIn("person[1] score = 0.500", output)
        self.assertIn("person[1] box = [5.0, 6.0, 7.0, 8.0]", output)

    def test_prints_nothing_for_an_empty_list(self):
        stdout = io.StringIO()

        with contextlib.redirect_stdout(stdout):
            validate_script._print_detections("garment", [])

        self.assertEqual(stdout.getvalue(), "")


class SaveDebugImageTests(unittest.TestCase):
    def test_creates_a_file_at_the_expected_path(self):
        image_rgb = np.zeros((20, 30, 3), dtype=np.uint8)
        detection = DetectionResult(persons=[], garments=[])
        with tempfile.TemporaryDirectory() as tmp:
            results_dir = Path(tmp)

            path = validate_script._save_debug_image(image_rgb, detection, 149369, results_dir)

            self.assertEqual(path, results_dir / "debug_149369.jpg")
            self.assertTrue(path.exists())

    def test_preserves_image_dimensions(self):
        image_rgb = np.zeros((20, 30, 3), dtype=np.uint8)
        detection = DetectionResult(
            persons=[Detection("person", 0.9, (1.0, 1.0, 10.0, 15.0))], garments=[]
        )
        with tempfile.TemporaryDirectory() as tmp:
            results_dir = Path(tmp)

            path = validate_script._save_debug_image(image_rgb, detection, 1, results_dir)

            saved = cv2.imread(str(path))
            self.assertEqual(saved.shape, (20, 30, 3))

    def test_drawing_a_box_changes_the_pixels(self):
        image_rgb = np.zeros((50, 50, 3), dtype=np.uint8)
        detection = DetectionResult(
            persons=[Detection("person", 0.9, (5.0, 5.0, 40.0, 45.0))], garments=[]
        )
        with tempfile.TemporaryDirectory() as tmp:
            results_dir = Path(tmp)

            path = validate_script._save_debug_image(image_rgb, detection, 1, results_dir)

            saved = cv2.imread(str(path))
            self.assertFalse((saved == 0).all())

    def test_no_detections_leaves_the_image_unchanged(self):
        image_rgb = np.zeros((20, 30, 3), dtype=np.uint8)
        detection = DetectionResult(persons=[], garments=[])
        with tempfile.TemporaryDirectory() as tmp:
            results_dir = Path(tmp)

            path = validate_script._save_debug_image(image_rgb, detection, 1, results_dir)

            saved = cv2.imread(str(path))
            self.assertTrue((saved == 0).all())

    def test_person_and_garment_boxes_use_different_colors(self):
        image_rgb = np.zeros((50, 50, 3), dtype=np.uint8)
        detection = DetectionResult(
            persons=[Detection("person", 0.9, (0.0, 0.0, 20.0, 20.0))],
            garments=[Detection("pants", 0.9, (25.0, 25.0, 45.0, 45.0))],
        )
        with tempfile.TemporaryDirectory() as tmp:
            results_dir = Path(tmp)

            path = validate_script._save_debug_image(image_rgb, detection, 1, results_dir)

            saved = cv2.imread(str(path))
            person_edge_pixel = saved[0, 0]
            garment_edge_pixel = saved[25, 25]
            self.assertFalse((person_edge_pixel == garment_edge_pixel).all())


# --- main() 전체 흐름: 실제 모델/네트워크 대신 validate_product_image 를 fake 로 바꾼다 ---


class _FakeDetector:
    pass


class _FakePoseEstimator:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def _make_fake_validate(results_by_code: dict[int, ValidationResult]):
    def fake_validate_product_image(product, *, detector, pose_estimator):
        return results_by_code[product.product_code]

    return fake_validate_product_image


def _run_main(connection: FakeConnection, results_by_code: dict[int, ValidationResult], **kwargs):
    """main() 을 real DB/모델/CSV 파일 없이 실행하고, _write_csv_results 에 넘어간
    결과 목록을 돌려준다(CSV 내용 자체는 WriteCsvResultsTests 가 따로 검증한다)."""
    written: dict[str, list] = {}

    def fake_write_csv_results(results, results_dir=None):
        written["results"] = results
        pass_count = sum(1 for _, result in results if result.passed)
        return pass_count, len(results) - pass_count

    with (
        patch.object(validate_script, "get_connection", return_value=connection),
        patch.object(validate_script, "GroundingDinoDetector", return_value=_FakeDetector()),
        patch.object(validate_script, "PoseEstimator", return_value=_FakePoseEstimator()),
        patch.object(
            validate_script, "validate_product_image", _make_fake_validate(results_by_code)
        ),
        patch.object(validate_script, "_write_csv_results", fake_write_csv_results),
    ):
        validate_script.main(**kwargs)
    return written.get("results", [])


class MainDryRunTests(unittest.TestCase):
    def test_dry_run_does_not_touch_the_database(self):
        connection = FakeConnection(pending_rows=[_pending_row(1001), _pending_row(1002)])
        results_by_code = {
            1001: ValidationResult(ValidationStatus.PASS, ValidationReason.VALID),
            1002: ValidationResult(ValidationStatus.FAIL, ValidationReason.MULTIPLE_GARMENTS),
        }

        _run_main(connection, results_by_code, delete_failed=False, limit=None, commit=False)

        updates = [entry for entry in connection.executed if entry[0].startswith("UPDATE")]
        self.assertEqual(updates, [])
        self.assertEqual(connection.commit_count, 0)

    def test_dry_run_still_produces_results_for_the_csv(self):
        connection = FakeConnection(pending_rows=[_pending_row(1001), _pending_row(1002)])
        results_by_code = {
            1001: ValidationResult(ValidationStatus.PASS, ValidationReason.VALID),
            1002: ValidationResult(ValidationStatus.FAIL, ValidationReason.MULTIPLE_GARMENTS),
        }

        written = _run_main(
            connection, results_by_code, delete_failed=False, limit=None, commit=False
        )

        self.assertEqual([product.product_code for product, _ in written], [1001, 1002])
        self.assertTrue(written[0][1].passed)
        self.assertFalse(written[1][1].passed)


class MainCommitTests(unittest.TestCase):
    def test_commit_reflects_every_result_in_the_database(self):
        connection = FakeConnection(
            pending_rows=[_pending_row(1001), _pending_row(1002)],
            status_count_rows=[("PENDING", 0)],
        )
        results_by_code = {
            1001: ValidationResult(ValidationStatus.PASS, ValidationReason.VALID),
            1002: ValidationResult(ValidationStatus.FAIL, ValidationReason.BACK_VIEW),
        }

        _run_main(connection, results_by_code, delete_failed=False, limit=None, commit=True)

        updates = [entry for entry in connection.executed if entry[0].startswith("UPDATE")]
        self.assertEqual(len(updates), 2)
        self.assertEqual(updates[0][1], ("PASS", None, 1001))
        self.assertEqual(updates[1][1], ("FAIL", "BACK_VIEW", 1002))

    def test_commit_with_limit_only_commits_what_was_selected(self):
        # limit 이 실제로 몇 건을 돌려주는지는 storage.get_pending_products 의 책임이고
        # test_storage.py 에서 이미 검증했다. 여기서는 (1) LIMIT 파라미터가 SQL 에
        # 실제로 실렸는지, (2) 조회된 만큼만 커밋되는지를 함께 확인한다.
        connection = FakeConnection(
            pending_rows=[_pending_row(1001), _pending_row(1002)],
            status_count_rows=[("PENDING", 0)],
        )
        results_by_code = {
            1001: ValidationResult(ValidationStatus.PASS, ValidationReason.VALID),
            1002: ValidationResult(ValidationStatus.PASS, ValidationReason.VALID),
        }

        _run_main(connection, results_by_code, delete_failed=False, limit=10, commit=True)

        [(select_sql, select_params)] = [
            entry
            for entry in connection.executed
            if "validation_status = 'PENDING'" in entry[0]
        ]
        self.assertIn("LIMIT", select_sql)
        self.assertEqual(select_params, (10,))
        updates = [entry for entry in connection.executed if entry[0].startswith("UPDATE")]
        self.assertEqual(len(updates), 2)  # pending_rows 에 있던 만큼만


class MainDeleteFailedTests(unittest.TestCase):
    def test_delete_failed_only_deletes_and_never_validates(self):
        connection = FakeConnection(status_count_rows=[("FAIL", 3)], delete_rowcount=3)

        def fail_if_called(*args, **kwargs):
            raise AssertionError("--delete-failed 는 검증을 수행하면 안 된다")

        with (
            patch.object(validate_script, "get_connection", return_value=connection),
            patch.object(validate_script, "GroundingDinoDetector", side_effect=fail_if_called),
            patch.object(validate_script, "PoseEstimator", side_effect=fail_if_called),
            patch.object(validate_script, "validate_product_image", fail_if_called),
        ):
            validate_script.main(delete_failed=True)

        deletes = [entry for entry in connection.executed if entry[0].startswith("DELETE")]
        self.assertEqual(len(deletes), 1)
        self.assertTrue(connection.closed)


class _FakeDetectorWithResult:
    def __init__(self, detection_result: DetectionResult):
        self._result = detection_result
        self.calls: list[tuple] = []

    def detect(self, image_rgb, garment_labels):
        self.calls.append((image_rgb, tuple(garment_labels)))
        return self._result


class _FakePoseEstimatorWithLandmarks:
    def __init__(self, landmarks: BodyLandmarks | None):
        self._landmarks = landmarks
        self.calls = 0

    def get_landmarks(self, image_rgb):
        self.calls += 1
        return self._landmarks

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class MainSingleProductTests(unittest.TestCase):
    """--product-code (와 --debug) 전용. DB 를 절대 바꾸지 않는다."""

    def test_product_not_found_does_not_call_the_model_or_touch_the_db(self):
        connection = FakeConnection(product_rows=[])

        def fail_if_called(*args, **kwargs):
            raise AssertionError("상품을 못 찾았으면 검증을 시도하면 안 된다")

        with (
            patch.object(validate_script, "get_connection", return_value=connection),
            patch.object(validate_script, "GroundingDinoDetector", side_effect=fail_if_called),
            patch.object(validate_script, "PoseEstimator", side_effect=fail_if_called),
        ):
            validate_script.main(product_code=999999)

        writes = [e for e in connection.executed if e[0].startswith(("UPDATE", "DELETE"))]
        self.assertEqual(writes, [])
        self.assertTrue(connection.closed)

    def test_found_product_without_debug_does_not_touch_the_db(self):
        connection = FakeConnection(product_rows=[_pending_row(1001)])
        detection = DetectionResult(
            persons=[], garments=[Detection("sweatshirt", 0.9, (0.0, 0.0, 1.0, 1.0))]
        )

        fake_detector = _FakeDetectorWithResult(detection)
        fake_pose_estimator = _FakePoseEstimatorWithLandmarks(None)
        with (
            patch.object(validate_script, "get_connection", return_value=connection),
            patch.object(validate_script, "GroundingDinoDetector", return_value=fake_detector),
            patch.object(validate_script, "PoseEstimator", return_value=fake_pose_estimator),
            patch.object(validate_script, "download_image", return_value="fake-image"),
        ):
            validate_script.main(product_code=1001)

        writes = [e for e in connection.executed if e[0].startswith(("UPDATE", "DELETE"))]
        self.assertEqual(writes, [])

    def test_debug_prints_every_intermediate_value_for_a_bottom_product(self):
        # 하의 + garment 1개 + 사람 1명 + 상체 landmark 부족 + 하체 landmark 충분
        # -> allow_lower_body_fallback=True, pose_direction=FRONT, 최종 PASS.
        #
        # 실제 파일 저장(_save_debug_image 의 진짜 동작)은 SaveDebugImageTests 가 따로
        # 검증하므로, 여기서는 fake 로 바꿔서 "무엇을 넘겨 호출했는지·출력에 경로가
        # 찍히는지"만 본다. main() 은 _save_debug_image 를 항상 기본 results_dir 인자로
        # 부르므로(명시적으로 넘기지 않음), RESULTS_DIR 을 나중에 patch 해도 그 기본값에는
        # 반영되지 않는다 — 그래서 함수 자체를 fake 로 바꾼다.
        connection = FakeConnection(
            product_rows=[
                _pending_row(1001, main_category="하의", sub_category="슬림 팬츠")
            ]
        )
        detection = DetectionResult(
            persons=[Detection("person", 0.9, (5.0, 5.0, 40.0, 90.0))],
            garments=[Detection("slim pants", 0.85, (10.0, 50.0, 35.0, 95.0))],
        )
        landmarks = BodyLandmarks(
            nose_visibility=0.1,
            left_shoulder_x=0.5,
            left_shoulder_visibility=0.1,
            right_shoulder_x=0.5,
            right_shoulder_visibility=0.1,
            shoulder_mid_y=0.0,
            hip_mid_y=0.3,
            left_hip_visibility=0.9,
            right_hip_visibility=0.9,
            left_knee_visibility=0.9,
            right_knee_visibility=0.1,
            left_ankle_visibility=0.1,
            right_ankle_visibility=0.1,
        )
        fake_image = np.zeros((100, 50, 3), dtype=np.uint8)  # height=100, width=50
        fake_debug_path = Path("/fake/debug_1001.jpg")
        save_debug_image_calls = []

        def fake_save_debug_image(image_rgb, detection_arg, product_code, results_dir=None):
            save_debug_image_calls.append((image_rgb, detection_arg, product_code))
            return fake_debug_path

        fake_detector = _FakeDetectorWithResult(detection)
        fake_pose_estimator = _FakePoseEstimatorWithLandmarks(landmarks)
        stdout = io.StringIO()
        with (
            patch.object(validate_script, "get_connection", return_value=connection),
            patch.object(validate_script, "GroundingDinoDetector", return_value=fake_detector),
            patch.object(validate_script, "PoseEstimator", return_value=fake_pose_estimator),
            patch.object(validate_script, "download_image", return_value=fake_image),
            patch.object(validate_script, "_save_debug_image", fake_save_debug_image),
            contextlib.redirect_stdout(stdout),
        ):
            validate_script.main(product_code=1001, debug=True)

        self.assertEqual(len(save_debug_image_calls), 1)
        called_image, called_detection, called_code = save_debug_image_calls[0]
        self.assertIs(called_image, fake_image)
        self.assertEqual(called_detection, detection)
        self.assertEqual(called_code, 1001)

        output = stdout.getvalue()
        for expected in (
            "product_code = 1001",
            "main_category = 하의",
            "sub_category = 슬림 팬츠",
            "detection labels",
            "image width = 50",
            "image height = 100",
            "person_count = 1",
            "garment_count = 1",
            "person[0] label = person",
            "person[0] score = 0.900",
            "person[0] box = [5.0, 5.0, 40.0, 90.0]",
            "garment[0] label = slim pants",
            "garment[0] score = 0.850",
            "garment[0] box = [10.0, 50.0, 35.0, 95.0]",
            f"debug image = {fake_debug_path}",
            "allow_lower_body_fallback = True",
            "nose visibility",
            "left shoulder visibility",
            "right shoulder visibility",
            "left hip visibility",
            "right hip visibility",
            "left knee visibility",
            "right knee visibility",
            "left ankle visibility",
            "right ankle visibility",
            "has_sufficient_upper_body_landmarks = False",
            "has_sufficient_lower_body_landmarks = True",
            "pose_direction = FRONT",
            "final validation_status = PASS",
            "final validation_reason = VALID",
        ):
            self.assertIn(expected, output, msg=f"missing in debug output: {expected!r}")

        writes = [e for e in connection.executed if e[0].startswith(("UPDATE", "DELETE"))]
        self.assertEqual(writes, [])

    def test_debug_stops_early_on_category_mapping_not_found(self):
        connection = FakeConnection(
            product_rows=[
                _pending_row(1001, main_category="없는카테고리", sub_category="없는서브")
            ]
        )

        def fail_if_called(*args, **kwargs):
            raise AssertionError("label 매핑이 없으면 이미지를 내려받으면 안 된다")

        fake_detector = _FakeDetectorWithResult(DetectionResult(persons=[], garments=[]))
        fake_pose_estimator = _FakePoseEstimatorWithLandmarks(None)
        stdout = io.StringIO()
        with (
            patch.object(validate_script, "get_connection", return_value=connection),
            patch.object(validate_script, "GroundingDinoDetector", return_value=fake_detector),
            patch.object(validate_script, "PoseEstimator", return_value=fake_pose_estimator),
            patch.object(validate_script, "download_image", side_effect=fail_if_called),
            contextlib.redirect_stdout(stdout),
        ):
            validate_script.main(product_code=1001, debug=True)

        output = stdout.getvalue()
        self.assertIn("final validation_status = FAIL", output)
        self.assertIn("final validation_reason = CATEGORY_MAPPING_NOT_FOUND", output)

    def test_debug_prints_bbox_fallback_values_when_landmarks_are_none(self):
        # 하의 + garment 1개 + 사람 1명 + MediaPipe landmarks=None + bbox 조건 전부 만족
        # -> is_cropped_lower_body=True, pose_direction=FRONT, 최종 PASS.
        connection = FakeConnection(
            product_rows=[
                _pending_row(1001, main_category="하의", sub_category="슬림 팬츠")
            ]
        )
        detection = DetectionResult(
            persons=[Detection("person", 0.9, (100.0, 0.0, 400.0, 990.0))],
            garments=[Detection("slim pants", 0.85, (120.0, 200.0, 380.0, 950.0))],
        )
        fake_image = np.zeros((1000, 500, 3), dtype=np.uint8)  # height=1000

        fake_detector = _FakeDetectorWithResult(detection)
        fake_pose_estimator = _FakePoseEstimatorWithLandmarks(None)
        stdout = io.StringIO()
        with (
            patch.object(validate_script, "get_connection", return_value=connection),
            patch.object(validate_script, "GroundingDinoDetector", return_value=fake_detector),
            patch.object(validate_script, "PoseEstimator", return_value=fake_pose_estimator),
            patch.object(validate_script, "download_image", return_value=fake_image),
            patch.object(validate_script, "_save_debug_image", return_value=Path("/fake.jpg")),
            contextlib.redirect_stdout(stdout),
        ):
            validate_script.main(product_code=1001, debug=True)

        output = stdout.getvalue()
        for expected in (
            "landmarks = None",
            "person_top_ratio = 0.0000",
            "garment_top_ratio = 0.2000",
            "is_cropped_lower_body = True",
            "pose_direction = FRONT",
            "final validation_status = PASS",
            "final validation_reason = VALID",
        ):
            self.assertIn(expected, output, msg=f"missing in debug output: {expected!r}")
        self.assertIn("garment_person_area_ratio = 0.", output)
        self.assertIn("garment_in_person_ratio = 1.0000", output)

        writes = [e for e in connection.executed if e[0].startswith(("UPDATE", "DELETE"))]
        self.assertEqual(writes, [])

    def test_debug_does_not_print_bbox_values_for_a_top_product(self):
        # 상의는 allow_lower_body_fallback 이 False 라서 bbox 값 자체를 계산/출력하지 않는다.
        connection = FakeConnection(
            product_rows=[
                _pending_row(1001, main_category="상의", sub_category="스웨트셔츠")
            ]
        )
        detection = DetectionResult(
            persons=[Detection("person", 0.9, (100.0, 0.0, 400.0, 990.0))],
            garments=[Detection("sweatshirt", 0.85, (120.0, 200.0, 380.0, 950.0))],
        )
        fake_image = np.zeros((1000, 500, 3), dtype=np.uint8)

        fake_detector = _FakeDetectorWithResult(detection)
        fake_pose_estimator = _FakePoseEstimatorWithLandmarks(None)
        stdout = io.StringIO()
        with (
            patch.object(validate_script, "get_connection", return_value=connection),
            patch.object(validate_script, "GroundingDinoDetector", return_value=fake_detector),
            patch.object(validate_script, "PoseEstimator", return_value=fake_pose_estimator),
            patch.object(validate_script, "download_image", return_value=fake_image),
            patch.object(validate_script, "_save_debug_image", return_value=Path("/fake.jpg")),
            contextlib.redirect_stdout(stdout),
        ):
            validate_script.main(product_code=1001, debug=True)

        output = stdout.getvalue()
        self.assertIn("allow_lower_body_fallback = False", output)
        self.assertNotIn("is_cropped_lower_body", output)
        self.assertIn("pose_direction = UNCERTAIN", output)
        self.assertIn("final validation_reason = UNCERTAIN", output)


class ParseArgsTests(unittest.TestCase):
    def test_no_flags_means_dry_run_no_delete_and_no_limit(self):
        args = validate_script._parse_args([])

        self.assertFalse(args.delete_failed)
        self.assertFalse(args.commit)
        self.assertIsNone(args.limit)

    def test_limit_is_parsed_as_int(self):
        args = validate_script._parse_args(["--limit", "10"])

        self.assertEqual(args.limit, 10)
        self.assertFalse(args.delete_failed)

    def test_delete_failed_flag_still_works(self):
        args = validate_script._parse_args(["--delete-failed"])

        self.assertTrue(args.delete_failed)
        self.assertIsNone(args.limit)

    def test_commit_flag_defaults_to_false(self):
        args = validate_script._parse_args([])

        self.assertFalse(args.commit)

    def test_commit_flag_is_parsed(self):
        args = validate_script._parse_args(["--commit"])

        self.assertTrue(args.commit)

    def test_commit_and_limit_together_is_allowed(self):
        args = validate_script._parse_args(["--commit", "--limit", "10"])

        self.assertTrue(args.commit)
        self.assertEqual(args.limit, 10)

    def test_limit_zero_is_rejected(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            validate_script._parse_args(["--limit", "0"])

    def test_negative_limit_is_rejected(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            validate_script._parse_args(["--limit", "-1"])

    def test_non_integer_limit_is_rejected(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            validate_script._parse_args(["--limit", "abc"])

    def test_delete_failed_and_limit_together_is_rejected(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            validate_script._parse_args(["--delete-failed", "--limit", "10"])

    def test_delete_failed_and_commit_together_is_rejected(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            validate_script._parse_args(["--delete-failed", "--commit"])

    def test_product_code_is_parsed_as_int(self):
        args = validate_script._parse_args(["--product-code", "149369"])

        self.assertEqual(args.product_code, 149369)
        self.assertFalse(args.debug)

    def test_product_code_defaults_to_none(self):
        args = validate_script._parse_args([])

        self.assertIsNone(args.product_code)

    def test_product_code_with_debug_is_parsed(self):
        args = validate_script._parse_args(["--product-code", "149369", "--debug"])

        self.assertEqual(args.product_code, 149369)
        self.assertTrue(args.debug)

    def test_debug_without_product_code_is_rejected(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            validate_script._parse_args(["--debug"])

    def test_product_code_and_limit_together_is_rejected(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            validate_script._parse_args(["--product-code", "1", "--limit", "10"])

    def test_product_code_and_commit_together_is_rejected(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            validate_script._parse_args(["--product-code", "1", "--commit"])

    def test_product_code_and_delete_failed_together_is_rejected(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            validate_script._parse_args(["--product-code", "1", "--delete-failed"])


if __name__ == "__main__":
    unittest.main()
