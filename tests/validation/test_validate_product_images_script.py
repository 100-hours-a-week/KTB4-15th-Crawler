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

import scripts.validate_product_images as validate_script
from app.models.product import Product
from app.validation.models import ValidationReason, ValidationResult, ValidationStatus

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


def _pending_row(product_code: int) -> tuple:
    """app/validation/storage.py 의 _PRODUCT_COLUMNS 순서와 같은 raw DB row."""
    product = _product(product_code)
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
        raise AssertionError(f"이 fake 가 지원하지 않는 조회입니다: {self._last_sql}")

    @property
    def rowcount(self):
        return self._connection.delete_rowcount

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class FakeConnection:
    def __init__(self, *, pending_rows=(), status_count_rows=(), delete_rowcount=0):
        self.pending_rows = list(pending_rows)
        self.status_count_rows = list(status_count_rows)
        self.delete_rowcount = delete_rowcount
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


if __name__ == "__main__":
    unittest.main()
