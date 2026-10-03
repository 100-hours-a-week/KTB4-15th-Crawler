import csv
import tempfile
import unittest
from contextlib import redirect_stdout
from datetime import UTC, datetime
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import scripts.validate_all_product_images as batch_script
from app.models.product import Product
from app.validation.models import ValidationReason, ValidationResult, ValidationStatus


def _product(code: int) -> Product:
    return Product(
        product_code=code,
        product_name=f"상품 {code}",
        detail_url=f"https://example.com/products/{code}",
        image_url=f"https://example.com/images/{code}.jpg",
        price=10000,
        is_sold_out=False,
        main_category="상의",
        sub_category="스웨트셔츠",
        color=None,
        crawled_at=datetime(2026, 10, 2, tzinfo=UTC),
    )


_PASS = ValidationResult(ValidationStatus.PASS, ValidationReason.VALID)
_FAIL = ValidationResult(ValidationStatus.FAIL, ValidationReason.UNCERTAIN)


def _write_results(path: Path, rows: list[tuple[int, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as result_file:
        writer = csv.writer(result_file)
        writer.writerow(batch_script.CSV_COLUMNS)
        writer.writerows(rows)


class ValidateAllProductsTests(unittest.TestCase):
    def test_writes_only_pass_products_to_pass_csv_and_fail_products_to_fail_csv(self):
        products = [_product(1), _product(2), _product(3)]
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory)
            with patch.object(
                batch_script,
                "validate_product_image",
                side_effect=[_PASS, _FAIL, _PASS],
            ):
                summary = batch_script.validate_all_products(
                    products,
                    detector=object(),
                    pose_estimator=object(),
                    output_dir=output_dir,
                    progress_every=10,
                )

            with summary.pass_path.open(newline="", encoding="utf-8") as pass_file:
                pass_rows = list(csv.DictReader(pass_file))
            with summary.fail_path.open(newline="", encoding="utf-8") as fail_file:
                fail_rows = list(csv.DictReader(fail_file))

        self.assertEqual(tuple(pass_rows[0]), batch_script.CSV_COLUMNS)
        self.assertEqual(tuple(fail_rows[0]), batch_script.CSV_COLUMNS)
        self.assertEqual([row["product_code"] for row in pass_rows], ["1", "3"])
        self.assertEqual([row["product_code"] for row in fail_rows], ["2"])
        self.assertEqual((summary.total, summary.pass_count, summary.fail_count), (3, 2, 1))

    def test_unexpected_product_error_is_fail_and_batch_continues(self):
        products = [_product(1), _product(2)]
        with tempfile.TemporaryDirectory() as temporary_directory:
            with patch.object(
                batch_script,
                "validate_product_image",
                side_effect=[RuntimeError("boom"), _PASS],
            ):
                summary = batch_script.validate_all_products(
                    products,
                    detector=object(),
                    pose_estimator=object(),
                    output_dir=Path(temporary_directory),
                    progress_every=10,
                )
            with summary.fail_path.open(newline="", encoding="utf-8") as fail_file:
                fail_rows = list(csv.DictReader(fail_file))

        self.assertEqual([row["product_code"] for row in fail_rows], ["1"])
        self.assertEqual(summary.processed, 2)
        self.assertEqual(summary.exception_count, 1)
        self.assertEqual(summary.pass_count, 1)

    def test_empty_input_still_writes_exact_headers(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            summary = batch_script.validate_all_products(
                [],
                detector=object(),
                pose_estimator=object(),
                output_dir=Path(temporary_directory),
            )

            self.assertEqual(
                summary.pass_path.read_text(encoding="utf-8"),
                "product_code,image_url\n",
            )
            self.assertEqual(
                summary.fail_path.read_text(encoding="utf-8"),
                "product_code,image_url\n",
            )

    def test_resume_skips_existing_pass_and_fail_and_appends_only_missing_products(self):
        products = [_product(1), _product(2), _product(3), _product(4)]
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory)
            _write_results(output_dir / "pass.csv", [(1, products[0].image_url)])
            _write_results(output_dir / "fail.csv", [(2, products[1].image_url)])

            with patch.object(
                batch_script,
                "validate_product_image",
                side_effect=[_PASS, _FAIL],
            ) as validate:
                summary = batch_script.validate_all_products(
                    products,
                    detector=object(),
                    pose_estimator=object(),
                    output_dir=output_dir,
                    progress_every=10,
                )

            self.assertEqual(
                [call.args[0].product_code for call in validate.call_args_list], [3, 4]
            )
            with summary.pass_path.open(newline="", encoding="utf-8") as pass_file:
                pass_rows = list(csv.DictReader(pass_file))
            with summary.fail_path.open(newline="", encoding="utf-8") as fail_file:
                fail_rows = list(csv.DictReader(fail_file))
            self.assertEqual([row["product_code"] for row in pass_rows], ["1", "3"])
            self.assertEqual([row["product_code"] for row in fail_rows], ["2", "4"])
            self.assertEqual((summary.processed, summary.pass_count, summary.fail_count), (4, 2, 2))
            self.assertEqual(
                summary.pass_path.read_text(encoding="utf-8").count(
                    "product_code,image_url"
                ),
                1,
            )
            self.assertEqual(
                summary.fail_path.read_text(encoding="utf-8").count(
                    "product_code,image_url"
                ),
                1,
            )

            with patch.object(batch_script, "validate_product_image") as validate_again:
                batch_script.validate_all_products(
                    products,
                    detector=object(),
                    pose_estimator=object(),
                    output_dir=output_dir,
                    progress_every=10,
                )
            validate_again.assert_not_called()

            with summary.pass_path.open(newline="", encoding="utf-8") as pass_file:
                pass_codes = [row["product_code"] for row in csv.DictReader(pass_file)]
            with summary.fail_path.open(newline="", encoding="utf-8") as fail_file:
                fail_codes = [row["product_code"] for row in csv.DictReader(fail_file)]
            self.assertEqual(len(pass_codes), len(set(pass_codes)))
            self.assertEqual(len(fail_codes), len(set(fail_codes)))

    def test_keyboard_interrupt_flushes_progress_without_marking_product_failed(self):
        products = [_product(1), _product(2), _product(3)]
        stdout = StringIO()
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory)
            with (
                patch.object(
                    batch_script,
                    "validate_product_image",
                    side_effect=[_PASS, KeyboardInterrupt()],
                ),
                redirect_stdout(stdout),
            ):
                summary = batch_script.validate_all_products(
                    products,
                    detector=object(),
                    pose_estimator=object(),
                    output_dir=output_dir,
                    progress_every=10,
                )

            with summary.pass_path.open(newline="", encoding="utf-8") as pass_file:
                pass_rows = list(csv.DictReader(pass_file))
            with summary.fail_path.open(newline="", encoding="utf-8") as fail_file:
                fail_rows = list(csv.DictReader(fail_file))

        self.assertTrue(summary.interrupted)
        self.assertEqual(summary.processed, 1)
        self.assertEqual([row["product_code"] for row in pass_rows], ["1"])
        self.assertEqual(fail_rows, [])
        self.assertIn("Validation interrupted.", stdout.getvalue())
        self.assertIn("processed: 1/3", stdout.getvalue())
        self.assertIn("Run the same command again to resume.", stdout.getvalue())

    def test_rejects_product_code_present_in_both_result_files(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory)
            _write_results(output_dir / "pass.csv", [(1, "pass.jpg")])
            _write_results(output_dir / "fail.csv", [(1, "fail.jpg")])

            with self.assertRaisesRegex(ValueError, "양쪽에 중복"):
                batch_script.load_existing_results(output_dir)


class ParseArgsTests(unittest.TestCase):
    def test_defaults_to_all_products_and_production_output_directory(self):
        args = batch_script._parse_args([])

        self.assertIsNone(args.limit)
        self.assertEqual(args.output_dir, batch_script.RESULTS_DIR)
        self.assertEqual(args.progress_every, 100)

    def test_limit_output_and_progress_are_configurable(self):
        args = batch_script._parse_args(
            ["--limit", "5", "--output-dir", "/tmp/output", "--progress-every", "2"]
        )

        self.assertEqual(args.limit, 5)
        self.assertEqual(args.output_dir, Path("/tmp/output"))
        self.assertEqual(args.progress_every, 2)

    def test_main_suppresses_keyboard_interrupt_during_setup(self):
        existing_results = batch_script.ExistingResults(
            pass_codes=frozenset({1, 2}),
            fail_codes=frozenset({3}),
        )
        stdout = StringIO()
        with (
            patch.object(batch_script, "run", side_effect=KeyboardInterrupt()),
            patch.object(
                batch_script,
                "load_existing_results",
                return_value=existing_results,
            ),
            redirect_stdout(stdout),
        ):
            batch_script.main([])

        self.assertIn("processed: 3", stdout.getvalue())
        self.assertIn("PASS: 2", stdout.getvalue())
        self.assertIn("FAIL: 1", stdout.getvalue())
        self.assertIn("Run the same command again to resume.", stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
