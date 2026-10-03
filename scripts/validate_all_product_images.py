"""DB 전체 상품을 현재 production 이미지 검증 로직으로 분류해 CSV로 저장한다.

DB에서는 상품을 SELECT만 하며 validation 상태를 수정하거나 상품을 삭제하지 않는다.
결과는 validation_results/product_image_validation/pass.csv와 fail.csv에 기록한다.

    python scripts/validate_all_product_images.py

기존 결과 파일이 있으면 저장된 product_code를 건너뛰고 자동으로 이어서 실행한다.
소량 smoke test는 별도 디렉터리를 지정해 최종 결과와 분리한다.

    python scripts/validate_all_product_images.py --limit 10 \
        --output-dir /tmp/lookddak-product-image-validation-sample
"""

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.models.product import Product
from app.storage.postgres_storage import get_connection
from app.validation import storage
from app.validation.detector import GroundingDinoDetector
from app.validation.pose import PoseEstimator
from app.validation.product_image_validator import validate_product_image

RESULTS_DIR = PROJECT_ROOT / "validation_results" / "product_image_validation"
CSV_COLUMNS = ("product_code", "image_url")
FLUSH_EVERY = 25
MAX_PRINTED_ERRORS = 10


@dataclass(frozen=True)
class BatchSummary:
    total: int
    processed: int
    pass_count: int
    fail_count: int
    exception_count: int
    pass_path: Path
    fail_path: Path
    interrupted: bool = False


@dataclass(frozen=True)
class ExistingResults:
    pass_codes: frozenset[int]
    fail_codes: frozenset[int]

    @property
    def processed_codes(self) -> frozenset[int]:
        return self.pass_codes | self.fail_codes


def _read_result_codes(path: Path) -> frozenset[int]:
    """기존 결과 CSV의 product_code를 읽고 파일 형식을 검증한다."""
    if not path.exists() or path.stat().st_size == 0:
        return frozenset()

    with path.open(newline="", encoding="utf-8") as result_file:
        reader = csv.DictReader(result_file)
        if tuple(reader.fieldnames or ()) != CSV_COLUMNS:
            raise ValueError(
                f"잘못된 CSV 헤더입니다: {path} "
                f"(expected={CSV_COLUMNS}, actual={reader.fieldnames})"
            )

        codes: set[int] = set()
        for line_number, row in enumerate(reader, start=2):
            raw_code = row.get("product_code")
            try:
                product_code = int(raw_code) if raw_code is not None else None
            except ValueError as error:
                raise ValueError(
                    f"잘못된 product_code입니다: {path}:{line_number} ({raw_code!r})"
                ) from error
            if product_code is None:
                raise ValueError(f"product_code가 없습니다: {path}:{line_number}")
            if product_code in codes:
                raise ValueError(
                    f"중복 product_code입니다: {path}:{line_number} ({product_code})"
                )
            codes.add(product_code)
    return frozenset(codes)


def load_existing_results(output_dir: Path = RESULTS_DIR) -> ExistingResults:
    """PASS/FAIL CSV에 이미 저장된 상품 코드 집합을 불러온다."""
    pass_codes = _read_result_codes(output_dir / "pass.csv")
    fail_codes = _read_result_codes(output_dir / "fail.csv")
    overlap = pass_codes & fail_codes
    if overlap:
        sample = ", ".join(str(code) for code in sorted(overlap)[:5])
        raise ValueError(f"PASS/FAIL CSV 양쪽에 중복된 product_code가 있습니다: {sample}")
    return ExistingResults(pass_codes=pass_codes, fail_codes=fail_codes)


def _open_result_file(path: Path):
    """결과 파일을 append로 열고 빈 파일에만 헤더를 기록한다."""
    needs_header = not path.exists() or path.stat().st_size == 0
    result_file = path.open("a", newline="", encoding="utf-8", buffering=1)
    if needs_header:
        csv.writer(result_file).writerow(CSV_COLUMNS)
        result_file.flush()
    return result_file


def validate_all_products(
    products: list[Product],
    *,
    detector,
    pose_estimator,
    output_dir: Path = RESULTS_DIR,
    progress_every: int = 100,
    existing_results: ExistingResults | None = None,
) -> BatchSummary:
    """미처리 상품만 production validator로 판정해 두 CSV에 append한다.

    예상하지 못한 상품별 예외는 해당 상품만 FAIL로 기록하고 다음 상품을 계속한다.
    CSV는 FLUSH_EVERY건마다 flush하고 Ctrl+C 시에도 즉시 flush한다.
    """
    if progress_every < 1:
        raise ValueError("progress_every는 1 이상이어야 합니다")

    output_dir.mkdir(parents=True, exist_ok=True)
    pass_path = output_dir / "pass.csv"
    fail_path = output_dir / "fail.csv"
    if existing_results is None:
        existing_results = load_existing_results(output_dir)

    total = len(products)
    pass_count = len(existing_results.pass_codes)
    fail_count = len(existing_results.fail_codes)
    processed_product_codes = set(existing_results.processed_codes)
    processed = sum(
        product.product_code in processed_product_codes for product in products
    )
    exception_count = 0
    interrupted = False

    with (
        _open_result_file(pass_path) as pass_file,
        _open_result_file(fail_path) as fail_file,
    ):
        pass_writer = csv.writer(pass_file)
        fail_writer = csv.writer(fail_file)

        try:
            for product in products:
                if product.product_code in processed_product_codes:
                    continue

                try:
                    result = validate_product_image(
                        product,
                        detector=detector,
                        pose_estimator=pose_estimator,
                    )
                    passed = result.passed
                except Exception as error:  # noqa: BLE001 - 상품별 실패를 격리하는 batch 경계
                    passed = False
                    exception_count += 1
                    if exception_count <= MAX_PRINTED_ERRORS:
                        print(
                            f"product_code={product.product_code} 예외 → FAIL: "
                            f"{type(error).__name__}: {error}",
                            file=sys.stderr,
                        )
                    elif exception_count == MAX_PRINTED_ERRORS + 1:
                        print("추가 예외 상세 로그는 생략합니다.", file=sys.stderr)

                row = (product.product_code, product.image_url)
                if passed:
                    pass_writer.writerow(row)
                    pass_count += 1
                else:
                    fail_writer.writerow(row)
                    fail_count += 1
                processed_product_codes.add(product.product_code)
                processed += 1

                if processed % FLUSH_EVERY == 0:
                    pass_file.flush()
                    fail_file.flush()
                if processed % progress_every == 0 or processed == total:
                    print(
                        f"[{processed}/{total}] PASS={pass_count} FAIL={fail_count} "
                        f"exceptions={exception_count}"
                    )
        except KeyboardInterrupt:
            interrupted = True
        finally:
            pass_file.flush()
            fail_file.flush()

    summary = BatchSummary(
        total=total,
        processed=processed,
        pass_count=pass_count,
        fail_count=fail_count,
        exception_count=exception_count,
        pass_path=pass_path,
        fail_path=fail_path,
        interrupted=interrupted,
    )
    if interrupted:
        print("\nValidation interrupted.")
        print("\nSaved progress:")
        print(f"  processed: {summary.processed}/{summary.total}")
        print(f"  PASS: {summary.pass_count}")
        print(f"  FAIL: {summary.fail_count}")
        print("\nRun the same command again to resume.")
    return summary


def run(
    *,
    limit: int | None = None,
    output_dir: Path = RESULTS_DIR,
    progress_every: int = 100,
) -> BatchSummary:
    connection = get_connection()
    try:
        products = storage.get_all_products(connection, limit=limit)
    finally:
        connection.close()

    existing_results = load_existing_results(output_dir)
    processed_in_scope = sum(
        product.product_code in existing_results.processed_codes for product in products
    )
    remaining = len(products) - processed_in_scope

    if existing_results.processed_codes:
        print("Existing results found:")
        print(f"  PASS: {len(existing_results.pass_codes)}")
        print(f"  FAIL: {len(existing_results.fail_codes)}")
        print(f"  processed: {len(existing_results.processed_codes)}")
        print(f"  remaining: {remaining}")
        print("\nResuming validation...")
    else:
        print(f"검증 대상 상품: {len(products)}개")

    if remaining == 0:
        summary = validate_all_products(
            products,
            detector=None,
            pose_estimator=None,
            output_dir=output_dir,
            progress_every=progress_every,
            existing_results=existing_results,
        )
        print("처리할 미검증 상품이 없습니다.")
        return summary

    print("모델 로딩 중 (Grounding DINO, MediaPipe)...")
    detector = GroundingDinoDetector()
    with PoseEstimator() as pose_estimator:
        summary = validate_all_products(
            products,
            detector=detector,
            pose_estimator=pose_estimator,
            output_dir=output_dir,
            progress_every=progress_every,
            existing_results=existing_results,
        )

    if summary.interrupted:
        return summary

    print("완료")
    print(f"Total: {summary.total}")
    print(f"PASS: {summary.pass_count}")
    print(f"FAIL: {summary.fail_count}")
    print(f"Exceptions: {summary.exception_count}")
    print(f"PASS CSV: {summary.pass_path}")
    print(f"FAIL CSV: {summary.fail_path}")
    print("DB는 수정하지 않았습니다.")
    return summary


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"정수여야 합니다: {value!r}") from error
    if parsed < 1:
        raise argparse.ArgumentTypeError(f"1 이상이어야 합니다: {parsed}")
    return parsed


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--limit",
        type=_positive_int,
        default=None,
        metavar="N",
        help="전체 상품 중 product_code 순으로 최대 N개만 검증한다.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=RESULTS_DIR,
        metavar="PATH",
        help=f"CSV 저장 디렉터리. 기본값: {RESULTS_DIR}",
    )
    parser.add_argument(
        "--progress-every",
        type=_positive_int,
        default=100,
        metavar="N",
        help="N건마다 진행 상황을 출력한다. 기본값: 100",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    try:
        run(
            limit=args.limit,
            output_dir=args.output_dir,
            progress_every=args.progress_every,
        )
    except KeyboardInterrupt:
        # 모델 로딩이나 DB 조회 중 Ctrl+C가 들어온 경우에도 traceback 없이 끝낸다.
        existing_results = load_existing_results(args.output_dir)
        print("\nValidation interrupted before the validation loop completed.")
        print("\nSaved progress:")
        print(f"  processed: {len(existing_results.processed_codes)}")
        print(f"  PASS: {len(existing_results.pass_codes)}")
        print(f"  FAIL: {len(existing_results.fail_codes)}")
        print("\nRun the same command again to resume.")


if __name__ == "__main__":
    main()
