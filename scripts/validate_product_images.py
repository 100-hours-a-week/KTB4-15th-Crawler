"""29CM 상품 이미지를 로컬에서 검증하는 데이터 정제 도구.

운영 크롤러나 API 로 배포하는 기능이 아니다. Grounding DINO/MediaPipe 같은 무거운
의존성은 requirements-validation.txt 에만 있고 운영 requirements.txt 에는 없다.

기본 실행 = dry-run: PENDING 상품을 검증하고 CSV 를 만들지만 DB 는 바꾸지 않는다.

    python scripts/validate_product_images.py --limit 50

결과를 확인한 뒤 실제로 DB 에 반영하려면 --commit 을 붙인다(PASS/FAIL 기록).

    python scripts/validate_product_images.py --limit 50 --commit

사람이 fail_products.csv 를 확인한 뒤에만 FAIL 상품을 지운다. 검증은 하지 않는다.

    python scripts/validate_product_images.py --delete-failed
"""

import argparse
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.models.product import Product
from app.storage.postgres_storage import get_connection
from app.validation import storage
from app.validation.detector import GroundingDinoDetector
from app.validation.models import ValidationResult
from app.validation.pose import PoseEstimator
from app.validation.product_image_validator import validate_product_image

RESULTS_DIR = PROJECT_ROOT / "validation_results"
CSV_COLUMNS = (
    "product_code",
    "product_name",
    "image_url",
    "main_category",
    "sub_category",
    "validation_status",
    "validation_reason",
)


def _validate_products(
    products: list[Product], *, detector, pose_estimator
) -> list[tuple[Product, ValidationResult]]:
    """상품을 검증만 하고 DB 는 건드리지 않는다. DB 반영 여부는 호출하는 쪽이 결정한다."""
    total = len(products)
    results: list[tuple[Product, ValidationResult]] = []
    for index, product in enumerate(products, start=1):
        result = validate_product_image(product, detector=detector, pose_estimator=pose_estimator)
        results.append((product, result))
        if result.passed:
            print(f"[{index}/{total}] product_code={product.product_code} PASS")
        else:
            print(
                f"[{index}/{total}] product_code={product.product_code} "
                f"FAIL reason={result.reason.value}"
            )
    return results


def _commit_results(connection, results: list[tuple[Product, ValidationResult]]) -> None:
    """검증 결과를 DB 에 실제로 반영한다. --commit 일 때만 호출한다."""
    for product, result in results:
        reason = None if result.passed else result.reason.value
        storage.update_validation_result(
            connection, product.product_code, result.status.value, reason
        )


def _write_csv_results(
    results: list[tuple[Product, ValidationResult]], results_dir: Path = RESULTS_DIR
) -> tuple[int, int]:
    """이번 실행에서 나온 판정 결과로 CSV 두 개를 새로 만든다(덮어쓴다).

    DB 를 다시 조회하지 않는다. dry-run 에서는 DB 가 PENDING 그대로라 조회해도 이번
    실행 결과를 알 수 없고, --commit 에서도 "이번 실행에서 처리한 상품"만 CSV 에
    남기기 위해서다(--limit 을 쓰면 DB 전체 PASS/FAIL 과 이번 실행 대상이 다르다).
    """
    results_dir.mkdir(parents=True, exist_ok=True)
    pass_count = fail_count = 0
    pass_path = results_dir / "pass_products.csv"
    fail_path = results_dir / "fail_products.csv"
    with (
        pass_path.open("w", newline="", encoding="utf-8") as pass_file,
        fail_path.open("w", newline="", encoding="utf-8") as fail_file,
    ):
        pass_writer = csv.writer(pass_file)
        fail_writer = csv.writer(fail_file)
        pass_writer.writerow(CSV_COLUMNS)
        fail_writer.writerow(CSV_COLUMNS)
        for product, result in results:
            row = (
                product.product_code,
                product.product_name,
                product.image_url,
                product.main_category,
                product.sub_category,
                result.status.value,
                "" if result.passed else result.reason.value,
            )
            if result.passed:
                pass_writer.writerow(row)
                pass_count += 1
            else:
                fail_writer.writerow(row)
                fail_count += 1
    return pass_count, fail_count


def _delete_failed(connection) -> None:
    counts = storage.count_by_validation_status(connection)
    print(f"삭제 대상 FAIL 상품: {counts['FAIL']}건")
    if counts["FAIL"] == 0:
        return
    deleted = storage.delete_failed_products(connection)
    print(f"삭제 완료: {deleted}건")


def main(*, delete_failed: bool = False, limit: int | None = None, commit: bool = False) -> None:
    connection = get_connection()
    try:
        if delete_failed:
            _delete_failed(connection)
            return

        products = storage.get_pending_products(connection, limit=limit)
        total = len(products)
        print(f"PENDING 상품: {total}건")

        results: list[tuple[Product, ValidationResult]] = []
        if total == 0:
            print("검증할 상품이 없습니다.")
        else:
            print("모델 로딩 중 (Grounding DINO, MediaPipe)...")
            detector = GroundingDinoDetector()
            with PoseEstimator() as pose_estimator:
                results = _validate_products(
                    products, detector=detector, pose_estimator=pose_estimator
                )

        pass_count = sum(1 for _, result in results if result.passed)
        fail_count = len(results) - pass_count
        print()
        print(f"Total: {len(results)}")
        print(f"PASS: {pass_count}")
        print(f"FAIL: {fail_count}")

        if commit:
            _commit_results(connection, results)
            print("DB 에 반영했습니다.")
        else:
            print("dry-run 입니다. DB 의 validation_status/validation_reason 은 바꾸지 않았습니다.")

        written_pass, written_fail = _write_csv_results(results)
        print(
            f"CSV 생성 완료: pass_products.csv={written_pass}건, "
            f"fail_products.csv={written_fail}건"
        )

        if commit:
            counts = storage.count_by_validation_status(connection)
            if counts["PENDING"] > 0:
                print(f"아직 PENDING 상품이 {counts['PENDING']}건 남아 있습니다.")
    finally:
        connection.close()


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"정수여야 합니다: {value!r}") from error
    if parsed < 1:
        raise argparse.ArgumentTypeError(f"1 이상이어야 합니다: {parsed}")
    return parsed


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="상품 이미지를 로컬에서 검증한다.")
    parser.add_argument(
        "--delete-failed",
        action="store_true",
        help="검증을 실행하지 않고 validation_status='FAIL' 상품을 DB에서 삭제만 한다.",
    )
    parser.add_argument(
        "--commit",
        action="store_true",
        help="검증 결과를 DB에 실제로 반영한다. 기본은 dry-run(DB 를 바꾸지 않음)이다.",
    )
    parser.add_argument(
        "--limit",
        type=_positive_int,
        default=None,
        metavar="N",
        help="PENDING 상품 중 최대 N개만 조회해서 검증한다. 기본값은 전체다.",
    )
    args = parser.parse_args(argv)

    # --delete-failed 는 검증을 하지 않고 삭제만 하므로 --commit(검증 결과 반영),
    # --limit(검증 대상 제한)과 같이 쓰는 것은 의미가 없다. 조용히 무시하지 않고
    # 명시적으로 금지한다. --commit 과 --limit 은 함께 쓸 수 있어야 하므로 이 셋을
    # 하나의 mutually_exclusive_group 으로 묶지 않고 개별적으로 검사한다.
    if args.delete_failed and args.commit:
        parser.error("--delete-failed 는 --commit 과 함께 쓸 수 없습니다.")
    if args.delete_failed and args.limit is not None:
        parser.error("--delete-failed 는 --limit 과 함께 쓸 수 없습니다.")

    return args


if __name__ == "__main__":
    args = _parse_args()
    main(delete_failed=args.delete_failed, limit=args.limit, commit=args.commit)
