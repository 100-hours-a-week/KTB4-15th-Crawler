"""29CM 상품 이미지를 로컬에서 검증하는 데이터 정제 도구.

운영 크롤러나 API 로 배포하는 기능이 아니다. Grounding DINO/MediaPipe 같은 무거운
의존성은 requirements-validation.txt 에만 있고 운영 requirements.txt 에는 없다.

기본 실행 = dry-run: PENDING 상품을 검증하고 CSV 를 만들지만 DB 는 바꾸지 않는다.

    python scripts/validate_product_images.py --limit 50

결과를 확인한 뒤 실제로 DB 에 반영하려면 --commit 을 붙인다(PASS/FAIL 기록).

    python scripts/validate_product_images.py --limit 50 --commit

사람이 fail_products.csv 를 확인한 뒤에만 FAIL 상품을 지운다. 검증은 하지 않는다.

    python scripts/validate_product_images.py --delete-failed

특정 상품 하나가 왜 그렇게 판정됐는지 볼 때는 --product-code 로 그 상품 하나만
검증한다(DB 는 바꾸지 않는다). --debug 를 같이 주면 person/garment 수, bounding box,
landmark visibility 등 중간값을 전부 출력하고, person(빨강)/garment
(초록) box 를 그린 validation_results/debug_<product_code>.jpg 를 저장한다.

    python scripts/validate_product_images.py --product-code 149369 --debug
"""

import argparse
import csv
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError

import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.models.product import Product
from app.storage.postgres_storage import get_connection
from app.validation import rules, storage
from app.validation.detector import (
    BOTTOM_MAIN_CATEGORY,
    GroundingDinoDetector,
    check_cropped_lower_body,
    get_detection_labels,
)
from app.validation.models import DetectionResult, PoseDirection, ValidationResult
from app.validation.pose import (
    PoseEstimator,
    classify_direction,
    has_sufficient_lower_body_landmarks,
    has_sufficient_upper_body_landmarks,
)
from app.validation.product_image_validator import (
    download_image,
    validate_product_image,
)

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

# --debug 디버그 이미지 시각화 전용 값이다. Grounding DINO 탐지/threshold/NMS 에는
# 영향을 주지 않는다. 색상은 OpenCV 관례대로 BGR 순서다.
_PERSON_BOX_COLOR_BGR = (0, 0, 255)  # 빨강
_GARMENT_BOX_COLOR_BGR = (0, 255, 0)  # 초록
_BOX_THICKNESS = 2
_LABEL_FONT = cv2.FONT_HERSHEY_SIMPLEX
_LABEL_FONT_SCALE = 0.5
_LABEL_THICKNESS = 1


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


def _print_detections(label: str, detections) -> None:
    """--debug 전용. Grounding DINO 최종 중복 제거 후 detection 하나씩을 출력한다."""
    for index, detection in enumerate(detections):
        print(f"{label}[{index}] label = {detection.label}")
        print(f"{label}[{index}] score = {detection.score:.3f}")
        print(f"{label}[{index}] box = {list(detection.box)}")


def _draw_detections(image_bgr, detections, *, color) -> None:
    for detection in detections:
        x_min, y_min, x_max, y_max = (round(value) for value in detection.box)
        cv2.rectangle(image_bgr, (x_min, y_min), (x_max, y_max), color, _BOX_THICKNESS)
        text = f"{detection.label} {detection.score:.2f}"
        text_origin = (x_min, max(y_min - 5, 10))
        cv2.putText(
            image_bgr, text, text_origin, _LABEL_FONT, _LABEL_FONT_SCALE, color, _LABEL_THICKNESS
        )


def _save_debug_image(
    image_rgb,
    detection: DetectionResult,
    product_code: int,
    results_dir: Path = RESULTS_DIR,
) -> Path:
    """person(빨강)/garment(초록) box 를 그려 validation_results/debug_<code>.jpg 로 저장한다.

    시각화 전용이며 PASS/FAIL 판정에는 전혀 관여하지 않는다. --debug 일 때만 호출한다.
    """
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    _draw_detections(image_bgr, detection.persons, color=_PERSON_BOX_COLOR_BGR)
    _draw_detections(image_bgr, detection.garments, color=_GARMENT_BOX_COLOR_BGR)

    results_dir.mkdir(parents=True, exist_ok=True)
    debug_path = results_dir / f"debug_{product_code}.jpg"
    cv2.imwrite(str(debug_path), image_bgr)
    return debug_path


def _debug_validate_product(product: Product, *, detector, pose_estimator) -> None:
    """--product-code --debug 전용. 중간 판정값을 전부 출력한다.

    validate_product_image() 를 그대로 호출하지 않는 이유: 그 함수 내부에서만 쓰고
    버리는 값(detection label, person/garment 수, allow_lower_body_fallback, landmark,
    pose_direction)을 보여줘야 하기 때문이다. 같은 detector/pose/rules
    함수를 같은 순서로 그대로 호출할 뿐, 판정 로직 자체는 바꾸지 않는다.
    """
    print(f"product_code = {product.product_code}")
    print(f"main_category = {product.main_category}")
    print(f"sub_category = {product.sub_category}")

    labels = get_detection_labels(product.main_category, product.sub_category)
    print(f"detection labels = {labels}")
    if labels is None:
        print("final validation_status = FAIL")
        print("final validation_reason = CATEGORY_MAPPING_NOT_FOUND")
        return

    try:
        image_rgb = download_image(product.image_url)
    except (URLError, HTTPError, OSError, ValueError) as error:
        print(f"이미지 다운로드/디코딩 실패: {error}")
        print("final validation_status = FAIL")
        print("final validation_reason = IMAGE_LOAD_FAILED")
        return

    detection = detector.detect(image_rgb, labels)
    height, width = image_rgb.shape[:2]
    print(f"image width = {width}")
    print(f"image height = {height}")
    print(f"person_count = {detection.person_count}")
    print(f"garment_count = {detection.garment_count}")
    _print_detections("person", detection.persons)
    _print_detections("garment", detection.garments)

    debug_image_path = _save_debug_image(image_rgb, detection, product.product_code)
    print(f"debug image = {debug_image_path}")

    pose_direction = None
    if detection.person_count == 1:
        allow_lower_body_fallback = (
            product.main_category == BOTTOM_MAIN_CATEGORY and detection.garment_count == 1
        )
        print(f"allow_lower_body_fallback = {allow_lower_body_fallback}")

        landmarks = pose_estimator.get_landmarks(image_rgb)
        if landmarks is None:
            print("landmarks = None (MediaPipe 가 사람을 감지하지 못했습니다)")
            if allow_lower_body_fallback:
                # 하의 + garment 1개인데 MediaPipe 가 사람을 아예 못 잡은 경우다.
                # classify_direction(None, ...) 은 항상 UNCERTAIN 이므로(그대로 둔다),
                # bbox 기반으로 한 번 더 확인해서 왜 fallback 이 적용/미적용됐는지 보여준다.
                crop_check = check_cropped_lower_body(
                    person_box=detection.persons[0].box,
                    garment_box=detection.garments[0].box,
                    image_height=height,
                )
                print(f"person_top_ratio = {crop_check.person_top_ratio:.4f}")
                print(f"garment_top_ratio = {crop_check.garment_top_ratio:.4f}")
                print(f"garment_person_area_ratio = {crop_check.garment_person_area_ratio:.4f}")
                print(f"garment_in_person_ratio = {crop_check.garment_in_person_ratio:.4f}")
                print(f"is_cropped_lower_body = {crop_check.is_cropped_lower_body}")
                pose_direction = (
                    PoseDirection.FRONT
                    if crop_check.is_cropped_lower_body
                    else PoseDirection.UNCERTAIN
                )
            else:
                pose_direction = PoseDirection.UNCERTAIN
        else:
            print(f"nose visibility = {landmarks.nose_visibility:.3f}")
            print(f"left shoulder visibility = {landmarks.left_shoulder_visibility:.3f}")
            print(f"right shoulder visibility = {landmarks.right_shoulder_visibility:.3f}")
            print(f"left hip visibility = {landmarks.left_hip_visibility:.3f}")
            print(f"right hip visibility = {landmarks.right_hip_visibility:.3f}")
            print(f"left knee visibility = {landmarks.left_knee_visibility:.3f}")
            print(f"right knee visibility = {landmarks.right_knee_visibility:.3f}")
            print(f"left ankle visibility = {landmarks.left_ankle_visibility:.3f}")
            print(f"right ankle visibility = {landmarks.right_ankle_visibility:.3f}")
            print(
                "has_sufficient_upper_body_landmarks = "
                f"{has_sufficient_upper_body_landmarks(landmarks)}"
            )
            print(
                "has_sufficient_lower_body_landmarks = "
                f"{has_sufficient_lower_body_landmarks(landmarks)}"
            )
            pose_direction = classify_direction(
                landmarks, allow_lower_body_fallback=allow_lower_body_fallback
            )

        print(f"pose_direction = {pose_direction.value}")
    else:
        print("person_count != 1 이므로 MediaPipe 를 실행하지 않습니다.")

    result = rules.decide(detection, pose_direction)
    print(f"final validation_status = {result.status.value}")
    print(f"final validation_reason = {result.reason.value}")


def _run_single_product(connection, product_code: int, *, debug: bool) -> None:
    """--product-code 전용. 이 상품 하나만 조회해서 검증한다. DB 는 바꾸지 않는다."""
    product = storage.get_product_by_code(connection, product_code)
    if product is None:
        print(f"product_code={product_code} 상품을 찾을 수 없습니다.")
        return

    print("모델 로딩 중 (Grounding DINO, MediaPipe)...")
    detector = GroundingDinoDetector()
    with PoseEstimator() as pose_estimator:
        if debug:
            _debug_validate_product(product, detector=detector, pose_estimator=pose_estimator)
        else:
            result = validate_product_image(
                product, detector=detector, pose_estimator=pose_estimator
            )
            status_line = f"product_code={product.product_code} {result.status.value}"
            if not result.passed:
                status_line += f" reason={result.reason.value}"
            print(status_line)

    print("dry-run 입니다. DB 의 validation_status/validation_reason 은 바꾸지 않았습니다.")


def main(
    *,
    delete_failed: bool = False,
    limit: int | None = None,
    commit: bool = False,
    product_code: int | None = None,
    debug: bool = False,
) -> None:
    connection = get_connection()
    try:
        if delete_failed:
            _delete_failed(connection)
            return

        if product_code is not None:
            _run_single_product(connection, product_code, debug=debug)
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
    parser.add_argument(
        "--product-code",
        type=_positive_int,
        default=None,
        metavar="CODE",
        help="이 상품 하나만 조회해서 검증한다(PENDING 여부와 무관). 항상 dry-run 이다.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="--product-code 와 함께 써서 중간 판정값(landmark 등)을 출력한다.",
    )
    args = parser.parse_args(argv)

    # --delete-failed 는 검증을 하지 않고 삭제만 하므로 --commit(검증 결과 반영),
    # --limit(검증 대상 제한), --product-code 와 같이 쓰는 것은 의미가 없다. 조용히
    # 무시하지 않고 명시적으로 금지한다. --commit 과 --limit 은 함께 쓸 수 있어야
    # 하므로 이들을 하나의 mutually_exclusive_group 으로 묶지 않고 개별적으로 검사한다.
    if args.delete_failed and args.commit:
        parser.error("--delete-failed 는 --commit 과 함께 쓸 수 없습니다.")
    if args.delete_failed and args.limit is not None:
        parser.error("--delete-failed 는 --limit 과 함께 쓸 수 없습니다.")
    if args.delete_failed and args.product_code is not None:
        parser.error("--delete-failed 는 --product-code 와 함께 쓸 수 없습니다.")
    # --product-code 는 상품 하나만 보는 dry-run 전용 조회다. 배치 대상 제한(--limit)이나
    # DB 반영(--commit)과는 의미가 겹치거나 충돌하므로 함께 쓸 수 없게 한다.
    if args.product_code is not None and args.limit is not None:
        parser.error("--product-code 는 --limit 과 함께 쓸 수 없습니다.")
    if args.product_code is not None and args.commit:
        parser.error("--product-code 는 --commit 과 함께 쓸 수 없습니다(항상 dry-run 이다).")
    if args.debug and args.product_code is None:
        parser.error("--debug 는 --product-code 와 함께 써야 합니다.")

    return args


if __name__ == "__main__":
    args = _parse_args()
    main(
        delete_failed=args.delete_failed,
        limit=args.limit,
        commit=args.commit,
        product_code=args.product_code,
        debug=args.debug,
    )
