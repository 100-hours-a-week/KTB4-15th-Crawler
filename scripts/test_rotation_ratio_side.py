"""Quality Gate를 통과한 실제 상품으로 rotation ratio와 SIDE 정도를 실험한다.

샘플은 seed로 정한 후보 순서에서 person 1개, garment 1개, landmarks 존재,
production Landmark Quality Gate 통과 조건만으로 채택한다. 방향 결과와
rotation_ratio는 채택 후에만 계산하므로 sampling에는 영향을 주지 않는다.

    python scripts/test_rotation_ratio_side.py --collect 100 --seed 42
    python scripts/test_rotation_ratio_side.py --analyze CSV_PATH

    python scripts/test_rotation_ratio_side.py --contact-sheet 100 --seed 44 --per-sheet 25

--collect-side-too-large 는 같은 Quality Gate를 통과한 후보를 사람에게 하나씩
보여주고, 사람이 실제로 SIDE_TOO_LARGE라고 판단한 사례만 목표 개수만큼 모은다.
후보를 보여줄지 말지는 person/garment/landmarks/Quality Gate로만 정하고,
rotation_ratio/abs(rotation_ratio)/current_mediapipe_direction/기존 FRONT·SIDE
판정 결과는 전혀 쓰지 않는다(채택이 끝난 뒤 CSV 기록용으로만 계산한다).

    python scripts/test_rotation_ratio_side.py --collect-side-too-large 15 --seed 42
"""

import argparse
import csv
import math
import random
import statistics
import sys
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from urllib.error import HTTPError, URLError

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.storage.postgres_storage import get_connection
from app.validation.detector import GroundingDinoDetector, get_detection_labels
from app.validation.pose import (
    FRONT_ROTATION_RATIO,
    SIDE_ROTATION_RATIO,
    BodyLandmarks,
    PoseEstimator,
    classify_direction,
    is_direction_landmark_reliable,
)
from app.validation.product_image_validator import download_image

RESULTS_DIR = PROJECT_ROOT / "validation_results" / "rotation_ratio_side"
HUMAN_GROUPS = ("FRONT_OR_BACK", "SIDE_ALLOWED", "SIDE_TOO_LARGE")
HUMAN_LABEL_CHOICES = ("SIDE_TOO_LARGE", "SKIP", "EXCLUDE")
CONTACT_SHEETS_DIR = RESULTS_DIR / "contact_sheets"
CONTACT_SHEET_COLUMNS = 5
CONTACT_CELL_WIDTH = 320
CONTACT_IMAGE_HEIGHT = 340
CONTACT_LABEL_HEIGHT = 60
CONTACT_CELL_HEIGHT = CONTACT_IMAGE_HEIGHT + CONTACT_LABEL_HEIGHT
CONTACT_CELL_PADDING = 10
CONTACT_BACKGROUND_RGB = (245, 245, 245)
CONTACT_TEXT_RGB = (20, 20, 20)
CONTACT_BORDER_RGB = (180, 180, 180)
CSV_COLUMNS = (
    "product_code",
    "product_name",
    "main_category",
    "sub_category",
    "image_url",
    "rotation_ratio",
    "abs_rotation_ratio",
    "current_mediapipe_direction",
    "left_shoulder_visibility",
    "right_shoulder_visibility",
    "left_hip_visibility",
    "right_hip_visibility",
    "torso_scale",
    "landmark_reliable",
    "human_side_group",
    "human_note",
)
CONTACT_CSV_COLUMNS = (
    "sheet",
    "position",
    "product_code",
    "product_name",
    "main_category",
    "sub_category",
    "image_url",
    "rotation_ratio",
    "abs_rotation_ratio",
    "current_mediapipe_direction",
)


@dataclass(frozen=True)
class Candidate:
    product_code: int
    product_name: str
    main_category: str
    sub_category: str
    image_url: str


@dataclass(frozen=True)
class CollectionResult:
    csv_path: Path
    summary: dict[str, int]


@dataclass(frozen=True)
class ContactSheetItem:
    row: dict[str, object]
    cell_rgb: np.ndarray


@dataclass(frozen=True)
class ContactSheetResult:
    csv_path: Path
    sheet_paths: tuple[Path, ...]
    summary: dict[str, int]


@dataclass(frozen=True)
class GroupStatistics:
    count: int
    mean: float
    median: float
    p25: float
    p75: float
    minimum: float
    maximum: float


@dataclass(frozen=True)
class IqrOverlap:
    first_group: str
    second_group: str
    lower: float
    upper: float
    width: float


@dataclass(frozen=True)
class ThresholdSweepResult:
    side_threshold: float
    front_threshold: float
    correct: int
    total: int
    accuracy: float


@dataclass(frozen=True)
class AnalysisResult:
    labeled_count: int
    unlabeled_count: int
    excluded_count: int
    groups: dict[str, GroupStatistics]
    median_order_holds: bool | None
    iqr_overlaps: tuple[IqrOverlap, ...]
    current_threshold_accuracy: float
    threshold_sweep: tuple[ThresholdSweepResult, ...]


def _load_candidates(connection) -> list[Candidate]:
    """DB를 수정하지 않고 상의/하의 후보를 고정된 순서로 읽는다."""
    with connection.cursor() as cursor:
        cursor.execute(
            "SELECT product_code, product_name, main_category, sub_category, image_url "
            "FROM products WHERE main_category IN ('상의', '하의') ORDER BY product_code"
        )
        rows = cursor.fetchall()
    return [Candidate(*row) for row in rows]


def _randomized_candidates(candidates: list[Candidate], *, seed: int) -> list[Candidate]:
    """gate/direction/rotation_ratio를 보지 않는 재현 가능한 후보 순서다."""
    randomized = list(candidates)
    random.Random(seed).shuffle(randomized)
    return randomized


def _rotation_ratio(landmarks: BodyLandmarks) -> float:
    return (
        landmarks.left_shoulder_x - landmarks.right_shoulder_x
    ) / landmarks.torso_scale


def _csv_row(candidate: Candidate, landmarks: BodyLandmarks) -> dict[str, object]:
    """gate 통과가 확정된 뒤에만 방향과 rotation ratio를 계산한다."""
    rotation_ratio = _rotation_ratio(landmarks)
    return {
        "product_code": candidate.product_code,
        "product_name": candidate.product_name,
        "main_category": candidate.main_category,
        "sub_category": candidate.sub_category,
        "image_url": candidate.image_url,
        "rotation_ratio": f"{rotation_ratio:.12f}",
        "abs_rotation_ratio": f"{abs(rotation_ratio):.12f}",
        "current_mediapipe_direction": classify_direction(landmarks).value,
        "left_shoulder_visibility": f"{landmarks.left_shoulder_visibility:.6f}",
        "right_shoulder_visibility": f"{landmarks.right_shoulder_visibility:.6f}",
        "left_hip_visibility": f"{landmarks.left_hip_visibility:.6f}",
        "right_hip_visibility": f"{landmarks.right_hip_visibility:.6f}",
        "torso_scale": f"{landmarks.torso_scale:.6f}",
        "landmark_reliable": True,
        "human_side_group": "",
        "human_note": "",
    }


def _fresh_summary() -> Counter:
    return Counter(
        {
            "candidate_attempts": 0,
            "category_mapping_missing": 0,
            "image_load_failed": 0,
            "person_count_not_one": 0,
            "garment_count_not_one": 0,
            "landmarks_none": 0,
            "landmark_quality_bad": 0,
            "collected": 0,
        }
    )


def _collect_rows(
    candidates: list[Candidate],
    *,
    target: int,
    detector,
    pose_estimator,
    image_loader=download_image,
    on_accept: Callable[
        [Candidate, np.ndarray, BodyLandmarks, dict[str, object]], None
    ]
    | None = None,
) -> tuple[list[dict[str, object]], dict[str, int]]:
    """방향/ratio와 무관하게 조건을 통과한 순서대로 target개를 수집한다."""
    rows: list[dict[str, object]] = []
    summary = _fresh_summary()
    for candidate in candidates:
        summary["candidate_attempts"] += 1
        labels = get_detection_labels(candidate.main_category, candidate.sub_category)
        if labels is None:
            summary["category_mapping_missing"] += 1
            continue
        try:
            image_rgb = image_loader(candidate.image_url)
        except (URLError, HTTPError, OSError, ValueError):
            summary["image_load_failed"] += 1
            continue

        detection = detector.detect(image_rgb, labels)
        if detection.person_count != 1:
            summary["person_count_not_one"] += 1
        if detection.garment_count != 1:
            summary["garment_count_not_one"] += 1
        if detection.person_count != 1 or detection.garment_count != 1:
            continue

        # 이 이미지에 대한 유일한 MediaPipe 호출이다.
        landmarks = pose_estimator.get_landmarks(image_rgb)
        if landmarks is None:
            summary["landmarks_none"] += 1
            continue
        if not is_direction_landmark_reliable(landmarks):
            summary["landmark_quality_bad"] += 1
            continue

        # sampling 조건을 모두 통과한 뒤에만 ratio와 current direction을 계산한다.
        row = _csv_row(candidate, landmarks)
        if on_accept is not None:
            on_accept(candidate, image_rgb, landmarks, row)
        rows.append(row)
        summary["collected"] += 1
        print(
            f"[{len(rows)}/{target}] product_code={candidate.product_code} "
            f"candidate_attempt={summary['candidate_attempts']}"
        )
        if len(rows) == target:
            return rows, dict(summary)

    raise RuntimeError(f"샘플을 {target}개 확보하지 못했습니다: {len(rows)}개")


def _write_collection(
    rows: list[dict[str, object]], *, collect: int, seed: int, results_dir: Path
) -> Path:
    results_dir.mkdir(parents=True, exist_ok=True)
    output_path = results_dir / f"rotation_ratio_side_seed{seed}_n{collect}.csv"
    with output_path.open("w", newline="", encoding="utf-8-sig") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def collect(
    sample_count: int,
    *,
    seed: int,
    results_dir: Path = RESULTS_DIR,
    connection=None,
    detector=None,
    pose_estimator=None,
    image_loader=download_image,
) -> CollectionResult:
    if sample_count < 1:
        raise ValueError("--collect는 1 이상이어야 합니다")

    owns_connection = connection is None
    connection = connection if connection is not None else get_connection()
    try:
        detector = detector if detector is not None else GroundingDinoDetector()
        owns_pose_estimator = pose_estimator is None
        pose_estimator = pose_estimator if pose_estimator is not None else PoseEstimator()
        try:
            candidates = _randomized_candidates(_load_candidates(connection), seed=seed)
            rows, summary = _collect_rows(
                candidates,
                target=sample_count,
                detector=detector,
                pose_estimator=pose_estimator,
                image_loader=image_loader,
            )
            csv_path = _write_collection(
                rows, collect=sample_count, seed=seed, results_dir=results_dir
            )
            return CollectionResult(csv_path=csv_path, summary=summary)
        finally:
            if owns_pose_estimator:
                pose_estimator.close()
    finally:
        if owns_connection:
            connection.close()


def _build_contact_cell(candidate: Candidate, image_rgb: np.ndarray) -> np.ndarray:
    """원본 비율을 보존한 썸네일과 편향 없는 식별 정보만 한 cell에 그린다."""
    cell = np.full(
        (CONTACT_CELL_HEIGHT, CONTACT_CELL_WIDTH, 3),
        CONTACT_BACKGROUND_RGB,
        dtype=np.uint8,
    )
    cv2.putText(
        cell,
        str(candidate.product_code),
        (CONTACT_CELL_PADDING, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        CONTACT_TEXT_RGB,
        2,
        cv2.LINE_AA,
    )
    category_text = "TOP" if candidate.main_category == "상의" else "BOTTOM"
    cv2.putText(
        cell,
        category_text,
        (CONTACT_CELL_PADDING, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        CONTACT_TEXT_RGB,
        1,
        cv2.LINE_AA,
    )

    source_height, source_width = image_rgb.shape[:2]
    available_width = CONTACT_CELL_WIDTH - CONTACT_CELL_PADDING * 2
    available_height = CONTACT_IMAGE_HEIGHT - CONTACT_CELL_PADDING * 2
    scale = min(available_width / source_width, available_height / source_height)
    resized_width = max(1, round(source_width * scale))
    resized_height = max(1, round(source_height * scale))
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    thumbnail = cv2.resize(
        image_rgb,
        (resized_width, resized_height),
        interpolation=interpolation,
    )
    image_x = (CONTACT_CELL_WIDTH - resized_width) // 2
    image_y = CONTACT_LABEL_HEIGHT + (CONTACT_IMAGE_HEIGHT - resized_height) // 2
    cell[image_y : image_y + resized_height, image_x : image_x + resized_width] = thumbnail
    cv2.rectangle(
        cell,
        (0, 0),
        (CONTACT_CELL_WIDTH - 1, CONTACT_CELL_HEIGHT - 1),
        CONTACT_BORDER_RGB,
        1,
    )
    return cell


def _contact_sheet_item(
    candidate: Candidate,
    image_rgb: np.ndarray,
    landmarks: BodyLandmarks,
    row: dict[str, object],
) -> ContactSheetItem:
    # row의 ratio/direction 값은 매핑 CSV에만 쓰고 이미지에는 절대 그리지 않는다.
    return ContactSheetItem(
        row=dict(row),
        cell_rgb=_build_contact_cell(candidate, image_rgb),
    )


def _write_contact_sheets(
    items: list[ContactSheetItem], *, per_sheet: int, output_dir: Path
) -> tuple[Path, ...]:
    output_dir.mkdir(parents=True, exist_ok=True)
    sheet_paths: list[Path] = []
    for sheet_index, start in enumerate(range(0, len(items), per_sheet), start=1):
        page_items = items[start : start + per_sheet]
        row_count = math.ceil(len(page_items) / CONTACT_SHEET_COLUMNS)
        sheet_rgb = np.full(
            (
                row_count * CONTACT_CELL_HEIGHT,
                CONTACT_SHEET_COLUMNS * CONTACT_CELL_WIDTH,
                3,
            ),
            CONTACT_BACKGROUND_RGB,
            dtype=np.uint8,
        )
        for item_index, item in enumerate(page_items):
            row_index, column_index = divmod(item_index, CONTACT_SHEET_COLUMNS)
            y_start = row_index * CONTACT_CELL_HEIGHT
            x_start = column_index * CONTACT_CELL_WIDTH
            sheet_rgb[
                y_start : y_start + CONTACT_CELL_HEIGHT,
                x_start : x_start + CONTACT_CELL_WIDTH,
            ] = item.cell_rgb

        sheet_path = output_dir / f"sheet_{sheet_index:02d}.jpg"
        if not cv2.imwrite(str(sheet_path), cv2.cvtColor(sheet_rgb, cv2.COLOR_RGB2BGR)):
            raise OSError(f"contact sheet 저장에 실패했습니다: {sheet_path}")
        sheet_paths.append(sheet_path)
    return tuple(sheet_paths)


def _write_contact_mapping(
    items: list[ContactSheetItem],
    *,
    per_sheet: int,
    seed: int,
    sample_count: int,
    output_dir: Path,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"contact_sheet_seed{seed}_n{sample_count}.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=CONTACT_CSV_COLUMNS)
        writer.writeheader()
        for index, item in enumerate(items):
            row = item.row
            writer.writerow(
                {
                    "sheet": f"sheet_{index // per_sheet + 1:02d}.jpg",
                    "position": index % per_sheet + 1,
                    "product_code": row["product_code"],
                    "product_name": row["product_name"],
                    "main_category": row["main_category"],
                    "sub_category": row["sub_category"],
                    "image_url": row["image_url"],
                    "rotation_ratio": row["rotation_ratio"],
                    "abs_rotation_ratio": row["abs_rotation_ratio"],
                    "current_mediapipe_direction": row[
                        "current_mediapipe_direction"
                    ],
                }
            )
    return csv_path


def create_contact_sheets(
    sample_count: int,
    *,
    seed: int,
    per_sheet: int,
    results_dir: Path = RESULTS_DIR,
    connection=None,
    detector=None,
    pose_estimator=None,
    image_loader=download_image,
) -> ContactSheetResult:
    if sample_count < 1:
        raise ValueError("--contact-sheet는 1 이상이어야 합니다")
    if not 20 <= per_sheet <= 25:
        raise ValueError("--per-sheet는 20 이상 25 이하여야 합니다")

    owns_connection = connection is None
    connection = connection if connection is not None else get_connection()
    try:
        detector = detector if detector is not None else GroundingDinoDetector()
        owns_pose_estimator = pose_estimator is None
        pose_estimator = pose_estimator if pose_estimator is not None else PoseEstimator()
        try:
            candidates = _randomized_candidates(_load_candidates(connection), seed=seed)
            items: list[ContactSheetItem] = []

            def remember_accepted(
                candidate: Candidate,
                image_rgb: np.ndarray,
                landmarks: BodyLandmarks,
                row: dict[str, object],
            ) -> None:
                items.append(_contact_sheet_item(candidate, image_rgb, landmarks, row))

            rows, summary = _collect_rows(
                candidates,
                target=sample_count,
                detector=detector,
                pose_estimator=pose_estimator,
                image_loader=image_loader,
                on_accept=remember_accepted,
            )
            if len(rows) != len(items):
                raise RuntimeError("contact sheet 샘플과 CSV 행 개수가 일치하지 않습니다")

            contact_root = results_dir / "contact_sheets"
            sheet_dir = contact_root / f"seed{seed}_n{sample_count}_per{per_sheet}"
            sheet_paths = _write_contact_sheets(
                items, per_sheet=per_sheet, output_dir=sheet_dir
            )
            csv_path = _write_contact_mapping(
                items,
                per_sheet=per_sheet,
                seed=seed,
                sample_count=sample_count,
                output_dir=contact_root,
            )
            return ContactSheetResult(
                csv_path=csv_path,
                sheet_paths=sheet_paths,
                summary=summary,
            )
        finally:
            if owns_pose_estimator:
                pose_estimator.close()
    finally:
        if owns_connection:
            connection.close()


def _fresh_side_too_large_summary() -> Counter:
    return Counter(
        {
            "scanned": 0,
            "category_mapping_missing": 0,
            "image_load_failed": 0,
            "person_count_not_one": 0,
            "garment_count_not_one": 0,
            "landmarks_none": 0,
            "landmark_quality_bad": 0,
            "quality_gate_passed": 0,
            "side_too_large_collected": 0,
            "skipped": 0,
            "excluded": 0,
        }
    )


def _prompt_human_label(candidate: Candidate, *, input_fn=input) -> str:
    """사람이 볼 수 있게 후보 정보만 보여준다.

    rotation_ratio/abs_rotation_ratio/current_mediapipe_direction 등 모델이 낸 값은
    일부러 보여주지 않는다 — 사람이 모델 출력에 영향받지 않고 이미지만 보고
    판단하게 하기 위해서다.
    """
    print(f"product_code = {candidate.product_code}")
    print(f"product_name = {candidate.product_name}")
    print(f"main_category = {candidate.main_category}")
    print(f"sub_category = {candidate.sub_category}")
    print(f"image_url = {candidate.image_url}")
    while True:
        raw = input_fn("SIDE_TOO_LARGE / SKIP / EXCLUDE > ")
        normalized = raw.strip().upper()
        if normalized in HUMAN_LABEL_CHOICES:
            return normalized
        print(
            f"입력을 이해하지 못했습니다: {raw!r}. "
            f"{'/'.join(HUMAN_LABEL_CHOICES)} 중 하나를 입력하세요."
        )


def _collect_side_too_large_rows(
    candidates: list[Candidate],
    *,
    target: int,
    detector,
    pose_estimator,
    image_loader=download_image,
    input_fn=input,
) -> tuple[list[dict[str, object]], dict[str, int]]:
    """기존 --collect와 같은 Quality Gate(person 1개, garment 1개, landmarks 존재,
    is_direction_landmark_reliable)만으로 후보를 사람에게 보여줄지 정한다.

    rotation_ratio/abs_rotation_ratio/current_mediapipe_direction/기존 FRONT·SIDE
    판정 결과는 이 함수의 어떤 분기에도 쓰지 않는다 — gate를 통과한 뒤 CSV 한 줄을
    만들 때만(_csv_row) 계산한다. 사람이 SIDE_TOO_LARGE라고 답한 행만 수집하고,
    target개가 모이면 멈춘다. 후보를 다 돌았거나 Ctrl-C로 중단해도 예외를 던지지
    않고 그때까지 모은 결과를 그대로 돌려준다.
    """
    rows: list[dict[str, object]] = []
    summary = _fresh_side_too_large_summary()
    try:
        for candidate in candidates:
            summary["scanned"] += 1
            labels = get_detection_labels(candidate.main_category, candidate.sub_category)
            if labels is None:
                summary["category_mapping_missing"] += 1
                continue
            try:
                image_rgb = image_loader(candidate.image_url)
            except (URLError, HTTPError, OSError, ValueError):
                summary["image_load_failed"] += 1
                continue

            detection = detector.detect(image_rgb, labels)
            if detection.person_count != 1:
                summary["person_count_not_one"] += 1
            if detection.garment_count != 1:
                summary["garment_count_not_one"] += 1
            if detection.person_count != 1 or detection.garment_count != 1:
                continue

            # 이 이미지에 대한 유일한 MediaPipe 호출이다.
            landmarks = pose_estimator.get_landmarks(image_rgb)
            if landmarks is None:
                summary["landmarks_none"] += 1
                continue
            if not is_direction_landmark_reliable(landmarks):
                summary["landmark_quality_bad"] += 1
                continue

            summary["quality_gate_passed"] += 1
            # gate 통과가 확정된 뒤에만 ratio/방향을 계산한다(사람에게는 보여주지 않는다).
            row = _csv_row(candidate, landmarks)
            label = _prompt_human_label(candidate, input_fn=input_fn)
            if label == "SIDE_TOO_LARGE":
                row["human_side_group"] = "SIDE_TOO_LARGE"
                rows.append(row)
                summary["side_too_large_collected"] += 1
                print(
                    f"[{summary['side_too_large_collected']}/{target}] "
                    f"product_code={candidate.product_code} 수집됨"
                )
                if summary["side_too_large_collected"] == target:
                    break
            elif label == "SKIP":
                summary["skipped"] += 1
            else:
                summary["excluded"] += 1
    except KeyboardInterrupt:
        print("\n중단됨 (Ctrl-C). 지금까지 모은 결과를 저장합니다.")
    return rows, dict(summary)


def _write_side_too_large_collection(
    rows: list[dict[str, object]], *, target: int, seed: int, results_dir: Path
) -> Path:
    results_dir.mkdir(parents=True, exist_ok=True)
    output_path = results_dir / f"side_too_large_seed{seed}_n{target}.csv"
    with output_path.open("w", newline="", encoding="utf-8-sig") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def collect_side_too_large(
    target: int,
    *,
    seed: int,
    results_dir: Path = RESULTS_DIR,
    connection=None,
    detector=None,
    pose_estimator=None,
    image_loader=download_image,
    input_fn=input,
) -> CollectionResult:
    """사람이 SIDE_TOO_LARGE라고 분류한 사례가 target개 모일 때까지 수집한다.

    DB를 다 돌았거나 Ctrl-C로 중단해도 예외를 던지지 않고, 그때까지 모은 결과를
    CSV로 저장한 뒤 CollectionResult를 돌려준다(목표에 못 미쳐도 실패로 보지
    않는다 — 사람이 직접 보고 진행하는 수집이기 때문이다).
    """
    if target < 1:
        raise ValueError("--collect-side-too-large는 1 이상이어야 합니다")

    owns_connection = connection is None
    connection = connection if connection is not None else get_connection()
    try:
        detector = detector if detector is not None else GroundingDinoDetector()
        owns_pose_estimator = pose_estimator is None
        pose_estimator = pose_estimator if pose_estimator is not None else PoseEstimator()
        try:
            candidates = _randomized_candidates(_load_candidates(connection), seed=seed)
            rows, summary = _collect_side_too_large_rows(
                candidates,
                target=target,
                detector=detector,
                pose_estimator=pose_estimator,
                image_loader=image_loader,
                input_fn=input_fn,
            )
            csv_path = _write_side_too_large_collection(
                rows, target=target, seed=seed, results_dir=results_dir
            )
            return CollectionResult(csv_path=csv_path, summary=summary)
        finally:
            if owns_pose_estimator:
                pose_estimator.close()
    finally:
        if owns_connection:
            connection.close()


def _print_side_too_large_summary(summary: dict[str, int]) -> None:
    print("side_too_large collection summary:")
    for key in (
        "scanned",
        "category_mapping_missing",
        "image_load_failed",
        "person_count_not_one",
        "garment_count_not_one",
        "landmarks_none",
        "landmark_quality_bad",
        "quality_gate_passed",
        "side_too_large_collected",
        "skipped",
        "excluded",
    ):
        print(f"  {key}: {summary[key]}")


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    if lower_index == upper_index:
        return ordered[lower_index]
    fraction = position - lower_index
    return ordered[lower_index] + (ordered[upper_index] - ordered[lower_index]) * fraction


def _group_statistics(values: list[float]) -> GroupStatistics:
    return GroupStatistics(
        count=len(values),
        mean=statistics.fmean(values),
        median=statistics.median(values),
        p25=_percentile(values, 0.25),
        p75=_percentile(values, 0.75),
        minimum=min(values),
        maximum=max(values),
    )


def _normalize_human_group(value: str) -> str | None:
    normalized = value.strip().upper()
    if not normalized:
        return None
    if normalized in {"FRONT", "BACK", "FRONT_OR_BACK"}:
        return "FRONT_OR_BACK"
    if normalized in {"SIDE_ALLOWED", "SIDE_TOO_LARGE"}:
        return normalized
    raise ValueError(f"지원하지 않는 human_side_group입니다: {value!r}")


def _predicted_group(abs_rotation_ratio: float, *, side: float, front: float) -> str:
    if abs_rotation_ratio >= front:
        return "FRONT_OR_BACK"
    if abs_rotation_ratio >= side:
        return "SIDE_ALLOWED"
    return "SIDE_TOO_LARGE"


def _accuracy(labeled_values: list[tuple[float, str]], *, side: float, front: float) -> float:
    if not labeled_values:
        return 0.0
    correct = sum(
        _predicted_group(value, side=side, front=front) == group
        for value, group in labeled_values
    )
    return correct / len(labeled_values)


def _threshold_candidates(values: list[float]) -> list[float]:
    unique_values = sorted(set(values))
    if not unique_values:
        return []
    candidates = [0.0]
    if unique_values[0] > 0:
        candidates.append(unique_values[0] / 2)
    candidates.extend(
        (lower + upper) / 2 for lower, upper in pairwise(unique_values)
    )
    candidates.append(unique_values[-1] + 1e-12)
    return sorted(set(candidates))


def _threshold_sweep(
    labeled_values: list[tuple[float, str]], *, limit: int = 10
) -> tuple[ThresholdSweepResult, ...]:
    candidates = _threshold_candidates([value for value, _ in labeled_values])
    results: list[ThresholdSweepResult] = []
    total = len(labeled_values)
    for side_threshold in candidates:
        for front_threshold in candidates:
            if side_threshold >= front_threshold:
                continue
            correct = sum(
                _predicted_group(
                    value, side=side_threshold, front=front_threshold
                )
                == group
                for value, group in labeled_values
            )
            results.append(
                ThresholdSweepResult(
                    side_threshold=side_threshold,
                    front_threshold=front_threshold,
                    correct=correct,
                    total=total,
                    accuracy=correct / total if total else 0.0,
                )
            )
    results.sort(
        key=lambda result: (
            -result.accuracy,
            abs(result.side_threshold - SIDE_ROTATION_RATIO)
            + abs(result.front_threshold - FRONT_ROTATION_RATIO),
            result.side_threshold,
            result.front_threshold,
        )
    )
    return tuple(results[:limit])


def _iqr_overlap(
    first_group: str,
    first: GroupStatistics,
    second_group: str,
    second: GroupStatistics,
) -> IqrOverlap:
    lower = max(first.p25, second.p25)
    upper = min(first.p75, second.p75)
    width = max(0.0, upper - lower)
    return IqrOverlap(
        first_group=first_group,
        second_group=second_group,
        lower=lower,
        upper=upper,
        width=width,
    )


def analyze(csv_path: Path) -> AnalysisResult:
    with csv_path.open(newline="", encoding="utf-8-sig") as input_file:
        reader = csv.DictReader(input_file)
        required_columns = {"abs_rotation_ratio", "human_side_group"}
        missing_columns = required_columns - set(reader.fieldnames or ())
        if missing_columns:
            raise ValueError(f"CSV 필수 컬럼이 없습니다: {sorted(missing_columns)}")
        rows = list(reader)

    values_by_group: dict[str, list[float]] = {group: [] for group in HUMAN_GROUPS}
    labeled_values: list[tuple[float, str]] = []
    unlabeled_count = 0
    excluded_count = 0
    for row in rows:
        # 과거 실험 CSV와의 호환: 명시적으로 invalid 처리된 행은 분석에서 제외한다.
        if "human_sample_valid" in row and row["human_sample_valid"].strip().upper() == "NO":
            excluded_count += 1
            continue
        group = _normalize_human_group(row["human_side_group"])
        if group is None:
            unlabeled_count += 1
            continue
        value = float(row["abs_rotation_ratio"])
        values_by_group[group].append(value)
        labeled_values.append((value, group))

    if not labeled_values:
        raise ValueError("분석할 human_side_group 라벨이 없습니다")

    groups = {
        group: _group_statistics(values)
        for group, values in values_by_group.items()
        if values
    }
    if all(group in groups for group in HUMAN_GROUPS):
        median_order_holds: bool | None = (
            groups["FRONT_OR_BACK"].median
            > groups["SIDE_ALLOWED"].median
            > groups["SIDE_TOO_LARGE"].median
        )
        overlaps = (
            _iqr_overlap(
                "FRONT_OR_BACK",
                groups["FRONT_OR_BACK"],
                "SIDE_ALLOWED",
                groups["SIDE_ALLOWED"],
            ),
            _iqr_overlap(
                "SIDE_ALLOWED",
                groups["SIDE_ALLOWED"],
                "SIDE_TOO_LARGE",
                groups["SIDE_TOO_LARGE"],
            ),
        )
    else:
        median_order_holds = None
        overlaps = ()

    return AnalysisResult(
        labeled_count=len(labeled_values),
        unlabeled_count=unlabeled_count,
        excluded_count=excluded_count,
        groups=groups,
        median_order_holds=median_order_holds,
        iqr_overlaps=overlaps,
        current_threshold_accuracy=_accuracy(
            labeled_values,
            side=SIDE_ROTATION_RATIO,
            front=FRONT_ROTATION_RATIO,
        ),
        threshold_sweep=_threshold_sweep(labeled_values),
    )


def _print_collection_summary(summary: dict[str, int]) -> None:
    print("collection summary:")
    for key in (
        "candidate_attempts",
        "category_mapping_missing",
        "image_load_failed",
        "person_count_not_one",
        "garment_count_not_one",
        "landmarks_none",
        "landmark_quality_bad",
        "collected",
    ):
        print(f"  {key}: {summary[key]}")


def _print_analysis(result: AnalysisResult) -> None:
    print(f"labeled: {result.labeled_count}")
    print(f"unlabeled_ignored: {result.unlabeled_count}")
    print(f"excluded: {result.excluded_count}")
    for group in HUMAN_GROUPS:
        statistics_for_group = result.groups.get(group)
        if statistics_for_group is None:
            print(f"{group}: count=0")
            continue
        print(f"{group}:")
        print(f"  count: {statistics_for_group.count}")
        print(f"  mean: {statistics_for_group.mean:.6f}")
        print(f"  median: {statistics_for_group.median:.6f}")
        print(f"  p25: {statistics_for_group.p25:.6f}")
        print(f"  p75: {statistics_for_group.p75:.6f}")
        print(f"  min: {statistics_for_group.minimum:.6f}")
        print(f"  max: {statistics_for_group.maximum:.6f}")
    print(f"median_order_holds: {result.median_order_holds}")
    print("IQR overlaps:")
    if not result.iqr_overlaps:
        print("  insufficient groups")
    for overlap in result.iqr_overlaps:
        interval = (
            f"[{overlap.lower:.6f}, {overlap.upper:.6f}]"
            if overlap.width > 0
            else "none"
        )
        print(
            f"  {overlap.first_group} vs {overlap.second_group}: "
            f"width={overlap.width:.6f} interval={interval}"
        )
    print(
        "current_threshold_accuracy: "
        f"{result.current_threshold_accuracy:.4f} "
        f"(side={SIDE_ROTATION_RATIO:.2f}, front={FRONT_ROTATION_RATIO:.2f})"
    )
    print("exploratory_threshold_sweep_top:")
    for sweep in result.threshold_sweep:
        print(
            f"  side={sweep.side_threshold:.6f} front={sweep.front_threshold:.6f} "
            f"accuracy={sweep.accuracy:.4f} ({sweep.correct}/{sweep.total})"
        )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--collect", type=int, metavar="N")
    mode.add_argument(
        "--collect-side-too-large",
        type=int,
        metavar="N",
        dest="collect_side_too_large",
    )
    mode.add_argument("--contact-sheet", type=int, metavar="N", dest="contact_sheet")
    mode.add_argument("--analyze", type=Path, metavar="CSV_PATH")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--per-sheet", type=int, default=25)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    if args.collect is not None:
        result = collect(args.collect, seed=args.seed)
        _print_collection_summary(result.summary)
        print(f"CSV 생성 완료: {result.csv_path}")
        return
    if args.collect_side_too_large is not None:
        result = collect_side_too_large(args.collect_side_too_large, seed=args.seed)
        _print_side_too_large_summary(result.summary)
        print(f"CSV 생성 완료: {result.csv_path}")
        return
    if args.contact_sheet is not None:
        result = create_contact_sheets(
            args.contact_sheet,
            seed=args.seed,
            per_sheet=args.per_sheet,
        )
        _print_collection_summary(result.summary)
        for sheet_path in result.sheet_paths:
            print(f"contact sheet 생성 완료: {sheet_path}")
        print(f"매핑 CSV 생성 완료: {result.csv_path}")
        return
    try:
        result = analyze(args.analyze)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    _print_analysis(result)


if __name__ == "__main__":
    main()
