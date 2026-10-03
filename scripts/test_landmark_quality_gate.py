"""실제 상품 이미지로 landmark quality gate 를 수집하고 사람이 라벨링한 결과를 분석한다.

수집은 DB 를 읽기만 하며, seed 로 섞은 후보 순서에서 다음 조건을 만족하는 상품만
상의/하의별로 채택한다: person 1개, garment 1개, MediaPipe landmarks 존재.
gate 결과나 방향 판정 결과는 샘플 선택에 사용하지 않는다.

    python scripts/test_landmark_quality_gate.py --collect 30 --seed 42
    python scripts/test_landmark_quality_gate.py --analyze \
        validation_results/landmark_quality/landmark_quality_seed42_n30.csv
"""

import argparse
import csv
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from urllib.error import HTTPError, URLError

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.storage.postgres_storage import get_connection
from app.validation.detector import GroundingDinoDetector, get_detection_labels
from app.validation.pose import (
    BodyLandmarks,
    PoseEstimator,
    is_direction_landmark_reliable,
)
from app.validation.product_image_validator import download_image

RESULTS_DIR = PROJECT_ROOT / "validation_results" / "landmark_quality"
UPPER_CATEGORY = "상의"
LOWER_CATEGORY = "하의"
CSV_COLUMNS = (
    "product_code",
    "product_name",
    "main_category",
    "sub_category",
    "image_url",
    "left_shoulder_visibility",
    "right_shoulder_visibility",
    "left_hip_visibility",
    "right_hip_visibility",
    "torso_scale",
    "landmark_reliable",
    "human_expected_reliable",
    "human_note",
)


@dataclass(frozen=True)
class Candidate:
    product_code: int
    product_name: str
    main_category: str
    sub_category: str
    image_url: str


@dataclass(frozen=True)
class AnalysisResult:
    tp: int
    tn: int
    fp: int
    fn: int
    accuracy: float
    precision: float
    recall: float
    specificity: float
    mismatches: tuple[dict[str, str], ...]


def _load_candidates(connection, main_category: str) -> list[Candidate]:
    """고정된 product_code 순서로 후보를 SELECT만 한다."""
    with connection.cursor() as cursor:
        cursor.execute(
            "SELECT product_code, product_name, main_category, sub_category, image_url "
            "FROM products WHERE main_category = %s ORDER BY product_code",
            (main_category,),
        )
        rows = cursor.fetchall()
    return [Candidate(*row) for row in rows]


def _randomized_candidates(
    candidates: list[Candidate], *, seed: int, main_category: str
) -> list[Candidate]:
    """gate/direction 값과 무관한 재현 가능한 후보 순서를 만든다."""
    randomized = list(candidates)
    category_offset = 0 if main_category == UPPER_CATEGORY else 1
    random.Random(seed * 2 + category_offset).shuffle(randomized)
    return randomized


def _csv_row(candidate: Candidate, landmarks: BodyLandmarks) -> dict[str, object]:
    return {
        "product_code": candidate.product_code,
        "product_name": candidate.product_name,
        "main_category": candidate.main_category,
        "sub_category": candidate.sub_category,
        "image_url": candidate.image_url,
        "left_shoulder_visibility": f"{landmarks.left_shoulder_visibility:.6f}",
        "right_shoulder_visibility": f"{landmarks.right_shoulder_visibility:.6f}",
        "left_hip_visibility": f"{landmarks.left_hip_visibility:.6f}",
        "right_hip_visibility": f"{landmarks.right_hip_visibility:.6f}",
        "torso_scale": f"{landmarks.torso_scale:.6f}",
        "landmark_reliable": is_direction_landmark_reliable(landmarks),
        "human_expected_reliable": "",
        "human_note": "",
    }


def _collect_category(
    candidates: list[Candidate],
    *,
    target: int,
    detector,
    pose_estimator,
    image_loader=download_image,
) -> list[dict[str, object]]:
    """요청된 세 조건만으로 한 카테고리의 샘플을 채택한다.

    person/garment count 를 먼저 확인하고, 해당 조건을 통과한 이미지에서만
    get_landmarks()를 정확히 한 번 호출한다. None이면 제외하고 다음 후보로 간다.
    """
    rows: list[dict[str, object]] = []
    for attempt, candidate in enumerate(candidates, start=1):
        labels = get_detection_labels(candidate.main_category, candidate.sub_category)
        if labels is None:
            continue
        try:
            image_rgb = image_loader(candidate.image_url)
        except (URLError, HTTPError, OSError, ValueError):
            continue

        detection = detector.detect(image_rgb, labels)
        if detection.person_count != 1 or detection.garment_count != 1:
            continue

        landmarks = pose_estimator.get_landmarks(image_rgb)
        if landmarks is None:
            continue

        rows.append(_csv_row(candidate, landmarks))
        print(
            f"[{candidate.main_category} {len(rows)}/{target}] "
            f"product_code={candidate.product_code} candidate_attempt={attempt}"
        )
        if len(rows) == target:
            return rows

    raise RuntimeError(
        f"{candidates[0].main_category if candidates else 'unknown'} 샘플을 "
        f"{target}개 확보하지 못했습니다: {len(rows)}개"
    )


def _write_collection(
    rows: list[dict[str, object]], *, collect: int, seed: int, results_dir: Path
) -> Path:
    results_dir.mkdir(parents=True, exist_ok=True)
    output_path = results_dir / f"landmark_quality_seed{seed}_n{collect}.csv"
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
) -> Path:
    if sample_count < 2:
        raise ValueError("--collect는 2 이상이어야 합니다")

    upper_target = sample_count // 2
    lower_target = sample_count - upper_target
    owns_connection = connection is None
    owns_pose_estimator = pose_estimator is None
    connection = connection if connection is not None else get_connection()
    detector = detector if detector is not None else GroundingDinoDetector()
    pose_estimator = pose_estimator if pose_estimator is not None else PoseEstimator()
    try:
        upper_candidates = _randomized_candidates(
            _load_candidates(connection, UPPER_CATEGORY),
            seed=seed,
            main_category=UPPER_CATEGORY,
        )
        lower_candidates = _randomized_candidates(
            _load_candidates(connection, LOWER_CATEGORY),
            seed=seed,
            main_category=LOWER_CATEGORY,
        )
        upper_rows = _collect_category(
            upper_candidates,
            target=upper_target,
            detector=detector,
            pose_estimator=pose_estimator,
        )
        lower_rows = _collect_category(
            lower_candidates,
            target=lower_target,
            detector=detector,
            pose_estimator=pose_estimator,
        )
        return _write_collection(
            upper_rows + lower_rows,
            collect=sample_count,
            seed=seed,
            results_dir=results_dir,
        )
    finally:
        if owns_pose_estimator:
            pose_estimator.close()
        if owns_connection:
            connection.close()


def _divide(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def analyze(csv_path: Path) -> AnalysisResult:
    with csv_path.open(newline="", encoding="utf-8-sig") as input_file:
        reader = csv.DictReader(input_file)
        missing_columns = set(CSV_COLUMNS) - set(reader.fieldnames or ())
        if missing_columns:
            raise ValueError(f"CSV 필수 컬럼이 없습니다: {sorted(missing_columns)}")
        rows = list(reader)

    incomplete = [row["product_code"] for row in rows if not row["human_expected_reliable"]]
    if incomplete:
        raise ValueError(
            "human_expected_reliable이 비어 있습니다: " + ", ".join(incomplete)
        )

    counts = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}
    mismatches: list[dict[str, str]] = []
    for row in rows:
        human = row["human_expected_reliable"].strip().upper()
        if human not in {"YES", "NO"}:
            raise ValueError(
                f"product_code={row['product_code']}의 human_expected_reliable은 "
                "YES 또는 NO여야 합니다"
            )
        gate_text = row["landmark_reliable"].strip().lower()
        if gate_text not in {"true", "false"}:
            raise ValueError(
                f"product_code={row['product_code']}의 landmark_reliable 값이 "
                f"올바르지 않습니다: {row['landmark_reliable']!r}"
            )

        human_positive = human == "YES"
        gate_positive = gate_text == "true"
        if human_positive and gate_positive:
            outcome = "tp"
        elif not human_positive and not gate_positive:
            outcome = "tn"
        elif not human_positive and gate_positive:
            outcome = "fp"
        else:
            outcome = "fn"
        counts[outcome] += 1
        if outcome in {"fp", "fn"}:
            mismatches.append({**row, "outcome": outcome.upper()})

    total = sum(counts.values())
    return AnalysisResult(
        **counts,
        accuracy=_divide(counts["tp"] + counts["tn"], total),
        precision=_divide(counts["tp"], counts["tp"] + counts["fp"]),
        recall=_divide(counts["tp"], counts["tp"] + counts["fn"]),
        specificity=_divide(counts["tn"], counts["tn"] + counts["fp"]),
        mismatches=tuple(mismatches),
    )


def _print_analysis(result: AnalysisResult) -> None:
    print(f"TP: {result.tp}  # human YES / gate True")
    print(f"TN: {result.tn}  # human NO / gate False")
    print(f"FP: {result.fp}  # human NO / gate True")
    print(f"FN: {result.fn}  # human YES / gate False")
    print(f"accuracy: {result.accuracy:.4f}")
    print(f"precision: {result.precision:.4f}")
    print(f"recall: {result.recall:.4f}")
    print(f"specificity: {result.specificity:.4f}")
    print("mismatches:")
    if not result.mismatches:
        print("  none")
        return
    for row in result.mismatches:
        print(
            f"  {row['outcome']} product_code={row['product_code']} "
            f"left_shoulder={row['left_shoulder_visibility']} "
            f"right_shoulder={row['right_shoulder_visibility']} "
            f"left_hip={row['left_hip_visibility']} "
            f"right_hip={row['right_hip_visibility']} "
            f"torso_scale={row['torso_scale']} "
            f"note={row['human_note']!r}"
        )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--collect", type=int, metavar="N")
    mode.add_argument("--analyze", type=Path, metavar="CSV_PATH")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    if args.collect is not None:
        output_path = collect(args.collect, seed=args.seed)
        print(f"CSV 생성 완료: {output_path}")
        return
    try:
        result = analyze(args.analyze)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    _print_analysis(result)


if __name__ == "__main__":
    main()
