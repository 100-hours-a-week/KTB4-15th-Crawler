"""Grounding DINO 로 사람/의류를 탐지한다.

PyTorch + Hugging Face Transformers 의 zero-shot object detection 을 그대로 쓴다
(https://huggingface.co/docs/transformers/en/model_doc/grounding-dino).
"""

from collections.abc import Sequence

from app.validation.models import CroppedLowerBodyCheck, Detection, DetectionResult

# 실제 이미지로 조정이 필요한 값들은 이 파일 한 곳에 모아둔다 (요구사항 §5, §21).
DEFAULT_MODEL_ID = "IDEA-Research/grounding-dino-tiny"
BOX_THRESHOLD = 0.35
TEXT_THRESHOLD = 0.25
IOU_THRESHOLD = 0.5

PERSON_LABEL = "person"

# app/config/categories.py 의 실제 sub_category 값(52개)을 기준으로 작성했다.
# 하나의 sub_category 에 여러 label 을 둔 이유: 같은 옷이라도 이미지마다
# Grounding DINO 가 더 잘 반응하는 동의어가 다르기 때문이다.
SUB_CATEGORY_DETECTION_LABELS: dict[str, list[str]] = {
    # 상의
    "반소매 셔츠": ["short sleeve shirt", "shirt"],
    "반소매 티셔츠": ["t-shirt", "short sleeve t-shirt"],
    "피케/카라 티셔츠": ["polo shirt", "collared t-shirt"],
    "슬리브리스": ["sleeveless top", "tank top"],
    "스웨트셔츠": ["sweatshirt"],
    "후디": ["hoodie", "hooded sweatshirt"],
    "집업": ["zip-up jacket", "zip-up hoodie", "jacket"],
    "긴소매 셔츠": ["long sleeve shirt", "shirt"],
    "긴소매 티셔츠": ["long sleeve t-shirt", "t-shirt"],
    # 하의
    "부츠컷": ["bootcut pants", "pants"],
    "레깅스": ["leggings"],
    "쇼트": ["shorts"],
    "슬림 팬츠": ["slim pants", "pants"],
    "스트레이트 팬츠": ["straight pants", "pants"],
    "와이드 팬츠": ["wide leg pants", "pants"],
    "데님 팬츠": ["jeans", "denim pants"],
    "트레이닝 팬츠": ["track pants", "sweatpants"],
    "슬랙스": ["slacks", "dress pants"],
    "코튼 팬츠": ["cotton pants", "pants"],
    "기타 팬츠": ["pants", "trousers"],
    # 아우터 (categories.py 원문 표기를 그대로 key 로 쓴다: "테님 재킷", "배스트")
    "야상": ["field jacket", "jacket"],
    "블루종": ["bomber jacket", "jacket"],
    "바시티": ["varsity jacket", "jacket"],
    "테님 재킷": ["denim jacket", "jacket"],
    "퍼 재킷": ["fur jacket", "jacket"],
    "트레이닝 재킷": ["track jacket", "jacket"],
    "점퍼": ["jacket", "jumper"],
    "바람막이": ["windbreaker", "jacket"],
    "아노락": ["anorak", "jacket"],
    "플리스": ["fleece jacket", "fleece"],
    "무스탕": ["shearling jacket", "jacket"],
    "트렌치/맥코트": ["trench coat", "coat"],
    "나일론/코치 재킷": ["nylon jacket", "coach jacket"],
    "후드 집업": ["zip-up hoodie", "hooded jacket"],
    "배스트": ["vest"],
    "레더 재킷": ["leather jacket", "jacket"],
    "블레이저": ["blazer", "jacket"],
    "숏코트": ["short coat", "coat"],
    "하프코트": ["coat"],
    "롱코트": ["long coat", "coat"],
    "경량패딩": ["puffer jacket", "down jacket"],
    "숏패딩": ["puffer jacket", "down jacket"],
    "롱패딩": ["long puffer coat", "down coat"],
    "기타 아우터": ["jacket", "coat"],
    # 니트웨어
    "기타 니트": ["knit sweater", "sweater"],
    "크루넥": ["crew neck sweater", "sweater"],
    "브이넥": ["v-neck sweater", "sweater"],
    "터틀넥": ["turtleneck sweater", "sweater"],
    "폴로셔츠": ["polo shirt", "collared t-shirt"],
    "카디건": ["cardigan"],
    "베스트": ["vest"],
    "후드": ["hoodie", "hooded sweater"],
}

# main_category 는 저장 시 STORAGE_MAIN_CATEGORY_MAP 을 거쳐 "상의"/"하의" 둘 중
# 하나로만 저장된다 (아우터/니트웨어도 "상의"로 저장). sub_category mapping 이
# 없을 때만 쓰는 fallback 이다.
MAIN_CATEGORY_DETECTION_LABELS: dict[str, list[str]] = {
    "상의": ["shirt", "t-shirt", "sweater", "jacket", "top"],
    "하의": ["pants", "trousers", "shorts"],
}

# product_image_validator 가 하의 상품(상반신이 잘려도 정상일 수 있는 카테고리)을
# 구분할 때 쓴다. 위와 마찬가지로 저장된 main_category 값 기준이다.
BOTTOM_MAIN_CATEGORY = "하의"

# 하의 + person 1 + garment 1 + MediaPipe landmarks 없음(사람이 화면 위쪽 끝까지 잘려
# MediaPipe 가 사람 자체를 못 잡은 경우) 전용 bbox fallback threshold다. 실제 이미지로
# 조정이 필요하다(요구사항 §21과 같은 맥락).
CROPPED_LOWER_BODY_PERSON_TOP_RATIO_MAX = 0.01
CROPPED_LOWER_BODY_GARMENT_TOP_RATIO_MAX = 0.30
CROPPED_LOWER_BODY_GARMENT_PERSON_AREA_RATIO_MIN = 0.42
CROPPED_LOWER_BODY_GARMENT_IN_PERSON_RATIO_MIN = 0.90


def get_detection_labels(main_category: str, sub_category: str) -> list[str] | None:
    """sub_category 를 우선 사용하고 없으면 main_category 로 대체한다.

    둘 다 없으면 None 을 반환한다. 호출하는 쪽에서 CATEGORY_MAPPING_NOT_FOUND 로
    기록한다.
    """
    labels = SUB_CATEGORY_DETECTION_LABELS.get(sub_category)
    if labels is not None:
        return labels
    return MAIN_CATEGORY_DETECTION_LABELS.get(main_category)


Box = tuple[float, float, float, float]


def box_area(box: Box) -> float:
    """box 는 (x0, y0, x1, y1) 이다. 재사용 가능한 작은 함수로 분리해 중복 계산을 피한다."""
    x0, y0, x1, y1 = box
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)


def intersection_area(box_a: Box, box_b: Box) -> float:
    ax0, ay0, ax1, ay1 = box_a
    bx0, by0, bx1, by1 = box_b
    inter_x0, inter_y0 = max(ax0, bx0), max(ay0, by0)
    inter_x1, inter_y1 = min(ax1, bx1), min(ay1, by1)
    return max(0.0, inter_x1 - inter_x0) * max(0.0, inter_y1 - inter_y0)


def _iou(a: Box, b: Box) -> float:
    inter_area = intersection_area(a, b)
    if inter_area <= 0:
        return 0.0
    union_area = box_area(a) + box_area(b) - inter_area
    return inter_area / union_area if union_area > 0 else 0.0


def deduplicate_by_iou(
    detections: Sequence[Detection], *, iou_threshold: float = IOU_THRESHOLD
) -> list[Detection]:
    """동의어 label 이 같은 객체를 가리키는 경우를 하나로 합친다.

    confidence 가 높은 순으로 남기는 greedy NMS 다. label 이 달라도 같은 물체를
    가리키면(IoU 가 threshold 이상) 점수가 낮은 쪽을 버린다.
    """
    ordered = sorted(detections, key=lambda detection: detection.score, reverse=True)
    kept: list[Detection] = []
    for candidate in ordered:
        if all(_iou(candidate.box, other.box) < iou_threshold for other in kept):
            kept.append(candidate)
    return kept


def check_cropped_lower_body(
    *, person_box: Box, garment_box: Box, image_height: float
) -> CroppedLowerBodyCheck:
    """하의 + person 1 + garment 1 + MediaPipe landmarks 없음일 때만 호출한다.

    사람이 화면 위쪽 끝까지 잘려 있고(person_top_ratio), 하의가 화면 상단 30% 안에서
    시작하며(garment_top_ratio), 하의 박스가 사람 박스의
    CROPPED_LOWER_BODY_GARMENT_PERSON_AREA_RATIO_MIN 이상을 차지하고
    (garment_person_area_ratio), 하의 박스가 사람 박스 안에 거의 다 들어가 있으면
    (garment_in_person_ratio) 정상적으로 하의를 입은 크롭샷으로 본다.

    Grounding DINO 의 threshold/NMS 판정에는 관여하지 않는다. person_box/garment_box
    가 이미 dedup 을 끝낸 최종 detection 이라고 가정한다.
    """
    person_area = box_area(person_box)
    garment_area = box_area(garment_box)

    if image_height <= 0 or person_area <= 0 or garment_area <= 0:
        return CroppedLowerBodyCheck(
            person_top_ratio=1.0,
            garment_top_ratio=1.0,
            garment_person_area_ratio=0.0,
            garment_in_person_ratio=0.0,
            is_cropped_lower_body=False,
        )

    person_top_ratio = person_box[1] / image_height
    garment_top_ratio = garment_box[1] / image_height
    garment_person_area_ratio = garment_area / person_area
    garment_in_person_ratio = intersection_area(person_box, garment_box) / garment_area

    is_cropped_lower_body = (
        person_top_ratio <= CROPPED_LOWER_BODY_PERSON_TOP_RATIO_MAX
        and garment_top_ratio <= CROPPED_LOWER_BODY_GARMENT_TOP_RATIO_MAX
        and garment_person_area_ratio >= CROPPED_LOWER_BODY_GARMENT_PERSON_AREA_RATIO_MIN
        and garment_in_person_ratio >= CROPPED_LOWER_BODY_GARMENT_IN_PERSON_RATIO_MIN
    )
    return CroppedLowerBodyCheck(
        person_top_ratio=person_top_ratio,
        garment_top_ratio=garment_top_ratio,
        garment_person_area_ratio=garment_person_area_ratio,
        garment_in_person_ratio=garment_in_person_ratio,
        is_cropped_lower_body=is_cropped_lower_body,
    )


class GroundingDinoDetector:
    """모델을 한 번만 로드하고 여러 상품 검증에 재사용한다."""

    def __init__(
        self,
        *,
        model_id: str = DEFAULT_MODEL_ID,
        device: str | None = None,
        box_threshold: float = BOX_THRESHOLD,
        text_threshold: float = TEXT_THRESHOLD,
        iou_threshold: float = IOU_THRESHOLD,
    ) -> None:
        # 무거운 의존성은 실제 사용 시점(로컬 검증 실행)에만 필요하므로
        # 모듈 최상단이 아니라 여기서 import 한다.
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        self.device = device or self._default_device(torch)
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.iou_threshold = iou_threshold
        self._torch = torch
        self._processor = AutoProcessor.from_pretrained(model_id)
        self._model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(
            self.device
        )
        self._model.eval()

    @staticmethod
    def _default_device(torch_module) -> str:
        if torch_module.cuda.is_available():
            return "cuda"
        if torch_module.backends.mps.is_available():
            return "mps"
        return "cpu"

    def detect(self, image_rgb, garment_labels: Sequence[str]) -> DetectionResult:
        """image_rgb 는 HxWx3 RGB numpy 배열이다."""
        text_labels = [[PERSON_LABEL, *garment_labels]]
        height, width = image_rgb.shape[:2]

        inputs = self._processor(images=image_rgb, text=text_labels, return_tensors="pt").to(
            self.device
        )
        with self._torch.no_grad():
            outputs = self._model(**inputs)

        results = self._processor.post_process_grounded_object_detection(
            outputs,
            threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            target_sizes=[(height, width)],
        )[0]

        persons: list[Detection] = []
        garments: list[Detection] = []
        for box, score, label in zip(
            results["boxes"], results["scores"], results["text_labels"]
        ):
            detection = Detection(
                label=label,
                score=float(score),
                box=tuple(round(value, 2) for value in box.tolist()),
            )
            if label == PERSON_LABEL:
                persons.append(detection)
            else:
                garments.append(detection)

        return DetectionResult(
            persons=deduplicate_by_iou(persons, iou_threshold=self.iou_threshold),
            garments=deduplicate_by_iou(garments, iou_threshold=self.iou_threshold),
        )
