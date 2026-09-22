"""상품 한 건의 이미지 검증 전체 흐름을 담당한다.

Product → 이미지 다운로드/디코딩 → category → detection label → Grounding DINO
→ (사람이 1명이면) MediaPipe → rules → ValidationResult

DB 처리나 CSV 파일 생성은 여기서 하지 않는다.
"""

from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import cv2
import numpy as np

from app.models.product import Product
from app.validation import rules
from app.validation.detector import (
    BOTTOM_MAIN_CATEGORY,
    GroundingDinoDetector,
    get_detection_labels,
)
from app.validation.models import (
    PoseDirection,
    ValidationReason,
    ValidationResult,
    ValidationStatus,
)
from app.validation.pose import PoseEstimator

IMAGE_DOWNLOAD_TIMEOUT = 15
_USER_AGENT = "Mozilla/5.0 (compatible; LookDDak-ImageValidator/1.0)"

_CATEGORY_MAPPING_NOT_FOUND = ValidationResult(
    ValidationStatus.FAIL, ValidationReason.CATEGORY_MAPPING_NOT_FOUND
)
_IMAGE_LOAD_FAILED = ValidationResult(ValidationStatus.FAIL, ValidationReason.IMAGE_LOAD_FAILED)


def download_image(image_url: str, *, timeout: int = IMAGE_DOWNLOAD_TIMEOUT) -> np.ndarray:
    """원본 이미지를 그대로 내려받아 RGB numpy 배열로 디코딩한다.

    resize 는 하지 않는다. 실패하면 (URLError, HTTPError, OSError, ValueError) 중
    하나를 던진다.
    """
    request = Request(image_url, headers={"User-Agent": _USER_AGENT})
    with urlopen(request, timeout=timeout) as response:
        raw = response.read()
    if not raw:
        raise ValueError(f"이미지 응답이 비어 있습니다: {image_url}")

    image_bgr = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError(f"이미지를 디코딩할 수 없습니다: {image_url}")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def validate_product_image(
    product: Product,
    *,
    detector: GroundingDinoDetector,
    pose_estimator: PoseEstimator,
    image_loader=download_image,
) -> ValidationResult:
    """한 상품의 이미지를 검증한다. 예외를 던지지 않고 항상 ValidationResult 를 반환한다.

    단, 이미지 다운로드/디코딩 실패만 여기서 흡수한다(IMAGE_LOAD_FAILED). 그 외
    예상하지 못한 오류는 그대로 올려서 호출하는 쪽이 batch 진행 여부를 판단하게 한다.

    image_loader 는 테스트에서 실제 네트워크 호출 대신 fake 함수를 주입하기 위한
    자리다. 기본값은 실제 다운로드 함수인 download_image 다.
    """
    labels = get_detection_labels(product.main_category, product.sub_category)
    if labels is None:
        return _CATEGORY_MAPPING_NOT_FOUND

    try:
        image_rgb = image_loader(product.image_url)
    except (URLError, HTTPError, OSError, ValueError):
        return _IMAGE_LOAD_FAILED

    detection = detector.detect(image_rgb, labels)

    pose_direction: PoseDirection | None = None
    if detection.person_count == 1:
        # 하의 상품은 상반신이 잘려 찍혀도 정상일 수 있다. 그 경우에만 상체
        # landmark 가 부족할 때 하체 landmark 로 대신 판정하게 한다(pose.py 참고).
        allow_lower_body_fallback = (
            product.main_category == BOTTOM_MAIN_CATEGORY and detection.garment_count == 1
        )
        pose_direction = pose_estimator.estimate_direction(
            image_rgb, allow_lower_body_fallback=allow_lower_body_fallback
        )

    return rules.decide(detection, pose_direction)
