import re

# 숫자 자체가 아니라 뒤따르는 colors/컬러 단위까지 있어야 매칭한다.
# 문자 클래스로 앞자리를 2-9로 제한하면 "10 colors"처럼 두 자리 이상인
# 개수를 놓치므로, 숫자는 전부 캡처한 뒤 파이썬에서 임계값을 비교한다.
#
# \b 대신 부정 lookaround를 쓴다. "_"는 정규식에서 단어 문자로 취급돼
# \b가 성립하지 않으므로, "TEXT_2COLORS"처럼 언더스코어로 붙은 실제
# 상품명(29CM 응답에서 흔한 구분자)을 \b 기반 패턴은 놓친다. 대신 숫자
# 앞뒤에 다른 숫자/문자가 없는지 직접 확인해 모델 코드 속 숫자
# ("SH2101color" 같은 붙어있는 오탐)는 걸러내면서 "_", "-", "[", 공백,
# 문자열 시작/끝처럼 실제 구분자로 쓰이는 모든 문자는 허용한다.
_MULTI_COLOR_PATTERN = re.compile(
    r"(?<![0-9A-Za-z가-힣])(\d+)\s*(?:colors?|컬러)(?![0-9A-Za-z가-힣])",
    re.IGNORECASE,
)
_MULTI_COLOR_MIN_COUNT = 2


def is_multi_color_product(product_name: str) -> bool:
    """상품명이 2가지 이상 색상 옵션을 나타내는지 반환한다."""
    if not product_name:
        return False
    return any(
        int(match.group(1)) >= _MULTI_COLOR_MIN_COUNT
        for match in _MULTI_COLOR_PATTERN.finditer(product_name)
    )
