"""요청 main_category 기준 수집 quota 설정.

sub_category 기본 quota는 main_category quota를 sub_category 개수로 나눈
값이 아니라, 1차 수집 목표로 별도 지정한 값이다. 1차 수집 후 main_category
quota가 남으면 아직 소진(exhausted)되지 않은 같은 main_category의
sub_category에 부족분을 재분배해서 채운다.

저장되는 Product.main_category는 상의/하의로 정규화되어 아우터/니트웨어
구분이 사라지므로, quota는 반드시 API 요청 기준 main_category
(``STORAGE_MAIN_CATEGORY_MAP``의 key)로 관리한다.
"""

MAIN_CATEGORY_QUOTAS = {
    "상의": 10_000,
    "니트웨어": 10_000,
    "아우터": 10_000,
    "하의": 20_000,
}

SUB_CATEGORY_BASE_QUOTAS = {
    "상의": 1_000,
    "니트웨어": 1_000,
    "아우터": 400,
    "하의": 1_800,
}

TOTAL_QUOTA = sum(MAIN_CATEGORY_QUOTAS.values())
