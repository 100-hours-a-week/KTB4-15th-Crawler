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
