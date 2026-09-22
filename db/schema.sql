-- 29CM 상품 데이터셋(data/products.json)을 담는 초기 적재용 테이블.
-- 적용 방법: psql "$DATABASE_URL" -f db/schema.sql

CREATE TABLE IF NOT EXISTS products (
    id BIGSERIAL PRIMARY KEY,
    product_code BIGINT NOT NULL UNIQUE,
    product_name TEXT NOT NULL,
    price INTEGER NOT NULL,
    color VARCHAR(50),
    detail_url TEXT NOT NULL,
    image_url TEXT NOT NULL,
    main_category VARCHAR(20) NOT NULL,
    sub_category VARCHAR(100) NOT NULL,
    is_sold_out BOOLEAN NOT NULL,
    crawled_at TIMESTAMPTZ NOT NULL,
    -- 로컬 상품 이미지 검증 도구(app/validation/)가 쓰는 상태. PENDING/PASS/FAIL.
    -- 이미 만들어져 있는 DB에는 이 스키마를 다시 적용해도 컬럼이 추가되지 않으므로
    -- db/add_validation_columns.sql 을 대신 적용한다.
    validation_status VARCHAR(10) NOT NULL DEFAULT 'PENDING',
    validation_reason VARCHAR(50)
);
