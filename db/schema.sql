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
    crawled_at TIMESTAMPTZ NOT NULL
);
