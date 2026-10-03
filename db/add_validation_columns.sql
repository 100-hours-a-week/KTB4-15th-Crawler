-- db/schema.sql 을 적용하기 전에 이미 만들어져 있던 DB에 검증 컬럼을 추가한다.
-- CREATE TABLE IF NOT EXISTS 로는 기존 테이블에 새 컬럼이 생기지 않으므로 별도로 둔다.
-- 적용 방법: psql "$DATABASE_URL" -f db/add_validation_columns.sql
--
-- 여러 번 실행해도 안전하다(IF NOT EXISTS).

ALTER TABLE products
    ADD COLUMN IF NOT EXISTS validation_status VARCHAR(10) NOT NULL DEFAULT 'PENDING',
    ADD COLUMN IF NOT EXISTS validation_reason VARCHAR(50);
