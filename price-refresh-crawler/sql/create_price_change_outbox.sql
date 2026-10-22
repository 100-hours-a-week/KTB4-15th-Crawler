-- AI DB 가격 전달용 outbox. BE DB나 BE의 피팅용 outbox_event에는 적용하지 않는다.
-- 전제: public.products(product_code bigint)에 PRIMARY KEY 또는 UNIQUE 제약이 있다.
-- 이미 테이블이 있으면 정의가 달라도 덮어쓰지 않도록 IF NOT EXISTS 없이 실패시킨다.
-- 적용 후 확인: price_refresh.readiness.check_readiness 로 컬럼·제약·권한을 점검한다.
-- updated_at은 트리거 없이 상태를 변경하는 애플리케이션 코드가 명시적으로 갱신한다.

BEGIN;

CREATE TABLE public.price_change_outbox (
    id                 bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    scan_run_id        uuid NOT NULL,
    product_code       bigint NOT NULL,
    original_old_price integer NOT NULL,
    old_price          integer NOT NULL,
    new_price          integer NOT NULL,
    status             varchar(16) NOT NULL DEFAULT 'PENDING',
    attempt_count      integer NOT NULL DEFAULT 0,
    next_attempt_at    timestamptz DEFAULT CURRENT_TIMESTAMP,
    last_attempt_at    timestamptz,
    last_backend_price integer,
    last_http_status   smallint,
    last_error_code    varchar(64),
    last_error_message text,
    created_at         timestamptz NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at         timestamptz NOT NULL DEFAULT CURRENT_TIMESTAMP,
    completed_at       timestamptz,

    CONSTRAINT fk_price_change_outbox_product
        FOREIGN KEY (product_code)
        REFERENCES public.products(product_code) ON DELETE RESTRICT,
    CONSTRAINT uq_price_change_outbox_run_product
        UNIQUE (scan_run_id, product_code),
    CONSTRAINT ck_price_change_outbox_product
        CHECK (product_code > 0),
    CONSTRAINT ck_price_change_outbox_prices
        CHECK (original_old_price >= 0 AND old_price >= 0 AND new_price >= 0),
    CONSTRAINT ck_price_change_outbox_changed
        CHECK (original_old_price <> new_price),
    CONSTRAINT ck_price_change_outbox_status
        CHECK (status IN ('PENDING', 'RETRY_WAIT', 'DONE', 'NOT_FOUND', 'BLOCKED')),
    CONSTRAINT ck_price_change_outbox_attempt_count
        CHECK (attempt_count >= 0),
    CONSTRAINT ck_price_change_outbox_backend_price
        CHECK (last_backend_price IS NULL OR last_backend_price >= 0),
    CONSTRAINT ck_price_change_outbox_http_status
        CHECK (last_http_status IS NULL OR last_http_status BETWEEN 100 AND 599),
    CONSTRAINT ck_price_change_outbox_schedule
        CHECK (status NOT IN ('PENDING', 'RETRY_WAIT') OR next_attempt_at IS NOT NULL),
    CONSTRAINT ck_price_change_outbox_completion
        CHECK ((status = 'DONE') = (completed_at IS NOT NULL))
);

-- 예약 시각이 된 최초 전송·재전달 대기 기록 조회.
CREATE INDEX ix_price_change_outbox_due
    ON public.price_change_outbox (next_attempt_at, id)
    WHERE status IN ('PENDING', 'RETRY_WAIT');

-- 같은 상품의 앞선 미완료 기록 확인. NOT_FOUND는 후속 기록을 막지 않으므로 제외한다.
CREATE INDEX ix_price_change_outbox_unfinished_product
    ON public.price_change_outbox (product_code, id)
    WHERE status NOT IN ('DONE', 'NOT_FOUND');

-- DONE 완료 시각 기준 정리용.
CREATE INDEX ix_price_change_outbox_completed
    ON public.price_change_outbox (completed_at, id)
    WHERE status = 'DONE';

COMMIT;

-- 권한 부여 예시. 실제 DB명·계정명을 정한 뒤 별도로 적용한다(이 파일에서는 실행되지 않는다).
-- GRANT CONNECT ON DATABASE <ai_database> TO <price_refresh_role>;
-- GRANT USAGE ON SCHEMA public TO <price_refresh_role>;
-- GRANT SELECT ON public.products TO <price_refresh_role>;
-- GRANT UPDATE (price) ON public.products TO <price_refresh_role>;
-- GRANT SELECT, INSERT, UPDATE ON public.price_change_outbox TO <price_refresh_role>;
-- IDENTITY 열의 INSERT는 PostgreSQL 16 기준 시퀀스 권한 없이도 동작한다. 정책상 명시하려면:
-- GRANT USAGE, SELECT ON SEQUENCE public.price_change_outbox_id_seq TO <price_refresh_role>;
-- 완료 기록 정리용 DELETE는 별도 보수 계정에 부여한다.
-- GRANT SELECT, DELETE ON public.price_change_outbox TO <maintenance_role>;
