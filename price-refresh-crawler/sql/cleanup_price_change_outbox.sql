-- AI DB price_change_outbox 완료 기록 정리. 운영 DB에서 자동 실행하지 않는다(실행 예약은 별도 합의).
-- 삭제 대상: status = 'DONE' 이고 completed_at 이 실행 시각보다 5일을 초과해 지난 기록만.
-- PENDING·RETRY_WAIT·NOT_FOUND·BLOCKED 와 최근 완료된 DONE 은 created_at 과 관계없이 남긴다.
-- 한 번에 최대 1000건(completed_at, id 순)만 삭제하고 deleted_count 를 반환한다.
-- 결과가 1000이면 남은 대상이 있을 수 있으므로 0이 나올 때까지 제한된 횟수로 다시 실행한다.
-- 필요한 권한: public 스키마 USAGE, price_change_outbox 의 SELECT·DELETE. 다른 테이블은 변경하지 않는다.
-- 행 잠금(FOR UPDATE)은 UPDATE 권한이 필요하므로 쓰지 않는다. 대신 DELETE 에서 조건을 다시 확인하므로
-- 동시 전송·새 완료 기록이 있어도 조건에 맞지 않는 행은 삭제되지 않는다.

WITH expired AS (
    SELECT id
    FROM public.price_change_outbox
    WHERE status = 'DONE'
      AND completed_at < CURRENT_TIMESTAMP - INTERVAL '5 days'
    ORDER BY completed_at, id
    LIMIT 1000
), deleted AS (
    DELETE FROM public.price_change_outbox AS item
    USING expired
    WHERE item.id = expired.id
      AND item.status = 'DONE'
      AND item.completed_at < CURRENT_TIMESTAMP - INTERVAL '5 days'
    RETURNING item.id
)
SELECT count(*) AS deleted_count FROM deleted;
