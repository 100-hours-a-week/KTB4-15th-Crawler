# 가격 갱신 크롤러 (price-refresh-crawler) 이미지
# - 빌드 컨텍스트: price-refresh-crawler/ (docker build -f price-refresh-crawler/Dockerfile price-refresh-crawler)
# - 실행: ECS Scheduled Task (EventBridge Scheduler → RunTask). 실행 모드는 Task Definition의 command로 지정한다.
#   기본 CMD는 --help라서 이미지를 그냥 실행해도 DB에 쓰지 않는다.
ARG PYTHON_IMAGE=python:3.12-slim-bookworm

# =========================
# 1. Builder stage: 운영 의존성을 venv에 설치
# =========================
FROM ${PYTHON_IMAGE} AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /build
COPY requirements.txt .
RUN pip install -r requirements.txt

# =========================
# 2. Test stage: 단위 테스트 (DB·네트워크 없이 실행, 실 DB 테스트는 환경 변수 없으면 skip)
# =========================
FROM builder AS test

WORKDIR /app
COPY price_refresh ./price_refresh
COPY diagnostics ./diagnostics
COPY tests ./tests
RUN python -m unittest discover -s tests -p 'test_*.py' \
 && touch /tmp/.tests-passed

# =========================
# 3. Runtime stage: 운영 venv와 코드만 포함, non-root 실행
# =========================
FROM ${PYTHON_IMAGE} AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    HOME="/home/app"

RUN groupadd --system --gid 1001 app \
 && useradd --system --uid 1001 --gid app --home-dir /home/app --shell /usr/sbin/nologin app \
 && mkdir -p /home/app \
 && chown app:app /home/app

# test stage가 성공해야만 최종 이미지가 만들어지도록 한다.
COPY --from=test /tmp/.tests-passed /tmp/.tests-passed
COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
COPY price_refresh ./price_refresh

USER app

ENTRYPOINT ["python", "-m", "price_refresh"]
CMD ["--help"]
