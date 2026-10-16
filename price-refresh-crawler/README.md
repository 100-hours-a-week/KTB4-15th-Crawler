# 가격 갱신 크롤러 — 1단계 가격 조회

29CM 목록에서 상품 코드와 가격만 수집하는 독립 실행 폴더입니다. Python 3.9 이상과 표준 라이브러리만 사용합니다. 기존 수집 크롤러를 설치하거나 부모 폴더를 Python 경로에 추가할 필요가 없습니다.

설계: https://app.notion.com/p/3f8b7bbac8c780c8b70ecaced419b235

이번 단계에는 DB 조회·수정, 기존 가격 비교, 추천 이력·필터, 백엔드 전달, 예약 실행과 실제 서버 배포가 없습니다. 출력은 DB와 비교한 변경 목록이 아니라 이번 실행에서 조회한 코드별 최대 유효 가격입니다. 개발·후속 배포에 공통으로 사용할 최소 Docker 실행 환경만 추가했습니다.

## 구성

```text
price-refresh-crawler/
├── price_refresh/
│   ├── __main__.py             # python -m price_refresh 진입점
│   ├── main.py                 # CLI, JSON 출력, 종료 코드
│   ├── settings.py             # 환경변수 / 설정 객체 검증
│   ├── parser.py               # 응답 구조 및 코드·가격 검증
│   ├── scanner.py              # 순차 교대 조회, 카테고리 상한, 전체 최대 가격
│   └── source/
│       ├── categories.py       # 기존 CATEGORIES 전체 복사, 순서 유지
│       ├── client.py           # 기존 목록 요청 계약, 연결·응답 타임아웃 분리
│       └── fixture_client.py   # 외부 접속 없는 실행용 가짜 소스
├── tests/
│   ├── test_diagnostics.py     # 요청 횟수·조건부 세 번째 요청·대기 검증
│   ├── test_price_refresh.py
│   └── fixtures/listing_response.json
├── diagnostics/
│   ├── runtime.py             # 네트워크 없는 런타임 정보
│   └── compare_clients.py     # 기존/새 클라이언트 최대 3회 비교
├── Dockerfile                 # 공식 Python 이미지의 버전·OS·digest 고정
├── .dockerignore              # 작은 소스·테스트 파일만 빌드에 허용
├── requirements.txt           # 외부 의존성 없음
├── .env.example               # 요청 설정 예시, 비밀 값 없음
└── .gitignore
```

## 공통 Docker 실행 환경

개발 기준은 **공식 `python:3.13.16-slim-trixie`, Linux ARM64**입니다. Dockerfile은 아래 multi-platform index digest를 고정합니다. OS·Python·라이브러리 구성이 바뀌는 태그 갱신을 자동으로 따라가지 않습니다.

| 항목 | 고정값 또는 2026-10-15 실측 |
| --- | --- |
| 기본 이미지 | `docker.io/library/python:3.13.16-slim-trixie` |
| Index digest — Dockerfile의 FROM | `sha256:70729b46c69b4f1e97c4822c1af3df53a1476cf5ddc6c087c0c10bc3a5678c2f` |
| ARM64/v8 manifest digest | `sha256:cf97c3b79da1c706532c7042f851654d97d3b88c8ca6520acda9cf4479ea9b14` |
| AMD64 manifest digest — 실행 미검증 | `sha256:8fb4cfa1a2616d7b8e0c2175cc6ad68f5729c34ea8488c0b360d2934b7be9024` |
| 검증 호스트 / Docker 엔진 | macOS ARM64 / Docker Desktop Linux aarch64, 29.6.2 |
| 컨테이너 Python | `/usr/local/bin/python`, **3.13.16** |
| 컨테이너 TLS 라이브러리 | **OpenSSL 3.5.7 9 Jun 2026** |
| OS / CPU | Debian GNU/Linux 13 (trixie), `linux/arm64` (`aarch64`) |
| 추가 패키지 | 없음 — 표준 라이브러리만 사용 |
| 실행 사용자 | UID/GID `10001:10001` |

선택 근거는 로컬 Python 3.13 계열의 성공 기록, 현재 1단계의 표준 라이브러리 의존성, 공식 이미지의 ARM64·AMD64 지원입니다. [Python 3.13.16 공식 릴리스](https://www.python.org/downloads/release/python-31316/)와 [Docker Official Images의 Python 태그 목록](https://github.com/docker-library/official-images/blob/master/library/python)에서 버전과 Debian slim 태그를 확인했습니다. 향후 보안 패치 반영 시에는 버전·digest를 명시적으로 갱신하고 다시 검증해야 합니다.

기존 AI 서버 Dockerfile은 Python 3.12를 사용하지만, 가격 크롤러는 AI 모듈이나 의존성을 공유하지 않으므로 그 버전에 맞출 제약은 발견되지 않았습니다. AI 저장소의 `.github/workflows/ai-backend-cd.yml:49`와 개발 CD는 ARM 러너에서 이미지를 빌드합니다. 이는 실제 EC2 CPU를 직접 확인한 결과가 아닙니다. **후속 배포 단계에서 서버의 아키텍처·실제 네트워크·접근 권한을 확인해야 합니다.** 서버가 AMD64라면 해당 플랫폼으로 빌드·조회 검증을 별도로 수행해야 하며, 로컬 ARM64 성공을 대신 적용하지 않습니다.

아래 명령은 `price-refresh-crawler/` 안에서 실행합니다. Docker Desktop/엔진이 실행 중이어야 합니다.

```sh
docker build --platform linux/arm64 -t lookddak-price-refresh:py3.13.16 .

# 런타임 확인 — 네트워크 없음
docker run --rm --platform linux/arm64 --network none --read-only \
  lookddak-price-refresh:py3.13.16 -m diagnostics.runtime

# 오프라인 테스트 — 임시 파일만 /tmp tmpfs에 작성
docker run --rm --platform linux/arm64 --network none --read-only \
  --tmpfs /tmp:rw,nosuid,nodev,size=64m \
  lookddak-price-refresh:py3.13.16 -m unittest discover -s tests -v

# fixture 실행 — HTTP 연결과 실제 대기 없음
docker run --rm --platform linux/arm64 --network none --read-only \
  lookddak-price-refresh:py3.13.16 -m price_refresh \
  --fixture tests/fixtures/listing_response.json --category-limit 1
```

기본 `docker run --rm lookddak-price-refresh:py3.13.16`은 도움말만 출력하고 종료합니다. 모든 실행은 일회성 명령이며 포트를 열거나 상시 프로세스·예약 작업을 만들지 않습니다. 이미지에는 기존 수집 코드, `.env`, 인증 정보, `.git`, 실제 수집 결과, DB dump가 들어가지 않습니다. `COPY` 대상과 빌드 컨텍스트를 소스·작은 fixture·테스트·진단 코드로 제한했습니다.

코드를 수정하며 같은 런타임에서 테스트하려면 새 폴더의 소스만 읽기 전용으로 마운트합니다. `--network none`을 유지합니다.

```sh
docker run --rm --platform linux/arm64 --network none --read-only \
  --tmpfs /tmp:rw,nosuid,nodev,size=64m \
  --mount "type=bind,src=$PWD/price_refresh,dst=/app/price_refresh,readonly" \
  --mount "type=bind,src=$PWD/diagnostics,dst=/app/diagnostics,readonly" \
  --mount "type=bind,src=$PWD/tests,dst=/app/tests,readonly" \
  lookddak-price-refresh:py3.13.16 -m unittest discover -s tests -v
```

## Docker에서 제한된 실제 조회 비교

기존 클라이언트는 진단할 때만 **파일 하나를 읽기 전용 마운트**합니다. 운영 이미지와 새 크롤러는 이 파일 없이 독립 실행됩니다. 아래 첫 명령은 가짜 전송으로 두 클라이언트가 만드는 URL·메서드·명시적 헤더·본문 바이트가 같은지 확인하며, API를 호출하지 않습니다.

```sh
docker run --rm --platform linux/arm64 --network none --read-only \
  --mount "type=bind,src=$PWD/../app/crawler/client.py,dst=/legacy/client.py,readonly" \
  lookddak-price-refresh:py3.13.16 -m diagnostics.compare_clients \
  --legacy-client /legacy/client.py
```

실제 요청은 다음 명령의 `--live`를 명시했을 때만 발생합니다. 전체 scanner는 호출하지 않습니다.

```sh
docker run --rm --platform linux/arm64 --read-only \
  --mount "type=bind,src=$PWD/../app/crawler/client.py,dst=/legacy/client.py,readonly" \
  lookddak-price-refresh:py3.13.16 -m diagnostics.compare_clients \
  --legacy-client /legacy/client.py --live
```

진단 순서는 고정돼 있습니다.

1. 기존 클라이언트 `MOST_REVIEWED`, 1페이지, 크기 50.
2. 새 클라이언트 `MOST_REVIEWED`, 같은 조건.
3. 위 두 요청이 모두 성공했을 때만 새 클라이언트 `RECOMMENDED`, 1페이지, 크기 50.

카테고리는 `CATEGORIES[0]`인 **상의 / 반소매 셔츠**, `largeId=272100100`, `middleId=272103100`, **`smallId=272103105`**입니다. 과거 중분류 전체 조회와 조건이 다르지만, 이번 기존·새 비교에서는 동일한 세부 카테고리를 사용합니다.

각 응답·실패 후 마지막 요청을 포함해 1초 대기합니다. 최대 3회이며 재시도하지 않습니다. 기존 클라이언트는 원래의 단일 10초 타임아웃, 새 클라이언트는 연결 5초·연결 후 소켓 I/O 20초를 사용합니다. 요청 헤더와 TLS 기본 설정·인증서 검증은 바꾸지 않습니다. 추가 요청을 막기 위해 진단용 기존 opener에서만 리다이렉트를 차단하며, 원본 파일은 수정하지 않습니다. 프록시 설정이 있으면 변경하지 않고 실조회 전에 중단합니다.

실패 본문을 출력하는 기존 로거는 진단 동안만 비활성화합니다. 응답 상태·형식·소요 시간, 필요한 응답 헤더, Cloudflare/CAPTCHA 표시 여부, 최대 3개의 코드·가격과 파서 비교만 JSON으로 출력합니다. 쿠키나 인증 값은 수집·추가·출력하지 않습니다.

### 2026-10-15 검증 결과

위 단일 후보 이미지를 Linux ARM64로 빌드했습니다. **네트워크 없는 테스트 30개 통과**, fixture에서 코드별 가격 5개 정상 추출, 두 요청 생성기의 계약 일치 확인 후 실제 조회했습니다.

| 클라이언트 / 정렬 | 시작 시각 KST | HTTP / Content-Type | 소요 시간 | data.list / 상품 수 | 첫 3개 원본·파서 일치 |
| --- | --- | --- | ---: | --- | --- |
| 기존 / MOST_REVIEWED | 13:43:23.381 | 200 / application/json | 0.522초 | 있음 / 50 | 3/3 |
| 새 / MOST_REVIEWED | 13:43:24.911 | 200 / application/json | 0.133초 | 있음 / 50 | 3/3 |
| 새 / RECOMMENDED | 13:43:26.050 | 200 / application/json | 0.546초 | 있음 / 50 | 3/3 |

리뷰순의 첫 세 상품은 기존·새 모두 `2092944:37270`, `2650956:39600`, `2650955:37270`이었습니다. 새 추천순은 `3242193:38100`, `2527708:51710`, `3263645:37720`이었습니다. 모든 코드는 파서에서 숫자 문자열로 변환됐으며 가격은 일치했습니다.

**실제 API 요청 총 3회**, 다른 후보 이미지 0개, 재시도·추가 페이지·실제 scanner 실행 0회입니다. JSON·목록 구조·연결·파싱 오류는 없었습니다. 새 요청에서 연결 5초, 응답 헤더를 기다리기 전 소켓 타임아웃 20초를 관측했습니다. 실제 타임아웃을 유발해 종료 동작을 검증한 것은 아닙니다.

이 결과는 이 Docker 환경을 개발 기준으로 고정할 근거입니다. Conda가 필수라거나 Python·TLS 버전 중 하나가 403의 유일한 원인이었다는 증거는 아닙니다. 실제 서버에서의 아키텍처·출구 네트워크·접근 판정, 전체 카테고리·마지막 페이지·장시간 실행은 미검증입니다. 서버 배포·예약 등록·DB·백엔드·2단계 작업은 수행하지 않았습니다.

## 외부 접속 없는 실행

저장소의 `crawling/` 위치에서 시작하는 예시입니다. 이후 명령은 모두 새 폴더 안에서 실행합니다.

```sh
cd price-refresh-crawler
python3 -m price_refresh --fixture tests/fixtures/listing_response.json --category-limit 1
python3 -m unittest discover -s tests -v
```

`--fixture`는 각 정렬의 1페이지에 지정한 JSON을 그대로 돌려주고, 2페이지부터 정상 빈 목록을 돌려주는 시뮬레이션입니다. 실제 페이지네이션의 녹화·재생 기능은 아닙니다. fixture 실행에서는 실제 대기를 하지 않으며 HTTP 연결을 만들지 않습니다. `--category-limit 1`은 기존 순서의 첫 세부 카테고리만 선택합니다. 미지정 시 53개 전체를 순서대로 사용합니다.

표준 출력에는 JSON만, 로그에는 표준 오류 스트림을 사용합니다. 위 예시의 출력 중 가격 부분은 다음과 같습니다.

```json
{
  "prices": {
    "3263645": 31430,
    "3263646": 32000,
    "3263647": 29000,
    "3263648": 35000,
    "3263649": 33000
  },
  "categories": [
    {
      "main_category": "상의",
      "sub_category": "반소매 셔츠",
      "small_id": "272103105",
      "seen_unique_count": 5,
      "page_requests": 4,
      "stop_reason": "lists_exhausted",
      "sort_status": {
        "RECOMMENDED": "exhausted",
        "MOST_REVIEWED": "exhausted"
      },
      "errors": []
    }
  ]
}
```

새 파일로 저장하려면 `--output`을 지정합니다. 부모 디렉터리는 미리 만들어야 하며 기존 파일은 덮어쓰지 않습니다.

```sh
mkdir -p output
python3 -m price_refresh --fixture tests/fixtures/listing_response.json --category-limit 1 --output output/fixture-prices.json
```

## 실제 조회 명령

`--fixture`를 생략하면 실제 29CM 목록을 요청합니다. 소수 페이지 점검 예시는 **첫 카테고리, 두 정렬 합산 최대 2회**로 제한합니다.

```sh
PRICE_REFRESH_MAX_PAGE_REQUESTS=2 python3 -m price_refresh --category-limit 1
```

기본 설정으로 전체 카테고리를 조회하는 명령은 다음과 같습니다.

```sh
python3 -m price_refresh
```

초기 구현은 기존 코드·fixture로 확인했고, 이후 위 Docker 검증에서 실제 첫 페이지 3회를 확인했습니다. 이 절의 scanner 명령은 사용 방법 설명이며 이번 Docker 검증에서는 실행하지 않았습니다.

## 요청 설정

| 환경변수 | 기본값 | 적용 |
| --- | ---: | --- |
| `PRICE_REFRESH_REQUEST_DELAY` | 1 | 응답 완료 후 초 단위 대기. 오류에도 적용 |
| `PRICE_REFRESH_CONNECT_TIMEOUT` | 5 | TCP/TLS 연결의 타임아웃 |
| `PRICE_REFRESH_READ_TIMEOUT` | 20 | 연결 후 송신 및 응답 헤더·본문 소켓 I/O 타임아웃 |
| `PRICE_REFRESH_CONCURRENCY` | 1 | 1만 허용, 모든 조회를 순차 실행 |
| `PRICE_REFRESH_PAGE_SIZE` | 50 | 기존 기본값 유지. 1..50 허용 |
| `PRICE_REFRESH_MAX_UNIQUE` | 500 | 세부 카테고리별 고유 코드 상한. 1..500 허용 |
| `PRICE_REFRESH_MAX_PAGE_REQUESTS` | 100 | 세부 카테고리별 두 정렬 합산 요청 상한. 1..100 허용 |

상한·페이지 크기는 제한된 점검을 위해 줄일 수 있습니다. 지원 근거 없는 페이지 크기 확대와 조회 상한 초과는 거부합니다. 타임아웃은 양의 유한수, 대기 시간은 0 이상의 유한수여야 합니다.

`.env` 자동 로딩 의존성은 없습니다. 필요하면 예시를 복사한 뒤 셸에서 명시적으로 내보냅니다.

```sh
cp .env.example .env
set -a
. ./.env
set +a
```

기존 HTTP 클라이언트는 `urllib.request.urlopen(request, timeout=...)`에 단일 타임아웃을 전달합니다. 기본 인자는 10초이며 기존 설정 파일의 `CRAWL_TIMEOUT` 기본값은 15초입니다. 이를 그대로 복사하면 연결 5초·응답 대기 20초를 독립적으로 적용할 수 없습니다.

새 클라이언트는 `urllib.request`가 내부적으로 사용하는 표준 라이브러리 `http.client.HTTPSConnection`을 직접 사용합니다. `timeout=5`로 `connect()`를 완료하고, 연결된 `sock.settimeout(20)`을 적용한 후 POST와 응답 읽기를 수행합니다. 이는 개별 소켓 I/O의 대기 제한이며 전체 응답·페이지·작업의 총 소요 시간 제한이 아닙니다. DNS 조회 시간도 별도의 5초 전체 마감으로 보장하지 않습니다. TLS 인증서 검증은 표준 기본값을 사용합니다. 연결은 요청마다 닫으며 자동 재시도·리다이렉트·프록시 환경변수 처리를 추가하지 않았습니다.

관련 표준 동작: [http.client](https://docs.python.org/3.9/library/http.client.html), [socket.settimeout](https://docs.python.org/3.9/library/socket.html#socket.socket.settimeout).

## 순회와 가격 검증

각 세부 카테고리에서 `RECOMMENDED` 1페이지 → `MOST_REVIEWED` 1페이지 → `RECOMMENDED` 2페이지 → `MOST_REVIEWED` 2페이지 순서로 요청합니다. 한 정렬이 정상 빈 목록이면 그 정렬만 종료합니다. 짧은 페이지를 끝으로 추측하지 않습니다.

카테고리마다 다음 중 먼저 충족한 조건에서 종료합니다.

1. 두 정렬 합산 고유 코드 500개: `unique_limit`.
2. 두 정렬 모두 정상 빈 목록: `lists_exhausted`.
3. 두 정렬 합산 페이지 요청 100회: `page_limit`.

코드는 양의 정수 또는 ASCII 숫자 문자열만 허용합니다. 결과는 선행 0을 제거한 숫자 문자열로 통일합니다. bool·음수·0·소수·공백·부호 포함 문자열은 코드에서 제외합니다. 유효 코드라면 가격이 잘못돼도 500개 집계에 포함합니다. 이 집계는 카테고리마다 초기화하며, 실행 전체의 중복 제거와 구분합니다.

가격은 `itemInfo.displayPrice`의 0 이상 `int`만 허용합니다. 실제 0원은 유지합니다. bool·실수(100.0 포함)·숫자 문자열·null·필드 누락·음수는 제외합니다. 부가 필드나 품절·색상·상품명 필터를 요구하지 않습니다. 각 코드의 유효 가격은 두 정렬·전체 카테고리를 통틀어 최대값으로 합칩니다.

한 페이지에서는 응답 순서대로 500번째 고유 코드의 가격까지 처리합니다. 그 뒤 중복 항목의 더 높은 가격이나 새 코드는 처리하지 않고 추가 페이지도 요청하지 않습니다. 따라서 모든 채택 상품을 두 정렬에서 각각 확인한다고 보장하지 않습니다.

## 오류와 로그

정상 `{"data":{"list":[]}}`와 구조 오류·HTTP 오류·타임아웃을 구분합니다. `data.list` 누락·null·잘못된 타입·JSON 오류를 빈 목록으로 바꾸지 않습니다. 목록 내부의 잘못된 코드·가격은 해당 후보만 제외하고 페이지별 제외 개수를 로그에 남깁니다.

조회 또는 응답 구조 오류는 카테고리·정렬·페이지·메시지를 기록하고 해당 정렬을 `error` 상태로 중단합니다. 다른 정렬과 다음 카테고리는 계속 조회합니다. 실패 요청도 100회 상한에 포함하며 다시 요청하지 않습니다. 두 정렬이 모두 비활성화됐는데 오류가 있으면 `lookup_error`로 기록합니다. 상품·페이지 상한으로 먼저 멈추더라도 발생한 오류는 `errors`에 남습니다. 상한에 도달했을 때의 `active`는 목록의 끝을 확인하지 못했다는 뜻입니다.

각 카테고리의 시작·정렬 종료·조회 오류·최종 종료 이유·고유 코드 수·요청 수를 로그로 남깁니다. 유효 가격이 없는 코드는 `prices`에 넣지 않으며 조회 실패를 0원으로 바꾸지 않습니다.

CLI 종료 코드는 오류 없는 실행이면 0, 조회 오류가 있는 부분 결과 또는 설정·파일 오류면 1입니다. 조회 오류가 있어도 확보한 `prices`와 카테고리별 오류를 JSON으로 제공합니다. `page_limit`은 조회 범위의 상한에 따른 종료이며 그 자체를 요청 실패로 처리하지 않습니다.

## 기존 코드에서 가져온 부분과 확인 범위

| 기존 파일 | 활용 내용 |
| --- | --- |
| `app/crawler/client.py` | `TwentyNineCmClient.fetch_listing`의 endpoint, POST, query, headers, category facets, 정렬 상수, 1부터 시작하는 page, 기본 size 50. 전송부는 타임아웃 분리를 위해 변경 |
| `app/config/categories.py` | `CATEGORIES` 53개와 세부 카테고리 순서를 파일 그대로 복사. 외부 import 없음 |
| `app/crawler/parser.py` | `data.list`, `itemId`, `itemInfo.displayPrice` 경로 확인. 기존 Product 모델과 필수 부가 필드는 가져오지 않음 |
| `tests/fixtures/listing_response.json` | 원본 파일을 그대로 복사. `itemEvent`에도 가격이 있지만 `itemInfo`만 사용하는지 검증 |
| `app/config/settings.py`, `requirements.txt` | 기존 기본 page size 및 요청 설정 확인. 기존 DB 라이브러리 `psycopg`는 이 단계에 필요 없어 의존성에서 제외 |

기존 quota 수집기는 import하거나 복사하지 않았고, 기존 수집 파일도 수정하지 않았습니다. 새 폴더에서 부모 저장소 모듈을 참조하지 않습니다.

확보한 fixture는 예시 이미지 URL 등을 포함하고 있으며 가공되지 않은 실제 HTTP 원본 캡처인지는 확인할 수 없습니다. 마지막 페이지 메타데이터가 없어 해당 필드를 새로 가정하지 않았습니다. 위 Docker 실조회에서 한 세부 카테고리의 두 정렬 첫 페이지와 50개 반환만 확인했으며, 전체 정렬 순위의 정확성·마지막 페이지 형식·다른 카테고리는 미검증입니다.

## 테스트

`python3 -m unittest discover -s tests -v`로 교대 순서, 한 정렬 종료, 고유 500개와 페이지 중간 경계, 반복 코드·합산 100회, 카테고리별 집계 분리, 전역 최대 가격, 잘못된 코드·가격과 0원, 정상 빈 목록·응답 오류, 대기 순서, 연결·응답 타임아웃, 부분 실패 출력, 독립 실행을 확인합니다. 테스트는 fake client·fake connection·가짜 대기를 사용하고 외부 네트워크나 실제 1초 대기를 사용하지 않습니다.
