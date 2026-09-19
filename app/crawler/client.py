"""29CM 상품 목록 Fetch/XHR JSON API 클라이언트."""

import json
import logging
from typing import Any, Callable, Mapping, Optional
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


logger = logging.getLogger(__name__)

LISTING_ENDPOINT = "https://display-bff-api.29cm.co.kr/api/v1/listing/items"
RECOMMENDED = "RECOMMENDED"
MOST_REVIEWED = "MOST_REVIEWED"
DEFAULT_HEADERS = {
    "Content-Type": "application/json",
    "Accept": "*/*",
    "Origin": "https://www.29cm.co.kr",
    "Referer": "https://www.29cm.co.kr/",
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/537.36.0.0 Safari/537.36"
    ),
}


class TwentyNineCmClient:
    """카테고리와 정렬 조건으로 29CM 원본 JSON을 POST 요청한다."""

    def __init__(
        self,
        *,
        endpoint: str = LISTING_ENDPOINT,
        query_params: Optional[Mapping[str, str]] = None,
        headers: Optional[Mapping[str, str]] = None,
        timeout: float = 10.0,
        opener: Optional[Callable[..., Any]] = None,
    ) -> None:
        self.endpoint = endpoint
        self.query_params = {"colorchipVariant": "treatment", **(query_params or {})}
        self.headers = {**DEFAULT_HEADERS, **(headers or {})}
        self.timeout = timeout
        self._opener = opener or urlopen

    def fetch_listing(
        self,
        *,
        largeId: str,
        middleId: str,
        smallId: str,
        sort: str,
        page: int = 1,
        size: int = 50,
        query_params: Optional[Mapping[str, str]] = None,
        headers: Optional[Mapping[str, str]] = None,
    ) -> Mapping[str, Any]:
        """목록 API의 원본 JSON 객체를 반환한다."""
        params = {**self.query_params, **(query_params or {})}
        body = {
            "pageType": "CATEGORY_PLP",
            "sortType": sort,
            "facets": {
                "categoryFacetInputs": [
                    {
                        "largeId": int(largeId),
                        "middleId": int(middleId),
                        "smallId": int(smallId),
                    }
                ]
            },
            "pageRequest": {"page": page, "size": size},
        }
        request = Request(
            f"{self.endpoint}?{urlencode(params)}",
            data=json.dumps(body).encode("utf-8"),
            headers={**self.headers, **(headers or {})},
            method="POST",
        )
        try:
            with self._opener(request, timeout=self.timeout) as response:
                response_body = response.read().decode("utf-8")
                status = getattr(response, "status", None) or response.getcode()
                content_type = response.headers.get("Content-Type", "")
        except HTTPError as error:
            response_body = error.read().decode("utf-8", errors="replace")
            status = error.code
            content_type = error.headers.get("Content-Type", "")
            self._log_response_failure(status, content_type, response_body)
            raise ValueError(f"29CM 목록 API 요청 실패: HTTP {status}") from error

        if status < 200 or status >= 300:
            self._log_response_failure(status, content_type, response_body)
            raise ValueError(f"29CM 목록 API 요청 실패: HTTP {status}")
        try:
            payload = json.loads(response_body)
        except json.JSONDecodeError as error:
            self._log_response_failure(status, content_type, response_body)
            raise ValueError("29CM 목록 API 응답이 JSON이 아닙니다") from error
        if not isinstance(payload, Mapping):
            raise ValueError("29CM 목록 응답은 JSON 객체여야 합니다")
        return payload

    @staticmethod
    def _log_response_failure(status: int, content_type: str, body: str) -> None:
        logger.error(
            "29CM 목록 API 실패: status=%s content_type=%s body_prefix=%r",
            status,
            content_type,
            body[:500],
        )


def fetch_listing(
    *,
    largeId: str,
    middleId: str,
    smallId: str,
    sort: str,
    page: int = 1,
    size: int = 50,
    endpoint: str = LISTING_ENDPOINT,
    query_params: Optional[Mapping[str, str]] = None,
    headers: Optional[Mapping[str, str]] = None,
    timeout: float = 10.0,
) -> Mapping[str, Any]:
    """주입한 JSON API 설정으로 목록을 한 번 요청하는 편의 함수."""
    return TwentyNineCmClient(
        endpoint=endpoint,
        query_params=query_params,
        headers=headers,
        timeout=timeout,
    ).fetch_listing(
        largeId=largeId,
        middleId=middleId,
        smallId=smallId,
        sort=sort,
        page=page,
        size=size,
    )
