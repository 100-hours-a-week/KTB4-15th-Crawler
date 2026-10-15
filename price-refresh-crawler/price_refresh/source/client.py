"""Request contract adapted from app/crawler/client.py; separate timeouts."""

import http.client
import json
from collections.abc import Mapping
from typing import Any, Callable
from urllib.parse import urlencode, urlsplit

from ..settings import Settings


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


class ListingRequestError(RuntimeError):
    """HTTP, transport or JSON failure, distinct from an empty listing."""


class TwentyNineCmClient:
    def __init__(self, settings: Settings, *,
                 connection_factory: Callable[..., Any] = http.client.HTTPSConnection):
        self.settings = settings
        self._connection_factory = connection_factory

    def fetch_listing(self, *, largeId: str, middleId: str, smallId: str,
                      sort: str, page: int = 1, size: int = 50) -> Mapping:
        body = {
            "pageType": "CATEGORY_PLP",
            "sortType": sort,
            "facets": {"categoryFacetInputs": [{
                "largeId": int(largeId), "middleId": int(middleId),
                "smallId": int(smallId),
            }]},
            "pageRequest": {"page": page, "size": size},
        }
        url = urlsplit(LISTING_ENDPOINT)
        path = f"{url.path}?{urlencode({'colorchipVariant': 'treatment'})}"
        connection = self._connection_factory(
            url.hostname, timeout=self.settings.connect_timeout
        )
        try:
            # TCP/TLS setup uses connect_timeout. Switch before sending/reading.
            connection.connect()
            connection.sock.settimeout(self.settings.read_timeout)
            connection.request("POST", path, body=json.dumps(body).encode("utf-8"),
                               headers=dict(DEFAULT_HEADERS))
            with connection.getresponse() as response:
                raw = response.read()
                if not 200 <= response.status < 300:
                    raise ListingRequestError(f"29CM HTTP {response.status}")
            payload = json.loads(raw.decode("utf-8"))
            if not isinstance(payload, Mapping):
                raise ListingRequestError("29CM JSON response must be an object")
            return payload
        except (OSError, http.client.HTTPException, ValueError) as error:
            raise ListingRequestError(f"29CM request failed: {error}") from error
        finally:
            connection.close()
