"""29CM BFF listing API HTTP client 테스트."""

import json
import unittest
from urllib.parse import parse_qs, urlparse

from app.crawler.client import RECOMMENDED, TwentyNineCmClient
from test_parser import FIXTURE_PATH


class _Response:
    def __init__(self, payload, status=200, content_type="application/json"):
        self.payload = json.dumps(payload).encode("utf-8")
        self.status = status
        self.headers = {"Content-Type": content_type}

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return None

    def read(self):
        return self.payload

    def getcode(self):
        return self.status


class ClientTests(unittest.TestCase):
    def setUp(self):
        self.response = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))

    def test_builds_post_json_request(self):
        requests = []

        def opener(request, timeout):
            requests.append((request, timeout))
            return _Response(self.response)

        client = TwentyNineCmClient(
            endpoint="https://example.test/api/items",
            query_params={"extra": "value"},
            headers={"X-Test": "test"},
            timeout=3,
            opener=opener,
        )
        result = client.fetch_listing(
            largeId="272100100",
            middleId="272103100",
            smallId="272103108",
            sort=RECOMMENDED,
        )

        request = requests[0][0]
        query = parse_qs(urlparse(request.full_url).query)
        body = json.loads(request.data.decode("utf-8"))
        self.assertEqual(query["colorchipVariant"], ["treatment"])
        self.assertEqual(query["extra"], ["value"])
        self.assertEqual(request.method, "POST")
        self.assertEqual(body["sortType"], RECOMMENDED)
        self.assertEqual(body["pageRequest"], {"page": 1, "size": 50})
        self.assertEqual(
            body["facets"]["categoryFacetInputs"][0],
            {"largeId": 272100100, "middleId": 272103100, "smallId": 272103108},
        )
        self.assertEqual(request.headers["X-test"], "test")
        self.assertEqual(requests[0][1], 3)
        self.assertEqual(result, self.response)


if __name__ == "__main__":
    unittest.main()
