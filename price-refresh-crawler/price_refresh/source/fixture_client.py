"""Offline simulation: fixture on page 1 of each sort, then empty pages."""

import json
from pathlib import Path

from ..parser import listing_items


class FixtureClient:
    def __init__(self, path: Path):
        self.payload = json.loads(path.read_text(encoding="utf-8"))
        listing_items(self.payload)

    def fetch_listing(self, *, largeId, middleId, smallId, sort, page=1, size=50):
        return self.payload if page == 1 else {"data": {"list": []}}
