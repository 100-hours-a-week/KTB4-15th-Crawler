"""Validate the observed data.list / itemId / itemInfo.displayPrice fields."""

import re
from collections.abc import Mapping
from typing import Any, Optional


class ListingResponseError(ValueError):
    """Malformed listing envelope, never a normal empty page."""


def listing_items(payload: Any) -> list:
    if not isinstance(payload, Mapping):
        raise ListingResponseError("Response must be a JSON object")
    data = payload.get("data")
    if not isinstance(data, Mapping) or not isinstance(data.get("list"), list):
        raise ListingResponseError("Expected data.list to be an array")
    return data["list"]


def product_code(item: Any) -> Optional[str]:
    if not isinstance(item, Mapping):
        return None
    value = item.get("itemId")
    if type(value) is int and value > 0:
        return str(value)
    if type(value) is str and re.fullmatch(r"[0-9]+", value):
        # Avoid int conversion limits for unusually long identifiers.
        normalized = value.lstrip("0")
        return normalized or None
    return None


def display_price(item: Any) -> Optional[int]:
    if not isinstance(item, Mapping):
        return None
    info = item.get("itemInfo")
    if not isinstance(info, Mapping):
        return None
    price = info.get("displayPrice")
    # bool, float (even 1.0), numeric strings and null are not integer KRW.
    return price if type(price) is int and price >= 0 else None
