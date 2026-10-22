from dataclasses import dataclass

from .parser import display_price, product_code


def normalize_code(value):
    return product_code({"itemId": value})


def valid_price(value):
    return display_price({"itemInfo": {"displayPrice": value}})


def code_order(code):
    return len(code), code


@dataclass(frozen=True)
class PriceChange:
    product_code: str
    old_price: int
    new_price: int


@dataclass
class Comparison:
    changes: list[PriceChange]
    invalid_stored_rows: int


def compare_prices(stored_rows, observed_prices):
    stored = {}
    invalid = 0
    for raw_code, raw_price in stored_rows:
        code, price = normalize_code(raw_code), valid_price(raw_price)
        if code is None or price is None:
            invalid += 1
            continue
        if code in stored and stored[code] != price:
            raise ValueError("Conflicting stored prices for one product code")
        stored[code] = price
    current = {}
    for raw_code, raw_price in observed_prices.items():
        code, price = normalize_code(raw_code), valid_price(raw_price)
        if code is None or price is None:
            continue
        if code in current and current[code] != price:
            raise ValueError("Conflicting finalized scanner prices")
        current[code] = price
    changes = [PriceChange(code, stored[code], current[code])
               for code in sorted(stored.keys() & current.keys(), key=code_order)
               if stored[code] != current[code]]
    return Comparison(changes, invalid)
