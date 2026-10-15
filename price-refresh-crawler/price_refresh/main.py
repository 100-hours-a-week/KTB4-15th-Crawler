"""CLI for raw collected prices; no stored-price comparison is performed."""

import argparse
import json
import logging
from dataclasses import asdict
from pathlib import Path

from .scanner import collect_prices
from .settings import Settings
from .source.categories import CATEGORIES
from .source.client import TwentyNineCmClient
from .source.fixture_client import FixtureClient


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Collect 29CM listing prices")
    parser.add_argument("--fixture", type=Path, help="Offline listing JSON; no network or real waits")
    parser.add_argument("--category-limit", type=int, help="Scan only the first N configured categories")
    parser.add_argument("--output", type=Path, help="Write JSON to a new file instead of stdout")
    args = parser.parse_args(argv)
    if args.category_limit is not None and args.category_limit < 1:
        parser.error("--category-limit must be positive")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        settings = Settings.from_env()
        categories = CATEGORIES[:args.category_limit]
        if args.fixture:
            client = FixtureClient(args.fixture)
            result = collect_prices(client, categories, settings, sleep=lambda _: None)
        else:
            result = collect_prices(TwentyNineCmClient(settings), categories, settings)
        output = json.dumps(asdict(result), ensure_ascii=False, indent=2) + "\n"
        if args.output:
            # Avoid overwriting results from an earlier run.
            with args.output.open("x", encoding="utf-8") as file:
                file.write(output)
        else:
            print(output, end="")
        return 1 if any(category.errors for category in result.categories) else 0
    except (OSError, ValueError) as error:
        logging.error("execution_failed error=%s", error)
        return 1
