"""CLI for collected prices and read-only price comparison previews."""

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
from .dry_run import run_dry_run
from .repository import (
    DatabaseSettings, FixturePriceRepository, PostgresPriceRepository, RepositoryError,
)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Collect 29CM listing prices")
    parser.add_argument("--fixture", type=Path, help="Offline listing JSON; no network or real waits")
    parser.add_argument("--category-limit", type=int, help="Scan only the first N configured categories")
    parser.add_argument("--output", type=Path, help="Write JSON to a new file instead of stdout")
    parser.add_argument("--dry-run", action="store_true", help="Read stored prices, compare and preview only")
    parser.add_argument("--stored-products-fixture", type=Path,
                        help="Offline DB substitute; requires --dry-run and --fixture")
    args = parser.parse_args(argv)
    if args.category_limit is not None and args.category_limit < 1:
        parser.error("--category-limit must be positive")
    if args.stored_products_fixture and not (args.dry_run and args.fixture):
        parser.error("--stored-products-fixture requires --dry-run and --fixture")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        settings = Settings.from_env()
        categories = CATEGORIES[:args.category_limit]
        if args.fixture:
            client = FixtureClient(args.fixture)
            collect = lambda: collect_prices(client, categories, settings, sleep=lambda _: None)
        else:
            collect = lambda: collect_prices(TwentyNineCmClient(settings), categories, settings)
        if args.dry_run:
            repository = (FixturePriceRepository(args.stored_products_fixture)
                          if args.stored_products_fixture else
                          PostgresPriceRepository(DatabaseSettings.from_env()))
            payload = run_dry_run(repository, collect)
            failed = payload["recommendation_lookup"]["status"] == "failed" or payload["scan"]["has_errors"]
        else:
            result = collect()
            payload = asdict(result)
            failed = any(category.errors for category in result.categories)
        output = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
        if args.output:
            # Avoid overwriting results from an earlier run.
            with args.output.open("x", encoding="utf-8") as file:
                file.write(output)
        else:
            print(output, end="")
        return 1 if failed else 0
    except RepositoryError as error:
        logging.error("dry_run_failed code=%s", error.code)
        return 1
    except (OSError, ValueError) as error:
        logging.error("execution_failed error=%s", error)
        return 1
