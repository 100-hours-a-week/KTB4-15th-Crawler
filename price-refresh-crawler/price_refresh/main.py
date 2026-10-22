"""CLI for collected prices, previews, apply + outbox recording, outbox delivery and its control."""

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
from .apply import run_apply
from .backend import BackendClient, BackendError, BackendSettings
from .delivery import EXIT_CODES, DeliverySettings, run_delivery
from .delivery_control import DeliveryControlError, fetch_delivery_status, run_resume
from .writer import DeliverySession, PostgresApplySession
from .logging_config import configure_logging
from .repository import (
    DatabaseSettings, FixturePriceRepository, PostgresPriceRepository, RepositoryError,
)


def write_json(payload, destination):
    output = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if destination:
        with destination.open("x", encoding="utf-8") as file:
            file.write(output)
    else:
        print(output, end="")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Collect 29CM listing prices")
    parser.add_argument("--fixture", type=Path, help="Offline listing JSON; no network or real waits")
    parser.add_argument("--category-limit", type=int, help="Scan only the first N configured categories")
    parser.add_argument("--output", type=Path, help="Write JSON to a new file instead of stdout")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--dry-run", action="store_true", help="Read stored prices, compare and preview only")
    modes.add_argument("--apply", action="store_true",
                       help="Apply prices and queue recommended changes in the outbox under a DB "
                            "session lock; never sends to the backend")
    modes.add_argument("--deliver", action="store_true",
                       help="One bounded outbox delivery run to the backend, then exit; "
                            "exit 0 done, 3 authentication stop, 1 failure")
    modes.add_argument("--delivery-status", action="store_true",
                       help="Read-only: is outbox delivery stopped by an authentication failure? "
                            "Exit 0 allowed, 3 stopped, 1 check failed")
    modes.add_argument("--resume-delivery", action="store_true",
                       help="Explicitly clear the authentication stop under the delivery lock; "
                            "sends nothing")
    parser.add_argument("--stored-products-fixture", type=Path,
                        help="Offline DB substitute; requires --dry-run and --fixture")
    args = parser.parse_args(argv)
    if args.category_limit is not None and args.category_limit < 1:
        parser.error("--category-limit must be positive")
    if args.apply and (args.fixture or args.stored_products_fixture):
        parser.error("--apply cannot use listing or stored-products fixtures")
    if args.stored_products_fixture and not (args.dry_run and args.fixture):
        parser.error("--stored-products-fixture requires --dry-run and --fixture")
    control = args.deliver or args.delivery_status or args.resume_delivery
    if control and (args.fixture or args.category_limit is not None):
        parser.error("--deliver/--delivery-status/--resume-delivery take no listing options")
    configure_logging()
    if args.deliver:
        return deliver(args)
    if control:
        return delivery_control(args)
    try:
        settings = Settings.from_env()
        categories = CATEGORIES[:args.category_limit]
        if args.fixture:
            client = FixtureClient(args.fixture)
            collect = lambda: collect_prices(client, categories, settings, sleep=lambda _: None)
        else:
            collect = lambda: collect_prices(TwentyNineCmClient(settings), categories, settings)
        if args.apply:
            # Backend settings are not read: delivery is a separate outbox job.
            report = run_apply(PostgresApplySession(DatabaseSettings.from_env()), collect,
                               emit=lambda payload: write_json(payload, args.output))
            return 0 if report.status in {"completed", "skipped_concurrent"} else 1
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
        write_json(payload, args.output)
        return 1 if failed else 0
    except RepositoryError as error:
        logging.error("database_configuration_or_read_failed code=%s", error.code)
        return 1
    except (OSError, ValueError) as error:
        logging.error("execution_failed error=%s", error)
        return 1


def deliver(args):
    """All configuration is validated before the database is opened; 29CM settings are not read."""
    try:
        database = DatabaseSettings.from_env()
        client = BackendClient(BackendSettings.from_env())
        settings = DeliverySettings.from_env()
    except RepositoryError as error:
        logging.error("database_configuration_failed code=%s", error.code)
        return 1
    except BackendError as error:
        logging.error("backend_configuration_failed code=%s", error.code)
        return 1
    except ValueError as error:
        logging.error("delivery_configuration_failed code=%s", error)
        return 1
    try:
        report = run_delivery(DeliverySession(database), client, settings,
                              emit=lambda payload: write_json(payload, args.output))
    except OSError as error:
        logging.error("execution_failed error=%s", type(error).__name__)
        return 1
    return EXIT_CODES.get(report.status, 1)


def delivery_control(args):
    try:
        settings = DatabaseSettings.from_env()
        if args.resume_delivery:
            report = run_resume(DeliverySession(settings), emit=lambda payload: write_json(payload, args.output))
            return 0 if report.status in {"resumed", "already_allowed"} else 1
        try:
            stop = fetch_delivery_status(settings)
        except DeliveryControlError as error:
            logging.error("delivery_status_failed code=%s", error.code)
            write_json({"mode": "delivery_status", "status": "check_failed", "delivery_allowed": None,
                        "error": error.code}, args.output)
            return 1
        write_json({"mode": "delivery_status", "status": "stopped" if stop.stopped else "allowed",
                    "delivery_allowed": not stop.stopped, "auth_failed_count": stop.auth_failed_count,
                    "http_statuses": list(stop.http_statuses)}, args.output)
        return 3 if stop.stopped else 0
    except RepositoryError as error:
        logging.error("database_configuration_or_read_failed code=%s", error.code)
        return 1
    except OSError as error:
        logging.error("execution_failed error=%s", type(error).__name__)
        return 1
