"""Read-only products access, compatible with the AI DATABASE_URL/psycopg 3."""

import json
import os
from contextlib import closing
from dataclasses import dataclass, field
from typing import Protocol

from .comparison import code_order, normalize_code


class RepositoryError(RuntimeError):
    """Only a safe error code; never retain a driver message containing a DSN."""

    def __init__(self, code):
        allowed = {
            "invalid_database_url", "invalid_database_settings", "database_url_missing",
            "recommendation_column_missing", "database_read_failed", "invalid_product_code",
            "invalid_recommendation_result", "incomplete_recommendation_result",
            "invalid_database_fixture", "recommendation_lookup_failed",
        }
        self.code = code if code in allowed else "database_read_failed"
        super().__init__(self.code)


class PriceRepository(Protocol):
    def fetch_prices(self) -> list[tuple]: ...
    def fetch_recommendations(self, product_codes: list[str]) -> dict[str, bool]: ...


@dataclass(frozen=True)
class DatabaseSettings:
    url: str = field(repr=False)
    connect_timeout: int = 5
    recommendation_batch_size: int = 1000

    def __post_init__(self):
        if not isinstance(self.url, str) or not self.url.strip().startswith(("postgresql://", "postgres://")):
            raise RepositoryError("invalid_database_url")
        for value in (self.connect_timeout, self.recommendation_batch_size):
            if type(value) is not int or value < 1:
                raise RepositoryError("invalid_database_settings")

    @classmethod
    def from_env(cls):
        url = os.getenv("DATABASE_URL", "").strip()
        if not url:
            raise RepositoryError("database_url_missing")
        try:
            return cls(url, int(os.getenv("PRICE_REFRESH_DB_CONNECT_TIMEOUT", "5")),
                       int(os.getenv("PRICE_REFRESH_RECOMMENDATION_BATCH_SIZE", "1000")))
        except (ValueError, TypeError):
            raise RepositoryError("invalid_database_settings") from None


def recommendation_rows(rows, requested):
    requested = set(requested)
    result = {}
    for raw_code, flag in rows:
        code = normalize_code(raw_code)
        if code not in requested or type(flag) is not bool or code in result:
            raise RepositoryError("invalid_recommendation_result")
        result[code] = flag
    if result.keys() != requested:
        raise RepositoryError("incomplete_recommendation_result")
    return result


class PostgresPriceRepository:
    def __init__(self, settings, *, connect=None):
        self.settings = settings
        self._connect = connect

    def _read(self, operation, *, recommendations=False):
        try:
            connect = self._connect
            if connect is None:
                import psycopg
                connect = psycopg.connect
            # No connection context manager: it would commit on normal exit.
            # Closing a non-autocommit connection discards its read transaction.
            with closing(connect(self.settings.url, connect_timeout=self.settings.connect_timeout,
                                 autocommit=False)) as connection:
                connection.read_only = True  # before the first statement/BEGIN
                with connection.cursor() as cursor:
                    return operation(cursor)
        except RepositoryError:
            raise
        except Exception as error:
            code = ("recommendation_column_missing"
                    if recommendations and getattr(error, "sqlstate", None) == "42703"
                    else "database_read_failed")
            raise RepositoryError(code) from None

    def fetch_prices(self):
        def read(cursor):
            cursor.execute("SELECT product_code, price FROM products")
            return cursor.fetchall()
        return self._read(read)

    def fetch_recommendations(self, product_codes):
        codes = sorted(set(product_codes), key=code_order)
        if not codes:
            return {}
        # products.product_code is bigint; never interpolate values into SQL.
        if any(normalize_code(code) != code or len(code) > 19
               or int(code) > 9223372036854775807 for code in codes):
            raise RepositoryError("invalid_product_code")

        def read(cursor):
            rows = []
            for start in range(0, len(codes), self.settings.recommendation_batch_size):
                batch = codes[start:start + self.settings.recommendation_batch_size]
                cursor.execute(
                    "SELECT product_code, has_been_recommended FROM products "
                    "WHERE product_code = ANY(%s)", ([int(code) for code in batch],),
                )
                rows.extend(cursor.fetchall())
            return recommendation_rows(rows, codes)
        return self._read(read, recommendations=True)


class FixturePriceRepository:
    """Explicit offline DB substitute; never connects, even if DATABASE_URL is set."""

    def __init__(self, path):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or not isinstance(payload.get("products"), list):
            raise RepositoryError("invalid_database_fixture")
        if any(not isinstance(row, dict) or "product_code" not in row or "price" not in row
               for row in payload["products"]):
            raise RepositoryError("invalid_database_fixture")
        self.payload = payload

    def fetch_prices(self):
        return [(row["product_code"], row["price"]) for row in self.payload["products"]]

    def fetch_recommendations(self, product_codes):
        if self.payload.get("recommendation_error") == "recommendation_column_missing":
            raise RepositoryError("recommendation_column_missing")
        flags = self.payload.get("recommendations")
        if not isinstance(flags, list):
            raise RepositoryError("recommendation_lookup_failed")
        requested = set(product_codes)
        rows = [(row.get("product_code"), row.get("has_been_recommended"))
                for row in flags if isinstance(row, dict)
                and normalize_code(row.get("product_code")) in requested]
        return recommendation_rows(rows, requested)
