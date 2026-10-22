"""Read-only schema/privilege preflight before any 29CM request or price write.

Only pg_catalog and has_*_privilege() are queried: no table data is read or changed,
nothing is created, and privileges are never probed by writing and rolling back.
"""

from dataclasses import dataclass

SCHEMA = "public"
PRODUCTS = "products"
OUTBOX = "price_change_outbox"
# (column, format_type(), NOT NULL required)
PRODUCT_COLUMNS = (
    ("product_code", "bigint", True),
    ("price", "integer", True),
    # Nullability is owned by the recommendation service; a NULL is never read as FALSE.
    ("has_been_recommended", "boolean", False),
)
OUTBOX_COLUMNS = (
    ("id", "bigint", True),
    ("scan_run_id", "uuid", True),
    ("product_code", "bigint", True),
    ("original_old_price", "integer", True),
    ("old_price", "integer", True),
    ("new_price", "integer", True),
    ("status", "character varying(16)", True),
    ("attempt_count", "integer", True),
    ("next_attempt_at", "timestamp with time zone", False),
    ("last_attempt_at", "timestamp with time zone", False),
    ("last_backend_price", "integer", False),
    ("last_http_status", "smallint", False),
    ("last_error_code", "character varying(64)", False),
    ("last_error_message", "text", False),
    ("created_at", "timestamp with time zone", True),
    ("updated_at", "timestamp with time zone", True),
    ("completed_at", "timestamp with time zone", False),
)


class ReadinessError(RuntimeError):
    """The check itself could not run; the driver message is deliberately dropped."""

    def __init__(self, code):
        self.code = code
        super().__init__(code)


@dataclass(frozen=True)
class ReadinessIssue:
    code: str
    target: str


def _one(cursor, sql, params=()):
    cursor.execute(sql, params)
    return cursor.fetchone()


def _tables(cursor):
    cursor.execute(
        "SELECT c.relname, c.oid FROM pg_catalog.pg_class c "
        "JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace "
        "WHERE n.nspname = %s AND c.relname = ANY(%s) AND c.relkind IN ('r', 'p')",
        (SCHEMA, [PRODUCTS, OUTBOX]),
    )
    return dict(cursor.fetchall())


def _columns(cursor, table_oid):
    cursor.execute(
        "SELECT attname, pg_catalog.format_type(atttypid, atttypmod), attnotnull, attidentity "
        "FROM pg_catalog.pg_attribute WHERE attrelid = %s::oid AND attnum > 0 AND NOT attisdropped",
        (table_oid,),
    )
    return {name: (type_name, not_null, identity)
            for name, type_name, not_null, identity in cursor.fetchall()}


def _check_columns(cursor, table, oid, expected, issues):
    actual = _columns(cursor, oid)
    for name, type_name, not_null in expected:
        target = f"{table}.{name}"
        if name not in actual:
            issues.append(ReadinessIssue("column_missing", target))
        elif actual[name][0] != type_name:
            issues.append(ReadinessIssue("column_type_mismatch", target))
        elif not_null and not actual[name][1]:
            issues.append(ReadinessIssue("column_definition_mismatch", target))
    return actual


def _has_unique(cursor, table_oid, columns):
    """A unique index (PK, UNIQUE or unique index) on exactly these columns, any order."""
    return _one(cursor,
        "SELECT EXISTS (SELECT 1 FROM pg_catalog.pg_index i WHERE i.indrelid = %s::oid "
        "AND i.indisunique AND i.indisvalid AND i.indpred IS NULL AND i.indexprs IS NULL "
        "AND i.indnkeyatts = %s AND (SELECT array_agg(a.attname::text ORDER BY a.attname) "
        "FROM pg_catalog.pg_attribute a WHERE a.attrelid = i.indrelid "
        "AND a.attnum = ANY(i.indkey[0:i.indnkeyatts - 1])) = %s)",
        (table_oid, len(columns), sorted(columns)))[0]


def _has_product_fk(cursor, outbox_oid, products_oid):
    return _one(cursor,
        "SELECT EXISTS (SELECT 1 FROM pg_catalog.pg_constraint c "
        "JOIN pg_catalog.pg_attribute src ON src.attrelid = c.conrelid AND src.attnum = c.conkey[1] "
        "JOIN pg_catalog.pg_attribute dst ON dst.attrelid = c.confrelid AND dst.attnum = c.confkey[1] "
        "WHERE c.contype = 'f' AND c.conrelid = %s::oid AND c.confrelid = %s::oid "
        "AND cardinality(c.conkey) = 1 AND src.attname = 'product_code' "
        "AND dst.attname = 'product_code')",
        (outbox_oid, products_oid))[0]


def _check_privileges(cursor, tables, issues):
    if not _one(cursor, "SELECT has_schema_privilege(%s, 'USAGE')", (SCHEMA,))[0]:
        issues.append(ReadinessIssue("privilege_missing", f"USAGE schema {SCHEMA}"))
    checks = []
    if PRODUCTS in tables:
        checks += [("SELECT", PRODUCTS, None), ("UPDATE", PRODUCTS, "price")]
    if OUTBOX in tables:
        checks += [(privilege, OUTBOX, None) for privilege in ("SELECT", "INSERT", "UPDATE")]
    for privilege, table, column in checks:
        if column is None:
            allowed = _one(cursor, "SELECT has_table_privilege(%s::oid, %s)",
                           (tables[table], privilege))[0]
        else:
            allowed = _one(cursor, "SELECT has_column_privilege(%s::oid, %s::text, %s)",
                           (tables[table], column, privilege))[0]
        if not allowed:
            target = f"{table}.{column}" if column else table
            issues.append(ReadinessIssue("privilege_missing", f"{privilege} {target}"))


def check_readiness(connection):
    """Return every detected problem; an empty list means collection may start.

    Uses the caller's connection without committing or closing it. On a
    non-autocommit connection only catalog reads happen inside the open transaction.
    """
    issues = []
    try:
        with connection.cursor() as cursor:
            tables = _tables(cursor)
            for table, expected in ((PRODUCTS, PRODUCT_COLUMNS), (OUTBOX, OUTBOX_COLUMNS)):
                if table not in tables:
                    issues.append(ReadinessIssue("table_missing", f"{SCHEMA}.{table}"))
                    continue
                actual = _check_columns(cursor, table, tables[table], expected, issues)
                if table == OUTBOX and "id" in actual and actual["id"][2] not in {"a", "d"}:
                    issues.append(ReadinessIssue("column_definition_mismatch", f"{OUTBOX}.id identity"))
            if PRODUCTS in tables and not _has_unique(cursor, tables[PRODUCTS], ["product_code"]):
                issues.append(ReadinessIssue("constraint_missing", f"{PRODUCTS} unique(product_code)"))
            if OUTBOX in tables:
                if not _has_unique(cursor, tables[OUTBOX], ["scan_run_id", "product_code"]):
                    issues.append(ReadinessIssue(
                        "constraint_missing", f"{OUTBOX} unique(scan_run_id, product_code)"))
                if PRODUCTS in tables and not _has_product_fk(cursor, tables[OUTBOX], tables[PRODUCTS]):
                    issues.append(ReadinessIssue(
                        "constraint_missing", f"{OUTBOX} foreign key(product_code)"))
            _check_privileges(cursor, tables, issues)
    except Exception:
        raise ReadinessError("readiness_check_failed") from None
    return issues
