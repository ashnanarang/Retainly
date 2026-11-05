import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_DB_PATH = PROJECT_DIR / "data" / "cancellation_cases.sqlite3"
_SUPABASE_CLIENT = None


def _remote_config() -> tuple[str | None, str | None]:
    return (
        os.environ.get("SUPABASE_URL"),
        os.environ.get("SUPABASE_SERVICE_ROLE_KEY"),
    )


def _remote_required() -> bool:
    return os.environ.get("CANCELLATION_REQUIRE_REMOTE_DB", "").lower() in {"1", "true", "yes"}


def _supabase_client():
    global _SUPABASE_CLIENT
    url, key = _remote_config()
    if not url or not key:
        if _remote_required():
            raise sqlite3.OperationalError(
                "Remote case storage is required but Supabase secrets are not configured."
            )
        return None
    if _SUPABASE_CLIENT is not None:
        return _SUPABASE_CLIENT
    try:
        from supabase import create_client
        _SUPABASE_CLIENT = create_client(url, key)
        return _SUPABASE_CLIENT
    except Exception as exc:
        raise sqlite3.OperationalError("Could not initialize Supabase case storage.") from exc


def remote_storage_configured() -> bool:
    url, key = _remote_config()
    return bool(url and key)


def database_path() -> Path:
    configured_path = os.environ.get("CANCELLATION_DB_PATH")
    path = Path(configured_path).expanduser() if configured_path else DEFAULT_DB_PATH
    if not path.is_absolute():
        path = PROJECT_DIR / path
    return path


@contextmanager
def _connection():
    path = database_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=10)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    try:
        with connection:
            yield connection
    finally:
        connection.close()


def initialize_database() -> None:
    client = _supabase_client()
    if client is not None:
        try:
            client.table("cancellation_cases").select("case_id").limit(1).execute()
        except Exception as exc:
            raise sqlite3.OperationalError(
                "Supabase is configured, but the cancellation_cases table is unavailable. Apply supabase_schema.sql."
            ) from exc
        return

    with _connection() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS cancellation_cases (
                case_id INTEGER PRIMARY KEY AUTOINCREMENT,
                customer_id TEXT NOT NULL,
                request_date TEXT NOT NULL,
                plan TEXT,
                contract_type TEXT,
                tenure REAL,
                monthly_charge REAL,
                primary_reason TEXT NOT NULL,
                secondary_reason TEXT,
                customer_comments TEXT,
                stay_request TEXT,
                retention_action TEXT NOT NULL,
                outcome TEXT NOT NULL,
                cancellation_status TEXT NOT NULL DEFAULT 'Not cancelled',
                cancellation_date TEXT,
                owner TEXT,
                follow_up_date TEXT,
                notes TEXT,
                created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
                updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
                CHECK (cancellation_status != 'Cancelled' OR cancellation_date IS NOT NULL)
            )
            """
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_cases_request_date ON cancellation_cases(request_date)"
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_cases_customer_date ON cancellation_cases(customer_id, request_date)"
        )


def find_duplicate(customer_id: str, request_date: str, primary_reason: str) -> int | None:
    client = _supabase_client()
    if client is not None:
        try:
            response = (
                client.table("cancellation_cases")
                .select("case_id")
                .eq("customer_id", customer_id)
                .eq("request_date", request_date)
                .eq("primary_reason", primary_reason)
                .order("case_id", desc=True)
                .limit(1)
                .execute()
            )
            rows = response.data or []
        except Exception as exc:
            raise sqlite3.OperationalError("Could not check for a duplicate cancellation case.") from exc
        return int(rows[0]["case_id"]) if rows else None

    with _connection() as connection:
        row = connection.execute(
            """
            SELECT case_id FROM cancellation_cases
            WHERE customer_id = ? AND request_date = ? AND primary_reason = ?
            ORDER BY case_id DESC LIMIT 1
            """,
            (customer_id, request_date, primary_reason),
        ).fetchone()
    return int(row["case_id"]) if row else None


def create_case(case: dict[str, Any]) -> int:
    fields = (
        "customer_id",
        "request_date",
        "plan",
        "contract_type",
        "tenure",
        "monthly_charge",
        "primary_reason",
        "secondary_reason",
        "customer_comments",
        "stay_request",
        "retention_action",
        "outcome",
        "cancellation_status",
        "cancellation_date",
        "owner",
        "follow_up_date",
    )
    record = {field: case.get(field) for field in fields}
    client = _supabase_client()
    if client is not None:
        try:
            response = client.table("cancellation_cases").insert(record).execute()
            rows = response.data or []
        except Exception as exc:
            raise sqlite3.OperationalError("Could not save the cancellation case to remote storage.") from exc
        if not rows or rows[0].get("case_id") is None:
            raise sqlite3.DatabaseError("The case insert did not return a case ID.")
        return int(rows[0]["case_id"])

    values = [record[field] for field in fields]
    placeholders = ", ".join("?" for _ in fields)
    with _connection() as connection:
        cursor = connection.execute(
            f"INSERT INTO cancellation_cases ({', '.join(fields)}) VALUES ({placeholders})",
            values,
        )
        if cursor.lastrowid is None:
            raise sqlite3.DatabaseError("The case insert did not return a case ID.")
        return int(cursor.lastrowid)


def list_cases() -> pd.DataFrame:
    client = _supabase_client()
    if client is not None:
        try:
            response = (
                client.table("cancellation_cases")
                .select("*")
                .order("request_date", desc=True)
                .order("case_id", desc=True)
                .execute()
            )
            return pd.DataFrame(response.data or [])
        except Exception as exc:
            raise sqlite3.OperationalError("Could not load cancellation cases from remote storage.") from exc

    with _connection() as connection:
        return pd.read_sql_query(
            "SELECT * FROM cancellation_cases ORDER BY request_date DESC, case_id DESC",
            connection,
        )


def get_case(case_id: int) -> dict[str, Any] | None:
    client = _supabase_client()
    if client is not None:
        try:
            response = client.table("cancellation_cases").select("*").eq("case_id", case_id).limit(1).execute()
            rows = response.data or []
        except Exception as exc:
            raise sqlite3.OperationalError("Could not retrieve the selected cancellation case.") from exc
        return rows[0] if rows else None

    with _connection() as connection:
        row = connection.execute(
            "SELECT * FROM cancellation_cases WHERE case_id = ?", (case_id,)
        ).fetchone()
    return dict(row) if row else None


def update_case(case_id: int, updates: dict[str, Any]) -> None:
    allowed_fields = {
        "retention_action",
        "outcome",
        "cancellation_status",
        "cancellation_date",
        "owner",
        "follow_up_date",
        "notes",
    }
    if not updates or not set(updates).issubset(allowed_fields):
        raise ValueError("Only case management fields can be updated.")

    client = _supabase_client()
    if client is not None:
        from datetime import datetime, timezone
        payload = {**updates, "updated_at": datetime.now(timezone.utc).isoformat()}
        try:
            response = (
                client.table("cancellation_cases")
                .update(payload)
                .eq("case_id", case_id)
                .select("case_id")
                .execute()
            )
            if not response.data:
                raise ValueError(f"Cancellation case {case_id} was not found.")
        except ValueError:
            raise
        except Exception as exc:
            raise sqlite3.OperationalError("Could not update the cancellation case in remote storage.") from exc
        return

    assignments = ", ".join(f"{field} = ?" for field in updates)
    values = list(updates.values())
    values.append(case_id)
    with _connection() as connection:
        cursor = connection.execute(
            f"""
            UPDATE cancellation_cases
            SET {assignments}, updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
            WHERE case_id = ?
            """,
            values,
        )
        if cursor.rowcount == 0:
            raise ValueError(f"Cancellation case {case_id} was not found.")

