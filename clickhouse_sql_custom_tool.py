from __future__ import annotations

import csv
import io
import os
import re
from typing import Any

import httpx
from langflow.custom.custom_component.component import Component
from langflow.io import BoolInput, IntInput, MultilineInput, Output, SecretStrInput, StrInput
from langflow.schema.data import Data

_ENV_PREFIX = "DEV_"


def _truthy_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _int_env(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value.strip())
    except ValueError:
        return default


def _normalize_base_url(host: str) -> str:
    host = (host or "").strip()
    if not host:
        return ""
    if host.startswith(("http://", "https://")):
        return host.rstrip("/")
    return f"https://{host}:8443"


def _strip_trailing_semicolons(sql: str) -> str:
    return (sql or "").strip().rstrip(";").strip()


def _secret_to_str(value: Any) -> str:
    if value is None:
        return ""
    if hasattr(value, "get_secret_value"):
        return str(value.get_secret_value() or "")
    return str(value)


_FORBIDDEN_SQL = re.compile(
    r"\b("
    r"insert|update|delete|drop|alter|create|truncate|attach|detach|optimize|system|grant|revoke|kill"
    r")\b",
    flags=re.IGNORECASE,
)


def _validate_single_statement_select_only(sql: str) -> str | None:
    normalized = _strip_trailing_semicolons(sql)
    if not normalized:
        return "SQL query is empty."
    if ";" in normalized:
        return "Only a single SQL statement is allowed."
    if _FORBIDDEN_SQL.search(normalized):
        return "Only SELECT queries are allowed (select_only=true)."
    first_token = normalized.lstrip().split(None, 1)[0].lower()
    if first_token not in {"select", "with", "show", "describe", "explain"}:
        return "Only read-only queries are allowed (SELECT/WITH/SHOW/DESCRIBE/EXPLAIN)."
    return None


class ClickHouseSQLTool(Component):
    display_name = "ClickHouse SQL"
    description = "Execute a SQL query against ClickHouse over HTTPS and return CSVWithNames results."
    icon = "Database"
    name = "ClickHouseSQL"

    inputs = [
        MultilineInput(
            name="query",
            display_name="SQL Query",
            info="SQL query text to execute.",
            required=True,
            tool_mode=True,
        ),
        StrInput(
            name="host",
            display_name="Host Override",
            info="Optional. If blank, resolves from env vars (DEV_CH_HOST).",
            advanced=True,
        ),
        StrInput(
            name="user",
            display_name="User Override",
            info="Optional. If blank, resolves from env vars (DEV_CH_USER).",
            advanced=True,
        ),
        SecretStrInput(
            name="password",
            display_name="Password Override",
            info="Optional. If blank, resolves from env vars (DEV_CH_PASSWORD).",
            advanced=True,
        ),
        StrInput(
            name="database",
            display_name="Database Override",
            info="Optional. If blank, resolves from env vars (DATABASE or DEV_CH_DATABASE).",
            advanced=True,
        ),
        IntInput(
            name="timeout_seconds",
            display_name="Timeout (seconds)",
            info="Request timeout in seconds. If not set, uses QUERY_TIMEOUT_SECONDS or 60.",
            value=60,
            advanced=True,
        ),
        BoolInput(
            name="verify_ssl",
            display_name="Verify SSL",
            info="Whether to verify TLS certificates. If not set, uses CH_VERIFY_SSL or false.",
            value=False,
            advanced=True,
        ),
        IntInput(
            name="max_rows_to_parse",
            display_name="Max Rows To Parse",
            info="Parse at most N rows into structured output. Raw CSV is always returned.",
            value=1000,
            advanced=True,
        ),
        BoolInput(
            name="select_only",
            display_name="SELECT-only (recommended)",
            info="Reject non read-only SQL and multi-statement queries.",
            value=True,
            advanced=True,
        ),
    ]

    outputs = [
        Output(display_name="Result", name="result", method="run_clickhouse_query"),
    ]

    def _resolve_config(self) -> tuple[dict[str, Any] | None, str | None]:
        prefix = _ENV_PREFIX

        host = (self.host or "").strip() or os.getenv(f"{prefix}CH_HOST") or os.getenv("CH_HOST") or ""
        user = (self.user or "").strip() or os.getenv(f"{prefix}CH_USER") or os.getenv("CH_USER") or ""
        password = (
            _secret_to_str(self.password).strip() or os.getenv(f"{prefix}CH_PASSWORD") or os.getenv("CH_PASSWORD") or ""
        )
        database = (
            (self.database or "").strip()
            or os.getenv("DATABASE")
            or os.getenv(f"{prefix}CH_DATABASE")
            or os.getenv("CH_DATABASE")
            or ""
        )

        timeout_seconds = int(self.timeout_seconds or 0) or _int_env("QUERY_TIMEOUT_SECONDS", 60)

        verify_ssl = bool(self.verify_ssl)
        if os.getenv("CH_VERIFY_SSL") is not None:
            verify_ssl = _truthy_env("CH_VERIFY_SSL", verify_ssl)

        base_url = _normalize_base_url(host)

        missing = []
        if not base_url:
            missing.append("CH_HOST (or ENV-specific *_CH_HOST) / host input")
        if not user:
            missing.append("CH_USER (or ENV-specific *_CH_USER) / user input")
        if not password:
            missing.append("CH_PASSWORD (or ENV-specific *_CH_PASSWORD) / password input")
        if not database:
            missing.append("DATABASE (or legacy CH_DATABASE / *_CH_DATABASE) / database input")
        if missing:
            return None, "Missing required configuration: " + ", ".join(missing)

        return (
            {
                "base_url": base_url,
                "user": user,
                "password": password,
                "database": database,
                "timeout_seconds": timeout_seconds,
                "verify_ssl": verify_ssl,
                "headers": {"X-ClickHouse-Database": database, "X-ClickHouse-Format": "CSVWithNames"},
            },
            None,
        )

    async def run_clickhouse_query(self) -> Data:
        if self.select_only:
            err = _validate_single_statement_select_only(self.query)
            if err:
                self.status = f"Rejected query: {err}"
                return Data(data={"success": False, "error": err, "raw_csv": ""})

        cfg, cfg_err = self._resolve_config()
        if cfg_err or cfg is None:
            self.status = cfg_err or "Missing configuration"
            return Data(data={"success": False, "error": self.status, "raw_csv": ""})

        sql_to_run = _strip_trailing_semicolons(self.query)
        try:
            async with httpx.AsyncClient(timeout=cfg["timeout_seconds"], verify=cfg["verify_ssl"]) as client:
                response = await client.get(
                    cfg["base_url"],
                    params={"query": sql_to_run},
                    headers=cfg["headers"],
                    auth=(cfg["user"], cfg["password"]),
                )
                response.raise_for_status()

            raw_csv = response.text or ""
            if not raw_csv.strip():
                msg = "Empty response from ClickHouse."
                self.status = msg
                return Data(data={"success": False, "error": msg, "raw_csv": ""})

            csv_reader = csv.reader(io.StringIO(raw_csv))
            try:
                columns = next(csv_reader)
            except StopIteration:
                columns = []

            max_rows = int(self.max_rows_to_parse or 0)
            rows: list[dict[str, str]] = []
            for i, row in enumerate(csv_reader):
                if max_rows and i >= max_rows:
                    break
                rows.append({col: (row[idx] if idx < len(row) else "") for idx, col in enumerate(columns)})

            self.status = f"ClickHouse query ok ({len(rows)} row(s) parsed)"
            return Data(
                data={
                    "success": True,
                    "raw_csv": raw_csv,
                    "columns": columns,
                    "rows": rows,
                    "row_count": len(rows),
                    "database": cfg["database"],
                }
            )
        except httpx.TimeoutException as e:
            msg = f"Timeout contacting ClickHouse: {e}"
            self.status = msg
            return Data(data={"success": False, "error": msg, "raw_csv": ""})
        except httpx.HTTPStatusError as e:
            msg = f"HTTP error from ClickHouse: {e.response.status_code} {e.response.text}"
            self.status = msg
            return Data(data={"success": False, "error": msg, "raw_csv": e.response.text or ""})
        except Exception as e:
            msg = f"Error executing ClickHouse query: {str(e)}"
            self.status = msg
            return Data(data={"success": False, "error": msg, "raw_csv": ""})

