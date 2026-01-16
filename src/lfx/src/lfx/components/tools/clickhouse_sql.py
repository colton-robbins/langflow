from __future__ import annotations

import csv
import io
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
from dotenv import find_dotenv, load_dotenv
from langchain.tools import StructuredTool
from langchain_core.tools import ToolException
from pydantic import BaseModel, Field

from lfx.base.langchain_utilities.model import LCToolComponent
from lfx.field_typing import Tool
from lfx.inputs.inputs import BoolInput, DropdownInput, IntInput, MultilineInput, SecretStrInput, StrInput
from lfx.log.logger import logger
from lfx.schema.data import Data

_ENV_PREFIX: dict[str, str] = {"prod": "PROD_", "stage": "STAGE_", "dev": "DEV_"}


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


_FORBIDDEN_SQL = re.compile(
    r"\b("
    r"insert|update|delete|drop|alter|create|truncate|attach|detach|optimize|system|grant|revoke|kill"
    r")\b",
    flags=re.IGNORECASE,
)


def _validate_single_statement_select_only(sql: str) -> str | None:
    """Return an error string if invalid, else None."""
    normalized = _strip_trailing_semicolons(sql)
    if not normalized:
        return "SQL query is empty."

    # Reject multi-statement SQL. (Allow a single trailing ';' only, which we stripped above.)
    if ";" in normalized:
        return "Only a single SQL statement is allowed."

    if _FORBIDDEN_SQL.search(normalized):
        return "Only SELECT queries are allowed (select_only=true)."

    first_token = normalized.lstrip().split(None, 1)[0].lower()
    if first_token not in {"select", "with", "show", "describe", "explain"}:
        return "Only read-only queries are allowed (SELECT/WITH/SHOW/DESCRIBE/EXPLAIN)."

    return None


@dataclass(frozen=True)
class _ClickHouseResolvedConfig:
    base_url: str
    user: str
    password: str
    database: str
    timeout_seconds: int
    verify_ssl: bool

    @property
    def headers(self) -> dict[str, str]:
        return {
            "X-ClickHouse-Database": self.database,
            "X-ClickHouse-Format": "CSVWithNames",
        }


class ClickHouseSQLToolComponent(LCToolComponent):
    display_name = "ClickHouse SQL"
    description = "Execute a SQL query against ClickHouse over HTTPS and return CSVWithNames results."
    icon = "Database"
    name = "ClickHouseSQLTool"

    inputs = [
        StrInput(
            name="name",
            display_name="Tool Name",
            info="The name exposed to an agent when using this as a tool.",
            value="clickhouse_sql",
        ),
        StrInput(
            name="description",
            display_name="Tool Description",
            info="The description exposed to an agent when using this as a tool.",
            value="Execute a SQL query against ClickHouse and return results (CSVWithNames).",
        ),
        MultilineInput(
            name="query",
            display_name="SQL Query",
            info="SQL query text to execute.",
            required=True,
            tool_mode=True,
        ),
        DropdownInput(
            name="environment",
            display_name="Environment",
            info="Which environment-specific credentials to use when reading env vars.",
            options=["prod", "stage", "dev"],
            value="prod",
            advanced=True,
        ),
        StrInput(
            name="host",
            display_name="Host",
            info="Optional override. If blank, resolves from env vars (e.g. PROD_CH_HOST).",
            advanced=True,
        ),
        StrInput(
            name="user",
            display_name="User",
            info="Optional override. If blank, resolves from env vars (e.g. PROD_CH_USER).",
            advanced=True,
        ),
        SecretStrInput(
            name="password",
            display_name="Password",
            info="Optional override. If blank, resolves from env vars (e.g. PROD_CH_PASSWORD).",
            advanced=True,
        ),
        StrInput(
            name="database",
            display_name="Database",
            info="Optional override. If blank, resolves from env vars (DATABASE or *CH_DATABASE).",
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

    class ClickHouseSQLSchema(BaseModel):
        query: str = Field(..., description="SQL query text to execute.")

    def _load_env(self) -> None:
        """Best-effort .env loading (matches the minimal runner behavior)."""
        try:
            dotenv_path = find_dotenv(usecwd=True)
            if not dotenv_path:
                candidate = Path(__file__).resolve().parents[3] / ".env"
                if candidate.exists():
                    dotenv_path = str(candidate)
            load_dotenv(dotenv_path=dotenv_path or None)
        except Exception:  # noqa: BLE001
            # Don't fail component execution due to dotenv issues.
            return

    def _resolve_config(self) -> tuple[_ClickHouseResolvedConfig | None, str | None]:
        self._load_env()

        env_normalized = (getattr(self, "environment", "") or "prod").strip().lower()
        if env_normalized not in _ENV_PREFIX:
            env_normalized = "prod"
        prefix = _ENV_PREFIX[env_normalized]

        host = (getattr(self, "host", None) or "").strip() or os.getenv(f"{prefix}CH_HOST") or os.getenv("CH_HOST") or ""
        user = (getattr(self, "user", None) or "").strip() or os.getenv(f"{prefix}CH_USER") or os.getenv("CH_USER") or ""

        password_raw: Any = getattr(self, "password", None)
        password = (str(password_raw).strip() if password_raw is not None else "") or os.getenv(f"{prefix}CH_PASSWORD") or os.getenv("CH_PASSWORD") or ""

        database = (
            (getattr(self, "database", None) or "").strip()
            or os.getenv("DATABASE")
            or os.getenv(f"{prefix}CH_DATABASE")
            or os.getenv("CH_DATABASE")
            or ""
        )

        timeout_seconds = int(getattr(self, "timeout_seconds", None) or 0) or _int_env("QUERY_TIMEOUT_SECONDS", 60)
        verify_ssl = bool(getattr(self, "verify_ssl", False))
        if os.getenv("CH_VERIFY_SSL") is not None:
            # If the env var is present, it takes precedence over the default.
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
            _ClickHouseResolvedConfig(
                base_url=base_url,
                user=user,
                password=password,
                database=database,
                timeout_seconds=timeout_seconds,
                verify_ssl=verify_ssl,
            ),
            None,
        )

    def _execute_query(self, sql: str) -> Data:
        if getattr(self, "select_only", True):
            err = _validate_single_statement_select_only(sql)
            if err:
                self.status = f"Rejected query: {err}"
                return Data(data={"success": False, "error": err, "raw_csv": ""}, text=f"Error: {err}")

        cfg, cfg_err = self._resolve_config()
        if cfg_err or cfg is None:
            self.status = cfg_err or "Missing configuration"
            return Data(data={"success": False, "error": self.status, "raw_csv": ""}, text=f"Error: {self.status}")

        sql_to_run = _strip_trailing_semicolons(sql)
        try:
            with httpx.Client(timeout=cfg.timeout_seconds, verify=cfg.verify_ssl) as client:
                response = client.get(
                    cfg.base_url,
                    params={"query": sql_to_run},
                    headers=cfg.headers,
                    auth=(cfg.user, cfg.password),
                )
                response.raise_for_status()

            raw_csv = response.text or ""
            if not raw_csv.strip():
                msg = "Empty response from ClickHouse."
                self.status = msg
                return Data(data={"success": False, "error": msg, "raw_csv": ""}, text=f"Error: {msg}")

            csv_reader = csv.reader(io.StringIO(raw_csv))
            try:
                columns = next(csv_reader)
            except StopIteration:
                columns = []

            max_rows = int(getattr(self, "max_rows_to_parse", 1000) or 0)
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
                },
                text=raw_csv,
            )
        except httpx.TimeoutException as e:
            msg = f"Timeout contacting ClickHouse: {e}"
            logger.debug(msg, exc_info=True)
            self.status = msg
            return Data(data={"success": False, "error": msg, "raw_csv": ""}, text=f"Error: {msg}")
        except httpx.HTTPStatusError as e:
            msg = f"HTTP error from ClickHouse: {e.response.status_code} {e.response.text}"
            logger.debug(msg, exc_info=True)
            self.status = msg
            return Data(data={"success": False, "error": msg, "raw_csv": e.response.text or ""}, text=f"Error: {msg}")
        except Exception as e:  # noqa: BLE001
            msg = f"Error executing ClickHouse query: {e!s}"
            logger.debug(msg, exc_info=True)
            self.status = msg
            return Data(data={"success": False, "error": msg, "raw_csv": ""}, text=f"Error: {msg}")

    def build_tool(self) -> Tool:
        def run_clickhouse_sql(query: str) -> dict[str, Any]:
            result = self._execute_query(query)
            if not result.data.get("success"):
                raise ToolException(result.data.get("error") or "ClickHouse SQL tool error")
            return result.data

        tool = StructuredTool.from_function(
            name=self.name,
            description=self.description,
            func=run_clickhouse_sql,
            args_schema=self.ClickHouseSQLSchema,
        )
        self.status = "ClickHouse SQL Tool created"
        return tool

    def run_model(self) -> list[Data]:
        # Direct component execution (non-agent path).
        result = self._execute_query(self.query)
        return [result]

