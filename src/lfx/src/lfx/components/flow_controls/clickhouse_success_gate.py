from __future__ import annotations

from typing import Any

from lfx.custom.custom_component.component import Component
from lfx.io import DataInput, IntInput, MessageTextInput, MultilineInput, Output
from lfx.schema.data import Data
from lfx.schema.message import Message


class ClickHouseSuccessGateComponent(Component):
    """Gate execution on ClickHouse tool success.

    This component is designed to make your flow deterministic:
    - It reads the ClickHouse tool's `Data.data["success"]` boolean.
    - If success=True: emits the query result payload to the next step.
    - If success=False: emits a retry prompt (up to max_retries).
    - If attempts exceed max_retries: emits a fixed fallback message.

    This avoids relying on an agent's tool-calling loop and guarantees your Answer Agent
    only runs when ClickHouse returned success=True.
    """

    display_name = "ClickHouse Success Gate"
    description = "Routes based on ClickHouse tool `success` field, with retry + fallback."
    icon = "split"
    name = "ClickHouseSuccessGate"

    inputs = [
        DataInput(
            name="clickhouse_result",
            display_name="ClickHouse Result",
            info="Connect the ClickHouse SQL component output here (Data with success/raw_csv/error).",
            required=True,
        ),
        MessageTextInput(
            name="user_question",
            display_name="User Question",
            info="Original user question (used to build retry prompts).",
            required=False,
        ),
        MessageTextInput(
            name="last_sql",
            display_name="Last SQL",
            info="The SQL query that was executed (optional but recommended for better retries).",
            required=False,
            advanced=True,
        ),
        IntInput(
            name="max_retries",
            display_name="Max Retries",
            info="Maximum number of retries when ClickHouse returns success=false.",
            value=3,
            advanced=True,
        ),
        MultilineInput(
            name="too_complex_message",
            display_name="Too Complex Message",
            info="Returned when max retries are exhausted.",
            value="I apologize, but your question is too complex for me to answer with the available data. Please try rephrasing your question or breaking it down into simpler parts.",
            advanced=True,
        ),
    ]

    outputs = [
        Output(display_name="Success", name="success", method="success_output", group_outputs=True),
        Output(display_name="Retry", name="retry", method="retry_output", group_outputs=True),
        Output(display_name="Give Up", name="give_up", method="give_up_output", group_outputs=True),
    ]

    def _attempts_key(self) -> str:
        return f"{self._id}_attempts"

    def _get_attempts(self) -> int:
        return int(self.ctx.get(self._attempts_key(), 0) or 0)

    def _set_attempts(self, value: int) -> None:
        self.update_ctx({self._attempts_key(): int(value)})

    def _read_success_and_payload(self) -> tuple[bool | None, dict[str, Any]]:
        """Returns (success_flag, payload_dict)."""
        payload: dict[str, Any] = {}
        if isinstance(self.clickhouse_result, Data):
            payload = self.clickhouse_result.data or {}
        elif isinstance(self.clickhouse_result, dict):
            payload = self.clickhouse_result
        return payload.get("success"), payload

    def success_output(self) -> Message:
        success_flag, payload = self._read_success_and_payload()

        if success_flag is True:
            # Reset attempts on success
            self._set_attempts(0)

            # Stop other branches
            self.stop("retry")
            self.stop("give_up")
            self.graph.exclude_branch_conditionally(self._id, output_name="retry")
            self.graph.exclude_branch_conditionally(self._id, output_name="give_up")

            # Prefer raw_csv as text payload (answer agent can summarize it)
            raw_csv = payload.get("raw_csv")
            if isinstance(raw_csv, str) and raw_csv.strip():
                out_text = raw_csv
            else:
                out_text = str(payload)

            self.status = f"ClickHouse success (attempts reset)"
            return Message(text=out_text)

        # Not success -> stop this branch
        self.stop("success")
        return Message(text="")

    def retry_output(self) -> Message:
        success_flag, payload = self._read_success_and_payload()

        if success_flag is False:
            attempts = self._get_attempts() + 1
            self._set_attempts(attempts)

            if attempts <= int(self.max_retries or 0):
                # Stop other branches
                self.stop("success")
                self.stop("give_up")
                self.graph.exclude_branch_conditionally(self._id, output_name="success")
                self.graph.exclude_branch_conditionally(self._id, output_name="give_up")

                error = payload.get("error") or payload.get("message") or "Unknown error"
                question = (self.user_question or "").strip()
                last_sql = (self.last_sql or "").strip()

                prompt_parts = [
                    "The previous ClickHouse SQL execution failed.",
                    f"Attempt {attempts}/{int(self.max_retries or 0)}.",
                    f"Error: {error}",
                ]
                if question:
                    prompt_parts.append(f"User question: {question}")
                if last_sql:
                    prompt_parts.append(f"Last SQL:\n{last_sql}")
                prompt_parts.append("Generate a corrected ClickHouse SELECT query only (no explanation).")

                retry_prompt = "\n\n".join(prompt_parts)
                self.status = f"Retrying (attempt {attempts}/{int(self.max_retries or 0)})"
                return Message(text=retry_prompt)

            # If attempts exceeded max retries, stop this branch (give_up will fire)
            self.stop("retry")
            return Message(text="")

        # Not failed -> stop this branch
        self.stop("retry")
        return Message(text="")

    def give_up_output(self) -> Message:
        success_flag, _payload = self._read_success_and_payload()
        attempts = self._get_attempts()

        # Fire only when we have a failure and we've exhausted attempts
        if success_flag is False and attempts >= int(self.max_retries or 0):
            # Stop other branches
            self.stop("success")
            self.stop("retry")
            self.graph.exclude_branch_conditionally(self._id, output_name="success")
            self.graph.exclude_branch_conditionally(self._id, output_name="retry")

            self.status = f"Max retries reached ({attempts})"
            # Reset attempts so a new user request starts fresh
            self._set_attempts(0)
            return Message(text=str(self.too_complex_message or ""))

        self.stop("give_up")
        return Message(text="")

