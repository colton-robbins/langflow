from __future__ import annotations

import json
from time import perf_counter
from typing import Any, AsyncIterator, cast

from langchain.agents import AgentExecutor
from langchain_core.messages import HumanMessage

from lfx.base.agents.callback import AgentAsyncHandler
from lfx.base.agents.events import (
    CHAIN_EVENT_HANDLERS,
    TOOL_EVENT_HANDLERS,
    ExceptionWithMessageError,
    OnTokenFunctionType,
    SendMessageFunctionType,
    ToolContent,
    handle_on_tool_end,
)
from lfx.components.langchain_utilities.tool_calling import ToolCallingAgentComponent
from lfx.components.models_and_agents.memory import MemoryComponent
from lfx.io import BoolInput, IntInput, MessageTextInput, Output
from lfx.log.logger import logger
from lfx.schema.message import Message


class ClickHouseAgentComponent(ToolCallingAgentComponent):
    display_name: str = "ClickHouse Agent"
    description: str = "Agent that intercepts ClickHouse tool results and passes them to the next agent on success."
    icon = "Database"
    name = "ClickHouseAgent"

    inputs = [
        *ToolCallingAgentComponent.inputs,
        BoolInput(
            name="pass_results_on_success",
            display_name="Pass Results on Success",
            info="If enabled, when a ClickHouse tool succeeds, the results are passed directly to the next agent instead of generating a response.",
            value=True,
        ),
        MessageTextInput(
            name="tool_names",
            display_name="Tool Names to Intercept",
            info="Comma-separated list of tool names to intercept (e.g., 'ClickHouseSQL,ClickHouse'). Defaults to intercepting any tool with 'clickhouse' in the name.",
            value="",
            advanced=True,
        ),
        MessageTextInput(
            name="context_id",
            display_name="Context ID",
            info="The context ID of the chat. Adds an extra layer to the local memory.",
            value="",
            advanced=True,
        ),
        IntInput(
            name="n_messages",
            display_name="Number of Chat History Messages",
            value=100,
            info="Number of chat history messages to retrieve.",
            advanced=True,
            show=True,
        ),
    ]

    outputs = [
        Output(display_name="Response", name="response", method="message_response"),
    ]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._tool_succeeded = False
        self._captured_result = None

    async def _send_message_noop(self, message: Message, *, skip_db_update: bool = False) -> Message:
        """Local no-op function for send_message_callback when not connected to ChatOutput."""
        return message

    async def get_memory_data(self):
        """Retrieve chat history messages from memory.
        
        Uses session_id and context_id from incoming message if available,
        otherwise uses component/graph values. This ensures the agent retrieves
        messages from the same conversation context as Chat Input.
        """
        # Use session_id and context_id from incoming message if available, otherwise use component/graph values
        incoming_session_id = None
        incoming_context_id = None
        if isinstance(self.input_value, Message):
            if hasattr(self.input_value, "session_id"):
                incoming_session_id = self.input_value.session_id
            if hasattr(self.input_value, "context_id"):
                incoming_context_id = self.input_value.context_id
        
        # Prefer incoming message's session_id/context_id, then component's, then graph's
        session_id = incoming_session_id or (self.graph.session_id if hasattr(self, "graph") else None) or ""
        context_id = incoming_context_id or getattr(self, "context_id", "") or ""
        n_messages = getattr(self, "n_messages", 100)
        
        messages = (
            await MemoryComponent(**self.get_base_args())
            .set(
                session_id=session_id,
                context_id=context_id,
                order="Ascending",
                n_messages=n_messages,
            )
            .retrieve_messages()
        )
        return [
            message for message in messages if getattr(message, "id", None) != getattr(self.input_value, "id", None)
        ]

    async def message_response(self) -> Message:
        """Execute the agent and return the response."""
        if not self.tools:
            msg = "No tools provided. Please connect at least one tool."
            raise ValueError(msg)

        # Retrieve chat history from memory if not already provided via input
        if not hasattr(self, "chat_history") or not self.chat_history:
            self.chat_history = await self.get_memory_data()
            await logger.adebug(f"Retrieved {len(self.chat_history)} chat history messages")
            if isinstance(self.chat_history, Message):
                self.chat_history = [self.chat_history]

        agent = self.create_agent_runnable()

        handle_parsing_errors = True
        verbose = hasattr(self, "verbose") and self.verbose
        max_iterations = hasattr(self, "max_iterations") and self.max_iterations

        runnable = AgentExecutor.from_agent_and_tools(
            agent=agent,
            tools=self.tools or [],
            handle_parsing_errors=handle_parsing_errors,
            verbose=verbose,
            max_iterations=max_iterations,
        )

        # Convert input_value to proper format
        lc_message = None
        if isinstance(self.input_value, Message):
            lc_message = self.input_value.to_lc_message()
        elif isinstance(self.input_value, str):
            lc_message = HumanMessage(content=self.input_value)
        else:
            msg = f"Invalid input type: {type(self.input_value)}"
            raise TypeError(msg)

        input_dict = {"input": lc_message}

        # Add system_prompt to input_dict
        if hasattr(self, "system_prompt") and self.system_prompt:
            input_dict["system_prompt"] = self.system_prompt

        if hasattr(self, "chat_history") and self.chat_history:
            messages = self._data_to_messages_skip_empty(self.chat_history)
            input_dict["chat_history"] = messages

        # Get session_id safely
        if hasattr(self, "graph"):
            session_id = self.graph.session_id
        elif hasattr(self, "_session_id"):
            session_id = self._session_id
        else:
            session_id = None

        agent_message = Message(
            text="",
            sender="Machine",
            sender_name="AI",
            session_id=session_id,
        )

        on_token_callback = None
        if self._event_manager:
            on_token_callback = cast("OnTokenFunctionType", self._event_manager.on_token)

        try:
            # Process agent events with tool interception
            # If a ClickHouse tool succeeds, this will return early with the results
            result = await self._process_agent_events_with_tool_interception(
                runnable.astream_events(
                    input_dict,
                    config={"callbacks": [AgentAsyncHandler(self.log), *self._get_shared_callbacks()]},
                    version="v2",
                ),
                agent_message,
                cast("SendMessageFunctionType", self._send_message_noop),
                on_token_callback,
            )

            # Return the result (either intercepted tool results or normal agent response)
            self.status = result
            return result

        except ExceptionWithMessageError as e:
            await logger.aerror(f"ExceptionWithMessageError: {e}")
            raise
        except Exception as e:
            await logger.aerror(f"Error: {e}")
            raise

    async def _process_agent_events_with_tool_interception(
        self,
        agent_executor: AsyncIterator[dict[str, Any]],
        agent_message: Message,
        send_message_callback: SendMessageFunctionType,
        send_token_callback: OnTokenFunctionType | None = None,
    ) -> Message:
        """Process agent events and intercept ClickHouse tool results."""
        initial_message_id = agent_message.id if hasattr(agent_message, "id") else None

        try:
            tool_blocks_map: dict[str, ToolContent] = {}
            had_streaming = False
            start_time = perf_counter()

            # Determine which tool names to intercept
            tool_names_to_intercept = []
            if self.tool_names:
                tool_names_to_intercept = [name.strip().lower() for name in self.tool_names.split(",") if name.strip()]
            else:
                # Default: intercept any tool with 'clickhouse' in the name
                tool_names_to_intercept = ["clickhouse"]

            async for event in agent_executor:
                # Intercept tool end events
                if event["event"] == "on_tool_end":
                    # Process tool end normally
                    agent_message, start_time = await handle_on_tool_end(
                        event, agent_message, tool_blocks_map, send_message_callback, start_time
                    )

                    # Check if this is a tool we should intercept
                    tool_output = event["data"].get("output", "")
                    tool_name = event.get("name", "").lower()

                    # Check if tool name matches our interception list
                    should_intercept = any(
                        intercept_name in tool_name for intercept_name in tool_names_to_intercept
                    ) or any(
                        tool_name in intercept_name for intercept_name in tool_names_to_intercept
                    )

                    if should_intercept and self.pass_results_on_success:
                        await logger.adebug(f"Intercepting tool output for: {tool_name}")

                        # Normalize tool output to a dict when possible
                        parsed: dict[str, Any] | None = None
                        success_flag = None
                        try:
                            if isinstance(tool_output, str):
                                parsed = json.loads(tool_output)
                            elif isinstance(tool_output, dict):
                                parsed = tool_output
                        except (json.JSONDecodeError, AttributeError):
                            parsed = None

                        if parsed is not None:
                            success_flag = parsed.get("success")
                        elif isinstance(tool_output, str):
                            # Not JSON, check string content
                            lower = tool_output.lower()
                            if "success" in lower and "true" in lower:
                                success_flag = True
                            elif "success" in lower and "false" in lower:
                                success_flag = False

                        # If tool succeeded, capture result and return immediately
                        if success_flag is True:
                            self._tool_succeeded = True
                            await logger.ainfo(f"Tool {tool_name} succeeded, capturing results")

                            # Build payload for downstream agent
                            if parsed is not None:
                                payload_obj = parsed
                                raw_csv = parsed.get("raw_csv")
                                if isinstance(raw_csv, str) and raw_csv.strip():
                                    payload_text = raw_csv
                                else:
                                    payload_text = json.dumps(parsed, separators=(",", ":"), ensure_ascii=False)
                            else:
                                payload_text = str(tool_output)
                                payload_obj = {"success": True, "output": payload_text}

                            # Include original user prompt for context
                            try:
                                user_q = (
                                    self.input_value.text
                                    if isinstance(self.input_value, Message) and hasattr(self.input_value, "text")
                                    else str(self.input_value)
                                )
                            except Exception:
                                user_q = ""

                            if user_q and isinstance(payload_obj, dict) and "user_question" not in payload_obj:
                                payload_obj = {**payload_obj, "user_question": user_q}

                            # Store captured result
                            self._captured_result = Message(text=payload_text, data=payload_obj)

                            # Return immediately to prevent agent from generating its own response
                            # The downstream agent will receive this result
                            return self._captured_result

                # Process other tool events
                elif event["event"] in TOOL_EVENT_HANDLERS:
                    tool_handler = TOOL_EVENT_HANDLERS[event["event"]]
                    agent_message, start_time = await tool_handler(
                        event, agent_message, tool_blocks_map, send_message_callback, start_time
                    )

                # Process chain events
                elif event["event"] in CHAIN_EVENT_HANDLERS:
                    chain_handler = CHAIN_EVENT_HANDLERS[event["event"]]

                    if event["event"] in ("on_chain_stream", "on_chat_model_stream"):
                        had_streaming = True
                        agent_message, start_time = await chain_handler(
                            event,
                            agent_message,
                            send_message_callback,
                            send_token_callback,
                            start_time,
                            had_streaming=had_streaming,
                            message_id=initial_message_id,
                        )
                    else:
                        agent_message, start_time = await chain_handler(
                            event, agent_message, send_message_callback, None, start_time, had_streaming=had_streaming
                        )

            agent_message.properties.state = "complete"
            agent_message = await send_message_callback(message=agent_message)

        except Exception as e:
            raise ExceptionWithMessageError(agent_message, str(e)) from e

        return agent_message
