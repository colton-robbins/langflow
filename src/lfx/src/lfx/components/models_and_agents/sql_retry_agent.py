import json
import re
from typing import Any, AsyncIterator, cast
from time import perf_counter

from langchain.agents import AgentExecutor, BaseMultiActionAgent, BaseSingleActionAgent
from langchain_core.messages import HumanMessage
from langchain_core.runnables import Runnable
from langchain_core.tools import StructuredTool

from lfx.base.agents.agent import LCToolsAgentComponent
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
from lfx.base.models.model_input_constants import (
    ALL_PROVIDER_FIELDS,
    MODEL_DYNAMIC_UPDATE_FIELDS,
    MODEL_PROVIDERS_DICT,
    MODEL_PROVIDERS_LIST,
    MODELS_METADATA,
)
from lfx.base.models.model_utils import get_model_name
from lfx.components.helpers import CurrentDateComponent
from lfx.components.langchain_utilities.tool_calling import ToolCallingAgentComponent
from lfx.components.models_and_agents.memory import MemoryComponent
from lfx.custom.custom_component.component import get_component_toolkit
from lfx.custom.utils import update_component_build_config
from lfx.helpers.base_model import build_model_from_schema
from lfx.inputs.inputs import BoolInput, SecretStrInput, StrInput
from lfx.io import DropdownInput, IntInput, MessageTextInput, MultilineInput, Output, TableInput
from lfx.log.logger import logger
from lfx.memory import delete_message
from lfx.schema.data import Data
from lfx.schema.dotdict import dotdict
from lfx.schema.message import Message
from lfx.schema.table import EditMode


def set_advanced_true(component_input):
    component_input.advanced = True
    return component_input


class SQLRetryAgentComponent(ToolCallingAgentComponent):
    display_name: str = "SQL Retry Agent"
    description: str = "SQL agent with automatic retry logic based on tool success/failure."
    documentation: str = "https://docs.langflow.org/agents"
    icon = "database"
    beta = False
    name = "SQLRetryAgent"

    memory_inputs = [set_advanced_true(component_input) for component_input in MemoryComponent().inputs]

    # Filter out json_mode from OpenAI inputs
    if "OpenAI" in MODEL_PROVIDERS_DICT:
        openai_inputs_filtered = [
            input_field
            for input_field in MODEL_PROVIDERS_DICT["OpenAI"]["inputs"]
            if not (hasattr(input_field, "name") and input_field.name == "json_mode")
        ]
    else:
        openai_inputs_filtered = []

    inputs = [
        DropdownInput(
            name="agent_llm",
            display_name="Model Provider",
            info="The provider of the language model that the agent will use to generate responses.",
            options=[*MODEL_PROVIDERS_LIST],
            value="OpenAI",
            real_time_refresh=True,
            refresh_button=False,
            input_types=[],
            options_metadata=[MODELS_METADATA[key] for key in MODEL_PROVIDERS_LIST if key in MODELS_METADATA],
            external_options={
                "fields": {
                    "data": {
                        "node": {
                            "name": "connect_other_models",
                            "display_name": "Connect other models",
                            "icon": "CornerDownLeft",
                        }
                    }
                },
            },
        ),
        SecretStrInput(
            name="api_key",
            display_name="API Key",
            info="The API key to use for the model.",
            required=True,
        ),
        StrInput(
            name="base_url",
            display_name="Base URL",
            info="The base URL of the API.",
            required=True,
            show=False,
        ),
        StrInput(
            name="project_id",
            display_name="Project ID",
            info="The project ID of the model.",
            required=True,
            show=False,
        ),
        IntInput(
            name="max_output_tokens",
            display_name="Max Output Tokens",
            info="The maximum number of tokens to generate.",
            show=False,
        ),
        *openai_inputs_filtered,
        MultilineInput(
            name="system_prompt",
            display_name="Agent Instructions",
            info="System Prompt: Initial instructions and context provided to guide the agent's behavior.",
            value="You are a SQL expert that generates and executes SQL queries. When a query fails, analyze the error and try again with corrections.",
            advanced=False,
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
        *LCToolsAgentComponent.get_base_inputs(),
        BoolInput(
            name="add_current_date_tool",
            display_name="Current Date",
            advanced=True,
            info="If true, will add a tool to the agent that returns the current date.",
            value=True,
        ),
        IntInput(
            name="max_retries",
            display_name="Max Retries",
            info="Maximum number of retries when SQL execution fails.",
            value=3,
            advanced=True,
        ),
        MultilineInput(
            name="max_retries_message",
            display_name="Max Retries Message",
            info="Message to display when max retries is reached.",
            value="I apologize, but your question is too complex for me to answer with the available data. Please try rephrasing your question or breaking it down into simpler parts.",
            advanced=True,
        ),
    ]

    outputs = [
        Output(name="response", display_name="Response", method="message_response", group_outputs=True),
        Output(name="max_retries_reached", display_name="Max Retries", method="handle_max_retries", group_outputs=True),
    ]

    def _pre_run_setup(self):
        """Reset per-run state so outputs behave deterministically across runs."""
        for attr in (
            "_success_result",
            "_max_retries_result",
            "_agent_result",
            "_retry_count",
            "_tool_failed",
            "_last_error",
            "_captured_tool_payload",
        ):
            if hasattr(self, attr):
                delattr(self, attr)

    async def get_agent_requirements(self):
        """Get the agent requirements for the agent."""
        llm_model, display_name = await self.get_llm()
        if llm_model is None:
            msg = "No language model selected. Please choose a model to proceed."
            raise ValueError(msg)
        self.model_name = get_model_name(llm_model, display_name=display_name)

        # Get memory data
        self.chat_history = await self.get_memory_data()
        await logger.adebug(f"Retrieved {len(self.chat_history)} chat history messages")
        if isinstance(self.chat_history, Message):
            self.chat_history = [self.chat_history]

        # Add current date tool if enabled
        if self.add_current_date_tool:
            if not isinstance(self.tools, list):
                self.tools = []
            current_date_tool = (await CurrentDateComponent(**self.get_base_args()).to_toolkit()).pop(0)

            if not isinstance(current_date_tool, StructuredTool):
                msg = "CurrentDateComponent must be converted to a StructuredTool"
                raise TypeError(msg)
            self.tools.append(current_date_tool)

        # Set shared callbacks for tracing the tools used by the agent
        self.set_tools_callbacks(self.tools, self._get_shared_callbacks())

        return llm_model, self.chat_history, self.tools

    async def message_response(self) -> Message:
        """Main entry point with retry logic.

        On success, this returns a clean payload message built from the ClickHouse tool output
        (CSV/JSON), suitable for piping into a downstream \"answer\" agent.
        """
        # If already executed, return stored result
        if hasattr(self, "_success_result"):
            return self._success_result
        
        # If max retries reached, stop this output
        if hasattr(self, "_max_retries_result"):
            self.stop("response")
            return Message(text="")
        
        try:
            llm_model, self.chat_history, self.tools = await self.get_agent_requirements()
            
            # Initialize retry counter
            self._retry_count = 0
            self._tool_failed = False
            self._last_error = None
            
            # Set up and run agent with retry logic
            self.set(
                llm=llm_model,
                tools=self.tools or [],
                chat_history=self.chat_history,
                input_value=self.input_value,
                system_prompt=self.system_prompt,
            )
            
            agent = self.create_agent_runnable()
            result = await self.run_agent_with_retry(agent)

            # Store result
            self._agent_result = result
            
            # Check if we hit max retries during execution
            if hasattr(self, "_max_retries_result"):
                # Max retries was hit, stop this output
                self.stop("response")
                self.graph.exclude_branch_conditionally(self._id, "response")
                return Message(text="")
            
            # Success - store, stop max_retries output, and return
            self._success_result = result
            self.stop("max_retries_reached")
            self.graph.exclude_branch_conditionally(self._id, "max_retries_reached")
            return result

        except (ValueError, TypeError, KeyError) as e:
            await logger.aerror(f"{type(e).__name__}: {e!s}")
            raise
        except ExceptionWithMessageError as e:
            await logger.aerror(f"ExceptionWithMessageError occurred: {e}")
            raise
        except Exception as e:
            await logger.aerror(f"Unexpected error: {e!s}")
            raise

    async def run_agent_with_retry(
        self,
        agent: Runnable | BaseSingleActionAgent | BaseMultiActionAgent | AgentExecutor,
    ) -> Message:
        """Run agent with automatic retry on tool failure."""

        async def _send_message_noop_local(
            message: Message,
            id_: str | None = None,  # noqa: ARG001
            *,
            skip_db_update: bool = False,  # noqa: ARG001
        ) -> Message:
            """No-op send_message callback.

            Note: This is defined inside the method so it still exists when the user
            pastes this class into a Langflow \"Custom Component\" code block.
            """
            return message
        
        if isinstance(agent, AgentExecutor):
            runnable = agent
        else:
            handle_parsing_errors = hasattr(self, "handle_parsing_errors") and self.handle_parsing_errors
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
        
        agent_message = await Message.create(
            text="",
            sender="Machine",
            sender_name="AI",
            session_id=self.graph.session_id,
        )
        
        on_token_callback = None
        if self._event_manager:
            on_token_callback = cast("OnTokenFunctionType", self._event_manager.on_token)
        
        # RETRY LOOP
        while self._retry_count <= self.max_retries:
            try:
                # Process agent events with tool interception
                result = await self._process_agent_events_with_tool_check(
                    runnable.astream_events(
                        input_dict,
                        config={"callbacks": [AgentAsyncHandler(self.log), *self._get_shared_callbacks()]},
                        version="v2",
                    ),
                    agent_message,
                    cast("SendMessageFunctionType", _send_message_noop_local),
                    on_token_callback,
                )
                
                # Check if tool failed
                if self._tool_failed:
                    self._retry_count += 1
                    
                    if self._retry_count <= self.max_retries:
                        # Retry - add error context to input
                        error_context = f"\n\nPrevious attempt failed with error: {self._last_error}\nPlease correct the SQL query and try again. Attempt {self._retry_count}/{self.max_retries}."
                        
                        # Update input with error context
                        if isinstance(lc_message, HumanMessage):
                            lc_message = HumanMessage(content=lc_message.content + error_context)
                            input_dict["input"] = lc_message
                        
                        # Reset failure flag
                        self._tool_failed = False
                        
                        # Log retry
                        await logger.ainfo(f"Retrying SQL query (attempt {self._retry_count}/{self.max_retries})")
                        
                        # Create new agent message for retry
                        agent_message = await Message.create(
                            text="",
                            sender="Machine",
                            sender_name="AI",
                            session_id=self.graph.session_id,
                        )
                        
                        # Continue loop for retry
                        continue
                    else:
                        # Max retries reached - route to max_retries output
                        await logger.awarning(f"Max retries ({self.max_retries}) reached for SQL query")
                        
                        # Stop normal response output
                        self.stop("response")
                        self.graph.exclude_branch_conditionally(self._id, "response")
                        
                        # Store result for max_retries handler
                        self._max_retries_result = Message(text=self.max_retries_message)
                        
                        return self._max_retries_result
                
                # Success - return result
                self.status = result
                return result
                
            except ExceptionWithMessageError as e:
                if hasattr(e, "agent_message") and hasattr(e.agent_message, "id"):
                    msg_id = e.agent_message.id
                    await delete_message(id_=msg_id)
                await self._send_message_event(e.agent_message, category="remove_message")
                await logger.aerror(f"ExceptionWithMessageError: {e}")
                raise
            except Exception as e:
                await logger.aerror(f"Error: {e}")
                raise
        
        # Should not reach here, but safety fallback
        return Message(text=self.max_retries_message)

    async def _process_agent_events_with_tool_check(
        self,
        agent_executor: AsyncIterator[dict[str, Any]],
        agent_message: Message,
        send_message_callback: "SendMessageFunctionType",
        send_token_callback: "OnTokenFunctionType | None" = None,
    ) -> Message:
        """Process agent events and check tool results for success/failure."""
        
        initial_message_id = agent_message.id if hasattr(agent_message, "id") else None
        
        try:
            tool_blocks_map: dict[str, ToolContent] = {}
            had_streaming = False
            start_time = perf_counter()
            
            async for event in agent_executor:
                # Intercept tool end events
                if event["event"] == "on_tool_end":
                    # Process tool end normally
                    agent_message, start_time = await handle_on_tool_end(
                        event, agent_message, tool_blocks_map, send_message_callback, start_time
                    )
                    
                    # CHECK TOOL OUTPUT FOR SUCCESS FLAG
                    tool_output = event["data"].get("output", "")
                    tool_name = event.get("name", "")
                    
                    # Check if this is a ClickHouse or SQL tool
                    if "clickhouse" in tool_name.lower() or "sql" in tool_name.lower():
                        await logger.adebug(f"Checking tool output: {tool_output}")
                        
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
                            if success_flag is False:
                                self._last_error = parsed.get("error", "Unknown error")
                        elif isinstance(tool_output, str):
                            # Not JSON, check string content
                            lower = tool_output.lower()
                            if "success" in lower and "true" in lower:
                                success_flag = True
                            elif "success" in lower and "false" in lower:
                                success_flag = False
                                self._last_error = tool_output
                        
                        # Set failure flag if tool failed
                        if success_flag is False:
                            self._tool_failed = True
                            await logger.awarning(f"Tool failed: {self._last_error}")
                            # Don't continue processing - will retry
                            break
                        elif success_flag is True:
                            self._tool_failed = False
                            await logger.ainfo("Tool succeeded")

                            # Build a clean payload for downstream answer agents.
                            # Prefer raw_csv when present; otherwise, pass compact JSON.
                            if parsed is None:
                                payload_text = str(tool_output)
                                payload_obj: dict[str, Any] = {"success": True, "output": payload_text}
                            else:
                                payload_obj = parsed
                                raw_csv = parsed.get("raw_csv")
                                if isinstance(raw_csv, str) and raw_csv.strip():
                                    payload_text = raw_csv
                                else:
                                    payload_text = json.dumps(parsed, separators=(",", ":"), ensure_ascii=False)

                            # Include original user prompt for the answer agent (helps grounding).
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

                            self._captured_tool_payload = payload_obj
                            # Return immediately after successful execution to prevent the SQL agent from
                            # generating its own narrative response. The downstream answer agent will respond.
                            return Message(text=payload_text)
                
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
        
        return await Message.create(**agent_message.model_dump())

    def handle_max_retries(self) -> Message:
        """Return the max retries message - only fires when max retries reached."""
        # If success happened, this output should return empty (it's stopped)
        if hasattr(self, "_success_result"):
            return Message(text="")
        
        # Only return message if max retries was actually reached
        if hasattr(self, "_max_retries_result"):
            return self._max_retries_result
        
        # Shouldn't reach here, but safety fallback
        return Message(text="")

    async def get_memory_data(self):
        messages = (
            await MemoryComponent(**self.get_base_args())
            .set(
                session_id=self.graph.session_id,
                context_id=self.context_id,
                order="Ascending",
                n_messages=self.n_messages,
            )
            .retrieve_messages()
        )
        return [
            message for message in messages if getattr(message, "id", None) != getattr(self.input_value, "id", None)
        ]

    async def get_llm(self):
        if not isinstance(self.agent_llm, str):
            return self.agent_llm, None

        try:
            provider_info = MODEL_PROVIDERS_DICT.get(self.agent_llm)
            if not provider_info:
                msg = f"Invalid model provider: {self.agent_llm}"
                raise ValueError(msg)

            component_class = provider_info.get("component_class")
            display_name = component_class.display_name
            inputs = provider_info.get("inputs")
            prefix = provider_info.get("prefix", "")

            return self._build_llm_model(component_class, inputs, prefix), display_name

        except (AttributeError, ValueError, TypeError, RuntimeError) as e:
            await logger.aerror(f"Error building {self.agent_llm} language model: {e!s}")
            msg = f"Failed to initialize language model: {e!s}"
            raise ValueError(msg) from e

    def _build_llm_model(self, component, inputs, prefix=""):
        model_kwargs = {}
        for input_ in inputs:
            if hasattr(self, f"{prefix}{input_.name}"):
                model_kwargs[input_.name] = getattr(self, f"{prefix}{input_.name}")
        return component.set(**model_kwargs).build_model()

    def set_component_params(self, component):
        provider_info = MODEL_PROVIDERS_DICT.get(self.agent_llm)
        if provider_info:
            inputs = provider_info.get("inputs")
            prefix = provider_info.get("prefix")
            model_kwargs = {}
            for input_ in inputs:
                if hasattr(self, f"{prefix}{input_.name}"):
                    model_kwargs[input_.name] = getattr(self, f"{prefix}{input_.name}")
            return component.set(**model_kwargs)
        return component

    def delete_fields(self, build_config: dotdict, fields: dict | list[str]) -> None:
        """Delete specified fields from build_config."""
        for field in fields:
            if build_config is not None and field in build_config:
                build_config.pop(field, None)

    def update_input_types(self, build_config: dotdict) -> dotdict:
        """Update input types for all fields in build_config."""
        for key, value in build_config.items():
            if isinstance(value, dict):
                if value.get("input_types") is None:
                    build_config[key]["input_types"] = []
            elif hasattr(value, "input_types") and value.input_types is None:
                value.input_types = []
        return build_config

    async def update_build_config(
        self, build_config: dotdict, field_value: str, field_name: str | None = None
    ) -> dotdict:
        if field_name in ("agent_llm",):
            build_config["agent_llm"]["value"] = field_value
            provider_info = MODEL_PROVIDERS_DICT.get(field_value)
            if provider_info:
                component_class = provider_info.get("component_class")
                if component_class and hasattr(component_class, "update_build_config"):
                    build_config = await update_component_build_config(
                        component_class, build_config, field_value, "model_name"
                    )

            provider_configs: dict[str, tuple[dict, list[dict]]] = {
                provider: (
                    MODEL_PROVIDERS_DICT[provider]["fields"],
                    [
                        MODEL_PROVIDERS_DICT[other_provider]["fields"]
                        for other_provider in MODEL_PROVIDERS_DICT
                        if other_provider != provider
                    ],
                )
                for provider in MODEL_PROVIDERS_DICT
            }
            if field_value in provider_configs:
                fields_to_add, fields_to_delete = provider_configs[field_value]
                for fields in fields_to_delete:
                    self.delete_fields(build_config, fields)
                if field_value == "OpenAI" and not any(field in build_config for field in fields_to_add):
                    build_config.update(fields_to_add)
                else:
                    build_config.update(fields_to_add)
                build_config["agent_llm"]["input_types"] = []
                build_config["agent_llm"]["display_name"] = "Model Provider"
            elif field_value == "connect_other_models":
                self.delete_fields(build_config, ALL_PROVIDER_FIELDS)
                custom_component = DropdownInput(
                    name="agent_llm",
                    display_name="Language Model",
                    info="The provider of the language model that the agent will use to generate responses.",
                    options=[*MODEL_PROVIDERS_LIST],
                    real_time_refresh=True,
                    refresh_button=False,
                    input_types=["LanguageModel"],
                    placeholder="Awaiting model input.",
                    options_metadata=[MODELS_METADATA[key] for key in MODEL_PROVIDERS_LIST if key in MODELS_METADATA],
                    external_options={
                        "fields": {
                            "data": {
                                "node": {
                                    "name": "connect_other_models",
                                    "display_name": "Connect other models",
                                    "icon": "CornerDownLeft",
                                },
                            }
                        },
                    },
                )
                build_config.update({"agent_llm": custom_component.to_dict()})
            build_config = self.update_input_types(build_config)

            default_keys = [
                "code",
                "_type",
                "agent_llm",
                "tools",
                "input_value",
                "add_current_date_tool",
                "system_prompt",
                "agent_description",
                "max_iterations",
                "handle_parsing_errors",
                "verbose",
            ]
            missing_keys = [key for key in default_keys if key not in build_config]
            if missing_keys:
                msg = f"Missing required keys in build_config: {missing_keys}"
                raise ValueError(msg)
        if (
            isinstance(self.agent_llm, str)
            and self.agent_llm in MODEL_PROVIDERS_DICT
            and field_name in MODEL_DYNAMIC_UPDATE_FIELDS
        ):
            provider_info = MODEL_PROVIDERS_DICT.get(self.agent_llm)
            if provider_info:
                component_class = provider_info.get("component_class")
                component_class = self.set_component_params(component_class)
                prefix = provider_info.get("prefix")
                if component_class and hasattr(component_class, "update_build_config"):
                    if isinstance(field_name, str) and isinstance(prefix, str):
                        field_name_without_prefix = field_name.replace(prefix, "")
                    else:
                        field_name_without_prefix = field_name
                    build_config = await update_component_build_config(
                        component_class, build_config, field_value, field_name_without_prefix
                    )
        return dotdict({k: v.to_dict() if hasattr(v, "to_dict") else v for k, v in build_config.items()})

    async def _get_tools(self) -> list:
        component_toolkit = get_component_toolkit()
        tools_names = self._build_tools_names()
        agent_description = self.get_tool_description()
        description = f"{agent_description}{tools_names}"

        tools = component_toolkit(component=self).get_tools(
            tool_name="Call_SQL_Retry_Agent",
            tool_description=description,
            callbacks=self.get_langchain_callbacks(),
        )
        if hasattr(self, "tools_metadata"):
            tools = component_toolkit(component=self, metadata=self.tools_metadata).update_tools_metadata(tools=tools)

        return tools
