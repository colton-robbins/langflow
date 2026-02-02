from __future__ import annotations

import json
import re
from collections.abc import Sequence
from typing import TYPE_CHECKING

from pydantic import ValidationError

from lfx.components.models_and_agents.memory import MemoryComponent

if TYPE_CHECKING:
    from langchain_core.tools import Tool

from lfx.base.agents.agent import LCToolsAgentComponent
from lfx.base.agents.events import ExceptionWithMessageError
from lfx.components.helpers import CurrentDateComponent
from lfx.components.langchain_utilities.tool_calling import ToolCallingAgentComponent
from lfx.custom.custom_component.component import get_component_toolkit
from lfx.helpers.base_model import build_model_from_schema
from lfx.io import BoolInput, HandleInput, IntInput, MessageTextInput, MultilineInput, Output, SecretStrInput, TableInput
from lfx.log.logger import logger
from lfx.schema.data import Data
from lfx.schema.dotdict import dotdict
from lfx.schema.message import Message
from lfx.schema.table import EditMode


def sanitize_tool_name(name: str) -> str:
    """Sanitize tool name to match AWS Bedrock pattern: ^[a-zA-Z0-9_-]+$
    
    Replaces invalid characters (like $, spaces, etc.) with underscores.
    
    Args:
        name: Original tool name
        
    Returns:
        Sanitized tool name that matches the required pattern
    """
    if not name:
        return "unnamed_tool"
    
    # Replace any non-alphanumeric characters (except _ and -) with underscores
    sanitized = re.sub(r"[^a-zA-Z0-9_-]", "_", str(name))
    
    # Ensure it starts with a letter or underscore (not a number or special char)
    if sanitized and not sanitized[0].isalpha() and sanitized[0] != "_":
        sanitized = f"tool_{sanitized}"
    
    # Remove consecutive underscores
    sanitized = re.sub(r"_+", "_", sanitized)
    
    # Remove leading/trailing underscores
    sanitized = sanitized.strip("_")
    
    return sanitized or "unnamed_tool"


def deduplicate_tools(tools):
    """Deduplicate tools by name to prevent duplicate tool errors in model bindings.
    
    Some models (like AWS Bedrock ConverseStream) throw validation errors when
    duplicate tool names are present in the tool configuration. This function
    ensures each tool name appears only once, keeping the first occurrence.
    Also sanitizes tool names to ensure they match AWS Bedrock requirements.
    
    Args:
        tools: Sequence of tools that may contain duplicates
        
    Returns:
        List of deduplicated tools with sanitized names, preserving order of first occurrence
    """
    if not tools:
        return []
    
    seen_names = set()
    deduplicated = []
    
    for tool in tools:
        # Get tool name - tools can have 'name' attribute or be callable
        tool_name = None
        if hasattr(tool, "name"):
            tool_name = tool.name
        elif hasattr(tool, "__name__"):
            tool_name = tool.__name__
        elif callable(tool):
            tool_name = getattr(tool, "__name__", str(tool))
        else:
            tool_name = str(tool)
        
        # Sanitize the tool name to ensure it matches AWS Bedrock requirements
        sanitized_name = sanitize_tool_name(tool_name)
        
        # Only add if we haven't seen this sanitized name before
        if sanitized_name not in seen_names:
            seen_names.add(sanitized_name)
            # Update the tool's name to the sanitized version
            if hasattr(tool, "name"):
                tool.name = sanitized_name
            deduplicated.append(tool)
    
    return deduplicated


def set_advanced_true(component_input):
    component_input.advanced = True
    return component_input


class AgentComponent(ToolCallingAgentComponent):
    display_name: str = "Agent"
    description: str = "Define the agent's instructions, then enter a task to complete using tools."
    documentation: str = "https://docs.langflow.org/agents"
    icon = "bot"
    beta = False
    name = "Agent"

    memory_inputs = [set_advanced_true(component_input) for component_input in MemoryComponent().inputs]

    inputs = [
        HandleInput(
            name="model",
            display_name="Language Model",
            info="Connect a Language Model component",
            input_types=["LanguageModel"],
            required=True,
        ),
        SecretStrInput(
            name="api_key",
            display_name="API Key",
            info="Model Provider API key",
            real_time_refresh=True,
            advanced=True,
        ),
        MultilineInput(
            name="system_prompt",
            display_name="Agent Instructions",
            info="System Prompt: Initial instructions and context provided to guide the agent's behavior.",
            value="You are a helpful assistant that can use tools to answer questions and perform tasks.",
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
        MultilineInput(
            name="format_instructions",
            display_name="Output Format Instructions",
            info="Generic Template for structured output formatting. Valid only with Structured response.",
            value=(
                "You are an AI that extracts structured JSON objects from unstructured text. "
                "Use a predefined schema with expected types (str, int, float, bool, dict). "
                "Extract ALL relevant instances that match the schema - if multiple patterns exist, capture them all. "
                "Fill missing or ambiguous values with defaults: null for missing values. "
                "Remove exact duplicates but keep variations that have different field values. "
                "Always return valid JSON in the expected format, never throw errors. "
                "If multiple objects can be extracted, return them all in the structured format."
            ),
            advanced=True,
        ),
        TableInput(
            name="output_schema",
            display_name="Output Schema",
            info=(
                "Schema Validation: Define the structure and data types for structured output. "
                "No validation if no output schema."
            ),
            advanced=True,
            required=False,
            value=[],
            table_schema=[
                {
                    "name": "name",
                    "display_name": "Name",
                    "type": "str",
                    "description": "Specify the name of the output field.",
                    "default": "field",
                    "edit_mode": EditMode.INLINE,
                },
                {
                    "name": "description",
                    "display_name": "Description",
                    "type": "str",
                    "description": "Describe the purpose of the output field.",
                    "default": "description of field",
                    "edit_mode": EditMode.POPOVER,
                },
                {
                    "name": "type",
                    "display_name": "Type",
                    "type": "str",
                    "edit_mode": EditMode.INLINE,
                    "description": ("Indicate the data type of the output field (e.g., str, int, float, bool, dict)."),
                    "options": ["str", "int", "float", "bool", "dict"],
                    "default": "str",
                },
                {
                    "name": "multiple",
                    "display_name": "As List",
                    "type": "boolean",
                    "description": "Set to True if this output field should be a list of the specified type.",
                    "default": "False",
                    "edit_mode": EditMode.INLINE,
                },
            ],
        ),
        *LCToolsAgentComponent.get_base_inputs(),
        # removed memory inputs from agent component
        # *memory_inputs,
        BoolInput(
            name="add_current_date_tool",
            display_name="Current Date",
            advanced=True,
            info="If true, will add a tool to the agent that returns the current date.",
            value=True,
        ),
    ]
    outputs = [
        Output(name="response", display_name="Response", method="message_response"),
    ]

    def _get_tool_name_mapping(self) -> dict[str, str]:
        """Get a mapping of old tool names to new tool names.
        
        Override this method in a subclass to rename tools. Return a dictionary
        where keys are the original tool names and values are the new names.
        
        This will rename ALL tools with the same old name to the same new name.
        If you have multiple tools with the same method name, they will all be
        renamed to the same new name.
        
        Example:
            return {
                "SEARCH_DOCUMENTS": "scaffold_search",
                "AS_DATAFRAME": "scaffold_get_dataframe"
            }
        
        Returns:
            dict[str, str]: Mapping of old_name -> new_name
        """
        return {}

    def _extract_component_info_from_tool(self, tool: Tool) -> dict | None:
        """Extract component information from a tool.
        
        Attempts to extract component attributes (like persist_directory) from the
        tool's function closure or metadata. This is useful for Chroma DB tools
        where we want to use the persist_directory in the tool name.
        
        Args:
            tool: The tool object
            
        Returns:
            dict | None: Dictionary with component info (e.g., {'persist_directory': '...'})
                        or None if info cannot be extracted
        """
        # First, check if component info is stored in metadata
        if hasattr(tool, "metadata") and tool.metadata:
            component_info = tool.metadata.get("component_info")
            if component_info:
                return component_info
        
        # Try to extract from function closure
        # The tool's func/coroutine wraps the component method
        func = getattr(tool, "func", None) or getattr(tool, "coroutine", None)
        if func:
            try:
                # Try to access closure variables
                if hasattr(func, "__closure__") and func.__closure__:
                    for cell in func.__closure__:
                        if cell.cell_contents:
                            obj = cell.cell_contents
                            # Check if it's a Component instance
                            from lfx.custom.custom_component.component import Component
                            if isinstance(obj, Component):
                                info = {}
                                # Extract persist_directory if it exists
                                if hasattr(obj, "persist_directory"):
                                    persist_dir = getattr(obj, "persist_directory", None)
                                    if persist_dir:
                                        info["persist_directory"] = persist_dir
                                # Extract collection_name if it exists
                                if hasattr(obj, "collection_name"):
                                    collection = getattr(obj, "collection_name", None)
                                    if collection:
                                        info["collection_name"] = collection
                                if info:
                                    return info
            except Exception:
                # If extraction fails, continue without component info
                pass
        
        return None

    def _get_tool_name_for_index(self, tool: Tool, index: int, original_name: str) -> str | None:
        """Get a specific name for a tool based on its position/index.
        
        Override this method to provide custom names for each tool individually.
        This method is called for each tool in order, allowing you to set specific
        names based on the tool's position in the list.
        
        By default, this method will try to use persist_directory from Chroma DB
        components if available.
        
        Args:
            tool: The tool object being renamed
            index: Zero-based index of the tool in the tools list
            original_name: The original name of the tool
            
        Returns:
            str | None: The new name for this tool, or None to use default renaming
            
        Example:
            def _get_tool_name_for_index(self, tool, index, original_name):
                # Use persist_directory if available
                component_info = self._extract_component_info_from_tool(tool)
                if component_info and "persist_directory" in component_info:
                    persist_dir = component_info["persist_directory"]
                    # Extract just the directory name or use full path
                    dir_name = persist_dir.split("/")[-1] if persist_dir else None
                    if dir_name:
                        return f"search_{dir_name}"
                return None
        """
        # Default implementation: try to use persist_directory for Chroma tools
        component_info = self._extract_component_info_from_tool(tool)
        if component_info and "persist_directory" in component_info:
            persist_dir = component_info["persist_directory"]
            if persist_dir:
                from lfx.base.tools.component_tool import _format_tool_name
                # Extract directory name from path
                import os
                dir_name = os.path.basename(persist_dir.rstrip("/\\"))
                if dir_name:
                    # Create name based on original tool name and directory
                    base_name = original_name.lower().replace("_", "")
                    new_name = f"{base_name}_{dir_name}"
                    return _format_tool_name(new_name)
        
        return None

    def _should_make_tool_names_unique(self) -> bool:
        """Whether to add unique suffixes when multiple tools have the same name.
        
        Override this method to return True if you want tools with the same
        old name to be renamed with unique suffixes (e.g., scaffold_search_1,
        scaffold_search_2, etc.). Default is False, which means all tools
        with the same old name will be renamed to the same new name.
        
        Returns:
            bool: True to add unique suffixes, False to use the same name for all
        """
        return False

    def _rename_tools(self, tools: list[Tool]) -> list[Tool]:
        """Rename tools based on the name mapping or index-based naming.
        
        This method renames tools by updating their name, tags, and metadata
        attributes. The renaming happens before tools are bound to the agent.
        
        The renaming follows this priority:
        1. If _get_tool_name_for_index() returns a name, use that
        2. Otherwise, use _get_tool_name_mapping() with optional unique suffixes
        
        Args:
            tools: List of tools to potentially rename
            
        Returns:
            List of tools with renamed tools updated
        """
        from lfx.base.tools.component_tool import _format_tool_name
        
        name_mapping = self._get_tool_name_mapping()
        make_unique = self._should_make_tool_names_unique()
        
        # Track counts for each new name if we need unique names
        new_name_counts: dict[str, int] = {}
        
        renamed_tools = []
        for index, tool in enumerate(tools):
            original_name = tool.name
            new_name = None
            
            # First, try index-based naming (allows specific names for each tool)
            custom_name = self._get_tool_name_for_index(tool, index, original_name)
            if custom_name is not None:
                new_name = custom_name
            # Otherwise, use mapping-based naming
            elif name_mapping and original_name in name_mapping:
                base_new_name = name_mapping[original_name]
                
                # Add suffix if we need unique names
                if make_unique:
                    count = new_name_counts.get(base_new_name, 0) + 1
                    new_name_counts[base_new_name] = count
                    # First occurrence keeps original name, subsequent ones get suffix
                    new_name = base_new_name if count == 1 else f"{base_new_name}_{count}"
                else:
                    new_name = base_new_name
            
            # Apply the new name if we have one
            if new_name:
                formatted_new_name = _format_tool_name(new_name)
                
                # Update tool name
                tool.name = formatted_new_name
                
                # Update tags (first tag is typically the tool name)
                if tool.tags:
                    tool.tags = [formatted_new_name] + tool.tags[1:]
                else:
                    tool.tags = [formatted_new_name]
                
                # Update metadata if present
                if hasattr(tool, "metadata") and tool.metadata:
                    tool.metadata["display_name"] = formatted_new_name
                
                logger.debug(f"Renamed tool '{original_name}' to '{formatted_new_name}' (index {index})")
            
            renamed_tools.append(tool)
        
        return renamed_tools

    async def get_agent_requirements(self):
        """Get the agent requirements for the agent."""
        from langchain_core.tools import StructuredTool

        # With HandleInput, model is already a BaseLanguageModel instance
        llm_model = self.model
        if llm_model is None:
            msg = "No language model connected. Please connect a Language Model component to the model input."
            raise ValueError(msg)

        # Get memory data
        self.chat_history = await self.get_memory_data()
        await logger.adebug(f"Retrieved {len(self.chat_history)} chat history messages")
        if isinstance(self.chat_history, Message):
            self.chat_history = [self.chat_history]

        # Add current date tool if enabled
        if self.add_current_date_tool:
            if not isinstance(self.tools, list):  # type: ignore[has-type]
                self.tools = []
            current_date_tool = (await CurrentDateComponent(**self.get_base_args()).to_toolkit()).pop(0)

            if not isinstance(current_date_tool, StructuredTool):
                msg = "CurrentDateComponent must be converted to a StructuredTool"
                raise TypeError(msg)
            self.tools.append(current_date_tool)

        # Deduplicate tools before binding to prevent validation errors
        # (e.g., AWS Bedrock ConverseStream throws ValidationException for duplicate tool names)
        if self.tools:
            self.tools = deduplicate_tools(self.tools)

        # Rename tools if custom renaming is configured
        self.tools = self._rename_tools(self.tools)

        # Final sanitization pass to ensure all tool names are valid for AWS Bedrock
        # This catches any tools that might have invalid names after renaming
        if self.tools:
            for tool in self.tools:
                if hasattr(tool, "name") and tool.name:
                    sanitized_name = sanitize_tool_name(tool.name)
                    if sanitized_name != tool.name:
                        await logger.awarning(
                            f"Tool name '{tool.name}' was sanitized to '{sanitized_name}' "
                            "to match AWS Bedrock requirements (pattern: ^[a-zA-Z0-9_-]+$)"
                        )
                        tool.name = sanitized_name
                        # Update tags if they match the old name
                        if hasattr(tool, "tags") and tool.tags:
                            tool.tags = [sanitized_name if tag == tool.name else tag for tag in tool.tags]

        # Set shared callbacks for tracing the tools used by the agent
        self.set_tools_callbacks(self.tools, self._get_shared_callbacks())

        return llm_model, self.chat_history, self.tools

    async def message_response(self) -> Message:
        try:
            llm_model, self.chat_history, self.tools = await self.get_agent_requirements()
            # Set up and run agent
            self.set(
                llm=llm_model,
                tools=self.tools or [],
                chat_history=self.chat_history,
                input_value=self.input_value,
                system_prompt=self.system_prompt,
            )
            agent = self.create_agent_runnable()
            result = await self.run_agent(agent)

            # Store result for potential JSON output
            self._agent_result = result

        except (ValueError, TypeError, KeyError) as e:
            await logger.aerror(f"{type(e).__name__}: {e!s}")
            raise
        except ExceptionWithMessageError as e:
            await logger.aerror(f"ExceptionWithMessageError occurred: {e}")
            raise
        # Avoid catching blind Exception; let truly unexpected exceptions propagate
        except Exception as e:
            await logger.aerror(f"Unexpected error: {e!s}")
            raise
        else:
            return result

    def _preprocess_schema(self, schema):
        """Preprocess schema to ensure correct data types for build_model_from_schema."""
        processed_schema = []
        for field in schema:
            processed_field = {
                "name": str(field.get("name", "field")),
                "type": str(field.get("type", "str")),
                "description": str(field.get("description", "")),
                "multiple": field.get("multiple", False),
            }
            # Ensure multiple is handled correctly
            if isinstance(processed_field["multiple"], str):
                processed_field["multiple"] = processed_field["multiple"].lower() in [
                    "true",
                    "1",
                    "t",
                    "y",
                    "yes",
                ]
            processed_schema.append(processed_field)
        return processed_schema

    async def build_structured_output_base(self, content: str):
        """Build structured output with optional BaseModel validation."""
        json_pattern = r"\{.*\}"
        schema_error_msg = "Try setting an output schema"

        # Try to parse content as JSON first
        json_data = None
        try:
            json_data = json.loads(content)
        except json.JSONDecodeError:
            json_match = re.search(json_pattern, content, re.DOTALL)
            if json_match:
                try:
                    json_data = json.loads(json_match.group())
                except json.JSONDecodeError:
                    return {"content": content, "error": schema_error_msg}
            else:
                return {"content": content, "error": schema_error_msg}

        # If no output schema provided, return parsed JSON without validation
        if not hasattr(self, "output_schema") or not self.output_schema or len(self.output_schema) == 0:
            return json_data

        # Use BaseModel validation with schema
        try:
            processed_schema = self._preprocess_schema(self.output_schema)
            output_model = build_model_from_schema(processed_schema)

            # Validate against the schema
            if isinstance(json_data, list):
                # Multiple objects
                validated_objects = []
                for item in json_data:
                    try:
                        validated_obj = output_model.model_validate(item)
                        validated_objects.append(validated_obj.model_dump())
                    except ValidationError as e:
                        await logger.aerror(f"Validation error for item: {e}")
                        # Include invalid items with error info
                        validated_objects.append({"data": item, "validation_error": str(e)})
                return validated_objects

            # Single object
            try:
                validated_obj = output_model.model_validate(json_data)
                return [validated_obj.model_dump()]  # Return as list for consistency
            except ValidationError as e:
                await logger.aerror(f"Validation error: {e}")
                return [{"data": json_data, "validation_error": str(e)}]

        except (TypeError, ValueError) as e:
            await logger.aerror(f"Error building structured output: {e}")
            # Fallback to parsed JSON without validation
            return json_data

    async def json_response(self) -> Data:
        """Convert agent response to structured JSON Data output with schema validation."""
        # Always use structured chat agent for JSON response mode for better JSON formatting
        try:
            system_components = []

            # 1. Agent Instructions (system_prompt)
            agent_instructions = getattr(self, "system_prompt", "") or ""
            if agent_instructions:
                system_components.append(f"{agent_instructions}")

            # 2. Format Instructions
            format_instructions = getattr(self, "format_instructions", "") or ""
            if format_instructions:
                system_components.append(f"Format instructions: {format_instructions}")

            # 3. Schema Information from BaseModel
            if hasattr(self, "output_schema") and self.output_schema and len(self.output_schema) > 0:
                try:
                    processed_schema = self._preprocess_schema(self.output_schema)
                    output_model = build_model_from_schema(processed_schema)
                    schema_dict = output_model.model_json_schema()
                    schema_info = (
                        "You are given some text that may include format instructions, "
                        "explanations, or other content alongside a JSON schema.\n\n"
                        "Your task:\n"
                        "- Extract only the JSON schema.\n"
                        "- Return it as valid JSON.\n"
                        "- Do not include format instructions, explanations, or extra text.\n\n"
                        "Input:\n"
                        f"{json.dumps(schema_dict, indent=2)}\n\n"
                        "Output (only JSON schema):"
                    )
                    system_components.append(schema_info)
                except (ValidationError, ValueError, TypeError, KeyError) as e:
                    await logger.aerror(f"Could not build schema for prompt: {e}", exc_info=True)

            # Combine all components
            combined_instructions = "\n\n".join(system_components) if system_components else ""
            llm_model, self.chat_history, self.tools = await self.get_agent_requirements()
            self.set(
                llm=llm_model,
                tools=self.tools or [],
                chat_history=self.chat_history,
                input_value=self.input_value,
                system_prompt=combined_instructions,
            )

            # Create and run structured chat agent
            try:
                structured_agent = self.create_agent_runnable()
            except (NotImplementedError, ValueError, TypeError) as e:
                await logger.aerror(f"Error with structured chat agent: {e}")
                raise
            try:
                result = await self.run_agent(structured_agent)
            except (
                ExceptionWithMessageError,
                ValueError,
                TypeError,
                RuntimeError,
            ) as e:
                await logger.aerror(f"Error with structured agent result: {e}")
                raise
            # Extract content from structured agent result
            if hasattr(result, "content"):
                content = result.content
            elif hasattr(result, "text"):
                content = result.text
            else:
                content = str(result)

        except (
            ExceptionWithMessageError,
            ValueError,
            TypeError,
            NotImplementedError,
            AttributeError,
        ) as e:
            await logger.aerror(f"Error with structured chat agent: {e}")
            # Fallback to regular agent
            content_str = "No content returned from agent"
            return Data(data={"content": content_str, "error": str(e)})

        # Process with structured output validation
        try:
            structured_output = await self.build_structured_output_base(content)

            # Handle different output formats
            if isinstance(structured_output, list) and structured_output:
                if len(structured_output) == 1:
                    return Data(data=structured_output[0])
                return Data(data={"results": structured_output})
            if isinstance(structured_output, dict):
                return Data(data=structured_output)
            return Data(data={"content": content})

        except (ValueError, TypeError) as e:
            await logger.aerror(f"Error in structured output processing: {e}")
            return Data(data={"content": content, "error": str(e)})

    async def get_memory_data(self):
        # TODO: This is a temporary fix to avoid message duplication. We should develop a function for this.
        # Use session_id and context_id from incoming message if available, otherwise use component/graph values
        # This ensures the agent retrieves messages from the same conversation context as Chat Input
        incoming_session_id = None
        incoming_context_id = None
        if isinstance(self.input_value, Message):
            if hasattr(self.input_value, "session_id"):
                incoming_session_id = self.input_value.session_id
            if hasattr(self.input_value, "context_id"):
                incoming_context_id = self.input_value.context_id
        
        # Prefer incoming message's session_id/context_id, then component's, then graph's
        session_id = incoming_session_id or self.graph.session_id or ""
        context_id = incoming_context_id or self.context_id or ""
        
        messages = (
            await MemoryComponent(**self.get_base_args())
            .set(
                session_id=session_id,
                context_id=context_id,
                order="Ascending",
                n_messages=self.n_messages,
            )
            .retrieve_messages()
        )
        return [
            message for message in messages if getattr(message, "id", None) != getattr(self.input_value, "id", None)
        ]

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
        self,
        build_config: dotdict,
        field_value: list[dict],
        field_name: str | None = None,
    ) -> dotdict:
        # With HandleInput, no dynamic model options needed
        # Model is connected via handle from Language Model component
        
        # Update input types for all fields
        build_config = self.update_input_types(build_config)

        return dotdict({k: v.to_dict() if hasattr(v, "to_dict") else v for k, v in build_config.items()})

    async def _get_tools(self) -> list[Tool]:
        component_toolkit = get_component_toolkit()
        tools_names = self._build_tools_names()
        agent_description = self.get_tool_description()
        # TODO: Agent Description Depreciated Feature to be removed
        description = f"{agent_description}{tools_names}"

        tools = component_toolkit(component=self).get_tools(
            tool_name="Call_Agent",
            tool_description=description,
            # here we do not use the shared callbacks as we are exposing the agent as a tool
            callbacks=self.get_langchain_callbacks(),
        )
        if hasattr(self, "tools_metadata"):
            tools = component_toolkit(component=self, metadata=self.tools_metadata).update_tools_metadata(tools=tools)

        return tools
