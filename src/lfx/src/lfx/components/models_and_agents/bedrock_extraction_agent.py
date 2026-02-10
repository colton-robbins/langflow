from __future__ import annotations

from lfx.base.agents.agent import LCToolsAgentComponent
from lfx.components.models_and_agents.agent import AgentComponent
from lfx.io import (
    BoolInput,
    HandleInput,
    IntInput,
    MessageTextInput,
    MultilineInput,
    Output,
    TableInput,
)
from lfx.schema.table import EditMode


class BedrockExtractionAgentComponent(AgentComponent):
    """Agent component specialized for multi-modal information extraction using AWS Bedrock.
    
    This agent is optimized for extracting structured information from text and images.
    Connect an AWS Bedrock Converse component (e.g., Sonnet 4.5) to the model input.
    It supports multi-modal inputs and is optimized for information extraction tasks.
    """
    display_name: str = "Bedrock Extraction Agent"
    description: str = (
        "Multi-modal agent for extracting structured information from text and images. "
        "Connect an AWS Bedrock model component (e.g., Sonnet 4.5) for best results."
    )
    documentation: str = "https://docs.langflow.org/agents"
    icon = "Amazon"
    beta = False
    name = "BedrockExtractionAgent"

    inputs = [
        HandleInput(
            name="model",
            display_name="Language Model",
            info="Connect an AWS Bedrock Converse component (e.g., Sonnet 4.5) for multi-modal extraction",
            input_types=["LanguageModel"],
            required=True,
        ),
        MultilineInput(
            name="system_prompt",
            display_name="Agent Instructions",
            info="System Prompt: Instructions for information extraction. Default focuses on structured extraction.",
            value=(
                "You are an expert information extraction agent. Your task is to extract structured information "
                "from text and images. When processing multi-modal inputs (text and images), carefully analyze "
                "both modalities to extract all relevant information. Return extracted information in a clear, "
                "structured format. If asked to extract specific fields, ensure all requested information is "
                "captured accurately. For images, describe what you see and extract any visible text, data, "
                "or structured information."
            ),
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
                "You are an AI that extracts structured JSON objects from unstructured text and images. "
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
        *LCToolsAgentComponent.get_base_inputs(),  # Get base inputs (tools, input_value, handle_parsing_errors, verbose, max_iterations, agent_description)
        BoolInput(
            name="stream",
            display_name="Stream",
            info="Stream the response from the agent. When enabled, the agent will stream tokens as they are generated.",
            advanced=True,
            value=False,
        ),
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
