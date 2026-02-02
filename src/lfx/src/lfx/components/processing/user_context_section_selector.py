import json
from typing import Any

from lfx.custom import Component
from lfx.inputs import DataInput, MultiselectInput
from lfx.io import Output
from lfx.log.logger import logger
from lfx.schema import Data
from lfx.schema.dotdict import dotdict


class UserContextSectionSelectorComponent(Component):
    display_name = "User Context Section Selector"
    description = "Extract and select specific sections from user context data object."
    icon = "file-json"
    name = "UserContextSectionSelector"

    inputs = [
        DataInput(
            name="user_context_data",
            display_name="User Context Data",
            info="Data object containing user context with sections to extract.",
            required=True,
        ),
        MultiselectInput(
            name="selected_sections",
            display_name="Selected Sections",
            info="Select one or more sections to include in the output context.",
            options=[],
            value=[],
            dynamic=True,
            real_time_refresh=True,
        ),
    ]

    outputs = [
        Output(display_name="Selected Context", name="selected_context", method="extract_sections"),
    ]

    def _extract_sections_from_data(self, data: Any) -> dict[str, Any]:
        """Extract all sections from the user_context object."""
        sections = {}

        # Handle Data object
        if isinstance(data, Data):
            data_dict = data.model_dump()
            # Check if it has a 'data' key
            if "data" in data_dict:
                data_dict = data_dict["data"]
        elif isinstance(data, dict):
            data_dict = data
        else:
            # Try to parse as JSON string
            try:
                data_dict = json.loads(str(data))
            except (json.JSONDecodeError, TypeError):
                logger.warning("Could not parse input data as JSON or dict")
                return sections

        # Extract user_context if present
        user_context = data_dict.get("user_context", data_dict)

        # Extract all top-level sections
        if isinstance(user_context, dict):
            for key, value in user_context.items():
                sections[key] = value

        return sections

    def update_build_config(
        self, build_config: dotdict, field_value: Any, field_name: str | None = None
    ) -> dotdict:
        """Update build configuration to populate section options dynamically."""
        if field_name == "user_context_data" and field_value is not None:
            try:
                sections = self._extract_sections_from_data(field_value)
                section_names = list(sections.keys())
                
                if section_names:
                    build_config["selected_sections"]["options"] = section_names
                    logger.info(f"Updated section options: {section_names}")
                else:
                    logger.warning("No sections found in user context data")
                    build_config["selected_sections"]["options"] = []

                # Remove any selected sections that are no longer available
                if isinstance(build_config["selected_sections"]["value"], list):
                    current_selections = build_config["selected_sections"]["value"]
                    build_config["selected_sections"]["value"] = [
                        s for s in current_selections if s in section_names
                    ]
            except Exception as e:  # noqa: BLE001
                logger.warning(f"Error extracting sections from user context data: {e}")
                build_config["selected_sections"]["options"] = []
        elif field_name is None and self.user_context_data:
            # Called during initialization - try to populate options from current data
            try:
                sections = self._extract_sections_from_data(self.user_context_data)
                section_names = list(sections.keys())
                if section_names:
                    build_config["selected_sections"]["options"] = section_names
            except Exception as e:  # noqa: BLE001
                logger.debug(f"Could not extract sections during initialization: {e}")

        return build_config

    def extract_sections(self) -> Data:
        """Extract selected sections from user context data."""
        if not self.user_context_data:
            msg = "User context data is required."
            raise ValueError(msg)

        if not self.selected_sections:
            msg = "At least one section must be selected."
            raise ValueError(msg)

        # Extract all sections
        all_sections = self._extract_sections_from_data(self.user_context_data)

        # Build the selected context
        selected_context = {}
        for section_name in self.selected_sections:
            if section_name in all_sections:
                selected_context[section_name] = all_sections[section_name]
            else:
                logger.warning(f"Section '{section_name}' not found in user context data")

        # Wrap in user_context structure to match original format
        output_data = {"user_context": selected_context}

        logger.info(f"Extracted {len(selected_context)} sections: {list(selected_context.keys())}")

        return Data(data=output_data)
