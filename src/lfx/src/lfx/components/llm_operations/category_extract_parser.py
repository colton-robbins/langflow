from lfx.custom.custom_component.component import Component
from lfx.inputs.inputs import HandleInput, MessageTextInput
from lfx.schema.data import Data
from lfx.schema.message import Message
from lfx.template.field.base import Output


class ParserComponent(Component):
    display_name = "Parser"
    description = "Searches input text for category names and outputs the first match found."
    documentation: str = "https://docs.langflow.org/parser"
    icon = "braces"

    # Define the exact category terms to search for
    CATEGORIES = [
        "demographics",
        "group_info",
        "medical_claim",
        "j_code",
        "medical_savings",
        "pharm_and_phys",
        "pharmacy_claim",
        "opioid_and_benzo",
        "pharmacy_savings",
        "Clarification needed",
    ]

    inputs = [
        HandleInput(
            name="input_data",
            display_name="Input Text",
            input_types=["Message", "Data", "str"],
            info="Accepts a string, Message, or Data object containing text to search for category names.",
            required=True,
        ),
    ]

    outputs = [
        Output(
            display_name="Category",
            name="parsed_text",
            info="The first category name found in the input text, or 'Clarification needed' if none found.",
            method="parse_combined_text",
        ),
    ]

    def _extract_text(self, input_data):
        """Extract text string from various input types."""
        if isinstance(input_data, str):
            return input_data
        elif hasattr(input_data, "text"):
            # Handle Message objects
            return str(input_data.text)
        elif isinstance(input_data, Data):
            # Handle Data objects
            if hasattr(input_data, "data") and isinstance(input_data.data, dict):
                # Check various text fields in data
                for key in ["text", "content", "page_content"]:
                    if key in input_data.data:
                        return str(input_data.data[key])
                # If no text field found, convert entire data to string
                return str(input_data.data)
            return str(input_data)
        elif isinstance(input_data, dict):
            # Handle dict objects
            for key in ["text", "content", "page_content"]:
                if key in input_data:
                    return str(input_data[key])
            return str(input_data)
        else:
            return str(input_data)

    def parse_combined_text(self) -> Message:
        """Search input text for category names and return the first match found."""
        # Extract text from input
        search_text = self._extract_text(self.input_data)
        
        if not search_text or not search_text.strip():
            self.status = "No input text provided."
            return Message(text="./Clarification needed")

        self.log(f"Searching text for categories: '{search_text[:100]}...'")

        # Search for each category term (exact match, case-sensitive)
        # Check in order, return first match found
        for category in self.CATEGORIES:
            if category in search_text:
                self.log(f"Found category '{category}' in input text")
                self.status = f"Category: {category}"
                return Message(text=f"./{category}")

        # If no category found, return default
        self.log("No category found in input text")
        self.status = "Category: Clarification needed"
        return Message(text="./Clarification needed")
