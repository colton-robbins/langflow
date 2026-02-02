import re

from lfx.custom.custom_component.component import Component
from lfx.inputs.inputs import HandleInput, MessageTextInput
from lfx.schema.data import Data
from lfx.schema.message import Message
from lfx.template.field.base import Output


class ExtractSpecialInstructionsComponent(Component):
    display_name = "Extract Special Instructions"
    description = "Extracts the 'special instructions' section from input text, deduplicates entries, and passes them along."
    documentation: str = "https://docs.langflow.org/components-processing#extract-special-instructions"
    icon = "fileText"

    inputs = [
        HandleInput(
            name="input_data",
            display_name="Input Text",
            input_types=["Message", "Data", "str"],
            info="Accepts text, Message, or Data object containing special instructions section.",
            required=True,
        ),
        MessageTextInput(
            name="section_header",
            display_name="Section Header",
            value="special_instructions",
            info="The header text that identifies the special instructions section (e.g., 'special_instructions'). Case-insensitive matching.",
            advanced=True,
        ),
        MessageTextInput(
            name="separator",
            display_name="Separator",
            value="\n",
            info="Separator used to split instructions into individual entries. Default is newline.",
            advanced=True,
        ),
    ]

    outputs = [
        Output(
            display_name="Deduplicated Instructions",
            name="deduplicated_instructions",
            info="The deduplicated special instructions as a Message.",
            method="extract_and_deduplicate",
        ),
    ]

    def _extract_text(self, input_data):
        """Extract text string from various input types."""
        # Always convert to string explicitly
        if isinstance(input_data, str):
            return str(input_data)
        elif hasattr(input_data, "text"):
            # Handle Message objects
            text_value = input_data.text
            if text_value is None:
                return ""
            return str(text_value)
        elif isinstance(input_data, Data):
            # Handle Data objects
            if hasattr(input_data, "data") and isinstance(input_data.data, dict):
                # Check various text fields in data
                for key in ["text", "content", "page_content"]:
                    if key in input_data.data:
                        value = input_data.data[key]
                        if value is not None:
                            return str(value)
                # If no text field found, convert entire data to string
                return str(input_data.data)
            return str(input_data)
        elif isinstance(input_data, dict):
            # Handle dict objects
            for key in ["text", "content", "page_content"]:
                if key in input_data:
                    value = input_data[key]
                    if value is not None:
                        return str(value)
            return str(input_data)
        else:
            return str(input_data)

    def _extract_all_sections(self, text: str, section_header: str) -> list[str]:
        """Extract all special instructions sections from text."""
        # Ensure text is a string
        text = str(text) if text is not None else ""
        
        if not text or not text.strip():
            return []
        
        # Ensure section_header is a string
        section_header = str(section_header) if section_header is not None else ""
        
        # Try both the exact header and with underscores/spaces converted
        header_variations = [
            section_header,
            section_header.replace(" ", "_"),
            section_header.replace("_", " "),
        ]
        
        all_contents = []
        
        for header_var in header_variations:
            if not header_var:
                continue
            
            # Simple string search (case-insensitive)
            text_lower = text.lower()
            header_lower = header_var.lower()
            
            # Find all occurrences of the header followed by colon
            search_pattern = f"{header_lower}:"
            start_idx = 0
            
            while True:
                idx = text_lower.find(search_pattern, start_idx)
                if idx < 0:
                    break
                
                # Find the colon position in the original text
                colon_idx = text.find(":", idx)
                if colon_idx >= 0:
                    # Get text after the colon
                    after_colon = text[colon_idx + 1:]
                    # Find the next newline
                    newline_idx = after_colon.find("\n")
                    if newline_idx >= 0:
                        # Extract up to the newline
                        content = after_colon[:newline_idx].strip()
                    else:
                        # No newline found, take everything
                        content = after_colon.strip()
                    
                    if content:
                        all_contents.append(content)
                        self.log(f"Found section with header '{header_var}': {content[:100]}...")
                
                # Move past this match to find the next one
                start_idx = idx + len(search_pattern)
            
            # If we found matches with this variation, break (don't try other variations)
            if all_contents:
                break
        
        if not all_contents:
            self.log(f"Section header '{section_header}' not found in text")
        
        return all_contents

    def _extract_instructions(self, text: str) -> list[str]:
        """Extract individual instructions from text."""
        section_header = self.section_header or "special_instructions"
        separator = self.separator or "\n"
        
        # Extract all sections
        all_sections = self._extract_all_sections(text, section_header)
        
        if not all_sections:
            self.log(f"Could not extract any sections with header '{section_header}'. Text preview: {text[:200]}...")
            return []
        
        self.log(f"Extracted {len(all_sections)} section(s)")
        
        # Collect all instructions from all sections
        all_instructions = []
        
        for section_text in all_sections:
            self.log(f"Processing section text (length: {len(section_text)}): {section_text[:200]}...")
            
            # First check for bullet points or numbered lists
            bullet_pattern = r"[•\-\*]\s*(.+?)(?=\n[•\-\*]|\n\d+\.|\n\n|$)"
            numbered_pattern = r"\d+\.\s*(.+?)(?=\n\d+\.|\n[•\-\*]|\n\n|$)"
            
            bullet_matches = re.findall(bullet_pattern, section_text, re.MULTILINE)
            numbered_matches = re.findall(numbered_pattern, section_text, re.MULTILINE)
            
            if bullet_matches:
                instructions = [match.strip() for match in bullet_matches if match.strip()]
                self.log(f"Found {len(instructions)} instructions using bullet pattern")
                all_instructions.extend(instructions)
            elif numbered_matches:
                instructions = [match.strip() for match in numbered_matches if match.strip()]
                self.log(f"Found {len(instructions)} instructions using numbered pattern")
                all_instructions.extend(instructions)
            else:
                # Split by separator and clean up
                instructions = [item.strip() for item in section_text.split(separator) if item.strip()]
                self.log(f"Found {len(instructions)} instructions using separator '{separator}'")
                all_instructions.extend(instructions)
        
        return all_instructions

    def _deduplicate(self, instructions: list[str]) -> list[str]:
        """Deduplicate instructions while preserving order."""
        seen = set()
        deduplicated = []
        for instruction in instructions:
            instruction_lower = instruction.lower().strip()
            if instruction_lower and instruction_lower not in seen:
                seen.add(instruction_lower)
                deduplicated.append(instruction)
        return deduplicated

    def extract_and_deduplicate(self) -> Message:
        """Extract special instructions from text, deduplicate them, and return as Message."""
        # Extract text from input
        input_text = self._extract_text(self.input_data)
        
        # Ensure it's a string
        input_text = str(input_text) if input_text is not None else ""
        
        self.log(f"Input text length: {len(input_text)}")
        if input_text:
            self.log(f"Input text preview: {input_text[:300]}...")
            # Also check end of text in case section is there
            if len(input_text) > 300:
                self.log(f"Input text ending: ...{input_text[-300:]}")
        
        if not input_text or not input_text.strip():
            self.status = "No input text provided."
            self.log("No input text provided")
            return Message(text="")

        # Get section header and ensure it's a string
        section_header = str(self.section_header) if hasattr(self, "section_header") and self.section_header else "special_instructions"
        self.log(f"Looking for section header: '{section_header}' (type: {type(section_header)})")

        # Extract instructions from the text
        instructions = self._extract_instructions(input_text)

        if not instructions:
            self.status = f"No special instructions section found with header '{section_header}'."
            self.log(f"No instructions extracted. Input text length: {len(input_text)}")
            return Message(text="")

        # Deduplicate instructions
        deduplicated = self._deduplicate(instructions)
        separator = self.separator or "\n"
        result_text = separator.join(deduplicated)

        self.log(f"Extracted {len(instructions)} instruction(s), {len(deduplicated)} unique after deduplication.")
        self.status = f"Found {len(deduplicated)} unique instruction(s)."

        return Message(text=result_text)
