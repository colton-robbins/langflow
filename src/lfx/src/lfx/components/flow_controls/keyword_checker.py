import json

from lfx.custom.custom_component.component import Component
from lfx.inputs.inputs import BoolInput, HandleInput, MessageTextInput
from lfx.io import Output
from lfx.schema.message import Message


def extract_text_from_input(input_value) -> str:
    """Extract text from any input type."""
    if isinstance(input_value, str):
        return input_value
    
    if isinstance(input_value, Message):
        return input_value.text or str(input_value)
    
    if isinstance(input_value, dict):
        try:
            return json.dumps(input_value, default=str)
        except Exception:
            return str(input_value)
    
    return str(input_value) if input_value is not None else ""


class KeywordCheckerComponent(Component):
    display_name = "Keyword Checker"
    description = "Checks if a keyword exists anywhere in the input text and routes to appropriate output."
    icon = "search"
    name = "KeywordChecker"

    inputs = [
        HandleInput(
            name="input_text",
            display_name="Input Text",
            input_types=["Message", "Data", "str"],
            info="Text to search for the keyword.",
            required=True,
        ),
        MessageTextInput(
            name="keyword",
            display_name="Keyword",
            info="The keyword to search for in the input text.",
            required=True,
        ),
        BoolInput(
            name="case_sensitive",
            display_name="Case Sensitive",
            info="Whether the keyword search should be case-sensitive.",
            value=False,
            advanced=False,
        ),
    ]

    outputs = [
        Output(
            display_name="Found",
            name="found",
            method="check_keyword_found",
            info="Returns the input text if keyword is found.",
        ),
        Output(
            display_name="Not Found",
            name="not_found",
            method="check_keyword_not_found",
            info="Returns the input text if keyword is not found.",
        ),
    ]

    def keyword_exists(self, text: str, keyword: str, case_sensitive: bool) -> bool:
        """Check if keyword exists in text based on case sensitivity setting."""
        if not text or not keyword:
            return False
        
        if case_sensitive:
            return keyword in text
        else:
            return keyword.lower() in text.lower()

    def check_keyword_found(self) -> Message:
        """Return input text if keyword is found."""
        text = extract_text_from_input(self.input_text)
        keyword = extract_text_from_input(self.keyword)
        case_sensitive = self.case_sensitive
        
        message = Message(text=text)
        
        if self.keyword_exists(text, keyword, case_sensitive):
            self.status = f"Keyword '{keyword}' found in input text."
            self.log(f"Keyword '{keyword}' found in input text.")
        else:
            self.log(f"Keyword '{keyword}' not found in input text.")
        
        return message

    def check_keyword_not_found(self) -> Message:
        """Return input text if keyword is not found."""
        text = extract_text_from_input(self.input_text)
        keyword = extract_text_from_input(self.keyword)
        case_sensitive = self.case_sensitive
        
        message = Message(text=text)
        
        if not self.keyword_exists(text, keyword, case_sensitive):
            self.status = f"Keyword '{keyword}' not found in input text."
            self.log(f"Keyword '{keyword}' not found in input text.")
        else:
            self.log(f"Keyword '{keyword}' found in input text.")
        
        return message
