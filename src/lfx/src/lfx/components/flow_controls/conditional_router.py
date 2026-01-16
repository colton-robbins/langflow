import json

from lfx.custom.custom_component.component import Component
from lfx.io import MessageTextInput, Output
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


class ConditionalRouterComponent(Component):
    display_name = "Category Router"
    description = "Routes input to 5 categories based on keywords."
    icon = "split"
    name = "ConditionalRouter"

    inputs = [
        MessageTextInput(
            name="input_text",
            display_name="Input Text",
            info="Text to categorize and route.",
            required=True,
        ),
    ]

    outputs = [
        Output(
            display_name="Analytics",
            name="analytics_agent",
            method="route_analytics",
        ),
        Output(
            display_name="Knowledge Documents",
            name="knowledge_documents",
            method="route_knowledge_documents",
        ),
        Output(
            display_name="PCM Knowledge",
            name="pcm_knowledge",
            method="route_pcm_knowledge",
        ),
        Output(
            display_name="Client Knowledge",
            name="client_knowledge",
            method="route_client_knowledge",
        ),
        Output(
            display_name="Other",
            name="other",
            method="route_other",
        ),
    ]

    def get_category(self, text: str) -> str:
        """Simple keyword matching - returns category name."""
        text_lower = text.lower()
        
        # Check for specific patterns
        if "validated" in text_lower and "pcm" in text_lower:
            return "pcm_knowledge"
        
        if "validated" in text_lower and "client" in text_lower:
            return "client_knowledge"
        
        if "validated" in text_lower and ("knowledge" in text_lower or "document" in text_lower):
            return "knowledge_documents"
        
        if "escalate" in text_lower and "analytics" in text_lower:
            return "analytics_agent"
        
        # Simple keyword checks
        if "analytics" in text_lower:
            return "analytics_agent"
        
        if "pcm" in text_lower:
            return "pcm_knowledge"
        
        if "client" in text_lower:
            return "client_knowledge"
        
        if "document" in text_lower or "knowledge" in text_lower:
            return "knowledge_documents"
        
        return "other"  # Default

    def route_analytics(self) -> Message:
        """Route for analytics queries."""
        text = extract_text_from_input(self.input_text)
        message = Message(text=text)
        
        if self.get_category(text) == "analytics_agent":
            self.status = message
        
        return message

    def route_knowledge_documents(self) -> Message:
        """Route for document/knowledge queries."""
        text = extract_text_from_input(self.input_text)
        message = Message(text=text)
        
        if self.get_category(text) == "knowledge_documents":
            self.status = message
        
        return message

    def route_pcm_knowledge(self) -> Message:
        """Route for PCM company info."""
        text = extract_text_from_input(self.input_text)
        message = Message(text=text)
        
        if self.get_category(text) == "pcm_knowledge":
            self.status = message
        
        return message

    def route_client_knowledge(self) -> Message:
        """Route for client info."""
        text = extract_text_from_input(self.input_text)
        message = Message(text=text)
        
        if self.get_category(text) == "client_knowledge":
            self.status = message
        
        return message

    def route_other(self) -> Message:
        """Route for other queries that don't match specific categories."""
        text = extract_text_from_input(self.input_text)
        message = Message(text=text)
        
        if self.get_category(text) == "other":
            self.status = message
        
        return message
