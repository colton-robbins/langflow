from lfx.custom.custom_component.component import Component
from lfx.inputs.inputs import HandleInput
from lfx.schema.data import Data
from lfx.schema.message import Message
from lfx.template.field.base import Output


class ParallelDistributorComponent(Component):
    display_name = "Parallel Distributor"
    description = "Distributes input data to multiple outputs as Message objects for parallel execution. All outputs receive the same input Message simultaneously, ensuring downstream nodes run in parallel."
    documentation: str = "https://docs.langflow.org/components-processing#parallel-distributor"
    icon = "Split"

    inputs = [
        HandleInput(
            name="input_data",
            display_name="Input Data",
            input_types=["Message", "Data", "str"],
            info="The input data to distribute to multiple outputs. Accepts Message, Data, or string.",
            required=True,
        ),
    ]

    outputs = [
        Output(
            display_name="Output 1",
            name="output_1",
            info="First parallel output as Message.",
            method="distribute_output_1",
            types=["Message"],
            group_outputs=True,
        ),
        Output(
            display_name="Output 2",
            name="output_2",
            info="Second parallel output as Message.",
            method="distribute_output_2",
            types=["Message"],
            group_outputs=True,
        ),
        Output(
            display_name="Output 3",
            name="output_3",
            info="Third parallel output as Message.",
            method="distribute_output_3",
            types=["Message"],
            group_outputs=True,
        ),
        Output(
            display_name="Output 4",
            name="output_4",
            info="Fourth parallel output as Message.",
            method="distribute_output_4",
            types=["Message"],
            group_outputs=True,
        ),
        Output(
            display_name="Output 5",
            name="output_5",
            info="Fifth parallel output as Message.",
            method="distribute_output_5",
            types=["Message"],
            group_outputs=True,
        ),
        Output(
            display_name="Output 6",
            name="output_6",
            info="Sixth parallel output as Message.",
            method="distribute_output_6",
            types=["Message"],
            group_outputs=True,
        ),
        Output(
            display_name="Output 7",
            name="output_7",
            info="Seventh parallel output as Message.",
            method="distribute_output_7",
            types=["Message"],
            group_outputs=True,
        ),
    ]

    def _extract_text(self, input_data):
        """Extract text string from various input types."""
        self.log(f"Extracting text from input type: {type(input_data)}")
        
        if isinstance(input_data, str):
            self.log(f"Input is string: {input_data[:100] if len(input_data) > 100 else input_data}")
            return str(input_data)
        
        elif isinstance(input_data, Message):
            self.log(f"Input is Message object")
            text_value = input_data.text
            self.log(f"Message.text value: {repr(text_value)}")
            
            if hasattr(input_data, "get_text"):
                text_from_get = input_data.get_text()
                self.log(f"Message.get_text() value: {repr(text_from_get)}")
                if text_from_get and str(text_from_get).strip():
                    return str(text_from_get)
            
            if text_value is None or text_value == "":
                self.log("Message.text is empty, checking data dictionary")
                if hasattr(input_data, "data") and isinstance(input_data.data, dict):
                    self.log(f"Message.data keys: {list(input_data.data.keys())}")
                    for key in ["text", "content", "page_content"]:
                        if key in input_data.data:
                            value = input_data.data[key]
                            self.log(f"Found '{key}' in data: {repr(value)}")
                            if value is not None and str(value).strip():
                                return str(value)
                    self.log("No text found in data dictionary, checking all data values")
                    for key, value in input_data.data.items():
                        if isinstance(value, str) and value.strip():
                            self.log(f"Found non-empty string in data['{key}']: {value[:100]}")
                            return value
            
            if text_value:
                return str(text_value)
            self.log("No text found in Message, returning empty string")
            return ""
        
        elif hasattr(input_data, "text"):
            text_value = input_data.text
            self.log(f"Object has text attribute: {repr(text_value)}")
            if text_value is None:
                return ""
            return str(text_value)
        
        elif isinstance(input_data, Data):
            self.log(f"Input is Data object")
            if hasattr(input_data, "get_text"):
                text_from_get = input_data.get_text()
                self.log(f"Data.get_text() value: {repr(text_from_get)}")
                if text_from_get:
                    return str(text_from_get)
            if hasattr(input_data, "data") and isinstance(input_data.data, dict):
                self.log(f"Data.data keys: {list(input_data.data.keys())}")
                for key in ["text", "content", "page_content"]:
                    if key in input_data.data:
                        value = input_data.data[key]
                        if value is not None:
                            return str(value)
                return str(input_data.data)
            return str(input_data)
        
        elif isinstance(input_data, dict):
            self.log(f"Input is dict with keys: {list(input_data.keys())}")
            for key in ["text", "content", "page_content"]:
                if key in input_data:
                    value = input_data[key]
                    if value is not None:
                        return str(value)
            return str(input_data)
        else:
            self.log(f"Input is unknown type, converting to string")
            return str(input_data)

    def _to_message(self, input_data):
        """Convert input to Message object."""
        self.log(f"Converting to Message, input type: {type(input_data)}")
        
        if isinstance(input_data, Message):
            self.log(f"Input is already a Message")
            self.log(f"Message attributes: text={repr(input_data.text)}, sender={getattr(input_data, 'sender', None)}, session_id={getattr(input_data, 'session_id', None)}")
            
            text = self._extract_text(input_data)
            self.log(f"Extracted text length: {len(text)}, preview: {repr(text[:100]) if text else 'EMPTY'}")
            
            new_message = Message(
                text=text,
                sender=getattr(input_data, "sender", None),
                sender_name=getattr(input_data, "sender_name", None),
                session_id=getattr(input_data, "session_id", None),
                context_id=getattr(input_data, "context_id", None),
                files=getattr(input_data, "files", None),
            )
            self.log(f"Created new Message with text length: {len(new_message.text)}, text value: {repr(new_message.text)}")
            self.status = f"Distributed Message with text length: {len(new_message.text)}"
            return new_message
        
        text = self._extract_text(input_data)
        self.log(f"Extracted text length: {len(text)}, preview: {repr(text[:100]) if text else 'EMPTY'}")
        new_message = Message(text=text)
        self.log(f"Created new Message with text length: {len(new_message.text)}")
        return new_message

    def distribute_output_1(self) -> Message:
        """Distribute input to first output as Message."""
        self.log("Executing distribute_output_1")
        result = self._to_message(self.input_data)
        self.log(f"distribute_output_1 completed, result text length: {len(result.text)}")
        return result

    def distribute_output_2(self) -> Message:
        """Distribute input to second output as Message."""
        self.log("Executing distribute_output_2")
        result = self._to_message(self.input_data)
        self.log(f"distribute_output_2 completed, result text length: {len(result.text)}")
        return result

    def distribute_output_3(self) -> Message:
        """Distribute input to third output as Message."""
        self.log("Executing distribute_output_3")
        result = self._to_message(self.input_data)
        self.log(f"distribute_output_3 completed, result text length: {len(result.text)}")
        return result

    def distribute_output_4(self) -> Message:
        """Distribute input to fourth output as Message."""
        self.log("Executing distribute_output_4")
        result = self._to_message(self.input_data)
        self.log(f"distribute_output_4 completed, result text length: {len(result.text)}")
        return result

    def distribute_output_5(self) -> Message:
        """Distribute input to fifth output as Message."""
        self.log("Executing distribute_output_5")
        result = self._to_message(self.input_data)
        self.log(f"distribute_output_5 completed, result text length: {len(result.text)}")
        return result

    def distribute_output_6(self) -> Message:
        """Distribute input to sixth output as Message."""
        self.log("Executing distribute_output_6")
        result = self._to_message(self.input_data)
        self.log(f"distribute_output_6 completed, result text length: {len(result.text)}")
        return result

    def distribute_output_7(self) -> Message:
        """Distribute input to seventh output as Message."""
        self.log("Executing distribute_output_7")
        result = self._to_message(self.input_data)
        self.log(f"distribute_output_7 completed, result text length: {len(result.text)}")
        return result
