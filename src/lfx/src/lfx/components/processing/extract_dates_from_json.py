import json
import re
from json import JSONDecodeError
from typing import Any

from lfx.custom.custom_component.component import Component
from lfx.inputs.inputs import HandleInput
from lfx.schema.data import Data
from lfx.schema.message import Message
from lfx.template.field.base import Output


class ExtractDatesFromJSONComponent(Component):
    display_name = "Extract Dates From JSON"
    description = "Extracts dates from a JSON message. Returns the whole message and extracted dates as separate outputs."
    documentation: str = "https://docs.langflow.org/components-processing#extract-dates-from-json"
    icon = "calendar"

    inputs = [
        HandleInput(
            name="input_data",
            display_name="Input JSON",
            input_types=["Message", "Data", "str"],
            info="Accepts JSON as text, Message, or Data object containing date fields (start_date, end_date, multiple_periods).",
            required=True,
        ),
    ]

    outputs = [
        Output(
            display_name="Whole Message",
            name="whole_message",
            info="The original input message as a Message object.",
            method="get_whole_message",
            types=["Message"],
            group_outputs=True,
        ),
        Output(
            display_name="Dates Only",
            name="dates_only",
            info="Extracted dates as a JSON string containing start_date, end_date, and dates from multiple_periods.",
            method="extract_dates",
            types=["Message"],
            group_outputs=True,
        ),
    ]

    def _extract_text(self, input_data):
        """Extract text string from various input types."""
        self.log(f"[_extract_text] Input type: {type(input_data)}")
        
        if isinstance(input_data, str):
            self.log(f"[_extract_text] Input is string, length: {len(input_data)}")
            self.log(f"[_extract_text] String preview: {input_data[:200]}...")
            return str(input_data)
        elif hasattr(input_data, "text"):
            text_value = input_data.text
            self.log(f"[_extract_text] Input has 'text' attribute, value type: {type(text_value)}")
            if text_value is None:
                self.log("[_extract_text] Text value is None, returning empty string")
                return ""
            self.log(f"[_extract_text] Text value length: {len(str(text_value))}")
            self.log(f"[_extract_text] Text value preview: {str(text_value)[:200]}...")
            return str(text_value)
        elif isinstance(input_data, Data):
            self.log("[_extract_text] Input is Data object")
            if hasattr(input_data, "data") and isinstance(input_data.data, dict):
                self.log(f"[_extract_text] Data.data is dict with keys: {list(input_data.data.keys())}")
                json_str = json.dumps(input_data.data)
                self.log(f"[_extract_text] Converted to JSON string, length: {len(json_str)}")
                return json_str
            self.log("[_extract_text] Data object doesn't have dict data, converting to string")
            return str(input_data)
        elif isinstance(input_data, dict):
            self.log(f"[_extract_text] Input is dict with keys: {list(input_data.keys())}")
            json_str = json.dumps(input_data)
            self.log(f"[_extract_text] Converted to JSON string, length: {len(json_str)}")
            return json_str
        else:
            self.log(f"[_extract_text] Input is unknown type, converting to string: {type(input_data)}")
            return str(input_data)

    def _extract_json_from_text(self, text: str) -> str:
        """Extract JSON string from text, handling various formats."""
        self.log(f"[_extract_json_from_text] Starting with text length: {len(text) if text else 0}")
        
        if not text or not text.strip():
            self.log("[_extract_json_from_text] Text is empty or whitespace only")
            return ""
        
        text = text.strip()
        self.log(f"[_extract_json_from_text] Text after strip, length: {len(text)}")
        self.log(f"[_extract_json_from_text] Text starts with: {text[:50]}...")
        self.log(f"[_extract_json_from_text] Text ends with: ...{text[-50:]}")
        
        # Try to find JSON block in markdown code fences
        self.log("[_extract_json_from_text] Trying to find JSON in markdown code fences")
        json_block_pattern = r'```(?:json)?\s*\n?(.*?)\n?```'
        match = re.search(json_block_pattern, text, re.DOTALL)
        if match:
            extracted = match.group(1).strip()
            self.log(f"[_extract_json_from_text] Found JSON in code fence, extracted length: {len(extracted)}")
            self.log(f"[_extract_json_from_text] Extracted preview: {extracted[:200]}...")
            return extracted
        
        # Try to find JSON object/array pattern
        self.log("[_extract_json_from_text] Trying to find JSON object/array pattern")
        json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}|\[[^\[\]]*(?:\[[^\[\]]*\][^\[\]]*)*\]'
        match = re.search(json_pattern, text, re.DOTALL)
        if match:
            extracted = match.group(0).strip()
            self.log(f"[_extract_json_from_text] Found JSON pattern match, extracted length: {len(extracted)}")
            self.log(f"[_extract_json_from_text] Extracted preview: {extracted[:200]}...")
            return extracted
        
        # Strip markdown code fences if present at start/end
        self.log("[_extract_json_from_text] Trying to strip markdown code fences")
        original_text = text
        if text.startswith("```json"):
            text = text[7:]
            self.log("[_extract_json_from_text] Removed ```json from start")
        elif text.startswith("```"):
            text = text[3:]
            self.log("[_extract_json_from_text] Removed ``` from start")
        
        text = text.strip()
        if text.endswith("```"):
            text = text[:-3]
            self.log("[_extract_json_from_text] Removed ``` from end")
        
        text = text.strip()
        self.log(f"[_extract_json_from_text] Final extracted text length: {len(text)}")
        self.log(f"[_extract_json_from_text] Final text preview: {text[:200]}...")
        
        return text

    def _parse_json(self, text: str) -> Any:
        """Parse JSON string into a dictionary or list, trying multiple strategies."""
        self.log(f"[_parse_json] Starting parse, input text length: {len(text) if text else 0}")
        
        if not text or not text.strip():
            self.log("[_parse_json] Text is empty or whitespace only")
            return {}
        
        # Extract JSON from text
        self.log("[_parse_json] Extracting JSON from text")
        json_text = self._extract_json_from_text(text)
        
        if not json_text:
            self.log("[_parse_json] No JSON content found in text after extraction")
            return {}
        
        self.log(f"[_parse_json] Extracted JSON text length: {len(json_text)}")
        self.log(f"[_parse_json] Extracted JSON preview: {json_text[:300]}...")
        
        # Try parsing as-is first
        self.log("[_parse_json] Strategy 1: Trying to parse JSON as-is")
        try:
            parsed = json.loads(json_text)
            self.log(f"[_parse_json] Strategy 1 SUCCESS! Parsed type: {type(parsed)}")
            if isinstance(parsed, dict):
                self.log(f"[_parse_json] Parsed dict keys: {list(parsed.keys())}")
            elif isinstance(parsed, list):
                self.log(f"[_parse_json] Parsed list length: {len(parsed)}")
            return parsed
        except JSONDecodeError as e:
            self.log(f"[_parse_json] Strategy 1 FAILED: {e}")
        
        # Try cleaning up common issues
        self.log("[_parse_json] Strategy 2: Cleaning up JSON (removing trailing commas)")
        cleaned = json_text
        # Remove trailing commas before closing braces/brackets
        cleaned = re.sub(r',(\s*[}\]])', r'\1', cleaned)
        self.log(f"[_parse_json] Cleaned JSON length: {len(cleaned)}")
        self.log(f"[_parse_json] Cleaned JSON preview: {cleaned[:300]}...")
        
        try:
            parsed = json.loads(cleaned)
            self.log(f"[_parse_json] Strategy 2 SUCCESS! Parsed type: {type(parsed)}")
            if isinstance(parsed, dict):
                self.log(f"[_parse_json] Parsed dict keys: {list(parsed.keys())}")
            elif isinstance(parsed, list):
                self.log(f"[_parse_json] Parsed list length: {len(parsed)}")
            return parsed
        except JSONDecodeError as e:
            self.log(f"[_parse_json] Strategy 2 FAILED: {e}")
        
        # Try to extract just the first JSON object/array
        self.log("[_parse_json] Strategy 3: Extracting first complete JSON structure")
        try:
            # Find the first complete JSON structure
            brace_count = 0
            bracket_count = 0
            start_idx = -1
            
            for i, char in enumerate(json_text):
                if char == '{':
                    if start_idx == -1:
                        start_idx = i
                        self.log(f"[_parse_json] Found opening brace at index {i}")
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0 and start_idx != -1:
                        extracted = json_text[start_idx:i+1]
                        self.log(f"[_parse_json] Found complete object from {start_idx} to {i+1}, length: {len(extracted)}")
                        self.log(f"[_parse_json] Extracted preview: {extracted[:200]}...")
                        try:
                            parsed = json.loads(extracted)
                            self.log(f"[_parse_json] Strategy 3 SUCCESS! Parsed type: {type(parsed)}")
                            if isinstance(parsed, dict):
                                self.log(f"[_parse_json] Parsed dict keys: {list(parsed.keys())}")
                            return parsed
                        except JSONDecodeError as e:
                            self.log(f"[_parse_json] Strategy 3 FAILED to parse extracted object: {e}")
                            break
                elif char == '[':
                    if start_idx == -1:
                        start_idx = i
                        self.log(f"[_parse_json] Found opening bracket at index {i}")
                    bracket_count += 1
                elif char == ']':
                    bracket_count -= 1
                    if bracket_count == 0 and start_idx != -1:
                        extracted = json_text[start_idx:i+1]
                        self.log(f"[_parse_json] Found complete array from {start_idx} to {i+1}, length: {len(extracted)}")
                        try:
                            parsed = json.loads(extracted)
                            self.log(f"[_parse_json] Strategy 3 SUCCESS! Parsed type: {type(parsed)}")
                            return parsed
                        except JSONDecodeError as e:
                            self.log(f"[_parse_json] Strategy 3 FAILED to parse extracted array: {e}")
                            break
        except Exception as e:
            self.log(f"[_parse_json] Strategy 3 EXCEPTION: {e}")
        
        self.log(f"[_parse_json] ALL STRATEGIES FAILED. JSON text preview: {json_text[:200]}...")
        return {}

    def _is_date_field(self, key: str) -> bool:
        """Check if a key name suggests it's a date field."""
        key_lower = key.lower()
        date_indicators = [
            'date', 'time', 'period', 'range', 'start', 'end', 
            'from', 'to', 'begin', 'finish', 'created', 'updated',
            'modified', 'timestamp', 'since', 'until'
        ]
        result = any(indicator in key_lower for indicator in date_indicators)
        self.log(f"[_is_date_field] Key '{key}' (lower: '{key_lower}') is_date_field: {result}")
        return result

    def _is_date_value(self, value: Any) -> bool:
        """Check if a value looks like a date."""
        self.log(f"[_is_date_value] Checking value: {value} (type: {type(value)})")
        
        if not isinstance(value, str):
            self.log(f"[_is_date_value] Value is not a string, returning False")
            return False
        
        # Common date patterns
        date_patterns = [
            r'\d{4}-\d{2}-\d{2}',  # YYYY-MM-DD
            r'\d{2}/\d{2}/\d{4}',  # MM/DD/YYYY
            r'\d{4}/\d{2}/\d{2}',  # YYYY/MM/DD
            r'\d{2}-\d{2}-\d{4}',  # MM-DD-YYYY
            r'\d{4}\.\d{2}\.\d{2}',  # YYYY.MM.DD
        ]
        
        for pattern in date_patterns:
            if re.match(pattern, value):
                self.log(f"[_is_date_value] Value '{value}' matches pattern '{pattern}', returning True")
                return True
        
        self.log(f"[_is_date_value] Value '{value}' doesn't match any date pattern, returning False")
        return False

    def _extract_dates_recursive(self, obj: Any, path: str = "", dates: dict = None) -> dict:
        """Recursively extract date fields from JSON structure."""
        if dates is None:
            dates = {}
        
        self.log(f"[_extract_dates_recursive] Processing object at path '{path}', type: {type(obj)}")
        
        if isinstance(obj, dict):
            self.log(f"[_extract_dates_recursive] Processing dict with {len(obj)} keys: {list(obj.keys())}")
            for key, value in obj.items():
                current_path = f"{path}.{key}" if path else key
                self.log(f"[_extract_dates_recursive] Processing key '{key}' at path '{current_path}', value type: {type(value)}")
                
                # Check if this is a known date field
                is_date = self._is_date_field(key)
                if is_date:
                    if key not in dates or dates[key] is None:
                        dates[key] = value
                        self.log(f"[_extract_dates_recursive] ✓ Found date field '{current_path}': {value}")
                    else:
                        self.log(f"[_extract_dates_recursive] Date field '{key}' already exists in dates dict")
                
                # Also check standard field names
                standard_names = ["start_date", "end_date", "startDate", "endDate", 
                                 "start_date", "end_date", "date_start", "date_end",
                                 "from_date", "to_date", "fromDate", "toDate"]
                if key in standard_names:
                    dates[key] = value
                    self.log(f"[_extract_dates_recursive] ✓ Found standard date field '{current_path}': {value}")
                
                # Recurse into nested structures
                if isinstance(value, (dict, list)):
                    self.log(f"[_extract_dates_recursive] Recursing into nested {type(value).__name__} at '{current_path}'")
                    self._extract_dates_recursive(value, current_path, dates)
                else:
                    self.log(f"[_extract_dates_recursive] Value at '{current_path}' is not nested structure, skipping recursion")
        
        elif isinstance(obj, list):
            self.log(f"[_extract_dates_recursive] Processing list with {len(obj)} items")
            for i, item in enumerate(obj):
                current_path = f"{path}[{i}]" if path else f"[{i}]"
                self.log(f"[_extract_dates_recursive] Processing list item {i} at path '{current_path}', type: {type(item)}")
                if isinstance(item, (dict, list)):
                    self.log(f"[_extract_dates_recursive] Recursing into nested {type(item).__name__} at '{current_path}'")
                    self._extract_dates_recursive(item, current_path, dates)
                elif self._is_date_value(item):
                    dates[f"{current_path}_date"] = item
                    self.log(f"[_extract_dates_recursive] ✓ Found date value at '{current_path}': {item}")
                else:
                    self.log(f"[_extract_dates_recursive] List item {i} is not a date value or nested structure")
        
        else:
            self.log(f"[_extract_dates_recursive] Object at '{path}' is not dict or list, type: {type(obj)}")
        
        self.log(f"[_extract_dates_recursive] Returning dates dict with {len(dates)} entries: {list(dates.keys())}")
        return dates

    def _extract_dates_from_json(self, json_data: Any) -> dict:
        """Extract date fields from JSON data (dict, list, or nested structures)."""
        self.log(f"[_extract_dates_from_json] Starting extraction, input type: {type(json_data)}")
        dates = {}
        
        if json_data is None:
            self.log("[_extract_dates_from_json] JSON data is None, returning empty dict")
            return dates
        
        # Handle dict
        if isinstance(json_data, dict):
            self.log(f"[_extract_dates_from_json] JSON data is dict with {len(json_data)} keys")
            self.log(f"[_extract_dates_from_json] Dict keys: {list(json_data.keys())}")
            
            # FIRST: Check top-level standard fields explicitly (most reliable)
            self.log("[_extract_dates_from_json] STEP 1: Checking top-level standard fields FIRST")
            standard_fields = {
                "start_date": ["start_date", "startDate", "date_start", "from_date", "fromDate"],
                "end_date": ["end_date", "endDate", "date_end", "to_date", "toDate"],
                "multiple_periods": ["multiple_periods", "multiplePeriods", "periods", "date_periods"]
            }
            
            for target_key, possible_keys in standard_fields.items():
                self.log(f"[_extract_dates_from_json] Checking for {target_key} with possible keys: {possible_keys}")
                found = False
                for key in possible_keys:
                    if key in json_data:
                        dates[target_key] = json_data[key]
                        self.log(f"[_extract_dates_from_json] ✓✓✓ FOUND {target_key} as '{key}': {json_data[key]}")
                        found = True
                        break
                    else:
                        self.log(f"[_extract_dates_from_json] Key '{key}' not found in JSON")
                if not found:
                    self.log(f"[_extract_dates_from_json] ✗ {target_key} not found with any of the possible keys")
            
            self.log(f"[_extract_dates_from_json] After standard field check, dates dict has {len(dates)} entries: {list(dates.keys())}")
            
            # SECOND: Extract dates recursively for any other date fields
            self.log("[_extract_dates_from_json] STEP 2: Starting recursive extraction for other date fields")
            recursive_dates = self._extract_dates_recursive(json_data)
            self.log(f"[_extract_dates_from_json] After recursive extraction, found {len(recursive_dates)} entries: {list(recursive_dates.keys())}")
            
            # Merge recursive dates (don't overwrite standard fields)
            for key, value in recursive_dates.items():
                if key not in dates:  # Only add if not already found as standard field
                    dates[key] = value
                    self.log(f"[_extract_dates_from_json] Added recursive date field '{key}': {value}")
                else:
                    self.log(f"[_extract_dates_from_json] Skipping recursive date '{key}' (already found as standard field)")
        
        # Handle list (might contain date objects)
        elif isinstance(json_data, list):
            self.log(f"[_extract_dates_from_json] JSON data is list with {len(json_data)} items")
            self.log("[_extract_dates_from_json] Starting recursive extraction on list")
            dates = self._extract_dates_recursive(json_data)
            self.log(f"[_extract_dates_from_json] After recursive extraction, dates dict has {len(dates)} entries: {list(dates.keys())}")
        
        else:
            self.log(f"[_extract_dates_from_json] JSON data is not a dict or list, type: {type(json_data)}, value: {json_data}")
        
        self.log(f"[_extract_dates_from_json] ========== FINAL dates dict: {dates}")
        self.log(f"[_extract_dates_from_json] ========== FINAL dates dict keys: {list(dates.keys())}")
        self.log(f"[_extract_dates_from_json] ========== FINAL dates dict length: {len(dates)}")
        self.log(f"[_extract_dates_from_json] ========== FINAL dates dict is empty: {not dates}")
        if dates:
            for key, value in dates.items():
                self.log(f"[_extract_dates_from_json] ==========   '{key}': {value}")
        return dates

    def get_whole_message(self) -> Message:
        """Return the whole input message as-is."""
        input_text = self._extract_text(self.input_data)
        input_text = str(input_text) if input_text is not None else ""
        
        self.log(f"Whole Message - Input text length: {len(input_text)}")
        if input_text:
            self.log(f"Whole Message - Input text preview: {input_text[:300]}...")
        
        if not input_text or not input_text.strip():
            self.status = "No input text provided."
            self.log("No input text provided")
            return Message(text="")
        
        self.status = "Returned whole message."
        return Message(text=input_text)

    def extract_dates(self) -> Message:
        """Extract dates from JSON and return as JSON string."""
        self.log("=" * 80)
        self.log("[extract_dates] STARTING DATE EXTRACTION")
        self.log("=" * 80)
        
        self.log("[extract_dates] Step 1: Extracting text from input_data")
        input_text = self._extract_text(self.input_data)
        input_text = str(input_text) if input_text is not None else ""
        
        self.log(f"[extract_dates] Step 1 complete - Input text length: {len(input_text)}")
        if input_text:
            self.log(f"[extract_dates] Input text preview (first 500 chars): {input_text[:500]}...")
            if len(input_text) > 500:
                self.log(f"[extract_dates] Input text ending (last 200 chars): ...{input_text[-200:]}")
        
        if not input_text or not input_text.strip():
            self.status = "No input text provided."
            self.log("[extract_dates] ERROR: No input text provided")
            return Message(text="")
        
        self.log("[extract_dates] Step 2: Parsing JSON from text")
        json_data = self._parse_json(input_text)
        
        self.log(f"[extract_dates] Step 2 complete - Parsed JSON type: {type(json_data)}")
        
        if json_data is None:
            self.status = "Failed to parse JSON."
            self.log("[extract_dates] ERROR: Failed to parse JSON - returned None")
            return Message(text="")
        
        if isinstance(json_data, dict):
            self.log(f"[extract_dates] Parsed JSON is dict with {len(json_data)} keys")
            self.log(f"[extract_dates] Parsed JSON keys: {list(json_data.keys())}")
            for key, value in json_data.items():
                self.log(f"[extract_dates]   - Key '{key}': {type(value).__name__} = {str(value)[:100]}")
        elif isinstance(json_data, list):
            self.log(f"[extract_dates] Parsed JSON is list with {len(json_data)} items")
            for i, item in enumerate(json_data[:5]):  # Log first 5 items
                self.log(f"[extract_dates]   - Item {i}: {type(item).__name__} = {str(item)[:100]}")
        else:
            self.log(f"[extract_dates] Parsed JSON type: {type(json_data)}, value: {json_data}")
        
        if json_data == {} or json_data == []:
            self.status = "Parsed JSON is empty."
            self.log("[extract_dates] ERROR: Parsed JSON is empty")
            return Message(text="")
        
        self.log("[extract_dates] Step 3: Extracting dates from parsed JSON")
        dates = self._extract_dates_from_json(json_data)
        
        self.log(f"[extract_dates] Step 3 complete - Extracted dates dict")
        self.log(f"[extract_dates] Dates dict: {dates}")
        self.log(f"[extract_dates] Dates dict keys: {list(dates.keys())}")
        self.log(f"[extract_dates] Dates dict length: {len(dates)}")
        self.log(f"[extract_dates] Dates dict is empty: {not dates}")
        
        if not dates:
            self.status = "No date fields found in JSON."
            self.log("[extract_dates] ERROR: No date fields found in JSON")
            if isinstance(json_data, dict):
                self.log(f"[extract_dates] Available JSON keys: {list(json_data.keys())}")
                self.log(f"[extract_dates] Full JSON structure:")
                for key, value in json_data.items():
                    self.log(f"[extract_dates]   '{key}': {type(value).__name__} = {str(value)[:200]}")
            elif isinstance(json_data, list):
                self.log(f"[extract_dates] JSON is a list with {len(json_data)} items")
            return Message(text="")
        
        self.log("[extract_dates] Step 4: Converting dates dict to JSON string")
        dates_json = json.dumps(dates, indent=2)
        
        self.log(f"[extract_dates] Step 4 complete - Dates JSON string length: {len(dates_json)}")
        self.log(f"[extract_dates] Dates JSON:\n{dates_json}")
        self.status = f"Extracted {len(dates)} date field(s): {', '.join(dates.keys())}"
        
        self.log("=" * 80)
        self.log("[extract_dates] DATE EXTRACTION COMPLETE - SUCCESS")
        self.log("=" * 80)
        
        return Message(text=dates_json)
