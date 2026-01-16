from typing import Any

from lfx.custom import Component
from lfx.io import BoolInput, HandleInput, MessageInput, MessageTextInput, MultilineInput, Output, TableInput
from lfx.schema.message import Message
from lfx.schema.table import EditMode


class SmartRouterComponent(Component):
    display_name = "Smart Router"
    description = "Routes an input message using LLM-based categorization."
    icon = "route"
    name = "SmartRouter"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._matched_category = None
        self._categorization_done = False

    def __getattr__(self, name: str):
        """Dynamically handle process_category_N method calls."""
        if name.startswith("process_category_"):
            try:
                category_index = int(name.split("_")[-1])
                return lambda: self._process_category(category_index)
            except (ValueError, IndexError):
                pass
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    inputs = [
        HandleInput(
            name="llm",
            display_name="Language Model",
            info="LLM to use for categorization.",
            input_types=["LanguageModel"],
            required=True,
        ),
        MessageTextInput(
            name="input_text",
            display_name="Input",
            info="The primary text input for the operation.",
            required=True,
        ),
        TableInput(
            name="routes",
            display_name="Routes",
            info=(
                "Define the categories for routing. Each row should have a route/category name "
                "and optionally a custom output value."
            ),
            table_schema=[
                {
                    "name": "route_category",
                    "display_name": "Route Name",
                    "type": "str",
                    "description": "Name for the route (used for both output name and category matching)",
                    "edit_mode": EditMode.INLINE,
                },
                {
                    "name": "route_description",
                    "display_name": "Route Description",
                    "type": "str",
                    "description": "Description of when this route should be used (helps LLM understand the category)",
                    "default": "",
                    "edit_mode": EditMode.POPOVER,
                },
                {
                    "name": "output_value",
                    "display_name": "Route Message (Optional)",
                    "type": "str",
                    "description": (
                        "Optional message to send when this route is matched."
                        "Leave empty to pass through the original input text."
                    ),
                    "default": "",
                    "edit_mode": EditMode.POPOVER,
                },
            ],
            value=[
                {
                    "route_category": "Positive",
                    "route_description": "Positive feedback, satisfaction, or compliments",
                    "output_value": "",
                },
                {
                    "route_category": "Negative",
                    "route_description": "Complaints, issues, or dissatisfaction",
                    "output_value": "",
                },
            ],
            real_time_refresh=True,
            required=True,
        ),
        MessageInput(
            name="message",
            display_name="Override Output",
            info=(
                "Optional override message that will replace both the Input and Output Value "
                "for all routes when filled."
            ),
            required=False,
            advanced=True,
        ),
        BoolInput(
            name="enable_else_output",
            display_name="Include Else Output",
            info="Include an Else output for cases that don't match any route.",
            value=False,
            advanced=True,
        ),
        MultilineInput(
            name="custom_prompt",
            display_name="Additional Instructions",
            info=(
                "Additional instructions for LLM-based categorization. "
                "These will be added to the base prompt. "
                "Use {input_text} for the input text and {routes} for the available categories."
            ),
            advanced=True,
        ),
    ]

    outputs: list[Output] = []

    def update_outputs(self, frontend_node: dict, field_name: str, field_value: Any) -> dict:
        """Create a dynamic output for each category in the categories table."""
        if field_name in {"routes", "enable_else_output"}:
            frontend_node["outputs"] = []

            # Get the routes data - either from field_value (if routes field) or from component state
            routes_data = field_value if field_name == "routes" else getattr(self, "routes", [])

            # Create an output for each category with a unique method name
            for i, row in enumerate(routes_data):
                route_category = row.get("route_category", f"Category {i + 1}")
                method_name = f"process_category_{i}"
                
                # Add output with unique method name (method will be handled by __getattr__)
                frontend_node["outputs"].append(
                    Output(
                        display_name=route_category,
                        name=f"category_{i + 1}_result",
                        method=method_name,
                        group_outputs=True,
                    )
                )
            # Add default output only if enabled
            if field_name == "enable_else_output":
                enable_else = field_value
            else:
                enable_else = getattr(self, "enable_else_output", False)

            if enable_else:
                frontend_node["outputs"].append(
                    Output(display_name="Else", name="default_result", method="default_response", group_outputs=True)
                )
        return frontend_node

    def _extract_text_from_response(self, response: Any) -> str:
        """Extract text from various LLM response formats."""
        if response is None:
            return ""
        
        # Try common attributes
        if hasattr(response, "content"):
            text = response.content
        elif hasattr(response, "text"):
            text = response.text
        elif hasattr(response, "output"):
            text = response.output
        elif isinstance(response, dict):
            # Try common dict keys
            text = (
                response.get("content")
                or response.get("text")
                or response.get("output")
                or response.get("message")
                or response.get("log")
            )
            # If still None, try to extract from nested structures
            if text is None:
                # Check for nested content (e.g., return_values.log)
                if "return_values" in response and isinstance(response["return_values"], dict):
                    text = response["return_values"].get("log") or response["return_values"].get("output")
                # If still None, convert dict to string and extract first meaningful text
                if text is None:
                    text = str(response)
        elif isinstance(response, (list, tuple)) and len(response) > 0:
            # Try to extract from first element
            text = self._extract_text_from_response(response[0])
        else:
            text = str(response)
        
        # Clean up the text
        if text is None:
            text = ""
        else:
            text = str(text).strip().strip('"').strip("'")
        
        return text

    def _normalize_text(self, text: str) -> str:
        """Normalize text for comparison - remove extra whitespace, quotes, etc."""
        if not text:
            return ""
        # Remove quotes, normalize whitespace
        normalized = text.strip().strip('"').strip("'").strip()
        # Replace multiple spaces with single space
        normalized = " ".join(normalized.split())
        return normalized.lower()

    def _find_matching_category(self, categorization: str, categories: list) -> int | None:
        """Find matching category with exact and fuzzy matching."""
        if not categorization:
            return None
        
        categorization_normalized = self._normalize_text(categorization)
        
        # First pass: exact case-insensitive match after normalization
        for i, category in enumerate(categories):
            route_category = category.get("route_category", "")
            if not route_category:
                continue
            
            route_category_normalized = self._normalize_text(route_category)
            
            self.status = (
                f"Comparing '{categorization}' (normalized: '{categorization_normalized}') "
                f"with category {i + 1}: route_category='{route_category}' "
                f"(normalized: '{route_category_normalized}')"
            )
            
            # Exact match after normalization
            if categorization_normalized == route_category_normalized:
                self.status = f"EXACT MATCH FOUND! Category {i + 1} ('{route_category}') matched with '{categorization}'"
                return i
        
        # Second pass: check if categorization contains route_category or vice versa
        # Only do this if exact match failed, and be more conservative
        for i, category in enumerate(categories):
            route_category = category.get("route_category", "")
            if not route_category:
                continue
            
            route_category_normalized = self._normalize_text(route_category)
            
            # Only match if one is clearly contained in the other (avoid false positives)
            # Check if route category is contained in categorization (e.g., "other" in "other stuff")
            if route_category_normalized and route_category_normalized in categorization_normalized:
                # Make sure it's not just a partial word match
                # Check if it's a whole word or exact match
                if categorization_normalized == route_category_normalized or \
                   categorization_normalized.startswith(route_category_normalized + " ") or \
                   categorization_normalized.endswith(" " + route_category_normalized) or \
                   (" " + route_category_normalized + " ") in categorization_normalized:
                    self.status = f"PARTIAL MATCH FOUND! Category {i + 1} ('{route_category}') matched (contained in '{categorization}')"
                    return i
            
            # Check if categorization is contained in route category
            if categorization_normalized and categorization_normalized in route_category_normalized:
                if route_category_normalized == categorization_normalized or \
                   route_category_normalized.startswith(categorization_normalized + " ") or \
                   route_category_normalized.endswith(" " + categorization_normalized) or \
                   (" " + categorization_normalized + " ") in route_category_normalized:
                    self.status = f"PARTIAL MATCH FOUND! Category {i + 1} ('{route_category}') matched (contains '{categorization}')"
                    return i
        
        return None

    def _perform_categorization(self) -> int | None:
        """Perform LLM categorization once and cache the result. Returns matched category index or None."""
        if self._categorization_done:
            return self._matched_category
            
        self._categorization_done = True
        self._matched_category = None

        # Get categories and input text
        categories = getattr(self, "routes", [])
        input_text = getattr(self, "input_text", "")

        # Find the matching category using LLM-based categorization
        matched_category = None
        llm = getattr(self, "llm", None)

        if llm and categories:
            # Create prompt for categorization
            category_info = []
            for i, category in enumerate(categories):
                cat_name = category.get("route_category", f"Category {i + 1}")
                cat_desc = category.get("route_description", "")
                if cat_desc and cat_desc.strip():
                    category_info.append(f'"{cat_name}": {cat_desc}')
                else:
                    category_info.append(f'"{cat_name}"')

            categories_text = "\n".join([f"- {info}" for info in category_info if info])

            # Create base prompt
            base_prompt = (
                f"You are a text classifier. Given the following text and categories, "
                f"determine which category best matches the text.\n\n"
                f'Text to classify: "{input_text}"\n\n'
                f"Available categories:\n{categories_text}\n\n"
                f"Respond with ONLY the exact category name that best matches the text. "
                f'If none match well, respond with "NONE".\n\n'
                f"Category:"
            )

            # Use custom prompt as additional instructions if provided
            custom_prompt = getattr(self, "custom_prompt", "")
            if custom_prompt and custom_prompt.strip():
                # Format custom prompt with variables (support both {input_text} and {user_input})
                simple_routes = ", ".join(
                    [f'"{cat.get("route_category", f"Category {i + 1}")}"' for i, cat in enumerate(categories)]
                )
                
                # Check if custom prompt looks like a complete prompt (contains role tags, structured format)
                # If so, use it as the main prompt instead of additional instructions
                custom_lower = custom_prompt.lower()
                is_complete_prompt = (
                    "<role>" in custom_lower 
                    or "<categories>" in custom_lower 
                    or "<rules>" in custom_lower
                    or "<output>" in custom_lower
                    or custom_prompt.count("\n") > 5  # Multi-line structured prompt
                )
                
                # Replace placeholders safely - support both {input_text} and {user_input}
                # Use string replacement to avoid issues with XML-style tags
                formatted_custom = custom_prompt.replace("{user_input}", input_text)
                formatted_custom = formatted_custom.replace("{input_text}", input_text)
                formatted_custom = formatted_custom.replace("{routes}", simple_routes)
                
                if is_complete_prompt:
                    self.status = "Using custom prompt as main prompt"
                    prompt = formatted_custom
                else:
                    self.status = "Using custom prompt as additional instructions"
                    prompt = f"{base_prompt}\n\nAdditional Instructions:\n{formatted_custom}"
            else:
                self.status = "Using default prompt for LLM categorization"
                prompt = base_prompt

            self.status = f"Prompt sent to LLM:\n{prompt}"

            try:
                # Use the LLM to categorize
                if hasattr(llm, "invoke"):
                    response = llm.invoke(prompt)
                    categorization = self._extract_text_from_response(response)
                else:
                    response = llm(prompt)
                    categorization = self._extract_text_from_response(response)

                self.status = f"LLM response: '{categorization}'"

                # Find matching category based on LLM response
                matched_category = self._find_matching_category(categorization, categories)

                if matched_category is None:
                    self.status = (
                        f"No match found for '{categorization}'. Available categories: "
                        f"{[category.get('route_category', '') for category in categories]}"
                    )

            except RuntimeError as e:
                self.status = f"Error in LLM categorization: {e!s}"
        else:
            self.status = "No LLM provided for categorization"

        self._matched_category = matched_category
        return matched_category

    def _process_category(self, category_index: int) -> Message:
        """Process a specific category and return message if it matches."""
        # Perform categorization (will use cached result if already done)
        matched_category = self._perform_categorization()
        
        # Get categories and input text
        categories = getattr(self, "routes", [])
        input_text = getattr(self, "input_text", "")
        
        # Check if this category is the matched one
        if matched_category == category_index:
            route_category = categories[category_index].get("route_category", f"Category {category_index + 1}")
            self.status = f"Categorized as {route_category}"
            
            # Check if there's an override output (takes precedence over everything)
            override_output = getattr(self, "message", None)
            if (
                override_output
                and hasattr(override_output, "text")
                and override_output.text
                and str(override_output.text).strip()
            ):
                return Message(text=str(override_output.text))
            if override_output and isinstance(override_output, str) and override_output.strip():
                return Message(text=str(override_output))
            
            # Check if there's a custom output value for this category
            custom_output = categories[category_index].get("output_value", "")
            if custom_output and str(custom_output).strip() and str(custom_output).strip().lower() != "none":
                return Message(text=str(custom_output))
            
            # Use input as default output
            return Message(text=input_text)
        
        # This category doesn't match, return empty
        self.status = f"Category {category_index + 1} does not match"
        return Message(text="")

    def default_response(self) -> Message:
        """Handle the else case when no conditions match."""
        # Check if else output is enabled
        enable_else = getattr(self, "enable_else_output", False)
        if not enable_else:
            self.status = "Else output is disabled"
            return Message(text="")

        # Perform categorization (will use cached result if already done)
        matched_category = self._perform_categorization()
        
        input_text = getattr(self, "input_text", "")

        # If a category matched, don't output from else
        if matched_category is not None:
            self.status = f"Match found (Category {matched_category + 1}), stopping else output"
            return Message(text="")

        # No case matches, check for override output first, then use input as default
        override_output = getattr(self, "message", None)
        if (
            override_output
            and hasattr(override_output, "text")
            and override_output.text
            and str(override_output.text).strip()
        ):
            self.status = "Routed to Else (no match) - using override output"
            return Message(text=str(override_output.text))
        if override_output and isinstance(override_output, str) and override_output.strip():
            self.status = "Routed to Else (no match) - using override output"
            return Message(text=str(override_output))
        self.status = "Routed to Else (no match) - using input as default"
        return Message(text=input_text)
