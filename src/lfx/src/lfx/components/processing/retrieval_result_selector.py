import re
from typing import Any

from lfx.base.models.chat_result import get_chat_result
from lfx.components.models_and_agents.memory import MemoryComponent
from lfx.custom import Component
from lfx.inputs import HandleInput, MessageTextInput, MultilineInput
from lfx.io import IntInput, Output
from lfx.schema import Data, Message
from lfx.utils.constants import MESSAGE_SENDER_USER


def set_advanced_true(component_input):
    component_input.advanced = True
    return component_input


class RetrievalResultSelectorComponent(Component):
    display_name = "Retrieval Result Selector"
    description = "Uses an LLM to select the best result from multiple retrieval results without rewriting the content."
    icon = "git-branch"
    name = "RetrievalResultSelector"

    memory_inputs = [set_advanced_true(component_input) for component_input in MemoryComponent().inputs]

    inputs = [
        MessageTextInput(
            name="retrieval_results",
            display_name="Retrieval Results",
            info="The formatted retrieval results from a vector search component (e.g., Chroma Search Agent). Results should be formatted with '--- Result N ---' delimiters.",
            required=True,
        ),
        HandleInput(
            name="judge_llm",
            display_name="Judge LLM",
            input_types=["LanguageModel"],
            required=True,
            info="LLM that will evaluate and select the most appropriate retrieval result.",
        ),
        MultilineInput(
            name="selection_criteria",
            display_name="Selection Criteria",
            info="Optional instructions for how the LLM should evaluate and select results. If not provided, defaults to selecting the most relevant result.",
            value="",
            advanced=True,
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
        MessageTextInput(
            name="fallback_index",
            display_name="Fallback Index",
            info="If LLM selection fails, use this result index (0-based). Defaults to 0 (first result).",
            value="0",
            advanced=True,
        ),
    ]

    outputs = [
        Output(
            display_name="Selected Result",
            name="selected_result",
            method="select_result",
            types=["Message"],
        ),
    ]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._selected_index: int | None = None
        self._parsed_results: list[str] = []
        self._chat_history: list[Message] = []

    def _parse_retrieval_results(self, results_text: str) -> list[str]:
        """Parse retrieval results that are formatted with '--- Result N ---' delimiters."""
        if not results_text or not results_text.strip():
            return []

        # Split by the pattern "--- Result N ---" where N is a number
        pattern = r"---\s*Result\s+(\d+)\s*---"
        parts = re.split(pattern, results_text)

        # After splitting, we get: [text_before, num1, text_after_num1, num2, text_after_num2, ...]
        # We want to extract the text parts (odd indices) and pair them with their numbers
        results = []
        for i in range(1, len(parts), 2):
            if i + 1 < len(parts):
                result_num = int(parts[i])
                result_content = parts[i + 1].strip()
                if result_content:
                    results.append(result_content)

        # If no pattern matches, try splitting by double newlines as fallback
        if not results:
            self.log("No '--- Result N ---' pattern found. Trying alternative parsing...")
            # Try splitting by multiple newlines
            parts = re.split(r"\n\n+", results_text.strip())
            results = [part.strip() for part in parts if part.strip()]

        return results

    def _create_system_prompt(self) -> str:
        """Create system prompt for the judge LLM."""
        base_prompt = """You are an expert at evaluating retrieval results and selecting the most relevant one.

Your task is to analyze multiple retrieval results and select the index (0-based) of the result that best answers the user's query from the conversation history.

Important:
- Return ONLY the index number (0, 1, 2, etc.) of the best result
- Do not provide any explanation, reasoning, or rewritten content
- Just return the number
- Consider the full conversation history context when selecting
- If multiple results seem equally relevant, select the first one (lowest index)
- If no result seems relevant, return 0 (first result) as fallback"""

        if self.selection_criteria and self.selection_criteria.strip():
            return f"""{base_prompt}

Additional Selection Criteria:
{self.selection_criteria}"""

        return base_prompt

    async def get_memory_data(self):
        """Retrieve chat history messages from memory."""
        # TODO: This is a temporary fix to avoid message duplication. We should develop a function for this.
        messages = (
            await MemoryComponent(**self.get_base_args())
            .set(
                session_id=self.graph.session_id,
                context_id=self.context_id,
                order="Ascending",
                n_messages=self.n_messages,
            )
            .retrieve_messages()
        )
        return messages

    def _format_chat_history(self, chat_history: list[Message]) -> str:
        """Format chat history for inclusion in the prompt."""
        if not chat_history:
            return ""
        
        formatted_messages = []
        for msg in chat_history:
            sender = getattr(msg, "sender_name", None) or getattr(msg, "sender", "Unknown")
            text = msg.text if hasattr(msg, "text") else str(msg)
            formatted_messages.append(f"{sender}: {text}")
        
        return "\n".join(formatted_messages)

    async def select_result(self) -> Message:
        """Main method to select the best retrieval result."""
        if not self.retrieval_results:
            error_msg = "Retrieval results are required"
            self.status = error_msg
            self.log(f"Validation Error: {error_msg}", "error")
            raise ValueError(error_msg)

        if not self.judge_llm:
            error_msg = "Judge LLM is required"
            self.status = error_msg
            self.log(f"Validation Error: {error_msg}", "error")
            raise ValueError(error_msg)

        try:
            # Parse retrieval results
            results_text = ""
            if isinstance(self.retrieval_results, Message):
                results_text = self.retrieval_results.text
            elif isinstance(self.retrieval_results, str):
                results_text = self.retrieval_results
            else:
                results_text = str(self.retrieval_results)

            self._parsed_results = self._parse_retrieval_results(results_text)

            if not self._parsed_results:
                error_msg = "No results found in retrieval results. Make sure results are formatted with '--- Result N ---' delimiters."
                self.status = error_msg
                self.log(error_msg, "error")
                raise ValueError(error_msg)

            self.log(f"Parsed {len(self._parsed_results)} retrieval results")

            # Retrieve chat history from memory
            try:
                self._chat_history = await self.get_memory_data()
                self.log(f"Retrieved {len(self._chat_history)} chat history messages")
            except Exception as e:
                self.log(f"Could not retrieve chat history: {e}", "warning")
                self._chat_history = []

            # Extract query from the most recent user message in chat history
            query_text = ""
            if self._chat_history:
                # Find the most recent user message
                for msg in reversed(self._chat_history):
                    sender = getattr(msg, "sender", None) or getattr(msg, "sender_name", None)
                    if sender == MESSAGE_SENDER_USER or (hasattr(msg, "text") and msg.text):
                        query_text = msg.text if hasattr(msg, "text") else str(msg)
                        break
                
                if not query_text:
                    # Fallback: use the most recent message
                    most_recent = self._chat_history[-1]
                    query_text = most_recent.text if hasattr(most_recent, "text") else str(most_recent)
            
            if not query_text:
                error_msg = "No query found in chat history. Please ensure chat history contains user messages."
                self.status = error_msg
                self.log(error_msg, "error")
                raise ValueError(error_msg)

            # Prepare prompt for judge LLM
            system_prompt = self._create_system_prompt()

            # Format results for the LLM
            results_for_llm = []
            for i, result in enumerate(self._parsed_results):
                # Truncate very long results to avoid token limits
                preview = result[:1000] + "..." if len(result) > 1000 else result
                results_for_llm.append(f"Result {i}:\n{preview}")

            # Build user message with conversation history
            user_message_parts = []
            
            # Add conversation history
            if self._chat_history:
                history_text = self._format_chat_history(self._chat_history)
                user_message_parts.append(f"Conversation History:\n{history_text}\n")
            
            user_message_parts.append(f"\nRetrieval Results:\n{chr(10).join(results_for_llm)}")
            user_message_parts.append("\nBased on the conversation history above, select the index (0-based) of the result that best answers the most recent user query.\nReturn ONLY the index number:")

            user_message_content = "\n".join(user_message_parts)

            self.log("Requesting result selection from judge LLM...")
            self.status = "Judge LLM analyzing results..."

            # Call the judge LLM
            user_message = Message(text=user_message_content)
            response = get_chat_result(
                runnable=self.judge_llm,
                input_value=user_message,
                system_message=system_prompt,
            )

            # Parse response - handle different response types
            if isinstance(response, Message):
                response_text = response.text
            elif hasattr(response, "content"):
                response_text = str(response.content)
            else:
                response_text = str(response)

            selected_index = self._parse_judge_response(response_text.strip())

            self._selected_index = selected_index
            self.log(f"Judge LLM selected index: {selected_index}")

            # Return the actual result content
            if 0 <= selected_index < len(self._parsed_results):
                selected_content = self._parsed_results[selected_index]
                self.status = f"Selected result {selected_index}"
                return Message(text=selected_content)
            else:
                # Index out of bounds, use fallback
                fallback_idx = int(self.fallback_index) if self.fallback_index else 0
                if 0 <= fallback_idx < len(self._parsed_results):
                    self._selected_index = fallback_idx
                    self.log(f"Selected index out of bounds, using fallback index: {fallback_idx}", "warning")
                    selected_content = self._parsed_results[fallback_idx]
                    self.status = f"Selected result {fallback_idx} (fallback)"
                    return Message(text=selected_content)
                else:
                    # Last resort: return first result
                    self._selected_index = 0
                    self.log("Using default fallback: first result", "warning")
                    selected_content = self._parsed_results[0]
                    self.status = "Selected result 0 (default fallback)"
                    return Message(text=selected_content)

        except Exception as e:
            error_msg = f"Error during result selection: {type(e).__name__} - {e!s}"
            self.log(f"{error_msg}", "error")
            self.status = error_msg

            # Fallback to default index and return the result content
            try:
                fallback_idx = int(self.fallback_index) if self.fallback_index else 0
                if 0 <= fallback_idx < len(self._parsed_results):
                    self._selected_index = fallback_idx
                    self.log(f"Using fallback index: {fallback_idx}", "warning")
                    selected_content = self._parsed_results[fallback_idx]
                    return Message(text=selected_content)
            except (ValueError, TypeError):
                pass

            # Last resort: return first result
            if self._parsed_results:
                self._selected_index = 0
                self.log("Using default fallback: first result", "warning")
                selected_content = self._parsed_results[0]
                return Message(text=selected_content)
            else:
                error_msg = "No results available to return"
                self.log(error_msg, "error")
                return Message(text=error_msg)

    def _parse_judge_response(self, response_content: str) -> int:
        """Parse the judge's response to extract result index."""
        try:
            # Extract first number from response
            cleaned_response = "".join(filter(str.isdigit, response_content.strip()))
            if not cleaned_response:
                self.log(
                    f"Judge LLM response was non-numeric: '{response_content}'. Defaulting to index 0.",
                    "warning",
                )
                return 0

            selected_index = int(cleaned_response)

            if 0 <= selected_index < len(self._parsed_results):
                return selected_index

            log_msg = (
                f"Judge LLM selected index {selected_index} is out of bounds "
                f"(0-{len(self._parsed_results) - 1}). Defaulting to index 0."
            )
            self.log(log_msg, "warning")
            return 0

        except ValueError:
            self.log(
                f"Could not parse judge LLM response to integer: '{response_content}'. Defaulting to index 0.",
                "warning",
            )
            return 0
        except Exception as e:
            self.log(f"Error parsing judge response '{response_content}': {e!s}. Defaulting to index 0.", "error")
            return 0

