import re

from langchain.agents import create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate

from lfx.base.agents.agent import LCToolsAgentComponent
from lfx.inputs.inputs import (
    DataInput,
    HandleInput,
    MessageTextInput,
)
from lfx.schema.data import Data
from lfx.schema.message import Message


class ToolCallingAgentComponent(LCToolsAgentComponent):
    display_name: str = "Tool Calling Agent"
    description: str = "An agent designed to utilize various tools seamlessly within workflows."
    icon = "LangChain"
    name = "ToolCallingAgent"

    inputs = [
        *LCToolsAgentComponent.get_base_inputs(),
        HandleInput(
            name="llm",
            display_name="Language Model",
            input_types=["LanguageModel"],
            required=True,
            info="Language model that the agent utilizes to perform tasks effectively.",
        ),
        MessageTextInput(
            name="system_prompt",
            display_name="System Prompt",
            info="System prompt to guide the agent's behavior.",
            value="You are a helpful assistant that can use tools to answer questions and perform tasks.",
        ),
        DataInput(
            name="chat_history",
            display_name="Chat Memory",
            is_list=True,
            advanced=True,
            info="This input stores the chat history, allowing the agent to remember previous conversations.",
        ),
    ]

    def get_chat_history_data(self) -> list[Data] | None:
        return self.chat_history

    def _extract_system_prompt_text(self):
        """Extract system prompt text, handling both string and Message inputs.
        
        If system_prompt is a Message with template variables, this method
        will access all variables to ensure they are resolved before the agent runs.
        """
        if not hasattr(self, "system_prompt") or not self.system_prompt:
            return None
        
        # If it's a Message object, extract text and check for unresolved variables
        if isinstance(self.system_prompt, Message):
            # Check if Message has a template with variables
            if hasattr(self.system_prompt, "template") and self.system_prompt.template:
                # Extract variable names from template
                variable_pattern = r'\{([^{}]+)\}'
                template_variables = set(re.findall(variable_pattern, self.system_prompt.template))
                
                # Check if variables dict exists and has all required variables
                if hasattr(self.system_prompt, "variables") and self.system_prompt.variables:
                    # Access all template variables to create dependencies
                    # This ensures the component waits for all variables before executing
                    for var_name in template_variables:
                        if var_name in self.system_prompt.variables:
                            # Access the variable to create dependency
                            _ = self.system_prompt.variables[var_name]
                        elif hasattr(self, var_name):
                            # Variable might be passed as component attribute
                            _ = getattr(self, var_name, None)
            
            # Extract text from Message (this will use format_text if template exists)
            if hasattr(self.system_prompt, "text") and self.system_prompt.text:
                return self.system_prompt.text.strip()
            elif hasattr(self.system_prompt, "format_text"):
                # Format the template if it exists
                formatted = self.system_prompt.format_text()
                return formatted.strip() if formatted else None
            else:
                # Fallback to string representation
                return str(self.system_prompt).strip()
        
        # If it's a string, return it directly
        if isinstance(self.system_prompt, str):
            return self.system_prompt.strip()
        
        # Fallback: convert to string
        return str(self.system_prompt).strip() if self.system_prompt else None

    def create_agent_runnable(self):
        messages = []

        # Extract system prompt text, handling Message objects and template variables
        system_prompt_text = self._extract_system_prompt_text()
        
        # Only include system message if system_prompt is provided and not empty
        if system_prompt_text:
            messages.append(("system", "{system_prompt}"))

        messages.extend(
            [
                ("placeholder", "{chat_history}"),
                ("human", "{input}"),
                ("placeholder", "{agent_scratchpad}"),
            ]
        )

        prompt = ChatPromptTemplate.from_messages(messages)
        self.validate_tool_names()
        try:
            return create_tool_calling_agent(self.llm, self.tools or [], prompt)
        except NotImplementedError as e:
            message = f"{self.display_name} does not support tool calling. Please try using a compatible model."
            raise NotImplementedError(message) from e
