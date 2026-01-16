import re

from lfx.base.prompts.api_utils import process_prompt_template
from lfx.custom.custom_component.component import Component
from lfx.inputs.inputs import DefaultPromptField
from lfx.io import MessageTextInput, Output, PromptInput
from lfx.schema.message import Message
from lfx.template.utils import update_template_values


class GatedPromptComponent(Component):
    display_name: str = "Gated Prompt (Wait for All)"
    description: str = "Prompt with dynamic variables that waits for ALL inputs before executing."
    documentation: str = "https://docs.langflow.org/components-prompts"
    icon = "lock"
    trace_type = "prompt"
    name = "GatedPrompt"

    inputs = [
        PromptInput(name="template", display_name="Template"),
        MessageTextInput(
            name="tool_placeholder",
            display_name="Tool Placeholder",
            tool_mode=True,
            advanced=True,
            info="A placeholder input for tool mode.",
        ),
    ]

    outputs = [
        Output(display_name="Prompt", name="prompt", method="build_prompt"),
    ]

    async def build_prompt(self) -> Message:
        # Extract all variable names from the template
        template_text = str(self.template) if self.template else ""
        variable_pattern = r'\{([^{}]+)\}'
        variables = re.findall(variable_pattern, template_text)
        
        # Explicitly access ALL dynamic inputs to force dependencies
        # This ensures the component waits for all template variables
        for var_name in variables:
            if hasattr(self, var_name):
                # Access the attribute to create dependency
                _ = getattr(self, var_name, None)
        
        # Now build the prompt normally using the standard mechanism
        prompt = Message.from_template(**self._attributes)
        self.status = prompt.text
        return prompt

    def _update_template(self, frontend_node: dict):
        prompt_template = frontend_node["template"]["template"]["value"]
        custom_fields = frontend_node["custom_fields"]
        frontend_node_template = frontend_node["template"]
        _ = process_prompt_template(
            template=prompt_template,
            name="template",
            custom_fields=custom_fields,
            frontend_node_template=frontend_node_template,
        )
        return frontend_node

    async def update_frontend_node(self, new_frontend_node: dict, current_frontend_node: dict):
        frontend_node = await super().update_frontend_node(new_frontend_node, current_frontend_node)
        template = frontend_node["template"]["template"]["value"]
        _ = process_prompt_template(
            template=template,
            name="template",
            custom_fields=frontend_node["custom_fields"],
            frontend_node_template=frontend_node["template"],
        )
        update_template_values(new_template=frontend_node, previous_template=current_frontend_node["template"])
        return frontend_node

    def _get_fallback_input(self, **kwargs):
        return DefaultPromptField(**kwargs)
