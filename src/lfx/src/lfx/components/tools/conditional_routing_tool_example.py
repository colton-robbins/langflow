from lfx.custom.custom_component.component import Component
from lfx.io import BoolInput, MessageInput, MessageTextInput, Output, StrInput
from lfx.schema.message import Message


class ConditionalRoutingToolComponent(Component):
    display_name = "Conditional Routing Tool"
    description = "A tool that executes logic and routes to different paths based on success or failure, bypassing the calling agent."
    documentation: str = "https://docs.langflow.org/"
    icon = "split"
    name = "ConditionalRoutingTool"

    inputs = [
        MessageInput(
            name="input_message",
            display_name="Input Message",
            info="The input message to process. Can come from agent or another component.",
            required=True,
        ),
        MessageTextInput(
            name="success_keyword",
            display_name="Success Keyword",
            info="Keyword to check for success (e.g., 'completed', 'success', 'true'). Case-insensitive.",
            value="success",
            required=False,
        ),
        BoolInput(
            name="simulate_execution",
            display_name="Simulate Execution",
            info="If true, simulates tool execution for testing. In production, replace with actual logic.",
            value=True,
            advanced=True,
        ),
        StrInput(
            name="simulated_result",
            display_name="Simulated Result",
            info="For testing: The simulated result to return. Include success_keyword for success path.",
            value="Operation completed successfully",
            advanced=True,
        ),
    ]

    outputs = [
        Output(
            display_name="Success Path",
            name="success_output",
            method="handle_success",
            group_outputs=False,
            tool_mode=True,
        ),
        Output(
            display_name="Failure Path",
            name="failure_output",
            method="handle_failure",
            group_outputs=False,
            tool_mode=True,
        ),
        Output(
            display_name="Tool Result",
            name="tool_output",
            method="handle_tool_result",
            group_outputs=False,
            tool_mode=True,
        ),
    ]

    def execute_tool_logic(self) -> tuple[bool, str]:
        if self.simulate_execution:
            result = self.simulated_result
            is_success = self.success_keyword.lower() in result.lower()
            return is_success, result
        
        try:
            input_text = self.input_message.text if hasattr(self.input_message, 'text') else str(self.input_message)
            
            result = f"Processed: {input_text}"
            
            is_success = self.success_keyword.lower() in input_text.lower()
            
            return is_success, result
            
        except Exception as e:
            return False, f"Error during execution: {str(e)}"

    def handle_success(self) -> Message:
        is_success, result = self.execute_tool_logic()
        
        if is_success:
            self.stop("failure_output")
            self.stop("tool_output")
            
            self.graph.exclude_branch_conditionally(self._id, "failure_output")
            self.graph.exclude_branch_conditionally(self._id, "tool_output")
            
            self.status = "Success - routing to success path"
            return Message(text=result)
        
        self.stop("success_output")
        return Message(text="")

    def handle_failure(self) -> Message:
        is_success, result = self.execute_tool_logic()
        
        if not is_success:
            self.stop("success_output")
            self.stop("tool_output")
            
            self.graph.exclude_branch_conditionally(self._id, "success_output")
            self.graph.exclude_branch_conditionally(self._id, "tool_output")
            
            self.status = "Failure - routing to failure path"
            return Message(text=result)
        
        self.stop("failure_output")
        return Message(text="")

    def handle_tool_result(self) -> Message:
        is_success, result = self.execute_tool_logic()
        
        self.status = f"Tool executed: {'success' if is_success else 'failure'}"
        return Message(text=result)
