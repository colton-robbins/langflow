from unittest.mock import MagicMock, PropertyMock, patch

import pytest
from lfx.components.tools.conditional_routing_tool_example import ConditionalRoutingToolComponent
from lfx.schema.message import Message

from tests.base import ComponentTestBaseWithoutClient


class TestConditionalRoutingToolComponent(ComponentTestBaseWithoutClient):
    """Test cases for ConditionalRoutingToolComponent."""

    @pytest.fixture
    def component_class(self):
        """Return the component class to test."""
        return ConditionalRoutingToolComponent

    @pytest.fixture
    def default_kwargs(self):
        """Return the default kwargs for the component."""
        return {
            "input_message": Message(text="test input"),
            "success_keyword": "success",
            "simulate_execution": True,
            "simulated_result": "Operation completed successfully",
        }

    @pytest.fixture
    def file_names_mapping(self):
        """Return an empty list since this is a new component."""
        return []

    async def test_component_initialization(self, component_class, default_kwargs):
        """Test proper initialization of ConditionalRoutingToolComponent."""
        component = await self.component_setup(component_class, default_kwargs)
        assert component.display_name == "Conditional Routing Tool"
        assert "executes logic and routes to different paths" in component.description
        assert component.name == "ConditionalRoutingTool"
        assert component.icon == "split"

    async def test_inputs_configuration(self, component_class, default_kwargs):
        """Test that inputs are properly configured."""
        component = await self.component_setup(component_class, default_kwargs)
        expected_inputs = [
            "input_message",
            "success_keyword",
            "simulate_execution",
            "simulated_result",
        ]

        assert len(component.inputs) == len(expected_inputs)
        input_names = [inp.name for inp in component.inputs]

        for expected_input in expected_inputs:
            assert expected_input in input_names

    async def test_outputs_configuration(self, component_class, default_kwargs):
        """Test that outputs are properly configured."""
        component = await self.component_setup(component_class, default_kwargs)
        assert len(component.outputs) == 3
        output_names = [out.name for out in component.outputs]
        assert "success_output" in output_names
        assert "failure_output" in output_names
        assert "tool_output" in output_names

    async def test_execute_tool_logic_success(self, component_class, default_kwargs):
        """Test execute_tool_logic when result contains success keyword."""
        component = await self.component_setup(component_class, default_kwargs)
        component.simulate_execution = True
        component.simulated_result = "Operation completed successfully"
        component.success_keyword = "success"

        is_success, result = component.execute_tool_logic()

        assert is_success is True
        assert result == "Operation completed successfully"

    async def test_execute_tool_logic_failure(self, component_class, default_kwargs):
        """Test execute_tool_logic when result does not contain success keyword."""
        component = await self.component_setup(component_class, default_kwargs)
        component.simulate_execution = True
        component.simulated_result = "Operation failed with error"
        component.success_keyword = "success"

        is_success, result = component.execute_tool_logic()

        assert is_success is False
        assert result == "Operation failed with error"

    async def test_execute_tool_logic_non_simulated_success(self, component_class, default_kwargs):
        """Test execute_tool_logic in non-simulated mode with success keyword in input."""
        component = await self.component_setup(component_class, default_kwargs)
        component.simulate_execution = False
        component.input_message = Message(text="Please process this successfully")
        component.success_keyword = "success"

        is_success, result = component.execute_tool_logic()

        assert is_success is True
        assert "Processed: Please process this successfully" in result

    async def test_execute_tool_logic_non_simulated_failure(self, component_class, default_kwargs):
        """Test execute_tool_logic in non-simulated mode without success keyword."""
        component = await self.component_setup(component_class, default_kwargs)
        component.simulate_execution = False
        component.input_message = Message(text="Please process this request")
        component.success_keyword = "success"

        is_success, result = component.execute_tool_logic()

        assert is_success is False
        assert "Processed: Please process this request" in result

    async def test_handle_success_when_successful(self, component_class, default_kwargs):
        """Test handle_success when operation is successful."""
        component = await self.component_setup(component_class, default_kwargs)
        component.simulate_execution = True
        component.simulated_result = "Operation succeeded"
        component.success_keyword = "succeed"

        mock_graph = MagicMock()
        mock_graph.exclude_branch_conditionally = MagicMock()

        with (
            patch.object(component, "stop") as mock_stop,
            patch.object(type(component), "graph", new_callable=PropertyMock, return_value=mock_graph),
            patch.object(component, "_id", "test_id"),
        ):
            result = component.handle_success()

            assert isinstance(result, Message)
            assert result.text == "Operation succeeded"
            assert component.status == "Success - routing to success path"
            assert mock_stop.call_count == 2
            mock_stop.assert_any_call("failure_output")
            mock_stop.assert_any_call("tool_output")
            assert mock_graph.exclude_branch_conditionally.call_count == 2

    async def test_handle_success_when_not_successful(self, component_class, default_kwargs):
        """Test handle_success when operation is not successful."""
        component = await self.component_setup(component_class, default_kwargs)
        component.simulate_execution = True
        component.simulated_result = "Operation failed"
        component.success_keyword = "success"

        with patch.object(component, "stop") as mock_stop:
            result = component.handle_success()

            assert isinstance(result, Message)
            assert result.text == ""
            mock_stop.assert_called_once_with("success_output")

    async def test_handle_failure_when_failed(self, component_class, default_kwargs):
        """Test handle_failure when operation fails."""
        component = await self.component_setup(component_class, default_kwargs)
        component.simulate_execution = True
        component.simulated_result = "Operation failed with error"
        component.success_keyword = "success"

        mock_graph = MagicMock()
        mock_graph.exclude_branch_conditionally = MagicMock()

        with (
            patch.object(component, "stop") as mock_stop,
            patch.object(type(component), "graph", new_callable=PropertyMock, return_value=mock_graph),
            patch.object(component, "_id", "test_id"),
        ):
            result = component.handle_failure()

            assert isinstance(result, Message)
            assert result.text == "Operation failed with error"
            assert component.status == "Failure - routing to failure path"
            assert mock_stop.call_count == 2
            mock_stop.assert_any_call("success_output")
            mock_stop.assert_any_call("tool_output")
            assert mock_graph.exclude_branch_conditionally.call_count == 2

    async def test_handle_failure_when_successful(self, component_class, default_kwargs):
        """Test handle_failure when operation is successful."""
        component = await self.component_setup(component_class, default_kwargs)
        component.simulate_execution = True
        component.simulated_result = "Operation succeeded"
        component.success_keyword = "succeed"

        with patch.object(component, "stop") as mock_stop:
            result = component.handle_failure()

            assert isinstance(result, Message)
            assert result.text == ""
            mock_stop.assert_called_once_with("failure_output")

    async def test_handle_tool_result(self, component_class, default_kwargs):
        """Test handle_tool_result returns appropriate result."""
        component = await self.component_setup(component_class, default_kwargs)
        component.simulate_execution = True
        component.simulated_result = "Tool result"
        component.success_keyword = "success"

        result = component.handle_tool_result()

        assert isinstance(result, Message)
        assert result.text == "Tool result"
        assert "Tool executed: failure" in component.status

    async def test_case_insensitive_success_keyword(self, component_class, default_kwargs):
        """Test that success keyword matching is case-insensitive."""
        component = await self.component_setup(component_class, default_kwargs)
        component.simulate_execution = True
        component.simulated_result = "Operation SUCCEEDED"
        component.success_keyword = "success"

        is_success, result = component.execute_tool_logic()

        assert is_success is True

    async def test_success_keyword_partial_match(self, component_class, default_kwargs):
        """Test that success keyword matching works for partial matches."""
        component = await self.component_setup(component_class, default_kwargs)
        component.simulate_execution = True
        component.simulated_result = "The operation was successful and completed"
        component.success_keyword = "success"

        is_success, result = component.execute_tool_logic()

        assert is_success is True

    async def test_input_message_as_string(self, component_class, default_kwargs):
        """Test execute_tool_logic handles input_message that doesn't have text attribute."""
        component = await self.component_setup(component_class, default_kwargs)
        component.simulate_execution = False
        component.input_message = "raw string input"
        component.success_keyword = "success"

        is_success, result = component.execute_tool_logic()

        assert is_success is False
        assert "Processed: raw string input" in result

    async def test_multiple_outputs_independent_execution(self, component_class, default_kwargs):
        """
        Test that each output method can execute independently.
        
        This verifies the conditional routing pattern where only one
        output path should have meaningful results.
        """
        component = await self.component_setup(component_class, default_kwargs)
        component.simulate_execution = True
        component.simulated_result = "Operation succeeded"
        component.success_keyword = "succeed"

        mock_graph = MagicMock()
        mock_graph.exclude_branch_conditionally = MagicMock()

        with (
            patch.object(component, "stop"),
            patch.object(type(component), "graph", new_callable=PropertyMock, return_value=mock_graph),
            patch.object(component, "_id", "test_id"),
        ):
            success_result = component.handle_success()
            assert success_result.text == "Operation succeeded"

            failure_result = component.handle_failure()
            assert failure_result.text == ""

    async def test_error_handling_in_execute_logic(self, component_class, default_kwargs):
        """Test that errors in execute_tool_logic are handled gracefully."""
        component = await self.component_setup(component_class, default_kwargs)
        component.simulate_execution = False

        with patch.object(component.input_message, "text", new_callable=PropertyMock) as mock_text:
            mock_text.side_effect = Exception("Test error")

            is_success, result = component.execute_tool_logic()

            assert is_success is False
            assert "Error during execution" in result
            assert "Test error" in result
