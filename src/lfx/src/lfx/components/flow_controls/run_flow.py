from datetime import datetime
from typing import Any

from lfx.base.tools.run_flow import RunFlowBaseComponent
from lfx.inputs.inputs import HandleInput
from lfx.log.logger import logger
from lfx.schema.data import Data
from lfx.schema.dotdict import dotdict
from lfx.schema.message import Message


def extract_text_from_input(input_value: Any) -> str:
    """Extract text from any input type."""
    if input_value is None:
        return ""
    
    if isinstance(input_value, str):
        return input_value
    
    if isinstance(input_value, Message):
        return input_value.text or str(input_value)
    
    if isinstance(input_value, Data):
        if hasattr(input_value, "data") and isinstance(input_value.data, dict):
            # Check various text fields in data
            for key in ["text", "content", "page_content"]:
                if key in input_value.data:
                    return str(input_value.data[key])
            # If no text field found, convert entire data to string
            return str(input_value.data)
        return str(input_value)
    
    if isinstance(input_value, dict):
        # Handle dict objects
        for key in ["text", "content", "page_content"]:
            if key in input_value:
                return str(input_value[key])
        return str(input_value)
    
    return str(input_value) if input_value is not None else ""


class RunFlowComponent(RunFlowBaseComponent):
    display_name = "Run Flow"
    description = (
        "Executes another flow from within the same project. Can also be used as a tool for agents."
        " \n **Select a Flow to use the tool mode**"
    )
    documentation: str = "https://docs.langflow.org/run-flow"
    beta = True
    name = "RunFlow"
    icon = "Workflow"

    # Override default_keys to include our custom inputs
    default_keys = [
        "code",
        "_type",
        "flow_name_selected",
        "flow_id_selected",
        "session_id",
        "cache_flow",
        "user_input",
        "escalation_trigger",
    ]

    inputs = [
        HandleInput(
            name="user_input",
            display_name="User Input",
            input_types=["Message", "Data", "str"],
            info="The user input string to pass to the target flow's chat input.",
        ),
        HandleInput(
            name="escalation_trigger",
            display_name="Escalation Trigger",
            input_types=["Message", "Data", "str"],
            info="The escalation trigger from a router node. Both inputs must be present to trigger the flow.",
        ),
        *RunFlowBaseComponent.get_base_inputs(),
    ]
    outputs = RunFlowBaseComponent.get_base_outputs()

    def __init__(self, *args, **kwargs):
        """Initialize the component and ensure _flow_run_inputs exists."""
        super().__init__(*args, **kwargs)
        # Initialize _flow_run_inputs to ensure it exists before _pre_run_setup is called
        if not hasattr(self, "_flow_run_inputs"):
            self._flow_run_inputs = []

    async def update_build_config(
        self,
        build_config: dotdict,
        field_value: Any,
        field_name: str | None = None,
    ):
        missing_keys = [key for key in self.default_keys if key not in build_config]
        for key in missing_keys:
            if key == "flow_name_selected":
                build_config[key] = {"options": [], "options_metadata": [], "value": None}
            elif key == "flow_id_selected":
                build_config[key] = {"value": None}
            elif key == "cache_flow":
                build_config[key] = {"value": False}
            elif key in {"user_input", "escalation_trigger"}:
                # Initialize HandleInput fields - preserve existing value if present
                if key not in build_config:
                    # Get the input definition from self.inputs
                    input_def = next((inp for inp in self.inputs if inp.name == key), None)
                    if input_def:
                        # Convert Input to dict using model_dump or to_dict
                        if hasattr(input_def, "model_dump"):
                            build_config[key] = input_def.model_dump(by_alias=True, exclude_none=True)
                        elif hasattr(input_def, "to_dict"):
                            build_config[key] = input_def.to_dict()
                        else:
                            # Fallback: create dict from input attributes
                            build_config[key] = {
                                "name": input_def.name,
                                "display_name": getattr(input_def, "display_name", input_def.name),
                                "info": getattr(input_def, "info", ""),
                                "input_types": getattr(input_def, "input_types", []),
                                "value": getattr(input_def, "value", None),
                            }
            else:
                build_config[key] = {}
        if field_name == "flow_name_selected" and (build_config.get("is_refresh", False) or field_value is None):
            # refresh button was clicked or componented was initialized, so list the flows
            options: list[str] = await self.alist_flows_by_flow_folder()
            build_config["flow_name_selected"]["options"] = [flow.data["name"] for flow in options]
            build_config["flow_name_selected"]["options_metadata"] = []
            for flow in options:
                # populate options_metadata
                build_config["flow_name_selected"]["options_metadata"].append(
                    {"id": flow.data["id"], "updated_at": flow.data["updated_at"]}
                )
                # update selected flow if it is stale
                if str(flow.data["id"]) == self.flow_id_selected:
                    await self.check_and_update_stale_flow(flow, build_config)
        elif field_name in {"flow_name_selected", "flow_id_selected"} and field_value is not None:
            # flow was selected by name or id, so get the flow and update the bcfg
            try:
                # derive flow id if the field_name is flow_name_selected
                build_config["flow_id_selected"]["value"] = (
                    self.get_selected_flow_meta(build_config, "id") or build_config["flow_id_selected"]["value"]
                )
                updated_at = self.get_selected_flow_meta(build_config, "updated_at")
                await self.load_graph_and_update_cfg(
                    build_config, build_config["flow_id_selected"]["value"], updated_at
                )
            except Exception as e:
                msg = f"Error building graph for flow {field_value}"
                await logger.aexception(msg)
                raise RuntimeError(msg) from e

        return build_config

    def get_selected_flow_meta(self, build_config: dotdict, field: str) -> dict:
        """Get the selected flow's metadata from the build config."""
        return build_config.get("flow_name_selected", {}).get("selected_metadata", {}).get(field)

    def update_build_config_from_graph(self, build_config: dotdict, graph):
        """Override to ensure our custom inputs are preserved."""
        # Preserve our custom inputs before updating
        preserved_inputs = {
            "user_input": build_config.get("user_input"),
            "escalation_trigger": build_config.get("escalation_trigger"),
        }
        
        # Call parent method
        super().update_build_config_from_graph(build_config, graph)
        
        # Restore our custom inputs after updating
        for key, value in preserved_inputs.items():
            if value is not None:
                build_config[key] = value
            elif key not in build_config:
                # Re-initialize if it was removed
                input_def = next((inp for inp in self.inputs if inp.name == key), None)
                if input_def:
                    # Convert Input to dict using model_dump or to_dict
                    if hasattr(input_def, "model_dump"):
                        build_config[key] = input_def.model_dump(by_alias=True, exclude_none=True)
                    elif hasattr(input_def, "to_dict"):
                        build_config[key] = input_def.to_dict()
                    else:
                        # Fallback: create dict from input attributes
                        build_config[key] = {
                            "name": input_def.name,
                            "display_name": getattr(input_def, "display_name", input_def.name),
                            "info": getattr(input_def, "info", ""),
                            "input_types": getattr(input_def, "input_types", []),
                            "value": getattr(input_def, "value", None),
                        }

    async def load_graph_and_update_cfg(
        self,
        build_config: dotdict,
        flow_id: str,
        updated_at: str | datetime,
    ) -> None:
        """Load a flow's graph and update the build config."""
        graph = await self.get_graph(
            flow_id_selected=flow_id,
            updated_at=self.get_str_isots(updated_at),
        )
        self.update_build_config_from_graph(build_config, graph)

    def should_update_stale_flow(self, flow: Data, build_config: dotdict) -> bool:
        """Check if the flow should be updated."""
        return (
            (updated_at := self.get_str_isots(flow.data["updated_at"]))  # true updated_at date just fetched from db
            and (stale_at := self.get_selected_flow_meta(build_config, "updated_at"))  # outdated date in bcfg
            and self._parse_timestamp(updated_at) > self._parse_timestamp(stale_at)  # stale flow condition
        )

    async def check_and_update_stale_flow(self, flow: Data, build_config: dotdict) -> None:
        """Check if the flow should be updated and update it if necessary."""
        # TODO: improve contract/return value
        if self.should_update_stale_flow(flow, build_config):
            await self.load_graph_and_update_cfg(
                build_config,
                flow.data["id"],
                flow.data["updated_at"],
            )

    def get_str_isots(self, date: datetime | str) -> str:
        """Get a string timestamp from a datetime or string."""
        return date.isoformat() if hasattr(date, "isoformat") else date

    def _pre_run_setup(self) -> None:
        """Override to add user_input to the target flow's chat input. Requires both inputs to be present to trigger."""
        # Call parent's _pre_run_setup first (this initializes _flow_run_inputs from tweaks)
        super()._pre_run_setup()

        # Ensure _flow_run_inputs is initialized (should be set by parent, but check to be safe)
        if not hasattr(self, "_flow_run_inputs") or self._flow_run_inputs is None:
            self._flow_run_inputs = []

        # Check if both inputs are present
        user_input_value = getattr(self, "user_input", None)
        escalation_trigger_value = getattr(self, "escalation_trigger", None)

        # Only proceed if both inputs are present
        if not user_input_value or not escalation_trigger_value:
            self.log("Waiting for both user_input and escalation_trigger to be present.")
            return

        # We need to get the graph synchronously or cache it
        # Since we can't use await here, we'll set up the inputs in a way that works
        # The graph will be loaded when the flow actually runs
        try:
            # Extract text from user input only (not the trigger)
            user_text = extract_text_from_input(user_input_value)

            # Store just the user input text without any headers
            self._pending_chat_input = user_text
            self.log(f"Prepared user input: {user_text[:100]}...")
        except Exception as e:
            logger.warning(f"Could not prepare inputs: {e}")

    async def _prepare_flow_inputs(self):
        """Prepare the flow inputs by finding the chat input vertex and setting up the input."""
        if not hasattr(self, "_pending_chat_input"):
            return

        try:
            graph = await self.get_graph(
                flow_name_selected=self.flow_name_selected,
                flow_id_selected=self.flow_id_selected,
                updated_at=self._cached_flow_updated_at,
            )

            # Find the chat input vertex
            chat_input_vertex = None
            for vertex in graph.vertices:
                if vertex.is_input and "ChatInput" in vertex.id:
                    chat_input_vertex = vertex
                    break

            # Find Chroma components that might need search_query
            chroma_vertices = []
            for vertex in graph.vertices:
                # Check if it's a Chroma component (Chroma, ChromaSearchAgent, etc.)
                if "Chroma" in vertex.id or "chroma" in vertex.id.lower():
                    # Check if it has a search_query input
                    vertex_template = vertex.data.get("node", {}).get("template", {})
                    if "search_query" in vertex_template:
                        chroma_vertices.append(vertex)

            if chat_input_vertex:
                # Build input for the chat input
                chat_input = {
                    "components": [chat_input_vertex.id],
                    "input_value": self._pending_chat_input,
                    "type": "chat",
                }

                # Ensure _flow_run_inputs is a list before using it
                if not isinstance(self._flow_run_inputs, list):
                    self._flow_run_inputs = []

                # Clear any existing inputs for this chat input vertex and add ours
                self._flow_run_inputs = [
                    inp for inp in self._flow_run_inputs
                    if not (isinstance(inp, dict) and inp.get("components") == [chat_input_vertex.id])
                ]
                self._flow_run_inputs.append(chat_input)
                
                self.log(f"Passing user input to flow '{self.flow_name_selected}' chat input (vertex: {chat_input_vertex.id}).")
                
                # Also pass the search query directly to Chroma components via tweaks
                # Use the pending chat input directly (it's already just the user text)
                user_input_text = self._pending_chat_input
                    
                # Add tweaks for Chroma components to set search_query directly
                if chroma_vertices and user_input_text:
                    for chroma_vertex in chroma_vertices:
                        tweak_key = f"{chroma_vertex.id}{self.IOPUT_SEP}search_query"
                        self._attributes[tweak_key] = user_input_text
                        self.log(f"Setting search_query for Chroma component '{chroma_vertex.id}': '{user_input_text[:50]}...'")
                
                # Clear the pending input
                delattr(self, "_pending_chat_input")
            else:
                self.log(f"No ChatInput found in flow '{self.flow_name_selected}'. Cannot pass inputs.")
        except Exception as e:
            logger.warning(f"Could not pass inputs to chat input: {e}")

    async def _get_cached_run_outputs(
        self,
        *,
        user_id: str | None = None,
        tweaks: dict | None,
        inputs: dict | list[dict] | None,
        output_type: str,
    ):
        """Override to prepare flow inputs before getting cached outputs."""
        # Prepare the flow inputs if we have pending input
        if hasattr(self, "_pending_chat_input"):
            await self._prepare_flow_inputs()
        
        # Call parent method
        return await super()._get_cached_run_outputs(
            user_id=user_id,
            tweaks=tweaks,
            inputs=inputs,
            output_type=output_type,
        )
