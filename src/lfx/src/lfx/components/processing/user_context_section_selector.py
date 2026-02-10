import json
from typing import Any

from lfx.custom import Component
from lfx.inputs import DataInput, MultiselectInput
from lfx.io import Output
from lfx.log.logger import logger
from lfx.schema import Data, Message


class UserContextSectionSelectorComponent(Component):
    display_name = "User Context Section Selector"
    description = "Extract and select specific sections from user context data object."
    icon = "file-json"
    name = "UserContextSectionSelector"

    inputs = [
        DataInput(
            name="user_context_data",
            display_name="User Context Data",
            info="Data object containing user context with sections to extract.",
            required=True,
        ),
        MultiselectInput(
            name="selected_sections",
            display_name="Selected Sections",
            info="Select one or more fields to include in the output context. All individual fields are available.",
            options=[
                "user_context",
                "identity",
                "identity.user_id",
                "identity.name",
                "identity.email",
                "identity.role",
                "identity.role.id",
                "identity.role.name",
                "identity.is_super_admin",
                "identity.is_primary_user",
                "identity.is_internal_reviewer",
                "identity.is_help_desk",
                "permissions",
                "permissions.role_based",
                "permissions.role_based.can_create_folders",
                "permissions.role_based.can_edit_folders",
                "permissions.role_based.can_delete_folders",
                "permissions.role_based.can_share_folders",
                "permissions.role_based.can_upload_documents",
                "permissions.role_based.can_edit_documents",
                "permissions.role_based.can_delete_documents",
                "permissions.role_based.can_view_reports",
                "permissions.role_based.can_generate_reports",
                "permissions.role_based.can_view_claims",
                "permissions.role_based.can_view_proposals",
                "permissions.feature_access",
                "permissions.feature_access.full_groups_access",
                "permissions.feature_access.pbm_access",
                "permissions.feature_access.stoploss_access",
                "permissions.feature_access.demo_groups_access",
                "permissions.module_access",
                "permissions.module_access.document_manager",
                "permissions.module_access.document_manager.view",
                "permissions.module_access.document_manager.create",
                "permissions.module_access.document_manager.edit",
                "permissions.module_access.document_manager.delete",
                "permissions.module_access.reports",
                "permissions.module_access.reports.view",
                "permissions.module_access.reports.create",
                "permissions.module_access.reports.edit",
                "permissions.module_access.reports.delete",
                "permissions.module_access.pbm_rfp",
                "permissions.module_access.pbm_rfp.view",
                "permissions.module_access.pbm_rfp.create",
                "permissions.module_access.pbm_rfp.edit",
                "permissions.module_access.pbm_rfp.delete",
                "permissions.module_access.stoploss_rfp",
                "permissions.module_access.stoploss_rfp.view",
                "permissions.module_access.stoploss_rfp.create",
                "permissions.module_access.stoploss_rfp.edit",
                "permissions.module_access.stoploss_rfp.delete",
                "permissions.module_access.claims",
                "permissions.module_access.claims.view",
                "permissions.module_access.claims.create",
                "permissions.module_access.claims.edit",
                "permissions.module_access.claims.delete",
                "permissions.module_access.demo_groups",
                "permissions.module_access.demo_groups.view",
                "permissions.module_access.demo_groups.create",
                "permissions.module_access.demo_groups.edit",
                "permissions.module_access.demo_groups.delete",
                "organization",
                "organization.id",
                "organization.name",
                "organization.type",
                "organization.is_admin",
                "organization.hierarchy",
                "organization.hierarchy.is_root",
                "organization.hierarchy.parent_id",
                "organization.hierarchy.parent_name",
                "organization.hierarchy.child_count",
                "organization.hierarchy.depth",
                "organization.accessible_organizations",
                "organization.share_group_ids",
                "group_context",
                "session_state",
                "session_state.session_id",
                "session_state.session_uuid",
                "session_state.date_range",
                "session_state.selected_context",
                "session_state.selected_context.folders",
                "session_state.selected_context.files",
                "document_manager",
                "document_manager.enabled",
                "document_manager.folder_permissions",
                "document_manager.folder_permissions.sharing_types",
                "document_manager.folder_permissions.sharing_types.private_only",
                "document_manager.folder_permissions.sharing_types.public_to_my_organization",
                "document_manager.folder_permissions.sharing_types.specific_users",
                "document_manager.folder_permissions.access_levels",
                "document_manager.folder_permissions.access_levels.owner",
                "document_manager.folder_permissions.access_levels.admin",
                "document_manager.folder_permissions.access_levels.editor",
                "document_manager.folder_permissions.access_levels.viewer",
                "document_manager.folder_permissions.inheritance",
                "document_manager.system_folders",
                "document_manager.accessible_folder_count",
                "document_manager.accessible_file_count",
                "compliance",
                "compliance.hipaa_compliant",
                "compliance.data_handling_restrictions",
                "compliance.audit_enabled",
            ],
            value=[],
        ),
    ]

    outputs = [
        Output(
            display_name="Selected Context",
            name="selected_context",
            method="extract_sections",
            types=["Message", "Data"],
        ),
    ]

    def _extract_all_paths(self, obj: Any, path: str = "", paths: list[str] | None = None) -> list[str]:
        """Recursively extract all paths from a nested dictionary/object structure."""
        if paths is None:
            paths = []
        
        if isinstance(obj, dict):
            for key, value in obj.items():
                current_path = f"{path}.{key}" if path else key
                # Add the current path
                paths.append(current_path)
                # Recursively process nested structures
                if isinstance(value, (dict, list)):
                    self._extract_all_paths(value, current_path, paths)
        elif isinstance(obj, list) and obj:
            # For lists, process the first item to get structure
            # Don't add list indices, just process the structure
            if isinstance(obj[0], (dict, list)):
                self._extract_all_paths(obj[0], path, paths)
        
        return paths

    def _extract_sections_from_data(self, data: Any) -> dict[str, Any]:
        """Extract all sections from the user_context object."""
        sections = {}

        # Handle Data object
        if isinstance(data, Data):
            data_dict = data.model_dump()
            # Check if it has a 'data' key
            if "data" in data_dict:
                data_dict = data_dict["data"]
        elif isinstance(data, dict):
            data_dict = data
        else:
            # Try to parse as JSON string
            try:
                data_dict = json.loads(str(data))
            except (json.JSONDecodeError, TypeError):
                logger.warning("Could not parse input data as JSON or dict")
                return sections

        # Check if we have user_context at top level
        if "user_context" in data_dict:
            # Extract sections from inside user_context
            user_context = data_dict["user_context"]
            if isinstance(user_context, dict):
                # Add all individual sections inside user_context
                for key, value in user_context.items():
                    sections[key] = value
                # Also add user_context itself as an option (contains all sections)
                sections["user_context"] = user_context
        else:
            # If no user_context wrapper, treat the whole dict as sections
            if isinstance(data_dict, dict):
                for key, value in data_dict.items():
                    sections[key] = value

        return sections


    def _get_nested_value(self, data: dict, path: str) -> Any:
        """Get a nested value from a dictionary using dot notation path."""
        keys = path.split(".")
        current = data
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return None
        return current

    def _set_nested_value(self, data: dict, path: str, value: Any) -> None:
        """Set a nested value in a dictionary using dot notation path."""
        keys = path.split(".")
        current = data
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        current[keys[-1]] = value

    def extract_sections(self) -> Message:
        """Extract selected sections from user context data."""
        if not self.user_context_data:
            msg = "User context data is required."
            raise ValueError(msg)

        if not self.selected_sections:
            msg = "At least one section must be selected."
            raise ValueError(msg)

        # Extract all sections
        all_sections = self._extract_sections_from_data(self.user_context_data)

        # Separate top-level sections from nested subsections
        top_level_selections = []
        nested_selections = []
        
        for section_path in self.selected_sections:
            if "." in section_path:
                nested_selections.append(section_path)
            else:
                top_level_selections.append(section_path)

        # Build the selected context
        selected_context = {}
        
        # First, add all top-level sections
        for section_path in top_level_selections:
            if section_path in all_sections:
                selected_context[section_path] = all_sections[section_path]
            else:
                logger.warning(f"Section '{section_path}' not found in user context data")

        # Then, add nested subsections (they will merge into parent sections if parent was selected)
        for section_path in nested_selections:
            top_level = section_path.split(".")[0]
            if top_level in all_sections:
                top_level_data = all_sections[top_level]
                if isinstance(top_level_data, dict):
                    # Get the nested value
                    nested_value = self._get_nested_value(top_level_data, ".".join(section_path.split(".")[1:]))
                    if nested_value is not None:
                        # Initialize parent section if not already added
                        if top_level not in selected_context:
                            selected_context[top_level] = {}
                        # Merge nested value into parent section
                        self._set_nested_value(selected_context[top_level], ".".join(section_path.split(".")[1:]), nested_value)
                    else:
                        logger.warning(f"Subsection '{section_path}' not found in user context data")
                else:
                    logger.warning(f"Section '{top_level}' is not a dictionary, cannot access subsection '{section_path}'")
            else:
                logger.warning(f"Top-level section '{top_level}' not found in user context data")

        # If user_context itself was selected, return it as-is
        # Otherwise wrap selected sections in user_context structure
        if "user_context" in selected_context and len(selected_context) == 1:
            # If only user_context was selected, return it wrapped
            output_data = {"user_context": selected_context["user_context"]}
        else:
            # Remove user_context from selections if other sections are also selected
            # (since user_context contains all sections)
            if "user_context" in selected_context:
                selected_context.pop("user_context")
            # Wrap in user_context structure to match original format
            output_data = {"user_context": selected_context}

        logger.info(f"Extracted {len(selected_context)} sections: {list(selected_context.keys())}")

        # Stringify the output data as JSON for prompt templates
        json_string = json.dumps(output_data, indent=2, ensure_ascii=False)
        
        # Return Message object for prompt template compatibility
        # Prompt template variables expect Message type, and Message.text will contain the JSON string
        return Message(text=json_string, data=output_data)
