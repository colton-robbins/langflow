"""Excel Output component - outputs Data or DataFrame as a downloadable Excel file."""

from __future__ import annotations

import tempfile
from datetime import datetime
from pathlib import Path

from lfx.custom.custom_component.component import Component
from lfx.io import HandleInput, Output, StrInput
from lfx.schema import Data, DataFrame, Message


class ExcelOutputComponent(Component):
    """Output component that converts Data or DataFrame to Excel and returns a downloadable file."""

    display_name = "Excel Output"
    description = "Output Data or DataFrame as a downloadable Excel file."
    name = "ExcelOutput"
    icon = "Excel"

    inputs = [
        HandleInput(
            name="input_value",
            display_name="Input",
            info="Data or DataFrame to convert to Excel.",
            input_types=["Data", "DataFrame"],
            required=True,
        ),
        StrInput(
            name="file_name",
            display_name="File Name",
            info="Base name for the output file (without extension). Leave empty for timestamp.",
            value="",
        ),
    ]
    outputs = [
        Output(display_name="Excel File", name="file_output", method="build_excel_output"),
    ]

    async def build_excel_output(self) -> Message:
        """Convert input to Excel, upload to storage, and return Message with file for download."""
        if self.input_value is None:
            msg = "Input is required."
            raise ValueError(msg)

        flow_id = getattr(self, "graph", None) and getattr(self.graph, "flow_id", None)
        if not flow_id:
            msg = "Flow context is required for file output."
            raise ValueError(msg)

        if type(self.input_value) is DataFrame:
            df = self.input_value
        elif type(self.input_value) is Data:
            import pandas as pd

            df = pd.DataFrame(self.input_value.data)
        else:
            msg = f"Input must be Data or DataFrame, got {type(self.input_value).__name__}"
            raise TypeError(msg)

        base_name = (self.file_name or "").strip()
        if not base_name:
            base_name = f"output_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        if not base_name.endswith(".xlsx"):
            base_name = f"{base_name}.xlsx"

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / base_name
            df.to_excel(output_path, index=False, engine="openpyxl")

            await self._save_excel(output_path, str(flow_id), base_name)

        path_str = f"{flow_id}/{base_name}"
        display_name = base_name.removesuffix(".xlsx") if base_name.endswith(".xlsx") else base_name
        file_info = {"path": path_str, "name": display_name, "type": "xlsx"}

        message = Message(
            text=f"Excel file '{display_name}.xlsx' is ready for download.",
            files=[file_info],
        )
        self.status = message
        return message

    async def _save_excel(self, file_path: Path, flow_id: str, file_name: str) -> None:
        """Save the Excel file to storage using flow_id for Playground download compatibility."""
        from lfx.services.deps import get_storage_service

        if not file_path.exists():
            msg = f"File not found: {file_path}"
            raise FileNotFoundError(msg)

        storage_service = get_storage_service()
        file_content = file_path.read_bytes()
        await storage_service.save_file(flow_id=flow_id, file_name=file_name, data=file_content)
