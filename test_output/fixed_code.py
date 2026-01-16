"""
FIXED VERSION of split_csv_by_row.py

This version correctly splits CSV data into individual rows.

Key fixes:
1. Line 39: Use data_obj.to_dataframe() instead of DataFrame([data_obj])
2. Line 42: Use self.data_inputs.to_dataframe() instead of DataFrame([self.data_inputs])
3. Line 65: Use df.to_dict(orient='records') with explicit keyword argument
"""

from lfx.custom.custom_component.component import Component
from lfx.io import HandleInput, Output
from lfx.schema.data import Data
from lfx.schema.dataframe import DataFrame
from lfx.schema.message import Message


class SplitCSVByRowComponent(Component):
    display_name: str = "Split CSV by Row"
    description: str = "Split CSV/DataFrame into individual rows, one row per chunk."
    documentation: str = "https://docs.langflow.org/split-csv-by-row"
    icon = "scissors-line-dashed"
    name = "SplitCSVByRow"

    inputs = [
        HandleInput(
            name="data_inputs",
            display_name="Input",
            info="The DataFrame or Data to split into individual rows.",
            input_types=["Data", "DataFrame", "Message"],
            required=True,
        ),
    ]

    outputs = [
        Output(display_name="Chunks", name="dataframe", method="split_by_row"),
    ]

    def _convert_to_dataframe(self) -> DataFrame:
        """Convert input data to DataFrame format."""
        if isinstance(self.data_inputs, DataFrame):
            if not len(self.data_inputs):
                msg = "DataFrame is empty"
                raise TypeError(msg)
            return self.data_inputs
        elif isinstance(self.data_inputs, Message):
            # Convert Message to Data, then to DataFrame
            data_obj = self.data_inputs.to_data()
            return data_obj.to_dataframe()  # ✅ FIX: Properly expands nested data
        elif isinstance(self.data_inputs, Data):
            # Use to_dataframe() to properly expand nested CSV/structured data
            return self.data_inputs.to_dataframe()  # ✅ FIX: Properly expands nested data
        else:
            # Handle list of Data objects
            if not self.data_inputs:
                msg = "No data inputs provided"
                raise TypeError(msg)
            
            try:
                data_list = [input_ for input_ in self.data_inputs if isinstance(input_, Data)]
                if not data_list:
                    msg = f"No valid Data inputs found in {type(self.data_inputs)}"
                    raise TypeError(msg)
                return DataFrame(data_list)
            except (AttributeError, TypeError) as e:
                msg = f"Invalid input type in collection: {e}"
                raise TypeError(msg) from e

    def _split_dataframe_by_rows(self, df: DataFrame) -> list[dict]:
        """Split DataFrame into individual rows, one row per chunk."""
        if len(df) == 0:
            return []
        
        # Convert DataFrame to list of dictionaries (one per row)
        return df.to_dict(orient='records')  # ✅ FIX: Use explicit keyword argument

    def split_by_row(self) -> DataFrame:
        """Split the input data into individual rows, one row per chunk."""
        try:
            # Convert input to DataFrame
            df = self._convert_to_dataframe()
            
            # Split into individual rows (returns list of dictionaries)
            rows = self._split_dataframe_by_rows(df)
            
            # Create Data objects for each row with metadata
            chunk_data_list = []
            total_rows = len(rows)
            
            for idx, row_dict in enumerate(rows):
                chunk_metadata = {
                    "chunk_index": idx,
                    "total_chunks": total_rows,
                    "row_number": idx + 1,
                }
                
                # Merge row data with metadata
                complete_data = {
                    "chunk_metadata": chunk_metadata,
                    **row_dict,  # Include all original row columns
                }
                
                # Create a Data object for this single row
                chunk_data = Data(
                    data=complete_data,
                    text=f"Row {idx + 1} of {total_rows}"
                )
                chunk_data_list.append(chunk_data)
            
            return DataFrame(chunk_data_list)
            
        except Exception as e:
            msg = f"Error splitting CSV by row: {e}"
            raise TypeError(msg) from e
