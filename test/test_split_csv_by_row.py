"""Test the SplitCSVByRowComponent to verify CSV splitting works correctly."""

import sys
from pathlib import Path

# Add the split_csv_by_row.py directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from split_csv_by_row import SplitCSVByRowComponent
from lfx.schema.data import Data
from lfx.schema.dataframe import DataFrame


def test_split_csv_from_nested_data():
    """Test splitting a CSV that comes as a single Data object with nested records."""
    print("\n=== Test 1: Split CSV from nested Data object ===")
    
    # Simulate CSV data as it would come from a CSV parser
    csv_data = Data(data={
        "records": [
            {"name": "Alice", "age": 30, "city": "NYC"},
            {"name": "Bob", "age": 25, "city": "LA"},
            {"name": "Charlie", "age": 35, "city": "Chicago"},
        ]
    })
    
    # Create component and set input
    component = SplitCSVByRowComponent()
    component.data_inputs = csv_data
    
    # Split by row
    result = component.split_by_row()
    
    print(f"Input: Single Data object with 3 nested records")
    print(f"Output: DataFrame with {len(result)} rows")
    
    # Verify we got 3 separate chunks
    assert len(result) == 3, f"Expected 3 chunks, got {len(result)}"
    
    # Convert to list to inspect
    result_list = result.to_data_list()
    
    # Check first chunk
    first_chunk = result_list[0]
    print(f"\nFirst chunk data: {first_chunk.data}")
    assert first_chunk.data["name"] == "Alice"
    assert first_chunk.data["age"] == 30
    assert first_chunk.data["chunk_metadata"]["row_number"] == 1
    assert first_chunk.data["chunk_metadata"]["total_chunks"] == 3
    
    # Check second chunk
    second_chunk = result_list[1]
    assert second_chunk.data["name"] == "Bob"
    assert second_chunk.data["age"] == 25
    assert second_chunk.data["chunk_metadata"]["row_number"] == 2
    
    # Check third chunk
    third_chunk = result_list[2]
    assert third_chunk.data["name"] == "Charlie"
    assert third_chunk.data["age"] == 35
    assert third_chunk.data["chunk_metadata"]["row_number"] == 3
    
    print("\nTest 1 PASSED: CSV with nested records split correctly")


def test_split_csv_from_dataframe():
    """Test splitting a CSV that comes as a DataFrame."""
    print("\n=== Test 2: Split CSV from DataFrame ===")
    
    # Create DataFrame directly
    csv_dataframe = DataFrame([
        {"name": "David", "score": 95},
        {"name": "Eve", "score": 87},
    ])
    
    # Create component and set input
    component = SplitCSVByRowComponent()
    component.data_inputs = csv_dataframe
    
    # Split by row
    result = component.split_by_row()
    
    print(f"Input: DataFrame with 2 rows")
    print(f"Output: DataFrame with {len(result)} rows")
    
    # Verify we got 2 separate chunks
    assert len(result) == 2, f"Expected 2 chunks, got {len(result)}"
    
    result_list = result.to_data_list()
    
    # Check chunks
    assert result_list[0].data["name"] == "David"
    assert result_list[0].data["score"] == 95
    assert result_list[1].data["name"] == "Eve"
    assert result_list[1].data["score"] == 87
    
    print("\nTest 2 PASSED: DataFrame split correctly")


def test_split_csv_from_data_list():
    """Test splitting a list of Data objects."""
    print("\n=== Test 3: Split from list of Data objects ===")
    
    # Create list of Data objects
    data_list = [
        Data(data={"product": "Widget", "price": 10.99}),
        Data(data={"product": "Gadget", "price": 25.50}),
        Data(data={"product": "Doohickey", "price": 5.00}),
    ]
    
    # Create component and set input
    component = SplitCSVByRowComponent()
    component.data_inputs = data_list
    
    # Split by row
    result = component.split_by_row()
    
    print(f"Input: List of 3 Data objects")
    print(f"Output: DataFrame with {len(result)} rows")
    
    # Verify we got 3 separate chunks
    assert len(result) == 3, f"Expected 3 chunks, got {len(result)}"
    
    result_list = result.to_data_list()
    
    # Check chunks
    assert result_list[0].data["product"] == "Widget"
    assert result_list[1].data["product"] == "Gadget"
    assert result_list[2].data["product"] == "Doohickey"
    
    print("\nTest 3 PASSED: List of Data objects split correctly")


def test_metadata_correctness():
    """Test that metadata is correctly added to each chunk."""
    print("\n=== Test 4: Verify metadata correctness ===")
    
    csv_data = Data(data={
        "records": [
            {"item": "A"},
            {"item": "B"},
            {"item": "C"},
            {"item": "D"},
        ]
    })
    
    component = SplitCSVByRowComponent()
    component.data_inputs = csv_data
    result = component.split_by_row()
    result_list = result.to_data_list()
    
    # Check metadata for each chunk
    for idx, chunk in enumerate(result_list):
        metadata = chunk.data["chunk_metadata"]
        print(f"Chunk {idx}: {metadata}")
        
        assert metadata["chunk_index"] == idx
        assert metadata["row_number"] == idx + 1
        assert metadata["total_chunks"] == 4
        assert chunk.get_text() == f"Row {idx + 1} of 4"
    
    print("\nTest 4 PASSED: Metadata is correct for all chunks")


if __name__ == "__main__":
    try:
        test_split_csv_from_nested_data()
        test_split_csv_from_dataframe()
        test_split_csv_from_data_list()
        test_metadata_correctness()
        
        print("\n" + "=" * 60)
        print("ALL TESTS PASSED!")
        print("=" * 60)
    except AssertionError as e:
        print(f"\nTEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
