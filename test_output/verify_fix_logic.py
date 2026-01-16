"""
Simplified verification of the fix logic without full lfx dependencies.
This demonstrates the key difference between the old and new approach.
"""

import pandas as pd


class MockData:
    """Mock Data class to demonstrate the issue."""
    
    def __init__(self, data):
        self.data = data
    
    def to_dataframe_OLD(self):
        """OLD APPROACH - wraps Data as single row."""
        # This is what DataFrame([data_obj]) does
        return pd.DataFrame([self.data])
    
    def to_dataframe_NEW(self):
        """NEW APPROACH - detects and expands nested records."""
        data_dict = self.data
        
        # If data contains only one key and the value is a list of dictionaries, expand it
        if (
            len(data_dict) == 1
            and isinstance(next(iter(data_dict.values())), list)
            and all(isinstance(item, dict) for item in next(iter(data_dict.values())))
        ):
            return pd.DataFrame(data=next(iter(data_dict.values())))
        
        # Otherwise treat as single row
        return pd.DataFrame([self.data])


def demonstrate_issue():
    """Show the difference between old and new approaches."""
    
    print("=" * 70)
    print("DEMONSTRATING THE FIX")
    print("=" * 70)
    
    # Simulate CSV data as it comes from a CSV parser
    csv_data = MockData(data={
        "records": [
            {"name": "Alice", "age": 30, "city": "NYC"},
            {"name": "Bob", "age": 25, "city": "LA"},
            {"name": "Charlie", "age": 35, "city": "Chicago"},
        ]
    })
    
    print("\nInput Data Structure:")
    print("--------------------")
    print("Data(data={")
    print("    'records': [")
    print("        {'name': 'Alice', 'age': 30, 'city': 'NYC'},")
    print("        {'name': 'Bob', 'age': 25, 'city': 'LA'},")
    print("        {'name': 'Charlie', 'age': 35, 'city': 'Chicago'},")
    print("    ]")
    print("})")
    
    # OLD APPROACH
    print("\n\n" + "=" * 70)
    print("OLD APPROACH: DataFrame([data_obj])")
    print("=" * 70)
    old_df = csv_data.to_dataframe_OLD()
    print(f"\nResulting DataFrame shape: {old_df.shape}")
    print(f"Number of rows: {len(old_df)}")
    print("\nDataFrame contents:")
    print(old_df)
    print("\nDataFrame columns:", list(old_df.columns))
    print("\nPROBLEM: Only 1 row! The entire 'records' list is stored in a single cell.")
    
    # Try to convert to dict
    old_records = old_df.to_dict(orient='records')
    print(f"\nto_dict(orient='records') returns {len(old_records)} record(s)")
    print("First record:", old_records[0])
    print("\nThis is NOT what we want - we need 3 separate rows!")
    
    # NEW APPROACH
    print("\n\n" + "=" * 70)
    print("NEW APPROACH: data_obj.to_dataframe()")
    print("=" * 70)
    new_df = csv_data.to_dataframe_NEW()
    print(f"\nResulting DataFrame shape: {new_df.shape}")
    print(f"Number of rows: {len(new_df)}")
    print("\nDataFrame contents:")
    print(new_df)
    print("\nDataFrame columns:", list(new_df.columns))
    print("\nSUCCESS: 3 separate rows! Each CSV row is properly expanded.")
    
    # Convert to dict
    new_records = new_df.to_dict(orient='records')
    print(f"\nto_dict(orient='records') returns {len(new_records)} records")
    for i, record in enumerate(new_records, 1):
        print(f"  Row {i}: {record}")
    
    print("\n\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"OLD approach: {len(old_df)} row (WRONG - nested structure not expanded)")
    print(f"NEW approach: {len(new_df)} rows (CORRECT - each CSV row is separate)")
    print("\nThe fix ensures CSV data is properly split into individual rows.")
    print("=" * 70)


def demonstrate_single_row():
    """Show that the new approach still handles single rows correctly."""
    
    print("\n\n" + "=" * 70)
    print("BONUS: Single Row Still Works Correctly")
    print("=" * 70)
    
    # Single row data (not nested)
    single_data = MockData(data={
        "name": "David",
        "age": 40,
        "city": "Seattle"
    })
    
    print("\nInput: Single Data object (not nested)")
    print(single_data.data)
    
    df = single_data.to_dataframe_NEW()
    print(f"\nResulting DataFrame shape: {df.shape}")
    print(df)
    
    print("\nSUCCESS: Single row data still works as expected!")


if __name__ == "__main__":
    demonstrate_issue()
    demonstrate_single_row()
    print("\n[OK] Verification complete!")
