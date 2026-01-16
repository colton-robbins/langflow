# Before vs After: CSV Split By Row Fix

## Visual Comparison

### Input Data (CSV loaded as Data object)
```python
Data(data={
    "records": [
        {"name": "Alice", "age": 30, "city": "NYC"},
        {"name": "Bob", "age": 25, "city": "LA"},
        {"name": "Charlie", "age": 35, "city": "Chicago"}
    ]
})
```

---

## BEFORE the Fix

### Code (Lines 38-42)
```python
elif isinstance(self.data_inputs, Message):
    data_obj = self.data_inputs.to_data()
    return DataFrame([data_obj])  # ❌ WRONG
elif isinstance(self.data_inputs, Data):
    return DataFrame([self.data_inputs])  # ❌ WRONG
```

### Code (Line 64)
```python
return df.to_dict('records')  # ⚠️ Works but not explicit
```

### Result
```
DataFrame with 1 row:
┌─────────────────────────────────────────────┐
│ records                                     │
├─────────────────────────────────────────────┤
│ [{'name': 'Alice', ...}, {'name': 'Bob'...}]│
└─────────────────────────────────────────────┘

❌ PROBLEM: Entire CSV is in one cell!
❌ Split by row produces 1 chunk instead of 3
```

---

## AFTER the Fix

### Code (Lines 38-42)
```python
elif isinstance(self.data_inputs, Message):
    data_obj = self.data_inputs.to_data()
    return data_obj.to_dataframe()  # ✅ CORRECT
elif isinstance(self.data_inputs, Data):
    return self.data_inputs.to_dataframe()  # ✅ CORRECT
```

### Code (Line 65)
```python
return df.to_dict(orient='records')  # ✅ Explicit keyword arg
```

### Result
```
DataFrame with 3 rows:
┌─────────┬─────┬─────────┐
│ name    │ age │ city    │
├─────────┼─────┼─────────┤
│ Alice   │ 30  │ NYC     │
│ Bob     │ 25  │ LA      │
│ Charlie │ 35  │ Chicago │
└─────────┴─────┴─────────┘

✅ SUCCESS: Each CSV row is separate!
✅ Split by row produces 3 chunks
```

---

## Output Chunks After Split

### Chunk 1
```python
Data(
    data={
        "chunk_metadata": {
            "chunk_index": 0,
            "total_chunks": 3,
            "row_number": 1
        },
        "name": "Alice",
        "age": 30,
        "city": "NYC"
    },
    text="Row 1 of 3"
)
```

### Chunk 2
```python
Data(
    data={
        "chunk_metadata": {
            "chunk_index": 1,
            "total_chunks": 3,
            "row_number": 2
        },
        "name": "Bob",
        "age": 25,
        "city": "LA"
    },
    text="Row 2 of 3"
)
```

### Chunk 3
```python
Data(
    data={
        "chunk_metadata": {
            "chunk_index": 2,
            "total_chunks": 3,
            "row_number": 3
        },
        "name": "Charlie",
        "age": 35,
        "city": "Chicago"
    },
    text="Row 3 of 3"
)
```

---

## Key Insight

The `Data.to_dataframe()` method already has the intelligence to:
1. Detect nested record structures like `{"records": [...]}`
2. Automatically expand them into multiple DataFrame rows
3. Handle single-row data correctly

By using this existing method instead of manually wrapping with `DataFrame([...])`, the component now works correctly for all CSV input formats.

---

## Impact

✅ CSV files now split correctly into individual rows
✅ Each row becomes a separate chunk with metadata
✅ Works with File component, Type Converter, and other CSV sources
✅ No breaking changes to existing functionality
✅ No linter errors
