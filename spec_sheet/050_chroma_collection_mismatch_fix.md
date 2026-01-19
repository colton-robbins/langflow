# Chroma Collection Name Mismatch Fix

## Issue Summary

Documents were being ingested into a Chroma vector store but could not be retrieved during search operations.

## Root Cause Analysis

The Chroma Ingest component and Chroma Search Agent component were using DIFFERENT collection names:

- Ingest Component: Writing to collection `langflow_v2`
- Search Agent: Reading from collection `langflow` (hardcoded default)

### Evidence from Logs

Ingest logs:
```
"Detected new data to ingest. Clearing existing collection 'langflow_v2' to ensure fresh start."
"SUCCESS: Added 12 document(s) to collection 'langflow_v2'"
"BUILD COMPLETE: Collection 'langflow_v2' final state: 12 document(s)"
```

Retrieve logs:
```
"Connected to Chroma DB at '/app/medical_savings' (collection: 'langflow', documents: 12)"
```

## Technical Details

The Chroma Search Agent component was missing the `collection_name` input field entirely. It attempted to read it with:

```python
collection_name = getattr(self, "collection_name", "langflow")
```

Since the input did not exist, it always defaulted to `"langflow"`, regardless of what collection name was used during ingestion.

## Solution

Added the `collection_name` input to the Chroma Search Agent component to match the Chroma Ingest component.

### Changes Made

File: `src/lfx/src/lfx/components/chroma/chroma_search_agent.py`

1. Added `collection_name` input field to the inputs list
2. Updated `build_vector_store()` method to properly handle the collection_name input (supporting both string and Message object types)
3. Ensured proper logging of the actual collection name being used

### New Input Field

```python
MessageTextInput(
    name="collection_name",
    display_name="Collection Name",
    info="Name of the Chroma collection to search. Must match the collection name used during ingestion.",
    required=False,
    show=True,
    value="langflow",
)
```

### Collection Name Handling

The collection name is now properly extracted from the input, handling multiple input types:
- String values
- Message objects (with .text attribute)
- Data objects (with .data dictionary)
- Defaults to "langflow" if empty or not provided

## Resolution

Users must now ensure the SAME collection name is used in both:
1. The Chroma Ingest component (Collection Name input)
2. The Chroma Search Agent component (Collection Name input - NEW)

## Testing Recommendation

To verify the fix:
1. Set both components to use the same collection name (e.g., `langflow_v2`)
2. Run ingestion to add documents
3. Run search queries - documents should now be retrievable
4. Check logs to confirm both components reference the same collection name

## Prevention

This issue highlights the importance of configuration consistency across related components. Future improvements could include:
- Component validation to warn when persist directory + collection name combinations don't match between ingest and search
- Shared configuration objects that ensure matching settings
- Better documentation about collection name requirements
