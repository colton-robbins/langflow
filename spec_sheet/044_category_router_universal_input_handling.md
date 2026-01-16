# Category Router - Universal Input Handling

## Final Solution
The router now handles LITERALLY ANY input type and searches for keywords anywhere in the structure.

## How It Works

### Two-Phase Text Extraction

#### Phase 1: `extract_text_from_input()` - Aggressive Searching
Converts ANY input to a searchable string:
- Plain strings → returned as-is
- Message objects → extracts `.text` or converts to string
- Dictionaries → converts to JSON string (searches entire structure!)
- Lists/Tuples → converts to JSON string
- Anything else → converts to string

**Result**: Keywords can be found ANYWHERE in nested structures, JSON, dicts, agent outputs, etc.

#### Phase 2: `extract_clean_text_for_output()` - Clean Output
Extracts the actual text content for output:
- Tries common text fields: `content`, `text`, `log`, `output`, `message`
- Recursively checks nested structures
- Falls back to searchable string if no clean text found

**Result**: Output message contains clean, readable text instead of JSON.

### Priority-Based Matching

The `get_best_match()` method uses smart priority logic:

1. **Phrase Combinations** (checked first):
   - "pcm" + "knowledge" → PCM Knowledge
   - "client" + ("info" OR "knowledge") → Client Knowledge
   - "knowledge" + "document" → Knowledge Documents

2. **Individual Keywords** (fallback):
   - "analytics", "group", "date" → Analytics Agent
   - "document", "documents" → Knowledge Documents
   - "pcm", "company info" → PCM Knowledge
   - "client info" → Client Knowledge

## Example Scenarios

### Scenario 1: Agent Output (Dict)
**Input:**
```json
{
  "content": "validated pcm knowledge",
  "type": "ai",
  "id": "run-123",
  "additional_kwargs": {}
}
```

**Processing:**
1. `extract_text_from_input()` → Converts entire dict to JSON string
2. Searches JSON for keywords → finds "pcm" and "knowledge"
3. `get_best_match()` → Returns "pcm_knowledge" (phrase combination)
4. `extract_clean_text_for_output()` → Extracts "validated pcm knowledge" from `content` field
5. Routes to **PCM Knowledge** with clean output: "validated pcm knowledge"

### Scenario 2: Nested Agent Return Values
**Input:**
```json
{
  "return_values": {
    "log": "validated client knowledge"
  },
  "type": "AgentFinish"
}
```

**Processing:**
1. `extract_text_from_input()` → Converts to JSON, searches entire structure
2. Finds "client" and "knowledge" in the JSON
3. `get_best_match()` → Returns "client_knowledge"
4. `extract_clean_text_for_output()` → Recursively extracts "validated client knowledge" from `return_values.log`
5. Routes to **Client Knowledge** with output: "validated client knowledge"

### Scenario 3: Plain String
**Input:** `"Show me analytics for my group"`

**Processing:**
1. `extract_text_from_input()` → Returns string as-is
2. Finds "analytics" and "group"
3. `get_best_match()` → Returns "analytics_agent"
4. `extract_clean_text_for_output()` → Returns string as-is
5. Routes to **Analytics Agent** with output: "Show me analytics for my group"

### Scenario 4: Message Object
**Input:** `Message(text="I need documents about knowledge base")`

**Processing:**
1. `extract_text_from_input()` → Extracts `.text` field
2. Finds "documents" and "knowledge"
3. `get_best_match()` → Returns "knowledge_documents" (phrase combination)
4. `extract_clean_text_for_output()` → Returns `.text`
5. Routes to **Knowledge Documents** with output: "I need documents about knowledge base"

## Why This Handles Literally Anything

1. **JSON Conversion**: Entire structures are converted to JSON strings, so keywords nested anywhere are found
2. **Multiple Fallbacks**: If JSON fails, uses string conversion
3. **Recursive Extraction**: Clean text extraction recursively digs into nested structures
4. **Type Agnostic**: Works with str, dict, list, tuple, Message, or custom objects
5. **No Assumptions**: Doesn't assume specific field names or structure

## Technical Details

### Search Strategy
- Input → Convert to searchable string (JSON/string representation)
- Search for keywords in that string
- Use priority logic to determine best match
- Extract clean text for output

### Output Strategy
- Try common text fields first: `content`, `text`, `log`, `output`, `message`
- Recursively check nested dicts
- Fallback to full searchable string if needed

### Edge Cases Handled
- Empty inputs → Returns empty string
- None values → Returns empty string
- Non-serializable objects → Uses `str()` conversion
- Circular references → JSON `default=str` handles it
- Complex nested structures → Recursive extraction
- Agent outputs → Specific field checking

## Testing Examples

```python
# Test 1: Dict with content
input1 = {"content": "validated pcm knowledge", "type": "ai"}
# Result: Routes to PCM Knowledge, output: "validated pcm knowledge"

# Test 2: Nested return values
input2 = {"return_values": {"log": "validated client knowledge"}}
# Result: Routes to Client Knowledge, output: "validated client knowledge"

# Test 3: Plain string
input3 = "I need analytics"
# Result: Routes to Analytics Agent, output: "I need analytics"

# Test 4: List with nested dict
input4 = [{"message": "show me documents"}]
# Result: Routes to Knowledge Documents, output: JSON or extracted text

# Test 5: Complex object
input5 = Message(text="company info from PCM")
# Result: Routes to PCM Knowledge, output: "company info from PCM"
```

## File Location
`src/lfx/src/lfx/components/flow_controls/conditional_router.py`

## Key Functions
1. `extract_text_from_input()` - Universal text extraction for searching
2. `extract_clean_text_for_output()` - Clean text extraction for output
3. `get_best_match()` - Priority-based keyword matching
4. 4 route methods - Each uses both extraction functions and priority matching
