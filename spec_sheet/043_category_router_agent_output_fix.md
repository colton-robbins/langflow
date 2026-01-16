# Category Router Agent Output Fix

## Problem
The router wasn't working when receiving input from agents because agents output structured objects, not plain strings.

### Example Agent Output Structure
```
{
  "content": "validated pcm knowledge",
  "additional_kwargs": {},
  "response_metadata": {...},
  "type": "ai",
  "name": null,
  "id": "run--53ebfc55-3a6c-437e-b3fe-7c49ae321507-0"
}

OR

{
  "return_values": {
    "log": "validated pcm knowledge"
  },
  "type": "AgentFinish"
}
```

The actual text "validated pcm knowledge" is nested inside `content` or `log` fields.

## Solution

### 1. Added Text Extraction Function
Created `extract_text_from_input()` function that handles multiple input types:
- Plain strings
- Message objects (extracts `.text`)
- Dict/structured objects from agents (extracts from `content`, `log`, `text`, `output`, or `return_values`)
- Fallback to string conversion

### 2. Added Priority Matching Logic
Created `get_best_match()` method that checks for specific phrase combinations FIRST before individual keywords.

#### Priority Order
1. **Phrase combinations** (highest priority):
   - "pcm" + "knowledge" → routes to `pcm_knowledge`
   - "client" + ("info" or "knowledge") → routes to `client_knowledge`
   - "knowledge" + "document" → routes to `knowledge_documents`

2. **Individual keywords** (fallback):
   - "analytics", "group", or "date" → `analytics_agent`
   - "document" or "documents" → `knowledge_documents`
   - "pcm" or "company info" → `pcm_knowledge`
   - "client info" → `client_knowledge`

### 3. Updated All Route Methods
All 4 route methods now:
1. Use `extract_text_from_input()` to get text from any input type
2. Use `get_best_match()` to determine correct route with priority logic
3. Only activate if they are the best match
4. Output the exact extracted text

## Why This Fixes "validated pcm knowledge"

Input: `"validated pcm knowledge"`

### Old Behavior (BROKEN)
- Checked "knowledge_documents" route first
- Found "knowledge" keyword → MATCHED incorrectly
- Routed to wrong category

### New Behavior (FIXED)
1. `get_best_match()` checks phrase combinations first
2. Finds BOTH "pcm" AND "knowledge" in the text
3. Returns `"pcm_knowledge"` as best match
4. Routes correctly to PCM Knowledge output

## Example Test Cases

### Input: "validated pcm knowledge"
- Extract: "validated pcm knowledge"
- Best match: `pcm_knowledge` (phrase combination detected)
- Routes to: PCM Knowledge output
- Output text: "validated pcm knowledge"

### Input: "validated knowledge and documents"
- Extract: "validated knowledge and documents"
- Best match: `knowledge_documents` (phrase combination detected)
- Routes to: Knowledge Documents output
- Output text: "validated knowledge and documents"

### Input: "validated client knowledge"
- Extract: "validated client knowledge"
- Best match: `client_knowledge` (phrase combination detected)
- Routes to: Client Knowledge output
- Output text: "validated client knowledge"

### Input: "escalate to analytics agent"
- Extract: "escalate to analytics agent"
- Best match: `analytics_agent` (keyword "analytics" detected)
- Routes to: Analytics Agent output
- Output text: "escalate to analytics agent"

## Key Changes Summary

1. Added `extract_text_from_input()` - handles agent structured outputs
2. Added `get_best_match()` - priority-based phrase matching
3. Updated all 4 route methods to use both new functions
4. Phrase combinations checked before individual keywords
5. Eliminates false matches when keywords overlap

## File Location
`src/lfx/src/lfx/components/flow_controls/conditional_router.py`
