# Category Keyword Router Implementation Plan

## Overview
Complete rewrite of the conditional router to create a simple keyword-based category router.

## Requirements
1. **Input**: Single string input
2. **Processing**: Match input to 4 categories based on keyword detection
3. **Outputs**: 4 separate output routes
4. **Output Value**: EXACT input string passes through to matched category

## Category Matching Logic

### Category 1: "escalate to analytics agent"
- Keywords: "analytics", "group", "date"
- Logic: Check if input contains ANY of these keywords

### Category 2: "validated knowledge and documents"  
- Keywords: "documents", "knowledge"
- Logic: Check if input contains ANY of these keywords

### Category 3: "validated pcm knowledge"
- Keywords: "PCM", "company info"
- Logic: Check if input contains ANY of these keywords

### Category 4: "validated client knowledge"
- Keywords: "client info"
- Logic: Check if input contains this keyword phrase

## Component Structure

### Inputs
1. `input_text` (MessageTextInput) - The string to categorize
2. `case_sensitive` (BoolInput) - Whether keyword matching is case-sensitive (default: False)

### Outputs
1. `analytics_agent` - Route for analytics/group/date queries
2. `knowledge_documents` - Route for documents/knowledge queries
3. `pcm_knowledge` - Route for PCM/company info queries
4. `client_knowledge` - Route for client info queries

### Methods
- `route_analytics()` - Returns input if matches analytics keywords
- `route_knowledge_documents()` - Returns input if matches documents/knowledge keywords
- `route_pcm_knowledge()` - Returns input if matches PCM keywords
- `route_client_knowledge()` - Returns input if matches client keywords
- `check_keywords()` - Helper method to check if input contains any keyword from a list

## Implementation Steps
1. Create new component class `CategoryKeywordRouter`
2. Define inputs (input_text, case_sensitive)
3. Define 4 outputs with corresponding methods
4. Implement keyword checking logic
5. Implement each route method to:
   - Check if input contains category keywords
   - If match: return input string and stop other routes
   - If no match: stop this route
6. Use `exclude_branch_conditionally()` for proper routing

## Key Differences from Old Router
- No cycles/iterations needed
- No max_iterations
- No complex condition evaluation
- Simple keyword matching
- 4 outputs instead of 2
- One-way routing (no loops)
