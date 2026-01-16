# Keyword Checker Component Plan

## Overview
Create a new Langflow component that checks if a keyword exists anywhere in the input text and routes accordingly.

## Component Details

### Name and Description
- **Component Name:** KeywordChecker
- **Display Name:** Keyword Checker
- **Description:** Checks if a keyword exists anywhere in the input text and routes to appropriate output
- **Category:** flow_controls
- **Icon:** search (Lucide icon)

### Inputs
1. **input_text** (MessageTextInput)
   - Display Name: Input Text
   - Info: Text to search for the keyword
   - Required: Yes
   - Input types: Message, Data, str

2. **keyword** (MessageTextInput)
   - Display Name: Keyword
   - Info: The keyword to search for in the input text
   - Required: Yes

3. **case_sensitive** (BoolInput)
   - Display Name: Case Sensitive
   - Info: Whether the keyword search should be case-sensitive
   - Default: False
   - Required: No

### Outputs
1. **found** (Output)
   - Display Name: Found
   - Method: check_keyword_found
   - Info: Returns the input text if keyword is found

2. **not_found** (Output)
   - Display Name: Not Found
   - Method: check_keyword_not_found
   - Info: Returns the input text if keyword is not found

### Logic
1. Extract text from input (handle Message, Data, str types)
2. Extract keyword from input
3. Check if keyword exists in text based on case_sensitive setting
4. Route to appropriate output based on result
5. Use self.status to indicate which path was taken

### File Location
`src/lfx/src/lfx/components/flow_controls/keyword_checker.py`

## Implementation Steps
1. Create the component file with proper imports
2. Define the KeywordCheckerComponent class
3. Implement text extraction helper method
4. Implement keyword checking logic
5. Implement both output methods (found/not_found)
6. Add proper logging and status updates

## Testing Approach
- Test with simple string input
- Test with Message object input
- Test with case-sensitive and case-insensitive modes
- Test with keyword present and absent
- Test with special characters and whitespace in keywords
