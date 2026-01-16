# Solution: Forcing Prompt to Wait for Multiple Inputs

## Problem

User has a flow where:
- User Input connects to BOTH Conditional Router AND Prompt
- Conditional Router also connects to Prompt
- Prompt executes BEFORE Router finishes, causing missing data

## Root Cause

Langflow's topological sorting allows components in the same "layer" to execute in parallel. The standard Prompt component doesn't enforce waiting for ALL inputs - it can execute as soon as ANY predecessor completes.

## Solution 1: Use the New Gated Prompt Component (RECOMMENDED)

I've created a new component specifically for this use case: **Gated Prompt**

### Location
`src/lfx/src/lfx/components/prompts/waiting_prompt.py`

### How It Works

The Gated Prompt component has **explicit, required inputs**:
- `user_input`: Primary user input
- `gate_input`: Input from conditional router (or any component you want to wait for)
- `template`: Prompt template with `{user_input}` and `{gate_input}` placeholders

**Key Feature**: The `build_prompt` method explicitly accesses BOTH inputs at the start:

```python
async def build_prompt(self) -> Message:
    # Explicitly access both inputs - this creates a dependency
    # The component CANNOT execute until both are available
    user_text = str(self.user_input) if self.user_input else ""
    gate_text = str(self.gate_input) if self.gate_input else ""
    # ... rest of the logic
```

This forces Langflow's dependency system to recognize that BOTH inputs must be satisfied before execution.

### Flow Structure

```
[User Input] ──┬──> [Conditional Router] ──┐
               │                            │
               └────────────────────────────┼──> [Gated Prompt]
                                            │    - user_input
                                            └──> - gate_input
```

### Usage

1. Add the **Gated Prompt** component to your flow
2. Connect User Input to the `user_input` port
3. Connect Conditional Router output to the `gate_input` port
4. Set your template:
   ```
   User asked: {user_input}
   Router decision: {gate_input}
   
   Please process this request appropriately.
   ```

### Why This Works

By making both inputs **explicit and required**, and by **accessing them at the beginning** of the build method, the component creates hard dependencies that Langflow's topological sort respects. The component literally cannot execute until both `self.user_input` and `self.gate_input` have values.

## Solution 2: Remove Direct Connection (Alternative)

If you don't want to use the Gated Prompt component, simply:

1. Remove the direct User Input → Prompt connection
2. Only connect: User Input → Router → Prompt
3. Modify your Router to include the original user input in its output

Example Router modification:

```python
def route_analytics(self) -> Message:
    text = extract_text_from_input(self.input_text)
    category = self.get_category(text)
    
    if category == "analytics_agent":
        # Include BOTH category and original input
        message = Message(
            text=f"[ANALYTICS] {text}",
            metadata={"category": "analytics", "original": text}
        )
        self.status = message
        return message
    return Message(text="")
```

## Solution 3: Create an Intermediate Combiner

Create a component that explicitly combines both inputs:

```python
class InputCombinerComponent(Component):
    inputs = [
        MessageTextInput(name="input_a", required=True),
        MessageTextInput(name="input_b", required=True),
    ]
    
    outputs = [Output(name="combined", method="combine")]
    
    async def combine(self) -> Message:
        # Accessing both forces waiting
        a = str(self.input_a)
        b = str(self.input_b)
        return Message(text=f"{a}\\n{b}")
```

## Technical Details

### How Langflow Determines Execution Order

1. **Topological Sort**: Analyzes graph connections to create execution layers
2. **Layer Execution**: Components in the same layer can execute in parallel
3. **No Implicit Waiting**: Components don't wait for ALL inputs, only for predecessors in the topological order

### Why Standard Prompt Doesn't Wait

The standard Prompt component:
- Creates dynamic inputs from template variables
- These dynamic inputs don't enforce "all must be present"
- Can execute as soon as template is available, even if some variables are empty

### Why Gated Prompt DOES Wait

The Gated Prompt component:
- Has **explicit, named inputs** (not dynamic)
- **Accesses both inputs immediately** in the build method
- This creates **hard dependencies** that topological sort respects
- Cannot execute until both dependencies are satisfied

## Testing

To verify the Gated Prompt works:

1. Add logging to your flow:
   ```python
   # In Router
   self.status = f"Router executing at {time.time()}"
   
   # In Gated Prompt (add this)
   self.status = f"Prompt executing at {time.time()} with gate: {gate_text[:50]}"
   ```

2. Run the flow and check execution order in logs
3. Verify the Prompt shows the Router's output, not empty values

## Comparison: Standard vs Gated Prompt

| Feature | Standard Prompt | Gated Prompt |
|---------|----------------|--------------|
| Input Type | Dynamic (from template) | Explicit, named |
| Waits for All Inputs | No | Yes |
| Execution Control | Loose | Strict |
| Use Case | Simple prompts | Multi-input coordination |
| Fresh User Input | Requires workaround | Built-in support |

## Best Practices

1. **Use Gated Prompt when**:
   - You have multiple input sources
   - Execution order matters
   - You need guaranteed input availability

2. **Use Standard Prompt when**:
   - Single input source
   - Simple template rendering
   - Execution order doesn't matter

3. **General Flow Design**:
   - Minimize parallel paths to the same component
   - Use explicit components for coordination
   - Test execution order with logging

## References

- Execution order analysis: `spec_sheet/025_execution_order_prompt_multiple_inputs.md`
- Graph topological sort: `src/lfx/src/lfx/graph/graph/base.py`
- Output required_inputs: `src/lfx/src/lfx/template/field/base.py` (line 205)
