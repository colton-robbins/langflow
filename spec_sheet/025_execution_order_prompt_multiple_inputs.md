# Langflow Execution Order Issue: Prompt with Multiple Inputs

## Problem Summary

When a Prompt component receives inputs from BOTH:
1. A User Input component (directly)
2. A Conditional Router (which also receives the User Input)

The Prompt may execute BEFORE the Conditional Router finishes, causing the router's variable to be missing or empty.

## Root Cause

### How Langflow Execution Works

Langflow uses **topological sorting** to determine execution order:

1. The graph analyzes all component connections (edges)
2. It builds a dependency graph based on these connections
3. Components execute in "layers" - each layer contains components whose dependencies are satisfied
4. A component executes as soon as its predecessors in the topological order have executed

### Why Your Flow Executes Out of Order

```
[User Input] ──┬──> [Conditional Router] ──> [Prompt {user_input} {router_output}]
               └─────────────────────────────> [Prompt {user_input} {router_output}]
```

**Execution sequence:**
1. **Layer 1**: User Input executes (no dependencies)
2. **Layer 2**: BOTH Conditional Router AND Prompt can execute (both depend only on User Input)
3. Prompt sees `{user_input}` has a value and may execute immediately
4. Router is still processing
5. Prompt executes with `{user_input}` populated but `{router_output}` is empty or undefined

### Key Issue: No "Required Input" Mechanism

Langflow does NOT have a traditional "required input" field. The Prompt component's dynamic variables (created from your template like `{user_input}` and `{router_output}`) do not enforce that ALL variables must have values before execution.

## Solutions

### Solution 1: Remove Direct User Input Connection (RECOMMENDED)

**Only** connect the Router to the Prompt, not the User Input directly:

```
[User Input] ──> [Conditional Router] ──> [Prompt {router_output}]
```

**If you need both the original input AND the routing decision:**
- Have your Conditional Router pass through the original input in its output
- Modify the router to include both the decision and the original text

### Solution 2: Modify Router to Include Original Input

Update your Conditional Router component to return BOTH the category decision AND the original input:

```python
def route_analytics(self) -> Message:
    text = extract_text_from_input(self.input_text)
    category = self.get_category(text)
    
    # Include both original input and routing decision
    if category == "analytics_agent":
        message = Message(
            text=f"Category: analytics | Input: {text}",
            metadata={"category": "analytics", "original_input": text}
        )
        self.status = message
        return message
    else:
        # Return empty or pass-through for non-matching routes
        return Message(text="")
```

Then in your Prompt template:
```
Process this request:
{router_output}
```

### Solution 3: Create a Combiner Component

Create a new component that explicitly waits for BOTH inputs:

```python
from lfx.custom.custom_component.component import Component
from lfx.io import MessageTextInput, Output
from lfx.schema.message import Message

class InputCombinerComponent(Component):
    display_name = "Input Combiner"
    description = "Combines user input and router output, ensuring both are present"
    icon = "combine"
    name = "InputCombiner"

    inputs = [
        MessageTextInput(
            name="user_input",
            display_name="User Input",
            info="Original user input",
            required=True,
        ),
        MessageTextInput(
            name="router_output",
            display_name="Router Output",
            info="Output from conditional router",
            required=True,
        ),
    ]

    outputs = [
        Output(
            display_name="Combined Output",
            name="combined",
            method="combine_inputs",
        ),
    ]

    async def combine_inputs(self) -> Message:
        # This ensures both inputs are available before execution
        user_text = str(self.user_input)
        router_text = str(self.router_output)
        
        combined = f"User Input: {user_text}\nRouter Output: {router_text}"
        return Message(text=combined)
```

Flow structure:
```
[User Input] ──┬──> [Conditional Router] ──┬──> [Input Combiner] ──> [Prompt]
               └────────────────────────────┘
```

### Solution 4: Use Sequential Flow Pattern

Restructure your flow to be explicitly sequential:

```
[User Input] ──> [Conditional Router] ──> [Pass Through / Store] ──> [Prompt]
```

Where the Pass Through component explicitly requires the router output before passing data to the Prompt.

## Understanding Langflow's Execution Model

### Key Concepts:

1. **Topological Sorting**: Components execute based on their position in the dependency graph, not based on how many inputs they have

2. **No Implicit Waiting**: A component does NOT automatically wait for all connected inputs - it waits for all PREDECESSOR components in the topological order

3. **Dynamic Prompt Variables**: Prompt template variables like `{user_input}` create dynamic input fields, but these don't enforce execution order

4. **Layer-Based Execution**: Components in the same "layer" (same topological level) may execute in parallel or in an undefined order

### How to Control Execution Order:

1. **Use Sequential Connections**: Create a clear chain where each component depends on the previous one
2. **Avoid Multiple Paths**: Don't connect the same source to both intermediate and final components
3. **Use Intermediate Components**: Add components that explicitly combine or validate inputs
4. **Test Execution Order**: Use the Langflow UI's execution visualization to see the actual order

## Recommended Approach for Your Use Case

For your specific situation with a Conditional Router feeding into a Prompt:

1. **Remove** the direct User Input → Prompt connection
2. **Only connect** Conditional Router → Prompt
3. **Ensure** your Router outputs contain all necessary information
4. **Structure your prompt template** to use only the router output:

```
Prompt Template:
You are processing a categorized request.

Input from router: {router_output}

Please respond appropriately based on the category.
```

This ensures the Prompt MUST wait for the Router to complete before it can execute, because the Router is now its only predecessor in the topological sort.

## Debugging Tips

1. **Check Component Logs**: Look at the execution order in Langflow's debug output
2. **Use Status Messages**: Set `self.status` in your components to see what's executing when
3. **Add Debug Components**: Create simple components that log when they execute
4. **Simplify Flow**: Test with minimal connections first, then add complexity

## References

- Graph execution: `src/lfx/src/lfx/graph/graph/base.py` (lines 2062-2086, 1832-1863)
- Topological sort: `src/lfx/src/lfx/graph/graph/utils.py` (line 461)
- Conditional routing: See `spec_sheet/022_conditional_routing_tool_architecture.md`
