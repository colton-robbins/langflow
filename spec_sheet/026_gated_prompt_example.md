# Gated Prompt Component - Usage Example

## Problem Scenario

You have a Conditional Router that needs to process user input, and you want the next agent to receive BOTH:
1. Fresh user input (not passed through the router)
2. The router's decision/output

## Flow Diagram

### BEFORE (Problematic Flow)

```
[User Input] ──┬──> [Conditional Router] ──┐
               │                            │
               └────────────────────────────┼──> [Standard Prompt]
                                            │    Template: {user_input} {router_output}
                                            │
                                            └──> PROBLEM: Prompt may execute before
                                                 Router finishes, missing {router_output}
```

**Issue**: Standard Prompt can execute as soon as it receives User Input, before Router completes.

### AFTER (Fixed with Gated Prompt)

```
[User Input] ──┬──> [Conditional Router] ──┐
               │                            │
               └────────────────────────────┼──> [Gated Prompt]
                                            │    - user_input: {user_input}
                                            └──> - gate_input: {router_output}
                                            
                                                 SOLUTION: Gated Prompt WAITS for both
                                                 inputs before executing
```

**Solution**: Gated Prompt has explicit inputs that create hard dependencies.

## Step-by-Step Setup

### Step 1: Add Components to Flow

1. **User Input** (Chat Input component)
2. **Conditional Router** (your existing router)
3. **Gated Prompt** (new component from `src/lfx/src/lfx/components/prompts/waiting_prompt.py`)
4. **Agent** or other downstream component

### Step 2: Connect Components

1. Connect **User Input** → **Conditional Router** `input_text`
2. Connect **User Input** → **Gated Prompt** `user_input`
3. Connect **Conditional Router** output → **Gated Prompt** `gate_input`
4. Connect **Gated Prompt** output → **Agent** or next component

### Step 3: Configure Gated Prompt Template

In the Gated Prompt's `template` field, write your prompt:

```
You are a helpful assistant processing a categorized request.

Original user question:
{user_input}

Category and routing information:
{gate_input}

Please provide an appropriate response based on the category.
```

### Step 4: Test Execution Order

Add status messages to verify order:

In your Conditional Router:
```python
def route_analytics(self) -> Message:
    text = extract_text_from_input(self.input_text)
    category = self.get_category(text)
    
    result = Message(text=f"Category: {category} | Input: {text}")
    self.status = f"Router completed: {category}"  # Shows in UI
    return result
```

The Gated Prompt will automatically show its status when it executes.

## Real-World Example

### Scenario: Customer Support Routing

You want to:
1. Get fresh user question
2. Route to appropriate knowledge base
3. Pass BOTH question and routing decision to agent

### Flow Configuration

**User Input**: "How do I reset my password?"

**Conditional Router** (routes based on keywords):
- Input: "How do I reset my password?"
- Output: "Category: ACCOUNT_MANAGEMENT | Keywords: password, reset"

**Gated Prompt**:
- Template:
  ```
  You are a customer support agent.
  
  Customer Question: {user_input}
  
  Routing Information: {gate_input}
  
  Based on the routing category, provide a helpful response using the appropriate knowledge base.
  ```

- Actual rendered prompt sent to Agent:
  ```
  You are a customer support agent.
  
  Customer Question: How do I reset my password?
  
  Routing Information: Category: ACCOUNT_MANAGEMENT | Keywords: password, reset
  
  Based on the routing category, provide a helpful response using the appropriate knowledge base.
  ```

**Agent** receives the complete context and can respond appropriately.

## Advanced Usage

### Multiple Gate Inputs

If you need to wait for MORE than 2 inputs, you can extend the Gated Prompt component:

```python
class MultiGatedPromptComponent(Component):
    inputs = [
        StrInput(name="template", required=True),
        MessageTextInput(name="input_1", required=True),
        MessageTextInput(name="input_2", required=True),
        MessageTextInput(name="input_3", required=True),
    ]
    
    async def build_prompt(self) -> Message:
        # Access ALL inputs to create dependencies
        i1 = str(self.input_1)
        i2 = str(self.input_2)
        i3 = str(self.input_3)
        template = str(self.template)
        
        # Replace placeholders
        result = template.replace("{input_1}", i1)
        result = result.replace("{input_2}", i2)
        result = result.replace("{input_3}", i3)
        
        return Message(text=result)
```

### Conditional Gating

If you want to gate only SOME of the time:

```python
async def build_prompt(self) -> Message:
    user_text = str(self.user_input)
    
    # Only wait for gate if user_text contains certain keywords
    if "analytics" in user_text.lower():
        gate_text = str(self.gate_input)  # This creates dependency
    else:
        gate_text = "No routing needed"
    
    # Build prompt...
```

## Troubleshooting

### Issue: Gated Prompt still executes too early

**Cause**: One of the inputs might not be properly connected or required.

**Solution**: 
1. Check all connections in the UI
2. Verify both inputs show as "connected" (not empty)
3. Add logging to confirm execution order

### Issue: Gated Prompt shows empty values

**Cause**: Router is returning empty Message or None.

**Solution**:
1. Check Router's output method returns a proper Message
2. Verify Router's routing logic matches your input
3. Add `self.status` logging in Router to see what it's outputting

### Issue: Want to use standard Prompt template syntax

**Cause**: Gated Prompt uses simple string replacement, not full template engine.

**Solution**: 
Extend the component to use Langflow's template system:

```python
from lfx.schema.message import Message

async def build_prompt(self) -> Message:
    # Access inputs to create dependencies
    user_text = str(self.user_input)
    gate_text = str(self.gate_input)
    
    # Use Message.from_template for advanced features
    return Message.from_template(
        template=self.template,
        user_input=user_text,
        gate_input=gate_text
    )
```

## Key Takeaways

1. **Gated Prompt guarantees execution order** by explicitly accessing all inputs
2. **Use it when you need fresh input** from multiple sources
3. **Simple to configure**: Just connect the inputs you want to wait for
4. **Extensible**: Easy to add more gate inputs if needed
5. **Debuggable**: Add status messages to verify execution order

## Next Steps

1. Copy the Gated Prompt component to your flow
2. Connect your User Input and Router to it
3. Configure your template with `{user_input}` and `{gate_input}`
4. Test and verify execution order
5. Extend as needed for your use case
