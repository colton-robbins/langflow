# SOLUTION SUMMARY: Forcing Prompt to Wait for Router

## Your Problem

You have:
- User Input → Conditional Router → Prompt
- User Input → Prompt (direct connection for fresh input)

**Issue**: Prompt executes BEFORE Router finishes, so router's output is missing.

## The Solution: Use Gated Prompt Component

I've created a new component specifically for this: **Gated Prompt**

### Location
`src/lfx/src/lfx/components/prompts/waiting_prompt.py`

### What It Does

The Gated Prompt has two explicit input ports:
1. `user_input` - Your fresh user input
2. `gate_input` - The router's output (or any component you want to wait for)

**It CANNOT execute until BOTH inputs have values.**

### How to Use It

#### 1. Update Your Flow Connections

**Remove** the direct User Input → Standard Prompt connection.

**Add** these connections:
```
[User Input] ──┬──> [Conditional Router] ──┐
               │                            │
               └────────────────────────────┼──> [Gated Prompt]
                                            │    - user_input (from User Input)
                                            └──> - gate_input (from Router)
```

#### 2. Configure the Template

In the Gated Prompt's `template` field:

```
User asked: {user_input}

Router determined: {gate_input}

Please respond appropriately.
```

The component will replace `{user_input}` with the fresh user input and `{gate_input}` with the router's output.

#### 3. Connect to Your Agent

Connect Gated Prompt's output to your Agent or next component.

## Why This Works

The Gated Prompt **explicitly accesses both inputs** at the start of its execution:

```python
async def build_prompt(self) -> Message:
    # These two lines force the component to wait
    user_text = str(self.user_input)    # Must have user_input
    gate_text = str(self.gate_input)    # Must have gate_input
    
    # Then builds the prompt
    # ...
```

This creates **hard dependencies** that Langflow's execution system respects. The component literally cannot run until both inputs are available.

## Quick Start

1. **Add the component** to your flow (it's in `src/lfx/src/lfx/components/prompts/`)
2. **Connect**:
   - User Input → Gated Prompt `user_input`
   - User Input → Router `input_text`
   - Router output → Gated Prompt `gate_input`
3. **Set template** with `{user_input}` and `{gate_input}` placeholders
4. **Test** - the prompt will now wait for the router!

## Alternative Solutions

If you don't want to use the Gated Prompt:

### Option 1: Remove Direct Connection
- Only connect: User Input → Router → Prompt
- Modify Router to include original input in its output

### Option 2: Create Custom Combiner
- Create a component that takes both inputs
- Explicitly access both in the build method
- Output combined result

## Documentation

Full details in:
- `spec_sheet/026_forcing_prompt_to_wait_solution.md` - Technical explanation
- `spec_sheet/026_gated_prompt_example.md` - Usage examples
- `spec_sheet/025_execution_order_prompt_multiple_inputs.md` - Why this happens

## Key Insight

**Langflow doesn't have "required inputs" that force waiting.** Components execute based on topological sort order, not based on how many inputs they have connected.

The solution is to **explicitly access all inputs** in your component's build method, which creates dependencies that the execution system respects.

## Testing

Add this to your Router to verify order:

```python
def route_analytics(self) -> Message:
    text = extract_text_from_input(self.input_text)
    category = self.get_category(text)
    
    result = Message(text=f"[{category}] {text}")
    self.status = f"Router completed at {time.time()}"
    return result
```

The Gated Prompt will show its execution time in status, and you'll see it always comes AFTER the router.
