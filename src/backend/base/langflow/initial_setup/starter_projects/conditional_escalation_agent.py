"""Conditional Escalation Agent Flow

This flow demonstrates a two-agent system with conditional escalation:
1. First agent checks conversation memory and handles simple queries
2. Conditional router determines if escalation is needed
3. Second agent handles complex queries requiring tools

The first agent escalates when:
- Real-time information is needed
- External data lookups are required
- Tool usage is necessary
- Complex analysis is needed
"""

from lfx.components.flow_controls.conditional_router import ConditionalRouterComponent
from lfx.components.input_output import ChatInput, ChatOutput
from lfx.components.models_and_agents.agent import AgentComponent
from lfx.components.models_and_agents.memory import MemoryComponent
from lfx.components.models_and_agents import PromptComponent
from lfx.components.openai.openai_chat_model import OpenAIModelComponent
from lfx.components.tools import SearchAPIComponent
from lfx.graph import Graph


def conditional_escalation_agent_graph():
    """Create a graph with conditional agent escalation."""
    
    # Initialize components
    llm_model = OpenAIModelComponent(model_name="gpt-4o-mini")
    search_tool = SearchAPIComponent()
    
    # Chat input/output
    chat_input = ChatInput()
    chat_output = ChatOutput()
    
    # Memory component to retrieve conversation history
    memory_component = MemoryComponent()
    memory_component.set(
        mode="Retrieve",
        n_messages=10,
        order="Descending",
    )
    
    # First Agent: Memory Checker
    # This agent checks conversation history and determines if escalation is needed
    first_agent = AgentComponent()
    first_agent.set(
        model=llm_model.build_model,
        system_prompt="""You are a helpful assistant that checks conversation history and handles simple queries.

Your responsibilities:
1. Review the conversation history provided to you
2. If the user's query can be answered from memory or is a simple question, respond directly
3. If the user's query requires any of the following, respond with EXACTLY: "ESCALATE: [brief reason]"
   - Real-time information (current events, weather, stock prices, etc.)
   - External data lookups or searches
   - Tool usage (calculations, API calls, etc.)
   - Complex analysis requiring external resources

Examples:
- User asks "What did we discuss earlier?" → Answer from memory
- User asks "What's the weather today?" → Respond: "ESCALATE: Real-time weather data needed"
- User asks "Search for recent news about AI" → Respond: "ESCALATE: External search required"
- User asks "Hello, how are you?" → Answer directly

Remember: Only escalate when tools or external data are truly needed. Simple questions and memory-based queries should be handled directly.""",
        n_messages=10,
        # No tools - this agent doesn't call tools
    )
    
    # Conditional Router: Determines if escalation is needed
    # We'll check the first agent's output text to see if it contains "ESCALATE"
    conditional_router = ConditionalRouterComponent()
    conditional_router.set(
        input_text=first_agent.message_response,
        operator="contains",
        match_text="ESCALATE",
        case_sensitive=False,
        # When escalation is needed (true), pass the original query to second agent
        true_case_message=chat_input.message_response,
        # When no escalation (false), pass the first agent's response directly
        false_case_message=first_agent.message_response,
    )
    
    # Second Agent: Tool Caller
    # This agent handles escalated queries and has access to tools
    second_agent = AgentComponent()
    second_agent.set(
        model=llm_model.build_model,
        system_prompt="""You are an advanced assistant with access to tools for handling complex queries.

When you receive an escalated query, use the appropriate tools to:
- Search for current information using the search tool
- Look up real-time data
- Perform complex operations

Provide a comprehensive and accurate answer using the tools available. If the query requires information that tools can provide, use them. Otherwise, provide the best answer you can based on your knowledge.

Remember to:
- Use tools when they can provide better or more current information
- Explain what you're doing when using tools
- Provide clear, helpful responses""",
        tools=[search_tool.build_tool],
        n_messages=10,
    )
    
    # Connect the flow
    # Chat Input → First Agent → Conditional Router
    #                                    ├─ true_result → Second Agent → Chat Output
    #                                    └─ false_result → Chat Output
    
    # Set up connections
    first_agent.set(input_value=chat_input.message_response)
    
    # Connect router outputs
    # True case (escalation needed): Route original query to second agent
    second_agent.set(input_value=conditional_router.true_result)
    
    # Connect outputs to chat_output
    # The conditional router ensures only one branch executes via its stop() mechanism
    # So we connect both paths - the router will stop the unused branch
    # In the UI, both connections should be made:
    # - conditional_router.false_result → chat_output (for no escalation)
    # - second_agent.message_response → chat_output (for escalation)
    
    # For the graph definition, we connect the default path (no escalation)
    # The UI will allow connecting both paths, and the conditional router handles execution
    chat_output.set(input_value=conditional_router.false_result)
    
    # Create the graph
    return Graph(
        start=chat_input,
        end=chat_output,
        flow_name="Conditional Escalation Agent",
        description="Two-agent system with conditional escalation based on conversation memory and query complexity. First agent handles simple queries, second agent handles complex queries requiring tools.",
    )

