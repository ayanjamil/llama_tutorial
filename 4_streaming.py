"""
Clean Web Search Agent with Streaming Output
- Uses Tavily for web search
- Uses OpenAI as the LLM
- Streams readable responses to the terminal
"""

import os
import asyncio
from dotenv import load_dotenv

# Import LlamaIndex modules
from llama_index.llms.openai import OpenAI
from llama_index.core.agent.workflow import AgentWorkflow
from llama_index.tools.tavily_research import TavilyToolSpec

# Event classes for streaming
from llama_index.core.agent.workflow import (
    AgentInput,
    AgentOutput,
    ToolCallResult,
    AgentStream,
)

# Load environment variables (expects TAVILY_API_KEY in .env)
load_dotenv()

# Setup OpenAI LLM
llm = OpenAI(model="gpt-4o-mini")

# Setup Tavily web search tool
tavily_tool = TavilyToolSpec(api_key=os.getenv("TAVILY_API_KEY"))

# Create an agent workflow with the Tavily tool
workflow = AgentWorkflow.from_tools_or_functions(
    tavily_tool.to_tool_list(),
    llm=llm,
    system_prompt="You're a helpful assistant that can search the web for information."
)

async def main():
    print("Client: What's the weather like in San Francisco?")

    handler = workflow.run(user_msg="What's the weather like in San Francisco?")

    # Collect agent's answer as it streams
    agent_response = ""

    async for event in handler.stream_events():
        if isinstance(event, AgentStream):
            # Stream only the assistant text response
            agent_response += event.delta
            print(event.delta, end="", flush=True)

        elif isinstance(event, ToolCallResult):
            # Summarize tool calls instead of dumping raw data
            print(f"\n[Tool Used: {event.tool_name}]")
            print(f"Query: {event.tool_kwargs}")
            print(f"Result snippet: {str(event.tool_output)[:200]}...")  # show first 200 chars

    # Print final clean output
    final_response = await handler
    print("\n\nAgent (final):", str(final_response))

if __name__ == "__main__":
    asyncio.run(main())
