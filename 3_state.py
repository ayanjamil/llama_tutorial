"""
Interactive Agent Script with Memory + Persistence

This script creates a terminal-based conversational agent that:
- Performs basic math operations (add, multiply).
- Uses Yahoo Finance tools for financial queries.
- Uses Google GenAI as the language model (can be swapped with OpenAI).
- Maintains conversational context (remembers past interactions).
- Saves and restores context to/from a file (`history.json`) so the agent
  remembers even after the script is restarted.
"""

# ==============================
# --- Imports ---
# ==============================
import asyncio
import json
import os
from dotenv import load_dotenv

# Language models (choose between OpenAI or Google GenAI)
from llama_index.llms.openai import OpenAI
from llama_index.llms.google_genai import GoogleGenAI

# Yahoo Finance tool wrapper
from llama_index.tools.yahoo_finance import YahooFinanceToolSpec

# Agent workflow + context management
from llama_index.core.agent.workflow import AgentWorkflow
from llama_index.core.workflow import Context, JsonSerializer

# ==============================
# --- Load Environment Variables ---
# ==============================
# Loads API keys (e.g., GOOGLE_API_KEY or OPENAI_API_KEY) from .env file
load_dotenv()

# ==============================
# --- Math Helper Functions ---
# ==============================
def multiply(a: float, b: float) -> float:
    """Multiply two numbers and return the result."""
    return a * b

def add(a: float, b: float) -> float:
    """Add two numbers and return the result."""
    return a + b

# ==============================
# --- Language Model Setup ---
# ==============================
# Option A: OpenAI (commented out)
# llm = OpenAI(model="gpt-4o-mini")

# Option B: Google GenAI (default)
# Uses GOOGLE_API_KEY from .env automatically if not set explicitly
llm = GoogleGenAI(model="gemini-2.0-flash")

# ==============================
# --- Tools Setup ---
# ==============================
# Yahoo Finance tools (stock data, etc.)
finance_tools = YahooFinanceToolSpec().to_tool_list()

# Add our custom math tools to the agent’s toolset
finance_tools.extend([multiply, add])

# ==============================
# --- Agent Workflow Setup ---
# ==============================
# The AgentWorkflow ties everything together:
# - The tools (math + finance)
# - The language model (LLM)
# - A system prompt describing what the agent can do
workflow = AgentWorkflow.from_tools_or_functions(
    finance_tools,
    llm=llm,
    system_prompt="You are an agent that can perform math and answer finance questions."
)

# ==============================
# --- Context Persistence Setup ---
# ==============================
# By default, the agent forgets past conversations when the program ends.
# To fix this, we serialize the Context to a JSON file (`history.json`)
# and reload it on the next run.

HISTORY_FILE = "history.json"

def load_context() -> Context:
    """Load saved context from history.json if it exists, else start fresh."""
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r") as f:
            ctx_dict = json.load(f)
        # Rebuild the Context object from saved state
        return Context.from_dict(workflow, ctx_dict, serializer=JsonSerializer())
    # If no history file, start a new context
    return Context(workflow)

def save_context(ctx: Context):
    """Save the current conversation context to history.json."""
    ctx_dict = ctx.to_dict(serializer=JsonSerializer())
    with open(HISTORY_FILE, "w") as f:
        json.dump(ctx_dict, f, indent=2)

# Initialize context (load from file if available)
ctx = load_context()

# ==============================
# --- Interactive Chat Loop ---
# ==============================
async def chat_loop():
    """
    Main interactive loop:
    - User types input in terminal (Client).
    - Agent responds with output (Agent).
    - Context is updated and saved after each turn.
    """
    print(" Interactive Agent Started! Type 'exit' or 'quit' to stop.\n")

    while True:
        # --- Get input from user ---
        user_msg = input("Client: ")

        # --- Exit condition ---
        if user_msg.lower() in ["exit", "quit"]:
            print("Ending chat. Saving history...")
            save_context(ctx)  # Save state before quitting
            print("History saved. Goodbye!")
            break

        # --- Run the workflow with the user’s message ---
        response = await workflow.run(user_msg=user_msg, ctx=ctx)

        # --- Print agent response ---
        print(f"Agent: {response}\n")

        # --- Save context after each turn ---
        save_context(ctx)

# ==============================
# --- Entry Point ---
# ==============================
if __name__ == "__main__":
    # Run the chat loop asynchronously
    asyncio.run(chat_loop())




# OUTPUT sample 
# (llama) ayan@ayan--Laptop-14-ec0xxx:~/Desktop/dev/llama_tuts/python-agents-tutorial$ python 3_state.py 
# 💬 Interactive Agent Started! Type 'exit' or 'quit' to stop.

# Client: hi my name is ayan 
# Agent: Hello Ayan, nice to meet you! How can I help you today?


# Client: i want to know how much is 10+12 
# Agent: 10 + 12 = 22


# Client: great can you tell me the price of apple stocks
# Agent: The current price of Apple stock (AAPL) is $229.31.


# Client: exit
# 👋 Ending chat. Saving history...
# ✅ History saved. Goodbye!


# (llama) ayan@ayan--Laptop-14-ec0xxx:~/Desktop/dev/llama_tuts/python-agents-tutorial$ python 3_state.py 
#  Interactive Agent Started! Type 'exit' or 'quit' to stop.

# Client: say my name
# Agent: Ayan


# Client: exit
# Ending chat. Saving history...
# History saved. Goodbye!
