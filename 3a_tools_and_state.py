"""
Interactive Name-Remembering Agent

This script demonstrates:
- How to build an interactive conversational agent using LlamaIndex.
- The agent has a custom tool (`set_name`) that stores a user's name.
- The agent uses a language model (Google GenAI by default, OpenAI as an option).
- A Context object is used to maintain memory across turns.
- Memory is saved into a file (`history.json`) so the agent remembers even after restart.
- The script runs an interactive terminal loop where the user types a message
  and the agent responds.
"""

import asyncio
import json
import os
from dotenv import load_dotenv

# Import required LlamaIndex modules
from llama_index.llms.openai import OpenAI
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.core.agent.workflow import AgentWorkflow
from llama_index.core.workflow import Context, JsonSerializer

# ================================================================
# Load Environment Variables
# ================================================================
# The API keys for OpenAI or Google GenAI are expected to be stored
# in a `.env` file in the project root. The `load_dotenv()` function
# loads these values into environment variables automatically.
#
# Example `.env` file:
#   GOOGLE_API_KEY=your_google_api_key
#   OPENAI_API_KEY=your_openai_api_key
load_dotenv()

# ================================================================
# Language Model Setup
# ================================================================
# You can choose between OpenAI or Google GenAI.
# Uncomment the OpenAI line if you want to use OpenAI.
# The default here is Google GenAI.
#
# Note: The API key will be automatically pulled from the environment.
# For Google, the variable is GOOGLE_API_KEY.
# For OpenAI, the variable is OPENAI_API_KEY.

# llm = OpenAI(model="gpt-4o-mini")   # Option A: OpenAI
llm = GoogleGenAI(model="gemini-2.0-flash")  # Option B: Google GenAI

# ================================================================
# Custom Tool Definition
# ================================================================
# This is a simple tool that lets the agent store a user's name
# inside the Context state. The Context works like a memory store.
async def set_name(ctx: Context, name: str) -> str:
    """
    Save the user's name into the context state.

    Parameters:
    - ctx: The Context object that holds agent state across turns.
    - name: The string name provided by the user.

    Returns:
    - A confirmation string indicating that the name was stored.
    """
    # Get the current state (state is a dictionary)
    state = await ctx.get("state")

    # Store the new name under the key "name"
    state["name"] = name

    # Update the context with the modified state
    await ctx.set("state", state)

    # Return a confirmation message back to the agent
    return f"Name set to {name}"

# ================================================================
# Agent Workflow Setup
# ================================================================
# The workflow defines:
# - What tools the agent has access to
# - Which language model to use
# - The initial state (in this case, the name is "unset")
workflow = AgentWorkflow.from_tools_or_functions(
    [set_name],   # Register the custom tool
    llm=llm,
    system_prompt="You are a helpful assistant that can remember a user's name.",
    initial_state={"name": "unset"},  # Default state before a name is set
)

# ================================================================
# Context Persistence (Save/Load)
# ================================================================
# The Context allows memory during a single session.
# To persist memory across sessions, we serialize the Context
# to a JSON file called `history.json`.
# This way, if you stop the program and restart, the agent
# will still remember the previous state.
HISTORY_FILE = "history.json"

def load_context() -> Context:
    """
    Load saved context from history.json if available.
    If no file exists, create a new empty context.
    """
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r") as f:
            ctx_dict = json.load(f)
        return Context.from_dict(workflow, ctx_dict, serializer=JsonSerializer())
    return Context(workflow)

def save_context(ctx: Context):
    """
    Save the current context into history.json for persistence.
    """
    ctx_dict = ctx.to_dict(serializer=JsonSerializer())
    with open(HISTORY_FILE, "w") as f:
        json.dump(ctx_dict, f, indent=2)

# ================================================================
# Interactive Chat Loop
# ================================================================
# This is the main loop where the user interacts with the agent.
# The user types input at the "Client:" prompt.
# The agent processes the message using the workflow and responds.
# After each turn, the context is saved so the conversation memory
# is preserved even if the program exits.
async def chat_loop():
    # Load previous context if it exists, otherwise start fresh
    ctx = load_context()

    print("Interactive Agent Started. Type 'exit' or 'quit' to end.\n")

    while True:
        # Prompt the user for input
        user_msg = input("Client: ")

        # If the user types exit, quit the loop
        if user_msg.lower() in ["exit", "quit"]:
            print("Ending chat. Saving memory...")
            save_context(ctx)
            print("Memory saved. Goodbye!")
            break

        # Run the workflow with the user message
        response = await workflow.run(user_msg=user_msg, ctx=ctx)

        # Print the agent's response
        print(f"Agent: {response}\n")

        # Save the context after each interaction
        save_context(ctx)

# ================================================================
# Entry Point
# ================================================================
# The script starts execution here.
# asyncio.run is used because the workflow and chat loop are asynchronous.
if __name__ == "__main__":
    asyncio.run(chat_loop())


# https://github.com/run-llama/python-agents-tutorial/blob/main/3a_tools_and_state.py - Link to the actuall code 