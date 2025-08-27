from dotenv import load_dotenv
load_dotenv()

# from llama_index.llms.openai import OpenAI
from llama_index.llms.google_genai import GoogleGenAI

from llama_index.core.agent.workflow import AgentWorkflow
from llama_index.tools.yahoo_finance import YahooFinanceToolSpec

def multiply(a: float, b: float) -> float:
    """Multiply two numbers and returns the product"""
    return a * b

def add(a: float, b: float) -> float:
    """Add two numbers and returns the sum"""
    return a + b

# llm = OpenAI(model="gpt-4o-mini")
llm = GoogleGenAI(
    model="gemini-2.0-flash",
    # api_key="some key",  # uses GOOGLE_API_KEY env var by default
)

finance_tools = YahooFinanceToolSpec().to_tool_list()
print ("FINANCIAL TOOLS ARE: ", finance_tools)
finance_tools.extend([multiply, add])
# we added more additonal capabiliteis to the finance_tool 

workflow = AgentWorkflow.from_tools_or_functions(
    finance_tools,
    llm=llm,
    system_prompt="You are an agent that can perform basic mathematical operations using tools."
)

async def main():
    response = await workflow.run(user_msg="What's the current stock price of NVIDIA?")
    print(response)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
