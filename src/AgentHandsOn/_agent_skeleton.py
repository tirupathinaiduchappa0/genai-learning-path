"""
UNIVERSAL AGENT SKELETON — Memorize this pattern.
Every interview agent follows these 5 steps.
"""

# ============================================
# STEP 1: IMPORTS (memorize these 6 lines)
# ============================================
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent
from dotenv import load_dotenv
import os

# ============================================
# STEP 2: LOAD ENV + CREATE LLM
# ============================================
load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY", "")

llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

# ============================================
# STEP 3: DEFINE TOOLS (this changes per task)
# ============================================
@tool
def my_tool(input_text: str) -> str:
    """Describe what this tool does. The agent reads this description."""
    # Your logic here
    return "result"

# ============================================
# STEP 4: CREATE AGENT
# ============================================
tools = [my_tool]
agent = create_react_agent(llm, tools, prompt="You are a helpful agent.")

# ============================================
# STEP 5: RUN
# ============================================
result = agent.invoke({"messages": [HumanMessage(content="Do the task")]})
print(result["messages"][-1].content)
