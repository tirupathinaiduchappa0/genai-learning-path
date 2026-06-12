# ============================================
# AGENT #3: DATABASE QUERY AGENT (Text-to-SQL)
# ============================================
# This agent converts natural language questions into SQL queries,
# executes them on a real SQLite database, and returns results.
#
# THE PATTERN (same as Agent #1):
#   Step 1: Imports
#   Step 2: Load env + Create LLM
#   Step 3: Define tools (get_schema, run_query, summarize_results)
#   Step 4: Create agent
#   Step 5: Run
# ============================================

# ============================================
# STEP 1: IMPORTS
# ============================================
import os
import sqlite3
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent
from dotenv import load_dotenv

# ============================================
# STEP 2: LOAD ENV + CREATE LLM
# ============================================
load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY", "")

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

# Database path (same folder as this script)
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ecommerce.db")


#Approach 1 (LangChain built-in)

#from langchain_community.utilities import SQLDatabase
#from langchain.chains import create_sql_query_chain

#db = SQLDatabase.from_uri("sqlite:///ecommerce.db")
#chain = create_sql_query_chain(llm, db)
#sql_query = chain.invoke({"question": "Top 3 customers by spending"})
#result = db.run(sql_query)

# Approach 2 (Agent with custom tools)

# ============================================
# STEP 3: TOOL 1 — Get database schema
# ============================================
@tool
def get_database_schema() -> str:
    """Get the database schema (table names, columns, types). Use this FIRST
    before writing any SQL query so you know what tables and columns exist."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Get all table names
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = cursor.fetchall()

    schema = ""
    for (table_name,) in tables:
        # Get column info for each table
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns = cursor.fetchall()

        schema += f"\nTABLE: {table_name}\n"
        schema += "  COLUMNS:\n"
        for col in columns:
            # col = (cid, name, type, notnull, default, pk)
            col_name = col[1]
            col_type = col[2]
            is_pk = " (PRIMARY KEY)" if col[5] else ""
            schema += f"    - {col_name} ({col_type}){is_pk}\n"

        # Get sample data (first 3 rows)
        cursor.execute(f"SELECT * FROM {table_name} LIMIT 3")
        rows = cursor.fetchall()
        if rows:
            schema += f"  SAMPLE DATA (first 3 rows): {rows}\n"

    conn.close()
    return schema


# ============================================
# STEP 4: TOOL 2 — Execute SQL query
# ============================================
@tool
def execute_sql_query(sql_query: str) -> str:
    """Execute a SQL query on the ecommerce database and return results.
    ONLY use SELECT statements. Never use DELETE, DROP, UPDATE, or INSERT.
    Always include LIMIT to prevent huge result sets."""
    # SECURITY: Block dangerous queries
    dangerous_keywords = ["DELETE", "DROP", "UPDATE", "INSERT", "ALTER", "TRUNCATE"]
    query_upper = sql_query.upper().strip()

    for keyword in dangerous_keywords:
        if keyword in query_upper:
            return f"BLOCKED: {keyword} queries are not allowed. Only SELECT is permitted."

    if not query_upper.startswith("SELECT"):
        return "BLOCKED: Only SELECT queries are allowed."

    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(sql_query)
        columns = [description[0] for description in cursor.description]
        rows = cursor.fetchall()
        conn.close()

        if not rows:
            return "Query returned no results."

        # Format as readable table
        result = f"Columns: {columns}\n"
        result += f"Rows returned: {len(rows)}\n\n"
        for row in rows:
            result += str(row) + "\n"

        return result

    except Exception as e:
        return f"SQL Error: {str(e)}. Please fix the query and try again."


# ============================================
# STEP 5: TOOL 3 — Validate SQL before execution
# ============================================
@tool
def validate_sql(sql_query: str) -> str:
    """Validate a SQL query for syntax errors WITHOUT executing it.
    Use this to check your query before running execute_sql_query."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        # EXPLAIN validates syntax without executing
        cursor.execute(f"EXPLAIN QUERY PLAN {sql_query}")
        plan = cursor.fetchall()
        conn.close()
        return f"Query is VALID. Execution plan: {plan}"
    except Exception as e:
        return f"Query is INVALID. Error: {str(e)}"


# ============================================
# STEP 6: CREATE AGENT + RUN
# ============================================
tools = [get_database_schema, execute_sql_query, validate_sql]

agent = create_react_agent(
    llm,
    tools,
    prompt=(
        "You are a database query agent for an eCommerce company.\n"
        "When a user asks a question about data:\n"
        "1. FIRST use get_database_schema to understand the tables and columns\n"
        "2. Write a SQL query to answer the question\n"
        "3. Use validate_sql to check the query syntax\n"
        "4. Use execute_sql_query to run it and get results\n"
        "5. Summarize the results in plain English for the user\n\n"
        "Rules:\n"
        "- ONLY use SELECT statements (never DELETE, DROP, UPDATE)\n"
        "- Always add LIMIT to prevent huge results\n"
        "- If a query fails, read the error and fix it\n"
        "- Present results clearly with numbers and context"
    ),
)

# ============================================
# STEP 7: TEST WITH MULTIPLE QUESTIONS
# ============================================
questions = [
    "Who are our top 3 customers by total spending?",
    "What is the most popular product category?",
    "Show me all orders that are still being processed",
    "What is the average order value?",
    "Which city has the most customers?",
]

# Run one question (change index to test different ones)
question = questions[0]
print(f"\n{'='*60}")
print(f"QUESTION: {question}")
print(f"{'='*60}\n")

result = agent.invoke(
    {"messages": [HumanMessage(content=question)]},
    {"recursion_limit": 20},
)

print(result["messages"][-1].content)
