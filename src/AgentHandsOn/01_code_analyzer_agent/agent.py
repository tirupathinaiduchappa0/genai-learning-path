# ============================================
# STEP 1: IMPORTS
# ============================================
import os
import ast #Abstract Syntax Tree
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent  # FIX 1: langgraph, not langchain
from dotenv import load_dotenv

# ============================================
# STEP 2: LOAD ENV + CREATE LLM
# ============================================
load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY", "")
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)


# ============================================
# STEP 3: TOOL 1 — Scan directory for Python files
# ============================================
@tool
def scan_directory(directory_path: str) -> str:
    """Recursively scan a directory and return all python file paths."""
    py_files = []                                          # FIX 2: indented INSIDE function
    for root, dirs, files in os.walk(directory_path):
        for f in files:
            if f.endswith(".py"):
                py_files.append(os.path.join(root, f))
    if not py_files:                                       # FIX 2: indented INSIDE function
        return "No python files found."
    return "\n".join(py_files)                             # FIX 2: indented INSIDE function


# ============================================
# STEP 4: TOOL 2 — Analyze unused imports in a file
# ============================================
@tool
def analyze_unused_imports(file_path: str) -> str:
    """Read a python file and find all unused imports."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            source = f.read()

        tree = ast.parse(source)                           # FIX 3: outside 'with' block

        # Collect all imported names
        imports = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    name = alias.asname or alias.name
                    imports[name] = node.lineno
            elif isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    name = alias.asname or alias.name
                    imports[name] = node.lineno

        # Collect all used names
        used_names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                used_names.add(node.id)
            elif isinstance(node, ast.Attribute):
                if isinstance(node.value, ast.Name):
                    used_names.add(node.value.id)

        # Find unused = imported but not used
        unused = {name: line for name, line in imports.items() if name not in used_names}  # FIX 4: imports.items()

        if not unused:
            return f"{file_path}: No unused imports found."

        report = f"{file_path}:\n"
        for name, line in unused.items():
            report += f"  Line {line}: '{name}' is imported but never used\n"
        return report                                      # FIX 5: OUTSIDE the for loop

    except Exception as e:
        return f"Error while analyzing {file_path}: {str(e)}"


# ============================================
# STEP 5: TOOL 3 — Generate final report
# ============================================
@tool                                                      # FIX 6: at top level, not inside another function
def generate_report(analysis_results: str) -> str:
    """Take the analysis results and generate the summary."""
    lines = analysis_results.strip().split("\n")
    total_unused = sum(1 for line in lines if "is imported but never used" in line)

    report = "=" * 50 + "\n"
    report += "UNUSED IMPORTS REPORT\n"
    report += "=" * 50 + "\n\n"                            # FIX 7: += not =
    report += analysis_results + "\n"
    report += "=" * 50 + "\n"
    report += f"Total unused imports found: {total_unused}\n"
    report += "=" * 50 + "\n"
    return report


# ============================================
# STEP 5b: TOOL 4 — Remove commented-out code from a file
# ============================================
@tool
def remove_commented_code(file_path: str) -> str:
    """Remove all comment-only lines from a Python file. Keeps shebang (#!) lines."""
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()                              # readlines() returns a list of strings

    clean_code = []
    for line in lines:
        stripped = line.strip()
        # Skip comment-only lines, but keep shebang (#!) lines
        if stripped.startswith("#") and not stripped.startswith("#!"):
            continue                                       # skip this line, don't add to clean_code
        clean_code.append(line)                            # keep non-comment lines

    with open(file_path, "w", encoding="utf-8") as f:
        f.writelines(clean_code)                           # FIX: writelines() for list, write() for string

    removed_count = len(lines) - len(clean_code)
    return f"{file_path}: Removed {removed_count} commented lines."




# ============================================
# STEP 6: CREATE AGENT + RUN
# ============================================
tools = [scan_directory, analyze_unused_imports, remove_commented_code, generate_report]

agent = create_react_agent(
    llm,
    tools,
    prompt=(                                               # newer LangGraph uses 'prompt' not 'state_modifier'
        "You are a code analyzer agent. When given a directory path:\n"
        "1. First use scan_directory to find all Python files\n"
        "2. Then use analyze_unused_imports on EACH file\n"
        "3. Finally use generate_report to create a summary\n"
        "Be thorough — analyze every file found."
    ),
)

# Build absolute path to sample_project (works from any working directory)
sample_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_project")

result = agent.invoke(
    {"messages": [HumanMessage(content=(
        f"Analyze this project directory for unused imports: {sample_dir}"
    ))]},
    {"recursion_limit": 30},
)

print(result["messages"][-1].content)
