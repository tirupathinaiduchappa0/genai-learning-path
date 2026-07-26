"""
===================================================================================
LESSON 7 — AGENT MEMORY & PROTOCOLS  (Senior DS / GenAI Real-Time Deep-Dive)
===================================================================================

COVERS (Section F of the question bank, part 2 of 2 — F1/F2/F3/F5 are in Lesson 6):
  F4. LangGraph memory & checkpointers (InMemorySaver, SQLiteSaver, PostgresSaver)
  F6. MCP server-client tool architecture
  F7. MCP vs A2A — what's the difference?

HOW TO READ:  THEORY -> RUNNABLE CODE -> INTERVIEW ANSWER -> RELATED CONCEPTS.

Run:  & "C:\\Projects\\Gen AI Udemy\\LangCGS\\.venv\\Scripts\\python.exe" "07_agent_memory_and_protocols.py"

NOTE ON WHAT'S REAL vs SIMULATED:
  - F4's InMemorySaver demo uses the REAL langgraph.checkpoint.memory.InMemorySaver
    against a real tiny compiled graph (it's installed in this venv).
  - SqliteSaver/PostgresSaver packages (langgraph-checkpoint-sqlite/-postgres) and
    the `mcp` SDK are NOT installed in this environment, so F4's DB-backed comparison
    and all of F6/F7 use faithful stdlib SIMULATIONS of their exact documented
    behavior/protocol — clearly labeled — so the mechanics are 100% accurate without
    adding new dependencies just for this lesson.
===================================================================================
"""

from __future__ import annotations

import json
import sqlite3
import sys
import tempfile
from dataclasses import dataclass, field
from typing import TypedDict

try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass


def banner(title: str) -> None:
    line = "=" * 83
    print("\n" + line + "\n" + title + "\n" + line)


def sub(title: str) -> None:
    print("\n" + "━" * 79 + "\n" + title + "\n" + "━" * 79)


# ===================================================================================
# F4. LANGGRAPH MEMORY & CHECKPOINTERS — DEEP DIVE
# ===================================================================================
#
# THE CONCEPT: CHECKPOINTERS ("SAVERS")
#   A LangGraph graph's STATE (the TypedDict/schema flowing through nodes) is, by
#   default, only alive for one `.invoke()` call — call it again and it starts fresh.
#   A CHECKPOINTER persists a SNAPSHOT of the graph's state after every step/node,
#   keyed by a `thread_id`. Passing the SAME thread_id on the next call resumes from
#   that saved state — this is how "conversation memory" and multi-turn chat actually
#   work under the hood: it's not the LLM remembering, it's the checkpointer replaying
#   accumulated state (usually the message list) back into the next invocation.
#   Checkpointers ALSO enable: time-travel/replay (inspect or resume from any past
#   checkpoint), human-in-the-loop (pause a graph at a checkpoint for approval), and
#   fault tolerance (resume after a crash from the last saved checkpoint instead of
#   restarting the whole run).
#
# InMemorySaver
#   Stores checkpoints in a plain Python dict, IN the running process's memory.
#   + Zero setup, fastest possible, perfect for local dev/testing/demos.
#   - LOST on process restart; NOT shared across multiple server processes/instances
#     (each worker has its own dict) — so it's unusable for real multi-instance
#     production chat where a user's next message might hit a different server.
#
# SqliteSaver
#   Persists checkpoints to a SQLite file (or in-memory SQLite for tests) using the
#   library `langgraph-checkpoint-sqlite`.
#   + Durable across restarts (a real file on disk); still zero external
#     infrastructure to run (no separate DB server); great for a single-instance app,
#     a desktop app, or local-first tools.
#   - SQLite has limited concurrent-WRITE support — fine for one process, risky for
#     multiple concurrent server instances writing to the same file; no built-in
#     replication/HA.
#
# PostgresSaver
#   Persists checkpoints to a real Postgres database using
#   `langgraph-checkpoint-postgres`.
#   + Full production durability, handles concurrent access from MANY server
#     instances correctly (that's what a real RDBMS is for), supports replication,
#     backups, and scaling — the standard choice for a multi-instance production
#     chat service.
#   - Requires running/operating an actual Postgres instance — real infrastructure
#     and ops overhead compared to the other two.
#
# COMPARISON — DURABILITY / SCALABILITY / CONCURRENCY / PRODUCTION-READINESS
#   InMemorySaver: durability NONE (gone on restart); scalability NONE (single
#     process only); concurrency fine within one process; production-readiness:
#     dev/test only.
#   SqliteSaver:   durability YES (disk file); scalability LOW (single writer
#     process, one file); concurrency limited (SQLite write-locking); production-
#     readiness: OK for single-instance / low-traffic / local apps.
#   PostgresSaver: durability YES (real DB, backups/replication); scalability HIGH
#     (many app instances against one DB); concurrency: proper multi-writer support;
#     production-readiness: YES, the standard for multi-instance production chat.
#
# WHICH TO CHOOSE FOR CHAT PERSISTENCE
#   Local dev/prototyping, a demo, or unit tests           -> InMemorySaver.
#   A single-instance app, desktop tool, or low-traffic app -> SqliteSaver.
#   A real production chat service behind a load balancer
#     with multiple app instances                           -> PostgresSaver.


class ChatState(TypedDict):
    messages: list[str]


def _demo_inmemory_saver_real() -> None:
    sub("F4a. InMemorySaver — REAL LangGraph checkpointer (runnable)")
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.graph import END, START, StateGraph

    def chat_node(state: ChatState) -> ChatState:
        # A trivial "agent": echoes how many turns have accumulated so far.
        turn = len(state["messages"]) + 1
        return {"messages": [f"turn {turn} response"]}

    builder = StateGraph(ChatState)
    builder.add_node("chat", chat_node)
    builder.add_edge(START, "chat")
    builder.add_edge("chat", END)

    # add_messages-style accumulation isn't used here for simplicity — we manage the
    # list append manually via the reducer default (replace) plus explicit reads.
    graph = builder.compile(checkpointer=InMemorySaver())

    thread_a = {"configurable": {"thread_id": "user-alice"}}
    thread_b = {"configurable": {"thread_id": "user-bob"}}

    # Two turns for Alice, on the SAME thread_id -> state persists between calls.
    state1 = graph.invoke({"messages": []}, config=thread_a)
    print(f"  Alice turn 1 -> {state1['messages']}")
    # LangGraph's checkpointer returns the LATEST full state; we manually accumulate
    # here by re-reading the saved snapshot to show the persistence explicitly.
    snapshot = graph.get_state(thread_a)
    accumulated = snapshot.values["messages"]
    state2 = graph.invoke({"messages": accumulated + ["(user's 2nd message)"]}, config=thread_a)
    print(f"  Alice turn 2 -> {state2['messages']} "
          f"(built on top of turn 1's saved state)")

    # A DIFFERENT thread_id (Bob) starts completely fresh — proves isolation per thread.
    state_bob = graph.invoke({"messages": []}, config=thread_b)
    print(f"  Bob (different thread_id) turn 1 -> {state_bob['messages']} "
          f"(isolated from Alice's history)")
    print("\n  -> the SAME thread_id resumes accumulated state; a DIFFERENT thread_id")
    print("     starts fresh. This IS how multi-user chat memory works.")


@dataclass
class SimpleCheckpoint:
    thread_id: str
    step: int
    state_json: str


class SqliteSaverSim:
    """Faithful behavioral simulation of langgraph-checkpoint-sqlite's core idea:
    checkpoints persisted to a REAL SQLite file on disk (survives process restart),
    keyed by thread_id. (Package not installed in this venv — same mechanics.)
    """

    def __init__(self, db_path: str) -> None:
        self.conn = sqlite3.connect(db_path)
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS checkpoints "
            "(thread_id TEXT, step INTEGER, state_json TEXT)"
        )
        self.conn.commit()

    def put(self, thread_id: str, step: int, state: dict) -> None:
        self.conn.execute(
            "INSERT INTO checkpoints VALUES (?, ?, ?)",
            (thread_id, step, json.dumps(state)),
        )
        self.conn.commit()

    def get_latest(self, thread_id: str) -> dict | None:
        row = self.conn.execute(
            "SELECT state_json FROM checkpoints WHERE thread_id=? "
            "ORDER BY step DESC LIMIT 1", (thread_id,)
        ).fetchone()
        return json.loads(row[0]) if row else None

    def close(self) -> None:
        self.conn.close()


def _demo_sqlite_saver_persistence() -> None:
    sub("F4b. SqliteSaver-style DURABILITY (runnable — real file, survives 'restart')")
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    saver1 = SqliteSaverSim(db_path)
    saver1.put("user-carol", step=1, state={"messages": ["hello"]})
    saver1.put("user-carol", step=2, state={"messages": ["hello", "how are you"]})
    print(f"  process 1: wrote 2 checkpoints to {db_path}")
    saver1.close()   # simulate the process/connection ending ("server restarted")

    saver2 = SqliteSaverSim(db_path)   # a brand-new connection/"process"
    restored = saver2.get_latest("user-carol")
    print(f"  process 2 (fresh connection, simulating a restart): "
          f"restored latest state -> {restored}")
    print("\n  -> unlike InMemorySaver, this state SURVIVED the 'restart' because it")
    print("     lives in a real file on disk, not just process memory.")
    saver2.close()

    import os
    os.remove(db_path)


def _demo_saver_comparison_table() -> None:
    sub("F4c. CHECKPOINTER COMPARISON (quick reference)")
    rows = [
        ("Dimension", "InMemorySaver", "SqliteSaver", "PostgresSaver"),
        ("Durability", "none (RAM only)", "yes (disk file)", "yes (real RDBMS)"),
        ("Multi-instance", "no", "risky (1 writer)", "yes"),
        ("Concurrency", "in-process only", "limited (locking)", "proper multi-writer"),
        ("Setup", "zero", "zero (just a file)", "run a Postgres server"),
        ("Best for", "dev/test/demo", "single-instance app", "production chat @scale"),
    ]
    widths = [14, 18, 20, 20]
    for r in rows:
        print("  " + "".join(str(c).ljust(w) for c, w in zip(r, widths)))


# INTERVIEW ANSWER (F4):
#   "A checkpointer persists the graph's state after every step, keyed by thread_id,
#    so the SAME thread_id resumes accumulated state on the next call — that's the
#    actual mechanism behind conversational memory, plus it enables time-travel,
#    human-in-the-loop pauses, and crash recovery. InMemorySaver keeps checkpoints in
#    a Python dict — zero setup but lost on restart and not shared across instances,
#    so dev/test only. SqliteSaver persists to a real file on disk — durable across
#    restarts, good for a single-instance app, but SQLite's write-locking makes it
#    risky for multiple concurrent server instances. PostgresSaver persists to a real
#    RDBMS — full durability plus correct multi-instance concurrency, which is what I'd
#    use for a production chat service behind a load balancer."
#
# RELATED CONCEPTS: thread_id as the multi-tenancy key; get_state/get_state_history
#   for time-travel and replay; interrupt() for human-in-the-loop checkpoints;
#   checkpoint pruning/retention policy at scale (you can't keep every checkpoint
#   forever); Redis-backed custom checkpointers as a fourth option for very high
#   throughput.


# ===================================================================================
# F6. MCP SERVER-CLIENT TOOL ARCHITECTURE
# ===================================================================================
#
# THE OVERALL ARCHITECTURE
#   MCP (Model Context Protocol) standardizes how an LLM application (the CLIENT,
#   e.g., an agent, Claude Desktop, an IDE assistant) connects to external TOOLS,
#   DATA, and CONTEXT exposed by an MCP SERVER. Before MCP, every app had to write
#   custom, bespoke integration code for every tool/data source it wanted to use —
#   MCP is the common protocol so ANY MCP client can talk to ANY MCP server without
#   custom glue code, the same way a browser can talk to any website via HTTP.
#   Transport: typically stdio (server as a local subprocess) or HTTP/SSE (server as
#   a remote service) — the client and server exchange JSON-RPC-style messages over
#   whichever transport is configured.
#
# HOW TOOLS ARE REGISTERED/DEFINED ON THE SERVER SIDE
#   The server author defines each tool with: a NAME, a DESCRIPTION (critical — this
#   is what the LLM reads to decide WHEN to call it), and an INPUT SCHEMA (typically
#   JSON Schema describing expected arguments and their types). The server exposes
#   these as part of its capabilities when a client connects. (This is exactly the
#   pattern behind FastMCP's `@tool` decorator — you write a Python function, decorate
#   it, and the framework auto-derives the schema from type hints + docstring.)
#
# HOW A CLIENT DISCOVERS WHAT TOOLS A SERVER PROVIDES
#   On connecting, the client sends a "list tools" request (e.g., `tools/list`); the
#   server responds with the full catalog: every tool's name, description, and input
#   schema. The client (or the LLM it wraps) now KNOWS what's available WITHOUT any
#   hard-coded integration — this discovery step is what makes MCP servers "plug and
#   play" for any compliant client.
#
# HOW A TOOL CALL FLOWS, CLIENT -> SERVER -> BACK
#   1) The LLM (inside the client app) decides, from the tool catalog + user's
#      request, that tool X should be called with certain arguments.
#   2) The client sends a "call tool" request (`tools/call`) with the tool name and
#      arguments, validated against the schema the server advertised.
#   3) The server executes the actual tool logic (e.g., hits an ERP API, queries a
#      DB) and returns a RESULT (or an error) back to the client.
#   4) The client feeds that result back to the LLM as an observation, continuing
#      the agent loop (exactly the ReAct-style Action -> Observation step from
#      Lesson 6).


@dataclass
class MCPTool:
    name: str
    description: str
    input_schema: dict


class MCPServerSim:
    """Faithful simulation of MCP server-side mechanics: tool registration + the
    tools/list and tools/call protocol operations. (The real `mcp` package isn't
    installed in this venv; this mirrors its documented request/response shape.)
    """

    def __init__(self) -> None:
        self._tools: dict[str, MCPTool] = {}
        self._impls: dict[str, callable] = {}

    def register_tool(self, name: str, description: str, input_schema: dict,
                      impl: callable) -> None:
        self._tools[name] = MCPTool(name, description, input_schema)
        self._impls[name] = impl

    def handle_list_tools(self) -> list[dict]:
        """Server-side handler for a client's `tools/list` request."""
        return [{"name": t.name, "description": t.description,
                "input_schema": t.input_schema} for t in self._tools.values()]

    def handle_call_tool(self, name: str, arguments: dict) -> dict:
        """Server-side handler for a client's `tools/call` request."""
        if name not in self._impls:
            return {"error": f"unknown tool: {name}"}
        try:
            result = self._impls[name](**arguments)
            return {"result": result}
        except Exception as e:  # surfaced back to the client as a tool error, not a crash
            return {"error": str(e)}


class MCPClientSim:
    """Faithful simulation of the client side: discover tools, then call one."""

    def __init__(self, server: MCPServerSim) -> None:
        self.server = server           # in real MCP this would be a transport (stdio/HTTP)
        self.catalog: list[dict] = []

    def discover_tools(self) -> list[dict]:
        self.catalog = self.server.handle_list_tools()
        return self.catalog

    def call_tool(self, name: str, **arguments) -> dict:
        return self.server.handle_call_tool(name, arguments)


def _demo_mcp_server_client() -> None:
    sub("F6. MCP SERVER <-> CLIENT PROTOCOL FLOW (runnable simulation)")

    server = MCPServerSim()
    server.register_tool(
        name="get_order_total",
        description="Look up the total dollar amount for a given order ID.",
        input_schema={"type": "object",
                      "properties": {"order_id": {"type": "integer"}},
                      "required": ["order_id"]},
        impl=lambda order_id: {4471: 89.99, 5002: 152.50}.get(order_id, 0.0),
    )
    server.register_tool(
        name="get_refund_policy",
        description="Return the current refund policy text.",
        input_schema={"type": "object", "properties": {}},
        impl=lambda: "Refunds within 30 days, items must be unused.",
    )

    client = MCPClientSim(server)

    print("1) Client connects and sends tools/list ->")
    catalog = client.discover_tools()
    for tool in catalog:
        print(f"     - {tool['name']}: {tool['description']}")
    print("   (the client/LLM now knows what's available with ZERO hard-coded")
    print("    integration code — this is the plug-and-play discovery step)")

    print("\n2) LLM decides to call get_order_total(order_id=4471) ->")
    response = client.call_tool("get_order_total", order_id=4471)
    print(f"   client sends tools/call -> server executes -> response: {response}")

    print("\n3) LLM calls an unregistered tool by mistake (error path) ->")
    bad_response = client.call_tool("delete_everything")
    print(f"   response: {bad_response}  (server rejects gracefully, no crash)")


# INTERVIEW ANSWER (F6):
#   "MCP standardizes how a client — an agent or app — connects to a server that
#    exposes tools, data, and context, over stdio or HTTP. The server registers each
#    tool with a name, a description the LLM reads to decide when to use it, and a
#    JSON-Schema input spec. On connecting, the client sends a tools/list request and
#    gets the full catalog back — that's discovery, and it's what makes any MCP
#    client work with any MCP server with no custom integration code. When the LLM
#    decides to use a tool, the client sends tools/call with the arguments, the
#    server executes the real logic and returns a result or error, and the client
#    feeds that back to the LLM as an observation, continuing the agent loop."
#
# RELATED CONCEPTS: FastMCP's @tool decorator auto-deriving schemas from type hints;
#   MCP resources vs tools (read-only context vs callable actions) vs prompts
#   (reusable prompt templates the server can expose); stdio vs HTTP+SSE transports;
#   auth/session management on top of the base protocol (your own Infor MCP server's
#   OAuth layer is exactly this).


# ===================================================================================
# F7. MCP vs A2A — WHAT'S THE DIFFERENCE?
# ===================================================================================
#
# WHAT PROBLEM EACH SOLVES
#   MCP (Model Context Protocol): standardizes how a SINGLE agent/LLM app connects
#   VERTICALLY to tools, data, and context it needs — "how does my agent talk to a
#   database, a file system, an ERP API, a search index." One client, one server,
#   exposing capabilities.
#   A2A (Agent-to-Agent protocol): standardizes how MULTIPLE AGENTS discover each
#   other and collaborate HORIZONTALLY — "how does Agent A ask Agent B (built by a
#   different team, possibly a different company, possibly a different framework) to
#   perform a task and return a result." It's about agent INTEROPERABILITY, not
#   agent-to-tool access.
#
# MCP'S ROLE
#   Connects an agent DOWN to its tools/data/context. Answers: "what can THIS agent
#   DO, and what does it know?"
#
# A2A'S ROLE
#   Connects agents ACROSS to each other. Answers: "which OTHER agents exist, what
#   are THEIR capabilities, and how do I hand off or delegate a (sub)task to one of
#   them and get a result back?" Includes agent discovery (capability cards/manifests
#   describing what an agent can do), task delegation, and result/status exchange —
#   conceptually similar to MCP's tool discovery, but between PEER AGENTS instead of
#   between an agent and a tool server.
#
# COMPLEMENTARY, NOT COMPETING
#   They solve DIFFERENT layers of the same overall system and are commonly used
#   TOGETHER: each individual agent uses MCP internally to reach ITS OWN tools/data,
#   while A2A lets that agent be discovered and invoked BY OTHER AGENTS as if it were
#   itself a capability. Analogy: MCP is like a function calling a library; A2A is
#   like two separate microservices calling each other's APIs.
#
# EXAMPLE SCENARIOS
#   MCP example: a customer-support agent uses MCP to query an order database tool
#   and a refund-policy knowledge-base tool — both are "its own" resources.
#   A2A example: that same support agent, mid-conversation, realizes the user's issue
#   is actually a shipping/logistics question outside its expertise, and uses A2A to
#   discover and delegate the task to a SEPARATE "Logistics Agent" (built by a
#   different team, possibly on a different framework entirely), which handles it
#   and returns a result the support agent relays to the user.


@dataclass
class AgentCard:
    """A2A-style capability manifest — how one agent advertises what it can do to
    OTHER agents (conceptually parallel to MCP's tool schema, but for agents)."""
    agent_id: str
    capabilities: list[str]
    description: str


class A2ARegistrySim:
    """Faithful simulation of A2A-style peer discovery + task delegation between
    independent agents. (No real `a2a` package involved — this mirrors the
    documented capability-discovery + delegate/return pattern.)
    """

    def __init__(self) -> None:
        self._agents: dict[str, AgentCard] = {}
        self._handlers: dict[str, callable] = {}

    def register_agent(self, card: AgentCard, handler: callable) -> None:
        self._agents[card.agent_id] = card
        self._handlers[card.agent_id] = handler

    def discover_agents_for(self, capability: str) -> list[AgentCard]:
        """A2A discovery: find OTHER agents that can handle a given capability."""
        return [c for c in self._agents.values() if capability in c.capabilities]

    def delegate_task(self, agent_id: str, task: str) -> dict:
        """A2A delegation: hand off a task to a peer agent, get its result back."""
        if agent_id not in self._handlers:
            return {"error": f"unknown agent: {agent_id}"}
        return {"agent_id": agent_id, "result": self._handlers[agent_id](task)}


def _demo_mcp_vs_a2a() -> None:
    sub("F7. MCP (agent<->tools) vs A2A (agent<->agent) — runnable side-by-side")

    # --- MCP layer: the support agent's OWN tools (from F6) ---
    print("MCP layer — the Support Agent's OWN tools/context (vertical):")
    server = MCPServerSim()
    server.register_tool("get_refund_policy", "Return refund policy text",
                         {"type": "object", "properties": {}},
                         lambda: "Refunds within 30 days.")
    mcp_client = MCPClientSim(server)
    mcp_client.discover_tools()
    print(f"  Support Agent discovers its OWN tool via MCP: "
          f"{[t['name'] for t in mcp_client.catalog]}")

    # --- A2A layer: discovering and delegating to a DIFFERENT, independent agent ---
    print("\nA2A layer — discovering and delegating to a PEER agent (horizontal):")
    registry = A2ARegistrySim()
    registry.register_agent(
        AgentCard("logistics-agent-v2", ["shipping_status", "carrier_tracking"],
                 "Handles shipping and carrier logistics questions."),
        handler=lambda task: f"Tracked: your package is in transit (task: {task!r})",
    )
    registry.register_agent(
        AgentCard("billing-agent-v1", ["invoice_lookup", "payment_dispute"],
                 "Handles billing and payment questions."),
        handler=lambda task: f"Billing resolved (task: {task!r})",
    )

    print("  Support Agent realizes it needs shipping help -> discovers via A2A:")
    matches = registry.discover_agents_for("shipping_status")
    for m in matches:
        print(f"    found peer agent: {m.agent_id} — {m.description}")

    print("\n  Support Agent DELEGATES the task to that peer agent via A2A:")
    result = registry.delegate_task("logistics-agent-v2",
                                    "where is order 4471's package")
    print(f"    delegation result: {result}")
    print("\n  -> MCP got the Support Agent its OWN data (refund policy). A2A let it")
    print("     find and hand off to a COMPLETELY SEPARATE agent it doesn't own or")
    print("     control the internals of — the two protocols solved different layers")
    print("     of the SAME support request, working together.")


# INTERVIEW ANSWER (F7):
#   "MCP solves agent-to-TOOL connectivity — how one agent reaches its own data and
#    capabilities, with tool discovery and structured calls. A2A solves agent-to-
#    AGENT interoperability — how independent agents, possibly built by different
#    teams or frameworks, discover each other's capabilities and delegate tasks to
#    one another, getting a result back. They're complementary layers of the same
#    system: an agent typically uses MCP internally for its own tools while being
#    discoverable and callable BY OTHER agents via A2A. A concrete example: a support
#    agent uses MCP to query its own refund-policy tool, then uses A2A to delegate a
#    shipping question to a separate logistics agent it doesn't own."
#
# RELATED CONCEPTS: multi-agent orchestration patterns (Lesson 6's supervisor/worker
#   sits on TOP of A2A-style delegation); capability cards/manifests as the A2A
#   analogue of MCP's tool schemas; when to build ONE agent with many MCP tools vs
#   MANY agents connected via A2A (organizational boundaries, independent deployment,
#   different tech stacks per team all push toward A2A).


# ===================================================================================
# GOLDEN LESSONS
# ===================================================================================
GOLDEN_LESSONS = """
  1. A checkpointer persists graph state per thread_id — same ID resumes, different ID starts fresh.
  2. That resume mechanic IS conversational memory — it's replayed state, not the LLM "remembering."
  3. InMemorySaver = dict in RAM: zero setup, gone on restart, not shared across instances. Dev/test only.
  4. SqliteSaver = real file on disk: durable across restarts, but single-writer-risky at scale.
  5. PostgresSaver = real RDBMS: durable + true multi-instance concurrency. The production choice.
  6. MCP = agent <-> TOOLS/DATA (vertical). Server registers tools with name+description+schema.
  7. MCP discovery: client sends tools/list -> gets the catalog -> zero hard-coded integration needed.
  8. MCP call flow: LLM decides -> client sends tools/call -> server executes -> result -> LLM observes.
  9. A2A = agent <-> AGENT (horizontal). Solves discovery + delegation between independent agents.
 10. MCP and A2A are complementary layers, not competitors — used TOGETHER in real multi-agent systems.
 11. Tool description quality matters as much as the schema — it's literally what the LLM reads to decide.
"""


if __name__ == "__main__":
    banner("LESSON 7 — AGENT MEMORY & PROTOCOLS")
    _demo_inmemory_saver_real()
    _demo_sqlite_saver_persistence()
    _demo_saver_comparison_table()
    _demo_mcp_server_client()
    _demo_mcp_vs_a2a()
    banner("GOLDEN LESSONS")
    print(GOLDEN_LESSONS)
