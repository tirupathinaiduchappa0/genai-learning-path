"""
🔗 Lesson 6.1 — LLMs, Prompt Templates, Chains & RAG Retrieval (Production-Grade)

This lesson ties EVERYTHING together. We've learned:
     Load documents
     Split into chunks
     Embed chunks into vectors
     Store in vector DB & retrieve

NOW we connect retrieval to an LLM to build a complete RAG pipeline:

    User Question
         ↓
    Embed the question
         ↓
    Search vector store for relevant chunks
         ↓
    Stuff those chunks into a prompt as "context"
         ↓
    Send prompt + context to LLM
         ↓
    LLM generates answer based ONLY on the context
         ↓
    Parse output → Return answer to user

TOPICS COVERED:
    1. LLM Setup — Groq (free API), Ollama (local, commented), OpenAI (paid, commented)
    2. ChatPromptTemplate — System messages, Human messages, variables
    3. LCEL Chains — The pipe (|) operator: prompt | llm | parser
    4. StrOutputParser — Converting LLM response to plain string
    5. RAG Retrieval Chain — Full pipeline: retriever → context → LLM → answer
    6. create_stuff_documents_chain & create_retrieval_chain (legacy helpers)
    7. Production LCEL RAG chain (modern approach)

Reference:
    https://python.langchain.com/docs/how_to/#chat-models
    https://python.langchain.com/docs/how_to/#prompts

Author: GenAI Learner
Date: 2026-04-12
"""

import logging
import os
from pathlib import Path

from dotenv import load_dotenv

# ─── Environment & Logging Setup ────────────────────────────────────────────
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

DATA_DIR: Path = Path(__file__).parent.parent / "DataIngestion" / "data"


# ═══════════════════════════════════════════════════════════════════════════════
# 🖥️ OLLAMA SETUP GUIDE (Local LLM — Commented, for future use)
# ═══════════════════════════════════════════════════════════════════════════════
#
# Ollama lets you run LLMs locally on your machine. No API key needed.
# But it requires disk space (~2-4 GB per model) and uses your CPU/GPU.
#
# INSTALLATION STEPS:
#   1. Download from: https://ollama.com/download
#      - Windows: Download the .exe installer and run it
#      - Mac: brew install ollama
#      - Linux: curl -fsSL https://ollama.com/install.sh | sh
#
#   2. Start the Ollama server:
#      $ ollama serve
#      (This runs in the background on http://localhost:11434)
#
#   3. Pull a model (downloads it to your machine):
#      $ ollama pull gemma:2b          # ~1.5 GB, lightweight, fast
#      $ ollama pull llama3.2:3b       # ~2 GB, good quality
#      $ ollama pull mistral           # ~4 GB, great quality
#      $ ollama pull mxbai-embed-large # Embedding model (~670 MB)
#
#   4. Test it:
#      $ ollama run gemma:2b "What is Python?"
#
#   5. Use in LangChain:
#      from langchain_community.llms import Ollama
#      llm = Ollama(model="gemma:2b")
#      response = llm.invoke("What is Python?")
#
# MODELS FOR DIFFERENT USE CASES:
#   gemma:2b       → Fastest, lowest memory (~1.5 GB), good for testing
#   llama3.2:3b    → Good balance of speed and quality (~2 GB)
#   mistral        → High quality, needs more RAM (~4 GB)
#   phi3:mini      → Microsoft's small model, good for coding (~2.3 GB)
#
# TO CHECK INSTALLED MODELS:
#   $ ollama list
#
# TO STOP OLLAMA:
#   $ ollama stop


# ═══════════════════════════════════════════════════════════════════════════════
# 1️⃣  LLM Setup — Groq (Our Primary LLM for this project)
# ═══════════════════════════════════════════════════════════════════════════════
#
# Groq provides FREE, ultra-fast LLM inference via API.
# We have GROQ_API_KEY in our .env file, so this works out of the box.
#
# Available Groq models:
#   llama-3.3-70b-versatile  → Best quality, 70B parameters
#   llama-3.1-8b-instant     → Fast, good quality, 8B parameters
#   gemma2-9b-it             → Google's Gemma 2, 9B parameters
#   mixtral-8x7b-32768       → Mixtral MoE, 32K context window
#
# WHY Groq over OpenAI?
#   - FREE (no credit card needed)
#   - Extremely fast (custom LPU hardware)
#   - Supports latest open-source models (Llama 3, Gemma, Mixtral)

def get_llm():
    """
    Create and return the LLM instance.

    Uses Groq as the primary LLM. Ollama and OpenAI are commented
    alternatives you can switch to based on your setup.
    """
    from langchain_groq import ChatGroq

    llm = ChatGroq(
        model="llama-3.1-8b-instant",   # Fast, good quality
        temperature=0.3,                  # Lower = more focused/deterministic
        max_tokens=512,                   # Cap response size (saves tokens)
    )
    logger.info("LLM initialized: Groq llama-3.1-8b-instant")
    return llm

    # ── ALTERNATIVE: Ollama (local, free, needs Ollama running) ──────────
    # from langchain_community.llms import Ollama
    # llm = Ollama(model="gemma:2b")
    # return llm

    # ── ALTERNATIVE: OpenAI (paid, highest quality) ──────────────────────
    # from langchain_openai import ChatOpenAI
    # llm = ChatOpenAI(model="gpt-4o", max_tokens=512)
    # return llm


# ═══════════════════════════════════════════════════════════════════════════════
# 2️⃣  ChatPromptTemplate — Structuring Messages for the LLM
# ═══════════════════════════════════════════════════════════════════════════════
#
# LLMs (especially chat models) expect messages in a specific format:
#
# ┌──────────────────┬────────────────────────────────────────────────────────┐
# │ Message Type     │ Purpose                                                │
# ├──────────────────┼────────────────────────────────────────────────────────┤
# │ SystemMessage    │ Sets the LLM's personality/role/rules.                │
# │                  │ "You are a helpful assistant specialized in Python."   │
# │                  │ The LLM follows these instructions for ALL responses. │
# ├──────────────────┼────────────────────────────────────────────────────────┤
# │ HumanMessage     │ The user's actual question or input.                  │
# │                  │ "What is a decorator in Python?"                      │
# │                  │ This is what the LLM responds to.                     │
# ├──────────────────┼────────────────────────────────────────────────────────┤
# │ AIMessage        │ Previous LLM responses (for conversation history).    │
# │                  │ Used to maintain context in multi-turn chats.         │
# └──────────────────┴────────────────────────────────────────────────────────┘
#
# ChatPromptTemplate.from_messages() creates a reusable template with VARIABLES.
# Variables are wrapped in {curly_braces} and filled in at runtime.
#
# PRODUCTION TIP:
#   Always use ChatPromptTemplate (not plain strings) because:
#   1. It validates that all required variables are provided
#   2. It properly formats messages for any LLM (OpenAI, Groq, Ollama)
#   3. It's composable with LCEL chains using the pipe operator

def demo_prompt_templates() -> None:
    """Demonstrate ChatPromptTemplate with system and human messages."""
    from langchain_core.prompts import ChatPromptTemplate

    # ── Basic prompt with system + human messages ────────────────────────
    # ("system", "...") → SystemMessage: sets the LLM's behavior
    # ("human", "...")  → HumanMessage: the user's question
    # {input} is a VARIABLE that gets filled when you call .invoke()
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert AI Engineer. Provide concise answers."),
        ("human", "{input}"),
    ])

    # Let's see what the template looks like
    logger.info("Prompt template: %s", prompt)
    logger.info("Required variables: %s", prompt.input_variables)
    # Output: ['input']

    # ── Format the prompt (fill in variables) ────────────────────────────
    # This is what gets sent to the LLM
    formatted = prompt.format_messages(input="What is LangChain?")
    for msg in formatted:
        logger.info("  %s: %s", msg.type, msg.content[:80])

    # ── Prompt with MULTIPLE variables ───────────────────────────────────
    # In RAG, we need both {context} and {input} (or {question})
    rag_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Answer based ONLY on the provided context."),
        ("human", "Context:\n{context}\n\nQuestion: {input}\n\nAnswer:"),
    ])
    logger.info("RAG prompt variables: %s", rag_prompt.input_variables)
    # Output: ['context', 'input']


# ═══════════════════════════════════════════════════════════════════════════════
# 3️⃣  LCEL Chains — The Pipe (|) Operator
# ═══════════════════════════════════════════════════════════════════════════════
#
# LCEL = LangChain Expression Language
# It uses the PIPE operator (|) to connect components into a chain.
#
# HOW THE PIPE WORKS (step by step):
#
#   chain = prompt | llm | output_parser
#
#   When you call chain.invoke({"input": "What is AI?"}):
#
#   Step 1: prompt receives {"input": "What is AI?"}
#           → Fills in the template variables
#           → Outputs: [SystemMessage("You are..."), HumanMessage("What is AI?")]
#
#   Step 2: llm receives the formatted messages
#           → Sends them to the LLM API (Groq/OpenAI/Ollama)
#           → Outputs: AIMessage(content="AI is a field of computer science...")
#
#   Step 3: output_parser receives the AIMessage
#           → Extracts just the text content (strips the AIMessage wrapper)
#           → Outputs: "AI is a field of computer science..." (plain string)
#
# EACH COMPONENT:
#   prompt        → Formats input into messages
#   llm           → Generates a response
#   output_parser → Cleans up the response
#
# WHY LCEL over old-style LLMChain?
#   - LLMChain is DEPRECATED in latest LangChain
#   - LCEL is composable, streamable, and supports async
#   - LCEL chains can be deployed directly with LangServe

def demo_lcel_chain() -> None:
    """
    Build a simple LCEL chain: prompt | llm | parser.

    This is the foundation of ALL LangChain applications.
    """
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate

    llm = get_llm()

    # ── Build the chain ──────────────────────────────────────────────────
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert AI Engineer. Give concise answers in 2-3 sentences."),
        ("human", "{input}"),
    ])

    # StrOutputParser extracts the plain text from the AIMessage
    # Without it, you'd get: AIMessage(content="...", response_metadata={...})
    # With it, you get just: "..."
    output_parser = StrOutputParser()

    # The PIPE operator connects them: prompt → llm → parser
    chain = prompt | llm | output_parser

    # ── Invoke the chain ─────────────────────────────────────────────────
    # .invoke() runs the chain end-to-end
    # Input: dict with the template variables
    # Output: plain string (thanks to StrOutputParser)
    response: str = chain.invoke({"input": "What is LangChain in one sentence?"})

    logger.info("--- Simple LCEL Chain ---")
    logger.info("Question: What is LangChain in one sentence?")
    logger.info("Answer: %s", response)

    # ── What happens WITHOUT StrOutputParser? ────────────────────────────
    chain_no_parser = prompt | llm
    raw_response = chain_no_parser.invoke({"input": "What is LangChain?"})
    logger.info("Without parser (raw AIMessage): type=%s", type(raw_response).__name__)
    logger.info("  content: %s...", raw_response.content[:100])
    logger.info("  response_metadata keys: %s", list(raw_response.response_metadata.keys()))
    # You get an AIMessage object with .content, .response_metadata, .usage_metadata
    # StrOutputParser just does: return ai_message.content


# ═══════════════════════════════════════════════════════════════════════════════
# 4️⃣  RAG Retrieval — How It ACTUALLY Works (The Full Picture)
# ═══════════════════════════════════════════════════════════════════════════════
#
# This is the CORE of what you're learning. Let me trace through every step:
#
# USER ASKS: "What is the speaker's view on democracy?"
#
# ┌─────────────────────────────────────────────────────────────────────────┐
# │ Step 1: EMBED THE QUESTION                                            │
# │   embeddings.embed_query("What is the speaker's view on democracy?")  │
# │   → [0.12, -0.34, 0.56, ...] (384-dim vector)                        │
# │                                                                        │
# │ Step 2: SEARCH VECTOR STORE                                           │
# │   vectorstore.similarity_search(query_vector, k=4)                    │
# │   → Compares query vector against ALL stored chunk vectors            │
# │   → Returns top 4 most similar chunks (Documents with text+metadata)  │
# │                                                                        │
# │ Step 3: FORMAT CONTEXT                                                │
# │   Take the 4 retrieved chunks and join them into one string:          │
# │   context = chunk1.page_content + "\n\n" + chunk2.page_content + ...  │
# │                                                                        │
# │ Step 4: BUILD THE PROMPT                                              │
# │   System: "Answer based ONLY on the provided context."                │
# │   Human: "Context: {context}\n\nQuestion: {question}\n\nAnswer:"      │
# │   → The LLM sees the relevant chunks + the question                   │
# │                                                                        │
# │ Step 5: LLM GENERATES ANSWER                                         │
# │   The LLM reads the context and generates an answer                   │
# │   It can ONLY use information from the context (not its training)     │
# │   This prevents hallucination!                                         │
# │                                                                        │
# │ Step 6: PARSE OUTPUT                                                  │
# │   StrOutputParser extracts the plain text answer                      │
# │   → "The speaker believes democracy must be made safe..."             │
# └─────────────────────────────────────────────────────────────────────────┘
#
# TWO WAYS TO BUILD THIS:
#
# A) MODERN WAY (LCEL) — Recommended, flexible, production-ready
#    chain = (
#        {"context": retriever | format_docs, "input": RunnablePassthrough()}
#        | prompt
#        | llm
#        | StrOutputParser()
#    )
#
# B) LEGACY WAY — create_stuff_documents_chain + create_retrieval_chain
#    These are convenience functions that build the same thing internally.
#    They still work but LCEL gives you more control.
#    "stuff" means: stuff ALL retrieved documents into the prompt at once.

def demo_rag_retrieval_chain() -> None:
    """
    Full RAG pipeline: Load → Split → Embed → Store → Retrieve → Answer.

    This is the complete end-to-end RAG chain using LCEL (modern approach).
    """
    from langchain_community.document_loaders import TextLoader
    from langchain_community.vectorstores import FAISS
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.runnables import RunnablePassthrough
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    # ── Step 1: Load ─────────────────────────────────────────────────────
    loader = TextLoader(str(DATA_DIR / "speech.txt"), encoding="utf-8")
    docs = loader.load()
    logger.info("Step 1 — Loaded %d document(s)", len(docs))

    # ── Step 2: Split ────────────────────────────────────────────────────
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    logger.info("Step 2 — Split into %d chunks", len(chunks))

    # ── Step 3: Embed + Store ────────────────────────────────────────────
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    logger.info("Step 3 — Created FAISS vector store")

    # ── Step 4: Create Retriever ─────────────────────────────────────────
    # as_retriever() wraps the vector store as a LangChain Retriever.
    # When you call retriever.invoke("some question"), it:
    #   1. Embeds the question
    #   2. Searches the vector store
    #   3. Returns top-k Documents
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # ── Step 5: Build the RAG prompt ─────────────────────────────────────
    # This prompt tells the LLM to answer ONLY from the context.
    # {context} will be filled with retrieved chunks.
    # {input} will be filled with the user's question.
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a helpful assistant. Answer the question based ONLY on "
         "the provided context. If the context doesn't contain the answer, "
         "say 'I don't have enough information to answer that.'"),
        ("human",
         "Context:\n{context}\n\n"
         "Question: {input}\n\n"
         "Answer:"),
    ])

    # ── Step 6: Build the LCEL RAG chain ─────────────────────────────────
    llm = get_llm()
    output_parser = StrOutputParser()

    # Helper function: joins retrieved Documents into a single string
    def format_docs(docs: list) -> str:
        """Join document page_contents with double newlines."""
        return "\n\n".join(doc.page_content for doc in docs)

    # THE LCEL RAG CHAIN — Here's how each piece connects:
    #
    # {"context": retriever | format_docs, "input": RunnablePassthrough()}
    #   ↑ This is a RunnableParallel — it runs TWO things in parallel:
    #
    #   "context": retriever | format_docs
    #     → Takes the input string (the question)
    #     → Passes it to retriever (which embeds + searches)
    #     → Gets back list[Document]
    #     → Passes to format_docs (which joins them into one string)
    #     → Result: one big context string
    #
    #   "input": RunnablePassthrough()
    #     → Takes the input string (the question)
    #     → Passes it through unchanged
    #     → Result: the original question string
    #
    #   Combined output: {"context": "chunk1\n\nchunk2\n\n...", "input": "What is...?"}
    #
    # | prompt
    #   → Receives {"context": "...", "input": "..."}
    #   → Fills in the template variables
    #   → Outputs formatted messages
    #
    # | llm
    #   → Receives messages, sends to Groq API
    #   → Outputs AIMessage
    #
    # | output_parser
    #   → Extracts plain text from AIMessage
    #   → Outputs: "The speaker believes..."
    rag_chain = (
        {"context": retriever | format_docs, "input": RunnablePassthrough()}
        | prompt
        | llm
        | output_parser
    )

    # ── Step 7: Ask questions! ───────────────────────────────────────────
    question1 = "What does the speaker believe about democracy?"
    answer1 = rag_chain.invoke(question1)
    logger.info("--- RAG Chain (LCEL) ---")
    logger.info("Q: %s", question1)
    logger.info("A: %s", answer1)

    question2 = "What is the speaker's view on the German people?"
    answer2 = rag_chain.invoke(question2)
    logger.info("Q: %s", question2)
    logger.info("A: %s", answer2)

    # ── Show what the retriever actually returns ─────────────────────────
    logger.info("--- What the retriever returns for question 1 ---")
    retrieved_docs = retriever.invoke(question1)
    for i, doc in enumerate(retrieved_docs):
        logger.info("  Chunk %d (%d chars): %s...", i, len(doc.page_content), doc.page_content[:80])


# ═══════════════════════════════════════════════════════════════════════════════
# 5️⃣  Legacy Helpers: create_stuff_documents_chain & create_retrieval_chain
# ═══════════════════════════════════════════════════════════════════════════════
#
# These are convenience functions from older LangChain versions.
# They build the same RAG chain internally, but with less control.
#
# "stuff" = stuff ALL retrieved documents into the prompt at once.
# (As opposed to "map_reduce" which processes each doc separately.)
#
# HOW THEY WORK:
#
#   document_chain = create_stuff_documents_chain(llm, prompt)
#     → Creates a chain that takes {"context": [Document, ...], "input": "..."}
#     → Joins all Documents into one string and fills the prompt
#     → Sends to LLM and returns the answer
#
#   retrieval_chain = create_retrieval_chain(retriever, document_chain)
#     → Wraps the document_chain with a retriever
#     → When you call retrieval_chain.invoke({"input": "..."}):
#       1. Retriever fetches relevant Documents
#       2. Passes them as "context" to the document_chain
#       3. Returns {"input": "...", "context": [...], "answer": "..."}
#
# NOTE: These still work in current LangChain but the LCEL approach
# (shown above) is preferred because it's more flexible and explicit.

def demo_legacy_retrieval_chain() -> None:
    """
    RAG using create_stuff_documents_chain + create_retrieval_chain.

    NOTE: In LangChain v1.x+, these functions have been REMOVED.
    They lived in langchain.chains which no longer exists.
    The LCEL approach (shown above) is the ONLY way going forward.

    This function is kept as DOCUMENTATION of the old pattern.
    If you're reading older tutorials/notebooks that use these,
    now you know: replace them with the LCEL chain from demo 3 above.

    OLD WAY (deprecated, does NOT work in LangChain v1.x):
        from langchain.chains.combine_documents import create_stuff_documents_chain
        from langchain.chains import create_retrieval_chain
        document_chain = create_stuff_documents_chain(llm, prompt)
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        result = retrieval_chain.invoke({"input": "..."})

    NEW WAY (LCEL, works in all versions):
        rag_chain = (
            {"context": retriever | format_docs, "input": RunnablePassthrough()}
            | prompt | llm | StrOutputParser()
        )
        answer = rag_chain.invoke("...")
    """
    logger.info("create_stuff_documents_chain & create_retrieval_chain are")
    logger.info("REMOVED in LangChain v1.x. Use the LCEL approach instead.")
    logger.info("See demo_rag_retrieval_chain() above for the modern pattern.")


# ═══════════════════════════════════════════════════════════════════════════════
# 🚀 Main — Run all demos
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logger.info("=" * 70)
    logger.info("🔗 LLMs, PROMPTS, CHAINS & RAG RETRIEVAL LESSON")
    logger.info("=" * 70)

    logger.info("\n🔹 1. ChatPromptTemplate — System & Human messages")
    demo_prompt_templates()

    logger.info("\n🔹 2. LCEL Chain — prompt | llm | parser")
    demo_lcel_chain()

    logger.info("\n🔹 3. Full RAG Retrieval Chain (LCEL — modern approach)")
    demo_rag_retrieval_chain()

    logger.info("\n🔹 4. Legacy Retrieval Chain (create_stuff_documents_chain)")
    demo_legacy_retrieval_chain()

    logger.info("\n" + "=" * 70)
    logger.info("✅ RAG Chains lesson complete!")
    logger.info("Next up: LCEL Deep Dive — RunnableParallel, RunnableLambda, etc.")
    logger.info("=" * 70)
