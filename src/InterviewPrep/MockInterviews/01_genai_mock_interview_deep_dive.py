"""
===================================================================================
MOCK INTERVIEW DEEP DIVE — PART 2 (Sections 5-16, Fully Elaborated)
===================================================================================

Continuation of 01_genai_mock_interview_deep_dive.py
These sections are expanded with full depth, examples from the transcript,
and interview-ready explanations.

SECTIONS IN THIS FILE:
    5.  RAG System Design — Invoice Chatbot (Full Design Discussion)
    6.  OCR Challenges with Structured Data (Tables, Invoices)
    7.  Retrieval Methods — Cosine Similarity vs KNN (The Real Truth)
    8.  Cheating Detection — Outlier Detection + Graph Databases
    9.  Attention Mechanism — The N² Complexity Problem
    10. Python Internals — Dictionary O(1) Lookup (Hash Tables)
    11. ML Metrics — Precision, Recall, F1 Score (With Real Scenarios)
    12. Object Detection — YOLO (How It Works, Bounding Boxes)
    13. Fine-Tuning vs RAG — Complete Decision Framework
    14. System Design — Accuracy Multiplication Problem
    15. AI Models Landscape (What You Must Know)
    16. Key Interview Takeaways (The Interviewer's Wisdom)
===================================================================================
"""


# =================================================================================
# SECTION 1: ML vs GENERATIVE AI — When to Use Which
# =================================================================================
"""
INTERVIEW QUESTION: "When would you use traditional ML vs Generative AI?"

TRADITIONAL ML:
    - Predictions (will this customer churn?)
    - Classification (spam or not spam?)
    - Clustering (group similar customers)
    - Regression (predict house price)
    - Anomaly detection (fraud detection)
    - Recommendation systems
    USE WHEN: Structured data, predictions, classification. Task is WELL-DEFINED.

GENERATIVE AI:
    - Text generation (descriptions, emails, summaries)
    - Code generation, Image generation, Conversation
    - Translation, Summarization, Q&A over documents (RAG)
    USE WHEN: Generate new content, understand natural language, unstructured data.

DECISION FRAMEWORK:
    "Predicting from structured data?" -> ML
    "Generating or understanding unstructured content?" -> GenAI
    "Simple classification with labeled data?" -> ML (GenAI is overkill)
    "Need reasoning, explanation, or creation?" -> GenAI

EXAMPLE: "500 movie reviews, classify as positive/negative/neutral"
    With labeled data -> Fine-tuned BERT or LSTM (cheaper, faster, offline)
    Without labeled data -> GPT zero-shot (expensive but instant, no training)

INTERVIEW ANSWER:
    "ML for prediction/classification with structured data — cheaper and faster.
    GenAI for generation, understanding, reasoning with unstructured data.
    For simple classification with labeled data, ML is sufficient — GenAI is overkill."
"""


# =================================================================================
# SECTION 2: EMBEDDINGS vs ENCODINGS
# =================================================================================
"""
ENCODING:
    - Converts categories to numbers. Does NOT capture meaning.
    - One-Hot: "cat" -> [1,0,0], "dog" -> [0,1,0] (sparse, no relationship)
    - Label: "cat" -> 0, "dog" -> 1 (implies false ordering)

EMBEDDING:
    - Dense vector capturing SEMANTIC MEANING. Similar words = similar vectors.
    - "king" -> [0.2, 0.8, 0.1, 0.9, ...] (384-1536 dimensions)
    - "queen" -> [0.2, 0.7, 0.1, 0.9, ...] (very similar!)
    - Famous: king - man + woman = queen

    | Aspect | Encoding | Embedding |
    | Purpose | Categories to numbers | Capture meaning |
    | Dimensions | High (sparse) | Low (dense) |
    | Semantic | None | Yes |
    | Learned | No (rule-based) | Yes (from data) |

EMBEDDING MODELS:
    - Word2Vec (word-level, older)
    - GloVe (Global Vectors, pre-computed lookup)
    - BERT (contextual — same word, different embedding based on context)
    - Sentence-Transformers (sentence-level, used in RAG)
    - OpenAI text-embedding-3-small (API-based)
"""


# =================================================================================
# SECTION 3: LSTM vs TRANSFORMERS
# =================================================================================
"""
LSTM (Long Short-Term Memory):
    - Processes text SEQUENTIALLY (word by word, left to right)
    - Has gates: forget gate, input gate, output gate
    - Good at remembering long-term dependencies
    - SLOW (can't parallelize — must process word 1 before word 2)
    - Used for: sentiment analysis, time series, sequence prediction

TRANSFORMER:
    - Processes ALL words SIMULTANEOUSLY (parallel)
    - Uses ATTENTION mechanism (each word attends to every other word)
    - Much faster (parallelizable on GPUs)
    - Better at capturing long-range dependencies
    - Foundation of GPT, BERT, Claude, Llama
    - Used for: everything NLP (replaced LSTMs almost entirely)

WHY TRANSFORMERS WON:
    1. Parallelization (LSTM is sequential, Transformer is parallel)
    2. Better long-range dependencies (attention sees all words at once)
    3. Scalable (can train on massive datasets efficiently)
    4. Transfer learning (pre-train once, fine-tune for any task)

WHEN TO STILL USE LSTM:
    - Very small datasets (Transformer needs lots of data)
    - Real-time streaming data (process one token at a time)
    - Resource-constrained environments (LSTM is lighter)

INTERVIEW ANSWER:
    "LSTMs process text sequentially — word by word — which is slow and struggles
    with very long sequences. Transformers process all words simultaneously using
    attention, which is parallelizable and captures long-range dependencies better.
    That's why all modern LLMs (GPT, BERT, Claude) use Transformers. LSTMs are
    still useful for small datasets or streaming scenarios."
"""

# =================================================================================
# SECTION 5: RAG SYSTEM DESIGN — INVOICE CHATBOT (Full Design)
# =================================================================================
"""
INTERVIEW SCENARIO (from the mock interview):
    "I have 100 documents. Each document is one page long. Assume these are
    invoices. Your task is to build a chatbot that answers questions about
    these invoices. How would you design RAG for this?"

THE INTERVIEWER'S HIDDEN TESTS:
    1. Do you understand when RAG is even needed?
    2. Do you know the challenges with structured documents (tables)?
    3. Can you make practical decisions (not over-engineer)?
    4. Do you understand the tradeoffs?

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

STEP 1: ANALYZE THE PROBLEM FIRST (before jumping to solution)

    What are invoices?
    - STRUCTURED documents (tables with rows and columns)
    - Contains: invoice number, date, vendor name, line items, quantities, prices, totals
    - Financial data = ACCURACY IS CRITICAL (wrong number = wrong decision)
    - One page each = SHORT documents
    - 100 documents = SMALL dataset

    Key questions to ask yourself:
    - Do I even need RAG for 100 one-page documents?
    - What's special about invoices vs regular text documents?
    - What kind of questions will users ask?

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

STEP 2: DO WE EVEN NEED RAG?

    Let's calculate:
    - 100 invoices × 1 page × ~500 tokens per page = 50,000 tokens total
    - GPT-4o context window = 128,000 tokens
    - Claude context window = 200,000 tokens

    50,000 tokens FITS in the context window!

    So we have THREE options:
    Option A: Context stuffing (put all 100 invoices in one prompt)
        Pros: Simple, no RAG infrastructure needed
        Cons: Expensive per query ($0.125/query), slow, "lost in middle" problem

    Option B: RAG (retrieve only relevant invoices per query)
        Pros: Cheaper per query, precise, scalable
        Cons: More infrastructure, retrieval might miss relevant docs

    Option C: Hybrid (metadata filtering + RAG)
        Pros: Best of both worlds
        Cons: Slightly more complex

    BEST ANSWER: "For 100 one-page invoices, I'd use RAG with metadata filtering.
    Even though they fit in context, RAG gives better precision (only relevant
    invoices are retrieved) and lower cost per query. But I'd acknowledge that
    context stuffing is a valid option for this small dataset."

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

STEP 3: LOADING — THE TABLE PROBLEM

    Invoices have TABLES. This is the critical challenge.

    WRONG approach: Simple OCR or PyPDFLoader
        Input (PDF table):
            | Item        | Qty | Price  | Total  |
            | Widget A    | 10  | $5.00  | $50.00 |
            | Widget B    | 5   | $8.00  | $40.00 |

        OCR output (loses structure):
            "Item Qty Price Total Widget A 10 $5.00 $50.00 Widget B 5 $8.00 $40.00"

        Problem: You can't tell which price belongs to which item!
        If user asks "What's the price of Widget B?" — the system might
        return $5.00 (Widget A's price) because it's just flat text.

    RIGHT approach: Layout-aware parsing
        Option 1: LlamaParse — understands table structure
        Option 2: Unstructured with table extraction mode
        Option 3: Convert to Markdown (preserves table format)
        Option 4: GPT-4V (vision model reads the table image → JSON)

        Markdown output (preserves structure):
            | Item | Qty | Price | Total |
            |------|-----|-------|-------|
            | Widget A | 10 | $5.00 | $50.00 |
            | Widget B | 5 | $8.00 | $40.00 |

        Now the LLM can correctly answer "Widget B costs $8.00"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

STEP 4: CHUNKING STRATEGY FOR INVOICES

    Normal documents: chunk by 1000 characters with 200 overlap.
    Invoices: DIFFERENT strategy needed.

    Option A: One chunk per invoice (since each is one page)
        - Each invoice becomes ONE chunk
        - Simple, preserves all context within an invoice
        - 100 invoices = 100 chunks (very manageable)

    Option B: Chunk by sections (header, line items, totals)
        - More precise retrieval
        - But might lose context (header has vendor name, line items have prices)

    BEST FOR INVOICES: Option A (one chunk per invoice)
    Why: Invoices are short (one page). Splitting them loses context.
    The user might ask "What's the total for invoice from Vendor X?" —
    you need both the vendor name (header) AND the total (footer) in one chunk.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

STEP 5: EMBEDDING + STORAGE + METADATA

    Embed each invoice as one chunk.
    Store in FAISS (100 docs = tiny, no need for Pinecone).

    CRITICAL: Add METADATA to each chunk:
        {
            "invoice_number": "INV-2024-0042",
            "date": "2024-03-15",
            "vendor": "Acme Corp",
            "total_amount": 1250.00,
            "content": "... full invoice text ..."
        }

    Why metadata matters:
    - User asks "Show me invoices from March 2024"
      → Filter by date BEFORE vector search (faster, more accurate)
    - User asks "What's invoice INV-2024-0042 about?"
      → Exact match on invoice_number (no vector search needed)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

STEP 6: RETRIEVAL STRATEGY

    For financial documents: HYBRID SEARCH is essential.

    Why not just vector search?
    - "What's invoice INV-2024-0042?" — this needs EXACT keyword match
    - Vector search might return a semantically similar invoice (wrong one!)
    - Invoice numbers, dates, vendor names need KEYWORD matching

    Hybrid = Vector search + Keyword search (BM25)
    - Vector: finds semantically relevant invoices
    - Keyword: finds exact matches (invoice numbers, vendor names)
    - Combine results using RRF (Reciprocal Rank Fusion)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

STEP 7: GENERATION

    Strict prompt for financial accuracy:
        "You are an invoice analysis assistant.
        Answer ONLY from the provided invoice data.
        If a number or fact is not in the context, say 'Not found in the invoices.'
        NEVER guess or estimate financial numbers.
        Always cite which invoice your answer comes from."

    Why so strict?
    - Financial data: wrong number = wrong business decision
    - Hallucinated invoice total could cause payment errors
    - Citations let users verify the answer

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

STEP 8: THE INTERVIEWER'S TRAP QUESTIONS

    "Do you need MapReduce for 100 documents?"
    ANSWER: "No. 100 one-page documents is tiny. MapReduce is for millions of
    documents across distributed systems. For 100 docs, a simple Python loop
    processes them in seconds. Don't over-engineer."

    "Do you need a vector database like Pinecone?"
    ANSWER: "No. FAISS in-memory is perfect for 100 documents. Pinecone is for
    millions of vectors with persistence needs. For 100 docs, FAISS loads in
    milliseconds and searches instantly."

    KEY LESSON: Don't over-engineer. Match the solution to the problem scale.

INTERVIEW ANSWER (complete):
    "For 100 one-page invoices, I'd design a lightweight RAG system:
    1. Load with a layout-aware parser to preserve table structure.
    2. Keep each invoice as one chunk (they're short, don't split).
    3. Add rich metadata (invoice number, date, vendor, amount).
    4. Store in FAISS with hybrid search (vector + keyword).
    5. Use metadata filtering for exact queries (invoice number, date range).
    6. Strict generation prompt that prevents hallucination of financial numbers.
    I'd also consider whether RAG is even needed — 100 short docs could fit in
    context, but RAG gives better precision and lower cost per query."
"""


# =================================================================================
# SECTION 6: OCR CHALLENGES WITH STRUCTURED DATA
# =================================================================================
"""
FROM THE INTERVIEW:
    Interviewer: "If you're extracting text using OCR from invoices which have
    tables, do you see any issue with this approach?"

THE PROBLEM IN DETAIL:

    A PDF invoice looks like this visually:
    ┌─────────────────────────────────────────────────┐
    │  INVOICE #INV-2024-0042                          │
    │  Date: March 15, 2024                            │
    │  Vendor: Acme Corp                               │
    │                                                  │
    │  ┌──────────┬─────┬────────┬─────────┐          │
    │  │ Item     │ Qty │ Price  │ Total   │          │
    │  ├──────────┼─────┼────────┼─────────┤          │
    │  │ Widget A │ 10  │ $5.00  │ $50.00  │          │
    │  │ Widget B │ 5   │ $8.00  │ $40.00  │          │
    │  │ Widget C │ 20  │ $2.50  │ $50.00  │          │
    │  └──────────┴─────┴────────┴─────────┘          │
    │                                                  │
    │  Subtotal: $140.00                               │
    │  Tax (10%): $14.00                               │
    │  TOTAL: $154.00                                  │
    └─────────────────────────────────────────────────┘

    Simple OCR (PyPDFLoader) extracts this as FLAT TEXT:
        "INVOICE INV-2024-0042 Date March 15 2024 Vendor Acme Corp
         Item Qty Price Total Widget A 10 5.00 50.00 Widget B 5 8.00
         40.00 Widget C 20 2.50 50.00 Subtotal 140.00 Tax 10 14.00
         TOTAL 154.00"

    PROBLEMS WITH FLAT TEXT:
    1. No table structure — can't tell which price belongs to which item
    2. Numbers are disconnected from their labels
    3. "5.00" appears twice — is it Widget A's price or Widget C's total?
    4. Row relationships are lost (Widget B + 5 + $8.00 + $40.00 = one row)

WHY THIS MATTERS FOR RAG:
    If user asks: "What's the price per unit of Widget C?"
    - With flat text: LLM might say "$5.00" (wrong — that's Widget A)
    - With structured text: LLM correctly says "$2.50"

    Financial accuracy depends on preserving table structure!

SOLUTIONS (from simple to advanced):

    SOLUTION 1: Layout-aware parsers
        - LlamaParse (by LlamaIndex) — understands tables natively
        - Unstructured.io with partition_pdf(strategy="hi_res")
        - Docling (by IBM) — excellent table extraction
        These output markdown tables that preserve row/column relationships.

    SOLUTION 2: Convert PDF to Markdown/LaTeX
        - Tools like Marker, Nougat convert PDFs to markdown
        - Tables become proper markdown tables
        - LLMs understand markdown tables perfectly

    SOLUTION 3: Dedicated table extraction
        - Camelot (Python library) — extracts tables as DataFrames
        - Tabula — similar, Java-based
        - These give you structured data (CSV/DataFrame) from tables

    SOLUTION 4: Vision models (most powerful)
        - Send the PDF page as an IMAGE to GPT-4V or Claude Vision
        - Prompt: "Read this invoice table and return as JSON"
        - Output: structured JSON with proper row/column mapping
        - Most accurate but most expensive

    SOLUTION 5: The candidate's answer (from transcript) — LaTeX conversion
        - Convert PDF to LaTeX format
        - LaTeX preserves all structural elements (tables, figures)
        - AI can render and understand LaTeX natively
        - Good answer! Shows awareness of the problem.

WHICH TO USE WHEN:
    Simple tables (2-3 columns) → Layout-aware parser (LlamaParse)
    Complex tables (merged cells, nested) → Vision model (GPT-4V)
    High volume (thousands of invoices) → Camelot + custom pipeline
    Mixed content (text + tables + images) → Unstructured.io

INTERVIEW ANSWER:
    "Simple OCR loses table structure — numbers become disconnected from their
    columns. For invoices, I'd use a layout-aware parser like LlamaParse or
    Unstructured with hi-res strategy. These output markdown tables that
    preserve row/column relationships. For complex tables with merged cells,
    I'd use GPT-4V to read the page as an image and return structured JSON.
    The key insight: if you lose table structure at ingestion, no amount of
    good retrieval or generation can fix it downstream."
"""


# =================================================================================
# SECTION 7: RETRIEVAL — COSINE SIMILARITY vs KNN (The Real Truth)
# =================================================================================
"""
FROM THE INTERVIEW:
    Candidate said: "I'd use KNN to retrieve relevant chunks."
    Interviewer challenged: "Why KNN? Why not just cosine similarity?
    At the end of the day, KNN uses Euclidean distance which is also
    measuring semantic similarity. Why the two-stage process?"

    The candidate couldn't answer clearly. Here's the REAL answer:

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

WHAT IS COSINE SIMILARITY?
    Measures the ANGLE between two vectors.
    - 1.0 = vectors point in same direction (identical meaning)
    - 0.0 = vectors are perpendicular (unrelated)
    - -1.0 = vectors point in opposite directions

    Formula: cos(θ) = (A · B) / (|A| × |B|)

    Example:
        query_embedding = [0.2, 0.8, 0.1]
        doc1_embedding  = [0.2, 0.7, 0.1]  → cosine = 0.99 (very similar!)
        doc2_embedding  = [0.9, 0.1, 0.8]  → cosine = 0.35 (not similar)

    KEY PROPERTY: Scale-invariant. Only direction matters, not magnitude.
    "king" and "KING" would have same direction even if different magnitudes.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

WHAT IS KNN (K-Nearest Neighbors)?
    An ALGORITHM that finds the K closest points to a query point.
    It needs a DISTANCE METRIC to measure "closeness":
    - Euclidean distance (straight-line distance)
    - Cosine distance (1 - cosine similarity)
    - Manhattan distance (sum of absolute differences)

    So KNN is NOT a distance metric itself — it's an algorithm that USES
    a distance metric. If you use cosine distance inside KNN, you're
    essentially doing cosine similarity with extra steps.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

THE INTERVIEWER'S POINT (why this question was asked):

    The candidate said "I'd use KNN" but couldn't explain WHY it's better
    than simple cosine similarity. The truth is:

    FOR RAG RETRIEVAL, THEY ARE ESSENTIALLY THE SAME THING.

    What vector databases actually do:
    1. You give a query vector
    2. They compute similarity (cosine) between query and ALL stored vectors
    3. They return the top-K most similar ones

    That's it. No clustering. No KNN algorithm. Just similarity + top-K.

    FAISS, Pinecone, Chroma — they all do this internally.
    When you write:
        retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    You're saying: "Find the 4 most similar vectors using cosine similarity."

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

EUCLIDEAN vs COSINE — WHEN DOES IT MATTER?

    COSINE SIMILARITY:
    - Measures DIRECTION (angle between vectors)
    - Ignores magnitude (length of vector)
    - Best for: text embeddings (where direction = meaning)
    - "king" and "KING" have same direction = same meaning

    EUCLIDEAN DISTANCE:
    - Measures STRAIGHT-LINE distance between points
    - Affected by magnitude
    - Best for: when magnitude matters (e.g., user activity levels)

    FOR RAG: Always use COSINE SIMILARITY.
    Why: Embedding models produce vectors where DIRECTION captures meaning.
    Two chunks about the same topic point in the same direction regardless
    of how long the text is.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

WHAT VECTOR DATABASES ACTUALLY USE (under the hood):

    For EXACT search (small datasets like 100 invoices):
    - Brute force: compute cosine similarity with every vector
    - O(n) — fine for small datasets

    For APPROXIMATE search (millions of vectors):
    - HNSW (Hierarchical Navigable Small World) — graph-based
    - IVF (Inverted File Index) — cluster-based
    - These are APPROXIMATE nearest neighbor algorithms
    - They sacrifice tiny accuracy for massive speed gains
    - FAISS uses IVF + HNSW internally

INTERVIEW ANSWER:
    "For RAG retrieval, I use the vector database's built-in similarity search
    which is cosine similarity returning top-K results. I don't need a separate
    KNN algorithm because FAISS and Pinecone already implement optimized
    approximate nearest neighbor search internally. The key metric is cosine
    similarity because embedding models encode meaning in vector DIRECTION,
    not magnitude. For small datasets (100 docs), brute-force cosine similarity
    is fast enough. For millions of vectors, HNSW or IVF provides approximate
    results in milliseconds."
"""


# =================================================================================
# SECTION 8: CHEATING DETECTION — OUTLIERS + GRAPH DATABASES
# =================================================================================
"""
FROM THE INTERVIEW:
    "I have a coding platform. Cheaters are: (1) people scoring impossibly high,
    (2) groups with identical solutions. How do you detect them?"

    This tests: ML thinking, data analysis, graph databases, system design.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

TYPE 1: INDIVIDUAL CHEATERS (impossibly high scores)

    DETECTION METHOD 1 — Statistical Outlier Detection:
        - Calculate mean and standard deviation of all scores
        - Anyone > 3 standard deviations above mean = outlier
        - Visualization: BOX PLOT shows outliers clearly

        Box Plot:
            |----[====|====]----| ○  ○
            min  Q1  median Q3  max  outliers

        Q1 = 25th percentile, Q3 = 75th percentile
        IQR = Q3 - Q1
        Outlier = anything > Q3 + 1.5 × IQR

    DETECTION METHOD 2 — Time Analysis:
        - Problem difficulty: Hard (expected time: 45 min)
        - User solved it in: 3 minutes
        - Red flag! Either genius or cheating.
        - Compare solve time vs problem difficulty vs user's historical performance

    DETECTION METHOD 3 — Behavioral Analysis:
        - Window switching (left the platform during contest)
        - Copy-paste patterns (pasted large code blocks)
        - Typing speed anomalies (suddenly types 200 WPM)
        - Tab switching frequency

    DETECTION METHOD 4 — Historical Comparison:
        - User's rating: 1200 (intermediate)
        - Problem difficulty: 2500 (expert)
        - User solved it perfectly → suspicious
        - Compare current performance vs historical average

    THE INTERVIEWER'S TRAP: "What if they've been cheating from day 1?"
        - Their history is consistently high (no anomaly to detect)
        - Solution: behavioral analysis (window switching, typing patterns)
        - Or: compare with population (if EVERYONE at 1200 rating can't solve it,
          but this one person always can → suspicious)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

TYPE 2: GROUP CHEATERS (identical solutions)

    DETECTION METHOD 1 — Code Similarity:
        - Don't just compare text (variable names can be changed)
        - Compare AST (Abstract Syntax Tree) — structure of the code
        - If two solutions have identical AST but different variable names → copied
        - Tools: MOSS (Measure of Software Similarity), JPlag

    DETECTION METHOD 2 — Submission Time Correlation:
        - User A submits at 10:15:03
        - User B submits at 10:15:47 (44 seconds later)
        - User C submits at 10:16:12 (69 seconds later)
        - All three have identical solutions → B and C copied from A

    DETECTION METHOD 3 — Cluster Analysis:
        - Group solutions by similarity score
        - If a cluster of 5+ users have >95% similar code → cheating group
        - The earliest submitter in the cluster = likely source

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

GRAPH DATABASE FOR FINDING THE SOURCE:

    WHY GRAPH DB?
        - Relationships between cheaters form a NETWORK
        - "Who copied from whom?" is a GRAPH problem
        - Graph queries are natural: "Find all users who copied from User A"

    STRUCTURE:
        Nodes = Users
        Edges = "copied from" (directed: copier → source)
        Edge weight = similarity score

    HOW TO DETERMINE DIRECTION (who is the source):
        1. TIMESTAMP: Who submitted FIRST? → likely the source
        2. RATING: Higher-rated user → more likely to be the original solver
        3. SOLVE HISTORY: User who consistently solves hard problems → source

    GRAPH QUERIES:
        - "Who has the most incoming edges?" → most copied-from person (source)
        - "Find all paths from User X" → trace the cheating chain
        - "Find connected components" → identify cheating groups

    THE COMPLEX CASE (from interview):
        A solves the problem. B copies from A. C copies from A.
        But D copies from B, and E copies from C.
        The graph shows: A → B → D, A → C → E
        Graph centrality analysis reveals A as the ultimate source.

    Graph databases: Neo4j, Amazon Neptune, ArangoDB

INTERVIEW ANSWER:
    "For individual cheaters: statistical outlier detection (box plots, z-scores)
    combined with time analysis and behavioral signals (window switching, typing
    patterns). For group cheating: AST-based code similarity + submission time
    correlation. To find the source: build a directed graph where edges point
    from copier to source (based on timestamps). The node with highest in-degree
    or earliest submission is the original solver. Graph databases like Neo4j
    are ideal for querying these relationships — 'find all users who directly
    or indirectly copied from User A.'"
"""


# =================================================================================
# SECTION 9: ATTENTION MECHANISM — THE N² COMPLEXITY PROBLEM
# =================================================================================
"""
FROM THE INTERVIEW:
    "Attention is an n-squared complexity algorithm. How would you solve this?"
    The candidate couldn't answer. Here's the full explanation.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

WHAT IS ATTENTION? (The Basics)

    Attention is the core mechanism inside Transformers (GPT, BERT, Claude, Llama).
    It answers: "For each word in the sentence, which OTHER words should I pay
    attention to?"(Bidirectional Encoder Representations from Transformers)

    Example: "The cat sat on the mat because it was tired"
    When processing "it" → attention looks at ALL other words and decides:
    "it" should pay most attention to "cat" (because "it" refers to the cat).

    HOW IT WORKS (simplified):
    1. Each word becomes 3 vectors: Query (Q), Key (K), Value (V)
    2. For each word, compute: "How relevant is every other word to me?"
       Score = Q × K^T (dot product of query with all keys)
    3. Apply softmax to get attention weights (probabilities)
    4. Multiply weights × Values to get the output

    The formula: Attention(Q, K, V) = softmax(Q × K^T / sqrt(d)) × V

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

WHY IT'S O(n²):

    For a sequence of n tokens:
    - Each token's Query must be compared with EVERY token's Key
    - That's n × n = n² comparisons

    REAL NUMBERS:
        n = 100 tokens    → 10,000 comparisons (instant)
        n = 1,000 tokens  → 1,000,000 comparisons (fast)
        n = 10,000 tokens → 100,000,000 comparisons (slow)
        n = 100,000 tokens → 10,000,000,000 comparisons (VERY expensive!)
        n = 1,000,000 tokens → 1,000,000,000,000 comparisons (impossible naively)

    This is why:
    - Long-context models are EXPENSIVE to run
    - Processing a 100-page document costs more than a 1-page document
    - Token limits exist (you can't just feed infinite text)

    MEMORY is also O(n²):
    - The attention matrix (n × n) must be stored in GPU memory
    - For n = 100,000: that's 100,000 × 100,000 × 4 bytes = 40 GB just for attention!

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SOLUTIONS (Research Directions):

    SOLUTION 1: SPARSE ATTENTION (Longformer, BigBird)
        Instead of attending to ALL tokens, attend only to:
        - Nearby tokens (local window of size w)
        - A few global tokens (like [CLS] token)
        - Random tokens (for diversity)

        Complexity: O(n × w) instead of O(n²) where w << n
        Tradeoff: Might miss long-range dependencies

        Example: Longformer uses sliding window of 512 tokens
        + global attention on first token. Works for documents up to 16K tokens.

    SOLUTION 2: LINEAR ATTENTION (Performer, Linear Transformer)
        Approximate the softmax(Q × K^T) using kernel tricks.
        Instead of computing the full n×n matrix, decompose it into
        two smaller matrices that multiply in O(n) time.

        Complexity: O(n) instead of O(n²)
        Tradeoff: Approximation — slightly less accurate than full attention

    SOLUTION 3: FLASH ATTENTION (Most widely used in practice!)
        Doesn't reduce the MATHEMATICAL complexity (still O(n²)).
        But optimizes HOW the computation happens on GPU hardware.

        The insight: GPU memory has levels (fast SRAM vs slow HBM).
        Standard attention reads/writes the full n×n matrix to slow memory.
        Flash Attention computes attention in TILES that fit in fast SRAM.

        Result: 2-4x faster in practice, uses much less GPU memory.
        Used by: GPT-4, Claude, Llama, Mistral — basically everyone.

    SOLUTION 4: SLIDING WINDOW ATTENTION (Mistral)
        Each token only attends to the last W tokens (window size).
        Token at position 1000 attends to tokens 500-1000 (if W=500).

        Complexity: O(n × W) where W is fixed (e.g., 4096)
        Tradeoff: Can't directly attend to very early tokens.
        But: information propagates through layers (token 1000 → 500 → 1)

    SOLUTION 5: CHUNKED/HIERARCHICAL ATTENTION
        Split the sequence into chunks. Attend within chunks (local).
        Then attend across chunk summaries (global).

        Like reading a book: read each chapter (local), then connect chapters (global).

    SOLUTION 6: MULTI-QUERY ATTENTION (MQA) / GROUPED-QUERY ATTENTION (GQA)
        Instead of separate K and V for each attention head,
        SHARE K and V across multiple heads.
        Reduces memory and computation significantly.
        Used by: Llama 2, Llama 3, Mistral

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

WHICH IS USED IN PRACTICE (2026):

    Flash Attention → EVERYONE uses this (it's a no-brainer optimization)
    GQA (Grouped-Query Attention) → Llama 3, Mistral (reduces memory)
    Sliding Window → Mistral (enables long context cheaply)
    Sparse Attention → Longformer (for very long documents)

    The trend: Models are getting longer context windows (1M+ tokens)
    by combining multiple techniques: Flash Attention + GQA + Sliding Window.

INTERVIEW ANSWER:
    "Attention is O(n²) because each token computes similarity with every other
    token. For 100K tokens, that's 10 billion comparisons. Solutions include:
    Flash Attention (optimizes GPU memory access — 2-4x faster, used by everyone),
    Sparse Attention (attend only to nearby + global tokens — Longformer),
    Sliding Window (attend to last W tokens — Mistral), and Linear Attention
    (approximate in O(n) — Performer). In practice, Flash Attention + GQA +
    Sliding Window is the combination most modern models use to enable
    long context windows efficiently."
"""


# =================================================================================
# SECTION 10: PYTHON INTERNALS — DICTIONARY O(1) LOOKUP
# =================================================================================
"""
FROM THE INTERVIEW:
    "Why is Python dictionary lookup O(1)? What methodologies make it efficient?"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

HOW A HASH TABLE WORKS (What's inside a Python dict):

    Think of a hash table like a HOTEL with numbered rooms.

    When you check in (store a key-value pair):
    1. Your name ("Tirupathi") gets HASHED → a room number (e.g., 42)
    2. You go directly to room 42 and store your luggage (value)

    When you check out (lookup a key):
    1. Your name gets hashed again → same room number (42)
    2. Go directly to room 42 → get your luggage
    3. No searching through all rooms! Direct access = O(1)

    STEP BY STEP:
        my_dict = {}
        my_dict["name"] = "Tirupathi"

        Internally:
        1. Python computes hash("name") → 7392847 (a big number)
        2. 7392847 % table_size → 42 (maps to a slot in the array)
        3. Stores ("name", "Tirupathi") at slot 42

        my_dict["name"]  # Lookup

        Internally:
        1. Python computes hash("name") → 7392847 (same hash!)
        2. 7392847 % table_size → 42 (same slot!)
        3. Goes directly to slot 42 → returns "Tirupathi"
        4. NO iteration through all keys. Direct jump. O(1).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

WHAT IS A HASH FUNCTION?

    A function that converts ANY input into a FIXED-SIZE number.
    - hash("name") → 7392847
    - hash("age") → 2847391
    - hash("city") → 9182736

    Properties of a good hash function:
    1. DETERMINISTIC: same input always gives same output
    2. FAST: computes in constant time
    3. UNIFORM: distributes outputs evenly across the table
    4. DIFFERENT inputs should give DIFFERENT outputs (mostly)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

HASH COLLISIONS (When two keys map to the same slot):

    What if hash("name") % table_size == hash("city") % table_size?
    Both want slot 42! This is a COLLISION.

    Python handles this with OPEN ADDRESSING (probing):
    - If slot 42 is taken, try slot 43, then 44, then 45...
    - Until an empty slot is found

    WORST CASE: All keys collide → every lookup checks all slots → O(n)
    AVERAGE CASE: Good hash function → few collisions → O(1)

    Python keeps the table at most 2/3 full (load factor < 0.67).
    When it gets too full, it RESIZES (doubles the table) to reduce collisions.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

COMPARISON WITH OTHER DATA STRUCTURES:

    OPERATION           LIST        DICT        SET
    Lookup by value     O(n)        O(1)        O(1)
    Insert              O(1)*       O(1)        O(1)
    Delete              O(n)        O(1)        O(1)
    Check membership    O(n)        O(1)        O(1)

    * List append is O(1), but insert at position is O(n)

    WHY THIS MATTERS IN INTERVIEWS:
        # BAD: O(n) lookup
        if x in my_list:  # checks every element

        # GOOD: O(1) lookup
        if x in my_set:   # hash lookup, instant
        if x in my_dict:  # hash lookup, instant

    REAL EXAMPLE (from our coding prep):
        # Two Sum — finding complement
        seen = {}  # dict for O(1) lookup
        for num in arr:
            complement = target - num
            if complement in seen:  # O(1) check!
                return [seen[complement], i]
            seen[num] = i

        Without dict: nested loop O(n²). With dict: single pass O(n).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

WHAT CAN BE A DICT KEY? (Hashability)

    Only IMMUTABLE objects can be dict keys:
    ✅ str, int, float, tuple, frozenset, bool, None
    ❌ list, dict, set (mutable — can't be hashed)

    Why? If a key could change after being stored, the hash would change,
    and you'd never find it again (it's in slot 42 but now hashes to slot 87).

INTERVIEW ANSWER:
    "Python dicts use hash tables internally. When you store dict['name'] = 'Tirupathi',
    Python hashes the key 'name' to get a number, maps it to a slot in an internal
    array, and stores the value there. On lookup, it hashes the key again, goes
    directly to that slot — no iteration needed. That's why it's O(1) average.
    Collisions are handled with open addressing (probing to next slot). Python
    keeps the table at most 2/3 full and resizes when needed. Only immutable
    objects can be keys because the hash must stay constant."
"""


# =================================================================================
# SECTION 11: ML METRICS — PRECISION, RECALL, F1 (With Real Scenarios)
# =================================================================================
"""
FROM THE INTERVIEW:
    "Out of 100 data points, 90 were correctly classified. Calculate precision
    and recall. When would you prioritize each?"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

THE CONFUSION MATRIX (Foundation of all metrics):

    Imagine a cancer detection model. It predicts "Cancer" or "No Cancer."

                            ACTUAL
                        Cancer    No Cancer
    PREDICTED  Cancer    TP=80      FP=10       ← Predicted positive
               No Cancer FN=5       TN=905      ← Predicted negative

    TP (True Positive) = 80: Model said Cancer, actually IS cancer ✅
    FP (False Positive) = 10: Model said Cancer, actually NO cancer ❌ (false alarm)
    FN (False Negative) = 5: Model said No Cancer, actually IS cancer ❌ (MISSED!)
    TN (True Negative) = 905: Model said No Cancer, actually no cancer ✅

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PRECISION:
    "Of everything I PREDICTED as positive, how many are actually positive?"

    Formula: TP / (TP + FP) = 80 / (80 + 10) = 80/90 = 0.89 (89%)

    In plain English: "When I say someone has cancer, I'm right 89% of the time."

    HIGH PRECISION = Few false alarms.
    LOW PRECISION = Many false alarms (crying wolf too often).

RECALL (also called Sensitivity or True Positive Rate):
    "Of everything that IS actually positive, how many did I catch?"

    Formula: TP / (TP + FN) = 80 / (80 + 5) = 80/85 = 0.94 (94%)

    In plain English: "Of all actual cancer patients, I caught 94% of them."

    HIGH RECALL = Few missed cases.
    LOW RECALL = Many missed cases (letting positives slip through).

F1 SCORE:
    Harmonic mean of precision and recall. Balances both.

    Formula: 2 × (Precision × Recall) / (Precision + Recall)
           = 2 × (0.89 × 0.94) / (0.89 + 0.94)
           = 2 × 0.8366 / 1.83
           = 0.914 (91.4%)

    Use F1 when you need BOTH precision and recall to be good.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

WHEN TO PRIORITIZE WHICH (Real Scenarios):

    SCENARIO 1: CANCER DETECTION → Prioritize RECALL
        Why: Missing a cancer patient (FN) = life-threatening.
        A false alarm (FP) just means an extra test — inconvenient but safe.
        "I'd rather test 10 healthy people than miss 1 cancer patient."

    SCENARIO 2: SPAM FILTER → Prioritize PRECISION
        Why: Marking a real email as spam (FP) = user misses important mail.
        Missing a spam email (FN) = user sees one extra spam — annoying but not critical.
        "I'd rather let 1 spam through than block 1 real email."

    SCENARIO 3: FRAUD DETECTION (from interview) → F1 SCORE
        Why: Can't let fraudsters escape (need recall).
        But can't falsely accuse innocent people (need precision — lawsuits!).
        Both matter equally → F1 balances them.

    SCENARIO 4: SEARCH ENGINE → Prioritize RECALL (at top-k)
        Why: Users want to find ALL relevant results.
        Showing one irrelevant result is okay, but missing a relevant one is bad.

    SCENARIO 5: AUTONOMOUS DRIVING (pedestrian detection) → Prioritize RECALL
        Why: Missing a pedestrian (FN) = accident = death.
        False alarm (FP) = car brakes unnecessarily — safe but annoying.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ACCURACY vs PRECISION vs RECALL:

    ACCURACY = (TP + TN) / Total = (80 + 905) / 1000 = 98.5%

    Sounds great! But it's MISLEADING for imbalanced datasets.

    Example: 1000 people, only 5 have cancer.
    A model that ALWAYS says "No Cancer" gets 99.5% accuracy!
    But it catches ZERO cancer patients (recall = 0%).

    LESSON: Never use accuracy alone for imbalanced datasets.
    Use precision, recall, F1, or AUC-ROC instead.

INTERVIEW ANSWER:
    "Precision measures how accurate my positive predictions are — of everything
    I flagged, how many are actually positive. Recall measures coverage — of all
    actual positives, how many did I catch. For cancer detection, I prioritize
    recall because missing a patient is life-threatening. For spam filtering,
    I prioritize precision because blocking a real email is worse than letting
    spam through. For fraud detection, I use F1 because both false accusations
    and missed fraud are costly. I never rely on accuracy alone for imbalanced
    datasets — a model that always predicts 'no cancer' gets 99.5% accuracy
    but catches zero patients."
"""


# =================================================================================
# SECTION 12: OBJECT DETECTION — YOLO (How It Actually Works)
# =================================================================================
"""
FROM THE INTERVIEW:
    "You used YOLOv4 for pedestrian detection. Explain how it works.
    What's the output? How do you evaluate it?"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

WHAT IS YOLO?

    YOLO = You Only Look Once
    It's an object detection model that looks at an image ONCE and detects
    ALL objects in a single forward pass (hence "only look once").

    Before YOLO: Models would scan the image multiple times with sliding windows.
    YOLO: Processes the entire image in ONE shot → much faster (real-time).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

HOW YOLO WORKS (Step by Step):

    STEP 1: DIVIDE IMAGE INTO GRID
        The image is divided into an S × S grid (e.g., 13 × 13 = 169 cells).
        Each cell is responsible for detecting objects whose CENTER falls in that cell.

        ┌───┬───┬───┬───┬───┐
        │   │   │   │   │   │
        ├───┼───┼───┼───┼───┤
        │   │   │ X │   │   │  ← Object center is in this cell
        ├───┼───┼───┼───┼───┤
        │   │   │   │   │   │
        └───┴───┴───┴───┴───┘

    STEP 2: EACH CELL PREDICTS BOUNDING BOXES
        Each cell predicts B bounding boxes (typically B=2 or 3).
        Each bounding box has 5 values:
        - x, y: center of the box (relative to the cell)
        - w, h: width and height of the box (relative to image)
        - confidence: P(object exists) × IoU(predicted, actual)

    STEP 3: EACH CELL PREDICTS CLASS PROBABILITIES
        Each cell also predicts: P(class | object exists)
        For example: P(person)=0.9, P(car)=0.05, P(dog)=0.05

    STEP 4: COMBINE → FINAL PREDICTIONS
        Final score = confidence × class_probability
        "This box is a person with 85% confidence"

    STEP 5: NON-MAX SUPPRESSION (NMS)
        Multiple cells might detect the SAME object.
        NMS removes duplicate/overlapping boxes:
        - Keep the box with highest confidence
        - Remove all boxes that overlap with it by more than a threshold (IoU > 0.5)
        - Repeat for remaining boxes

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

THE OUTPUT FORMAT:

    YOLO returns a list of detections:
    [
        {"class": "person", "confidence": 0.95, "bbox": [x1, y1, x2, y2]},
        {"class": "person", "confidence": 0.87, "bbox": [x1, y1, x2, y2]},
        {"class": "car",    "confidence": 0.92, "bbox": [x1, y1, x2, y2]},
    ]

    bbox = [x1, y1, x2, y2] = top-left corner (x1,y1) and bottom-right corner (x2,y2)

    How to USE this output:
    - Count objects: len([d for d in detections if d["class"] == "person"]) → 2 people
    - Track objects: compare bounding boxes across video frames
    - Trigger alerts: if "person" detected in restricted area → alarm

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

EVALUATION METRICS FOR OBJECT DETECTION:

    IoU (Intersection over Union):
        Measures how much the PREDICTED box overlaps with the ACTUAL box.

        IoU = Area of Overlap / Area of Union

        IoU = 1.0 → perfect match (boxes are identical)
        IoU = 0.5 → 50% overlap (acceptable threshold)
        IoU = 0.0 → no overlap (completely wrong)

        Typically: IoU > 0.5 = correct detection (True Positive)

    mAP (mean Average Precision):
        The STANDARD metric for object detection.
        - Calculate precision-recall curve for each class
        - Compute area under the curve (AP) for each class
        - Average across all classes = mAP

        mAP@0.5 = mAP at IoU threshold 0.5 (standard)
        mAP@0.5:0.95 = average mAP at IoU thresholds 0.5, 0.55, ..., 0.95 (stricter)

    Confidence Threshold:
        Only keep detections above a certain confidence (e.g., 0.5).
        Higher threshold = fewer detections but more accurate.
        Lower threshold = more detections but more false positives.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

YOLO VERSIONS:
    YOLOv1 (2016) — original, revolutionary speed
    YOLOv3 (2018) — multi-scale detection, much better accuracy
    YOLOv4 (2020) — bag of tricks, best accuracy/speed tradeoff at the time
    YOLOv5 (2020) — PyTorch implementation, easy to use
    YOLOv8 (2023) — Ultralytics, state-of-the-art, easy API
    YOLO11 (2024) — latest, improved efficiency

INTERVIEW ANSWER:
    "YOLO divides the image into a grid. Each cell predicts bounding boxes with
    confidence scores and class probabilities. Non-max suppression removes
    duplicate detections. The output is a list of detected objects with class,
    confidence, and bounding box coordinates. I evaluate using mAP (mean Average
    Precision) at IoU threshold 0.5 — this measures both detection accuracy and
    localization quality. For my pedestrian detection project, I used YOLOv4
    and achieved good real-time performance for counting and tracking people."
"""


# =================================================================================
# SECTION 13: FINE-TUNING vs RAG — COMPLETE DECISION FRAMEWORK
# =================================================================================
"""
FROM THE INTERVIEW:
    Interviewer: "Where can fine-tuning be better than RAG?"
    This was the closing discussion — very insightful.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

THE THREE OPTIONS (Not just two!):

    The interviewer's key insight: "It's not always fine-tuning vs RAG.
    Sometimes you need NEITHER."

    OPTION 1: CONTEXT STUFFING (No RAG, No Fine-tuning)
        Put the entire knowledge base directly in the prompt.
        When: Documents are SHORT and FEW (< 50K tokens total).
        Example: 10 FAQ pages → just put them all in the system prompt.
        Pros: Simplest, no infrastructure, no training.
        Cons: Expensive per query if docs are large, "lost in middle" problem.

    OPTION 2: RAG (Retrieval-Augmented Generation)
        Retrieve relevant chunks, pass to LLM as context.
        When: Large knowledge base, factual Q&A, frequently updated docs.
        Example: 10,000 product manuals → retrieve relevant pages per query.
        Pros: Scalable, updatable, source citations, cost-effective.
        Cons: Retrieval can miss relevant docs, adds latency, accuracy multiplication.

    OPTION 3: FINE-TUNING
        Train the model on your specific data.
        When: Domain-specific STYLE or LANGUAGE that the LLM never saw.
        Example: Legal contracts in a specific format, medical terminology.
        Pros: Model "knows" the domain natively, consistent style.
        Cons: Expensive (GPU time), hard to update, can cause catastrophic forgetting.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

WHEN RAG IS BETTER THAN FINE-TUNING:

    1. Knowledge changes frequently
       RAG: Just update the documents. Instant.
       Fine-tuning: Retrain the model. Hours/days. Expensive.

    2. You need source citations
       RAG: "This answer came from page 42 of the manual."
       Fine-tuning: Model can't tell you WHERE it learned something.

    3. Multiple knowledge bases
       RAG: Agent picks which knowledge base to search.
       Fine-tuning: All knowledge is mixed together in weights.

    4. Accuracy is critical (financial, legal, medical)
       RAG: Answer is grounded in specific documents (verifiable).
       Fine-tuning: Answer comes from model weights (not verifiable).

    5. Cost-sensitive
       RAG: No GPU(Graphics Processing Unit) training needed. Just API calls.
       Fine-tuning: Needs GPU hours ($100-$10,000+ depending on model size).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

WHEN FINE-TUNING IS BETTER THAN RAG:

    1. Domain-specific LANGUAGE the LLM never saw
       Example: A specialized medical field with unique terminology.
       The LLM literally doesn't know these words exist.
       RAG can retrieve docs with these words, but the LLM might not
       understand them well enough to generate good answers.

    2. Specific output FORMAT that prompting can't achieve
       Example: "Always respond in this exact JSON schema with these 15 fields."
       If the format is very rigid and complex, fine-tuning teaches the model
       to always produce that format natively.

    3. The ROLE itself doesn't exist in training data
       Example: A very specialized advisor role that no public data covers.
       The LLM can't "pretend" to be something it has zero knowledge about.
       Fine-tuning gives it that knowledge.

    4. Consistent BEHAVIOR across all queries
       RAG: behavior depends on what's retrieved (inconsistent if retrieval varies).
       Fine-tuning: model behaves consistently regardless of retrieval.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SLMs (Small Language Models) — THE INTERVIEWER'S POINT:

    "Can't we just use SLMs for domain-specific tasks?"

    SLMs (1-7B parameters) are:
    - Cheaper to run (less GPU needed)
    - Faster inference (smaller model = faster)
    - Good for specialized, narrow tasks

    BUT:
    - They still need fine-tuning or distillation to be domain-specific
    - They can't handle complex reasoning as well as large models
    - You typically DISTILL from a large model → SLM (teacher-student)

    The process:
    1. Fine-tune a large model (70B) on your domain data
    2. Distill that knowledge into a small model (7B)
    3. Deploy the small model (cheaper, faster)

    So fine-tuning is STILL involved — just at a different stage.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

THE DECISION FLOWCHART:

    Is your knowledge base < 50K tokens?
        YES → Context stuffing (no RAG needed)
        NO ↓

    Does knowledge change frequently?
        YES → RAG (easy to update)
        NO ↓

    Do you need source citations?
        YES → RAG (can cite specific documents)
        NO ↓

    Does the LLM already understand your domain?
        YES → RAG (just give it the right context)
        NO → Fine-tuning (teach it the domain)

    Do you need both knowledge AND style?
        → RAG + Fine-tuning (fine-tune for style, RAG for knowledge)

INTERVIEW ANSWER:
    "I start by asking: do I even need RAG or fine-tuning? If the knowledge base
    is small enough to fit in context, I just stuff it in the prompt — simplest
    solution. For large, frequently-updated document collections with factual Q&A,
    RAG is ideal — it's cheaper, updatable, and gives source citations. Fine-tuning
    is for when the model needs to learn domain-specific language or output formats
    that don't exist in its training data. The key insight from the interviewer:
    every system you add multiplies error probability. So use the SIMPLEST approach
    that works — don't add RAG if context stuffing is sufficient."
"""


# =================================================================================
# SECTION 14: SYSTEM DESIGN — ACCURACY MULTIPLICATION PROBLEM
# =================================================================================
"""
FROM THE INTERVIEW (Interviewer's key lesson):
    "I have accuracy 0.9 for system A, 0.9 for system B, 0.9 for system C,
    working sequentially. What's the accuracy of the entire system?"

    Answer: 0.9 × 0.9 × 0.9 = 0.729 (72.9%)

    THREE systems at 90% each → overall only 73%!

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

WHY THIS MATTERS FOR AI SYSTEMS:

    A typical RAG pipeline has MULTIPLE sequential steps:

    Step 1: Document parsing (accuracy: 0.95)
        - Did we extract the text correctly? Tables preserved?

    Step 2: Chunking (accuracy: 0.95)
        - Did we chunk at the right boundaries? No context lost?

    Step 3: Embedding (accuracy: 0.90)
        - Does the embedding capture the meaning correctly?

    Step 4: Retrieval (accuracy: 0.85)
        - Did we retrieve the RIGHT chunks for this query?

    Step 5: Generation (accuracy: 0.90)
        - Did the LLM generate a correct answer from the context?

    COMBINED: 0.95 × 0.95 × 0.90 × 0.85 × 0.90 = 0.62 (62%!)

    Even with each step at 85-95%, the overall system is only 62% accurate!
    This is why production RAG is HARD.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

IMPLICATIONS AND SOLUTIONS:

    IMPLICATION 1: FEWER STEPS = HIGHER ACCURACY
        - Don't add steps unless they JUSTIFY their accuracy cost
        - Sometimes context stuffing (1 step) beats RAG (5 steps)
        - Every component you add MUST improve overall accuracy more than it costs

    IMPLICATION 2: EACH STEP MUST BE AS ACCURATE AS POSSIBLE
        - Improving retrieval from 85% to 95% has HUGE impact on overall accuracy
        - 0.95 × 0.95 × 0.90 × 0.95 × 0.90 = 0.69 (vs 0.62 before)
        - That one improvement (85→95) raised overall from 62% to 69%

    IMPLICATION 3: ADD QUALITY GATES (Corrective/Adaptive RAG)
        - Grade retrieved documents → catch bad retrieval early
        - Validate generated answers → catch hallucination
        - These gates ADD a step but IMPROVE overall accuracy by catching errors

    IMPLICATION 4: THE 95% → 100% GAP
        The interviewer's wisdom: "AI can get you to 95%. The last 5% is where
        your entire knowledge comes in."

        Getting a demo to work = easy (95%)
        Making it production-ready = hard (95% → 100%)
        The difference: error handling, edge cases, monitoring, fallbacks

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

REAL-WORLD EXAMPLES:

    EXAMPLE 1: Self-driving car
        Perception (95%) × Planning (95%) × Control (95%) = 85.7%
        At 85.7% accuracy, the car makes a mistake every ~7 decisions.
        That's why self-driving is SO hard — each component must be 99.9%+.

    EXAMPLE 2: RAG chatbot for customer support
        Retrieval (90%) × Generation (90%) = 81%
        1 in 5 answers is wrong. Unacceptable for production.
        Solution: Add grading (catches 80% of bad retrievals) → effective 96%

    EXAMPLE 3: Multi-agent system
        Agent 1 decision (90%) × Agent 2 decision (90%) × Agent 3 (90%) = 73%
        This is why multi-agent systems need VERIFICATION at each step.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

HOW TO DISCUSS THIS IN INTERVIEWS:

    When designing any system, always mention:
    1. "I'm aware that sequential steps multiply error rates"
    2. "That's why I add quality gates (grading, validation)"
    3. "I keep the architecture as simple as possible"
    4. "I measure accuracy at EACH step, not just end-to-end"
    5. "Sometimes the simplest solution (fewer steps) is the best"

INTERVIEW ANSWER:
    "In sequential systems, accuracy multiplies — three 90% steps give only 73%
    overall. This is why I design RAG pipelines with minimal sequential steps
    and add quality gates (document grading, answer validation) to catch errors
    early. I also measure accuracy at each step independently — if retrieval is
    the weakest link at 85%, I focus on improving that one component rather than
    adding more steps. Sometimes the simplest architecture (context stuffing with
    no RAG) outperforms a complex pipeline because fewer steps = fewer failure points."
"""


# =================================================================================
# SECTION 15: AI MODELS LANDSCAPE (What You Must Know in 2026)
# =================================================================================
"""
FROM THE INTERVIEW:
    The interviewer played a "name AI models" game to test awareness.
    You MUST know the major models, who made them, and what they're good at.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

FRONTIER MODELS (Closed-Source, Most Powerful):

    MODEL           COMPANY      BEST AT                    CONTEXT
    GPT-4o          OpenAI       General, coding, reasoning  128K tokens
    GPT-4o-mini     OpenAI       Fast, cheap, good enough    128K tokens
    Claude 3.5      Anthropic    Long docs, safety, coding   200K tokens
    Claude 4        Anthropic    Reasoning, agentic tasks    200K tokens
    Gemini 2.0      Google       Multimodal, long context    1M+ tokens
    Gemini Flash    Google       Fast, cheap                 1M tokens
    Grok            xAI          Real-time info (X/Twitter)  128K tokens

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

OPEN-SOURCE MODELS (Can run locally, free):

    MODEL           COMPANY      SIZES           BEST AT
    Llama 3.3      Meta          8B, 70B         General, tool calling
    Mistral/Mixtral Mistral AI   7B, 8x7B        Efficient, fast
    Qwen 2.5       Alibaba       7B, 72B         Coding, multilingual
    DeepSeek V3    DeepSeek      671B (MoE)      Reasoning, math
    DeepSeek R1    DeepSeek      Various         Chain-of-thought reasoning
    Gemma 2        Google        2B, 9B, 27B     Small, efficient
    Phi-4          Microsoft     14B             Small but capable
    StarCoder 2    BigCode       3B, 7B, 15B     Code generation
    Command R+     Cohere        104B            RAG-optimized

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

EMBEDDING MODELS (For RAG/Search):

    MODEL                       DIMS    COST      QUALITY
    all-MiniLM-L6-v2 (HF)      384     Free      Good (prototyping)
    text-embedding-3-small      1536    Cheap     Very Good
    text-embedding-3-large      3072    Medium    Excellent
    BGE-large (BAAI)            1024    Free      Very Good
    Voyage-3                    1024    Medium    Best for code/legal
    E5-Mistral                  4096    Free      Excellent (large)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

IMAGE/MULTIMODAL MODELS:

    DALL-E 3 (OpenAI) — text to image
    Midjourney v6 — highest quality images
    Stable Diffusion 3 (Stability AI) — open-source image generation
    Flux (Black Forest Labs) — fast, high quality
    GPT-4V / Claude Vision — image understanding (not generation)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

WHEN TO USE WHICH (Practical Guide):

    "I need the best quality, cost doesn't matter" → GPT-4o or Claude 4
    "I need good quality, budget-friendly" → GPT-4o-mini or Claude Haiku
    "I need to run locally (privacy)" → Llama 3.3 70B or Qwen 72B
    "I need fast + cheap for simple tasks" → Gemini Flash or GPT-4o-mini
    "I need coding assistance" → Claude 4 or Qwen 2.5
    "I need long document processing" → Gemini (1M context) or Claude (200K)
    "I need reasoning/math" → DeepSeek R1 or GPT-4o
    "I need embeddings for RAG" → all-MiniLM-L6-v2 (free) or text-embedding-3-small

WHY THIS MATTERS IN INTERVIEWS:
    Interviewers test if you're AWARE of the landscape. If you only know
    "ChatGPT", it shows you haven't explored. Knowing 10+ models and their
    strengths shows you're actively working in the field.
"""


# =================================================================================
# SECTION 16: KEY INTERVIEW TAKEAWAYS (The Interviewer's Wisdom)
# =================================================================================
"""
These are the GOLDEN LESSONS from the interviewer's closing remarks.
Memorize these — they show MATURITY and DEPTH.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

LESSON 1: "AI can get you to 95%. The last 5% is where YOUR knowledge matters."

    What this means:
    - Anyone can build a demo with AI tools (95% working)
    - Production requires understanding edge cases, error handling, monitoring
    - The last 5% is: security, scalability, cost optimization, reliability
    - This is why companies still need engineers — AI alone isn't enough

    How to show this in interviews:
    - Don't just describe the happy path
    - Mention: "What if the API goes down? What if retrieval fails?"
    - Show you think about production concerns, not just demos

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

LESSON 2: "The more systems you chain, the lower your accuracy."

    What this means:
    - 0.9 × 0.9 × 0.9 = 0.729 (three 90% systems = 73% overall)
    - Every step you add introduces potential failure
    - SIMPLER architectures often outperform complex ones

    How to show this in interviews:
    - "I keep the architecture as simple as possible"
    - "I only add RAG if context stuffing isn't sufficient"
    - "Each component must justify its existence"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

LESSON 3: "Good interviewers test THOUGHT PROCESS, not memorized facts."

    What this means:
    - If you don't know the exact answer, approach it LOGICALLY
    - Show HOW you think, not just WHAT you know
    - It's okay to say "I'm not sure, but here's how I'd approach it..."
    - Interviewers want to see you can REASON through new problems

    How to show this in interviews:
    - Think out loud: "Let me break this down..."
    - Consider tradeoffs: "Option A is faster but less accurate..."
    - Ask clarifying questions: "What's the scale? What's the latency requirement?"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

LESSON 4: "Don't use GenAI for everything. Simple ML works for simple tasks."

    What this means:
    - 500 reviews → Logistic Regression (not GPT-4)
    - Fraud detection → Random Forest or XGBoost (not LLM)
    - Using GenAI for simple classification = overkill = waste of money

    How to show this in interviews:
    - "For this task, a simple ML model would be sufficient and cheaper"
    - "I'd only use GenAI if the task requires language understanding"
    - Shows you're PRACTICAL, not just chasing hype

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

LESSON 5: "Production is about 100% accuracy or MANAGEABLE failure."

    What this means:
    - You can't always get 100% accuracy
    - But you MUST handle failures gracefully
    - When the system is wrong, it should: detect it, log it, fallback, alert

    How to show this in interviews:
    - "I add monitoring to detect when accuracy drops"
    - "I have fallback mechanisms (retry, secondary model, human escalation)"
    - "I log every decision for debugging and improvement"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

LESSON 6: "Open-source your work. Let others test it. Get feedback."

    What this means:
    - Portfolio projects with live demos > theoretical knowledge
    - Open-sourcing gets you: feedback, visibility, credibility
    - Interviewers can SEE your code, not just hear about it

    Your advantage:
    - DocSage is deployed on HuggingFace (live demo!)
    - Your code is on GitHub
    - You can say: "Here's the link, you can try it yourself"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

LESSON 7: "Consider interviews as conversations, not exams."

    What this means:
    - The interviewer doesn't know everything either
    - It's a two-way discussion, not a one-way test
    - Ask questions back, share your perspective
    - Show curiosity: "That's interesting, I hadn't thought about it that way"

    How to show this:
    - Don't just answer — engage: "In my experience, I found that..."
    - Ask: "How does your team handle this at [company]?"
    - Be genuine: "I don't know the exact answer, but here's my thinking..."

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

LESSON 8: "The standalone Prompt Engineer role is dying. Hybrid roles win."

    What this means:
    - Pure "prompt writing" is becoming basic literacy (like typing)
    - Companies want people who can BUILD systems + write prompts
    - AI Engineer = prompt skills + coding + system design + deployment
    - This is why your LangGraph/MCP experience is MORE valuable than just prompts

    Your positioning:
    - "I'm not just a prompt engineer — I design end-to-end AI workflows"
    - "I can write prompts AND build the systems that execute them"
    - "I understand both the AI side and the engineering side"
"""

print("=" * 60)
print("Mock Interview Deep Dive — Part 2 (Sections 5-16)")
print("=" * 60)
print()
print("All sections fully elaborated:")
print("  5.  RAG System Design (Invoice Chatbot)")
print("  6.  OCR Challenges (Tables in PDFs)")
print("  7.  Cosine Similarity vs KNN (The Real Truth)")
print("  8.  Cheating Detection (Outliers + Graph DBs)")
print("  9.  Attention N² Problem (6 Solutions)")
print("  10. Python Dict O(1) (Hash Tables Explained)")
print("  11. Precision, Recall, F1 (5 Real Scenarios)")
print("  12. YOLO Object Detection (Step by Step)")
print("  13. Fine-Tuning vs RAG (Complete Framework)")
print("  14. Accuracy Multiplication (System Design)")
print("  15. AI Models Landscape (30+ Models)")
print("  16. Key Interview Takeaways (8 Golden Lessons)")
print("=" * 60)
