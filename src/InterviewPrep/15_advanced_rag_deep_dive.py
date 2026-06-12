"""
===================================================================================
ADVANCED RAG DEEP DIVE — Senior-Level Interview Concepts
===================================================================================

These are the EXACT gaps exposed in the ETech interview.
Every section goes to the depth that senior interviewers expect.

SECTIONS:
    1. Vectorless RAG — Inverted Indexes, BM25, Tree-Based Retrieval
    2. RecursiveCharacterTextSplitter — Separator Precedence (Exact Order)
    3. Advanced Retrieval Techniques (7 Methods Beyond Basic Similarity)
    4. Context Window Overflow — How to Handle It
    5. Adaptive RAG vs Corrective RAG (CORRECT Definitions)
    6. Limiting Vector Search Scope — ANN, HNSW, IVF, Metadata Filtering
    7. GOLDEN LESSONS
===================================================================================
"""


# =================================================================================
# SECTION 1: VECTORLESS RAG — Inverted Indexes, BM25, Tree-Based Retrieval
# =================================================================================
"""
WHAT THE INTERVIEWER ASKED:
    "How does retrieval work WITHOUT vectors? What are tree-based indexes?
    How are indexes separated in vectorless RAG?"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

WHAT IS VECTORLESS RAG?

    RAG without embedding vectors. Instead of converting text to dense vectors
    and doing cosine similarity, you use TRADITIONAL information retrieval
    techniques that have existed for decades (before AI).

    These techniques use SPARSE representations (keyword-based) instead of
    DENSE representations (embedding-based).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

METHOD 1: INVERTED INDEX (The Foundation of All Search Engines)

    WHAT IS IT?
        A data structure that maps WORDS to the DOCUMENTS that contain them.
        It's "inverted" because instead of "document → words" (forward index),
        it's "word → documents" (inverted).

    EXAMPLE:
        Documents:
            Doc1: "The cat sat on the mat"
            Doc2: "The dog sat on the log"
            Doc3: "The cat chased the dog"

        Forward Index (normal):
            Doc1 → [the, cat, sat, on, the, mat]
            Doc2 → [the, dog, sat, on, the, log]
            Doc3 → [the, cat, chased, the, dog]

        INVERTED INDEX:
            "cat"    → [Doc1, Doc3]
            "dog"    → [Doc2, Doc3]
            "sat"    → [Doc1, Doc2]
            "mat"    → [Doc1]
            "log"    → [Doc2]
            "chased" → [Doc3]
            "the"    → [Doc1, Doc2, Doc3]
            "on"     → [Doc1, Doc2]

    HOW RETRIEVAL WORKS:
        Query: "cat sat"
        1. Look up "cat" in inverted index → [Doc1, Doc3]
        2. Look up "sat" in inverted index → [Doc1, Doc2]
        3. Intersection: Doc1 appears in BOTH → most relevant
        4. Return Doc1 as top result

    WHY IT'S FAST:
        - Direct lookup (like a dictionary/hashmap) — O(1) per word
        - No need to scan every document
        - Google, Elasticsearch, Lucene ALL use inverted indexes internally

    THIS IS WHAT ELASTICSEARCH/OPENSEARCH USES:(BM= Best Match)
        When you use BM25 or keyword search in production, the underlying
        data structure is an inverted index. It's NOT doing vector math.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

METHOD 2: BM25 (Best Match 25) — The Scoring Algorithm

    BM25 is the ALGORITHM that SCORES documents using the inverted index.
    It answers: "Of all documents containing my search terms, which is MOST relevant?"

    THE FORMULA (simplified intuition):
        BM25_score = sum for each query term:
            IDF(term) × (TF(term, doc) × (k1 + 1)) / (TF(term, doc) + k1 × (1 - b + b × doc_length / avg_length))

    WHAT EACH PART MEANS:

        TF (Term Frequency):
            How many times does the word appear in THIS document?
            More occurrences = more relevant (but with diminishing returns).
            "cat" appears 5 times in Doc1 → higher score than if it appears once.

        IDF (Inverse Document Frequency):
            How RARE is this word across ALL documents?
            Rare words are more informative than common words.
            "cat" appears in 2/3 docs → moderate IDF
            "chased" appears in 1/3 docs → high IDF (more informative!)
            "the" appears in 3/3 docs → very low IDF (useless word)

        Document Length Normalization:
            Longer documents naturally have more word occurrences.
            BM25 normalizes for this — a short doc with "cat" once
            might be more relevant than a long doc with "cat" once.

    WHY BM25 IS STILL USED IN 2026:
        - FAST (inverted index lookup, no GPU needed)
        - GREAT for exact keyword matching (invoice numbers, product codes)
        - No embedding model needed (cheaper)
        - Interpretable (you can explain WHY a doc was retrieved)
        - Combined with vectors in HYBRID search = best of both worlds

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

METHOD 3: TREE-BASED RETRIEVAL (What the interviewer specifically asked)

    WHAT ARE TREE-BASED INDEXES?
        Hierarchical data structures that organize documents in a TREE
        for efficient retrieval. Different from flat inverted indexes.

    TYPE 1: B-TREE INDEX (used in databases)
        - Balanced tree structure
        - Each node contains sorted keys
        - Lookup is O(log n) — very fast even for millions of records
        - Used by PostgreSQL, MySQL for indexed columns
        - Good for: range queries ("find all orders between date X and Y")

        Example:
                        [M]
                       /   \
                   [D, H]   [R, W]
                  / | \     / | \
               [A-C][E-G][I-L][N-Q][S-V][X-Z]

        To find "K": Start at root → go right of D, left of H → find in [I-L] leaf
        Only 3 comparisons instead of scanning all documents!

    TYPE 2: KD-TREE (for vector/spatial data)
        - Splits space into regions at each level
        - Each node splits on one dimension
        - Used for nearest-neighbor search in low dimensions
        - Problem: doesn't work well for high-dimensional vectors (>20 dims)
        - That's why we use HNSW instead for embeddings (384+ dims)

    TYPE 3: HIERARCHICAL DOCUMENT TREES (LlamaIndex's approach)
        - Documents organized in a tree: root → sections → paragraphs → sentences
        - Retrieval starts at the top and drills down
        - "Which section is relevant?" → "Which paragraph?" → "Which sentence?"
        - This is what LlamaIndex calls "tree index" or "hierarchical retrieval"

        Example:
            Root: "Company Annual Report 2024"
            ├── Section: "Financial Results"
            │   ├── Paragraph: "Q1 Revenue was $10M..."
            │   └── Paragraph: "Q2 Revenue was $12M..."
            ├── Section: "Product Updates"
            │   ├── Paragraph: "Launched Product X..."
            │   └── Paragraph: "Product Y improvements..."
            └── Section: "Future Plans"
                └── Paragraph: "Expanding to 5 new markets..."

        Query: "What was Q2 revenue?"
        → Tree traversal: Root → "Financial Results" → "Q2 Revenue was $12M"
        → Only searched 1 section instead of the entire document!

    TYPE 4: RAPTOR (Recursive Abstractive Processing for Tree-Organized Retrieval)
        - Clusters chunks at multiple levels of abstraction
        - Bottom: individual chunks (detailed)
        - Middle: summaries of chunk groups
        - Top: high-level document summaries
        - Query matches at the RIGHT level of abstraction
        - Detailed question → matches bottom chunks
        - High-level question → matches top summaries

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

HOW INDEXES ARE SEPARATED (what the interviewer asked):

    In production systems, you often have MULTIPLE index types working together:

    INDEX LAYER 1: Inverted Index (BM25/keyword)
        → Fast keyword lookup
        → Handles exact matches (invoice numbers, names)

    INDEX LAYER 2: Vector Index (HNSW/IVF)
        → Semantic similarity search
        → Handles meaning-based queries

    INDEX LAYER 3: Metadata Index (B-tree)
        → Filter by date, category, tenant_id
        → Narrows search space BEFORE vector/keyword search

    INDEX LAYER 4: Tree/Hierarchical Index
        → Navigate document structure
        → Drill down from section → paragraph → sentence

    These are SEPARATE data structures, often in the same database:
    - Weaviate: has inverted index + vector index + metadata index built-in
    - OpenSearch: has inverted index (native) + vector index (plugin)
    - Pinecone: has vector index + metadata filtering

INTERVIEW ANSWER:
    "Vectorless RAG uses inverted indexes and BM25 scoring instead of embeddings.
    An inverted index maps words to documents — lookup is O(1) per term. BM25
    scores documents by term frequency, inverse document frequency, and length
    normalization. For tree-based retrieval, documents are organized hierarchically
    — queries traverse the tree from top to bottom, searching only relevant branches.
    In production, I'd use multiple index layers together: inverted index for keywords,
    vector index for semantics, metadata index for filtering, and optionally a
    hierarchical index for structured documents. This is how systems like Weaviate
    and OpenSearch work internally."
"""


# =================================================================================
# SECTION 2: RecursiveCharacterTextSplitter — SEPARATOR PRECEDENCE
# =================================================================================
"""
WHAT THE INTERVIEWER ASKED:
    "What is the separator precedence order in RecursiveCharacterTextSplitter?
    How are separators prioritized?"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

THE EXACT DEFAULT SEPARATOR ORDER:

    separators = ["\\n\\n", "\\n", " ", ""]

    Priority 1: "\\n\\n" (double newline = paragraph break)
    Priority 2: "\\n" (single newline = line break)
    Priority 3: " " (space = word boundary)
    Priority 4: "" (empty string = character by character — last resort)

    NOTE: Some implementations also include ". " (sentence boundary) between
    "\\n" and " ". The LangChain docs show the above 4 as default.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

HOW THE ALGORITHM WORKS (Step by Step):

    Given: chunk_size=100, chunk_overlap=20, text="..."

    STEP 1: Try to split on "\\n\\n" (paragraph breaks)
        Split the text at every double newline.
        Check: Is each resulting piece <= 100 characters?
        If YES → done! Each paragraph is a chunk.
        If NO → some paragraphs are too long. Go to step 2 for those.

    STEP 2: For pieces still too long, try "\\n" (line breaks)
        Split the oversized piece at single newlines.
        Check: Is each piece <= 100 characters?
        If YES → done for this piece.
        If NO → go to step 3.

    STEP 3: For pieces still too long, try " " (spaces)
        Split at word boundaries.
        Check: Is each piece <= 100 characters?
        If YES → done.
        If NO → go to step 4.

    STEP 4: Last resort — split at "" (character level)
        Split character by character to force fit within chunk_size.
        This means a word might get cut in half — worst case.

    THE KEY INSIGHT: It's RECURSIVE because it tries the BEST separator first,
    and only falls back to worse separators for pieces that are still too long.
    This preserves the most natural text boundaries possible.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

VISUAL EXAMPLE:

    Text (300 chars):
        "Introduction to RAG\\n\\nRAG stands for Retrieval Augmented Generation.
        It combines retrieval with generation.\\n\\nHow it works\\nFirst, documents
        are chunked. Then embedded. Then stored in a vector database.\\n\\nConclusion
        \\nRAG is essential for production AI."

    chunk_size = 150

    STEP 1: Split on "\\n\\n" → 3 paragraphs:
        Chunk 1: "Introduction to RAG" (19 chars) ✅ fits
        Chunk 2: "RAG stands for... generation." (80 chars) ✅ fits
        Chunk 3: "How it works\\nFirst... database." (120 chars) ✅ fits
        Chunk 4: "Conclusion\\nRAG is essential..." (50 chars) ✅ fits

    All fit! No need to go to step 2. Paragraphs preserved perfectly.

    But if chunk_size = 50:
        Chunk 2 (80 chars) is TOO LONG → split on "\\n" → still too long
        → split on " " (word boundaries) → fits now

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

WHY THIS ORDER MATTERS:

    "\\n\\n" first: Paragraphs are the most natural semantic units.
        Each paragraph is usually about ONE topic. Best for retrieval.

    "\\n" second: Lines within a paragraph are still meaningful units.
        Better than splitting mid-sentence.

    " " third: Word boundaries. Sentences might get split, but at least
        words stay intact.

    "" last: Character-level split. Words get cut. Worst quality.
        Only happens if a single word is longer than chunk_size (very rare).

    THE GOAL: Preserve the LARGEST meaningful text units possible.
    Paragraphs > Lines > Sentences > Words > Characters.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

WHAT ABOUT OVERLAP?

    chunk_overlap = 200 means the LAST 200 characters of chunk N
    are REPEATED at the START of chunk N+1.

    WHY: If a sentence spans two chunks, both chunks have the full sentence.
    Without overlap, you'd lose context at boundaries.

    Example (overlap=20):
        Chunk 1: "...RAG combines retrieval with generation."
        Chunk 2: "with generation. How it works: First..."
                  ↑ these 20 chars overlap ↑

CUSTOM SEPARATORS (for specific content):

    # For code:
    separators = ["\\nclass ", "\\ndef ", "\\n\\n", "\\n", " ", ""]

    # For markdown:
    separators = ["\\n## ", "\\n### ", "\\n\\n", "\\n", " ", ""]

    # For legal documents:
    separators = ["\\nArticle ", "\\nSection ", "\\n\\n", "\\n", ". ", " ", ""]

INTERVIEW ANSWER:
    "RecursiveCharacterTextSplitter tries separators in priority order:
    double newline (paragraphs) first, then single newline (lines), then
    space (words), then empty string (characters) as last resort. It's
    'recursive' because it tries the best separator first and only falls
    back to worse ones for pieces that are still too long. This preserves
    the most natural text boundaries — paragraphs stay intact when possible.
    The overlap parameter ensures context isn't lost at chunk boundaries.
    For domain-specific content, I customize separators — for code I split
    on class/function definitions, for markdown on headers."
"""


# =================================================================================
# SECTION 3: ADVANCED RETRIEVAL TECHNIQUES (7 Methods Beyond Basic)
# =================================================================================
"""
WHAT THE INTERVIEWER ASKED:
    "What retrieval techniques do you know beyond MMR, similarity search,
    and keyword matching?"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

TECHNIQUE 1: PARENT-CHILD RETRIEVAL (Small-to-Big)

    WHAT: Store small chunks for precise retrieval, but pass the LARGER
    parent chunk to the LLM for more context.

    HOW:
        - Split document into large chunks (parents): 2000 chars each
        - Split each parent into small chunks (children): 200 chars each
        - Embed and index ONLY the children (small, precise)
        - When a child is retrieved, fetch its PARENT and pass that to LLM

    WHY: Small chunks match queries precisely (high retrieval accuracy).
    But small chunks alone lack context. The parent provides the full picture.

    Example:
        Child (retrieved): "Revenue was $10M in Q3"
        Parent (passed to LLM): Full paragraph about Q3 financials including
        context about growth rate, comparison to Q2, and projections.

    Code (LangChain):
        from langchain.retrievers import ParentDocumentRetriever
        from langchain.storage import InMemoryStore

        parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000)
        child_splitter = RecursiveCharacterTextSplitter(chunk_size=200)

        retriever = ParentDocumentRetriever(
            vectorstore=vectorstore,
            docstore=InMemoryStore(),
            child_splitter=child_splitter,
            parent_splitter=parent_splitter,
        )

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

TECHNIQUE 2: MULTI-QUERY RETRIEVAL

    WHAT: Generate MULTIPLE variations of the user's query, retrieve for
    each variation, then combine results.

    WHY: A single query might miss relevant docs due to wording.
    Multiple phrasings cast a wider net.

    Example:
        User query: "How does RAG prevent hallucination?"
        Generated variations:
            1. "What techniques does RAG use to reduce false information?"
            2. "How does retrieval-augmented generation improve accuracy?"
            3. "What is the role of grounding in RAG systems?"

        Retrieve for ALL 3 → union of results → more comprehensive coverage.

    Code:
        from langchain.retrievers import MultiQueryRetriever
        retriever = MultiQueryRetriever.from_llm(
            retriever=base_retriever, llm=llm
        )

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

TECHNIQUE 3: CONTEXTUAL COMPRESSION

    WHAT: After retrieval, COMPRESS each document to only the parts
    relevant to the query. Remove irrelevant sentences.

    WHY: A retrieved chunk might be 1000 chars but only 2 sentences
    are actually relevant. Compression extracts just those 2 sentences.
    This saves tokens and improves generation quality.

    Code:
        from langchain.retrievers import ContextualCompressionRetriever
        from langchain.retrievers.document_compressors import LLMChainExtractor

        compressor = LLMChainExtractor.from_llm(llm)
        retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=base_retriever
        )

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

TECHNIQUE 4: SELF-QUERY RETRIEVAL

    WHAT: The LLM extracts METADATA FILTERS from the user's query
    automatically, then applies them before vector search.

    Example:
        User: "Show me furniture products under $100"
        LLM extracts: filter = {"category": "furniture", "price": {"$lt": 100}}
        Then: vector search WITH this filter applied

    WHY: Combines natural language understanding with structured filtering.
    The user doesn't need to know the database schema.

    Code:
        from langchain.retrievers import SelfQueryRetriever
        retriever = SelfQueryRetriever.from_llm(
            llm=llm,
            vectorstore=vectorstore,
            document_contents="Product catalog",
            metadata_field_info=[
                {"name": "category", "type": "string", "description": "Product category"},
                {"name": "price", "type": "float", "description": "Product price in USD"},
            ]
        )

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

TECHNIQUE 5: ENSEMBLE RETRIEVAL (Hybrid)

    WHAT: Run MULTIPLE retrievers in parallel, combine results with
    Reciprocal Rank Fusion (RRF).

    Example: BM25 retriever + Vector retriever + Metadata retriever
    → combine all results → rerank → return top-k

    Code:
        from langchain.retrievers import EnsembleRetriever
        retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, vector_retriever],
            weights=[0.4, 0.6]  # 40% keyword, 60% semantic
        )

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

TECHNIQUE 6: HYPOTHETICAL DOCUMENT EMBEDDING (HyDE)

    WHAT: Generate a FAKE answer first, embed THAT, use it as search vector.
    The fake answer looks more like a document than a question does.

    Flow: Query → LLM generates hypothetical answer → Embed answer → Search

    WHY: Query embeddings and document embeddings live in different spaces.
    "What causes inflation?" (question) looks different from a paragraph
    explaining inflation (document). HyDE bridges this gap.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

TECHNIQUE 7: RERANKING (Cross-Encoder)

    WHAT: After initial retrieval (top-20), use a cross-encoder to
    re-score documents. Cross-encoder sees query AND document together.

    WHY: Initial retrieval (bi-encoder) embeds query and docs separately.
    Cross-encoder is more accurate because it processes them TOGETHER.

    Models: Cohere Rerank, BGE-reranker, ms-marco-MiniLM-L-6-v2

INTERVIEW ANSWER:
    "Beyond basic similarity and MMR, I know: Parent-child retrieval (retrieve
    small chunks, pass parent for context), Multi-query (generate query variations
    for wider coverage), Contextual compression (extract only relevant sentences),
    Self-query (LLM extracts metadata filters from natural language), Ensemble
    retrieval (combine BM25 + vector with RRF), HyDE (embed hypothetical answer
    instead of question), and Reranking (cross-encoder re-scores top results).
    In production, I'd combine ensemble retrieval + reranking + parent-child
    for the best accuracy."
"""


# =================================================================================
# SECTION 4: CONTEXT WINDOW OVERFLOW — How to Handle It
# =================================================================================
"""
INTERVIEW QUESTION: "During retrieval, if the retrieved documents exceed the
context window or token limit, how would you handle it?"

THIS IS A PRODUCTION PROBLEM. You retrieve top-k documents, but their combined
size exceeds what the LLM can process. What do you do?

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

THE PROBLEM IN NUMBERS:

    LLM context window: 8,192 tokens (Llama 3.1 8B)
    System prompt: ~200 tokens
    User question: ~50 tokens
    Conversation history: ~500 tokens
    Available for retrieved docs: 8,192 - 750 = ~7,400 tokens

    But you retrieved 4 chunks × 1,000 tokens each = 4,000 tokens (fits!)
    What if you retrieved 10 chunks × 1,000 tokens = 10,000 tokens? (OVERFLOW!)

    Even with larger models (128K context), at scale you want to minimize
    tokens for COST and LATENCY reasons.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SOLUTION 1: REDUCE K (retrieve fewer documents)

    Instead of top-10, retrieve top-3 or top-4.
    Fewer docs = fewer tokens = fits in context.

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    Tradeoff: Might miss relevant information in docs 4-10.
    When to use: When your chunks are large or context window is small.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SOLUTION 2: CONTEXTUAL COMPRESSION (Shrink the chunks)

    After retrieval, COMPRESS each chunk to only the relevant parts.
    A "compressor" LLM reads the chunk and extracts only sentences
    relevant to the user's question. Everything else is removed.

    Before compression: 1,000 token chunk (full paragraph about many topics)
    After compression: 200 tokens (only the sentences relevant to the question)

    Code:
        from langchain.retrievers import ContextualCompressionRetriever
        from langchain.retrievers.document_compressors import LLMChainExtractor

        compressor = LLMChainExtractor.from_llm(llm)
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=base_retriever
        )

    How it works:
        1. Retrieve top-10 chunks (10,000 tokens)
        2. For each chunk, ask LLM: "Extract only parts relevant to the question"
        3. Each chunk shrinks from 1,000 → 150-300 tokens
        4. Total: 10 × 200 = 2,000 tokens (fits!)

    Tradeoff: Extra LLM calls for compression (adds latency and cost).
    When to use: When you need broad coverage (many docs) but limited context.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SOLUTION 3: MAP-REDUCE (Process chunks in batches)

    Instead of stuffing ALL chunks into one prompt:
    1. MAP: Send each chunk separately to the LLM with the question
       → Get a partial answer from each chunk
    2. REDUCE: Combine all partial answers into one final answer

    Example:
        Chunk 1 → LLM → "Revenue in Q1 was $5M"
        Chunk 2 → LLM → "Revenue in Q2 was $7M"
        Chunk 3 → LLM → "Revenue in Q3 was $10M"
        REDUCE → LLM → "Total revenue across Q1-Q3 was $22M, with Q3 being highest."

    Code (LangChain):
        from langchain.chains import MapReduceDocumentsChain

    Tradeoff: Multiple LLM calls (expensive, slow). But handles unlimited docs.
    When to use: Summarizing very long documents (100+ pages).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SOLUTION 4: RERANKING + TRUNCATION

    1. Retrieve top-20 chunks
    2. Rerank with CrossEncoder (most relevant first)
    3. Take only top-3 after reranking
    4. If still too long, truncate each chunk to first 500 tokens

    This gives you the BEST 3 chunks (not just the first 3 from vector search).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SOLUTION 5: HIERARCHICAL RETRIEVAL (Small-to-Big)

    Store TWO versions of each chunk:
    - Small chunk (128 tokens) — used for RETRIEVAL (precise matching)
    - Parent chunk (1024 tokens) — passed to LLM (more context)

    Retrieve using small chunks (precise), but pass parent chunks to LLM.
    Since you retrieve fewer parents (they're larger), total token count is controlled.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SOLUTION 6: TOKEN COUNTING + DYNAMIC K

    Before sending to LLM, COUNT the tokens:
    - If total < limit → send all
    - If total > limit → remove the LEAST relevant chunk and recount
    - Repeat until it fits

    Code sketch:
        import tiktoken
        encoder = tiktoken.encoding_for_model("gpt-4")

        max_context_tokens = 7000
        selected_docs = []
        current_tokens = 0

        for doc in ranked_docs:  # already sorted by relevance
            doc_tokens = len(encoder.encode(doc.page_content))
            if current_tokens + doc_tokens <= max_context_tokens:
                selected_docs.append(doc)
                current_tokens += doc_tokens
            else:
                break  # stop adding docs when limit reached

    This dynamically adjusts K based on actual token counts.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SUMMARY TABLE:

    SOLUTION                 COMPLEXITY    QUALITY    COST
    Reduce K                 Low           Medium     Low
    Contextual Compression   Medium        High       Medium (extra LLM calls)
    Map-Reduce               High          High       High (many LLM calls)
    Reranking + Truncation   Medium        High       Low
    Hierarchical (Small-Big) Medium        High       Low
    Dynamic K (token count)  Low           High       Low

    BEST FOR INTERVIEWS: "I'd use reranking + dynamic K with token counting.
    Retrieve top-20, rerank to get the best, then fill the context window
    greedily until the token limit is reached."

INTERVIEW ANSWER:
    "When retrieved documents exceed the context window, I use a combination
    of reranking and dynamic token-based selection. First, I retrieve a larger
    set (top-20) and rerank with a cross-encoder to get the most relevant docs
    first. Then I greedily add documents to the context until the token limit
    is reached — using tiktoken to count tokens precisely. This ensures I always
    use the maximum context available without overflow. For very long documents,
    I'd use contextual compression to extract only the relevant sentences from
    each chunk before passing to the LLM."
"""


# =================================================================================
# SECTION 5: ADAPTIVE RAG vs CORRECTIVE RAG (CORRECT Definitions)
# =================================================================================
"""
INTERVIEW QUESTION: "What's the difference between Adaptive RAG and Corrective RAG?"

YOU GOT THIS WRONG IN THE INTERVIEW. Here's the CORRECT understanding:

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CORRECTIVE RAG (CRAG) — Quality check AFTER retrieval:

    WHEN: After the retriever returns documents.
    WHAT: Evaluate if retrieved documents are relevant. If NOT → correct.

    Flow:
        User question → Retriever → [GRADE DOCUMENTS] → relevant? 
            YES → Generate answer
            NO → Rewrite query and retry OR fall back to web search

    The "correction" happens AFTER retrieval fails.
    It's a QUALITY GATE that catches bad retrieval results.

    In DocSage: Your grade_node IS Corrective RAG.
    It checks document relevance and rewrites if irrelevant.

    KEY POINT: Corrective RAG always RETRIEVES first, then checks quality.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ADAPTIVE RAG — Dynamic routing BEFORE retrieval:

    WHEN: BEFORE any retrieval happens.
    WHAT: A router decides the BEST STRATEGY for this specific question.

    Flow:
        User question → [ROUTER/CLASSIFIER] → decides strategy:
            "Simple factual question" → direct LLM answer (no retrieval needed)
            "Question about uploaded docs" → vector search retrieval
            "Question needing current info" → web search
            "Complex multi-part question" → decompose into sub-queries
            "Question about specific entity" → knowledge graph lookup

    The "adaptation" is choosing the RIGHT RETRIEVAL STRATEGY per question.
    Not every question needs the same approach!

    KEY POINT: Adaptive RAG decides HOW to retrieve (or whether to retrieve at all)
    BEFORE any retrieval happens. It's a ROUTING decision.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

THE CRITICAL DIFFERENCE:

    CORRECTIVE RAG:
        - Happens AFTER retrieval
        - Checks: "Are these retrieved docs good enough?"
        - If NO → fix it (rewrite query, try again, web search fallback)
        - It's REACTIVE (fixes problems after they occur)

    ADAPTIVE RAG:
        - Happens BEFORE retrieval
        - Decides: "What's the best strategy for THIS question?"
        - Routes to different retrieval methods based on question type
        - It's PROACTIVE (chooses the right path upfront)

    ANALOGY:
        Corrective RAG = A doctor who treats you AFTER you get sick
        Adaptive RAG = A health advisor who recommends the RIGHT diet BEFORE you get sick

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

HOW ADAPTIVE RAG ROUTING WORKS:

    OPTION 1: LLM-based router
        router_prompt = "Classify this question:
            - 'simple' if it can be answered without retrieval
            - 'vector_search' if it needs document retrieval
            - 'web_search' if it needs current/real-time information
            - 'multi_query' if it's complex and needs decomposition"

        route = llm.invoke(router_prompt + question)
        if route == "simple": return llm.invoke(question)
        elif route == "vector_search": return rag_pipeline(question)
        elif route == "web_search": return web_search(question)
        elif route == "multi_query": return decompose_and_retrieve(question)

    OPTION 2: Classifier-based router (faster, cheaper)
        Train a small classifier on question types.
        Input: question text → Output: strategy label
        No LLM call needed for routing (saves tokens).

    OPTION 3: Rule-based router (simplest)
        if "current" in question or "today" in question: → web search
        if "compare" in question: → multi-query
        else: → vector search

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CAN YOU USE BOTH TOGETHER? YES!

    User question
        ↓
    [ADAPTIVE ROUTER] → decides: "use vector search" (BEFORE retrieval)
        ↓
    Vector search retrieves top-4 docs
        ↓
    [CORRECTIVE GRADER] → checks: "are these docs relevant?" (AFTER retrieval)
        ↓
    YES → Generate answer
    NO → Rewrite query and retry (or fall back to web search)

    DocSage uses BOTH:
    - Adaptive: The agent DECIDES which tool to call (routing)
    - Corrective: The grade node checks document relevance (quality gate)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

YOUR MISTAKE IN THE INTERVIEW:
    You said: "Adaptive RAG comes after the retriever"
    CORRECT: "Adaptive RAG comes BEFORE retrieval — it's a routing decision"

    Remember: Adaptive = ROUTE (before). Corrective = CHECK (after).

INTERVIEW ANSWER:
    "Adaptive RAG and Corrective RAG solve different problems at different stages.
    Adaptive RAG is a ROUTING layer BEFORE retrieval — it classifies the question
    and decides the best retrieval strategy (vector search, web search, direct answer,
    or multi-query decomposition). Not every question needs the same approach.
    Corrective RAG is a QUALITY GATE AFTER retrieval — it evaluates if the retrieved
    documents are relevant, and if not, rewrites the query or falls back to web search.
    In my DocSage project, I use both: the agent routes to the right tool (adaptive),
    and the grade node checks document relevance (corrective)."
"""


# =================================================================================
# SECTION 6: LIMITING VECTOR SEARCH SCOPE (ANN, HNSW, IVF, Metadata Filtering)
# =================================================================================
"""
INTERVIEW QUESTION: "You have embeddings for a PDF. User asks a question, it gets
embedded too. You do cosine similarity against ALL vectors. But if you have millions
of vectors, how do you LIMIT the search instead of comparing against every single one?"

THIS IS ABOUT APPROXIMATE NEAREST NEIGHBOR (ANN) ALGORITHMS.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

THE PROBLEM:

    Brute force: Compare query vector against ALL N vectors.
    - 100 vectors → 100 comparisons (instant)
    - 1,000,000 vectors → 1,000,000 comparisons (SLOW — seconds per query)
    - 100,000,000 vectors → impossible in real-time

    We need to search ONLY a SUBSET of vectors, not all of them.
    This is called Approximate Nearest Neighbor (ANN) search.
    "Approximate" because we might miss the absolute best match,
    but we find a very good match MUCH faster.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

METHOD 1: IVF (Inverted File Index) — Cluster-Based

    HOW IT WORKS:
    1. OFFLINE (at index time): Cluster all vectors into K groups using K-means.
       Example: 1,000,000 vectors → 1,000 clusters of ~1,000 vectors each.

    2. ONLINE (at query time):
       a. Find which cluster(s) the query vector is closest to (compare against 1,000 centroids)
       b. Search ONLY within those clusters (compare against ~1,000-3,000 vectors)
       c. Return top results

    INSTEAD OF: 1,000,000 comparisons
    YOU DO: 1,000 (find cluster) + 3,000 (search within) = 4,000 comparisons
    SPEEDUP: 250x faster!

    TRADEOFF: Might miss vectors in neighboring clusters (approximate, not exact).
    FIX: Search top-3 nearest clusters instead of just 1 (nprobe parameter).

    ANALOGY: Like a library. Instead of checking every book, you first go to
    the RIGHT SECTION (cluster), then search within that section only.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

METHOD 2: HNSW (Hierarchical Navigable Small World) — Graph-Based

    HOW IT WORKS:
    1. OFFLINE: Build a multi-layer graph connecting similar vectors.
       - Top layer: few nodes, long-range connections (highway)
       - Middle layers: more nodes, medium connections
       - Bottom layer: all nodes, short-range connections (local streets)

    2. ONLINE (at query time):
       a. Start at the top layer (highway) — quickly navigate to the right region
       b. Drop to middle layer — get closer
       c. Drop to bottom layer — find exact nearest neighbors

    ANALOGY: Like navigating a city.
    - Highway (top layer): quickly get to the right neighborhood
    - Main roads (middle): get to the right street
    - Walking (bottom): find the exact house

    INSTEAD OF: Checking every house in the city
    YOU DO: Highway → main road → walk = find the house in ~log(N) steps

    SPEEDUP: For 1M vectors, ~40-50 comparisons instead of 1,000,000.
    TRADEOFF: Uses more memory (stores the graph structure).

    THIS IS WHAT MOST VECTOR DATABASES USE:
    - FAISS uses HNSW
    - Pinecone uses HNSW
    - Qdrant uses HNSW
    - Weaviate uses HNSW

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

METHOD 3: METADATA FILTERING (Pre-filter before vector search)

    BEFORE doing any vector comparison, FILTER by metadata:
    - "Only search documents from tenant_id = 'company_a'"
    - "Only search documents from category = 'financial'"
    - "Only search documents from date > '2024-01-01'"

    This reduces the search space BEFORE vector comparison even starts.

    Example:
        1,000,000 total vectors
        Filter: tenant_id = "company_a" → 50,000 vectors
        Now search only 50,000 instead of 1,000,000

    Code:
        results = vectorstore.similarity_search(
            query,
            filter={"tenant_id": "company_a", "category": "financial"},
            k=4
        )

    This is the SIMPLEST and most effective way to limit search scope.
    Every production vector database supports metadata filtering.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

METHOD 4: NAMESPACE/PARTITION ISOLATION

    Physically separate vectors into different partitions:
    - Pinecone: namespaces (each tenant gets their own namespace)
    - Weaviate: multi-tenant classes
    - Qdrant: collections or payload filtering

    Query ONLY searches within the specified namespace.
    Vectors in other namespaces are NEVER even considered.

    index.query(vector=query_vec, namespace="tenant_a", top_k=4)
    # Only searches tenant_a's vectors. Others don't exist in this search.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

METHOD 5: QUANTIZATION (Reduce vector size for faster comparison)

    Instead of comparing full 1536-dimensional float vectors:
    - Product Quantization (PQ): compress vectors to 1/4 size
    - Scalar Quantization: convert float32 → int8 (4x smaller)

    Smaller vectors = faster comparisons = more vectors fit in memory.
    Tradeoff: Slight accuracy loss (usually <5%).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SUMMARY — HOW VECTOR DATABASES LIMIT SEARCH:

    LAYER 1: Namespace/Partition → physically separate data
    LAYER 2: Metadata filtering → reduce candidates before search
    LAYER 3: ANN algorithm (HNSW/IVF) → search only relevant subset
    LAYER 4: Quantization → faster individual comparisons

    Combined: 100M vectors → namespace (1M) → filter (100K) → HNSW (~50 comparisons)
    Result: Find nearest neighbors in milliseconds, not minutes.

INTERVIEW ANSWER:
    "You don't compare against every vector. Vector databases use Approximate
    Nearest Neighbor algorithms like HNSW (graph-based navigation) or IVF
    (cluster-based search) to limit comparisons to a small subset. HNSW builds
    a multi-layer graph — at query time, you navigate from top layer (highway)
    to bottom layer (local) in ~log(N) steps instead of N comparisons. On top
    of that, metadata filtering pre-filters by tenant, category, or date BEFORE
    vector search even starts. And namespace isolation physically separates
    different tenants' vectors. Combined, a query against 100 million vectors
    takes milliseconds because you're actually comparing against maybe 50-100
    vectors after all the filtering and navigation."
"""


# =================================================================================
# SECTION 8: KNOWLEDGE BASE UPDATES + INDEX RELATIONSHIPS IN VECTORLESS RAG
# =================================================================================
"""
INTERVIEW QUESTIONS:
    1. "How do you update the knowledge base dynamically? Rebuild everything or add incrementally?"
    2. "How are relationships between indexes found in vectorless RAG?"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PART A: UPDATING KNOWLEDGE BASE — VECTOR RAG vs VECTORLESS RAG

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

VECTOR RAG — How to update:

    You do NOT rebuild the entire pipeline. You do INCREMENTAL updates.

    ADDING NEW DOCUMENTS:
        1. Detect new/changed documents (hash comparison or timestamp)
        2. Chunk only the NEW documents
        3. Embed only the NEW chunks
        4. UPSERT into the vector database (add new vectors, update changed ones)

        # Pinecone example:
        index.upsert(vectors=new_vectors, namespace="tenant_a")

        # FAISS example (trickier — FAISS doesn't support native upsert):
        vectorstore.add_documents(new_chunks)  # appends to existing index

    UPDATING EXISTING DOCUMENTS:
        1. Identify which document changed (by doc_id in metadata)
        2. DELETE old vectors for that document
        3. Re-chunk, re-embed the updated document
        4. INSERT new vectors

        # Pinecone:
        index.delete(filter={"doc_id": "report_v1"})  # remove old
        index.upsert(vectors=new_vectors)              # add updated

    DELETING DOCUMENTS:
        index.delete(filter={"doc_id": "obsolete_report"})

    KEY POINT: You NEVER rebuild the entire index for a single document change.
    Vector databases support incremental operations (add, update, delete).

    WHEN TO REBUILD ENTIRELY:
        - Changing the embedding model (all vectors must be re-computed)
        - Changing chunk size/strategy (all chunks change)
        - Major schema change in metadata
        - Periodic maintenance (compact, optimize indexes)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

VECTORLESS RAG (BM25 / Inverted Index) — How to update:

    Even EASIER than vector RAG because no embedding step needed.

    ADDING NEW DOCUMENTS:
        1. Tokenize the new document (split into terms)
        2. Update the inverted index: for each term, add the new doc_id to its posting list

        Before: "revenue" → [doc_1, doc_5, doc_12]
        After adding doc_99: "revenue" → [doc_1, doc_5, doc_12, doc_99]

        That's it. No embedding, no vectors. Just update the term → document mapping.

    UPDATING EXISTING DOCUMENTS:
        1. Remove old terms for that doc_id from the inverted index
        2. Tokenize the updated document
        3. Add new terms to the inverted index

    DELETING DOCUMENTS:
        Remove the doc_id from ALL posting lists where it appears.

    KEY ADVANTAGE OF VECTORLESS: Updates are INSTANT.
        - No embedding computation needed (which takes time for large docs)
        - Just update the term-document mapping
        - This is why search engines (Elasticsearch, OpenSearch) handle
          real-time updates so well — they use inverted indexes

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PRODUCTION PATTERN — INCREMENTAL INGESTION PIPELINE:

    ┌─────────────────────────────────────────────────────────────────┐
    │  DOCUMENT CHANGE DETECTION                                       │
    │  - Watch folder / S3 bucket for new/modified files               │
    │  - Compare file hash with stored hash                            │
    │  - Trigger: "doc_123 was modified" or "doc_456 is new"           │
    └──────────────────────────┬──────────────────────────────────────┘
                               ↓
    ┌─────────────────────────────────────────────────────────────────┐
    │  PROCESS ONLY CHANGED DOCUMENTS                                  │
    │  - If modified: delete old chunks → re-chunk → re-embed → upsert│
    │  - If new: chunk → embed → insert                                │
    │  - If deleted: remove from index                                 │
    └──────────────────────────┬──────────────────────────────────────┘
                               ↓
    ┌─────────────────────────────────────────────────────────────────┐
    │  UPDATE METADATA TRACKING                                        │
    │  - Store: doc_id, file_hash, last_updated, chunk_count           │
    │  - Next time: compare hashes to detect changes                   │
    └─────────────────────────────────────────────────────────────────┘

    This runs on a SCHEDULE (every hour, daily) or on FILE CHANGE EVENTS.
    You NEVER rebuild the entire pipeline for incremental changes.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PART B: HOW RELATIONSHIPS BETWEEN INDEXES ARE FOUND IN VECTORLESS RAG

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Your answer (GraphQL/graph databases) was NOT correct for this context.
Here's what they were actually asking about:

In vectorless RAG (tree-based indexes like LlamaIndex), documents are organized
in a HIERARCHICAL TREE STRUCTURE. The "relationships between indexes" refers to
how different nodes in this tree connect to each other.

LLAMAINDEX TREE STRUCTURE:

    ┌─────────────────────────────────────────────────────────────────┐
    │                    ROOT INDEX (Summary)                           │
    │         "This knowledge base covers HR, Finance, Legal"          │
    └──────────┬──────────────────┬──────────────────┬────────────────┘
               ↓                  ↓                  ↓
    ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
    │  HR INDEX    │   │ FINANCE INDEX│   │ LEGAL INDEX  │
    │  (Sub-tree)  │   │  (Sub-tree)  │   │  (Sub-tree)  │
    └──────┬───────┘   └──────┬───────┘   └──────┬───────┘
           ↓                  ↓                  ↓
    ┌────────────┐     ┌────────────┐     ┌────────────┐
    │ Leave      │     │ Quarterly  │     │ Contracts  │
    │ Policy     │     │ Reports    │     │ NDAs       │
    │ Attendance │     │ Budgets    │     │ Compliance │
    └────────────┘     └────────────┘     └────────────┘

HOW RELATIONSHIPS WORK:

    1. PARENT-CHILD RELATIONSHIPS:
       Each node knows its parent and children.
       "Leave Policy" → parent is "HR INDEX" → parent is "ROOT INDEX"

    2. SUMMARY RELATIONSHIPS:
       Each parent node contains a SUMMARY of all its children.
       "HR INDEX" summary: "Contains leave policy, attendance rules, benefits info"
       This summary is used for ROUTING — to decide which sub-tree to search.

    3. SIBLING RELATIONSHIPS:
       Nodes at the same level can reference each other.
       "Leave Policy" and "Attendance" are siblings under "HR INDEX."

    4. CROSS-REFERENCES:
       A legal contract might reference a financial report.
       These cross-document links are stored as metadata or explicit edges.

HOW RETRIEVAL WORKS IN THIS TREE:

    User asks: "What is the leave policy for remote workers?"

    Step 1: Query hits ROOT INDEX
            → ROOT summary mentions "HR" covers leave policies
            → ROUTE to HR INDEX (not Finance or Legal)

    Step 2: Query hits HR INDEX
            → HR summary mentions "Leave Policy" covers leave rules
            → ROUTE to Leave Policy node

    Step 3: Search within Leave Policy documents
            → Find relevant chunks about remote worker leave

    This is TREE TRAVERSAL — you navigate DOWN the tree, guided by summaries.
    You DON'T search all documents. You search only the relevant BRANCH.

    This is the "relationship between indexes" they were asking about:
    - Parent-child (hierarchy)
    - Summary-based routing (which branch to take)
    - Cross-references (links between related documents)

HOW THIS DIFFERS FROM GRAPH RAG:
    Tree-based (LlamaIndex): Hierarchical, top-down navigation, summary-guided
    Graph RAG: Entity-relationship network, can traverse in any direction,
              finds connections like "Person A works at Company B which signed Contract C"

    Your answer about graph databases would be correct for GRAPH RAG,
    but NOT for tree-based vectorless RAG. They're different structures.

INTERVIEW ANSWER (Knowledge Base Updates):
    "I use incremental updates, never rebuild the entire pipeline. For vector RAG:
    detect changed documents by hash comparison, delete old vectors for that doc_id,
    re-chunk and re-embed only the changed document, and upsert new vectors. For
    vectorless RAG (inverted index): even simpler — just update the term-document
    posting lists. No embedding needed. I run this on a schedule or trigger on
    file change events. Full rebuild is only needed when changing the embedding
    model or chunking strategy."

INTERVIEW ANSWER (Index Relationships in Vectorless RAG):
    "In tree-based vectorless RAG like LlamaIndex, documents are organized in a
    hierarchical tree. Each parent node contains a summary of its children. During
    retrieval, the query traverses the tree top-down — the summary at each level
    guides routing to the correct branch. Relationships are: parent-child hierarchy,
    summary-based routing, and cross-document references stored as metadata. This
    is different from Graph RAG which uses entity-relationship networks for
    multi-hop reasoning across documents."
"""


# =================================================================================
# SECTION 9: ALL CHUNKING STRATEGIES — Complete Deep Dive
# =================================================================================
"""
INTERVIEW CONTEXT: Chunking is asked in EVERY RAG interview. Interviewers want to
know: which strategies exist, when to use which, and the tradeoffs of each.

THE FUNDAMENTAL PROBLEM:
    LLMs have limited context windows. You can't feed a 100-page PDF in one shot.
    You must break it into smaller pieces (chunks) for:
    1. RETRIEVAL: Find the specific piece relevant to the question
    2. PRECISION: Smaller chunks = more targeted answers
    3. COST: Fewer tokens per query = cheaper

    But chunking WRONG = bad retrieval = bad answers.
    The chunk must be: small enough for precision, large enough for context.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

STRATEGY 1: FIXED-SIZE CHUNKING (Naive — Don't use in production)

    WHAT: Cut every N characters regardless of content.
    HOW: Split at exactly 500 characters, then 500 more, etc.

    Code:
        from langchain_text_splitters import CharacterTextSplitter
        splitter = CharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separator=""  # splits at exact character count
        )

    EXAMPLE:
        Text: "The garden bench is made of solid teak wood. It seats 3 people
               comfortably. The dimensions are 150cm x 60cm x 90cm. It weighs
               25kg and can support up to 200kg. Assembly is required..."

        Fixed split at 100 chars might cut:
        Chunk 1: "The garden bench is made of solid teak wood. It seats 3 people comfortably. The dimensions are 15"
        Chunk 2: "0cm x 60cm x 90cm. It weighs 25kg and can support up to 200kg. Assembly is required..."

        PROBLEM: "150cm" got split into "15" and "0cm" — BROKEN!

    PROS: Simple, fast, predictable chunk sizes.
    CONS: Cuts sentences in half. Loses context at boundaries. Breaks words/numbers.
    WHEN TO USE: Never in production. Only for quick prototyping.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

STRATEGY 2: RECURSIVE CHARACTER TEXT SPLITTING (Default — Good for most cases)

    WHAT: Try to split on natural boundaries in priority order.
    HOW: First try \\n\\n (paragraphs), if chunk still too big try \\n (lines),
         then ". " (sentences), then " " (words), then "" (characters).

    SEPARATOR PRECEDENCE ORDER (memorize this!):
        ["\\n\\n", "\\n", ". ", " ", ""]

        Priority 1: \\n\\n (double newline = paragraph break) — BEST boundary
        Priority 2: \\n (single newline = line break)
        Priority 3: ". " (period + space = sentence end)
        Priority 4: " " (space = word boundary)
        Priority 5: "" (character = last resort, like fixed-size)

    WHY THIS ORDER:
        Paragraphs are the most meaningful boundaries (one topic per paragraph).
        If a paragraph is too long, fall back to lines.
        If a line is too long, fall back to sentences.
        Only split mid-word as absolute last resort.

    Code:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\\n\\n", "\\n", ". ", " ", ""]
        )

    HOW IT ACTUALLY WORKS (step by step):
        1. Try to split the document on "\\n\\n" (paragraphs)
        2. Check each resulting piece: is it <= chunk_size?
           YES → keep it as a chunk
           NO → try splitting that piece on "\\n" (next separator)
        3. Repeat down the priority list until all pieces fit

    OVERLAP (chunk_overlap=200):
        The last 200 characters of chunk N are repeated at the start of chunk N+1.
        This prevents context loss at boundaries.

        Chunk 1: "...The bench weighs 25kg and supports 200kg."
        Chunk 2: "supports 200kg. Assembly requires a Phillips screwdriver..."
                  ↑ overlap — "supports 200kg" appears in BOTH chunks

    PROS: Respects natural text boundaries. Good default for most documents.
    CONS: Doesn't understand TOPIC shifts (might put two topics in one chunk).
    WHEN TO USE: Default choice. Works well for 80% of use cases.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

STRATEGY 3: SEMANTIC CHUNKING (Smart — Splits on topic boundaries)

    WHAT: Use an embedding model to detect WHERE the topic changes in the text.
    Split at those natural topic boundaries instead of character count.

    HOW IT WORKS:
        1. Split text into sentences
        2. Embed each sentence
        3. Compare adjacent sentence embeddings (cosine similarity)
        4. Where similarity DROPS significantly = topic boundary = split here

        Sentence 1: "The bench is made of teak." → embedding A
        Sentence 2: "It seats 3 people." → embedding B (similar to A — same topic)
        Sentence 3: "It weighs 25kg." → embedding C (similar to B — same topic)
        Sentence 4: "Our return policy allows 30 days." → embedding D (DIFFERENT! — new topic)
        ↑ SPLIT HERE — topic changed from "product specs" to "return policy"

    Code:
        from langchain_experimental.text_splitter import SemanticChunker
        from langchain_openai import OpenAIEmbeddings

        embeddings = OpenAIEmbeddings()
        chunker = SemanticChunker(
            embeddings,
            breakpoint_threshold_type="percentile",  # or "standard_deviation"
            breakpoint_threshold_amount=95  # split at top 5% dissimilarity
        )
        chunks = chunker.split_text(document_text)

    PROS: Each chunk is about ONE topic. Much better retrieval precision.
    CONS: Slower (needs embedding model for chunking). Variable chunk sizes.
    WHEN TO USE: When retrieval precision matters. Documents with multiple topics.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

STRATEGY 4: PARENT-CHILD (HIERARCHICAL) CHUNKING — Small-to-Big Retrieval

    WHAT: Store TWO versions of each chunk:
        - CHILD chunk (small, 128-256 tokens) — used for RETRIEVAL
        - PARENT chunk (large, 1024-2048 tokens) — passed to LLM for GENERATION

    WHY: Small chunks give PRECISE retrieval (exact match to question).
    But small chunks lack CONTEXT for the LLM to generate a good answer.
    Solution: Retrieve using small, generate using large.

    HOW IT WORKS:
        1. Split document into LARGE parent chunks (1024 tokens)
        2. Split each parent into SMALL child chunks (256 tokens)
        3. Store both. Link children to their parent (parent_id metadata).
        4. At query time:
           a. Embed the question
           b. Search against CHILD chunks (precise matching)
           c. When a child matches, retrieve its PARENT chunk
           d. Pass the PARENT to the LLM (more context)

    EXAMPLE:
        Parent chunk (1024 tokens): Full section about "Garden Bench Specifications"
        Child chunks (256 tokens each):
            - Child 1: "Material: solid teak wood, weather-resistant..."
            - Child 2: "Dimensions: 150cm x 60cm x 90cm, seats 3..."
            - Child 3: "Weight capacity: 200kg, assembly required..."

        User asks: "What are the dimensions of the garden bench?"
        → Child 2 matches (precise: "Dimensions: 150cm...")
        → Retrieve Parent (full specifications section)
        → LLM gets full context to answer comprehensively

    Code (LlamaIndex):
        from llama_index.core.node_parser import HierarchicalNodeParser
        from llama_index.core.node_parser import get_leaf_nodes, get_root_nodes

        parser = HierarchicalNodeParser.from_defaults(
            chunk_sizes=[2048, 512, 128]  # 3 levels: grandparent, parent, child
        )
        nodes = parser.get_nodes_from_documents(documents)

    PROS: Best retrieval precision + best generation context. Production gold standard.
    CONS: More complex to implement. More storage (multiple versions of same content).
    WHEN TO USE: Production RAG where quality matters. This is the BEST technique.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

STRATEGY 5: DOCUMENT-AWARE CHUNKING (For structured documents)

    WHAT: Respect the document's OWN structure (headers, sections, tables).
    HOW: Split on headers (H1, H2, H3) or section boundaries.

    For MARKDOWN:
        from langchain_text_splitters import MarkdownHeaderTextSplitter

        headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
        ]
        splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
        chunks = splitter.split_text(markdown_text)
        # Each chunk = one section with its header as metadata

    For HTML:
        from langchain_text_splitters import HTMLHeaderTextSplitter
        splitter = HTMLHeaderTextSplitter(headers_to_split_on=[("h1", "Header 1"), ("h2", "Header 2")])

    For PDFs with sections:
        Use UnstructuredLoader with "by_title" strategy:
        loader = UnstructuredLoader(file_path, strategy="by_title")

    PROS: Each chunk is a complete section. Headers become metadata for filtering.
    CONS: Only works for well-structured documents. Sections might be too long/short.
    WHEN TO USE: Documentation, wikis, markdown files, structured PDFs.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

STRATEGY 6: SENTENCE-LEVEL CHUNKING (For precision-critical tasks)

    WHAT: Each sentence is its own chunk.
    HOW: Split on sentence boundaries (period + space, question mark, etc.)

    Code:
        from langchain_text_splitters import SentenceTransformersTokenTextSplitter
        # Or simply:
        import nltk
        sentences = nltk.sent_tokenize(text)

    PROS: Maximum precision — each chunk is exactly one statement.
    CONS: No context. A single sentence often isn't enough for the LLM.
    WHEN TO USE: Combined with parent-child (sentences as children, paragraphs as parents).
                 Or for fact-checking (verify individual claims).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

STRATEGY 7: CODE-AWARE CHUNKING (For source code)

    WHAT: Split code by functions, classes, or logical blocks — not by character count.
    HOW: Use language-specific parsers that understand code structure.

    Code:
        from langchain_text_splitters import Language, RecursiveCharacterTextSplitter

        python_splitter = RecursiveCharacterTextSplitter.from_language(
            language=Language.PYTHON,
            chunk_size=1000,
            chunk_overlap=100,
        )
        # Splits on: class definitions, function definitions, then lines

    Supported languages: Python, JavaScript, TypeScript, Java, Go, Ruby, etc.

    PROS: Each chunk is a complete function or class. Preserves code logic.
    CONS: Some functions are very long (might still need splitting).
    WHEN TO USE: Code documentation, code search, code review agents.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

STRATEGY 8: AGENTIC CHUNKING (LLM decides where to split)

    WHAT: Use an LLM to decide the optimal split points.
    HOW: Feed the document to an LLM and ask: "Where should I split this
         document into meaningful, self-contained chunks?"

    The LLM reads the content and identifies natural topic boundaries
    based on UNDERSTANDING, not just patterns.

    PROS: Best quality splits (LLM understands meaning).
    CONS: EXPENSIVE (LLM call for every document at ingestion time). Slow.
    WHEN TO USE: High-value documents where chunking quality is critical.
                 Small number of very important documents.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

COMPARISON TABLE:

    STRATEGY              PRECISION   CONTEXT   SPEED    COMPLEXITY   PRODUCTION
    Fixed-size            Low         Low       Fast     Low          ❌ Never
    Recursive Character   Medium      Medium    Fast     Low          ✅ Default
    Semantic              High        Medium    Slow     Medium       ✅ Good
    Parent-Child          High        High      Medium   High         ✅ Best
    Document-Aware        High        High      Fast     Medium       ✅ Structured docs
    Sentence-Level        Very High   Low       Fast     Low          ⚠️ With parent
    Code-Aware            High        High      Fast     Medium       ✅ Code only
    Agentic (LLM)         Very High   High      Slow     High         ⚠️ High-value only

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

RECOMMENDED CHUNK SIZES BY DOCUMENT TYPE:

    General text/articles:    chunk_size=1000, overlap=200
    Legal/Medical documents:  chunk_size=1500, overlap=300 (need more context)
    Code files:               chunk_size=500, overlap=100 (functions are shorter)
    FAQ/Q&A pairs:            chunk_size=300, overlap=0 (each Q&A is self-contained)
    Tables/Invoices:          Don't chunk — keep each table/invoice as one chunk

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

INTERVIEW ANSWER (when asked "What chunking strategies do you know?"):
    "I know 8 strategies, and I choose based on the document type and use case:

    For most documents, I start with RecursiveCharacterTextSplitter (1000 chars,
    200 overlap) — it splits on natural boundaries in priority order: paragraphs,
    lines, sentences, words.

    For production where quality matters, I use Parent-Child (hierarchical) chunking —
    small chunks for precise retrieval, parent chunks for context-rich generation.
    This is the gold standard.

    For documents with clear structure (markdown, HTML), I use document-aware chunking
    that respects headers and sections.

    For topic-diverse documents, semantic chunking detects topic shifts using
    embedding similarity and splits at those boundaries.

    For code, I use language-aware splitters that split on function/class boundaries.

    The key principle: the chunk must be small enough for precise retrieval but
    large enough to contain meaningful context. Parent-child solves this tradeoff
    by separating retrieval granularity from generation context."
"""


# =================================================================================
# SECTION 10: GOLDEN LESSONS (was Section 9)
# =================================================================================
"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

GOLDEN LESSON 1: "RAG is not just vector search."
    Production RAG involves: BM25, inverted indexes, tree structures, metadata
    filtering, ANN algorithms, reranking, compression, and routing. Vector
    similarity is just ONE piece of the puzzle.

GOLDEN LESSON 2: "Know the EXACT separator order for RecursiveCharacterTextSplitter."
    \\n\\n → \\n → . → space → character. This is asked in EVERY deep RAG interview.
    It's not random — it preserves the most meaningful boundaries first.

GOLDEN LESSON 3: "Adaptive = ROUTE before. Corrective = CHECK after."
    Never confuse these again. Adaptive decides the strategy upfront.
    Corrective catches failures after retrieval. Use BOTH in production.

GOLDEN LESSON 4: "HNSW is how vector databases actually work."
    When someone asks "how do you search millions of vectors fast?" — the answer
    is HNSW (graph-based navigation in log(N) steps). Know this cold.

GOLDEN LESSON 5: "Context overflow has 6 solutions, not just 'reduce K'."
    Compression, map-reduce, reranking + truncation, hierarchical retrieval,
    dynamic token counting. Show you know multiple approaches.

GOLDEN LESSON 6: "The deeper the interview goes, the closer you are to the offer."
    ETech went deep because they're SERIOUS about hiring someone good.
    Surface-level interviews = they'll hire anyone. Deep interviews = they want
    the BEST. You being in that room means they see potential. Fill the gaps
    and come back stronger.

GOLDEN LESSON 7: "Every failed interview is a free lesson."
    ETech just gave you a FREE masterclass on what senior RAG interviews look like.
    Now you know EXACTLY what to study. Most people never get this clarity.
    Use it. Master these 6 topics and the next deep interview will be yours.
"""

print("=" * 60)
print("Advanced RAG Deep Dive — Complete")
print("=" * 60)
print()
print("7 Sections:")
print("  1. Vectorless RAG (BM25, Inverted Indexes, Tree Structures)")
print("  2. RecursiveCharacterTextSplitter Separator Precedence")
print("  3. Advanced Retrieval Techniques (7 methods)")
print("  4. Context Window Overflow (6 solutions)")
print("  5. Adaptive RAG vs Corrective RAG (CORRECT definitions)")
print("  6. Limiting Vector Search (HNSW, IVF, Metadata, Namespaces)")
print("  7. GOLDEN LESSONS")
print()
print("These fill EVERY gap from the ETech interview.")
print("=" * 60)
