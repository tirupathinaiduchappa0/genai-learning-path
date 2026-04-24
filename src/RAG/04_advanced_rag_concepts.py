"""
RAG Lesson 4: Advanced RAG Concepts - Interview Guide and Revision Reference

This lesson covers ALL the advanced RAG concepts that are commonly asked
in GenAI/ML interviews and used in production systems. No runnable code
here - this is a CONCEPTS-ONLY lesson designed for quick revision.

HOW TO USE THIS FILE:
    - Read it section by section to build understanding.
    - Before an interview, skim the "Interview Answer" boxes.
    - Each section has: What, Why, How, When to Use, Interview Answer.

TABLE OF CONTENTS:
    Section 1:  Advanced Chunking Strategies
    Section 2:  Semantic Chunking
    Section 3:  Hybrid Search (Dense + Sparse)
    Section 4:  Re-Ranking Techniques
    Section 5:  MMR (Maximal Marginal Relevance)
    Section 6:  Query Enhancement Techniques
    Section 7:  Multimodal RAG
    Section 8:  Multi-Agent RAG
    Section 9:  RAG Evaluation Metrics
    Section 10: Production RAG Architecture Patterns
    Section 11: RAG vs Fine-Tuning (When to Use Which)
    Section 12: Common RAG Failure Modes and Fixes
    Section 13: Master Comparison Table
    Section 14: Top 25 Interview Questions with Answers

Author: GenAI Learner
"""


# ==============================================================================
# SECTION 1: ADVANCED CHUNKING STRATEGIES
# ==============================================================================
#
# WHAT YOU ALREADY KNOW (from Lesson 01 - Data Chunking):
#   - RecursiveCharacterTextSplitter: splits by characters with overlap.
#   - chunk_size=1000, chunk_overlap=200 is the standard starting point.
#   - Separators: ["\n\n", "\n", ". ", " ", ""] in priority order.
#
# WHAT YOU MIGHT NOT KNOW:
#
# 1. FIXED-SIZE CHUNKING (what you've been doing)
#    Split every N characters/tokens regardless of content.
#    Pros: Simple, predictable chunk sizes.
#    Cons: Can split mid-sentence, mid-paragraph, mid-idea.
#
# 2. DOCUMENT-STRUCTURE CHUNKING
#    Split based on the document's own structure:
#    - MarkdownHeaderTextSplitter: splits by # headers.
#    - HTMLHeaderTextSplitter: splits by <h1>, <h2>, etc.
#    - Use FIRST to split by structure, THEN by size within each section.
#    Pros: Preserves logical sections. Each chunk = one topic.
#    Cons: Sections can be very uneven in size.
#
# 3. SENTENCE-LEVEL CHUNKING
#    Split by sentences using NLP sentence tokenizers (spaCy, NLTK).
#    Then group N sentences into a chunk.
#    Pros: Never splits mid-sentence.
#    Cons: Sentences vary in length. Chunks can be uneven.
#
# 4. SEMANTIC CHUNKING (see Section 2 for deep dive)
#    Split based on MEANING, not characters.
#    Uses embeddings to detect where the topic changes.
#    Pros: Each chunk = one coherent idea.
#    Cons: Slower (requires embedding computation). More complex.
#
# 5. PARENT-CHILD CHUNKING (a.k.a. Small-to-Big)
#    Create SMALL chunks for retrieval (better precision).
#    But return the PARENT (larger) chunk for context (better recall).
#    How: Split into small chunks (200 tokens). Each small chunk
#    stores a reference to its parent chunk (1000 tokens).
#    At retrieval: search small chunks, return parent chunks.
#    Pros: Best of both worlds - precise search + rich context.
#    Cons: More complex indexing. Storage overhead.
#
# 6. SLIDING WINDOW CHUNKING
#    Overlapping windows that slide across the document.
#    chunk_overlap in RecursiveCharacterTextSplitter does this.
#    Pros: No information lost at boundaries.
#    Cons: Redundant data. More chunks = more storage + slower search.
#
# INTERVIEW ANSWER:
#   "In production, I use a layered chunking strategy. First, I split
#   by document structure (headers, sections) to preserve logical units.
#   Then within each section, I use RecursiveCharacterTextSplitter with
#   overlap. For high-precision use cases, I use parent-child chunking
#   where small chunks are indexed for search but parent chunks are
#   returned for context. The choice depends on the document type and
#   the precision/recall tradeoff needed."
#
# WHEN TO USE WHICH:
#   Fixed-size     -> Default starting point. Works for most cases.
#   Structure-based -> PDFs, markdown docs, HTML pages with clear headers.
#   Semantic       -> Research papers, legal docs where topic boundaries matter.
#   Parent-child   -> When you need precise retrieval + rich context.
#   Sliding window -> When boundary information loss is critical.


# ==============================================================================
# SECTION 2: SEMANTIC CHUNKING
# ==============================================================================
#
# WHAT IT IS:
#   Semantic chunking splits text based on MEANING, not character count.
#   It uses embeddings to detect where the TOPIC CHANGES in a document.
#
# HOW IT WORKS:
#   1. Split the document into sentences.
#   2. Embed each sentence using an embedding model.
#   3. Compare adjacent sentence embeddings (cosine similarity).
#   4. When similarity DROPS below a threshold -> that's a topic boundary.
#   5. Group sentences between boundaries into chunks.
#
# EXAMPLE:
#   Sentence 1: "The transformer uses self-attention."     -> embedding A
#   Sentence 2: "Multi-head attention runs in parallel."   -> embedding B
#   Sentence 3: "The model was trained on WMT datasets."   -> embedding C
#   Sentence 4: "Training took 3.5 days on 8 GPUs."       -> embedding D
#
#   Similarity(A, B) = 0.92 (high -> same topic: attention)
#   Similarity(B, C) = 0.45 (LOW -> topic change! attention -> training)
#   Similarity(C, D) = 0.88 (high -> same topic: training)
#
#   Result: Chunk 1 = [Sentence 1, 2], Chunk 2 = [Sentence 3, 4]
#
# LANGCHAIN IMPLEMENTATION:
#   from langchain_experimental.text_splitter import SemanticChunker
#   from langchain_openai import OpenAIEmbeddings
#
#   chunker = SemanticChunker(
#       OpenAIEmbeddings(),
#       breakpoint_threshold_type="percentile",  # or "standard_deviation"
#   )
#   chunks = chunker.split_documents(docs)
#
# BREAKPOINT STRATEGIES:
#   "percentile"          -> Split at the Nth percentile of similarity drops.
#   "standard_deviation"  -> Split when drop exceeds N standard deviations.
#   "interquartile"       -> Split at outlier drops (IQR method).
#
# PROS:
#   - Each chunk is a coherent IDEA, not an arbitrary text slice.
#   - Better retrieval quality (chunks match queries semantically).
#   - Reduces "lost in the middle" problem (relevant info at chunk edges).
#
# CONS:
#   - Slower (requires embedding every sentence).
#   - Depends on embedding model quality.
#   - Chunk sizes are unpredictable (some very small, some very large).
#   - Requires langchain_experimental package.
#
# INTERVIEW ANSWER:
#   "Semantic chunking uses embeddings to detect topic boundaries in text.
#   Instead of splitting every N characters, it embeds each sentence and
#   measures cosine similarity between adjacent sentences. When similarity
#   drops significantly, that's a topic change and a chunk boundary. This
#   produces chunks where each chunk is one coherent idea, which improves
#   retrieval quality. I'd use it for research papers or legal documents
#   where topic boundaries are important but not marked by headers."


# ==============================================================================
# SECTION 3: HYBRID SEARCH (Dense + Sparse Retrieval)
# ==============================================================================
#
# WHAT IT IS:
#   Hybrid search combines TWO retrieval methods:
#   - DENSE retrieval (vector/semantic search) -> embeddings + cosine similarity.
#   - SPARSE retrieval (keyword/lexical search) -> BM25 / TF-IDF.
#   The results are MERGED using a fusion algorithm.
#
# WHY IT EXISTS:
#   Dense search is great at SEMANTIC matching:
#     Query: "How do neural networks learn?"
#     Finds: "Backpropagation adjusts weights..." (semantically similar)
#
#   But dense search FAILS at exact keyword matching:
#     Query: "Error code ERR_CONNECTION_REFUSED"
#     Dense search might return generic networking docs.
#     BM25 would find the EXACT error code match.
#
#   Sparse search (BM25) is great at EXACT matching:
#     Query: "Python 3.12 release notes"
#     BM25 finds docs with exact "Python 3.12" keyword.
#
#   But sparse search FAILS at semantic understanding:
#     Query: "How to fix slow database queries?"
#     BM25 might miss docs about "query optimization" or "indexing strategies."
#
#   Hybrid search gets the BEST OF BOTH WORLDS.
#
# HOW IT WORKS:
#   1. Run the query through BOTH retrievers (dense + sparse).
#   2. Each retriever returns a ranked list of documents.
#   3. MERGE the two lists using Reciprocal Rank Fusion (RRF):
#      RRF_score(doc) = sum(1 / (k + rank_i)) for each retriever i
#      where k is a constant (usually 60).
#   4. Return the top-N documents by combined RRF score.
#
# RECIPROCAL RANK FUSION (RRF) EXPLAINED:
#   Dense retriever ranks: [Doc_A (rank 1), Doc_C (rank 2), Doc_B (rank 3)]
#   Sparse retriever ranks: [Doc_B (rank 1), Doc_A (rank 2), Doc_D (rank 3)]
#
#   RRF scores (k=60):
#   Doc_A: 1/(60+1) + 1/(60+2) = 0.0164 + 0.0161 = 0.0325 (top!)
#   Doc_B: 1/(60+3) + 1/(60+1) = 0.0159 + 0.0164 = 0.0323
#   Doc_C: 1/(60+2) + 0         = 0.0161
#   Doc_D: 0         + 1/(60+3) = 0.0159
#
#   Final ranking: [Doc_A, Doc_B, Doc_C, Doc_D]
#   Doc_A wins because it ranked well in BOTH retrievers.
#
# IMPLEMENTATION OPTIONS:
#   - Pinecone: Native hybrid search (dense + sparse in one index).
#   - Weaviate: Native hybrid search with configurable alpha.
#   - Elasticsearch: BM25 + dense vector search.
#   - LangChain EnsembleRetriever: Combines any two retrievers.
#
#   from langchain.retrievers import EnsembleRetriever
#   from langchain_community.retrievers import BM25Retriever
#
#   bm25 = BM25Retriever.from_documents(docs, k=4)
#   faiss = vectorstore.as_retriever(search_kwargs={"k": 4})
#   hybrid = EnsembleRetriever(
#       retrievers=[bm25, faiss],
#       weights=[0.4, 0.6],  # 40% keyword, 60% semantic
#   )
#
# INTERVIEW ANSWER:
#   "Hybrid search combines dense retrieval (embeddings + cosine similarity)
#   with sparse retrieval (BM25 keyword matching). Dense search excels at
#   semantic understanding but misses exact keywords. BM25 excels at exact
#   matching but misses semantic relationships. By merging results using
#   Reciprocal Rank Fusion, we get documents that are both semantically
#   relevant AND keyword-matched. In production, I'd use Pinecone or
#   Weaviate for native hybrid search, or LangChain's EnsembleRetriever
#   to combine FAISS with BM25."


# ==============================================================================
# SECTION 4: RE-RANKING TECHNIQUES
# ==============================================================================
#
# WHAT IT IS:
#   Re-ranking is a SECOND-PASS scoring step after initial retrieval.
#   The retriever returns top-K candidates (fast but approximate).
#   The re-ranker scores each candidate MORE ACCURATELY (slower but precise).
#
# WHY IT EXISTS:
#   Embedding retrieval uses a BI-ENCODER: query and doc embedded SEPARATELY.
#   Fast (pre-compute doc embeddings) but approximate.
#   Re-ranking uses a CROSS-ENCODER: query and doc processed TOGETHER.
#   The model sees both simultaneously -> more accurate relevance score.
#
#   Bi-encoder (retriever): "Does this doc LOOK similar to the query?"
#   Cross-encoder (re-ranker): "Does this doc actually ANSWER the query?"
#
# HOW IT WORKS:
#   1. Retriever returns top-20 candidates (fast, approximate).
#   2. Re-ranker scores each of the 20 candidates (slow, accurate).
#   3. Re-order by re-ranker scores. Return top-5.
#   This is the "retrieve-then-rerank" pattern.
#
# POPULAR RE-RANKERS:
#   - Cohere Rerank API: Production-grade, easy to use.
#   - Jina Reranker: Open-source, self-hostable.
#   - BGE Reranker (BAAI): Open-source cross-encoder.
#   - ColBERT: Late-interaction model (faster than cross-encoder).
#   - FlashRank: Lightweight, fast re-ranker.
#
# LANGCHAIN USAGE:
#   from langchain.retrievers import ContextualCompressionRetriever
#   from langchain_cohere import CohereRerank
#   reranker = CohereRerank(model="rerank-english-v3.0", top_n=5)
#   retriever_with_rerank = ContextualCompressionRetriever(
#       base_compressor=reranker,
#       base_retriever=vectorstore.as_retriever(search_kwargs={"k": 20}),
#   )
#
# INTERVIEW ANSWER:
#   "Re-ranking is a two-stage retrieval pattern. A fast bi-encoder retriever
#   returns top-K candidates. Then a cross-encoder re-ranker scores each by
#   processing query and document TOGETHER, producing more accurate scores.
#   Retrieve top-20 with FAISS, re-rank to top-5 with Cohere Rerank."


# ==============================================================================
# SECTION 5: MMR (MAXIMAL MARGINAL RELEVANCE)
# ==============================================================================
#
# WHAT IT IS:
#   MMR balances RELEVANCE and DIVERSITY in retrieval results.
#   Instead of top-K most similar (which may all say the same thing),
#   MMR returns docs that are relevant AND diverse from each other.
#
# THE PROBLEM:
#   If your vector store has 5 chunks from the same paragraph,
#   standard search returns all 5 (redundant context).
#   MMR says: "Give me relevant docs, but each should add NEW info."
#
# THE FORMULA:
#   MMR = argmax [lambda * sim(query, doc) - (1-lambda) * max(sim(doc, selected))]
#   lambda = 1.0 -> pure relevance. lambda = 0.0 -> pure diversity.
#   lambda = 0.5 -> balanced (recommended).
#
# LANGCHAIN USAGE:
#   retriever = vectorstore.as_retriever(
#       search_type="mmr",
#       search_kwargs={"k": 6, "lambda_mult": 0.5},
#   )
#
# INTERVIEW ANSWER:
#   "MMR iteratively selects documents that are relevant to the query but
#   diverse from already-selected documents. Lambda controls the tradeoff:
#   1.0 is pure relevance, 0.0 is pure diversity, 0.5 is balanced. I use
#   MMR when the vector store has overlapping content."


# ==============================================================================
# SECTION 6: QUERY ENHANCEMENT TECHNIQUES
# ==============================================================================
#
# TECHNIQUE 1: QUERY EXPANSION (MultiQueryRetriever)
#   Generate MULTIPLE phrasings of the query, retrieve for each, merge.
#   Original: "How does attention work?"
#   Expanded: ["How does attention work?",
#              "What is self-attention in transformers?",
#              "Explain the attention mechanism in neural networks"]
#   Retrieve for all 3, merge, deduplicate. Catches different phrasings.
#
# TECHNIQUE 2: QUERY DECOMPOSITION
#   Break a COMPLEX query into SIMPLER sub-queries.
#   Original: "Compare transformer attention with RNN attention"
#   Decomposed: ["How does transformer attention work?",
#                "How does RNN attention work?",
#                "Which is better for long sequences?"]
#
# TECHNIQUE 3: STEP-BACK PROMPTING
#   Ask a MORE GENERAL question first for broader context.
#   Original: "What learning rate was used in the transformer paper?"
#   Step-back: "What are the training hyperparameters of the transformer?"
#
# TECHNIQUE 4: HyDE (Hypothetical Document Embeddings)
#   Generate a HYPOTHETICAL answer first (without retrieval).
#   Use the hypothetical answer as the search query.
#   Why: The hypothetical answer is closer in embedding space to the
#   actual answer documents than the question itself.
#
# TECHNIQUE 5: CONTEXTUAL COMPRESSION
#   After retrieval, COMPRESS each document to only the relevant parts.
#   A 1000-token chunk might have only 2 relevant sentences.
#   Compression extracts just those 2 sentences. Reduces noise.
#
# INTERVIEW ANSWER:
#   "I use query expansion for ambiguous queries, decomposition for complex
#   multi-part questions, HyDE for abstract questions where the hypothetical
#   answer is closer to actual docs in embedding space, and contextual
#   compression to extract only relevant sentences from retrieved chunks."


# ==============================================================================
# SECTION 7: MULTIMODAL RAG
# ==============================================================================
#
# WHAT IT IS:
#   Standard RAG works with TEXT only. Multimodal RAG handles IMAGES,
#   TABLES, CHARTS, and other non-text content.
#
# THREE APPROACHES:
#
# Approach 1: TEXT EXTRACTION (OCR)
#   Use OCR (Tesseract, AWS Textract) to extract text from images/tables.
#   Treat extracted text as regular chunks.
#   Pros: Simple. Works with existing pipeline. Cons: Loses visual context.
#
# Approach 2: MULTIMODAL EMBEDDINGS (CLIP)
#   Embed BOTH text and images into the SAME vector space.
#   At retrieval, the query matches both text and images.
#   Pros: Images are first-class citizens. Cons: More complex pipeline.
#
# Approach 3: VISION LLM SUMMARIZATION
#   Use GPT-4V/Claude 3/Gemini to DESCRIBE each image as text.
#   Index the description. Match against it at retrieval.
#   Pros: Rich descriptions. Cons: Expensive (vision LLM calls).
#
# MULTIMODAL AGENTIC RAG:
#   An agent DECIDES: search text KB or image DB? Use OCR or vision LLM?
#   Routes based on query type.
#
# INTERVIEW ANSWER:
#   "Multimodal RAG handles non-text content via three approaches: OCR for
#   text extraction, CLIP for same-space text+image embeddings, or vision
#   LLM summarization. In production, I'd use vision LLM for quality or
#   OCR for cost efficiency."


# ==============================================================================
# SECTION 8: MULTI-AGENT RAG AND AUTONOMOUS RAG
# ==============================================================================
#
# MULTI-AGENT RAG:
#   Multiple specialized agents collaborate on RAG:
#   Agent 1 (Planner): Analyzes query, decides strategy.
#   Agent 2 (Retriever): Executes retrieval.
#   Agent 3 (Grader): Evaluates document quality.
#   Agent 4 (Generator): Produces the answer.
#   Agent 5 (Reviewer): Checks for hallucination.
#   A SUPERVISOR coordinates all of them.
#
# AUTONOMOUS RAG:
#   Fully autonomous system that decides WHEN, WHAT, and HOW MUCH to
#   retrieve. Self-evaluates and self-corrects without human intervention.
#   Autonomous RAG = Adaptive RAG + memory + learning + full autonomy.
#
# GRAPH RAG (Microsoft):
#   Builds a KNOWLEDGE GRAPH from documents (entities + relationships).
#   Retrieval traverses the graph instead of vector search.
#   Better for questions about entity relationships.
#   More expensive to build (requires entity extraction pipeline).
#
# INTERVIEW ANSWER:
#   "Multi-Agent RAG uses specialized agents coordinated by a supervisor.
#   Autonomous RAG makes all decisions independently. Graph RAG builds a
#   knowledge graph for entity-relationship queries. Each pattern suits
#   different complexity levels and use cases."


# ==============================================================================
# SECTION 9: RAG EVALUATION METRICS
# ==============================================================================
#
# You can't improve what you can't measure. RAG evaluation is CRITICAL
# for production systems. Here are the key metrics:
#
# 1. CONTEXT RELEVANCE (Retrieval Quality)
#    "Are the retrieved documents relevant to the query?"
#    Measured by: grading each doc (like we did in CRAG/Adaptive RAG).
#    Tools: RAGAS framework, LangSmith evaluators.
#
# 2. FAITHFULNESS (Hallucination Check)
#    "Is the generated answer grounded in the retrieved documents?"
#    The answer should ONLY contain information from the context.
#    Measured by: checking each claim in the answer against the docs.
#    Tools: RAGAS faithfulness metric, custom hallucination graders.
#
# 3. ANSWER RELEVANCE
#    "Does the generated answer actually address the question?"
#    The answer might be grounded in docs but not answer the question.
#    Measured by: comparing the answer to the original question.
#    Tools: RAGAS answer relevancy, custom answer graders.
#
# 4. CONTEXT PRECISION
#    "How much of the retrieved context is actually useful?"
#    If you retrieve 10 chunks but only 2 are relevant, precision is low.
#    High precision = less noise for the LLM.
#
# 5. CONTEXT RECALL
#    "Did we retrieve ALL the relevant information?"
#    If the answer requires info from 5 chunks but we only found 3,
#    recall is low. High recall = complete information.
#
# 6. ANSWER CORRECTNESS
#    "Is the answer factually correct?"
#    Compared against a ground truth answer (if available).
#    Measured by: semantic similarity + factual overlap.
#
# THE RAGAS FRAMEWORK:
#   RAGAS (Retrieval Augmented Generation Assessment) is the standard
#   framework for evaluating RAG pipelines. It provides:
#   - Faithfulness, Answer Relevancy, Context Precision, Context Recall.
#   - Works with LangChain and LangSmith.
#   - Can run automated evaluations on test datasets.
#
# INTERVIEW ANSWER:
#   "I evaluate RAG pipelines using four key metrics: Context Relevance
#   (are retrieved docs relevant?), Faithfulness (is the answer grounded
#   in the docs?), Answer Relevance (does it address the question?), and
#   Context Precision/Recall (did we get the right amount of context?).
#   I use the RAGAS framework for automated evaluation and LangSmith for
#   tracing and debugging individual queries."


# ==============================================================================
# SECTION 10: PRODUCTION RAG ARCHITECTURE PATTERNS
# ==============================================================================
#
# PATTERN 1: NAIVE RAG (what most tutorials teach)
#   Query -> Retrieve -> Generate.
#   No grading, no routing, no correction.
#   Fine for demos. NOT for production.
#
# PATTERN 2: ADVANCED RAG (what production systems use)
#   Query Enhancement -> Hybrid Retrieval -> Re-ranking -> Generation.
#   Adds pre-retrieval and post-retrieval optimization.
#   Good for most production use cases.
#
# PATTERN 3: MODULAR RAG (what enterprise systems use)
#   Pluggable modules: Router, Retriever, Grader, Generator, Validator.
#   Each module can be swapped independently.
#   This is what Agentic/Corrective/Adaptive RAG implement.
#
# PRODUCTION CHECKLIST:
#   [ ] Chunking strategy optimized for your document types.
#   [ ] Hybrid search (dense + sparse) for robust retrieval.
#   [ ] Re-ranking for precision (Cohere Rerank or similar).
#   [ ] MMR for diversity (avoid redundant context).
#   [ ] Query enhancement (expansion or decomposition).
#   [ ] Document grading (filter irrelevant chunks).
#   [ ] Hallucination detection (post-generation check).
#   [ ] Answer validation (does it address the question?).
#   [ ] Evaluation pipeline (RAGAS metrics on test set).
#   [ ] Observability (LangSmith tracing for every query).
#   [ ] Caching (cache frequent queries to reduce latency/cost).
#   [ ] Guardrails (input validation, output filtering).


# ==============================================================================
# SECTION 11: RAG vs FINE-TUNING (When to Use Which)
# ==============================================================================
#
# This is one of the MOST COMMON interview questions.
#
# RAG (Retrieval-Augmented Generation):
#   - Retrieves external knowledge at QUERY TIME.
#   - Knowledge can be updated WITHOUT retraining the model.
#   - The model stays general-purpose.
#   - Best for: factual Q&A, knowledge bases, documentation.
#   - Cost: per-query (retrieval + LLM call).
#
# FINE-TUNING:
#   - Trains the model on domain-specific data BEFORE deployment.
#   - Knowledge is baked INTO the model weights.
#   - Updating knowledge requires RETRAINING.
#   - Best for: style/tone adaptation, domain-specific reasoning.
#   - Cost: upfront (training) + per-query (inference).
#
# WHEN TO USE RAG:
#   - Knowledge changes frequently (news, product docs, prices).
#   - You need citations/sources for answers.
#   - You have a large knowledge base (too big to fine-tune on).
#   - You need to add new knowledge without retraining.
#
# WHEN TO USE FINE-TUNING:
#   - You need the model to adopt a specific style/tone.
#   - Domain-specific reasoning (medical, legal, financial).
#   - The knowledge is stable and doesn't change often.
#   - You need faster inference (no retrieval step).
#
# WHEN TO USE BOTH (RAG + Fine-Tuning):
#   - Fine-tune for domain reasoning + RAG for current knowledge.
#   - Example: Fine-tune a medical LLM for clinical reasoning,
#     then use RAG to retrieve the latest drug interactions.
#
# INTERVIEW ANSWER:
#   "RAG and fine-tuning solve different problems. RAG retrieves external
#   knowledge at query time, so it's best when knowledge changes frequently
#   and you need citations. Fine-tuning bakes knowledge into model weights,
#   so it's best for style adaptation and domain-specific reasoning. In
#   production, I often combine both: fine-tune for domain reasoning and
#   use RAG for up-to-date factual knowledge."


# ==============================================================================
# SECTION 12: COMMON RAG FAILURE MODES AND FIXES
# ==============================================================================
#
# FAILURE 1: "Lost in the Middle"
#   Problem: LLMs pay more attention to the beginning and end of context,
#            ignoring information in the middle.
#   Fix: Put the most relevant chunks FIRST. Use re-ranking.
#        Keep context short (fewer, better chunks).
#
# FAILURE 2: Irrelevant Retrieval
#   Problem: Vector search returns semantically similar but irrelevant docs.
#   Fix: Hybrid search (add BM25). Re-ranking. Document grading.
#        Better chunking (semantic chunking).
#
# FAILURE 3: Hallucination from Context
#   Problem: LLM generates plausible but wrong info even with good context.
#   Fix: Hallucination grading (Adaptive RAG). Lower temperature.
#        Explicit "only use provided context" in the prompt.
#
# FAILURE 4: Incomplete Retrieval
#   Problem: The answer requires info from multiple chunks, but only some
#            were retrieved.
#   Fix: Increase k (retrieve more docs). Query decomposition.
#        Parent-child chunking (return larger parent chunks).
#
# FAILURE 5: Stale Knowledge
#   Problem: Vector store contains outdated information.
#   Fix: Web search fallback (CRAG). Regular re-indexing pipeline.
#        Metadata filtering by date.
#
# FAILURE 6: Ambiguous Queries
#   Problem: User query is vague, retriever returns scattered results.
#   Fix: Query expansion. Query decomposition. Clarification prompts.
#        HyDE (hypothetical document embeddings).
#
# FAILURE 7: Context Window Overflow
#   Problem: Too many retrieved chunks exceed the LLM's context window.
#   Fix: Contextual compression. Re-ranking to top-N.
#        Summarize chunks before passing to LLM.
#        Use a long-context model (128K+ tokens).


# ==============================================================================
# SECTION 13: MASTER COMPARISON TABLE
# ==============================================================================
#
# Technique         | Stage        | What It Does                    | When to Use
# ------------------|--------------|--------------------------------|------------------
# Semantic Chunking | Indexing     | Splits by meaning, not chars   | Research papers
# Parent-Child      | Indexing     | Small search, big return       | Precision + context
# Hybrid Search     | Retrieval    | Dense + Sparse combined        | Always in production
# BM25              | Retrieval    | Keyword/lexical matching       | Exact term queries
# MMR               | Retrieval    | Relevance + diversity balance  | Redundant content
# Re-Ranking        | Post-Retrieval| Cross-encoder re-scoring      | Always in production
# Query Expansion   | Pre-Retrieval| Multiple query phrasings       | Ambiguous queries
# Query Decomposition| Pre-Retrieval| Break complex into simple     | Multi-part questions
# HyDE              | Pre-Retrieval| Hypothetical answer as query   | Abstract questions
# Contextual Compress| Post-Retrieval| Extract relevant sentences   | Long chunks
# Doc Grading       | Post-Retrieval| Filter irrelevant docs        | CRAG/Adaptive RAG
# Hallucination Check| Post-Generation| Is answer grounded?         | Adaptive RAG
# Answer Grading    | Post-Generation| Does answer address query?   | Adaptive RAG
# Web Search Fallback| Correction  | External knowledge when local fails| CRAG
# ------------------|--------------|--------------------------------|------------------


# ==============================================================================
# SECTION 14: TOP 25 INTERVIEW QUESTIONS WITH ANSWERS
# ==============================================================================
#
# Q1: What is RAG and why is it needed?
# A: RAG combines retrieval with generation. LLMs have a knowledge cutoff
#    and can hallucinate. RAG grounds the LLM's response in retrieved
#    documents, providing up-to-date, factual, and citable answers.
#
# Q2: Explain the RAG pipeline end-to-end.
# A: Indexing: Load docs -> chunk -> embed -> store in vector DB.
#    Retrieval: Query -> embed -> similarity search -> top-K docs.
#    Generation: Query + retrieved docs -> LLM prompt -> answer.
#
# Q3: What chunking strategy would you use for a legal document?
# A: Structure-based chunking first (split by sections/clauses), then
#    semantic chunking within sections to preserve legal concepts.
#    Parent-child for precise search with full clause context.
#
# Q4: What is the difference between dense and sparse retrieval?
# A: Dense uses embeddings + cosine similarity (semantic matching).
#    Sparse uses BM25/TF-IDF (keyword matching). Dense catches meaning,
#    sparse catches exact terms. Hybrid combines both.
#
# Q5: How does re-ranking improve retrieval quality?
# A: Initial retrieval uses a fast bi-encoder (approximate). Re-ranking
#    uses a slow cross-encoder that processes query+doc together for
#    more accurate scoring. Retrieve 20, re-rank to top 5.
#
# Q6: What is MMR and when would you use it?
# A: MMR balances relevance and diversity. It prevents returning
#    redundant documents that all say the same thing. Use when the
#    vector store has overlapping content.
#
# Q7: Explain Agentic RAG vs Corrective RAG vs Adaptive RAG.
# A: Agentic: Agent decides which tool/source to query.
#    Corrective: Grades docs, falls back to web search if irrelevant.
#    Adaptive: Routes query first, grades docs, validates generation
#    for hallucination and answer relevance. Adaptive is the most complete.
#
# Q8: How do you handle hallucination in RAG?
# A: Post-generation hallucination grading (is the answer grounded in
#    the retrieved docs?). Lower temperature. Explicit prompt instructions
#    to only use provided context. Adaptive RAG's validation loop.
#
# Q9: What is HyDE and when would you use it?
# A: HyDE generates a hypothetical answer without retrieval, then uses
#    that answer as the search query. The hypothetical answer is closer
#    in embedding space to actual documents than the question. Use for
#    abstract or conceptual questions.
#
# Q10: How would you evaluate a RAG pipeline?
# A: Four metrics: Context Relevance (are docs relevant?), Faithfulness
#     (is answer grounded?), Answer Relevance (does it address the query?),
#     Context Precision/Recall. Use RAGAS framework + LangSmith tracing.
#
# Q11: RAG vs Fine-Tuning - when to use which?
# A: RAG for frequently changing knowledge, citations needed, large KBs.
#     Fine-tuning for style/tone, domain reasoning, stable knowledge.
#     Combine both: fine-tune for reasoning + RAG for current facts.
#
# Q12: What is the "lost in the middle" problem?
# A: LLMs pay more attention to the start and end of context, ignoring
#     the middle. Fix: put most relevant chunks first (re-ranking),
#     keep context short, use fewer but better chunks.
#
# Q13: How do you handle multimodal content in RAG?
# A: Three approaches: (1) OCR to extract text from images, (2) multimodal
#     embeddings (CLIP) for same-space text+image search, (3) vision LLM
#     to describe images, then index descriptions as text.
#
# Q14: What is Graph RAG?
# A: Instead of chunking text, Graph RAG extracts entities and relationships
#     to build a knowledge graph. Retrieval traverses the graph. Better for
#     questions about entity relationships. More expensive to build.
#
# Q15: How would you handle a RAG system with 10 million documents?
# A: Approximate Nearest Neighbor (ANN) search with FAISS/Pinecone.
#     Metadata filtering to narrow search space. Hierarchical indexing
#     (coarse search first, then fine search). Caching frequent queries.
#
# Q16: What is contextual compression?
# A: After retrieval, compress each chunk to only the sentences relevant
#     to the query. Reduces noise in the context. LangChain has
#     ContextualCompressionRetriever for this.
#
# Q17: How do you handle multi-turn conversations in RAG?
# A: Use conversation history as additional context. Reformulate the
#     current query using chat history (standalone question generation).
#     LangChain's create_history_aware_retriever does this.
#
# Q18: What is query decomposition and when would you use it?
# A: Breaking a complex question into simpler sub-questions. Retrieve
#     for each sub-question separately, then synthesize. Use for
#     multi-part or comparative questions.
#
# Q19: How do you keep a RAG knowledge base up to date?
# A: Scheduled re-indexing pipeline. Incremental updates (add new docs
#     without re-indexing everything). Metadata timestamps for filtering.
#     Web search fallback for real-time information (CRAG pattern).
#
# Q20: What vector databases would you use in production?
# A: Pinecone (managed, scalable, hybrid search). Weaviate (open-source,
#     hybrid search). Qdrant (open-source, fast). Chroma (simple, local).
#     FAISS (library, not a DB, good for prototyping).
#
# Q21: What is the difference between a retriever and a vector store?
# A: A vector store is the DATABASE that stores embeddings (FAISS, Pinecone).
#     A retriever is the INTERFACE that queries the vector store and returns
#     documents. In LangChain: vectorstore.as_retriever() creates a retriever.
#
# Q22: How do you handle conflicting information in retrieved documents?
# A: Re-ranking to prioritize authoritative sources. Metadata filtering
#     by date (prefer recent). Explicit prompt: "If sources conflict,
#     note the disagreement." Multi-source citation in the answer.
#
# Q23: What is the role of embeddings in RAG?
# A: Embeddings convert text into dense vectors that capture semantic meaning.
#     Similar texts have similar vectors (high cosine similarity). This enables
#     semantic search: find documents that MEAN the same thing as the query,
#     even if they use different words.
#
# Q24: How would you reduce latency in a production RAG system?
# A: Cache frequent queries. Use fast embedding models (all-MiniLM-L6-v2).
#     Pre-compute document embeddings. Use ANN search (not exact).
#     Reduce k (fewer retrieved docs). Streaming for perceived speed.
#     Async retrieval + generation in parallel where possible.
#
# Q25: Design a production RAG system for a customer support platform.
# A: Knowledge base: FAQ docs + product manuals + troubleshooting guides.
#     Indexing: Structure-based chunking + hybrid search (BM25 + dense).
#     Retrieval: Hybrid search -> re-ranking -> MMR for diversity.
#     Generation: RAG prompt with citations. Hallucination check.
#     Routing: Classify query type (FAQ, technical, billing) -> route
#     to specialized retrievers. Fallback: web search for unknown topics.
#     Evaluation: RAGAS metrics on test set. LangSmith for tracing.
#     Human-in-the-loop: Escalate low-confidence answers to human agents.


# ==============================================================================
# QUICK REVISION CHEAT SHEET
# ==============================================================================
#
# PRE-RETRIEVAL OPTIMIZATION:
#   Query Expansion    -> Multiple phrasings of the same question.
#   Query Decomposition -> Break complex into simple sub-queries.
#   HyDE               -> Hypothetical answer as search query.
#   Step-Back Prompting -> Ask a broader question first.
#
# INDEXING OPTIMIZATION:
#   Semantic Chunking  -> Split by meaning, not characters.
#   Parent-Child       -> Small chunks for search, big for context.
#   Structure-Based    -> Split by headers/sections first.
#
# RETRIEVAL OPTIMIZATION:
#   Hybrid Search      -> Dense (embeddings) + Sparse (BM25).
#   MMR                -> Relevance + diversity balance.
#
# POST-RETRIEVAL OPTIMIZATION:
#   Re-Ranking         -> Cross-encoder re-scores top-K candidates.
#   Contextual Compression -> Extract only relevant sentences.
#   Document Grading   -> Filter irrelevant chunks (CRAG).
#
# POST-GENERATION VALIDATION:
#   Hallucination Check -> Is answer grounded in docs?
#   Answer Grading      -> Does answer address the question?
#   Self-Correction Loop -> Rewrite + retry if validation fails.
#
# RAG PATTERNS (progressive complexity):
#   Basic RAG       -> retrieve -> generate (no checks)
#   Agentic RAG     -> agent picks source -> grade -> generate
#   Corrective RAG  -> retrieve -> grade -> web fallback -> generate
#   Adaptive RAG    -> route -> retrieve -> grade -> generate -> validate
