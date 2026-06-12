"""
===================================================================================
RAG EVALUATION DEEP DIVE — Senior AI Engineer Interview Mastery
===================================================================================

This lesson covers EVERYTHING senior interviewers test on RAG evaluation.
Required for: PwC, ETech, JPMC, and any senior GenAI role.

WHY THIS MATTERS:
    Building RAG = easy. Anyone can do it in 50 lines.
    Evaluating RAG correctly = HARD. This is what separates juniors from seniors.
    Senior interviewers ALWAYS ask: "How do you know your RAG is working?"

SECTIONS:
    1.  Why RAG Evaluation is Hard (The Two Failures)
    2.  Component Evaluation vs End-to-End Evaluation
    3.  RETRIEVAL Evaluation Metrics (Precision@k, Recall@k, MRR, NDCG)
    4.  GENERATION Evaluation Metrics (Faithfulness, Answer Relevance)
    5.  RAGAS Framework — The 6 Core Metrics in Depth
    6.  Building a Gold-Standard Evaluation Dataset
    7.  Evaluation Pipeline — From Dev to Production
    8.  LLM-as-Judge Pattern (How RAGAS Actually Works)
    9.  Tools — RAGAS, LangSmith, DeepEval, TruLens (Compared)
    10. Continuous Evaluation in Production (Drift Detection)
    11. Common Evaluation Pitfalls and How to Avoid Them
    12. 25+ Interview Q&A
    13. GOLDEN LESSONS

This is the depth a 20+ year experienced architect would test.
===================================================================================
"""


# =================================================================================
# SECTION 1: WHY RAG EVALUATION IS HARD (The Two Failures)
# =================================================================================
"""
THE FUNDAMENTAL PROBLEM:

    Traditional software: input → output → check if output matches expected.
    "2 + 2 = 4" → easy to test.

    RAG systems: input → retrieve → generate → output.
    "What is our refund policy?" → ???
    There's NO single correct answer. Many wordings could be correct.
    AND there are TWO places where things can fail.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

THE TWO FAILURES IN RAG:

    FAILURE 1 — RETRIEVAL FAILURE:
        The retriever didn't find the right documents.
        Even if the LLM is perfect, it has nothing to work with.
        Symptom: "I don't have enough information" or wrong answer from wrong docs.

    FAILURE 2 — GENERATION FAILURE:
        The retriever found the right docs, but the LLM:
        - Hallucinated (made up facts not in the docs)
        - Misread the docs (got the meaning wrong)
        - Ignored the docs (used training data instead)
        - Gave a vague/irrelevant answer

    ANY EVALUATION MUST DIAGNOSE BOTH.
    If you only check the final answer, you can't tell WHERE it went wrong.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

WHY YOU CAN'T USE TRADITIONAL METRICS:

    Traditional ML metrics (accuracy, F1) require EXACT MATCHES.
    RAG produces FREE-FORM TEXT. There's no exact match.

    Example:
        Question: "What is our refund policy?"
        Answer 1: "We offer 30-day refunds for unused products."
        Answer 2: "Customers can return products within 30 days for a full refund."
        Answer 3: "Refunds are available within 30 days of purchase."

    All three are CORRECT. All three say something different.
    Traditional accuracy = 0% match (no exact same string).
    But semantically, all three convey the same information.

    THIS IS WHY RAG NEEDS SPECIALIZED EVALUATION METRICS.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

THE INTERVIEWER'S TEST:

    Junior answer: "I check if the answer looks right."
    Mid-level: "I use RAGAS framework."
    SENIOR: "I evaluate retrieval and generation SEPARATELY using a labeled
    eval dataset, measuring context precision/recall for retrieval and
    faithfulness/answer relevance for generation. I run this continuously
    in production with drift detection."

INTERVIEW ANSWER:
    "RAG evaluation is hard because there are two independent failure modes —
    retrieval can fail (wrong documents) or generation can fail (hallucination,
    misreading). And the output is free-form text, so traditional accuracy
    metrics don't work. I evaluate both stages SEPARATELY using component-level
    metrics, then end-to-end metrics on top. Without this separation, you can't
    diagnose WHERE the failure occurred when accuracy drops."
"""


# =================================================================================
# SECTION 2: COMPONENT EVALUATION vs END-TO-END EVALUATION
# =================================================================================
"""
THE TWO LEVELS OF EVALUATION:

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

LEVEL 1: COMPONENT EVALUATION (Diagnose specific stages)

    Evaluate each STAGE of the pipeline independently.
    Tells you WHICH component is failing.

    STAGES TO EVALUATE:

    A) CHUNKING quality
       - Are chunks self-contained (have enough context)?
       - Are chunks too large (too much noise)?
       - Are chunks too small (lost context)?
       Metric: Manual review + retrieval quality downstream

    B) EMBEDDING quality
       - Do similar texts produce similar vectors?
       - Domain-specific test: do legal terms cluster correctly?
       Metric: STS (Semantic Textual Similarity) on domain test set

    C) RETRIEVAL quality (most important)
       - Does the retriever find the relevant chunks?
       - Are they ranked correctly?
       Metrics: Precision@k, Recall@k, MRR, NDCG

    D) GENERATION quality (given correct context)
       - Given perfect docs, does the LLM produce good answers?
       Metrics: Faithfulness, Answer Relevance

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

LEVEL 2: END-TO-END EVALUATION (User-facing quality)

    Evaluate the COMPLETE pipeline as a black box.
    Tells you WHAT THE USER experiences.
    In AI and RAG, "black box" simply means:
    You judge a system only by its inputs and outputs, without caring about (or looking at) what happens inside. 
    White Box Evaluation  ----> You inspect the internals.

    Black-box evaluation tells you that something is wrong. White-box evaluation tells you why it's wrong

    METRICS:
    - Answer Correctness (vs ground truth)
    - Answer Relevance (does it address the question)
    - User satisfaction (if you have user feedback)
    - Latency (response time)
    - Cost per query

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

THE EVALUATION HIERARCHY (How to debug a failing RAG system):

    Step 1: Run end-to-end eval. Is overall answer correctness < 70%?
        YES → Something is broken. Continue to Step 2.

    Step 2: Run RETRIEVAL eval. Is Recall@4 < 80%?
        YES → Retrieval is failing. Fix that first.
              Possible causes: bad chunking, wrong embedding model,
              poor search method (vector only vs hybrid).

    Step 3: If retrieval is OK but answer is wrong, run GENERATION eval.
        Is Faithfulness < 80%?
        YES → LLM is hallucinating. Fix the prompt or model.

    Step 4: If both retrieval and faithfulness are OK but answer is irrelevant,
        check Answer Relevance.
        The LLM is finding facts but not answering the question.
        Fix: prompt engineering.

INTERVIEW ANSWER:
    "I evaluate at two levels. Component-level: chunking quality, embedding
    quality, retrieval (Precision@k, Recall@k, MRR), and generation (Faithfulness,
    Answer Relevance) independently. End-to-end: Answer Correctness and user
    satisfaction. Component evaluation lets me DIAGNOSE failures — if end-to-end
    accuracy drops, I check each stage to find the broken component. Without
    this separation, you're guessing."
"""


# =================================================================================
# SECTION 3: RETRIEVAL EVALUATION METRICS (Precision@k, Recall@k, MRR, NDCG)
# =================================================================================
"""
These metrics evaluate the RETRIEVER independently — given a query, did it
return the right documents? You need a LABELED eval dataset where each query
has known "relevant documents" (ground truth).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

METRIC 1: PRECISION@k

    Definition: Of the top-k retrieved docs, what fraction are RELEVANT?

    Formula: (relevant docs in top-k) / k

    Of all the documents we predicted as relevant (positive), how many were actually relevant?
    precission = TP / (TP + FP)

    Example:
        Query: "What is our refund policy?"
        Ground truth relevant docs: [doc_5, doc_12, doc_99]
        Top-4 retrieved: [doc_5, doc_12, doc_42, doc_77]
        Relevant in top-4: doc_5, doc_12 = 2
        Precision@4 = 2/4 = 0.5 (50%)

    INTERPRETATION:
        High Precision@k = retrieved docs are mostly relevant (low noise)
        Low Precision@k = many irrelevant docs in top-k (noise problem)

    WHEN IT MATTERS: When you want CLEAN context for the LLM.
        High noise = LLM gets confused or hallucinates.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

METRIC 2: RECALL@k

    Definition: Of all RELEVANT docs, how many did we retrieve in top-k?

    Formula: (relevant docs in top-k) / (total relevant docs)

    of all the actually positive (relevant) docs, how many did we predict as positive (retrieve)?
        REcall = TP / (TP + FN)

    Example (same as above):
        Total relevant docs: 3 (doc_5, doc_12, doc_99)
        Relevant in top-4: 2 (doc_5, doc_12)
        Recall@4 = 2/3 = 0.67 (67%)
        We MISSED doc_99!

    INTERPRETATION:
        High Recall@k = we captured most of the relevant info
        Low Recall@k = we MISSED important docs (won't be in LLM context)

    WHEN IT MATTERS: When the LLM needs ALL relevant info to answer.
        Missing one critical doc = wrong answer.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PRECISION vs RECALL TRADE-OFF:

    Increase k (retrieve more docs):
    - Recall goes UP (more chance to capture relevant docs)
    - Precision goes DOWN (more irrelevant docs in the mix)

    Decrease k (retrieve fewer docs):
    - Precision goes UP (only most-similar docs)
    - Recall goes DOWN (might miss relevant docs)

    OPTIMAL K depends on your application:
    - Critical info needed (medical, legal) → favor RECALL (k=10)
    - Clean context needed (chatbot) → favor PRECISION (k=3-4)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

METRIC 3: MRR (Mean Reciprocal Rank)

    Definition: How EARLY does the FIRST relevant document appear?

    Formula: MRR = average(1 / rank_of_first_relevant_doc)

    Example:
        Query 1: First relevant doc at rank 1 → 1/1 = 1.0
        Query 2: First relevant doc at rank 3 → 1/3 = 0.33
        Query 3: First relevant doc at rank 5 → 1/5 = 0.20
        MRR = (1.0 + 0.33 + 0.20) / 3 = 0.51

    INTERPRETATION:
        MRR = 1.0 → relevant doc is ALWAYS first
        MRR = 0.5 → relevant doc is on average at rank 2
        MRR = 0.1 → relevant doc is on average at rank 10

    WHEN IT MATTERS: When users see top-1 result.
        For chatbots that mainly use top-1 chunk → MRR is critical.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

METRIC 4: NDCG (Normalized Discounted Cumulative Gain)

    Most sophisticated. Considers BOTH relevance AND position.

    INTUITION:
    - A relevant doc at position 1 is worth MORE than at position 5.
    - Some docs are "very relevant" (gain=3) vs "somewhat relevant" (gain=1).

    Formula (simplified):
        DCG = sum(gain_i / log2(rank_i + 1))
        NDCG = DCG / Ideal DCG (best possible ranking)

    NDCG = 1.0 means the ranking is PERFECT.

    USED BY: Google, Amazon, Netflix for ranking quality.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CHEAT SHEET — When to Use Which:

    METRIC          ANSWERS                                  USE WHEN
    Precision@k     "Are retrieved docs clean?"              You want low noise
    Recall@k        "Did we capture all relevant info?"       You can't afford to miss
    MRR             "Is the first result usually right?"     Top-1 matters most
    NDCG            "Is the ranking order optimal?"           Position matters

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CODE EXAMPLE:

    def precision_at_k(retrieved, relevant, k):
        retrieved_at_k = retrieved[:k]
        hits = len(set(retrieved_at_k) & set(relevant))
        return hits / k

    def recall_at_k(retrieved, relevant, k):
        retrieved_at_k = retrieved[:k]
        hits = len(set(retrieved_at_k) & set(relevant))
        return hits / len(relevant) if relevant else 0

    def mrr(retrieved, relevant):
        for rank, doc in enumerate(retrieved, start=1):
            if doc in relevant:
                return 1 / rank
        return 0

INTERVIEW ANSWER:
    "I evaluate retrieval with four metrics: Precision@k measures how clean the
    retrieved docs are (low noise), Recall@k measures whether we captured all
    relevant info (low miss rate), MRR measures how early the first relevant
    doc appears, and NDCG measures the overall ranking quality considering both
    relevance and position. I track Recall@4 most carefully because if the
    relevant info isn't in the context, the LLM can't possibly answer correctly —
    no amount of generation tuning can fix bad retrieval."
"""


# =================================================================================
# SECTION 4: GENERATION EVALUATION METRICS
# =================================================================================
"""
After retrieval, the LLM generates an answer. We need to evaluate the GENERATION.
Even with perfect docs, the LLM can fail in several ways:

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

METRIC 1: FAITHFULNESS (Most Important)

    "Are all claims in the answer SUPPORTED by the retrieved context?"

    HOW IT WORKS:
        1. Decompose the answer into individual CLAIMS (atomic statements)
        2. For each claim, check: "Is this supported by the context?"
        3. Faithfulness = (supported claims) / (total claims)

    Example:
        Question: "What's our refund policy?"
        Context: "Customers can return products within 30 days. Shipping is free."
        Answer: "Customers can return products within 30 days. Refunds take 5-7 days."

        Claims:
        - "Customers can return within 30 days" → SUPPORTED (in context) ✓
        - "Refunds take 5-7 days" → NOT IN CONTEXT (hallucination!) ✗

        Faithfulness = 1/2 = 0.5

    INTERPRETATION:
        Faithfulness = 1.0 → every word in answer is grounded
        Faithfulness < 0.9 → hallucination problem

    THIS DETECTS HALLUCINATION DIRECTLY.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

METRIC 2: ANSWER RELEVANCE

    "Does the answer ACTUALLY ANSWER the question?"

    HOW IT WORKS:
        1. From the answer, generate SYNTHETIC questions (what question would this answer fit?)
        2. Compare those questions to the original question
        3. High similarity = high relevance

    Example:
        Question: "What's our refund policy?"
        Answer: "Our company was founded in 2010 and has 500 employees."
        → Faithful (if context says this)
        → BUT not relevant to the question!
        Answer Relevance = LOW

    INTERPRETATION:
        Catches off-topic answers, even if they're factually correct.
        Common with poor prompt engineering.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

METRIC 3: ANSWER CORRECTNESS (vs Ground Truth)

    "Is the answer CORRECT compared to the expected answer?"

    Requires GROUND TRUTH (what the right answer should be).

    HOW IT WORKS:
        Compares semantic similarity + factual overlap with ground truth.
        Two parts:
        a) Semantic similarity (do they MEAN the same thing?)
        b) Factual overlap (do the same facts appear?)

    Example:
        Question: "What's our refund policy?"
        Ground Truth: "30-day refund for unused items"
        Answer: "Customers can return unused products within 30 days for refund"
        Answer Correctness = HIGH (different words, same meaning + facts)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

THE THREE-WAY DIAGNOSTIC:

    Faithfulness LOW + Answer Relevance HIGH:
        → LLM answered the question but made up facts (hallucination)
        → Fix: stronger prompt, better model

    Faithfulness HIGH + Answer Relevance LOW:
        → LLM stuck to the docs but didn't answer the actual question
        → Fix: prompt engineering, query rewriting

    Faithfulness LOW + Answer Relevance LOW:
        → Total mess. Probably wrong context retrieved.
        → Fix: improve retrieval first

    Faithfulness HIGH + Answer Relevance HIGH + Correctness LOW:
        → The retrieved context didn't have the right answer
        → Fix: improve retrieval (Recall@k)

INTERVIEW ANSWER:
    "I evaluate generation with three metrics. Faithfulness measures if every
    claim in the answer is supported by the context — this directly detects
    hallucination. Answer Relevance measures if the answer addresses the actual
    question — catches off-topic answers. Answer Correctness compares to ground
    truth for factual accuracy. Together they form a diagnostic — if Faithfulness
    is high but Relevance is low, the LLM is being too literal. If Faithfulness
    is low, it's hallucinating. This separation tells me exactly what to fix."
"""


# =================================================================================
# SECTION 5: RAGAS FRAMEWORK — The 6 Core Metrics in Depth
# =================================================================================
"""
RAGAS = THE INDUSTRY STANDARD for RAG evaluation.
Open-source Python framework. Used by OpenAI, Anthropic, IBM, and most enterprises.

When the interviewer asks "How do you evaluate RAG?" — the EXPECTED answer is RAGAS.

GitHub: https://github.com/explodinggradients/ragas
Install: pip install ragas

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

RAGAS HAS 6 CORE METRICS organized into THREE CATEGORIES:

    CATEGORY 1: GENERATION QUALITY (need: question, answer, context)
        1. Faithfulness — Is answer grounded in context?
        2. Answer Relevance — Does answer address the question?

    CATEGORY 2: RETRIEVAL QUALITY (need: question, context, ground_truth)
        3. Context Precision — Are retrieved docs ranked correctly?
        4. Context Recall — Did we retrieve all relevant info?
        5. Context Relevance — Are retrieved docs relevant to question?

    CATEGORY 3: END-TO-END (need: question, answer, ground_truth)
        6. Answer Correctness — Is final answer correct?

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

METRIC 1: FAITHFULNESS

    What it measures: Hallucination
    Range: 0 to 1 (higher is better)
    Requires: question, answer, context

    HOW RAGAS COMPUTES IT:
        1. Use LLM to extract atomic claims from the answer
        2. For each claim, ask LLM: "Can this be inferred from the context?"
        3. Faithfulness = supported_claims / total_claims

    Code:
        from ragas.metrics import faithfulness
        from ragas import evaluate

        result = evaluate(dataset, metrics=[faithfulness])
        print(result)  # {"faithfulness": 0.85}

    Production threshold: > 0.90

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

METRIC 2: ANSWER RELEVANCE

    What it measures: Does the answer address the question?
    Range: 0 to 1 (higher is better)
    Requires: question, answer

    HOW RAGAS COMPUTES IT:
        1. From the answer, generate N synthetic questions (e.g., 3)
           "What question would this answer fit?"
        2. Embed the original question and the synthetic questions
        3. Average cosine similarity = answer relevance

    Logic: If the answer truly addresses the question, the synthetic
    questions generated FROM the answer should be similar to the original.

    Code:
        from ragas.metrics import answer_relevancy
        result = evaluate(dataset, metrics=[answer_relevancy])

    Production threshold: > 0.85

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

METRIC 3: CONTEXT PRECISION

    What it measures: Are the retrieved chunks RANKED correctly?
    Range: 0 to 1 (higher is better)
    Requires: question, context (chunks list), ground_truth

    HOW RAGAS COMPUTES IT:
        For each chunk in retrieved context (in order):
            Ask LLM: "Is this chunk useful for answering the question?"
            (with knowledge of the ground_truth answer)
        Higher precision = relevant chunks appear FIRST.

    Logic: If the retriever returned [irrelevant, relevant, relevant], that's
    worse than [relevant, relevant, irrelevant] — even though both have 2/3
    relevant. Context Precision penalizes wrong ordering.

    Production threshold: > 0.80

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

METRIC 4: CONTEXT RECALL

    What it measures: Did we retrieve ALL the info needed?
    Range: 0 to 1 (higher is better)
    Requires: question, context, ground_truth

    HOW RAGAS COMPUTES IT:
        1. Decompose the GROUND TRUTH into claims
        2. For each ground truth claim, check: "Is this claim
           supported by the retrieved context?"
        3. Context Recall = (supported claims) / (total ground truth claims)

    Logic: If ground truth has 5 facts and only 3 are in retrieved context,
    Recall = 3/5 = 0.6. We MISSED 2 important facts.

    Production threshold: > 0.85

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

METRIC 5: CONTEXT RELEVANCE (newer metric)

    What it measures: Are retrieved chunks relevant to the question?
    Range: 0 to 1 (higher is better)
    Requires: question, context

    HOW IT WORKS:
        For each chunk, LLM extracts only sentences relevant to the question.
        Score = relevant_sentences / total_sentences

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

METRIC 6: ANSWER CORRECTNESS

    What it measures: Is the answer correct vs ground truth?
    Range: 0 to 1 (higher is better)
    Requires: question, answer, ground_truth

    Two components:
    a) Semantic similarity (cosine similarity of answer + ground_truth embeddings)
    b) Factual overlap (LLM checks: do they agree on facts?)

    Final score = weighted average of both.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

COMPLETE RAGAS EXAMPLE:

    from ragas import evaluate
    from ragas.metrics import (
        faithfulness, answer_relevancy,
        context_precision, context_recall, answer_correctness
    )
    from datasets import Dataset

    # Build evaluation dataset
    data = {
        "question": ["What is our refund policy?"],
        "answer": ["Customers can return within 30 days for a refund."],
        "contexts": [["Customers can return products within 30 days. Shipping is free."]],
        "ground_truth": ["30-day refund for unused items"]
    }
    dataset = Dataset.from_dict(data)

    # Run evaluation
    result = evaluate(
        dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
            answer_correctness,
        ]
    )

    print(result)
    # Output:
    # {
    #   "faithfulness": 1.0,
    #   "answer_relevancy": 0.92,
    #   "context_precision": 0.85,
    #   "context_recall": 1.0,
    #   "answer_correctness": 0.88
    # }

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

INTERPRETATION CHEATSHEET:

    SCENARIO                                 LIKELY CAUSE
    Low Faithfulness                         LLM hallucinating, weak prompt
    Low Answer Relevance                     LLM going off-topic
    Low Context Precision                    Retriever returning noise (wrong order)
    Low Context Recall                       Retriever missing relevant docs
    Low Answer Correctness, others HIGH      Ground truth and context disagree
    All metrics LOW                          Total system failure (start over)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PRODUCTION TARGETS (for senior interviews):

    Faithfulness:        > 0.90 (no hallucination tolerance)
    Answer Relevance:    > 0.85
    Context Precision:   > 0.80
    Context Recall:      > 0.85
    Answer Correctness:  > 0.80

INTERVIEW ANSWER:
    "I use RAGAS framework with 6 core metrics organized in three categories.
    For generation: Faithfulness (no hallucination — every claim must be in
    context) and Answer Relevance (does it address the question). For retrieval:
    Context Precision (correct ranking), Context Recall (capture all relevant info),
    Context Relevance (only relevant chunks). For end-to-end: Answer Correctness
    against ground truth. RAGAS uses LLM-as-judge for these — it decomposes
    answers into claims and verifies each. Production targets: Faithfulness > 0.90,
    Recall > 0.85. If Faithfulness drops below 0.90, I know hallucination is the
    issue and fix the prompt or model."
"""


# =================================================================================
# SECTION 6: BUILDING A GOLD-STANDARD EVALUATION DATASET
# =================================================================================
"""
Without a labeled eval dataset, you CAN'T measure anything reliably.
This is the most underrated part of RAG evaluation.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

WHAT IS A GOLD-STANDARD DATASET?

    A curated collection of (question, expected_answer, expected_documents)
    triples that represent what the system SHOULD do.

    For each question, you know:
    - The CORRECT answer (ground truth)
    - The DOCUMENTS that should be retrieved (relevance labels)

    This is your "test set" — you run RAG on these and measure how well it does.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

HOW TO BUILD ONE (THREE METHODS):

    METHOD 1: HUMAN-CURATED (best quality, slowest)
        - Domain experts write 50-100 representative questions
        - For each, write the ideal answer
        - Mark which document(s) contain the answer
        - Time: 1-2 weeks for a small team
        - Quality: HIGH (gold standard)

    METHOD 2: LLM-GENERATED (faster, requires validation)
        - Use an LLM to generate questions from each document
        - "Read this doc, generate 3 questions a user might ask"
        - LLM also generates the expected answer
        - Human validates a sample (10-20%)
        - Time: 1-2 days
        - Quality: MEDIUM (good for prototyping)

    METHOD 3: USER-LOG-BASED (most realistic)
        - Mine real user queries from production logs
        - Manually label correct answers
        - Best for production drift detection
        - Quality: HIGH (matches real usage)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

DATASET STRUCTURE (Industry Standard):

    [
        {
            "id": "q_001",
            "question": "What is our refund policy?",
            "expected_answer": "Customers can return unused products within 30 days for a full refund.",
            "expected_doc_ids": ["doc_5", "doc_12"],
            "category": "policy",
            "difficulty": "easy",
            "needs_multi_hop": false
        },
        {
            "id": "q_002",
            "question": "How does the refund policy compare with our shipping policy?",
            "expected_answer": "Refunds: 30 days. Shipping: free over $50.",
            "expected_doc_ids": ["doc_5", "doc_12", "doc_8"],
            "category": "policy",
            "difficulty": "medium",
            "needs_multi_hop": true
        },
        ...
    ]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CATEGORIES TO INCLUDE (for thorough testing):

    1. SIMPLE FACTUAL ("What is X?") — most common
    2. COMPARISON ("How does X compare to Y?") — multi-doc
    3. NUMERICAL ("How many X?") — exact answers needed
    4. LIST ("List all X") — completeness matters
    5. REASONING ("Why does X?") — needs multi-hop
    6. EDGE CASES (questions WITHOUT answers in docs) — should say "I don't know"
    7. ADVERSARIAL (prompt injection attempts) — should refuse

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

DATASET SIZE GUIDELINES:

    DEV/PROTOTYPE:    10-30 examples (catch obvious bugs)
    BETA/STAGING:     100-200 examples (statistical confidence)
    PRODUCTION:       500-2000 examples (covers all categories)
    LARGE-SCALE:      5000+ examples (regression testing)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

THE GOLDEN RULE:

    You can't IMPROVE what you can't MEASURE.
    Without a labeled eval set, every "fix" is just a guess.
    With one, you can prove improvements rigorously.

INTERVIEW ANSWER:
    "I build a gold-standard eval dataset of 100-200 examples covering simple
    factual, comparison, list, reasoning, edge cases, and adversarial categories.
    For prototyping, I use LLM-generated questions from documents with human
    validation. For production, I mine real user queries with manually-labeled
    answers — this catches drift and matches real usage patterns. Each entry
    has question, expected answer, expected doc IDs, category, and difficulty.
    Without this, every metric improvement is just a guess."
"""


# =================================================================================
# SECTION 7: EVALUATION PIPELINE — From Dev to Production
# =================================================================================
"""
HOW TO INTEGRATE EVALUATION INTO YOUR DEVELOPMENT WORKFLOW:

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

THE 4-STAGE EVALUATION PIPELINE:

    ┌─────────────────────────────────────────────────────────────────┐
    │  STAGE 1: DEV-TIME EVAL (during development)                     │
    │  Run on every commit / PR                                        │
    │  - Small eval set (50 examples)                                  │
    │  - Fast metrics (Precision@k, basic faithfulness)                │
    │  - Goal: catch regressions FAST                                  │
    └──────────────────────────┬──────────────────────────────────────┘
                               ↓
    ┌─────────────────────────────────────────────────────────────────┐
    │  STAGE 2: STAGING EVAL (before production deployment)             │
    │  Run before each release                                         │
    │  - Full eval set (500 examples)                                  │
    │  - All RAGAS metrics                                              │
    │  - Multi-category breakdown                                       │
    │  - Goal: prove quality > thresholds                              │
    └──────────────────────────┬──────────────────────────────────────┘
                               ↓
    ┌─────────────────────────────────────────────────────────────────┐
    │  STAGE 3: PRODUCTION SAMPLING (live system)                       │
    │  Sample 1-5% of real user queries                                 │
    │  - Run async (don't block users)                                  │
    │  - Compare with daily/weekly trends                               │
    │  - Goal: detect drift, regressions in real usage                 │
    └──────────────────────────┬──────────────────────────────────────┘
                               ↓
    ┌─────────────────────────────────────────────────────────────────┐
    │  STAGE 4: USER FEEDBACK LOOP                                      │
    │  Thumbs up/down on every response                                │
    │  Mine "down-voted" responses for analysis                         │
    │  - Add new failure cases to eval set                              │
    │  - Goal: continuous improvement                                   │
    └─────────────────────────────────────────────────────────────────┘

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CI/CD INTEGRATION:

    # .github/workflows/rag_eval.yml
    name: RAG Evaluation

    on:
      pull_request:
        paths:
          - "src/rag/**"
          - "src/prompts/**"

    jobs:
      evaluate:
        steps:
          - name: Run RAG evaluation
            run: |
              python evaluate.py --dataset eval_set.json
          - name: Check thresholds
            run: |
              python check_thresholds.py \
                --faithfulness 0.90 \
                --recall 0.85 \
                --answer_correctness 0.80
          - name: Fail if regression
            run: |
              if metrics_dropped > 5%:
                exit 1

    This blocks PRs that drop quality. Forces improvements only.

INTERVIEW ANSWER:
    "I have a 4-stage evaluation pipeline. Dev-time: 50 examples on every commit
    via CI/CD — fails the build if metrics drop. Staging: 500 examples with full
    RAGAS metrics before deployment. Production: 1-5% sampling of real queries
    asynchronously — detects drift. User feedback: thumbs up/down on every
    response, mine down-votes to grow the eval set. This makes evaluation
    continuous, not a one-time activity."
"""


# =================================================================================
# SECTION 8: LLM-AS-JUDGE PATTERN (How RAGAS Actually Works)
# =================================================================================
"""
A KEY ARCHITECTURAL INSIGHT: Most modern RAG metrics use an LLM to evaluate
another LLM's output. This is called "LLM-as-Judge."

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

THE PATTERN:

    Generation LLM: produces the answer (the "student")
    Judge LLM: evaluates the answer (the "teacher")

    The Judge LLM is given a prompt like:
        "Given this question: {q}
         And this context: {c}
         Is this answer faithful to the context?
         Score 0-1 with reasoning."

    The Judge LLM returns a score and a justification.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

WHY USE LLM-AS-JUDGE?

    1. NO TEMPLATE NEEDED: Works for free-form text answers (no exact match)
    2. SEMANTIC UNDERSTANDING: Knows that "30 days" and "thirty days" are equal
    3. SCALABLE: Can evaluate thousands of examples (humans can't)
    4. MULTI-DIMENSIONAL: Can score multiple aspects (faithfulness, fluency, etc.)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CHALLENGES:

    1. JUDGE BIAS: Judge LLM may have its own biases (favors verbose answers, etc.)
    2. POSITION BIAS: Judge favors the first option in pairwise comparisons
    3. SELF-PREFERENCE: GPT-4 judges GPT-4 answers more favorably
    4. COST: Each evaluation = LLM call ($$$)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

BEST PRACTICES FOR LLM-AS-JUDGE:

    1. Use a STRONGER model as judge (e.g., GPT-4 to judge GPT-3.5)
    2. CHAIN-OF-THOUGHT: ask judge to reason BEFORE scoring
    3. STRUCTURED OUTPUT: force JSON output with specific fields
    4. CALIBRATE: validate judge against human ratings on a sample
    5. AGGREGATE: run multiple judges and average

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

EXAMPLE LLM-AS-JUDGE PROMPT:

    [Role]
    You are an expert evaluator of AI-generated answers.

    [Inputs]
    Question: {question}
    Context: {context}
    Answer: {answer}

    [Task]
    Determine if the answer is FAITHFUL to the context.
    "Faithful" means: every claim in the answer must be supportable by the context.

    Step 1: List each factual claim in the answer.
    Step 2: For each claim, check if the context supports it.
    Step 3: Compute faithfulness = (supported claims) / (total claims)

    Respond ONLY in this JSON format:
    {
        "claims": [{"claim": "...", "supported": true/false}],
        "faithfulness_score": 0.0-1.0,
        "reasoning": "brief explanation"
    }

INTERVIEW ANSWER:
    "RAGAS metrics use the LLM-as-Judge pattern — a stronger LLM evaluates the
    answers from the generation LLM. The judge gets a structured prompt with
    chain-of-thought reasoning, returns JSON output, and produces both a score
    and justification. To avoid bias, I use a different model class as judge
    (GPT-4 for GPT-3.5 outputs) and calibrate against human ratings on a sample.
    LLM-as-judge is what makes modern RAG evaluation scalable — humans can't
    review thousands of examples, but an LLM can."
"""


# =================================================================================
# SECTION 9: TOOLS — RAGAS, LangSmith, DeepEval, TruLens (Compared)
# =================================================================================
"""
THE 4 MAJOR RAG EVALUATION TOOLS in 2026:

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

TOOL 1: RAGAS (Most Popular, Open-Source)

    What: Framework focused specifically on RAG evaluation
    Strengths:
    - 6 production-ready metrics
    - Easy to integrate (a few lines of code)
    - Works with any LLM
    - Free, open-source
    Weaknesses:
    - Less integration with deployment platforms
    - Not designed for tracing or live monitoring

    Code:
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy
        result = evaluate(dataset, metrics=[faithfulness, answer_relevancy])

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

TOOL 2: LangSmith (Best for LangChain/LangGraph)

    What: LangChain's full LLMOps platform — tracing + evaluation + monitoring
    Strengths:
    - End-to-end visibility into LangChain/LangGraph apps
    - Production tracing (every LLM call logged)
    - Custom evaluators
    - Cost and latency tracking
    Weaknesses:
    - Best with LangChain ecosystem (not framework-agnostic)
    - Paid for production usage

    Code:
        from langsmith import Client
        from langsmith.evaluation import evaluate

        client = Client()
        result = evaluate(
            lambda inputs: rag_chain.invoke(inputs),
            data="my_eval_dataset",
            evaluators=[faithfulness_evaluator]
        )

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

TOOL 3: DeepEval (Most Comprehensive)

    What: Like Pytest but for LLM applications
    Strengths:
    - 14+ metrics (more than RAGAS)
    - Test-driven (run as unit tests)
    - Synthetic dataset generation
    - Open-source
    Weaknesses:
    - Steeper learning curve
    - Less RAG-focused (general LLM eval)

    Code:
        from deepeval import assert_test
        from deepeval.metrics import HallucinationMetric

        def test_rag():
            test_case = LLMTestCase(input="...", actual_output="...", retrieval_context=["..."])
            metric = HallucinationMetric(threshold=0.5)
            assert_test(test_case, [metric])

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

TOOL 4: TruLens (Best for Production Observability)

    What: Open-source tool for evaluation + tracking + drift detection
    Strengths:
    - Production-grade observability
    - Custom feedback functions
    - Dashboards and visualizations
    - Free, open-source
    Weaknesses:
    - More setup overhead
    - Smaller community than RAGAS/LangSmith

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

WHICH TO CHOOSE?

    USE CASE                              RECOMMENDED
    Quick RAG evaluation in Python        RAGAS
    Built with LangChain / LangGraph       LangSmith
    Need test-driven LLM evaluation       DeepEval
    Need production observability         TruLens
    Just starting out                     RAGAS (easiest entry)
    Enterprise production                 LangSmith + RAGAS

INTERVIEW ANSWER:
    "I use RAGAS for evaluation because it's open-source, has the right metrics
    for RAG (faithfulness, answer relevance, context precision/recall), and
    integrates with any LLM. For production observability with LangChain apps,
    I'd add LangSmith for tracing and live monitoring. For test-driven LLM
    development with unit tests, DeepEval. The choice depends on the stack —
    but RAGAS is the universal starting point for any RAG evaluation."
"""


# =================================================================================
# SECTION 10: CONTINUOUS EVALUATION IN PRODUCTION (Drift Detection)
# =================================================================================
"""
A RAG system isn't done after deployment. It MUST be continuously evaluated
because real-world conditions change.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

WHAT IS DRIFT?

    The phenomenon where your system's quality DEGRADES over time, even
    without code changes. Causes:

    1. DATA DRIFT — your knowledge base content changes
       Example: docs get updated, new docs added, old docs removed
       Effect: retrieval quality changes

    2. QUERY DRIFT — user behavior changes
       Example: new product launches → questions about it dominate
       Effect: previously-rare query types become common

    3. MODEL DRIFT — LLM updates change behavior
       Example: provider updates GPT-4 → answers shift
       Effect: faithfulness scores can drop

    4. EMBEDDING DRIFT — embedding model changes
       Example: HuggingFace pushes new version
       Effect: similarity scores change

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

HOW TO DETECT DRIFT:

    1. CONTINUOUS METRIC TRACKING
       Run RAGAS on 100 sampled queries DAILY.
       Plot: faithfulness, recall, latency over time.
       Alert if metric drops > 10% week-over-week.

    2. ANOMALY DETECTION
       Use statistical tests (e.g., KS test) to detect distribution shifts.
       New query types appearing? → potential drift.

    3. USER FEEDBACK MONITORING
       Sudden spike in negative feedback (👎) = drift signal.
       Mine the down-voted queries for patterns.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PRODUCTION EVAL DASHBOARD (What to track):

    Daily/Weekly metrics:
    - Faithfulness (target: > 0.90)
    - Answer Relevance (target: > 0.85)
    - Context Recall (target: > 0.85)
    - User satisfaction (👍 / total)
    - p95 latency (target: < 5 seconds)
    - Cost per query

    Alerts:
    - Faithfulness drops 5% in 24 hours
    - New query patterns appearing
    - Latency p95 > threshold
    - Cost spike anomaly

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

REMEDIATION WHEN DRIFT IS DETECTED:

    Quality drop?
    → Rerun on golden dataset to confirm
    → Diagnose: retrieval issue or generation issue?
    → Roll back recent changes
    → Update prompts or retraining

    New query patterns?
    → Add to eval set
    → Update retrieval (maybe add new docs)
    → Retest

    Latency spike?
    → Check infrastructure (DB, LLM API)
    → Optimize or scale up

INTERVIEW ANSWER:
    "Production RAG isn't 'set and forget' — it requires continuous evaluation.
    I track key metrics daily on sampled production queries, detect drift via
    week-over-week trend monitoring with statistical tests. User feedback
    (thumbs up/down) is a leading indicator. When drift is detected, I diagnose
    whether it's data drift (knowledge base changes), query drift (user behavior
    shift), or model drift (LLM provider updates) — each requires different
    remediation."
"""


# =================================================================================
# SECTION 11: COMMON EVALUATION PITFALLS AND HOW TO AVOID THEM
# =================================================================================
"""
THESE ARE THE MISTAKES THAT TRIP UP MOST CANDIDATES:

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PITFALL 1: TESTING ON THE SAME DATA YOU TUNED ON
    Mistake: Use the same docs for tuning AND testing.
    Result: Inflated metrics that don't generalize.
    Fix: Hold out 20% of dataset as TEST set, never used for tuning.

PITFALL 2: ONLY TESTING THE HAPPY PATH
    Mistake: All test queries are "How does X work?"
    Result: System fails on edge cases (questions WITHOUT answers).
    Fix: Include negative test cases (queries the system should refuse).

PITFALL 3: NO GROUND TRUTH ANSWERS
    Mistake: Just check "does the answer look right?"
    Result: No reliable metrics. Subjective.
    Fix: Build labeled eval dataset with ground truth.

PITFALL 4: MEASURING ONLY END-TO-END
    Mistake: Only check final answer quality.
    Result: Can't diagnose WHERE failures happen.
    Fix: Component-level metrics (retrieval + generation separately).

PITFALL 5: TRUSTING ONLY ONE METRIC
    Mistake: Optimize for Faithfulness alone.
    Result: System refuses to answer anything (perfectly faithful but useless).
    Fix: Track multiple metrics in balance (Faithfulness + Relevance + Recall).

PITFALL 6: EVALUATING ONLY ONCE
    Mistake: Evaluate at launch, never again.
    Result: Quality degrades silently (drift).
    Fix: Continuous evaluation in production.

PITFALL 7: USING EXACT MATCH METRICS
    Mistake: BLEU, ROUGE, exact string match.
    Result: Penalizes semantically correct but worded differently answers.
    Fix: Use LLM-as-Judge metrics (faithfulness, answer relevance).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

THE SENIOR ENGINEER'S CHECKLIST:

    ✅ I have a labeled eval dataset (not just informal testing)
    ✅ I evaluate retrieval and generation separately
    ✅ I use RAGAS or equivalent (no exact-match metrics)
    ✅ I include edge cases and adversarial tests
    ✅ I have CI/CD integration (block bad commits)
    ✅ I monitor production continuously (drift detection)
    ✅ I track multiple metrics (no single metric optimization)
    ✅ I have user feedback loop (👍/👎)
    ✅ I version my eval dataset (track changes over time)

INTERVIEW ANSWER:
    "Common evaluation pitfalls: testing on tuning data (always have a held-out
    test set), only testing happy paths (need negative + adversarial cases),
    using exact-match metrics (use LLM-as-judge instead), measuring only
    end-to-end (component metrics diagnose failures better), and one-time
    evaluation (need continuous in production). I avoid these with a 9-point
    checklist: labeled dataset, separate component metrics, RAGAS-style metrics,
    CI/CD integration, drift detection, multi-metric tracking, user feedback,
    and dataset versioning."
"""


# =================================================================================
# SECTION 12: 25+ INTERVIEW Q&A
# =================================================================================
"""
TIER 1 — MUST NAIL (these are the questions you couldn't answer in PwC):

Q1: "How do you evaluate RAG?"
A: "I evaluate at two levels — component-level for diagnostics (retrieval +
   generation separately) and end-to-end for user-facing quality. I use RAGAS
   framework with 6 metrics: Faithfulness and Answer Relevance for generation,
   Context Precision/Recall/Relevance for retrieval, Answer Correctness against
   ground truth for end-to-end. I track these on a labeled eval dataset."

Q2: "What is RAGAS?"
A: "RAGAS is the industry-standard open-source framework for RAG evaluation.
   It uses LLM-as-Judge to evaluate answers without needing exact-match metrics.
   Six core metrics across retrieval, generation, and end-to-end quality."

Q3: "What is Faithfulness?"
A: "It measures hallucination — for every claim in the answer, RAGAS checks
   if the context supports it. Faithfulness = supported_claims / total_claims.
   Scores below 0.90 indicate hallucination problems. Production target > 0.90."

Q4: "Difference between Context Precision and Context Recall?"
A: "Precision asks: are retrieved docs relevant and ranked correctly? (low noise)
   Recall asks: did we retrieve all the relevant info? (no missing facts).
   They trade off — increase k for higher recall, lower precision. For critical
   info I prioritize recall; for clean context I prioritize precision."

Q5: "Why can't you use traditional accuracy metrics for RAG?"
A: "RAG produces free-form text. There's no exact match — the same idea can
   be expressed many ways. Traditional metrics like BLEU or exact-match would
   penalize correct answers worded differently. That's why we need semantic
   metrics like Faithfulness and Answer Relevance, computed via LLM-as-Judge."

Q6: "How do you build an eval dataset?"
A: "100-200 examples covering simple factual, comparison, list, reasoning,
   edge cases, and adversarial categories. For prototype, LLM-generates
   questions from documents, validated by humans. For production, mine real
   user queries with manual ground truth labeling. Each entry has question,
   expected_answer, expected_doc_ids, category, difficulty."

Q7: "What is LLM-as-Judge?"
A: "Pattern where a stronger LLM evaluates outputs of another LLM. Judge gets
   a chain-of-thought prompt and returns structured JSON output with score and
   reasoning. Used by RAGAS to compute faithfulness, relevance, etc. To avoid
   bias, I use a different model class as judge and calibrate with humans."

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

TIER 2 — SENIOR-LEVEL QUESTIONS:

Q8: "How would you debug a RAG system showing 60% answer correctness?"
A: "I'd run component evaluations to localize the failure. First check
   Recall@4 — if < 80%, retrieval is missing relevant docs. If retrieval is
   good, check Faithfulness — if < 90%, LLM is hallucinating. If both are
   good but answer is irrelevant, check Answer Relevance for prompt issues."

Q9: "What's drift in RAG and how do you detect it?"
A: "Drift = quality degradation over time without code changes. Causes: data
   drift (docs change), query drift (user behavior changes), model drift (LLM
   provider updates), embedding drift (model versions). I detect via daily
   metric tracking with week-over-week comparison, statistical tests for
   distribution shifts, and user feedback monitoring."

Q10: "How do you integrate RAG evaluation into CI/CD?"
A: "On every PR, run a small eval set (50 examples) with thresholds. Fail
   the build if Faithfulness drops > 5% or Recall drops > 5%. On staging,
   run full eval (500 examples) before deployment. In production, sample
   1-5% of queries asynchronously for live monitoring."

Q11: "Why use LLM-as-Judge instead of human evaluators?"
A: "Scalability and cost. Humans can review maybe 100 examples/day per person.
   LLM-as-Judge can do 10,000+/hour. Costs cents per evaluation vs dollars per
   human hour. Calibration: validate LLM-as-Judge against humans on a sample,
   confirm correlation > 0.85, then trust at scale."

Q12: "How do you balance Faithfulness and Answer Relevance?"
A: "Track both. If you only optimize Faithfulness, the LLM becomes overly
   conservative — refuses to answer or only restates context. If you only
   optimize Answer Relevance, the LLM hallucinates to seem helpful. The
   sweet spot is Faithfulness > 0.90 AND Answer Relevance > 0.85, monitored
   together."

Q13: "What's the difference between Answer Correctness and Faithfulness?"
A: "Faithfulness checks if answer is grounded in CONTEXT. Answer Correctness
   checks if answer matches GROUND TRUTH. An answer can be Faithful (grounded
   in retrieved docs) but Incorrect (the docs themselves are wrong/outdated)."

Q14: "How would you evaluate a RAG system at scale (1M queries/day)?"
A: "Sampling-based: 1-5% of queries get evaluated asynchronously, doesn't block
   responses. Cheap LLM as judge for cost control. Alerting on metric anomalies.
   Daily aggregation for trend analysis. Weekly deep dive on failure cases."

Q15: "What metric is most important — Faithfulness or Answer Correctness?"
A: "Depends on use case. For policy/legal/medical: Faithfulness > Correctness
   (better to say nothing than be wrong). For Q&A bots with verified knowledge:
   Correctness > Faithfulness (need accurate facts). Both should be > 0.85."

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

TIER 3 — ARCHITECT-LEVEL:

Q16: "How would you design eval for a multi-agent RAG system?"
A: "Evaluate at THREE levels: (1) per-tool quality (each tool's accuracy),
   (2) routing quality (did agent pick the right tool?), (3) end-to-end
   answer quality. For tool routing, label which tool SHOULD be called for
   each query and measure routing accuracy."

Q17: "Limitations of LLM-as-Judge?"
A: "Position bias (judge favors first option), self-preference (GPT-4 favors
   GPT-4 answers), verbosity bias (longer answers seem better), and cost.
   Mitigations: randomize order, use different judge model class, use
   chain-of-thought judging, calibrate with humans periodically."

Q18: "How do you evaluate without ground truth?"
A: "Use reference-free metrics: Faithfulness (just needs context + answer)
   and Answer Relevance (just needs question + answer). For Answer Correctness
   without ground truth, use confidence scoring or ensemble of LLMs voting."

Q19: "What's a good eval dataset SIZE?"
A: "Statistical confidence: 30 examples = noisy, 100 = decent confidence,
   500 = high confidence, 2000+ = covers all categories. For each category
   (factual, comparison, edge case), need 30-50 examples minimum."

Q20: "How do you eval RAG that retrieves multiple modalities (text + images)?"
A: "Adapt metrics: for image retrieval, use image-text similarity (CLIP-based).
   For multimodal generation (image + text answers), break into modality-specific
   metrics + overall coherence."

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

TIER 4 — PRODUCTION GOTCHAS:

Q21: "What if your eval set has bias?"
A: "Eval sets reflect what you test. If only happy-path queries, system seems
   perfect but fails real edge cases. Mitigation: diversify categories,
   include adversarial cases, mine real user queries periodically."

Q22: "How do you eval cost-quality tradeoffs?"
A: "Multi-dimensional eval: track quality metrics (Faithfulness, etc.) AND
   cost per query AND latency. Different LLMs/configurations create different
   points on the Pareto frontier. Pick based on budget."

Q23: "When do you re-evaluate after model upgrades?"
A: "BEFORE rolling out: full eval suite on staging. AFTER rollout: enhanced
   monitoring for first week to catch issues. Establish quality baseline before
   any model change so you can detect regressions."

Q24: "Eval metrics that matter for chat vs Q&A?"
A: "Q&A: Faithfulness, Answer Correctness, Recall@k. Chat: also need turn
   coherence, conversation memory, context retention. Multi-turn questions
   need eval across full conversation, not isolated turns."

Q25: "How do you eval RAG when ground truth is debatable?"
A: "For subjective questions, use multiple human raters and report agreement.
   For LLM-as-Judge, use multiple judges and report consensus. Acknowledge
   subjectivity in metrics rather than pretending one answer is 'correct'."
"""


# =================================================================================
# SECTION 13: GOLDEN LESSONS
# =================================================================================
"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

GOLDEN LESSON 1: "What you can't measure, you can't improve."
    Without an eval dataset and metrics, every optimization is a guess.
    Build the eval pipeline FIRST, then iterate.

GOLDEN LESSON 2: "Evaluate retrieval and generation SEPARATELY."
    Component metrics diagnose failures. End-to-end metrics measure user impact.
    You need both. Together they tell you WHAT broke and WHY.

GOLDEN LESSON 3: "RAGAS is the industry standard. Know it cold."
    Senior interviewers expect you to name RAGAS metrics by heart:
    Faithfulness, Answer Relevance, Context Precision, Context Recall,
    Context Relevance, Answer Correctness.

GOLDEN LESSON 4: "Faithfulness > 0.90 is non-negotiable in production."
    Hallucination is the #1 risk in RAG. Without high faithfulness, your
    system is unreliable. This is the LLM equivalent of "no compiler errors."

GOLDEN LESSON 5: "LLM-as-Judge is the only way to scale evaluation."
    Humans don't scale to 1M queries. LLM-as-Judge does.
    Calibrate with humans periodically, then trust at scale.

GOLDEN LESSON 6: "Drift is real. Plan for it."
    Production RAG quality degrades over time without intervention.
    Continuous evaluation + drift detection + alerting are essential.

GOLDEN LESSON 7: "Don't optimize one metric — optimize the SYSTEM."
    Faithfulness alone = uselessly conservative system.
    Answer Relevance alone = hallucinating system.
    The sweet spot is multi-metric balance.

GOLDEN LESSON 8: "Build the eval set BEFORE the system."
    The eval set defines SUCCESS. Without it, you don't know what you're
    building toward. It's the spec, the test plan, and the user requirements
    all in one.

GOLDEN LESSON 9: "Senior engineers TRACK quality. Juniors HOPE for quality."
    The difference between a junior and senior RAG engineer:
    - Junior: "It seems to work."
    - Senior: "Faithfulness is at 0.92, recall@4 is at 0.87, p95 latency
      is 3.2s, weekly trend is stable."

GOLDEN LESSON 10: "PwC asked you this for a reason — they want SENIOR thinking."
    The question 'how do you evaluate RAG?' separates juniors from seniors.
    Now you know what 20-year veterans expect to hear. Use this depth in
    every interview.
"""

print("=" * 60)
print("RAG Evaluation Deep Dive — Complete")
print("=" * 60)
print()
print("13 Sections:")
print("  1.  Why RAG Evaluation is Hard (Two Failures)")
print("  2.  Component vs End-to-End Evaluation")
print("  3.  Retrieval Metrics (Precision@k, Recall@k, MRR, NDCG)")
print("  4.  Generation Metrics (Faithfulness, Answer Relevance, Correctness)")
print("  5.  RAGAS Framework — 6 Core Metrics in Depth")
print("  6.  Building Gold-Standard Eval Datasets")
print("  7.  Eval Pipeline (Dev → Staging → Production)")
print("  8.  LLM-as-Judge Pattern")
print("  9.  Tools Compared (RAGAS, LangSmith, DeepEval, TruLens)")
print("  10. Continuous Evaluation + Drift Detection")
print("  11. Common Pitfalls + Senior Checklist")
print("  12. 25 Interview Q&A (Tiers 1-4)")
print("  13. GOLDEN LESSONS (10 lessons)")
print()
print("This is the depth a 20+ year experienced architect would test.")
print("=" * 60)
