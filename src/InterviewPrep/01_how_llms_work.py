"""

Yes, the Eisenhower Matrix (from Stephen Covey's 7 Habits) is a GREAT answer for this. It shows structured thinking. Let me give you the correct quadrants with a real software engineering example — no need for a full lesson file, just memorize this:

THE 4 QUADRANTS (Eisenhower Matrix / Covey's Time Management):

URGENT	NOT URGENT
IMPORTANT	Q1: DO FIRST	Q2: SCHEDULE
Production bug crashing the app	Building new AI feature, code reviews, learning
NOT IMPORTANT	Q3: DELEGATE	Q4: ELIMINATE
Unnecessary meetings, some emails	Social media, random browsing
Real-world software engineering example:

"I use the Eisenhower Matrix to prioritize. Let me give a real example from my week:

Q1 (Urgent + Important) — Do immediately: Production API returning 500 errors → fix NOW, customers are affected.

Q2 (Not Urgent + Important) — Schedule and protect this time: Building the new agentic RAG feature, writing tests, code reviews, learning new frameworks. This is where REAL progress happens. I block 2-3 hours daily for deep work here.

Q3 (Urgent + Not Important) — Delegate or minimize: Attending status meetings that could be an email, responding to non-critical Slack messages. I batch these into specific time slots.

Q4 (Not Urgent + Not Important) — Eliminate: Unnecessary notifications, random browsing. I turn these off during focus time."

Your 30-second interview answer:

"I prioritize using the Eisenhower Matrix — categorizing tasks by urgency and importance. Production issues go in Q1 — fix immediately. Feature development and learning go in Q2 — I protect dedicated focus time for these daily. Meetings and non-critical requests go in Q3 — I batch them. And I eliminate Q4 distractions during work hours. In agile, I also use sprint priorities — whatever the team committed to in sprint planning gets done first, and I communicate blockers early if priorities conflict."



GenAI Interview Prep - Lesson 1: How LLMs Work Internally

THIS IS THE #1 QUESTION IN GENAI INTERVIEWS:
    "When you type a question into ChatGPT, what happens internally?"

THE 30-SECOND ANSWER (memorize this):
    "A generative AI model like ChatGPT is based on the Transformer
    architecture. When a user inputs text, it is first TOKENIZED into
    smaller pieces. These tokens are converted into EMBEDDINGS (dense
    vectors that capture meaning). The embeddings pass through multiple
    Transformer layers using SELF-ATTENTION, which lets each token look
    at every other token to understand context(decides how much 
    attentionto pay to each one). The model then predicts
    the NEXT TOKEN probabilistically and generates the response token
    by token. It was trained in three phases: pre-training on massive
    text data, fine-tuning with human examples, and RLHF where humans
    ranked answers to improve quality."

TABLE OF CONTENTS:
    Step 1: Tokenization (breaking text into pieces)
    Step 2: Embeddings (converting tokens to numbers)
    Step 3: Transformer Architecture (the brain)
    Step 4: Next-Token Prediction (how it generates text)
    Step 5: Decoding Strategies (temperature, top-k, top-p)
    Step 6: Training Phases (pretraining, fine-tuning, RLHF)
    Step 7: What LLMs CAN and CANNOT do
    Step 8: Interview Q&A (15 questions with answers)

Author: GenAI Learner
"""


# ==============================================================================
# STEP 1: TOKENIZATION - Breaking Text into Pieces
# ==============================================================================
#
# The LLM doesn't see words. It sees TOKENS.
# Tokenization breaks your input into smaller pieces the model can process.
#
#   "Hello, how are you?" -> ["Hello", ",", " how", " are", " you", "?"]
#   Each token gets a NUMBER: "Hello" -> 15339, "," -> 11, " how" -> 1268
#
# WHY NOT JUST USE WORDS?
#   Some words are rare. Subword tokenization handles ANY word by breaking
#   it into known pieces: "unhappiness" -> ["un", "happ", "iness"]
#   The model can handle words it has never seen before.
#
# ALGORITHMS:
#   BPE (Byte Pair Encoding) - Used by GPT. Merges frequent character pairs.
#   WordPiece - Used by BERT. Uses likelihood instead of frequency.
#   SentencePiece - Used by LLaMA. Works on raw text directly.
#
# TOKEN COUNT RULE OF THUMB:
#   1 token ~ 4 characters ~ 0.75 words
#   "Hello world" = 2 tokens
#   A page of text ~ 500-800 tokens
#   GPT-4 context window: ~128K tokens
#
# INTERVIEW ANSWER:
#   "Tokenization breaks input text into subword tokens using BPE.
#   Each token maps to a number in the model's vocabulary. This lets
#   the model handle any word by breaking it into known subword pieces."


# ==============================================================================
# STEP 2: EMBEDDINGS - Converting Tokens to Numbers
# ==============================================================================
#
# Each token ID is converted into a dense VECTOR (list of numbers)
# called an embedding. This vector captures the MEANING of the token.
#
#   "king"  -> [0.2, -0.5, 0.8, 0.1, ...]  (768 or 1536 dimensions)
#   "queen" -> [0.3, -0.4, 0.7, 0.2, ...]  (similar to "king"!)
#   "car"   -> [-0.8, 0.3, -0.1, 0.9, ...]  (very different)
#
# WHY VECTORS?
#   Computers can't understand words but can do math on numbers.
#   Similar words have similar vectors (cosine similarity).
#   Famous example: "king" - "man" + "woman" ~ "queen"
#
# HOW IT CONNECTS TO YOUR RAG PROJECT:
#   In DocSage, we use HuggingFace embeddings to convert document chunks
#   into vectors and store them in FAISS. When a user asks a question,
#   the question is also embedded, and we find similar document vectors.
#   SAME concept - embeddings are everywhere in GenAI.
#
# INTERVIEW ANSWER:
#   "Embeddings convert tokens into dense numerical vectors that capture
#   semantic meaning. Similar words have similar vectors. This is the
#   foundation of both LLM understanding and RAG retrieval."


# ==============================================================================
# STEP 3: TRANSFORMER ARCHITECTURE - The Brain
# ==============================================================================
#
# The Transformer is the architecture behind ALL modern LLMs
# (GPT, LLaMA, Claude, Gemini). From the 2017 paper "Attention Is All You Need."
#
# Before Transformers: RNNs processed tokens ONE AT A TIME (slow, forgets).
# Transformers: process ALL tokens IN PARALLEL (fast, remembers everything).
#
# KEY COMPONENTS:
#
# 1. SELF-ATTENTION (the most important concept)
#    Each token looks at EVERY OTHER token and decides how much attention
#    to pay to each one.
#
#    Example: "The cat sat on the mat because it was tired"
#    When processing "it", self-attention figures out "it" refers to "cat"
#    (not "mat") by looking at all other tokens.
#
#    Example:
#
#    “The bank near the river”
#
#   “bank” pays attention to “river” → meaning = river bank
#    not financial bank
#
#
#    HOW: For each token, compute three vectors:
#      Query (Q): "What am I looking for?"
#      Key (K):   "What do I contain?"
#      Value (V): "What information do I provide?"
#      Attention = softmax(Q x K^T / sqrt(d)) x V
#
#    Don't memorize the formula. Just know: Q asks, K answers, V provides info.
#
# 2. MULTI-HEAD ATTENTION
#    Instead of one attention calculation, do it N times in parallel (N "heads").
#    Each head learns different things:
#      Head 1: grammar relationships
#      Head 2: semantic meaning
#      Head 3: positional/distance patterns
#    GPT-3 has 96 heads. Results are concatenated.
#
# 3. POSITIONAL ENCODING (Order Matters!)
#    Transformers process all tokens in parallel - they don't know ORDER.
#    "Dog bites man" vs "Man bites dog" - same tokens, different meaning!
#    Positional encoding adds position info to each embedding:
#      token_input = token_embedding + position_embedding
#    This tells the model: "this token is at position 1, this at position 5."
#
# 4. FEED-FORWARD LAYERS
#    After attention, each token passes through a neural network
#    (two linear layers + activation). Adds non-linear processing.
#
# 5. LAYER STACKING
#    Many layers stacked: GPT-3 has 96 layers, GPT-4 has 120+.
#    Each layer refines understanding further.
#
# ENCODER vs DECODER:
#   ENCODER-ONLY:    BERT (understands text, classification)
#   DECODER-ONLY:    GPT, LLaMA, Claude (generates text)
#   ENCODER-DECODER:  T5, BART (translates, summarizes)
#   GPT is DECODER-ONLY. It only generates the next token.
#
# INTERVIEW ANSWER:
#   "The Transformer uses self-attention to let each token attend to every
#   other token, understanding context in parallel. It has multi-head
#   attention for multiple patterns, positional encoding for word order,
#   and stacked layers for deeper understanding. GPT is a decoder-only
#   Transformer that generates text one token at a time."


# ==============================================================================
# STEP 4: NEXT-TOKEN PREDICTION - How It Generates Text
# ==============================================================================
#
# An LLM is fundamentally a NEXT-TOKEN PREDICTOR.
# Given all previous tokens, it predicts the probability of every possible next token.
#
#   Input: "The capital of France is"
#   Output probabilities: "Paris" (92%), "Lyon" (3%), "Berlin" (1%), ...
#   Picks "Paris". Adds it. Now predicts next token after "Paris":
#   Input: "The capital of France is Paris"
#   Output: "." (85%), "," (8%), "and" (2%), ...
#
#   This continues token by token until a stop token or max_tokens limit.
#
# THIS IS WHY IT'S CALLED "AUTOREGRESSIVE":
#   Each prediction depends on ALL previous tokens (including ones it just
#   generated). It regresses on its own output.
#
# IMPORTANT: It does NOT search the internet in real time (by default).
#   It generates from patterns learned during training.
#   That's why RAG exists - to give it external knowledge at query time.
#
# INTERVIEW ANSWER:
#   "LLMs generate text autoregressively - predicting one token at a time.
#   At each step, the model outputs probabilities for all possible next
#   tokens, picks one, adds it to the sequence, and repeats. It doesn't
#   search the internet - it generates from learned patterns. That's why
#   we use RAG to provide external knowledge."


# ==============================================================================
# STEP 5: DECODING STRATEGIES - How It Chooses Words
# ==============================================================================
#
# The model doesn't always pick the highest probability token.
# Different strategies control creativity vs accuracy.
#
# 1. GREEDY DECODING
#    Always pick the highest probability token.
#    Fast but boring - always gives the same answer.
#    temperature=0 in our code does this.
#
# 2. TEMPERATURE (the most important hyperparameter)
#    Controls randomness/creativity.
#      temperature = 0.0 -> deterministic (always same answer)
#      temperature = 0.3 -> slightly creative (good for RAG - what we use)
#      temperature = 0.7 -> creative (good for stories, brainstorming)
#      temperature = 1.0 -> very random
#      temperature > 1.0 -> chaotic, incoherent
#
#    HOW IT WORKS: divides logits before softmax.
#    Lower temp -> sharper distribution (top token dominates)
#    Higher temp -> flatter distribution (more tokens compete)
#
# 3. TOP-K SAMPLING
#    Only consider the top K most probable tokens, ignore the rest.
#    top_k=50: pick from the 50 most likely tokens.
#    Prevents picking very unlikely tokens.
#
# 4. TOP-P (NUCLEUS) SAMPLING
#    Only consider tokens whose cumulative probability reaches P.
#    top_p=0.9: pick from tokens that together have 90% probability.
#    More adaptive than top-k (sometimes 10 tokens cover 90%, sometimes 100).
#    top_p (nucleus sampling) is a parameter in large language models that controls text generation by limiting the 
#    selection of next tokens to the smallest set whose cumulative probability exceeds a threshold p.
#
# 5. MAX_TOKENS
#    Hard limit on how many tokens to generate.
#    In DocSage: GENERATION_MAX_TOKENS = 1024
#
# HOW WE USE THESE IN DOCSAGE:
#   Agent LLM:     temperature=0   (deterministic tool selection)
#   Grading LLM:   temperature=0   (deterministic yes/no grading)
#   Generation LLM: temperature=0.3 (slightly creative but grounded)
#   Rewrite LLM:   temperature=0   (deterministic query rewriting)
#
# INTERVIEW ANSWER:
#   "Temperature controls creativity. 0 is deterministic, 0.7 is creative.
#   Top-k limits to the K most likely tokens. Top-p limits to tokens
#   covering P% cumulative probability. In my RAG project, I use temp=0
#   for the agent and grading (need deterministic decisions) and temp=0.3
#   for generation (slightly creative but grounded in documents)."


# ==============================================================================
# STEP 6: TRAINING PHASES - How LLMs Are Built
# ==============================================================================
#
# LLMs are trained in THREE phases:
#
# PHASE 1: PRE-TRAINING (the expensive part)
#   What: Predict the next word on MASSIVE text data (internet, books, code).
#   Data: Trillions of tokens (GPT-3: 300B tokens, GPT-4: estimated 13T).
#   Cost: Millions of dollars, thousands of GPUs, weeks/months.
#   Result: The model learns language, grammar, facts, reasoning patterns.
#   This is the "base model" - it can complete text but isn't helpful yet.
#
#   Think of it as: reading the entire internet and learning patterns.
#
# PHASE 2: FINE-TUNING (Supervised Fine-Tuning / SFT)
#   What: Train on high-quality (question, answer) pairs written by humans.
#   Data: Thousands of carefully crafted examples.
#   Result: The model learns to be HELPFUL - follow instructions, answer
#           questions, be polite, refuse harmful requests.
#   This turns the base model into an "assistant."
#
#   Think of it as: a tutor teaching the model how to be a good assistant.
#
# PHASE 3: RLHF (Reinforcement Learning from Human Feedback)
#   What: Humans RANK multiple model responses. A reward model learns
#         what humans prefer. The LLM is then trained to maximize this reward.
#   Process:
#     1. Model generates multiple answers to the same question.
#     2. Humans rank them: Answer A > Answer B > Answer C.
#     3. A reward model learns the ranking pattern.
#     4. The LLM is fine-tuned using PPO (a reinforcement learning algorithm)
#        to produce answers the reward model scores highly.
#   Result: Better, safer, more aligned responses.
#
#   Think of it as: learning from human preferences, not just examples.
#
# WHY ALL THREE PHASES?
#   Pre-training alone: model can complete text but isn't helpful.
#   + Fine-tuning: model follows instructions and answers questions.
#   + RLHF: model gives BETTER answers that humans actually prefer.
#
# INTERVIEW ANSWER:
#   "LLMs are trained in three phases. Pre-training on massive text data
#   teaches language patterns. Fine-tuning on curated Q&A pairs teaches
#   the model to be a helpful assistant. RLHF uses human rankings to
#   further improve response quality - humans rank multiple answers,
#   a reward model learns preferences, and the LLM optimizes for those
#   preferences using reinforcement learning."


# ==============================================================================
# STEP 7: WHAT LLMs CAN and CANNOT DO
# ==============================================================================
#
# CAN DO:
#   - Generate human-like text (answers, summaries, code, stories)
#   - Understand context and nuance in language
#   - Follow complex instructions
#   - Reason about problems (to some extent)
#   - Translate between languages
#   - Write and explain code
#
# CANNOT DO (important for interviews!):
#   - Search the internet in real time (needs RAG or tools)
#   - Remember previous conversations (needs memory/checkpointing)
#   - Access private/company data (needs RAG with your documents)
#   - Do precise math (needs calculator tools)
#   - Know events after training cutoff date (needs web search)
#   - Guarantee factual accuracy (can hallucinate)
#
# THIS IS WHY YOUR DOCSAGE PROJECT MATTERS:
#   Every "cannot do" above is solved by something you built:
#   - Can't search internet -> Tavily web search tool
#   - Can't remember -> MemorySaver with thread_id
#   - Can't access private data -> PDF/document RAG retrieval
#   - Can hallucinate -> Document grading + hallucination validation
#   - Can't know recent events -> Web search fallback (CRAG pattern)
#
# INTERVIEW ANSWER:
#   "LLMs are powerful text generators but they can't search the internet,
#   remember conversations, or access private data by default. That's
#   exactly why I built my RAG application - it gives the LLM access to
#   uploaded documents via retrieval, web search for current information,
#   conversation memory via checkpointing, and quality validation to
#   prevent hallucination."


# ==============================================================================
# STEP 8: THE COMPLETE PICTURE - End to End
# ==============================================================================
#
# When you type "What is Python?" into ChatGPT:
#
# 1. TOKENIZE: "What is Python?" -> [What][is][Python][?] -> [1234, 318, 11361, 30]
#
# 2. EMBED: Each token ID -> dense vector [0.2, -0.5, 0.8, ...]
#
# 3. ADD POSITION: vector + positional encoding (so model knows word order)
#
# 4. SELF-ATTENTION (x96 layers in GPT-3):
#    Each token attends to every other token.
#    "Python" attends strongly to "What" and "is" (understanding the question).
#
# 5. PREDICT NEXT TOKEN: Model outputs probabilities for ALL tokens.
#    "Python" -> "is" (85%), "was" (5%), ...
#    Picks "Python" as first output token.
#
# 6. AUTOREGRESSIVE LOOP:
#    "Python is" -> "a" (70%), "the" (10%), ...
#    "Python is a" -> "high" (60%), "popular" (15%), ...
#    "Python is a high" -> "-" (90%), ...
#    "Python is a high-" -> "level" (95%), ...
#    ... continues until stop token or max_tokens.
#
# 7. DETOKENIZE: Convert token IDs back to text.
#    Output: "Python is a high-level, general-purpose programming language..."
#
# Total time: ~1-3 seconds for a paragraph.
# Total parameters doing this: 175 BILLION (GPT-3) or 1.8 TRILLION (GPT-4).


# ==============================================================================
# INTERVIEW Q&A - 15 Questions You WILL Be Asked
# ==============================================================================
#
# Q1: How does ChatGPT generate responses?
# A: It tokenizes input, converts to embeddings, processes through Transformer
#    layers with self-attention, and predicts the next token autoregressively.
#    Each token is generated one at a time based on all previous tokens.
#
# Q2: What is self-attention?
# A: Self-attention lets each token look at every other token in the sequence
#    to understand context. It computes Query, Key, Value vectors for each
#    token. The attention score between two tokens tells the model how much
#    one token should influence the other's representation.
#
# Q3: What is the difference between encoder and decoder in Transformers?
# A: Encoder processes input and creates representations (BERT uses this for
#    understanding). Decoder generates output token by token (GPT uses this).
#    T5 uses both. GPT/LLaMA/Claude are decoder-only models.
#
# Q4: What is temperature in LLMs?
# A: Temperature controls randomness. 0 is deterministic (always same answer),
#    0.7 is creative, 1.0+ is very random. It divides logits before softmax -
#    lower temperature sharpens the distribution, higher flattens it.
#
# Q5: What is the difference between top-k and top-p?
# A: Top-k considers only the K most probable tokens. Top-p considers tokens
#    until cumulative probability reaches P. Top-p is more adaptive - it
#    adjusts the number of candidates based on the distribution shape.
#
# Q6: What is tokenization and why is it needed?
# A: Tokenization breaks text into subword pieces using BPE or similar.
#    It's needed because models work with numbers, not text. Subword
#    tokenization handles rare/unknown words by breaking them into known pieces.
#
# Q7: What are embeddings?
# A: Dense numerical vectors that capture semantic meaning. Similar words
#    have similar vectors. Used in both LLMs (token understanding) and
#    RAG (document retrieval via similarity search).
#
# Q8: What is positional encoding and why is it needed?
# A: Transformers process tokens in parallel, so they don't know word order.
#    Positional encoding adds position information to embeddings so the model
#    knows "Dog bites man" is different from "Man bites dog."
#
# Q9: What are the three training phases of an LLM?
# A: Pre-training (next-token prediction on massive data), fine-tuning
#    (supervised training on Q&A pairs), and RLHF (reinforcement learning
#    from human feedback - humans rank answers to improve quality).
#
# Q10: What is RLHF?
# A: Reinforcement Learning from Human Feedback. Humans rank multiple model
#     responses. A reward model learns these preferences. The LLM is then
#     trained to maximize the reward model's score using PPO algorithm.
#
# Q11: Why do LLMs hallucinate?
# A: LLMs predict the most probable next token based on patterns, not facts.
#     If the training data has gaps or the question is ambiguous, the model
#     generates plausible-sounding but incorrect text. RAG and grading help.
#
# Q12: What is the context window?
# A: The maximum number of tokens the model can process at once (input +
#     output combined). GPT-4: 128K tokens. LLaMA 3: 8K-128K. Longer
#     context = more information but slower and more expensive.
#
# Q13: How is RAG different from fine-tuning?
# A: RAG retrieves external knowledge at query time (dynamic, no retraining).
#     Fine-tuning bakes knowledge into model weights (static, needs retraining).
#     RAG for changing knowledge + citations. Fine-tuning for style/reasoning.
#
# Q14: What is the difference between GPT-3 and GPT-4?
# A: GPT-4 is larger (estimated 1.8T params vs 175B), has longer context
#     (128K vs 4K tokens), better reasoning, multimodal (can process images),
#     and more aligned via improved RLHF. GPT-4 is also rumored to be a
#     Mixture of Experts (MoE) architecture.
#
# Q15: Can you explain how your DocSage project uses these concepts?
# A: "DocSage uses embeddings to convert document chunks into vectors stored
#     in FAISS. When a user asks a question, the agent LLM (with bound tools)
#     decides which knowledge base to search. Retrieved documents are graded
#     for relevance using structured output. The generation LLM produces an
#     answer with temperature=0.3 for grounded creativity. Post-generation
#     validation checks for hallucination. The entire workflow is a LangGraph
#     state machine with conditional edges and conversation memory."
