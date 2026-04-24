"""
GenAI Interview Prep - Lesson 2: ML and Deep Learning Basics

WHY THIS LESSON EXISTS:
    GenAI interviewers ask ML/DL basics as a "sanity check." They don't
    expect you to code a neural network from scratch. They want to hear
    you explain concepts clearly in 2-3 sentences each. If you can do
    that, you pass. If you say "I don't know what supervised learning is,"
    it's a red flag even for a GenAI role.

    Your profile: 4 years dev (2.5 Java + 1.5 GenAI). They expect you
    to know WHAT these concepts are and HOW they connect to GenAI.
    They do NOT expect you to derive backpropagation math.

HOW TO USE THIS FILE:
    - Read each concept once to understand it.
    - Before an interview, skim the "Interview Answer" boxes.
    - Focus on the "How It Connects to GenAI" lines.

TABLE OF CONTENTS:
    PART A: Machine Learning Basics
        1. What is Machine Learning
        2. Supervised Learning
        3. Unsupervised Learning
        4. Reinforcement Learning
        5. Data Preprocessing
        6. Train/Test Split
        7. Model Evaluation Metrics
        8. Overfitting vs Underfitting
        9. Linear Regression
        10. Logistic Regression
        11. NLP Basics

    PART B: Deep Learning Basics
        12. What is a Neural Network
        13. How Neural Networks Learn (Backpropagation)
        14. Activation Functions
        15. CNN (Convolutional Neural Networks)
        16. RNN (Recurrent Neural Networks)
        17. LSTM (Long Short-Term Memory)
        18. The Attention Mechanism
        19. From RNN to Transformer (the evolution)

    PART C: Interview Q&A (20 questions)

Author: GenAI Learner
"""


# ██████████████████████████████████████████████████████████████████████████████
# PART A: MACHINE LEARNING BASICS
# ██████████████████████████████████████████████████████████████████████████████


# ==============================================================================
# 1. WHAT IS MACHINE LEARNING
# ==============================================================================
#
# WHAT: Machine Learning is a way for computers to LEARN PATTERNS from data
# instead of being explicitly programmed with rules.
#
# ANALOGY: Traditional programming = you write rules ("if email contains
# 'lottery', mark as spam"). ML = you give examples of spam/not-spam emails,
# and the computer LEARNS the rules itself.
#
# THREE TYPES:
#   Supervised Learning   -> learns from LABELED data (input + correct answer)
#   Unsupervised Learning -> finds PATTERNS in UNLABELED data (no answers)
#   Reinforcement Learning -> learns by TRIAL AND ERROR (rewards/penalties)
#
# HOW IT CONNECTS TO GENAI:
#   LLMs are trained using supervised learning (fine-tuning on Q&A pairs)
#   and reinforcement learning (RLHF). The pre-training phase is self-supervised
#   (predicting the next token is the "label").
#
# INTERVIEW ANSWER:
#   "Machine learning lets computers learn patterns from data instead of
#   being explicitly programmed. There are three types: supervised (labeled
#   data), unsupervised (finding patterns), and reinforcement (learning from
#   rewards). LLMs use all three: self-supervised pre-training, supervised
#   fine-tuning, and reinforcement learning via RLHF."


# ==============================================================================
# 2. SUPERVISED LEARNING
# ==============================================================================
#
# WHAT: Learning from LABELED data. Each training example has an input AND
# the correct output (the "label").
#
# EXAMPLES:
#   Input: Email text     -> Label: "spam" or "not spam"     (classification)
#   Input: House features -> Label: price ($350,000)          (regression)
#   Input: Image          -> Label: "cat" or "dog"            (classification)
#   Input: Question        -> Label: correct answer            (LLM fine-tuning!)
#
# TWO TYPES:
#   Classification: predict a CATEGORY (spam/not-spam, cat/dog, positive/negative)
#   Regression: predict a NUMBER (price, temperature, score)
#
# COMMON ALGORITHMS:
#   Linear Regression, Logistic Regression, Decision Trees, Random Forest,
#   SVM, Neural Networks, Gradient Boosting (XGBoost, LightGBM)
#
# HOW IT CONNECTS TO GENAI:
#   LLM fine-tuning IS supervised learning. You give the model (question, answer)
#   pairs and it learns to produce good answers. The document grading in our
#   DocSage project (binary yes/no) is also a classification task.
#
# INTERVIEW ANSWER:
#   "Supervised learning trains on labeled data where each example has an
#   input and correct output. Classification predicts categories, regression
#   predicts numbers. LLM fine-tuning is supervised learning — you train on
#   question-answer pairs so the model learns to be a helpful assistant."


# ==============================================================================
# 3. UNSUPERVISED LEARNING
# ==============================================================================
#
# WHAT: Finding PATTERNS in data WITHOUT labels. No correct answers provided.
# The algorithm discovers structure on its own.
#
# EXAMPLES:
#   Clustering: group similar customers together (K-Means)
#   Dimensionality Reduction: compress 1000 features to 50 (PCA)
#   Anomaly Detection: find unusual transactions (fraud detection)
#
# HOW IT CONNECTS TO GENAI:
#   Embeddings are a form of unsupervised learning — the model learns to
#   represent words as vectors without explicit labels. Clustering embeddings
#   can group similar documents together (useful for organizing knowledge bases).
#
# INTERVIEW ANSWER:
#   "Unsupervised learning finds patterns in unlabeled data. Common tasks
#   include clustering (grouping similar items) and dimensionality reduction.
#   In GenAI, embeddings are learned in an unsupervised way — the model
#   discovers that 'king' and 'queen' should have similar vectors without
#   being explicitly told."


# ==============================================================================
# 4. REINFORCEMENT LEARNING
# ==============================================================================
#
# WHAT: Learning by TRIAL AND ERROR. An agent takes actions in an environment
# and receives rewards or penalties. It learns to maximize rewards over time.
#
# ANALOGY: Teaching a dog tricks. Dog sits -> treat (reward). Dog jumps on
# table -> "no!" (penalty). Over time, dog learns what to do.
#
# KEY TERMS:
#   Agent: the learner (the LLM)
#   Environment: the context (the conversation)
#   Action: what the agent does (generate a response)
#   Reward: feedback signal (human ranking in RLHF)
#
# HOW IT CONNECTS TO GENAI:
#   RLHF (Reinforcement Learning from Human Feedback) is THE key technique
#   that makes ChatGPT helpful. Humans rank responses, a reward model learns
#   preferences, and the LLM is trained to maximize that reward using PPO.
#
# INTERVIEW ANSWER:
#   "Reinforcement learning trains an agent through rewards and penalties.
#   In GenAI, RLHF uses this: humans rank LLM responses, a reward model
#   learns what humans prefer, and the LLM is optimized to produce
#   higher-ranked responses. This is what makes ChatGPT helpful and safe."


# ==============================================================================
# 5. DATA PREPROCESSING
# ==============================================================================
#
# WHAT: Cleaning and preparing raw data before feeding it to a model.
# "Garbage in, garbage out" — bad data = bad model.
#
# KEY STEPS:
#   1. Handle Missing Values: fill with mean/median, or drop rows
#   2. Encoding: convert text categories to numbers
#      - Label Encoding: "red"=0, "blue"=1, "green"=2
#      - One-Hot Encoding: "red"=[1,0,0], "blue"=[0,1,0], "green"=[0,0,1]
#   3. Normalization/Scaling: bring all features to same range (0-1 or -1 to 1)
#      - MinMaxScaler: scales to [0, 1]
#      - StandardScaler: mean=0, std=1
#   4. Feature Selection: pick the most important features, drop irrelevant ones
#   5. Text Preprocessing (for NLP): lowercase, remove stopwords, tokenize
#
# HOW IT CONNECTS TO GENAI:
#   In RAG, data preprocessing = chunking + cleaning documents before embedding.
#   Our DocSage pipeline: load PDF -> split into chunks -> clean text -> embed.
#   This IS data preprocessing for GenAI.
#
# INTERVIEW ANSWER:
#   "Data preprocessing includes handling missing values, encoding categories,
#   normalizing features, and cleaning text. In my RAG project, preprocessing
#   means loading documents, chunking them with RecursiveCharacterTextSplitter,
#   and embedding them — this is the GenAI equivalent of data preprocessing."


# ==============================================================================
# 6. TRAIN/TEST SPLIT
# ==============================================================================
#
# WHAT: Splitting your data into two parts:
#   Training set (80%): model learns from this
#   Test set (20%): model is evaluated on this (never seen during training)
#
# WHY: If you test on the same data you trained on, the model just memorizes
# answers. Testing on unseen data shows if it actually LEARNED patterns.
#
# COMMON SPLITS:
#   80/20 (train/test) — simple
#   70/15/15 (train/validation/test) — with validation for tuning
#   Cross-validation: split into K folds, train K times, average results
#
# HOW IT CONNECTS TO GENAI:
#   LLM evaluation uses the same principle. RAGAS metrics test the RAG
#   pipeline on questions it hasn't seen. You don't evaluate on the same
#   questions you used to build the system.
#
# INTERVIEW ANSWER:
#   "Train/test split separates data so the model is evaluated on unseen
#   examples. Typically 80/20. This prevents memorization and tests true
#   generalization. In RAG evaluation, we use test questions the system
#   hasn't seen to measure retrieval quality and answer accuracy."


# ==============================================================================
# 7. MODEL EVALUATION METRICS
# ==============================================================================
#
# FOR CLASSIFICATION (predicting categories):
#
#   Accuracy: % of correct predictions. Simple but misleading if data is imbalanced.
#     Example: 95% emails are not spam. A model that always says "not spam"
#     has 95% accuracy but catches ZERO spam. Useless!
#
#   Precision: Of all predicted positives, how many are actually positive?
#     "When the model says spam, how often is it right?"
#     High precision = few false alarms.
#
#   Recall: Of all actual positives, how many did the model catch?
#     "Of all actual spam, how much did the model find?"
#     High recall = catches most spam (but may have false alarms).
#
#   F1 Score: Harmonic mean of precision and recall. Balances both.
#     F1 = 2 * (precision * recall) / (precision + recall)
#
#   Confusion Matrix: 2x2 table showing TP, FP, TN, FN.
#     True Positive (TP): predicted spam, actually spam
#     False Positive (FP): predicted spam, actually not spam
#     True Negative (TN): predicted not spam, actually not spam
#     False Negative (FN): predicted not spam, actually spam
#
# FOR REGRESSION (predicting numbers):
#   MAE (Mean Absolute Error): average of |actual - predicted|
#   MSE (Mean Squared Error): average of (actual - predicted)^2
#   RMSE: square root of MSE (same units as the target)
#   R-squared: how much variance the model explains (1.0 = perfect)
#
# HOW IT CONNECTS TO GENAI:
#   Our document grading is binary classification (relevant: yes/no).
#   RAG evaluation uses similar metrics: context precision, context recall,
#   faithfulness, answer relevance (from RAGAS framework).
#
# INTERVIEW ANSWER:
#   "For classification: accuracy, precision, recall, F1, confusion matrix.
#   Precision measures false alarm rate, recall measures catch rate. For
#   RAG evaluation, I use RAGAS metrics: context relevance, faithfulness,
#   and answer relevance — which are conceptually similar to precision
#   and recall applied to retrieval and generation."


# ==============================================================================
# 8. OVERFITTING vs UNDERFITTING
# ==============================================================================
#
# OVERFITTING: Model memorizes training data but fails on new data.
#   Like a student who memorizes answers but can't solve new problems.
#   Signs: high training accuracy, low test accuracy.
#   Fixes: more data, regularization, dropout, simpler model, early stopping.
#
# UNDERFITTING: Model is too simple to capture patterns.
#   Like a student who didn't study enough.
#   Signs: low training accuracy AND low test accuracy.
#   Fixes: more complex model, more features, train longer.
#
# THE SWEET SPOT: Good training accuracy AND good test accuracy.
#   The model learned real patterns, not just memorized examples.
#
# HOW IT CONNECTS TO GENAI:
#   LLM hallucination is related to overfitting on training patterns.
#   The model generates plausible text because it learned patterns, but
#   those patterns don't always match reality. RAG helps by grounding
#   the model in actual documents instead of relying on memorized patterns.
#
# INTERVIEW ANSWER:
#   "Overfitting means the model memorizes training data but fails on new
#   data. Underfitting means it's too simple to learn patterns. In GenAI,
#   hallucination is related — the model generates plausible text from
#   memorized patterns that may not be factually correct. RAG addresses
#   this by grounding responses in retrieved documents."


# ==============================================================================
# 9. LINEAR REGRESSION
# ==============================================================================
#
# WHAT: Predicts a continuous NUMBER by fitting a straight line through data.
#   y = mx + b (one feature) or y = w1*x1 + w2*x2 + ... + b (multiple features)
#
# EXAMPLE: Predict house price from square footage.
#   More sqft -> higher price (linear relationship).
#   The model finds the best line that minimizes prediction errors.
#
# KEY TERMS:
#   Weights (w): how much each feature matters
#   Bias (b): the baseline value
#   Loss function: MSE (mean squared error) — what the model minimizes
#   Gradient Descent: the algorithm that adjusts weights to reduce loss
#
# HOW IT CONNECTS TO GENAI:
#   Neural networks are built from linear regression + activation functions.
#   Each neuron computes: output = activation(w1*x1 + w2*x2 + ... + b)
#   That's linear regression with a non-linear twist.
#
# INTERVIEW ANSWER:
#   "Linear regression predicts a number by fitting a line to data using
#   y = wx + b. It minimizes MSE using gradient descent. Neural networks
#   are essentially stacked linear regressions with activation functions
#   added for non-linearity."


# ==============================================================================
# 10. LOGISTIC REGRESSION
# ==============================================================================
#
# WHAT: Predicts a CATEGORY (not a number) despite the name "regression."
# Uses a sigmoid function to output a probability between 0 and 1.
#
# EXAMPLE: Predict if an email is spam (1) or not spam (0).
#   probability = sigmoid(w1*x1 + w2*x2 + ... + b)
#   If probability > 0.5 -> spam. Else -> not spam.
#
# SIGMOID FUNCTION: squashes any number to range [0, 1].
#   sigmoid(x) = 1 / (1 + e^(-x))
#   Large positive x -> ~1.0
#   Large negative x -> ~0.0
#   x = 0 -> 0.5
#
# HOW IT CONNECTS TO GENAI:
#   The softmax function in Transformers is a multi-class version of sigmoid.
#   Instead of binary (spam/not-spam), softmax outputs probabilities across
#   ALL tokens in the vocabulary. The next-token prediction is essentially
#   logistic regression over 50,000+ classes.
#
# INTERVIEW ANSWER:
#   "Logistic regression predicts categories using a sigmoid function that
#   outputs probabilities between 0 and 1. In Transformers, the softmax
#   function is the multi-class version — it outputs probabilities across
#   the entire vocabulary for next-token prediction."


# ==============================================================================
# 11. NLP BASICS (Natural Language Processing)
# ==============================================================================
#
# WHAT: The field of AI that deals with human language.
# GenAI is a SUBSET of NLP. Everything we do is NLP.
#
# KEY NLP CONCEPTS:
#
#   Tokenization: breaking text into tokens (covered in Lesson 1)
#
#   Stopwords: common words like "the", "is", "and" that carry little meaning.
#     Traditional NLP removes them. LLMs keep them (they matter for context).
#
#   Stemming/Lemmatization: reducing words to root form.
#     "running", "runs", "ran" -> "run"
#     Traditional NLP uses this. LLMs handle it via subword tokenization(BPE-BYTE PAIR ENC).
#
#   TF-IDF: Term Frequency - Inverse Document Frequency.
#     Measures how important a word is to a document.
#     High TF-IDF = word appears often in THIS doc but rarely in others.
#     This is the "sparse" in hybrid search (BM25 is based on TF-IDF).
#
#   Word2Vec / GloVe: Early embedding models (before Transformers).
#     Learned word vectors from context. "king - man + woman = queen"
#     Replaced by Transformer-based embeddings (BERT, sentence-transformers).
#
#   Named Entity Recognition (NER): finding names, dates, locations in text.
#     "Tirupathi works at Infor in Hyderabad" -> Person: Tirupathi, Org: Infor, Location: Hyderabad
#
#   Sentiment Analysis: determining if text is positive, negative, or neutral.
#     "This product is amazing!" -> Positive
#
# HOW IT CONNECTS TO GENAI:
#   LLMs do ALL of these tasks without separate models. One LLM can tokenize,
#   understand sentiment, extract entities, and generate text. That's the power
#   of large language models — they're general-purpose NLP engines.
#
# INTERVIEW ANSWER:
#   "NLP is the field of AI dealing with human language. Key concepts include
#   tokenization, TF-IDF for keyword importance, embeddings for semantic
#   meaning, and tasks like NER and sentiment analysis. LLMs are general-purpose
#   NLP models that handle all these tasks without separate specialized models."


# ==============================================================================
# 11b. STEMMING vs LEMMATIZATION (Frequently Asked in Interviews!)
# ==============================================================================
#
# Interviewers LOVE this question because it tests whether you understand
# text preprocessing fundamentals. They are DIFFERENT concepts, not the same.
#
# STEMMING:
#   WHAT: Chops off word endings using RULES to get an approximate root.
#   HOW:  Just removes suffixes. No dictionary lookup. Fast but crude.
#
#   Examples:
#     "running"    -> "run"       (correct)
#     "runs"       -> "run"       (correct)
#     "studies"    -> "studi"     (WRONG! not a real word)
#     "university" -> "univers"   (WRONG! not a real word)
#     "better"     -> "better"    (doesn't know "good" is the root)
#     "caring"     -> "car"       (WRONG! removed too much)
#
#   ALGORITHMS:
#     Porter Stemmer: most common, English-focused, aggressive cutting
#     Snowball Stemmer: improved Porter, supports multiple languages
#     Lancaster Stemmer: most aggressive, often over-stems
#
#   CODE:
#     from nltk.stem import PorterStemmer
#     stemmer = PorterStemmer()
#     stemmer.stem("running")  # "run"
#     stemmer.stem("studies")  # "studi" (wrong but fast)
#
#   PROS: Very fast. No dictionary needed. Good for search engines.
#   CONS: Produces non-real words. Crude. Loses meaning sometimes.
#
#
# LEMMATIZATION:
#   WHAT: Reduces words to their DICTIONARY FORM (lemma) using vocabulary
#   and grammar rules. Always produces a REAL word.
#   HOW:  Looks up the word in a dictionary/vocabulary. Understands grammar.
#
#   Examples:
#     "running"    -> "run"       (correct, same as stemming)
#     "runs"       -> "run"       (correct, same as stemming)
#     "studies"    -> "study"     (CORRECT! unlike stemming's "studi")
#     "better"     -> "good"     (CORRECT! understands comparative form)
#     "caring"     -> "care"     (CORRECT! unlike stemming's "car")
#     "mice"       -> "mouse"    (CORRECT! understands irregular plurals)
#     "went"       -> "go"       (CORRECT! understands irregular verbs)
#     "are"        -> "be"       (CORRECT! understands verb conjugation)
#
#   ALGORITHMS:
#     WordNet Lemmatizer (NLTK): uses WordNet dictionary
#     spaCy Lemmatizer: uses language models, very accurate
#
#   CODE:
#     from nltk.stem import WordNetLemmatizer
#     lemmatizer = WordNetLemmatizer()
#     lemmatizer.lemmatize("studies")  # "study"
#     lemmatizer.lemmatize("better", pos="a")  # "good" (pos=adjective)
#
#   PROS: Always produces real words. More accurate. Understands grammar.
#   CONS: Slower (needs dictionary lookup). Needs POS tag for best results.
#
#
# COMPARISON TABLE:
#
#   Aspect          | Stemming              | Lemmatization
#   ----------------|-----------------------|-------------------------
#   Method          | Rule-based (chop)     | Dictionary-based (lookup)
#   Output          | May not be real word   | Always a real word
#   Speed           | Very fast              | Slower
#   Accuracy        | Lower                  | Higher
#   "studies"       | "studi" (wrong)        | "study" (correct)
#   "better"        | "better" (no change)   | "good" (correct)
#   "caring"        | "car" (wrong)          | "care" (correct)
#   "mice"          | "mice" (no change)     | "mouse" (correct)
#   Use case        | Search engines, IR     | Chatbots, NLP pipelines
#
#
# HOW INTERVIEWERS ASK THIS:
#
#   Question 1: "What is the difference between stemming and lemmatization?"
#   Answer: "Stemming chops word endings using rules — fast but can produce
#   non-real words like 'studi' from 'studies'. Lemmatization uses a dictionary
#   to find the actual root word — slower but always produces real words like
#   'study' from 'studies' or 'good' from 'better'."
#
#   Question 2: "When would you use stemming vs lemmatization?"
#   Answer: "Stemming for speed-critical tasks like search engine indexing
#   where approximate matching is fine. Lemmatization for accuracy-critical
#   tasks like chatbots, sentiment analysis, or text classification where
#   the actual word meaning matters."
#
#   Question 3: "Do LLMs use stemming or lemmatization?"
#   Answer: "Neither. LLMs use subword tokenization (BPE) which handles
#   word variations differently. 'running' might become ['run', 'ning'] and
#   'studies' becomes ['stud', 'ies']. The model learns the relationships
#   between these subwords during training. So stemming and lemmatization
#   are traditional NLP techniques — LLMs don't need them."
#
#   Question 4: "Do you use stemming/lemmatization in your RAG pipeline?"
#   Answer: "Not directly. My RAG pipeline uses embedding-based retrieval
#   where semantic similarity handles word variations automatically.
#   'running' and 'run' have similar embeddings, so the retriever finds
#   both without explicit stemming. However, BM25 (sparse retrieval in
#   hybrid search) can benefit from stemming to match word variants."
#
#
# HOW IT CONNECTS TO GENAI:
#   Traditional NLP: stemming/lemmatization were ESSENTIAL preprocessing steps.
#   Modern GenAI: LLMs handle word variations via subword tokenization and
#   embeddings. You don't need to stem or lemmatize before sending text to
#   an LLM. But understanding these concepts shows you know NLP fundamentals.


# ██████████████████████████████████████████████████████████████████████████████
# PART B: DEEP LEARNING BASICS
# ██████████████████████████████████████████████████████████████████████████████


# ==============================================================================
# 12. WHAT IS A NEURAL NETWORK
# ==============================================================================
#
# WHAT: A neural network is layers of connected "neurons" that learn patterns.
# Inspired by the human brain (loosely), but really just math.
#
# STRUCTURE:
#   Input Layer:  receives the data (e.g., pixel values of an image)
#   Hidden Layers: process the data (learn patterns)
#   Output Layer: produces the result (e.g., "cat" or "dog")
#
#   Each neuron computes: output = activation(w1*x1 + w2*x2 + ... + b)
#   That's linear regression + an activation function.
#
# WHY "DEEP" LEARNING?
#   "Deep" = many hidden layers. More layers = more complex patterns.
#   1-2 layers: simple patterns (lines, edges)
#   10+ layers: complex patterns (faces, objects)
#   96+ layers: language understanding (GPT-3)
#
# PARAMETERS:
#   Weights and biases in all neurons combined.
#   GPT-3: 175 BILLION parameters
#   GPT-4: estimated 1.8 TRILLION parameters
#   These parameters are what the model "learns" during training.
#
# INTERVIEW ANSWER:
#   "A neural network is layers of neurons where each computes a weighted
#   sum plus activation function. Deep learning means many layers — more
#   layers capture more complex patterns. GPT-3 has 96 layers and 175
#   billion parameters. These parameters are learned during training."


# ==============================================================================
# 13. HOW NEURAL NETWORKS LEARN (Backpropagation)
# ==============================================================================
#
# THE LEARNING PROCESS:
#   1. FORWARD PASS: Input goes through the network, produces a prediction.
#   2. LOSS CALCULATION: Compare prediction to correct answer. Compute error.
#   3. BACKWARD PASS (Backpropagation): Calculate how much each weight
#      contributed to the error. Propagate error backwards through layers.
#   4. WEIGHT UPDATE: Adjust weights to reduce error (gradient descent).
#   5. REPEAT: Do this millions of times with different examples.
#
# GRADIENT DESCENT:
#   The algorithm that adjusts weights. Imagine you're blindfolded on a hill
#   and want to reach the bottom (minimum error). You feel the slope under
#   your feet (gradient) and take a step downhill. Repeat until you reach
#   the bottom.
#
#   Learning Rate: how big each step is.
#     Too large: overshoot the minimum (bouncing around)
#     Too small: takes forever to converge
#     Just right: smooth convergence to minimum error
#
# INTERVIEW ANSWER:
#   "Neural networks learn through backpropagation: forward pass produces
#   a prediction, loss function measures the error, backpropagation
#   calculates gradients (how much each weight contributed to the error),
#   and gradient descent adjusts weights to reduce the error. This repeats
#   millions of times until the model converges."


# ==============================================================================
# 14. ACTIVATION FUNCTIONS
# ==============================================================================
#
# WHY NEEDED: Without activation functions, a neural network is just
# stacked linear equations — which simplifies to ONE linear equation.
# Activation functions add NON-LINEARITY so the network can learn
# complex patterns (curves, not just straight lines).
#
# COMMON ACTIVATION FUNCTIONS:
#
#   ReLU (Rectified Linear Unit): f(x) = max(0, x)
#     Most popular. Simple and fast. Used in hidden layers.
#     If x > 0, output = x. If x <= 0, output = 0.
#
#   Sigmoid: f(x) = 1 / (1 + e^(-x))
#     Outputs between 0 and 1. Used for binary classification output.
#     Problem: vanishing gradient (gradients become tiny in deep networks).
#
#   Tanh: f(x) = (e^x - e^(-x)) / (e^x + e^(-x))
#     Outputs between -1 and 1. Better than sigmoid for hidden layers.
#     Still has vanishing gradient problem.
#
#   Softmax: converts a vector of numbers into probabilities that sum to 1.
#     Used in the OUTPUT layer for multi-class classification.
#     THIS IS WHAT TRANSFORMERS USE for next-token prediction.
#
#   GELU (Gaussian Error Linear Unit): used in Transformers (GPT, BERT).
#     Smoother version of ReLU. Better for language models.
#
# INTERVIEW ANSWER:
#   "Activation functions add non-linearity so networks can learn complex
#   patterns. ReLU is most common for hidden layers. Softmax is used in
#   Transformer output layers to convert logits into token probabilities.
#   GELU is used inside Transformer layers for smoother gradients."


# ==============================================================================
# 15. CNN (Convolutional Neural Networks) - Brief
# ==============================================================================
#
# WHAT: Specialized neural networks for IMAGE processing.
# Uses "filters" that slide across the image to detect patterns.
#
# HOW: Small filters (3x3 or 5x5) scan the image:
#   Early layers: detect edges, corners
#   Middle layers: detect shapes, textures
#   Deep layers: detect objects, faces
#
# NOT DIRECTLY USED IN GENAI TEXT MODELS, but:
#   - Vision models (GPT-4V, CLIP) use CNNs or Vision Transformers
#   - Multimodal RAG uses image processing (covered in RAG Lesson 4)
#
# INTERVIEW ANSWER:
#   "CNNs use sliding filters to detect patterns in images — edges in
#   early layers, objects in deeper layers. They're not used in text LLMs
#   but are used in multimodal models like GPT-4V and CLIP for image
#   understanding."


# ==============================================================================
# 16. RNN (Recurrent Neural Networks) - Important!
# ==============================================================================
#
# WHAT: Neural networks designed for SEQUENTIAL data (text, time series).
# They process tokens ONE AT A TIME(Slow and may loose context sometimes) and maintain a "hidden state" that
# carries information from previous tokens.
#
# HOW:
#   Token 1 ("The") -> RNN -> hidden_state_1
#   Token 2 ("cat") -> RNN + hidden_state_1 -> hidden_state_2
#   Token 3 ("sat") -> RNN + hidden_state_2 -> hidden_state_3
#   ...
#   Each step uses the previous hidden state as memory.
#
# THE PROBLEM (why Transformers replaced RNNs):
#   1. SEQUENTIAL: processes one token at a time. Can't parallelize.
#      A 1000-token sentence needs 1000 sequential steps. SLOW.
#   2. VANISHING GRADIENT: by the time you reach token 1000, the
#      gradient from token 1 has nearly vanished. The model FORGETS
#      early tokens. Long-range dependencies are lost.
#   3. SHORT MEMORY: practically, RNNs remember ~20-50 tokens back.
#      Not enough for paragraphs or documents.
#
# INTERVIEW ANSWER:
#   "RNNs process sequences one token at a time, carrying a hidden state
#   as memory. The problem: they're sequential (can't parallelize) and
#   suffer from vanishing gradients (forget early tokens in long sequences).
#   This is why LSTMs and then Transformers(Process token paralley) were invented."


# ==============================================================================
# 17. LSTM (Long Short-Term Memory) - Important!
# ==============================================================================
#
# WHAT: An improved RNN that can remember information for LONGER sequences.
# Invented in 1997 by Hochreiter and Schmidhuber.
#
# HOW IT FIXES RNN's PROBLEM:
#   LSTM adds GATES that control what to remember and what to forget:
#
#   Forget Gate: "Should I forget the previous information?"
#     Looks at current input + previous state. Outputs 0 (forget) to 1 (keep).
#
#   Input Gate: "Should I store this new information?"
#     Decides what new information to add to the cell state.
#
#   Output Gate: "What should I output right now?"
#     Decides what part of the cell state to output.
#
#   Cell State: the long-term memory that flows through the network.
#     Gates control what enters and leaves the cell state.
#
# STILL HAS PROBLEMS:
#   1. Still SEQUENTIAL (one token at a time). Can't parallelize.
#   2. Better than RNN but still struggles with very long sequences (1000+ tokens).
#   3. Complex architecture = slow to train.
#
# THE EVOLUTION:
#   RNN (1986) -> LSTM (1997) -> Attention (2014) -> Transformer (2017)
#   Each step solved a problem of the previous one.
#
# INTERVIEW ANSWER:
#   "LSTM improves on RNN by adding gates (forget, input, output) that
#   control what to remember and forget. This allows longer memory than
#   basic RNNs. But LSTMs are still sequential and struggle with very
#   long sequences. Transformers solved this with parallel self-attention."


# ==============================================================================
# 18. THE ATTENTION MECHANISM (Bridge from DL to GenAI)
# ==============================================================================
#
# WHAT: Attention lets the model FOCUS on the most relevant parts of the
# input when producing each output. Invented in 2014 for machine translation.
#
# THE PROBLEM IT SOLVED:
#   In RNN/LSTM translation, the entire input sentence is compressed into
#   ONE fixed-size vector. For long sentences, information is lost.
#   "The cat that was sitting on the mat in the living room was tired"
#   -> compressed to one vector -> loses details.
#
#   Attention says: "Instead of compressing everything, let the decoder
#   LOOK BACK at all encoder states and focus on the relevant ones."
#
# EXAMPLE (translation):
#   Translating "The cat is tired" to French "Le chat est fatigue"
#   When generating "chat" (cat), attention focuses on "cat" in the input.
#   When generating "fatigue" (tired), attention focuses on "tired."
#
# SELF-ATTENTION (2017 - Transformers):
#   Instead of encoder-decoder attention, each token attends to EVERY
#   OTHER token in the SAME sequence. No RNN needed at all.
#   This is the key innovation of the Transformer.
#
# INTERVIEW ANSWER:
#   "Attention lets the model focus on relevant parts of the input for
#   each output token. It was invented for translation but self-attention
#   in Transformers took it further — each token attends to every other
#   token in the same sequence, enabling parallel processing and long-range
#   understanding and add positional encoding. This replaced RNNs and LSTMs entirely."


# ==============================================================================
# 19. FROM RNN TO TRANSFORMER - The Evolution (Interview Gold)
# ==============================================================================
#
# This is a GREAT interview answer when they ask "Why Transformers?"
#
# THE TIMELINE:
#
# 1986: RNN
#   Process sequences one token at a time.
#   Problem: vanishing gradient, forgets early tokens.
#
# 1997: LSTM
#   Added gates to control memory.
#   Better long-term memory than RNN.
#   Problem: still sequential, still slow, still limited memory.
#
# 2014: Attention Mechanism
#   Let the decoder look back at all encoder states and focus on the relevant ones."
#   Solved the information bottleneck.
#   Problem: still used RNN as the backbone.
#
# 2017: Transformer ("Attention Is All You Need")
#   Removed RNN entirely. ONLY attention.
#   Self-attention: each token attends to every other token.
#   Parallel processing: all tokens processed simultaneously.
#   Solved: speed (parallel), memory (attention to all tokens), scale.
#
# 2018-2025: The LLM Era
#   BERT (2018): encoder-only Transformer for understanding.
#   GPT-2 (2019): decoder-only Transformer for generation.
#   GPT-3 (2020): 175B parameters, few-shot learning.
#   ChatGPT (2022): GPT-3.5 + RLHF = conversational AI.
#   GPT-4 (2023): multimodal, 128K context, best reasoning.
#   LLaMA, Claude, Gemini (2023-2025): competition and open-source.
#
# INTERVIEW ANSWER:
#   "The evolution went RNN -> LSTM -> Attention -> Transformer. RNNs were
#   sequential and forgot early tokens. LSTMs added memory gates but were
#   still sequential. Attention let models focus on relevant parts. The
#   Transformer removed RNNs entirely, using only self-attention for parallel
#   processing. This enabled scaling to billions of parameters and is the
#   foundation of all modern LLMs."


# ██████████████████████████████████████████████████████████████████████████████
# PART C: INTERVIEW Q&A - 20 Questions
# ██████████████████████████████████████████████████████████████████████████████
#
# Q1: What is the difference between supervised and unsupervised learning?
# A: Supervised uses labeled data (input + correct answer) for tasks like
#    classification and regression. Unsupervised finds patterns in unlabeled
#    data, like clustering similar items. LLM fine-tuning is supervised;
#    embedding learning is unsupervised.
#
# Q2: What is overfitting and how do you prevent it?
# A: Overfitting is when the model memorizes training data but fails on new
#    data. Prevent with: more data, regularization, dropout, early stopping,
#    or simpler models. In GenAI, hallucination is related to overfitting
#    on training patterns.
#
# Q3: Explain precision vs recall.
# A: Precision: of all predicted positives, how many are correct (low false
#    alarms). Recall: of all actual positives, how many were caught (high
#    detection). F1 balances both. In RAG, context precision and recall
#    measure retrieval quality.
#
# Q4: What is gradient descent?
# A: The optimization algorithm that adjusts model weights to minimize error.
#    It calculates the gradient (slope) of the loss function and takes steps
#    in the direction that reduces loss. Learning rate controls step size.
#
# Q5: What is a neural network?
# A: Layers of neurons where each computes a weighted sum plus activation
#    function. Input layer receives data, hidden layers learn patterns,
#    output layer produces results. "Deep" means many hidden layers.
#
# Q6: What is backpropagation?
# A: The algorithm that calculates how much each weight contributed to the
#    error. It propagates the error backwards from output to input, computing
#    gradients at each layer. These gradients are used by gradient descent
#    to update weights.
#
# Q7: Why do we need activation functions?
# A: Without them, stacked linear layers simplify to one linear equation.
#    Activation functions add non-linearity so the network can learn complex
#    patterns. ReLU for hidden layers, softmax for classification output.
#
# Q8: What is the vanishing gradient problem?
# A: In deep networks, gradients become extremely small as they propagate
#    backwards through many layers. Early layers barely learn. RNNs suffer
#    from this with long sequences. LSTMs and Transformers solve it.
#
# Q9: What is the difference between RNN and LSTM?
# A: RNN carries a hidden state but suffers from vanishing gradients and
#    short memory. LSTM adds forget/input/output gates and a cell state
#    for long-term memory. Both are sequential. Transformers replaced both
#    with parallel self-attention.
#
# Q10: Why did Transformers replace RNNs?
# A: Three reasons: (1) Parallel processing — all tokens at once vs one at
#     a time. (2) Self-attention — each token sees every other token, no
#     information loss. (3) Scalability — can train on billions of parameters.
#     RNNs can't parallelize and forget early tokens.
#
# Q11: What is the attention mechanism?
# A: Attention lets the model focus on relevant parts of the input for each
#     output. Self-attention in Transformers lets each token attend to every
#     other token using Query, Key, Value vectors. This is the core of all
#     modern LLMs.
#
# Q12: What is transfer learning?
# A: Using a model trained on one task as a starting point for another task.
#     Instead of training from scratch, you take a pre-trained model (like
#     GPT or BERT) and fine-tune it on your specific data. This saves time,
#     data, and compute. All LLM usage is transfer learning.
#
# Q13: What is the difference between CNN and RNN?
# A: CNN is for spatial data (images) — uses filters to detect patterns.
#     RNN is for sequential data (text, time series) — processes one step
#     at a time with memory. Transformers handle both with self-attention.
#
# Q14: What is regularization?
# A: Techniques to prevent overfitting. L1/L2 regularization adds a penalty
#     for large weights. Dropout randomly disables neurons during training.
#     In LLMs, dropout is used in Transformer layers during training.
#
# Q15: What is the difference between classification and regression?
# A: Classification predicts categories (spam/not-spam, cat/dog).
#     Regression predicts continuous numbers (price, temperature).
#     Next-token prediction is classification over the vocabulary.
#
# Q16: What is feature engineering?
# A: Creating new input features from raw data to help the model learn.
#     Example: from a date, create "day of week", "is weekend", "month."
#     In NLP, features include word count, TF-IDF scores, embeddings.
#     LLMs do automatic feature engineering via their hidden layers.
#
# Q17: What is cross-validation?
# A: Splitting data into K folds, training K times (each fold as test once),
#     and averaging results. More reliable than a single train/test split.
#     K=5 or K=10 is common. Used to tune hyperparameters.
#
# Q18: What is the bias-variance tradeoff?
# A: High bias = underfitting (model too simple, misses patterns).
#     High variance = overfitting (model too complex, memorizes noise).
#     The goal is to find the sweet spot: complex enough to learn patterns
#     but simple enough to generalize to new data.
#
# Q19: What is an epoch, batch, and iteration?
# A: Epoch: one complete pass through the entire training dataset.
#     Batch: a subset of data processed at once (e.g., 32 examples).
#     Iteration: one weight update (one batch processed).
#     If dataset has 1000 examples and batch size is 100:
#     1 epoch = 10 iterations = 10 batches.
#
# Q20: How does all of this connect to your GenAI work?
# A: "ML basics like supervised learning and evaluation metrics are the
#     foundation. Deep learning concepts like neural networks, RNNs, and
#     attention led to the Transformer architecture that powers all LLMs.
#     In my DocSage project, I use embeddings (from DL), vector similarity
#     (from ML), structured output classification (supervised learning),
#     and the Transformer-based Groq LLMs for generation. Understanding
#     these fundamentals helps me debug issues, choose the right models,
#     and explain my system architecture in interviews."
