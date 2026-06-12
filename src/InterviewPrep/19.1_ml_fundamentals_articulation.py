"""
===================================================================================
ML FUNDAMENTALS — Articulation for Senior Interviews
===================================================================================

In your PwC interview, you couldn't clearly articulate the difference between
classification, regression, and clustering. This lesson fixes that.

The interviewer was testing if you understand the FUNDAMENTALS — even as a
GenAI engineer, you need to articulate ML basics CLEANLY.

SECTIONS:
    1.  The Big Picture — What is Machine Learning?
    2.  The 3 Learning Paradigms (Supervised, Unsupervised, Reinforcement)
    3.  Classification vs Regression vs Clustering (Articulation)
    4.  What is "Categorization"? (Common Confusion Cleared Up)
    5.  When to Use Which Algorithm (Decision Matrix)
    6.  Confusion Cleared — Common Mix-ups
    7.  How to Articulate These in Interviews
    8.  15+ Interview Q&A
    9.  GOLDEN LESSONS

Read this twice. The articulation matters more than the technical knowledge.
===================================================================================
"""


# =================================================================================
# SECTION 1: THE BIG PICTURE — What is Machine Learning?
# =================================================================================
"""
ML = Computers learning patterns from DATA, without being explicitly programmed.

    Traditional programming:  Rules + Data → Output
    Machine Learning:         Data + Output → Rules (the model)

EXAMPLE:
    Traditional: "If email contains 'lottery', mark as spam." (you write the rule)
    ML: Show the model 1000 emails with spam/not-spam labels → it LEARNS the rules.

THE 3 STEPS OF ML:
    1. TRAIN — feed labeled data, model learns patterns
    2. EVALUATE — test on unseen data, measure accuracy
    3. PREDICT — use model on new, real data

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

INTERVIEW ANSWER:
    "Machine learning is computers learning patterns from data instead of being
    explicitly programmed. Three steps: train on labeled data, evaluate on
    unseen data, then predict on new data. The output is a 'model' — a
    mathematical representation of the patterns it learned."
"""


# =================================================================================
# SECTION 2: THE 3 LEARNING PARADIGMS
# =================================================================================
"""
ML is divided into 3 fundamental paradigms based on what data the model learns from.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PARADIGM 1: SUPERVISED LEARNING (Most Common)

    DATA: Each example has an INPUT and a KNOWN OUTPUT (label).
    GOAL: Learn the mapping from input → output.

    Example: 1000 photos labeled "cat" or "dog". Model learns to predict.

    SUB-TYPES:
    - Classification (predict category)
    - Regression (predict number)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PARADIGM 2: UNSUPERVISED LEARNING

    DATA: Inputs WITHOUT labels.
    GOAL: Find hidden structure or patterns in the data.

    Example: 10,000 customer purchase records (no labels). Model groups
    similar customers together.

    SUB-TYPES:
    - Clustering (group similar items)
    - Dimensionality reduction (compress features)
    - Anomaly detection (find outliers)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PARADIGM 3: REINFORCEMENT LEARNING

    DATA: Agent takes actions, gets rewards/penalties.
    GOAL: Learn the best ACTIONS to maximize reward.

    Example: Robot learns to walk. Falls = penalty. Walks = reward.
    AlphaGo learning to play Go.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

QUICK MENTAL MODEL:

    Supervised:    "Here's data + answers, learn the mapping."
    Unsupervised:  "Here's data, find structure on your own."
    Reinforcement: "Try things, get rewards, learn what works."

INTERVIEW ANSWER:
    "Three paradigms. Supervised learns from labeled data — input/output pairs —
    for tasks like predicting cat vs dog. Unsupervised finds structure in
    unlabeled data, like grouping customers by behavior. Reinforcement learns
    by taking actions and getting rewards, like a robot learning to walk.
    Most production ML is supervised because labeled data gives the clearest
    signal."
"""


# =================================================================================
# SECTION 3: CLASSIFICATION vs REGRESSION vs CLUSTERING
# =================================================================================
"""
THIS IS THE EXACT QUESTION YOU COULDN'T ANSWER. Let's nail it.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CLASSIFICATION (Supervised, predicts CATEGORY)

    What it does: Predict which CATEGORY an input belongs to.
    Output: A LABEL from a fixed set of options.

    EXAMPLES:
    - Email spam? → "spam" or "not spam" (binary classification)
    - What's in this photo? → "cat", "dog", "horse" (multi-class)
    - Customer churn? → "will churn", "won't churn"
    - Disease diagnosis? → "covid", "flu", "cold"

    KEY: Output is one of a FIXED, FINITE set of categories.

    ALGORITHMS: Logistic Regression, Decision Trees, Random Forest, SVM,
                Neural Networks, XGBoost

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

REGRESSION (Supervised, predicts NUMBER)

    What it does: Predict a CONTINUOUS NUMBER.
    Output: A REAL NUMBER (could be any value in a range).

    EXAMPLES:
    - House price? → $350,000, $425,000, $580,000 (any number)
    - Tomorrow's temperature? → 28.5°C, 24.7°C
    - Stock price next week? → $145.32
    - How long until system fails? → 5.2 days

    KEY: Output is a number on a CONTINUOUS scale.

    ALGORITHMS: Linear Regression, Polynomial Regression, Random Forest
                Regressor, XGBoost Regressor, Neural Networks

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CLUSTERING (Unsupervised, groups items)

    What it does: GROUP similar items together. NO LABELS in training data.
    Output: GROUPS (clusters) of similar items.

    EXAMPLES:
    - Customer segmentation: group customers by buying behavior
    - Document clustering: group articles by topic (no pre-defined topics)
    - Image grouping: organize photos by similarity
    - Anomaly detection: find outliers (small clusters)

    KEY: Output is GROUPS that didn't exist before. The algorithm
    DISCOVERS them.

    ALGORITHMS: K-Means, DBSCAN, Hierarchical Clustering, Gaussian Mixture

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

THE ARTICULATION TABLE — Use this in your answer:

    ASPECT         CLASSIFICATION    REGRESSION         CLUSTERING
    Output type    Category/label    Number             Group ID
    Output range   Fixed set         Continuous range    Discovered groups
    Labeled data?  YES               YES                NO
    Paradigm       Supervised        Supervised         Unsupervised
    Question       "Which class?"    "How much?"        "What groups exist?"
    Example        Spam/not spam     House price        Customer segments

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

THE 30-SECOND ARTICULATION (memorize this):

    "Three different problem types. CLASSIFICATION predicts CATEGORIES from
    a fixed set — like spam vs not-spam, or cat vs dog. REGRESSION predicts
    CONTINUOUS NUMBERS — like house prices or temperatures. CLUSTERING
    GROUPS similar items WITHOUT labels — like segmenting customers by
    behavior. Classification and regression are SUPERVISED (need labeled
    data). Clustering is UNSUPERVISED (no labels needed)."
"""


# =================================================================================
# SECTION 4: WHAT IS "CATEGORIZATION"?
# =================================================================================
"""
The interviewer asked about "categorization" too. This causes confusion.

CATEGORIZATION is essentially synonymous with CLASSIFICATION in ML context.
Both mean: assigning items to predefined CATEGORIES.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

DIFFERENCES (subtle):

    CLASSIFICATION:
    - Technical ML term
    - Used in literature, research, ML libraries
    - Implies the model is making a prediction with learned patterns

    CATEGORIZATION:
    - More general / business term
    - Sometimes used for rule-based assignment too
    - "Categorize these emails" could mean ML or manual sorting

    In a senior ML interview, prefer CLASSIFICATION when discussing models.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

POSSIBLE CONFUSION: "What about taxonomies / hierarchical categorization?"

    HIERARCHICAL CLASSIFICATION:
    - Categories have parent-child relationships
    - Example: Animal → Mammal → Dog → Labrador

    MULTI-LABEL CLASSIFICATION:
    - One item belongs to MULTIPLE categories simultaneously
    - Example: a movie tagged as "Comedy" + "Romance" + "Action"

    These are still CLASSIFICATION, just more complex variants.

INTERVIEW ANSWER:
    "Categorization in ML context is the same as classification — assigning
    items to predefined categories. Classification is the technical term I'd
    use. Variants include binary (2 classes), multi-class (more than 2),
    multi-label (one item, multiple labels), and hierarchical (categories
    with parent-child relationships)."
"""


# =================================================================================
# SECTION 5: WHEN TO USE WHICH ALGORITHM
# =================================================================================
"""
DECISION MATRIX — Pick the right approach for the problem:

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

QUESTION YOU'RE ASKING                 PROBLEM TYPE        EXAMPLE ALGORITHMS

"Which category does this belong to?"  Classification      Logistic Regression,
                                                          Random Forest, XGBoost,
                                                          Neural Networks

"How much / how many?"                 Regression          Linear Regression,
                                                          Random Forest Regressor,
                                                          XGBoost Regressor

"What groups exist in my data?"        Clustering          K-Means, DBSCAN,
                                                          Hierarchical Clustering

"Is this an outlier?"                  Anomaly Detection   Isolation Forest,
                                                          Local Outlier Factor

"What's the optimal action?"           Reinforcement       Q-Learning, PPO, DQN

"Reduce features without info loss?"   Dim. Reduction      PCA, t-SNE, UMAP

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

WHEN TO USE GENAI vs CLASSICAL ML:

    USE CLASSICAL ML:
    - Tabular data (numbers, categories)
    - Predicting a specific number/class
    - Need explainability (regulated industries)
    - Limited data size or compute
    - Speed matters more than language understanding

    USE GENERATIVE AI (LLMs):
    - Free-form text in or out
    - Open-ended tasks (summarization, Q&A)
    - Tasks needing reasoning/conversation
    - Need to handle ambiguity
    - When you want zero-shot capability (no training)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

EXAMPLE DECISIONS:

    "Predict tomorrow's stock price"
    → Regression (XGBoost on historical features)
    → NOT GenAI — wrong tool

    "Classify customer support tickets into 5 categories"
    → Classification (start with a simple model)
    → GenAI is overkill if rules work

    "Group customers into segments"
    → Clustering (K-Means)
    → No labels needed

    "Generate marketing copy for new products"
    → GenAI (LLM)
    → Classical ML can't do this

INTERVIEW ANSWER:
    "I pick the algorithm based on the question. 'Which category?' →
    classification. 'How much?' → regression. 'What groups exist?' →
    clustering. For tabular numeric prediction, I start with classical
    ML (XGBoost is my default). For free-form text generation or
    reasoning, I use LLMs. Don't use GenAI for problems that classical
    ML solves cheaply and reliably."
"""


# =================================================================================
# SECTION 6: COMMON MIX-UPS CLEARED UP
# =================================================================================
"""
THESE ARE THE MISTAKES PEOPLE MAKE — AVOID THEM:

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

MIX-UP 1: "Classification has labels, regression doesn't"
    WRONG. Both are supervised — both need labels.
    Difference: classification labels are CATEGORIES, regression labels are NUMBERS.

MIX-UP 2: "Logistic Regression is regression"
    DECEPTIVE NAME. Logistic Regression is actually CLASSIFICATION.
    The 'regression' in the name refers to the math (logistic function for
    probability), but it OUTPUTS classes.

MIX-UP 3: "Clustering and Classification are the same"
    NO. Classification has predefined classes; clustering DISCOVERS groups.
    Classification is supervised; clustering is unsupervised.

MIX-UP 4: "K-Means can do classification"
    NO. K-Means is unsupervised — it finds clusters but doesn't know what
    they MEAN. You'd need to label the clusters AFTER, or use K-NN
    (different algorithm) for classification.

MIX-UP 5: "Regression always means linear regression"
    NO. Regression is the PROBLEM TYPE. Linear regression is one ALGORITHM.
    Random Forest, Neural Networks, XGBoost — all can do regression.

MIX-UP 6: "Categorization = unsupervised"
    NO. Categorization usually means classification (supervised).
    Unsupervised = clustering or dimensionality reduction.

INTERVIEW ANSWER (when asked these):
    "Logistic Regression despite its name is a CLASSIFICATION algorithm —
    'regression' refers to the math, not the problem type. K-Means is
    UNSUPERVISED — it finds clusters but you must interpret them. The
    PROBLEM TYPE (classification, regression, clustering) determines the
    approach; multiple ALGORITHMS can solve each problem type."
"""


# =================================================================================
# SECTION 7: HOW TO ARTICULATE THESE IN INTERVIEWS
# =================================================================================
"""
The interviewer doesn't just want correct facts — they want CLEAR ARTICULATION.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

THE STRUCTURED ANSWER FORMAT:

    Q: "What's the difference between classification and regression?"

    BAD ANSWER (rambling):
        "Classification is for like categories and regression is for numbers
        and they're both supervised but yeah classification has classes..."

    GOOD ANSWER (structured):
        "Both are supervised learning, but they differ in OUTPUT TYPE.
        Classification predicts a CATEGORY from a fixed set —
        like 'spam' or 'not spam.'
        Regression predicts a CONTINUOUS NUMBER —
        like a house price.
        Different output type means different algorithms and different
        evaluation metrics. Classification uses accuracy, F1, AUC.
        Regression uses MAE, RMSE, R-squared."

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

THE 4-PART ARTICULATION TEMPLATE:

    1. Common ground ("Both are X...")
    2. Key difference ("...but they differ in Y")
    3. Concrete examples ("classification: spam, regression: house price")
    4. Implication ("different metrics, different use cases")

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PRACTICE THESE 30-SECOND ANSWERS OUT LOUD:

    "Classification vs Regression?"
    → "Both supervised. Classification: predict CATEGORY. Regression:
       predict NUMBER. Spam detection vs house price."

    "Classification vs Clustering?"
    → "Classification: supervised, predefined classes. Clustering:
       unsupervised, discovers groups. Classify emails as spam vs
       cluster customers by behavior."

    "Supervised vs Unsupervised?"
    → "Supervised: labeled data, predict input → output. Unsupervised:
       no labels, find structure. Spam classifier vs customer segmentation."

INTERVIEW ANSWER FRAMEWORK:
    "When asked technical comparison questions, I use the 4-part template:
    common ground, key difference, examples, implications. This shows
    structured thinking, not just memorization."
"""


# =================================================================================
# SECTION 8: 15+ INTERVIEW Q&A
# =================================================================================
"""
TIER 1 — FOUNDATION (must know cold):

Q1: "What's machine learning?"
A: "Computers learning patterns from data instead of being explicitly programmed.
   Three steps: train, evaluate, predict. Output is a model — a mathematical
   representation of the learned patterns."

Q2: "Three types of ML?"
A: "Supervised (labeled data, predict input→output, like classification),
   Unsupervised (no labels, find structure, like clustering),
   Reinforcement (agent learns by taking actions and getting rewards)."

Q3: "Classification vs regression?"
A: "Both supervised. Classification predicts a CATEGORY from fixed set
   (spam/not-spam). Regression predicts a CONTINUOUS NUMBER (house price).
   Different outputs, different algorithms, different evaluation metrics."

Q4: "Classification vs clustering?"
A: "Classification: supervised, predefined classes (spam/not-spam).
   Clustering: unsupervised, discovers groups in unlabeled data
   (customer segments). Different paradigms, different problems."

Q5: "Give an example of each: classification, regression, clustering."
A: "Classification: predict if email is spam (yes/no).
   Regression: predict tomorrow's stock price (a number).
   Clustering: group customers by purchase behavior into segments."

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

TIER 2 — DEEPER UNDERSTANDING:

Q6: "Why is logistic regression called regression if it's classification?"
A: "Misleading name. The 'regression' refers to the underlying math (it
   uses a logistic function). But it OUTPUTS class probabilities and is
   used for classification. Don't be fooled by the name."

Q7: "Can clustering be supervised?"
A: "By definition, no — clustering is fundamentally unsupervised. If you
   have labels, you'd use classification instead. There's a related
   concept called 'semi-supervised learning' where some data is labeled,
   but pure clustering is unsupervised."

Q8: "How do you decide between classification and regression?"
A: "Look at the OUTPUT. If output is a discrete category (spam/not-spam,
   cat/dog/bird), it's classification. If output is a number on a
   continuous scale (price, temperature, age), it's regression."

Q9: "Common algorithms for each?"
A: "Classification: Logistic Regression, Random Forest, SVM, XGBoost,
   Neural Networks.
   Regression: Linear Regression, Random Forest Regressor, XGBoost
   Regressor, Neural Networks.
   Clustering: K-Means, DBSCAN, Hierarchical, Gaussian Mixture."

Q10: "Evaluation metrics for each?"
A: "Classification: accuracy, precision, recall, F1, AUC-ROC.
   Regression: MAE, MSE, RMSE, R-squared, MAPE.
   Clustering: silhouette score, Davies-Bouldin index, intra/inter-cluster distance."

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

TIER 3 — PRACTICAL APPLICATION:

Q11: "When would you NOT use machine learning?"
A: "When simple rules work fine, when you don't have enough data, when
   explainability is critical and ML can't provide it, when the problem
   has clear deterministic logic. Don't ML what if-statements can solve."

Q12: "GenAI vs classical ML — when to use which?"
A: "Classical ML for tabular data, specific number/class predictions,
   when explainability matters. GenAI (LLMs) for free-form text in/out,
   open-ended tasks, conversation, when handling ambiguity. Don't use
   GenAI for what XGBoost can do better and cheaper."

Q13: "How do you handle imbalanced data in classification?"
A: "Imbalanced = one class is much more frequent (e.g., 99% normal, 1%
   fraud). Techniques: oversampling minority (SMOTE), undersampling
   majority, class weights in the model, focal loss, ensemble methods.
   Use F1 or AUC instead of accuracy."

Q14: "What's overfitting and how do you prevent it?"
A: "Overfitting: model memorizes training data instead of learning general
   patterns. Symptoms: high training accuracy, low test accuracy. Prevention:
   more data, regularization (L1/L2), dropout, cross-validation, early stopping,
   simpler model architecture."

Q15: "What's the bias-variance tradeoff?"
A: "Bias: model is too simple, can't capture patterns (underfitting).
   Variance: model is too complex, memorizes noise (overfitting).
   Tradeoff: simpler models = high bias / low variance. Complex models =
   low bias / high variance. Goal: minimize total error by balancing both."

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

TIER 4 — FOR GENAI ENGINEERS:

Q16: "Is RAG a form of supervised or unsupervised?"
A: "Neither traditional category. RAG uses pre-trained LLMs (which were
   trained with self-supervised learning) and adds retrieval at inference
   time. The retrieval part is similarity-based (closer to unsupervised
   nearest-neighbor search). RAG itself doesn't 'learn' from your data —
   it retrieves and grounds."

Q17: "Can an LLM do classification?"
A: "Yes — zero-shot or few-shot. Zero-shot: 'Classify this email as spam
   or not spam.' Few-shot: provide 3-5 examples, then the new email.
   Works without training but more expensive and slower than a dedicated
   classifier. Use traditional ML for high-volume classification."

Q18: "Why might you fine-tune an LLM for classification?"
A: "When you have lots of labeled examples and want consistent behavior
   on a specific task. Fine-tuning makes the LLM a specialist. But for
   most cases, prompt engineering or RAG works without fine-tuning."

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SUPER-CONCISE ANSWERS (for rapid-fire questions):

    "Classification?" → "Predict a category. Spam/not-spam. Supervised."
    "Regression?" → "Predict a number. House prices. Supervised."
    "Clustering?" → "Group similar items. No labels. Unsupervised."
    "Categorization?" → "Same as classification."
    "Supervised?" → "Labeled data. Input → output mapping."
    "Unsupervised?" → "No labels. Find structure."
    "Reinforcement?" → "Take actions. Get rewards. Learn policy."
"""


# =================================================================================
# SECTION 9: GOLDEN LESSONS
# =================================================================================
"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

GOLDEN LESSON 1: "Articulation > Memorization."
    Knowing facts isn't enough. You must EXPLAIN them clearly. Use the
    4-part template: common ground, key difference, examples, implications.

GOLDEN LESSON 2: "Output type defines the problem type."
    Output is a category? → Classification.
    Output is a number? → Regression.
    Output is groups (no labels)? → Clustering.
    Always start by asking "what does the model output?"

GOLDEN LESSON 3: "Even GenAI engineers need ML fundamentals."
    Senior interviews test breadth + depth. You'll be asked basic ML
    even in a GenAI role. Don't dismiss the fundamentals.

GOLDEN LESSON 4: "Logistic Regression is CLASSIFICATION (despite the name)."
    This is a frequent test. Don't be fooled by misleading names.

GOLDEN LESSON 5: "GenAI isn't always the answer."
    For tabular numeric prediction, classical ML (XGBoost) is faster,
    cheaper, more accurate. Choose tools based on problems, not hype.

GOLDEN LESSON 6: "The 30-second answer matters more than the 5-minute one."
    Senior interviewers can tell the difference between a candidate who
    truly understands (concise, clear answer) vs one who's faking
    (rambling, name-dropping).

GOLDEN LESSON 7: "Practice articulation OUT LOUD."
    Write your answer down. Read it out loud. Refine it. The first time
    you say something, you stumble. The 10th time, it flows.

GOLDEN LESSON 8: "Senior ML = combining classical + modern."
    A senior knows when to use XGBoost, when to use an LLM, when to use
    embeddings + classifier hybrid. Versatility matters.
"""

print("=" * 60)
print("ML Fundamentals Articulation — Complete")
print("=" * 60)
print()
print("9 Sections:")
print("  1.  The Big Picture (What is ML)")
print("  2.  3 Learning Paradigms (Supervised/Unsupervised/Reinforcement)")
print("  3.  Classification vs Regression vs Clustering")
print("  4.  What is Categorization (cleared up)")
print("  5.  Decision Matrix (when to use which)")
print("  6.  Common Mix-ups (avoid these)")
print("  7.  Articulation Templates (how to answer)")
print("  8.  18 Interview Q&A (Tiers 1-4)")
print("  9.  GOLDEN LESSONS (8 lessons)")
print()
print("Read this twice. Articulate out loud. Master fundamentals.")
print("=" * 60)
