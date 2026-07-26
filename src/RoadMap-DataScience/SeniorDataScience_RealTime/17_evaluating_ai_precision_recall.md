# Lesson 17 — Evaluating AI Responses (Precision, Recall in Code)

## 1. The Question

> "How do you evaluate the quality of AI/model responses? How do you compute precision and recall in code? Define TP/FP/FN in your context."

---

## 2. Theory — the confusion matrix

For any classification (or any task you can frame as "flag / don't flag"), every prediction falls into one of four cells vs the ground truth:

|                | Actual Positive | Actual Negative |
|----------------|-----------------|-----------------|
| **Predicted +** | True Positive (TP)  | False Positive (FP) |
| **Predicted −** | False Negative (FN) | True Negative (TN)  |

From these:

- **Precision** = TP / (TP + FP) — *of everything I flagged as positive, how much was right?* Penalizes false alarms.
- **Recall** (sensitivity) = TP / (TP + FN) — *of everything that was actually positive, how much did I catch?* Penalizes misses.
- **F1** = harmonic mean = 2·P·R / (P + R) — single number balancing both.
- **Accuracy** = (TP + TN) / all — misleading on **imbalanced** data (99% "no fraud" → predict all "no" = 99% accuracy, 0 recall).

### The precision/recall trade-off
Tightening the decision threshold raises precision but lowers recall, and vice versa.
- **Optimize recall** when misses are costly: cancer screening, fraud, safety filters.
- **Optimize precision** when false alarms are costly: spam that blocks real email, flagging content for human review at scale.

---

## 3. Hands-on

### 3.1 From scratch

```python
def prf(y_true, y_pred, positive=1):
    tp = sum(t == positive and p == positive for t, p in zip(y_true, y_pred))
    fp = sum(t != positive and p == positive for t, p in zip(y_true, y_pred))
    fn = sum(t == positive and p != positive for t, p in zip(y_true, y_pred))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall    = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}

print(prf([1,1,0,1,0], [1,0,0,1,1]))
```

### 3.2 With scikit-learn (what you'd use in practice)

```python
from sklearn.metrics import (precision_score, recall_score, f1_score,
                             classification_report, confusion_matrix)

y_true = [1, 1, 0, 1, 0, 0, 1]
y_pred = [1, 0, 0, 1, 1, 0, 1]

print(precision_score(y_true, y_pred))
print(recall_score(y_true, y_pred))
print(f1_score(y_true, y_pred))
print(confusion_matrix(y_true, y_pred))
print(classification_report(y_true, y_pred))  # per-class + macro/micro avg

# multi-class: choose an averaging strategy
f1_score(y_true, y_pred, average="macro")   # unweighted mean over classes
f1_score(y_true, y_pred, average="micro")   # global TP/FP/FN
f1_score(y_true, y_pred, average="weighted")# weighted by class support
```

---

## 4. Applying this to *generative* AI (the harder part)

Precision/recall need discrete labels; generation is open-ended. You bridge the gap by framing sub-tasks:

- **RAG retrieval** → precision@k / recall@k over retrieved chunks (did we fetch the relevant ones?). See Lesson 10.
- **Classification-style LLM tasks** (intent, sentiment, ticket category) → standard precision/recall on the predicted labels.
- **Extraction/NER** → per-field precision/recall (did we extract the right entities?).
- **Safety/guardrails** → precision/recall of the "unsafe" classifier.
- **Open-ended text** → precision/recall don't apply directly; use **BERTScore**, **LLM-as-judge**, faithfulness/relevancy (Lesson 10), or human eval.

### Retrieval precision@k / recall@k

```python
def precision_recall_at_k(retrieved, relevant, k):
    retrieved_k = retrieved[:k]
    hits = len(set(retrieved_k) & set(relevant))
    precision = hits / k if k else 0.0
    recall = hits / len(relevant) if relevant else 0.0
    return precision, recall
```

---

## 5. Beyond P/R — the fuller toolkit

- **PR-AUC / ROC-AUC** — threshold-independent quality (PR-AUC better for imbalance).
- **Confusion matrix** — see *where* errors happen, not just aggregate.
- **Ranking metrics** — MRR, nDCG for search/recommendation.
- **Regression** — MAE/RMSE for numeric outputs.
- **Calibration** — do predicted probabilities match observed frequencies?

---

## 6. Real-time / production notes

- **Pick the metric that matches business cost** — don't default to accuracy on imbalanced data.
- **Threshold tuning:** choose the operating point on the PR curve that fits the cost of FP vs FN.
- **Slice metrics:** report per-segment (language, tenant, category) to catch hidden failures.
- **Track over time:** metrics on a versioned eval set in CI; watch for drift.
- **Human-in-the-loop:** sample and label production traffic to keep ground truth fresh.

---

## 7. Interview script

"I start from the confusion matrix. Precision is, of what I flagged positive, how much was correct — it punishes false alarms. Recall is, of the actual positives, how many I caught — it punishes misses. F1 balances them, and I avoid plain accuracy on imbalanced data. Which I optimize depends on cost: recall for fraud or safety where misses hurt, precision where false positives are expensive. In code I use sklearn's precision/recall/f1 with macro or weighted averaging for multi-class. For generative systems I frame measurable sub-tasks — precision@k and recall@k for retrieval, per-field precision/recall for extraction, label precision/recall for classification tasks — and for open-ended text I switch to LLM-as-judge, faithfulness, and human eval since precision/recall don't apply directly."
