# Product Attribute Generation — Interview Explainer (revise card)

> Practical way to explain the two-stage attribute enrichment feature confidently.

---

## ONE HONEST FRAMING NOTE FIRST (read this)

You didn't build the Java backend — the team did before you joined. **Don't claim you architected it from scratch.** If pressed, the safe, confident framing is:

> "This feature existed in our platform and I work with / on the GenAI side of it. Let me explain how it works end to end."

You *understand* it, you work *around* it — that's enough to explain it confidently without overclaiming. If he digs into a Java implementation detail you don't know, say: *"That part was built by the backend team before I joined; my focus is the LLM/prompt and mapping logic — here's how that works…"* Honesty + clear scope = senior, not weak.

---

## THE PROBLEM (say this first — always start with WHY)

> "When a customer adds a product, they usually give only 4–5 basic attributes — brand, model, size, length — plus a description and images. That's not enough for buyers who want full detail before purchasing. So we use an LLM to **enrich** each product with 20+ additional attributes, automatically."

---

## THE TWO STAGES (the core answer)

**Stage 1 — GENERATE (enrichment):**
> "An LLM takes the available product data — the few existing attributes, the title, the description, and the category — and generates 20+ relevant additional attributes as structured key–value pairs. For a laptop it might infer RAM type, screen refresh rate, weight, ports, warranty, and so on."

**Stage 2 — MAP & DEDUPLICATE (normalization):**
> "The generated attributes are then matched against our existing attribute catalog. If a generated attribute already exists — maybe under a slightly different name like 'Display Size' vs 'Screen Size' — we map it to the existing one instead of creating a duplicate. Only genuinely new attributes get added. This keeps the catalog clean and consistent."

> **The mental model:** Stage 1 = *create*. Stage 2 = *reconcile with what already exists*. Generation without mapping would flood the catalog with duplicates — that's the whole reason for two stages.

---

## WHY TWO STAGES / "WHY TWO LLMs?" (handle this carefully)

Be precise — it's **two stages**, not necessarily two different LLMs. Say:

> "It's a two-STAGE pipeline. Generation and mapping are separate concerns, so we keep them as separate steps. Generation is a creative task; mapping is a matching/normalization task. Separating them means each step has a focused prompt and is independently testable and debuggable — and we can use a cheaper model or even non-LLM matching for the mapping step."

**If he asks "why not one prompt that does both?":**
> "You could, but combining them makes the prompt overloaded and the output harder to validate. Separating gives cleaner, more reliable results and lets us swap the mapping strategy without touching generation."

---

## HOW THE MAPPING WORKS (3 honest options — pick the one you're sure of)

You may not know the exact internal method. Give the realistic options:

1. **Embedding similarity (most likely / strongest answer):**
   > "Each existing attribute name has an embedding. We embed each generated attribute and compare with cosine similarity — if it's above a threshold against an existing attribute, we map to that existing one; otherwise it's new."

2. **LLM-based matching:**
   > "A second LLM step is given the generated attribute plus the list of existing attributes and asked 'does this match any existing one?'"

3. **Rule/string matching (simplest):**
   > "Normalized string matching — lowercase, synonyms, fuzzy match — for exact and near-exact names."

> Safe line: *"The robust way is embedding-based semantic matching with a similarity threshold, so 'Display Size' and 'Screen Size' map together even though the strings differ."*

---

## INPUTS CONSIDERED WHEN GENERATING (have this ready)

- Existing attributes (brand, model, size, length…)
- Product **title / name**
- Product **description**
- **Category** (so attributes are category-appropriate — a laptop vs a shirt)
- *(Optionally)* images via a vision model — say "we could extend with vision to read attributes off product images" as a forward-looking point.

---

## END-TO-END FLOW (draw / narrate this)

```
Customer adds product (4-5 attrs + title + description + category)
        |
        v
[STAGE 1: GENERATE]  LLM -> 20+ candidate attributes (key-value, structured)
        |
        v
[STAGE 2: MAP/DEDUPE]  for each candidate:
        |   - compare to existing catalog attributes (embedding/LLM/rule)
        |   - match found?  -> map to existing attribute
        |   - no match?     -> add as NEW attribute
        v
Enriched product (existing + mapped + new unique attributes)
        |
        v
Saved to catalog -> richer product page for buyers
```

---

## STRUCTURED-OUTPUT POINT (drop this — sounds senior)

> "Because the attributes need to be machine-usable, Stage 1 returns **structured output** — JSON key–value pairs, not free text. We validate the schema before mapping, so a malformed LLM response doesn't corrupt the catalog."

---

## LIKELY FOLLOW-UPS → ONE-LINE ANCHORS

- **"How do you prevent duplicates?"** → "Stage 2 maps generated attrs to existing ones via semantic similarity; only unmatched ones are added."
- **"How do you stop the LLM hallucinating attributes?"** → "Generation is grounded in the product's title/description/category; we validate structure, and a human or rule check can gate low-confidence values."
- **"What if it generates a wrong value?"** → "Confidence thresholds + optional human review for new/low-confidence attributes before they go live."
- **"Why not just let users fill everything?"** → "Friction — users won't enter 25 fields. AI enrichment improves catalog quality without burdening the seller."
- **"How do you evaluate it?"** → "Sample audits against ground-truth products; precision of mapping (did we map correctly?) and duplicate-rate reduction."
- **"Cost at scale?"** → "Batch generation, cache by product category, use a smaller model for mapping or replace it with pure embedding matching."

---

## THE 30-SECOND SPOKEN VERSION (memorize this)

> "When a customer lists a product, they give only a few attributes. We use a two-stage LLM pipeline to enrich it. Stage one takes the title, description, category, and existing attributes and generates 20-plus relevant attributes as structured key-value pairs. Stage two maps those against our existing attribute catalog using semantic similarity — if an attribute already exists, even under a different name, we map to it; only truly new ones are added. That prevents duplicates and keeps the catalog clean and consistent, so buyers get richer product detail without the seller doing extra work."

---

**Revise the 30-second version + the end-to-end flow diagram. That covers 90% of what he'll ask.**
