"""
===================================================================================
LESSON 2 — TRANSFORMERS, ATTENTION & LLM INTERNALS  (Senior DS Real-Time Deep-Dive)
===================================================================================

COVERS (Section B of the question bank):
  B1. How an LLM works end-to-end  (tokens -> embeddings -> attention -> next token)
  B2. Masked (causal) self-attention  (the -inf-before-softmax trick)
  B3. Multi-head self-attention & the full Transformer architecture
  B4. Attention mechanisms compared  (self vs cross, causal vs bidirectional, ...)

HOW TO READ:
  THEORY (why) -> RUNNABLE NUMPY (proof of the mechanism) -> INTERVIEW ANSWER
  (the soundbite) -> RELATED CONCEPTS (their follow-ups).

  Run:  & "C:\\Projects\\Gen AI Udemy\\LangCGS\\.venv\\Scripts\\python.exe" "02_transformers_attention_llm_internals.py"

  NOTE: the demos use tiny RANDOM (untrained) weights. They prove the MECHANICS
  and shapes of attention/Transformers — not learned language. That's exactly what
  an interviewer wants to see you understand.
===================================================================================
"""

from __future__ import annotations

import sys

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

np.random.seed(7)
np.set_printoptions(precision=3, suppress=True)


def banner(title: str) -> None:
    line = "=" * 83
    print("\n" + line + "\n" + title + "\n" + line)


def sub(title: str) -> None:
    print("\n" + "━" * 79 + "\n" + title + "\n" + "━" * 79)


# ===================================================================================
# CORE MATH — the building blocks every demo below reuses
# ===================================================================================

def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically-stable softmax: subtract the max before exponentiating."""
    x = x - np.max(x, axis=axis, keepdims=True)   # stability: avoids overflow
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def scaled_dot_product_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                                 mask: np.ndarray | None = None):
    """The heart of the Transformer.  Attention(Q,K,V) = softmax(QKᵀ / √dₖ) V

    Q: (seq_q, d_k)   K: (seq_k, d_k)   V: (seq_k, d_v)
    mask: (seq_q, seq_k) with 0 where allowed, -inf where forbidden (added to scores).
    Returns (output (seq_q, d_v), attention_weights (seq_q, seq_k)).
    """
    d_k = Q.shape[-1]
    scores = Q @ K.T / np.sqrt(d_k)   # (seq_q, seq_k) — similarity of each query to each key
    if mask is not None:
        scores = scores + mask         # -inf entries -> ~0 after softmax
    weights = softmax(scores, axis=-1) # each row sums to 1: "how much to attend"
    output = weights @ V               # weighted sum of value vectors
    return output, weights


def sinusoidal_positional_encoding(seq_len: int, d_model: int) -> np.ndarray:
    """Original Transformer positional encoding (sine/cosine of varying frequency).

    Attention itself is ORDER-BLIND (permuting inputs permutes outputs identically),
    so we must inject position. PE[pos, 2i]=sin(pos/10000^(2i/d)), odd idx uses cos.
    """
    pe = np.zeros((seq_len, d_model))
    pos = np.arange(seq_len)[:, None]
    div = np.power(10000.0, np.arange(0, d_model, 2) / d_model)
    pe[:, 0::2] = np.sin(pos / div)
    pe[:, 1::2] = np.cos(pos / div)
    return pe


def layer_norm(x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """Normalize across the feature dim — stabilizes training, standard in Transformers."""
    mu = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps)


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


# ===================================================================================
# B1. HOW AN LLM WORKS END-TO-END  (input text -> generated text)
# ===================================================================================
#
# THE FULL PIPELINE (say these 6 stages in order — this is the money answer):
#
#   1) TOKENIZATION
#      Raw text -> tokens (integer IDs). A "token" is a sub-word unit, not a word:
#      BPE/WordPiece/SentencePiece merge frequent character pairs, so "tokenization"
#      might be ["token","ization"]. Sub-words balance vocab size vs. handling rare
#      words / typos / any language. Output: a list of int IDs.
#
#   2) EMBEDDINGS (+ POSITION)
#      Each token ID indexes a learned embedding matrix (vocab_size x d_model) ->
#      a dense vector capturing meaning. Attention is order-blind, so we ADD
#      positional information (sinusoidal, or learned/RoPE/ALiBi in modern LLMs).
#      Output: matrix (seq_len x d_model).
#
#   3) SELF-ATTENTION
#      Each token builds Query/Key/Value vectors and attends to every other token:
#      softmax(QKᵀ/√dₖ)·V. This is how the model mixes context — "it" learns to look
#      at the noun it refers to. In an LLM (decoder) this attention is CAUSAL (B2).
#
#   4) TRANSFORMER LAYERS (stacked N times)
#      Each block = Multi-Head Attention -> Add&Norm (residual + layernorm) ->
#      Feed-Forward Network (two linear layers + nonlinearity) -> Add&Norm.
#      Stacking many blocks (GPT-3: 96) builds deep, abstract representations.
#
#   5) NEXT-TOKEN PREDICTION (the LM head)
#      The final hidden vector of the LAST position is projected to vocab-size logits
#      (often tied to the embedding matrix). Softmax -> probability over the vocab.
#
#   6) SAMPLING + AUTOREGRESSIVE LOOP
#      Pick the next token (greedy / temperature / top-k / top-p), APPEND it to the
#      input, and repeat from step 1 until <eos> or max length. "Autoregressive" =
#      each new token conditions on all previous tokens.
#
#   KEY TAKEAWAY: an LLM is a next-token probability machine run in a loop. All the
#   intelligence lives in the learned embeddings + attention + FFN weights.


def _demo_llm_pipeline() -> None:
    sub("B1. TOY END-TO-END LLM PIPELINE (untrained — shows the mechanism)")

    # --- 1) Tokenization (toy word-level vocab) ---
    vocab = ["<pad>", "the", "cat", "sat", "on", "mat", "dog", "ran", "<eos>"]
    stoi = {w: i for i, w in enumerate(vocab)}
    itos = {i: w for w, i in stoi.items()}
    prompt = ["the", "cat", "sat"]
    ids = [stoi[w] for w in prompt]
    print(f"1) tokenize {prompt} -> ids {ids}")

    # --- Model params (random, untrained) ---
    d_model, V = 16, len(vocab)
    E = np.random.randn(V, d_model) * 0.1        # token embedding table
    W_out = np.random.randn(d_model, V) * 0.1    # LM head (project hidden -> vocab logits)
    Wq = np.random.randn(d_model, d_model) * 0.1
    Wk = np.random.randn(d_model, d_model) * 0.1
    Wv = np.random.randn(d_model, d_model) * 0.1
    W1 = np.random.randn(d_model, 4 * d_model) * 0.1  # FFN up-projection
    W2 = np.random.randn(4 * d_model, d_model) * 0.1  # FFN down-projection

    def forward_last_logits(seq_ids: list[int]) -> np.ndarray:
        # 2) embeddings + positional encoding
        x = E[seq_ids] + sinusoidal_positional_encoding(len(seq_ids), d_model)
        # 3) causal self-attention (single head here; multi-head shown in B3)
        Q, K, Vv = x @ Wq, x @ Wk, x @ Wv
        mask = np.triu(np.full((len(seq_ids), len(seq_ids)), -np.inf), k=1)  # causal
        attn, _ = scaled_dot_product_attention(Q, K, Vv, mask=mask)
        x = layer_norm(x + attn)                       # 4) Add & Norm
        ffn = relu(x @ W1) @ W2                         #    Feed-Forward
        x = layer_norm(x + ffn)                         #    Add & Norm
        # 5) LM head on the LAST position -> vocab logits
        return x[-1] @ W_out

    def sample(logits: np.ndarray, temperature: float = 1.0, top_p: float = 1.0) -> int:
        # temperature: <1 sharpens (more deterministic), >1 flattens (more random)
        probs = softmax(logits / max(temperature, 1e-6))
        if top_p < 1.0:  # nucleus: keep smallest set of tokens whose prob mass >= top_p
            order = np.argsort(probs)[::-1]
            cum = np.cumsum(probs[order])
            keep = order[:np.searchsorted(cum, top_p) + 1]
            mask = np.zeros_like(probs); mask[keep] = probs[keep]
            probs = mask / mask.sum()
        return int(np.random.choice(len(probs), p=probs))

    print("2) embed + add positional encoding -> matrix shape",
          (E[ids] + sinusoidal_positional_encoding(len(ids), d_model)).shape)
    logits = forward_last_logits(ids)
    print("3-5) attention + FFN + LM head -> vocab logits shape", logits.shape)
    print("     next-token probs:", softmax(logits))

    # 6) autoregressive generation loop
    gen = list(ids)
    for _ in range(4):
        nxt = sample(forward_last_logits(gen), temperature=0.8, top_p=0.9)
        gen.append(nxt)
        if itos[nxt] == "<eos>":
            break
    print("6) generated (untrained, random):", [itos[i] for i in gen])


# INTERVIEW ANSWER (B1):
#   "An LLM is a next-token predictor run in a loop. Text is tokenized into sub-word
#    IDs, each ID maps to a learned embedding plus positional info, then a stack of
#    Transformer blocks — multi-head self-attention + feed-forward with residuals and
#    layernorm — mixes context across tokens. The final position's hidden state is
#    projected to vocabulary logits, softmaxed to a probability distribution, and we
#    sample the next token, append it, and repeat autoregressively until <eos>."
#
# RELATED CONCEPTS: BPE/SentencePiece; tied embeddings; RoPE/ALiBi positions; KV-cache
#   (cache past K,V so generation is O(1) per new token, not O(n)); context window;
#   logits -> temperature/top-k/top-p; teacher forcing during training.


# ===================================================================================
# B2. MASKED (CAUSAL) SELF-ATTENTION
# ===================================================================================
#
# WHAT & WHY
#   In a decoder LLM, when predicting token t we must use ONLY tokens 1..t — never
#   t+1..n (those are the "answers" during training). Masked self-attention enforces
#   this: position t is forbidden from attending to future positions.
#
# WHERE
#   In the DECODER's self-attention (GPT, LLaMA — all decoder-only). Encoders (BERT)
#   use UNMASKED (bidirectional) attention because they see the whole sentence at once.
#
# THE MATH (this is the exact thing they want)
#   scores = QKᵀ / √dₖ                      # (seq, seq)
#   add a mask that is 0 on/below the diagonal and -∞ ABOVE it (future positions)
#   weights = softmax(scores + mask)        # e^(-∞) = 0  -> future gets ZERO weight
#   -> BEFORE softmax we add -∞, so after softmax those probabilities are exactly 0.
#   This is why a position literally cannot "see" the future — its attention weight
#   on future tokens is zero, so they contribute nothing to its output.
#
# WHY CRITICAL FOR GENERATION
#   Training uses "teacher forcing" on full sequences in parallel; the causal mask is
#   what makes each position's prediction depend only on the past, matching how the
#   model is actually used at inference (one token at a time). Without it, the model
#   would trivially cheat by copying the next token.


def _demo_causal_mask() -> None:
    sub("B2. CAUSAL MASKING (runnable — future weights become exactly 0)")
    seq, d = 4, 8
    x = np.random.randn(seq, d)
    Q, K, V = x, x, x  # tie for simplicity

    _, unmasked = scaled_dot_product_attention(Q, K, V)
    mask = np.triu(np.full((seq, seq), -np.inf), k=1)  # -inf strictly above diagonal
    _, masked = scaled_dot_product_attention(Q, K, V, mask=mask)

    print("Unmasked attention weights (each row attends to ALL positions):")
    print(unmasked)
    print("\nCausal-masked weights (upper triangle is 0 — cannot look ahead):")
    print(masked)
    print("Row 0 attends only to token 0; row 3 attends to tokens 0..3. ✔")


# INTERVIEW ANSWER (B2):
#   "Causal masking makes position t attend only to positions ≤ t. In code you add a
#    mask to the raw scores that's 0 on/below the diagonal and −∞ above it, THEN apply
#    softmax — e^(−∞) is 0, so future tokens get exactly zero attention weight. It
#    lives in the decoder's self-attention and it's what lets us train on full
#    sequences in parallel while preserving left-to-right, autoregressive behavior."
#
# RELATED CONCEPTS: teacher forcing; padding masks (ignore <pad>) vs causal masks;
#   bidirectional (BERT) vs causal (GPT); prefix-LM; why the KV-cache is consistent
#   with causality (you only ever cache the past).


# ===================================================================================
# B3. MULTI-HEAD SELF-ATTENTION & THE FULL TRANSFORMER ARCHITECTURE
# ===================================================================================
#
# SELF-ATTENTION BASICS (Q, K, V)
#   Each token is projected into a Query ("what am I looking for"), Key ("what do I
#   offer"), and Value ("what I actually pass on"). A token's output = weighted sum
#   of all Values, weighted by Query·Key similarity. That's how context flows.
#
# WHAT MULTI-HEAD ADDS (and why not just one big head)
#   Split d_model into h heads, each with its own Wq/Wk/Wv in a smaller subspace.
#   Each head learns a DIFFERENT relationship (syntax, coreference, position, topic)
#   in parallel. One head can only average one kind of relationship; multiple heads
#   attend to different things at different positions simultaneously.
#
# HOW HEADS COMBINE
#   Run attention per head -> CONCATENATE the head outputs -> multiply by an output
#   projection Wₒ to mix them back to d_model. (concat then linear = "combine views".)
#
# THE FULL ARCHITECTURE
#   Encoder block:  MHA (bidirectional) -> Add&Norm -> FFN -> Add&Norm
#   Decoder block:  MASKED MHA -> Add&Norm -> CROSS-ATTENTION (to encoder) -> Add&Norm
#                   -> FFN -> Add&Norm
#   - Positional encoding: injected at the input (attention is order-blind).
#   - Feed-forward: per-position 2-layer MLP (expand ~4x then contract) — adds capacity.
#   - Residual connections: x + sublayer(x) — gradients flow, deep stacks train.
#   - Layer normalization: stabilizes activations (pre-norm in modern LLMs).
#   Encoder-decoder (T5, translation) vs decoder-only (GPT/LLaMA) vs encoder-only (BERT).
#
# WHAT IT SOLVED vs RNN/LSTM
#   RNNs process sequentially (slow, no parallelism) and struggle with long-range
#   dependencies (vanishing gradients, info bottleneck through one hidden state).
#   Attention connects any two tokens in ONE step (path length O(1)) and is fully
#   parallelizable across the sequence -> longer context + massive training speedups.


def multi_head_attention(x: np.ndarray, n_heads: int, causal: bool = True):
    """Full multi-head self-attention on x (seq, d_model). Returns (out, per-head weights)."""
    seq, d_model = x.shape
    assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
    d_head = d_model // n_heads

    Wq = np.random.randn(d_model, d_model) * 0.1
    Wk = np.random.randn(d_model, d_model) * 0.1
    Wv = np.random.randn(d_model, d_model) * 0.1
    Wo = np.random.randn(d_model, d_model) * 0.1

    Q, K, V = x @ Wq, x @ Wk, x @ Wv
    mask = np.triu(np.full((seq, seq), -np.inf), k=1) if causal else None

    head_outputs, head_weights = [], []
    for h in range(n_heads):
        s = slice(h * d_head, (h + 1) * d_head)           # this head's subspace
        out_h, w_h = scaled_dot_product_attention(Q[:, s], K[:, s], V[:, s], mask)
        head_outputs.append(out_h)
        head_weights.append(w_h)

    concat = np.concatenate(head_outputs, axis=-1)         # (seq, d_model)
    return concat @ Wo, head_weights                        # output projection mixes heads


def _demo_multi_head() -> None:
    sub("B3. MULTI-HEAD ATTENTION + TRANSFORMER BLOCK (runnable)")
    seq, d_model, n_heads = 5, 16, 4
    x = np.random.randn(seq, d_model)

    mha_out, head_weights = multi_head_attention(x, n_heads=n_heads, causal=True)
    print(f"input (seq={seq}, d_model={d_model}), n_heads={n_heads}, "
          f"d_head={d_model // n_heads}")
    print(f"multi-head output shape: {mha_out.shape}  (back to d_model)")
    print(f"each of the {n_heads} heads has its own {seq}x{seq} attention pattern; "
          f"head 0 row sums: {head_weights[0].sum(axis=1)}")

    # One full pre-norm Transformer block: x -> +MHA -> +FFN
    h = x + mha_out
    h = layer_norm(h)
    W1 = np.random.randn(d_model, 4 * d_model) * 0.1
    W2 = np.random.randn(4 * d_model, d_model) * 0.1
    h = layer_norm(h + relu(h @ W1) @ W2)
    print(f"after full block (MHA + Add&Norm + FFN + Add&Norm): {h.shape}")


# INTERVIEW ANSWER (B3):
#   "Self-attention projects each token to Query/Key/Value; a token's output is the
#    Value-weighted sum of all tokens, weighted by Query·Key similarity. Multi-head
#    splits the model dimension into several heads so each learns a different relation
#    in parallel; we concat the heads and apply an output projection. A Transformer
#    block wraps multi-head attention and a position-wise feed-forward network, each
#    with a residual connection and layernorm. Versus RNNs it connects any two tokens
#    in one step and is fully parallelizable, which is why it scaled to LLMs."
#
# RELATED CONCEPTS: GQA/MQA (share K,V across heads to shrink KV-cache); pre-norm vs
#   post-norm; FFN as key-value memory; parameter count dominated by FFN + attention.


# ===================================================================================
# B4. ATTENTION MECHANISMS COMPARED
# ===================================================================================
#
#   SELF-ATTENTION vs CROSS-ATTENTION
#     Self:  Q, K, V all come from the SAME sequence (tokens attend to each other).
#     Cross: Q comes from one sequence (decoder), K & V from ANOTHER (encoder output).
#            Used in encoder-decoder models (translation, T5) so the decoder can look
#            at the source sentence while generating the target.
#
#   MASKED (CAUSAL) vs BIDIRECTIONAL
#     Causal:        each token sees only past tokens. Decoders / generation (GPT).
#     Bidirectional: each token sees the whole sequence. Encoders / understanding (BERT).
#
#   SINGLE-HEAD vs MULTI-HEAD
#     Single averages ONE relationship; multi-head learns several in parallel subspaces
#     then concatenates. Multi-head is standard everywhere.
#
#   WHERE EACH LIVES
#     Encoder-only (BERT): bidirectional self-attention.  Task: classification/embeddings.
#     Decoder-only (GPT/LLaMA): causal self-attention.     Task: generation.
#     Encoder-decoder (T5): encoder self-attn + decoder masked self-attn + cross-attn.
#
#   EFFICIENCY VARIANTS (why they exist)
#     Full attention is O(n²) in sequence length -> memory/compute explode on long
#     context. Variants trade a little accuracy for scale:
#       - Sparse / windowed (Longformer, BigBird): attend to a local window + a few
#         global tokens.
#       - FlashAttention: exact, but IO-aware — fuses ops to avoid materializing the
#         n×n matrix (faster + less memory, same math).
#       - Linear/low-rank (Performer, Linformer): approximate softmax attention.


def _demo_cross_attention_shapes() -> None:
    sub("B4. CROSS-ATTENTION SHAPES (decoder queries attend to encoder keys/values)")
    dec_len, enc_len, d = 3, 6, 8   # decoder length != encoder length is fine
    Q = np.random.randn(dec_len, d)  # from decoder
    K = np.random.randn(enc_len, d)  # from encoder
    V = np.random.randn(enc_len, d)  # from encoder
    out, w = scaled_dot_product_attention(Q, K, V)  # no causal mask in cross-attn
    print(f"decoder Q: {Q.shape}, encoder K/V: {K.shape}")
    print(f"attention weights: {w.shape}  (each of {dec_len} decoder tokens over "
          f"{enc_len} encoder tokens)")
    print(f"output: {out.shape}  (decoder-length, model-dim)")


# INTERVIEW ANSWER (B4):
#   "Self-attention has Q,K,V from one sequence; cross-attention has queries from the
#    decoder and keys/values from the encoder, so the decoder can read the source.
#    Causal attention (decoders) sees only the past; bidirectional (encoders) sees
#    everything. Multi-head beats single-head by learning several relations in
#    parallel. Because full attention is O(n²), long-context models use sparse/
#    windowed patterns or IO-aware exact kernels like FlashAttention."
#
# RELATED CONCEPTS: attention as soft dictionary lookup; O(n²) bottleneck; RAG as an
#   alternative to giant context windows; retrieval + attention complementarity.


# ===================================================================================
# GOLDEN LESSONS
# ===================================================================================
GOLDEN_LESSONS = """
  1. LLM = next-token predictor in a loop: tokenize -> embed(+pos) -> attention -> logits -> sample.
  2. A token is a SUB-WORD unit (BPE), not a word. Embeddings give meaning; positions give order.
  3. Attention = softmax(QKᵀ/√dₖ)·V. Q=what I want, K=what I offer, V=what I pass on.
  4. √dₖ scaling keeps scores from blowing up so softmax stays in a usable gradient range.
  5. Causal mask = add −∞ ABOVE the diagonal BEFORE softmax -> future weight is exactly 0.
  6. Multi-head = several relations in parallel subspaces; concat then output-project.
  7. Block = MHA -> Add&Norm -> FFN -> Add&Norm. Residuals + layernorm make deep stacks trainable.
  8. Self vs cross; causal (GPT/decoder) vs bidirectional (BERT/encoder). Know which model uses which.
  9. Full attention is O(n²) -> FlashAttention (exact) / sparse-windowed (approx) for long context.
 10. KV-cache makes generation O(1) per new token — a favorite senior follow-up.
"""


if __name__ == "__main__":
    banner("LESSON 2 — TRANSFORMERS, ATTENTION & LLM INTERNALS")
    _demo_llm_pipeline()
    _demo_causal_mask()
    _demo_multi_head()
    _demo_cross_attention_shapes()
    banner("GOLDEN LESSONS")
    print(GOLDEN_LESSONS)
