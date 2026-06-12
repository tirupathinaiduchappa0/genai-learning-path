# It means that the model's internal reasoning process is opaque and not directly interpretable. A neural network's decision-making process is distributed across millions or billions of numerical weights. It's impossible to look at the model and understand why it made a specific decision in a human-readable way, making its internal logic a "black box."

# why tokens for 10-10-2026 date is lesss when compareed with October, 10 2026
# In a diffusion-based image generation model, what is the purpose of the seed parameter?
#What is the primary goal of applying "quantization" to a language model?


#In AI and RAG, "black box" simply means:
#You judge a system only by its inputs and outputs, without caring about (or looking at) what happens inside.



#White Box Evaluation

#You inspect the internals.

#For a RAG system:

# Question
#    ↓
# Retriever
#    ↓
# Top-k Documents
#    ↓
# Re-ranker
#    ↓
# Prompt
#    ↓
# LLM
#    ↓
# Answer

# Now you measure things like:

# Retrieval Recall@K
# Precision@K
# Context relevance
# Chunk quality
# Hallucination rate
# Re-ranking accuracy

# You're opening the box and examining each component.


# Why AI people say "black box" so often

# Because many stakeholders don't care about internals.

# Product managers care about user satisfaction.
# Customers care about answer quality.
# Business teams care about cost and latency.

# They evaluate the system as a black box.

# Meanwhile:

# ML engineers
# RAG engineers
# Researchers

# often perform white-box evaluations to find the root cause of failures.

# A common saying is:

# Black-box evaluation tells you that something is wrong. White-box evaluation tells you why it's wrong.

# That's why in RAG projects you'll often see both:

# Component-level (white-box) evaluation
# End-to-end (black-box) evaluation
#Black-box evaluation tells you that something is wrong. White-box evaluation tells you why it's wrong

# MRR and NDCG are retrieval evaluation metrics used in RAG to measure how good your retriever is at bringing relevant documents/chunks to the top.MRR (Mean Reciprocal Rank)

# MRR measures:

# How high the first relevant document appears in the ranking.

# Formula:

#                     1
# RR  =           -------------------
#                Rank of first relevant result


# MRR=Average of RR across all queries
# Limitation

# MRR only cares about the first relevant result.

# NDCG (Normalized Discounted Cumulative Gain)

# NDCG is more sophisticated.

# It measures:

# How well all relevant documents are ranked, giving higher importance to top positions.

# Unlike MRR:

# Uses multiple relevant documents.
# Supports different relevance levels.
# Rewards putting highly relevant chunks near the top.


# An API defines how one software application communicates with another. MCP (Model Context Protocol) is a standardized way for AI models to connect to external tools, data sources, and services. In simple terms, APIs are the individual services, while MCP provides a common protocol that lets AI models discover and use those services consistently."

# One-liner:

# API = a specific interface to a service; MCP = a standard protocol that helps AI models interact with many services/tools in a unified way.


# Regularization is the process of adding constraints or penalties during training so that the model learns general patterns instead of memorizing the training data.