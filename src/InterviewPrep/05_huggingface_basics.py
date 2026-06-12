"""
GenAI Interview Prep - Lesson 5: Hugging Face Basics

WHY THIS LESSON EXISTS:
    You hosted DocSage on Hugging Face Spaces. Interviewers WILL ask:
    "What is Hugging Face? Why did you choose it over Streamlit Cloud?"

TABLE OF CONTENTS:
    1. What is Hugging Face
    2. Key Components
    3. Hugging Face Spaces (where DocSage is hosted)
    4. Why HF over Streamlit Cloud
    5. HF in Your Project
    6. Interview Q&A (10 questions)

Author: GenAI Learner
"""


# ==============================================================================
# 1. WHAT IS HUGGING FACE
# ==============================================================================
#
# ONE-LINE: Hugging Face is the "GitHub of AI" — a platform where the AI
# community shares models, datasets, and applications.
#
#   GitHub       = developers share CODE
#   Hugging Face = AI developers share MODELS, DATASETS, and AI APPS
#
# Founded 2016. Used by Google, Meta, Microsoft, Amazon, NVIDIA.
# 500,000+ models, 100,000+ datasets, 300,000+ AI apps on the platform.
# It's the #1 platform in the AI/ML ecosystem.
#
# INTERVIEW ANSWER:
#   "Hugging Face is the leading open-source AI platform — like GitHub but
#   for AI. It hosts models, datasets, and applications. I use it for
#   hosting my app, accessing open-source models, and using their embedding
#   models in my RAG pipeline."


# ==============================================================================
# 2. KEY COMPONENTS
# ==============================================================================
#
# a) HF HUB — Central repository for models, datasets, and apps.
#    Anyone can upload and share. 500K+ models available.
#
# b) TRANSFORMERS LIBRARY — Most popular AI library (140M+ downloads/month).
#    Unified API for ALL transformer models:
#      from transformers import pipeline
#      classifier = pipeline("sentiment-analysis")
#      classifier("I love this!")  # {'label': 'POSITIVE', 'score': 0.99}
#
# c) DATASETS LIBRARY — Easy access to thousands of datasets:
#      from datasets import load_dataset
#      dataset = load_dataset("imdb")
#
# d) SPACES — Free hosting for AI apps (Streamlit, Gradio, Docker).
#    Each Space gets a public URL. This is where DocSage lives.
#
# e) SENTENCE-TRANSFORMERS — Text embedding library (what we use!).
#    We use: sentence-transformers/all-MiniLM-L6-v2 for document embeddings.
#
# f) INFERENCE API — Call any model via API without downloading:
#    POST https://api-inference.huggingface.co/models/bert-base
#
# INTERVIEW ANSWER:
#   "HF has the Hub for sharing, transformers library for loading models,
#   Spaces for hosting apps, and sentence-transformers for embeddings.
#   In my project, I use HF embeddings for RAG and Spaces for deployment."


# ==============================================================================
# 3. HUGGING FACE SPACES (Where DocSage Lives)
# ==============================================================================
#
# Free hosting for AI applications. Git-based deployment.
#
# SUPPORTED SDKs:
#   Streamlit (what we use), Gradio (HF's own), Docker, Static HTML
#
# HOW IT WORKS:
#   1. Create a Space on huggingface.co (choose SDK: Streamlit)
#   2. Push code via git (like GitHub)
#   3. HF builds a Docker container automatically
#   4. App gets a public URL: huggingface.co/spaces/username/app-name
#   5. No server management needed
#
# FREE TIER: 2 vCPU, 16GB RAM, public URL, encrypted secrets, auto-restart.
#
# OUR DEPLOYMENT:
#   URL: https://huggingface.co/spaces/tirupathi0/docsage
#   API keys stored as HF Secrets (not in code).
#   Auto-deploys on every git push.
#
# INTERVIEW ANSWER:
#   "I deployed DocSage on HF Spaces. I push code via git, HF builds a
#   container automatically, and the app gets a public URL. API keys are
#   stored as encrypted secrets in Space settings."


# ==============================================================================
# 4. WHY HUGGING FACE OVER STREAMLIT CLOUD
# ==============================================================================
#
# Aspect              | Hugging Face Spaces      | Streamlit Cloud
# --------------------|--------------------------|------------------
# Target audience     | AI/ML community          | General data apps
# Free tier RAM       | 16GB                     | 1GB
# AI ecosystem        | Models + datasets + apps | Just app hosting
# GPU support         | Yes (paid tier)          | No
# Community           | AI recruiters browse HF  | Less AI visibility
# Custom Docker       | Yes                      | No
# Model access        | Direct access to 500K+   | No model hosting
#
# THE REAL REASON:
#   "I chose HF because (1) it's the standard in the AI community —
#   recruiters browse HF to evaluate candidates, (2) 16GB free RAM
#   for loading embedding models and FAISS indexes, (3) it's part of
#   the ecosystem where I also use their sentence-transformers library."


# ==============================================================================
# 5. HUGGING FACE IN YOUR PROJECT
# ==============================================================================
#
# You use HF in THREE ways in DocSage:
#
# 1. EMBEDDINGS: sentence-transformers/all-MiniLM-L6-v2
#    Converts document chunks into vectors for FAISS.
#    Free, runs locally, 90MB model, no API key needed.
#
# 2. HOSTING: Hugging Face Spaces (Streamlit SDK)
#    Public URL for the live demo. Git-based deployment.
#    Secrets for API keys. Auto-restart on crash.
#
# 3. ECOSYSTEM: Part of the broader HF community.
#    Your project is visible to AI recruiters and engineers.
#    Shows you're part of the AI community, not just using APIs.
#
# INTERVIEW ANSWER:
#   "I use Hugging Face in three ways: their sentence-transformers model
#   for document embeddings in my RAG pipeline, Spaces for hosting the
#   live demo, and the broader ecosystem for visibility in the AI community."


# ==============================================================================
# 6. INTERVIEW Q&A (10 Questions)
# ==============================================================================
#
# Q1: What is Hugging Face?
# A: The leading open-source AI platform — like GitHub for AI. Hosts 500K+
#    models, 100K+ datasets, and 300K+ AI applications. Used by Google,
#    Meta, Microsoft, and most AI companies.
#
# Q2: Why did you choose Hugging Face Spaces over Streamlit Cloud?
# A: Three reasons: (1) 16GB free RAM vs 1GB on Streamlit Cloud — needed
#    for embedding models and FAISS. (2) AI community visibility — recruiters
#    browse HF. (3) Part of the same ecosystem as my embedding model.
#
# Q3: What is the transformers library?
# A: HF's Python library for loading and using any transformer model with
#    a unified API. One line to load BERT, GPT-2, LLaMA, or any of 500K+
#    models. 140M+ downloads per month — most popular AI library.
#
# Q4: What are Hugging Face Spaces?
# A: Free hosting for AI applications. Supports Streamlit, Gradio, and
#    Docker. Push code via git, HF builds and deploys automatically.
#    Each Space gets a public URL with encrypted secrets for API keys.
#
# Q5: What embedding model do you use from Hugging Face?
# A: sentence-transformers/all-MiniLM-L6-v2. It's a 90MB model that
#    produces 384-dimensional embeddings. Free, runs locally, good quality
#    for RAG. In production, I'd consider OpenAI embeddings for better
#    accuracy or BAAI/bge-large for a stronger open-source option.
#
# Q6: What is the difference between Hugging Face and OpenAI?
# A: OpenAI provides proprietary models (GPT-4) via paid API — closed source.
#    Hugging Face is a platform for open-source models — anyone can share
#    and use models for free. HF hosts models from Meta (LLaMA), Google
#    (Gemma), Mistral, and thousands of community contributors.
#
# Q7: How do you deploy an app on Hugging Face Spaces?
# A: Create a Space (choose SDK), clone the git repo, add your code with
#    a README.md containing metadata (sdk, app_file), push via git. HF
#    builds a Docker container and deploys automatically. Add API keys
#    as encrypted secrets in Space settings.
#
# Q8: Is Hugging Face free?
# A: The free tier includes: model hosting, dataset hosting, Spaces with
#    2 vCPU and 16GB RAM, Inference API (rate-limited), and community
#    features. Paid tiers add GPU access, private repos, and more resources.
#
# Q9: What is Gradio?
# A: HF's own UI framework for building ML demos. Simpler than Streamlit
#    for quick model demos (input -> model -> output). I chose Streamlit
#    because DocSage needs a chat interface with sidebar, streaming, and
#    session state — which Streamlit handles better.
#
# Q10: How does Hugging Face handle secrets/API keys?
# A: Spaces have encrypted "Repository secrets" in Settings. You add
#     key-value pairs (like GROQ_API_KEY=xxx). HF injects them as
#     environment variables at runtime. They're never exposed in code
#     or logs. Similar to GitHub Actions secrets.
