"""
📋 Lesson 10.5 — Structured Output: Getting JSON/Objects from LLMs

═══════════════════════════════════════════════════════════════════
THE PROBLEM: LLMs Return Unstructured Text
═══════════════════════════════════════════════════════════════════

When you ask an LLM "Tell me about Inception", you get a wall of text.
But in production, you need STRUCTURED data:
    {"title": "Inception", "year": 2010, "director": "Christopher Nolan"}

Without structured output, you'd have to PARSE the text yourself —
fragile, error-prone, and breaks when the LLM changes its format.

THE SOLUTION: with_structured_output()
    model.with_structured_output(MySchema)
    Now the LLM is FORCED to return data matching your schema.
    No parsing needed. Validated automatically.

THREE SCHEMA APPROACHES:
    Approach     | Validation | Features          | When to Use
    -------------|------------|-------------------|---------------------------
    Pydantic     | Full       | Field descriptions| Production (recommended)
                 |            | nested models,    |
                 |            | custom validators |
    TypedDict    | None       | Lightweight,      | Quick prototyping, when
                 |            | dict output       | you don't need validation
    dataclass    | None       | Python-native,    | When you prefer dataclass
                 |            | object output     | syntax over Pydantic

HOW IT WORKS UNDER THE HOOD:
    model.with_structured_output(Movie) does this:
    1. Converts your schema into a JSON Schema
    2. Sends it to the LLM as a "tool" (function calling)
    3. LLM fills in the fields instead of generating free text
    4. LangChain parses the response into your schema object
    5. Returns a validated Pydantic/TypedDict/dataclass instance

WHY THIS MATTERS FOR GENAI:
    - RAG pipelines: Extract structured answers with citations
    - Agents: Get structured tool arguments
    - Data extraction: Pull entities from documents
    - API responses: Return typed JSON from LangServe endpoints

HOW TO RUN:
    $ python 05_structured_output.py

Author: GenAI Learner
Date: 2026-04-14
"""

import logging
import os

from dotenv import load_dotenv

load_dotenv()

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY", "")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def _get_llm():
    from langchain_groq import ChatGroq
    return ChatGroq(model="llama-3.1-8b-instant", temperature=0, max_tokens=512)


# ═══════════════════════════════════════════════════════════════════════════════
# 1️⃣  Pydantic — The Recommended Approach (Full Validation)
# ═══════════════════════════════════════════════════════════════════════════════
#
# Pydantic BaseModel is the RICHEST option:
#   - Field(description="...") tells the LLM what each field means
#   - Type hints enforce data types (str, int, float, list, etc.)
#   - Validators can add custom rules (e.g., rating must be 0-10)
#   - Nested models for complex structures (e.g., Movie with list[Actor])
#   - The LLM reads descriptions to fill fields accurately
#
# This is the SAME Pydantic you already learned in Lesson 13.
# Now you're using it to CONTROL LLM output, not just validate API input.

def demo_pydantic_basic() -> None:
    """Basic Pydantic structured output — flat model."""
    from pydantic import BaseModel, Field

    class Movie(BaseModel):
        """A movie with details."""
        title: str = Field(description="The title of the movie")
        year: int = Field(description="The year the movie was released")
        director: str = Field(description="The director of the movie")
        rating: float = Field(description="The movie's rating out of 10")

    llm = _get_llm()

    # with_structured_output() returns a NEW model that outputs Movie objects
    structured_llm = llm.with_structured_output(Movie)

    # Now .invoke() returns a Movie object, NOT an AIMessage!
    result: Movie = structured_llm.invoke("Provide details about the movie Inception")

    logger.info("--- Pydantic Basic ---")
    logger.info("Type: %s", type(result).__name__)  # Movie (not AIMessage!)
    logger.info("Title: %s", result.title)
    logger.info("Year: %d", result.year)
    logger.info("Director: %s", result.director)
    logger.info("Rating: %.1f", result.rating)

    # You can use it like any Pydantic model
    logger.info("As dict: %s", result.model_dump())
    logger.info("As JSON: %s", result.model_dump_json())


def demo_pydantic_nested() -> None:
    """Nested Pydantic models — complex structures."""
    from pydantic import BaseModel, Field
    from typing import Optional

    class Actor(BaseModel):
        """An actor in a movie."""
        name: str = Field(description="Actor's full name")
        role: str = Field(description="Character name in the movie")

    class MovieDetails(BaseModel):
        """Detailed movie information with cast."""
        title: str = Field(description="Movie title")
        year: int = Field(description="Release year")
        cast: list[Actor] = Field(description="List of main actors and their roles")
        genres: list[str] = Field(description="List of genres")
        budget: Optional[float] = Field(None, description="Budget in millions USD")

    llm = _get_llm()
    structured_llm = llm.with_structured_output(MovieDetails)

    result: MovieDetails = structured_llm.invoke("Provide details about the movie Inception")

    logger.info("--- Pydantic Nested ---")
    logger.info("Title: %s (%d)", result.title, result.year)
    logger.info("Genres: %s", result.genres)
    logger.info("Budget: %s million", result.budget)
    logger.info("Cast (%d actors):", len(result.cast))
    for actor in result.cast[:3]:
        logger.info("  - %s as %s", actor.name, actor.role)


def demo_pydantic_with_raw() -> None:
    """include_raw=True — get both the parsed object AND the raw AIMessage."""
    from pydantic import BaseModel, Field

    class Movie(BaseModel):
        """A movie with details."""
        title: str = Field(description="The title of the movie")
        year: int = Field(description="The year the movie was released")
        director: str = Field(description="The director of the movie")
        rating: float = Field(description="The movie's rating out of 10")

    llm = _get_llm()

    # include_raw=True returns a dict with 'raw', 'parsed', 'parsing_error'
    structured_llm = llm.with_structured_output(Movie, include_raw=True)
    result = structured_llm.invoke("Provide details about the movie Inception")

    logger.info("--- Pydantic with include_raw=True ---")
    logger.info("Keys: %s", list(result.keys()))
    logger.info("parsed: %s", result["parsed"])
    logger.info("parsing_error: %s", result["parsing_error"])
    logger.info("raw type: %s", type(result["raw"]).__name__)
    # raw = the full AIMessage with token usage, metadata, etc.
    # parsed = the Movie object
    # parsing_error = None if successful, error message if parsing failed
    # USE THIS when you need both the structured data AND token usage info.


# ═══════════════════════════════════════════════════════════════════════════════
# 2️⃣  TypedDict — Lightweight, No Validation (Returns dict)
# ═══════════════════════════════════════════════════════════════════════════════
#
# TypedDict is Python's built-in way to define typed dictionaries.
# It's LIGHTER than Pydantic — no runtime validation, no Field descriptions.
# The output is a plain dict (not an object with methods).
#
# WHEN TO USE:
#   - Quick prototyping where you don't need validation
#   - When you want dict output (easier to serialize)
#   - When Pydantic is overkill for simple schemas
#
# LIMITATION:
#   - No Field(description="...") — the LLM gets less guidance
#   - No runtime validation — if the LLM returns wrong types, no error
#   - Use Annotated[str, ..., "description"] for field descriptions

def demo_typeddict() -> None:
    """TypedDict structured output — returns a plain dict."""
    from typing_extensions import Annotated, TypedDict

    class MovieDict(TypedDict):
        """A movie with details."""
        title: Annotated[str, ..., "The title of the movie"]
        year: Annotated[int, ..., "The year the movie was released"]
        director: Annotated[str, ..., "The director of the movie"]
        rating: Annotated[float, ..., "The movie's rating out of 10"]

    llm = _get_llm()
    structured_llm = llm.with_structured_output(MovieDict)

    result: dict = structured_llm.invoke("Provide details about the movie Avengers")

    logger.info("--- TypedDict ---")
    logger.info("Type: %s", type(result).__name__)  # dict (not a class instance!)
    logger.info("Result: %s", result)
    logger.info("Title: %s", result.get("title"))
    logger.info("Year: %s", result.get("year"))
    # Access like a regular dict — result["title"], result.get("year")


# ═══════════════════════════════════════════════════════════════════════════════
# 3️⃣  dataclass — Python-Native, Object Output
# ═══════════════════════════════════════════════════════════════════════════════
#
# Python's @dataclass decorator creates classes with auto-generated
# __init__, __repr__, __eq__, etc. It's between Pydantic and TypedDict:
#   - Returns an OBJECT (not a dict) — you access fields with dot notation
#   - No runtime validation (unlike Pydantic)
#   - Simpler than Pydantic, more structured than TypedDict
#
# NOTE: dataclass with with_structured_output() works with some providers
# but Pydantic is more universally supported. Use Pydantic in production.
#
# The notebook shows dataclass with create_agent(response_format=...) which
# requires OpenAI. Here we show the concept and recommend Pydantic for Groq.

def demo_dataclass_concept() -> None:
    """Explain dataclass structured output (concept + comparison)."""
    from dataclasses import dataclass

    @dataclass
    class ContactInfo:
        """Contact information for a person."""
        name: str   # The name of the person
        email: str  # The email address
        phone: str  # The phone number

    # Dataclass creates objects with dot notation access
    contact = ContactInfo(name="John Doe", email="john@example.com", phone="555-1234")

    logger.info("--- dataclass (concept) ---")
    logger.info("Type: %s", type(contact).__name__)
    logger.info("Name: %s", contact.name)
    logger.info("Email: %s", contact.email)
    logger.info("As dict: %s", contact.__dict__)

    # NOTE: with_structured_output(ContactInfo) using @dataclass works with
    # some providers (OpenAI via create_agent with response_format=ContactInfo).
    # For Groq, use Pydantic BaseModel instead — it's the safest choice.
    logger.info("For LLM structured output, Pydantic is recommended over dataclass.")


# ═══════════════════════════════════════════════════════════════════════════════
# 4️⃣  Production Pattern: Data Extraction Pipeline
# ═══════════════════════════════════════════════════════════════════════════════
#
# Real-world use case: Extract structured data from unstructured text.
# This is one of the MOST COMMON GenAI production patterns.

def demo_production_extraction() -> None:
    """Production pattern: extract structured data from unstructured text."""
    from pydantic import BaseModel, Field
    from typing import Optional

    class JobPosting(BaseModel):
        """Structured job posting extracted from text."""
        title: str = Field(description="Job title")
        company: str = Field(description="Company name")
        location: str = Field(description="Job location")
        salary_min: Optional[int] = Field(None, description="Minimum salary in USD")
        salary_max: Optional[int] = Field(None, description="Maximum salary in USD")
        skills: list[str] = Field(description="Required technical skills")
        experience_years: Optional[int] = Field(None, description="Years of experience required")

    llm = _get_llm()
    extractor = llm.with_structured_output(JobPosting)

    # Unstructured text — like a job posting from a website
    raw_text = """
    We're hiring a Senior Python Developer at TechCorp in San Francisco.
    The role requires 5+ years of experience with Python, FastAPI, and PostgreSQL.
    Knowledge of Docker and Kubernetes is a plus. Salary range: $150,000 - $200,000.
    """

    result: JobPosting = extractor.invoke(f"Extract job posting details from:\n{raw_text}")

    logger.info("--- Production: Data Extraction ---")
    logger.info("Title: %s", result.title)
    logger.info("Company: %s", result.company)
    logger.info("Location: %s", result.location)
    logger.info("Salary: $%s - $%s", result.salary_min, result.salary_max)
    logger.info("Skills: %s", result.skills)
    logger.info("Experience: %s years", result.experience_years)


# ═══════════════════════════════════════════════════════════════════════════════
# 📊 COMPARISON: Pydantic vs TypedDict vs dataclass
# ═══════════════════════════════════════════════════════════════════════════════
#
# Feature              | Pydantic        | TypedDict       | dataclass
# ---------------------|-----------------|-----------------|------------------
# Output type          | Object          | dict            | Object
# Runtime validation   | Yes             | No              | No
# Field descriptions   | Field(desc=...) | Annotated       | Comments only
# Nested models        | Full support    | Full support    | Limited
# Custom validators    | Yes             | No              | No
# JSON serialization   | model_dump_json | json.dumps      | Manual
# LangChain support    | Best            | Good            | Limited
# Production use       | Recommended     | Prototyping     | Rare
#
# RULE OF THUMB:
#   Production → Pydantic (always)
#   Quick test → TypedDict
#   Legacy code → dataclass


# ═══════════════════════════════════════════════════════════════════════════════
# 🏭 INTERVIEW QUESTIONS — Structured Output
# ═══════════════════════════════════════════════════════════════════════════════
#
# Q1: What is with_structured_output() in LangChain?
# A:  A method that forces the LLM to return data matching a given schema
#     (Pydantic, TypedDict, or dataclass). Under the hood, it uses function
#     calling — the schema is sent as a "tool" and the LLM fills in the fields.
#
# Q2: Why is Pydantic preferred over TypedDict for structured output?
# A:  Pydantic provides runtime validation, Field descriptions (which help
#     the LLM understand what each field means), custom validators, nested
#     models, and built-in JSON serialization. TypedDict has none of these.
#
# Q3: What does include_raw=True do?
# A:  Returns a dict with 3 keys: 'parsed' (the schema object), 'raw'
#     (the full AIMessage with token usage and metadata), and 'parsing_error'
#     (None if successful). Use this when you need both structured data
#     AND token usage information.
#
# Q4: How is structured output used in production GenAI apps?
# A:  Data extraction (pull entities from documents), API responses
#     (return typed JSON from LangServe), agent tool arguments (structured
#     parameters for tool calls), and RAG answers with citations.
#
# Q5: What happens if the LLM returns data that doesn't match the schema?
# A:  With Pydantic, a validation error is raised (you can catch it).
#     With include_raw=True, the 'parsing_error' field contains the error
#     and 'parsed' is None. With TypedDict, no error — you get wrong data.


# ═══════════════════════════════════════════════════════════════════════════════
# 🚀 Main
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logger.info("=" * 70)
    logger.info("📋 STRUCTURED OUTPUT — Getting JSON/Objects from LLMs")
    logger.info("=" * 70)

    logger.info("\n🔹 1. Pydantic — Basic (flat model)")
    demo_pydantic_basic()

    logger.info("\n🔹 2. Pydantic — Nested (complex structures)")
    demo_pydantic_nested()

    logger.info("\n🔹 3. Pydantic — include_raw=True")
    demo_pydantic_with_raw()

    logger.info("\n🔹 4. TypedDict — Lightweight dict output")
    demo_typeddict()

    logger.info("\n🔹 5. dataclass — Concept & comparison")
    demo_dataclass_concept()

    logger.info("\n🔹 6. Production: Data Extraction Pipeline")
    demo_production_extraction()

    logger.info("\n" + "=" * 70)
    logger.info("✅ Structured Output lesson complete!")
    logger.info("Upcoming: Human-in-the-Loop, Middleware")
    logger.info("=" * 70)
