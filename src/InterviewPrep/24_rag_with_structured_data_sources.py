"""
===================================================================================
RAG WITH STRUCTURED DATA SOURCES — SQL, NoSQL, JSON, CSV Deep Dive
===================================================================================

THE QUESTION INTERVIEWERS KEEP ASKING:
    "Not all data comes from PDFs. What if it's a SQL database with 10 tables?
     Or MongoDB? Or 100 CSV files? How do you chunk, embed, and retrieve from
     STRUCTURED data?"

WHY THIS KEEPS COMING UP:
    95% of RAG tutorials show PDF -> chunk -> embed -> search. But production
    data mostly lives in databases. Interviewers test if you can handle REAL
    data sources, not just demo files.

THE CORE INSIGHT (memorize this — it's the foundation):
    Embedding models take TEXT. Vector DBs store TEXT embeddings.
    No matter WHERE the data lives — SQL, MongoDB, JSON, CSV — you
    must CONVERT each record into a MEANINGFUL TEXT REPRESENTATION
    before embedding. That conversion is the entire art.

SECTIONS:
    1. The Universal Pattern (any source -> text -> embed -> store)
    2. SQL/Relational Databases (MySQL, PostgreSQL, etc.)
    3. NoSQL/Document Databases (MongoDB, DynamoDB)
    4. CSV Files (single and bulk 100+ files)
    5. JSON Files / API Responses
    6. The Two Approaches: RAG vs Text-to-SQL (when to use which)
    7. Chunking Strategies for Structured Data
    8. Common Interview Q&A (with crisp answers)
    9. Complete Code Examples
    10. GOLDEN LESSONS
===================================================================================
"""


# =================================================================================
# SECTION 1: THE UNIVERSAL PATTERN (any source -> text -> embed -> store)
# =================================================================================
"""
No matter what the source is, the pipeline is ALWAYS:

    [Data Source] -> [Read/Extract] -> [Convert to Text] -> [Chunk] -> [Embed] -> [Vector DB]

The ONLY thing that changes between PDF, SQL, MongoDB, JSON, CSV is the
FIRST TWO steps (read + convert to text). Everything after that is identical.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

THE CONVERSION PRINCIPLE:

    Each "unit of information" (a row, a document, a record) becomes a
    TEXT STRING that a human would understand when reading it.

    WHY TEXT? Because embedding models are trained on text — they understand
    "Name: John, Age: 30, Department: Engineering" as a meaningful sentence.
    They do NOT understand [row_id=5, col3=30] — that's meaningless vectors.

    THE FORMULA FOR ANY ROW/RECORD:
        text = "field1: value1 | field2: value2 | field3: value3"
    or
        text = "Name: John. Age: 30. Department: Engineering. Role: Senior."

    That's it. That's the core transformation.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

WHAT CHANGES PER SOURCE:

    PDF        -> load pages, split text (already text)
    SQL        -> query rows, convert each row to text string
    MongoDB    -> read documents, flatten JSON to text string
    CSV        -> read rows, convert each row to text string
    JSON       -> parse objects, flatten each to text string

    After conversion, chunking/embedding/retrieval is IDENTICAL.

INTERVIEW ANSWER:
    "The pipeline is the same regardless of source — I just change the
    reader and the text-conversion step. For SQL, I query the table and
    convert each row into a natural-language text string like 'Name: X,
    Age: Y, Department: Z'. For MongoDB, I flatten each JSON document
    into a readable string. Once I have text, the rest — chunking,
    embedding, vector storage, retrieval — is identical to file-based RAG."
"""


# =================================================================================
# SECTION 2: SQL / RELATIONAL DATABASES (MySQL, PostgreSQL, etc.)
# =================================================================================
"""
The most common interview scenario: "I have a SQL table, how do I RAG it?"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

THE APPROACH — ROW-TO-DOCUMENT CONVERSION:

    Each row becomes one "document" (text string) in the vector store.

    Table: employees
    | id | name    | age | dept        | role           | salary |
    |----|---------|-----|-------------|----------------|--------|
    | 1  | John    | 30  | Engineering | Senior Dev     | 120000 |
    | 2  | Sarah   | 28  | Marketing   | Content Lead   | 95000  |

    Converted text documents:
    Doc 1: "Employee: John. Age: 30. Department: Engineering. Role: Senior Dev. Salary: 120000."
    Doc 2: "Employee: Sarah. Age: 28. Department: Marketing. Role: Content Lead. Salary: 95000."

    Now these text strings go through the normal pipeline:
    embed(doc_1) -> vector_1, embed(doc_2) -> vector_2 -> store in vector DB.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

WHAT IF THERE ARE MULTIPLE TABLES?

    Option A — Join first, then convert:
        If tables are related (employees + departments + projects),
        JOIN them into a denormalized view, then convert each row.
        This gives richer context per document.

        "Employee: John. Department: Engineering (Head: Mike, Budget: 2M).
         Current Project: DocSage (Status: Active, Due: Dec 2026)."

    Option B — Convert per table, tag with metadata:
        Each table becomes its own set of documents, tagged with metadata
        (table_name, primary_key). At retrieval, metadata filtering
        scopes the search to relevant tables.

    Option C — Combine into one summary document per entity:
        Group all data about one entity (e.g., one customer) across tables
        into a single rich document. Best for entity-centric Q&A.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

COLUMN SELECTION (the exact interview question you faced):

    Not all columns are useful for semantic search:
    - INCLUDE: descriptive columns (name, description, category, notes, role)
    - EXCLUDE: IDs, timestamps, internal flags, PII (if not needed)

    This is exactly the CSV-filtering code you discussed:
    excluded = {"id", "created_at", "internal_flag"}
    text = " | ".join(f"{col}: {val}" for col, val in row.items() if col not in excluded)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

WHEN TO CHUNK vs NOT CHUNK FOR SQL:

    Single row = short text (50-200 words typically):
        -> NO further chunking needed. Each row IS one chunk/document.

    Single row = very long text (e.g., a 'notes' or 'description' column
    with paragraphs of text):
        -> Chunk THAT column's text, but attach the row's other fields
           as metadata so you don't lose context.

    Multiple rows grouped (e.g., all orders for one customer):
        -> Combine into one document per entity, THEN chunk if too long.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

INCREMENTAL UPDATES (production concern):

    Tables change — new rows added, old rows updated, some deleted.
    - Track with a timestamp column (updated_at) or change-data-capture.
    - On each sync: query WHERE updated_at > last_sync_time.
    - Upsert changed rows (same deterministic ID = update in vector DB).
    - Delete removed rows by ID.

INTERVIEW ANSWER:
    "For SQL, I query the table — optionally joining related tables for
    richer context — and convert each row into a natural-language text
    string with column names as labels. I exclude non-semantic columns
    like IDs and timestamps. Each converted row becomes one document in
    the vector store. If a text column is very long, I chunk it further
    but attach the row's other fields as metadata. For updates, I use
    a timestamp-based sync — only re-process rows changed since last sync,
    upsert with deterministic IDs, and delete removed records."
"""


# =================================================================================
# SECTION 3: NoSQL / DOCUMENT DATABASES (MongoDB, DynamoDB)
# =================================================================================
"""
MongoDB documents are already semi-structured JSON — actually EASIER than SQL.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

MONGODB DOCUMENT EXAMPLE:

    {
        "_id": "abc123",
        "name": "DocSage",
        "type": "AI Agent",
        "description": "Enterprise document intelligence with agentic RAG...",
        "tech_stack": ["LangGraph", "FAISS", "Groq"],
        "status": "active",
        "created_at": "2026-01-15"
    }

    CONVERTED TO TEXT:
    "Project: DocSage. Type: AI Agent. Description: Enterprise document
     intelligence with agentic RAG. Tech Stack: LangGraph, FAISS, Groq.
     Status: Active."

    KEY DECISIONS:
    - Flatten nested objects into readable sentences.
    - Convert arrays to comma-separated text ("LangGraph, FAISS, Groq").
    - Exclude _id, timestamps (not semantically useful for search).
    - Keep the description field as-is (already text).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

HANDLING NESTED DOCUMENTS:

    {
        "customer": "Acme",
        "orders": [
            {"product": "Widget", "qty": 5, "total": 500},
            {"product": "Gadget", "qty": 2, "total": 300}
        ]
    }

    Option A — One document per top-level record (flatten everything):
        "Customer: Acme. Orders: Widget (qty 5, $500), Gadget (qty 2, $300)."

    Option B — One document per nested item (more granular retrieval):
        Doc 1: "Customer: Acme. Order: Widget, quantity 5, total $500."
        Doc 2: "Customer: Acme. Order: Gadget, quantity 2, total $300."
        (Repeat parent context in each child doc for retrieval context.)

    RULE: If you'll ask questions at the ORDER level ("which orders > $400?"),
    use Option B. If questions are at the CUSTOMER level, use Option A.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

RECURSIVE FLATTENING FUNCTION (handles any depth):

    def flatten_doc(doc, prefix=""):
        parts = []
        for key, value in doc.items():
            if key.startswith("_"): continue  # skip _id etc.
            full_key = f"{prefix}{key}" if not prefix else f"{prefix}.{key}"
            if isinstance(value, dict):
                parts.append(flatten_doc(value, full_key))
            elif isinstance(value, list):
                items = ", ".join(str(v) for v in value)
                parts.append(f"{full_key}: {items}")
            else:
                parts.append(f"{full_key}: {value}")
        return " | ".join(parts)

INTERVIEW ANSWER:
    "MongoDB documents are already semi-structured JSON, so conversion is
    straightforward. I flatten each document into a text string — nested
    objects become dotted keys, arrays become comma-separated values, and
    I skip internal fields like _id. For deeply nested documents with
    arrays of sub-objects (like orders inside a customer), I decide based
    on the query granularity: if users ask at the sub-item level, each
    sub-item becomes its own document with parent context attached."
"""


# =================================================================================
# SECTION 4: CSV FILES (single and bulk 100+ files)
# =================================================================================
"""
CSV is just SQL without the database — same row-to-text conversion applies.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SINGLE CSV:

    Same approach as SQL: headers are column names, each row becomes a document.

    import csv
    documents = []
    excluded = {"id", "created_at"}
    with open("employees.csv") as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = " | ".join(f"{k}: {v}" for k, v in row.items() if k not in excluded)
            documents.append(text)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

100 CSV FILES (the bulk interview question):

    Each CSV may represent a different entity/topic. Two strategies:

    Strategy A — TAG by source:
        for csv_file in csv_files:
            reader = csv.DictReader(open(csv_file))
            for row in reader:
                text = row_to_text(row)
                metadata = {"source": csv_file.name, "table": csv_file.stem}
                documents.append(Document(page_content=text, metadata=metadata))

        At retrieval, optionally filter by source/table metadata.

    Strategy B — COMBINE into categories first:
        Group CSVs by topic (all employee CSVs together, all product CSVs
        together), then process each group with appropriate column selection.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

THE INTERVIEWER'S POINT — "WHY NOT LOAD INTO A TABLE FIRST?":

    He was asking: "Why use a file loader when a CSV IS tabular data?"
    The correct answer: "You're right — I treat it AS structured data, not
    as a raw text file. I read it with DictReader (which gives me column
    names as keys), convert each row to a semantic text string using the
    column names as labels, and process from there. I'm NOT blindly
    splitting raw CSV text with a RecursiveTextSplitter — that would cut
    through rows and columns arbitrarily. Each ROW is my natural unit."

INTERVIEW ANSWER:
    "For CSV I treat it as structured data, not raw text. Each row becomes
    one document — I use DictReader to get column names as keys, convert
    each row to a labeled text string, and exclude non-semantic columns.
    For 100 CSVs I tag each document with the source filename as metadata
    so I can filter at retrieval time. I never blindly text-split a CSV
    file — the row is the natural document boundary."
"""


# =================================================================================
# SECTION 5: JSON FILES / API RESPONSES
# =================================================================================
"""
JSON comes in two shapes: array of objects (like CSV) or deeply nested.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SHAPE 1 — ARRAY OF FLAT OBJECTS (same as CSV/SQL):

    [
        {"name": "John", "role": "Developer", "skills": ["Python", "RAG"]},
        {"name": "Sarah", "role": "Designer", "skills": ["Figma", "UX"]}
    ]

    Convert:
    "Name: John. Role: Developer. Skills: Python, RAG."
    "Name: Sarah. Role: Designer. Skills: Figma, UX."

SHAPE 2 — DEEPLY NESTED (API responses, configs):

    {
        "company": "Infor",
        "departments": [
            {"name": "AI", "teams": [{"lead": "Tiru", "members": 5}]},
            {"name": "Backend", "teams": [...]}
        ]
    }

    Options:
    A. Flatten the whole thing into one long text (good for small JSONs).
    B. Extract each meaningful "entity" as a separate document:
        "Company: Infor. Department: AI. Team Lead: Tiru. Members: 5."
        "Company: Infor. Department: Backend. ..."

    Rule: each document should be self-contained — a reader should understand
    it without needing the parent. Repeat parent context in children.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

LangChain has loaders:
    - JSONLoader (jq-based extraction for nested paths)
    - But manual conversion gives you full control over text quality.

INTERVIEW ANSWER:
    "For JSON, if it's flat arrays, each object becomes one document with
    labeled fields — same as SQL rows. If it's deeply nested, I extract
    each meaningful entity as a separate document with parent context
    repeated. The key is that each embedded document must be self-
    contained and semantically meaningful as a standalone text."
"""


# =================================================================================
# SECTION 6: THE TWO APPROACHES — RAG vs TEXT-TO-SQL (when to use which)
# =================================================================================
"""
THIS IS THE SENIOR QUESTION. Interviewers ask: "Why not just query the DB directly?"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

TWO COMPLETELY DIFFERENT APPROACHES:

    APPROACH 1 — VECTORIZE THE DATA (RAG on structured data):
        Convert rows to text -> embed -> store in vector DB -> semantic search.
        Good for: fuzzy/semantic questions, descriptions, unstructured columns,
                  when exact SQL can't express the query.
        Example: "Find employees with AI experience" (no exact column for this).

    APPROACH 2 — TEXT-TO-SQL (LLM writes SQL directly):
        Give the LLM the schema -> it writes a SQL query -> execute on the DB.
        Good for: exact/aggregate questions, counts, sums, filters, joins.
        Example: "How many employees in Engineering earn above 100K?"
        (This is a precise query — SQL is the right tool, not vector search.)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

WHEN TO USE WHICH:

    QUESTION TYPE                         BEST APPROACH
    "Find similar products to X"          RAG (semantic similarity)
    "Total sales this quarter"            Text-to-SQL (aggregate)
    "Employees with AI experience"        RAG (fuzzy, no exact column)
    "Show all orders > $500"              Text-to-SQL (exact filter)
    "Describe the refund policy"          RAG (document retrieval)
    "Count customers by country"          Text-to-SQL (GROUP BY)

    HYBRID (the senior answer):
    "In production I use BOTH. A router/classifier decides: if the question
    needs exact data/aggregation, route to Text-to-SQL. If it needs semantic
    matching or fuzzy lookup, route to RAG. Some systems run both in parallel
    and fuse results."

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

TEXT-TO-SQL QUICK OVERVIEW (so you can mention it):

    1. Give the LLM the database schema (table names, columns, types).
    2. User asks a natural-language question.
    3. LLM generates a SQL query.
    4. Execute the SQL on the real database.
    5. Return the result to the user.

    Risks: SQL injection (validate/sandbox), wrong SQL (validate/retry),
    sensitive data exposure (access control).

INTERVIEW ANSWER:
    "There are two approaches for structured data. If the question needs
    SEMANTIC matching or fuzzy lookups — 'find similar products' — I
    vectorize the data and use RAG. If the question needs EXACT answers or
    aggregations — 'total sales this quarter' — I use Text-to-SQL where
    the LLM writes SQL against the schema. In production I often combine
    both with a router that classifies the query type and picks the right
    path. They're complementary, not competing."
"""


# =================================================================================
# SECTION 7: CHUNKING STRATEGIES FOR STRUCTURED DATA
# =================================================================================
"""
Chunking rules are DIFFERENT for structured vs unstructured data.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

RULE 1 — THE ROW IS YOUR NATURAL CHUNK.
    Don't blindly apply RecursiveCharacterTextSplitter on tabular data.
    A row is a self-contained unit of information. Splitting mid-row
    corrupts the meaning (half a record is useless).

    ONE ROW = ONE DOCUMENT (for short rows like employee records).

RULE 2 — LONG TEXT COLUMNS GET CHUNKED SEPARATELY.
    If a row has a 'description' column with 2000 words, chunk THAT column.
    But attach the other columns as METADATA so the chunk retains context.

    chunks = split(row["description"])
    for chunk in chunks:
        doc = Document(
            page_content=chunk,
            metadata={"name": row["name"], "id": row["id"], "table": "products"}
        )

RULE 3 — GROUP RELATED ROWS INTO ONE DOCUMENT WHEN IT MAKES SENSE.
    All orders for one customer -> one document per customer.
    All messages in one ticket -> one document per ticket.
    This gives richer context for entity-level questions.

RULE 4 — METADATA IS YOUR FRIEND.
    Always attach: source table name, primary key/ID, any categorical
    columns that are useful for filtering at retrieval time.
    This enables metadata-filtered retrieval: "search only in the
    products table" or "search only Engineering department."

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

COMPARISON:

    DATA TYPE        CHUNKING APPROACH
    PDF / long text  RecursiveCharacterTextSplitter (512/50)
    SQL row (short)  No chunking — one row = one document
    SQL row (long)   Chunk the long column, metadata = other columns
    MongoDB doc      Flatten to text; chunk if > 500 tokens
    CSV row          Same as SQL row
    JSON object      Same as MongoDB doc

INTERVIEW ANSWER:
    "For structured data the ROW is my natural chunk — I don't apply a text
    splitter on tabular data because splitting mid-row corrupts meaning.
    Each row becomes one document. If a row has a long text column like
    'description', I chunk that column separately but attach the other
    fields as metadata so context isn't lost. I also tag every document
    with the source table and primary key for metadata-filtered retrieval."
"""


# =================================================================================
# SECTION 8: COMMON INTERVIEW Q&A (with crisp answers)
# =================================================================================
"""
Every question you've faced (and will face) on this topic, with tight answers.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Q: "How do you chunk data from a SQL table?"
A: "Each row becomes one document — the row IS the chunk. I convert it to a
    labeled text string with column names as keys. No text splitter needed
    for short rows. If a column has long text, I chunk that column only and
    attach other fields as metadata."

Q: "What if there are 500 records?"
A: "500 documents in the vector store — one per row. That's small. Vector
    DBs handle millions. Each is an independent, searchable document."

Q: "How do you handle multiple related tables?"
A: "Option A: JOIN first into a denormalized view, then convert each row.
    Option B: process each table separately and tag with table-name metadata.
    Option C: group all data per entity across tables into one rich document.
    I choose based on what questions users will ask."

Q: "CSV has a header and rows — how do you handle it?"
A: "I use DictReader — it maps headers to keys automatically. Each row becomes
    a dict, which I convert to a labeled text string. I treat it as structured
    data, NOT as a raw file to blindly text-split."

Q: "What about MongoDB / JSON?"
A: "Flatten each document into a readable text string. Nested objects become
    dotted keys, arrays become comma-separated values. Each document becomes
    one searchable text in the vector store."

Q: "Why not just use Text-to-SQL instead of vectorizing?"
A: "Text-to-SQL is great for exact queries (counts, sums, filters). RAG on
    vectorized data is better for semantic/fuzzy questions ('find similar
    products'). In production I use both with a router that picks the right
    approach per query."

Q: "How do you handle column selection — not all columns are needed?"
A: "I define an exclusion set (IDs, timestamps, internal flags) and filter
    them out during the row-to-text conversion. Only semantically meaningful
    columns go into the embedding."

Q: "How do you handle updates — SQL data changes?"
A: "Timestamp-based incremental sync. Query WHERE updated_at > last_sync.
    Upsert changed rows (deterministic ID = source_table + primary_key).
    Delete removed records by ID. No full re-index needed."

Q: "What if rows have NULL / missing values?"
A: "Skip NULL fields in the text conversion — don't include 'salary: None'
    in the string. It adds noise without information."

Q: "How do you handle 100 CSV files?"
A: "Process each file: DictReader, row-to-text, tag with source filename as
    metadata. At retrieval, optionally filter by metadata to scope search
    to specific files/tables."

Q: "What about a very large table (10 million rows)?"
A: "Batch process: read in chunks (10K rows at a time), convert + embed in
    batch, upsert to vector DB. Optionally pre-filter: only index rows
    that are user-facing (active records, not archived). Use an async
    ingestion pipeline with a queue for production scale."

Q: "Do you embed the column NAMES or just values?"
A: "Both — the column name IS context. 'Senior Developer' alone is ambiguous;
    'Role: Senior Developer' is meaningful. Column names act as labels that
    help the embedding model understand what each value represents."
"""


# =================================================================================
# SECTION 9: COMPLETE CODE EXAMPLES (runnable)
# =================================================================================
"""
Copy-paste-ready patterns for each data source.
"""

import csv
import json
from io import StringIO


# ---- SQL-STYLE (simulated with list of dicts) ----

def sql_to_documents(rows, exclude_cols=None):
    """Convert SQL query results to embeddable text documents."""
    exclude_cols = exclude_cols or set()
    documents = []
    for row in rows:
        text = " | ".join(
            f"{col}: {val}" for col, val in row.items()
            if col not in exclude_cols and val is not None
        )
        documents.append(text)
    return documents


# Simulate SQL query result
sql_rows = [
    {"id": 1, "name": "John", "dept": "Engineering", "role": "Senior Dev", "exp": 8},
    {"id": 2, "name": "Sarah", "dept": "Marketing", "role": "Content Lead", "exp": 5},
]
docs = sql_to_documents(sql_rows, exclude_cols={"id"})
print("SQL docs:")
for d in docs:
    print(f"  {d}")


# ---- CSV ----

def csv_to_documents(csv_text, exclude_cols=None):
    """Convert CSV content to embeddable text documents."""
    exclude_cols = exclude_cols or set()
    documents = []
    reader = csv.DictReader(StringIO(csv_text))
    for row in reader:
        text = " | ".join(
            f"{k}: {v}" for k, v in row.items()
            if k not in exclude_cols and v
        )
        documents.append(text)
    return documents


sample_csv = "name,age,dept,role\nJohn,30,Engineering,Developer\nSarah,28,Marketing,Designer"
docs = csv_to_documents(sample_csv, exclude_cols=set())
print("\nCSV docs:")
for d in docs:
    print(f"  {d}")


# ---- MongoDB / JSON ----

def json_to_documents(records, exclude_keys=None):
    """Convert JSON records (list of dicts, possibly nested) to text documents."""
    exclude_keys = exclude_keys or set()
    documents = []
    for record in records:
        text = flatten_to_text(record, exclude_keys)
        documents.append(text)
    return documents


def flatten_to_text(obj, exclude_keys, prefix=""):
    """Recursively flatten a nested dict/JSON to a labeled text string."""
    parts = []
    for key, value in obj.items():
        if key in exclude_keys or key.startswith("_"):
            continue
        label = f"{prefix}{key}" if not prefix else f"{prefix}.{key}"
        if isinstance(value, dict):
            nested = flatten_to_text(value, exclude_keys, label)
            if nested:
                parts.append(nested)
        elif isinstance(value, list):
            items = ", ".join(str(v) for v in value)
            parts.append(f"{label}: {items}")
        else:
            parts.append(f"{label}: {value}")
    return " | ".join(parts)


mongo_docs = [
    {"_id": "abc", "name": "DocSage", "type": "AI Agent",
     "stack": ["LangGraph", "FAISS"], "status": "active"},
    {"_id": "def", "name": "MCP Server", "type": "Integration",
     "stack": ["FastMCP", "Python"], "status": "active"},
]
docs = json_to_documents(mongo_docs, exclude_keys={"_id", "status"})
print("\nMongoDB/JSON docs:")
for d in docs:
    print(f"  {d}")


# ---- BULK CSV (100 files simulation) ----

def bulk_csv_to_documents(file_paths, exclude_cols=None):
    """Process multiple CSV files, tag each doc with source filename."""
    exclude_cols = exclude_cols or set()
    all_docs = []
    for path in file_paths:
        with open(path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                text = " | ".join(
                    f"{k}: {v}" for k, v in row.items()
                    if k not in exclude_cols and v
                )
                all_docs.append({
                    "text": text,
                    "metadata": {"source": path, "table": path.split("/")[-1].replace(".csv", "")}
                })
    return all_docs

# Usage (not run here — no files): bulk_csv_to_documents(["emp.csv", "dept.csv"])
print("\nBulk CSV: process each file, tag with source filename as metadata.")


# =================================================================================
# SECTION 10: GOLDEN LESSONS
# =================================================================================
"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

GOLDEN LESSON 1 — THE ROW IS THE CHUNK.
    For structured data, never blindly apply RecursiveTextSplitter. The row
    (or document, or record) is your natural unit. One row = one embedded doc.

GOLDEN LESSON 2 — COLUMN NAMES ARE CONTEXT.
    Always include column names as labels: "Role: Developer" not just "Developer".
    This is what makes the embedding semantically meaningful.

GOLDEN LESSON 3 — EXCLUDE NOISE, KEEP SIGNAL.
    IDs, timestamps, internal flags are noise for embedding. Filter them before
    conversion. Only semantically meaningful fields should become text.

GOLDEN LESSON 4 — KNOW WHEN TO USE RAG vs TEXT-TO-SQL.
    Semantic/fuzzy questions -> RAG on vectorized data.
    Exact/aggregate questions -> Text-to-SQL.
    Production -> hybrid with a router. Say this and you sound senior.

GOLDEN LESSON 5 — METADATA ENABLES SCOPED RETRIEVAL.
    Tag every document with source table, file name, primary key, category.
    At retrieval time, metadata filters scope search to relevant subsets.

GOLDEN LESSON 6 — THE CONVERSION IS THE ART.
    The technical challenge is NOT in the embedding or the vector DB.
    It's in converting structured data into a TEXT representation that
    captures meaning clearly. Get that right, and retrieval works.

GOLDEN LESSON 7 — REPEAT PARENT CONTEXT IN CHILD DOCUMENTS.
    When flattening nested data (MongoDB, JSON), each child document must
    be self-contained. A user asks about "order 123" — the document for
    that order must include the customer name, not just the order fields.

GOLDEN LESSON 8 — NEVER SAY "I'd USE A CSV LOADER" AS YOUR FULL ANSWER.
    The interviewer wants to hear that you UNDERSTAND the conversion logic,
    not that you call a library function. Explain the row-to-text pattern
    and why you do it — then mention the loader as the implementation tool.

GOLDEN LESSON 9 — UPDATES ARE SOLVED WITH TIMESTAMPS + UPSERT.
    Track updated_at, sync only changed rows, upsert with deterministic IDs.
    This avoids full re-indexing and is the production-grade answer.

GOLDEN LESSON 10 — THIS IS THE SAME PIPELINE, DIFFERENT READER.
    Whether it's PDF, SQL, MongoDB, CSV, or JSON — the pipeline after
    text conversion is IDENTICAL: embed -> store -> retrieve -> generate.
    Only the read/convert step changes. Say this to frame your answer.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ONE-LINE CHEAT SHEET FOR RAPID-FIRE:

    Universal pattern: source -> read -> convert to text -> embed -> store -> retrieve.
    SQL:     query rows -> convert each to labeled text -> embed -> one doc per row.
    MongoDB: flatten JSON -> text string with dotted keys -> embed.
    CSV:     DictReader -> row-to-text (headers as keys) -> embed.
    JSON:    parse -> flatten objects -> text -> embed.
    Chunking: row IS the chunk (short rows); chunk only long text columns.
    RAG vs SQL: semantic/fuzzy -> RAG; exact/aggregate -> Text-to-SQL; prod -> both.
    Updates:  timestamp sync + upsert (no full re-index).
"""


# =================================================================================
# RUN SUMMARY
# =================================================================================

if __name__ == "__main__":
    print()
    print("=" * 70)
    print("LESSON 24: RAG WITH STRUCTURED DATA SOURCES")
    print("SQL, MongoDB, JSON, CSV — Interview Deep Dive")
    print("=" * 70)
    print()
    print("THE CORE INSIGHT:")
    print("  No matter the source, convert each record to a meaningful")
    print("  TEXT STRING, then the rest (embed, store, retrieve) is identical.")
    print()
    print("THE ROW-TO-TEXT FORMULA:")
    print('  text = " | ".join(f"{col}: {val}" for col, val in row.items())')
    print()
    print("RAG vs TEXT-TO-SQL:")
    print("  Semantic/fuzzy -> RAG on vectorized data")
    print("  Exact/aggregate -> Text-to-SQL")
    print("  Production     -> BOTH with a router")
    print()
    print("=" * 70)
