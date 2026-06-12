"""
===================================================================================
GEN AI FEATURES IN REAL PROJECT — Interview Revision (CSD/SXE Application)
===================================================================================

These are 3 real Gen AI features you implemented in a production enterprise
application (Cloud Suite Distribution / SXE). Use these to answer:
"Have you integrated Gen AI into a real application?"

FEATURES COVERED:
    1. AI-Powered Product Attribute Generation (LEAD WITH THIS — most impressive)
    2. Sell It / Support It — Product Intelligence
    3. Conversational Quotation Agent (MCP Server)

INTERVIEW STRATEGY:
    - Lead with Feature 1 (Attribute Generation) — most technically deep
    - If they ask "anything else?" → mention Feature 2 (Sell It/Support It)
    - If they ask about conversational AI → bring up Feature 3 (Quotation Agent)
===================================================================================
"""


# =================================================================================
# FEATURE 1: AI-POWERED PRODUCT ATTRIBUTE GENERATION (PRIMARY — LEAD WITH THIS)
# =================================================================================
"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
THE PROBLEM IT SOLVES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

In distribution/eCommerce, every product needs STRUCTURED ATTRIBUTES for:
    - Filtering and faceted search ("show me all pipes > 2 inches")
    - Product comparison on eCommerce sites
    - B2B procurement matching
    - Catalog enrichment
    - Compliance and specifications

THE OLD WAY (manual):
    A catalog manager manually researches each product, looks up specs from
    manufacturer PDFs, and types in 15-20 attributes per product.
    With 10,000+ SKUs → takes MONTHS of manual work.

THE NEW WAY (Gen AI):
    System takes product's basic info (name, brand, category, manufacturer)
    and asks the LLM to generate 20+ structured attributes automatically.
    A human reviews and approves. MONTHS → MINUTES.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
THE TWO-PASS LLM ARCHITECTURE (This is the WOW factor)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

We don't just call the LLM once. We use a TWO-PASS approach:

    PASS 1 — GENERATION:
        "Give me 20 attributes for this product based on its name, brand,
        category, and manufacturer model number."

        Input: Product name, brand, category, manufacturer, model
        Output: JSON array of {attribute, value, datatype, unitOfMeasure}

    PASS 2 — MAPPING:
        "Here are the AI-generated attribute names. Here are the EXISTING
        attribute names in our system. Map them intelligently."

        Input: AI names ["Length", "Material", "Thread Size"]
               Existing names ["Pipe Length", "Body Material"]
        Output: {"Length": "Pipe Length", "Material": "Body Material", "Thread Size": null}

WHY TWO PASSES IS BRILLIANT:
    - Avoids creating DUPLICATE attributes with slightly different names
      ("Length" vs "Pipe Length" vs "Overall Length")
    - Maintains data CONSISTENCY across the catalog
    - REUSES existing attribute definitions (which already have proper UOMs, validation)
    - New attributes that don't map get created fresh

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
COMPLETE TECHNICAL FLOW (Step by Step)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

STEP 1: User opens Product Setup screen for product "1-003"
        Clicks "Auto Generate Attributes" button

STEP 2: Frontend fetches the PROMPT TEMPLATE from config table (SASTA)
        promptId = "icsp-get-attributes"
        Gets: prompt template, model name, version

STEP 3: TEMPLATE VARIABLE SUBSTITUTION
        Replace placeholders with actual product data:
            {lookupnm}  → "TAP EXTENSION"
            {descrip1}  → "Tap Extension Size 3/16 Style B 8 inch"
            {brandcode} → "RIDGID"
            {prodcat}   → "PLUMBING"
            {mfgprod}   → "31005"
            {modelcode} → "E-110"
            {{attributeGroupValues}} → ["Length", "Material", "Diameter"]

STEP 4: LLM CALL #1 — ATTRIBUTE GENERATION
        POST /web/api/shared/genaiprompt
        Body: { prompttxt: "<filled prompt>", model: "claude-haiku-4-5", version: "..." }

        LLM Returns structured JSON:
        { "results": [
            { "attribute": "Length", "attributeDescription": "Overall length",
              "attributeValue": "8", "datatype": "Number", "unitOfMeasure": "in" },
            { "attribute": "Material", "attributeDescription": "Body material",
              "attributeValue": "Steel", "datatype": "Text", "unitOfMeasure": "" },
            ... (20+ attributes)
        ]}

STEP 5: LLM CALL #2 — PA MAPPING (maps AI names to existing system names)
        Uses "pa-mapping-prompt"
        Input: List1 = AI names, List2 = Existing PA names
        Output: { "resultMappings": { "Length": "Pipe Length", "Material": "Body Material" } }

STEP 6: DISPLAY TO USER IN GRID
        Shows: "15 new attributes, 5 mapped to existing"
        User can edit values, select/deselect, modify before saving
        THIS IS HUMAN-IN-THE-LOOP — AI does heavy lifting, human validates

STEP 7: USER CLICKS SAVE → Persist to AttributionService (Java/Spring Boot)
        POST /web/api/ic/attributeservicecreate
        { productId: "1-003", genAI: true, attributes: [...] }
        - If attribute doesn't exist → creates it in DynamoDB
        - If attribute exists → updates description if changed
        - Creates product-attribute-value assignment

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
BULK PROCESSING MODE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

For enriching entire product categories at once:
    - User selects multiple products (or product range)
    - Enables "GenAI" flag → triggers batch processing
    - Products batched in groups of 5 (configurable batch size)
    - Each batch: collects product fields, sends to LLM
    - Results stored in temp-table, then persisted via AttributionService
    - Final output: CSV file emailed to user

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
BUSINESS IMPACT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    METRIC                  BEFORE (Manual)         AFTER (Gen AI)
    Time per product        30-60 minutes           2-3 minutes (with review)
    Products per day        50 (one person)         10,000+ (batch mode)
    Data consistency        Low (human errors)      High (structured JSON output)
    Attribute duplication   Common                  Eliminated (PA mapping)
    Cost                    $4/product (manual)     $0.01/product (API cost)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
HOW TO EXPLAIN IN INTERVIEW (60-second version)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

"One of the most impactful Gen AI features I worked on was AI-powered product
attribute generation for our distribution ERP system.

The problem: distributors have thousands of SKUs(Stock Keeping Unit) but their product data is
sparse — just a name and maybe a category. For eCommerce, you need structured
attributes like dimensions, material, voltage, weight.

We built a two-pass LLM architecture. First, we send the product's basic info
to Claude and it returns 20+ structured attributes as JSON with name, value,
datatype, and unit of measure.

The clever part — we make a SECOND LLM call that maps the AI-generated
attribute names to existing attributes in our system. So 'Length' maps to
'Pipe Length' if that already exists. This prevents attribute duplication
and maintains catalog consistency.

The user sees a grid with all generated attributes, can edit values, and
saves. It's human-in-the-loop — AI does the heavy lifting, human validates.

For bulk processing, we batch products in groups of 5 and can enrich entire
categories in minutes instead of months.

The prompts are stored in a configuration table so business users can tune
them without code changes. We moved from Claude 3.7 Sonnet to Claude Haiku
4.5 for cost optimization on this high-volume use case."

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FOLLOW-UP QUESTIONS & ANSWERS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Q: "How do you handle hallucinations in attribute generation?"
A: "Three safeguards: (1) Human-in-the-loop — users review all attributes
   before saving. (2) UOM validation — AI-generated units are validated
   against our system's UOM master table. (3) Constrained output — the
   prompt forces strict JSON format with only Text/Number/Boolean datatypes."

Q: "How do you handle cost at scale?"
A: "We moved from Claude 3.7 Sonnet to Claude Haiku 4.5 for this feature
   since it's high-volume. We batch 5 products per LLM call in bulk mode.
   Prompts are optimized to return only structured JSON — no explanations."

Q: "What if the LLM returns bad data?"
A: "Multiple safeguards: JSON parse error handling, validation that response
   has a 'results' array, UOM validation against master data, duplicate
   detection against existing assignments, and user can always edit before saving."

Q: "Why two LLM calls instead of one?"
A: "If we just generated attributes and saved them directly, we'd end up with
   thousands of slightly different attribute names across the catalog — 'Length',
   'Pipe Length', 'Overall Length' all meaning the same thing. The mapping call
   ensures we reuse existing definitions and maintain catalog consistency."

Q: "How are prompts managed?"
A: "Prompts are stored in a configuration table (SASTA) with a promptId key.
   Business users can update prompt text without code deployments. The model
   name and version are also configurable — we can switch models without
   touching code."
"""


# =================================================================================
# FEATURE 2: SELL IT / SUPPORT IT — PRODUCT INTELLIGENCE
# =================================================================================
"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WHAT IT DOES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

When a sales/support rep views ANY product in the system, they see two
AI-powered buttons:

    "Sell It" — Generates selling points, key features, competitive advantages
    "Support It" — Generates troubleshooting guides, installation tips, compatibility

This gives reps INSTANT product knowledge without reading manuals or PDFs.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TECHNICAL FLOW
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

STEP 1: User clicks "Sell It" on product "Tap Extension 8-inch"

STEP 2: Frontend checks feature activation flag (INFOROS.GENAI)
        If disabled for this tenant → button is hidden

STEP 3: Frontend fetches prompt template from SASTA table
        promptId = "ai-product-support-info"
        Template: "Tell me about {{item}} for selling purposes. Include key
                   features, benefits, and competitive advantages."

STEP 4: Variable substitution
        {{item}} → "1-003 Tap Extension Size 3/16 Style B 8 inch"

STEP 5: Call Gen AI backend
        POST /SX/webuiWebHandler/shared/genaiprompt
        { prompttxt: "<filled prompt>", model: "claude-haiku", stream: true }

STEP 6: LLM generates response (streamed)

STEP 7: Frontend displays with TYPEWRITER ANIMATION effect
        Characters appear one by one (streaming response)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
KEY DESIGN DECISIONS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. CONFIGURABLE PROMPTS — stored in DB, not hardcoded. Business users can
   tune the prompt without code changes.

2. FEATURE FLAGS — tenant-level activation. Customers can enable/disable
   Gen AI per their subscription.

3. STREAMING — real-time typewriter effect. User sees response building
   instead of waiting 5 seconds for a blank screen.

4. CONTEXT FROM SCREEN — the prompt includes the product the user is
   currently viewing (IBC = Infor Business Context).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
BUSINESS VALUE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    - Sales reps close deals FASTER (instant product knowledge)
    - Support reps resolve issues FASTER (instant troubleshooting)
    - No training needed on 10,000+ products
    - New hires productive from day 1
    - Consistent messaging across all reps

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
HOW TO EXPLAIN IN INTERVIEW (30-second version)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

"We also built a 'Sell It / Support It' feature where sales reps can click
a button on any product and instantly get AI-generated selling points or
troubleshooting guides. It uses the same configurable prompt infrastructure —
the prompt template is stored in our config table with the product name as
a variable. The response streams back with a typewriter effect for better UX.
It's feature-flagged per tenant so customers can enable/disable it."

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FOLLOW-UP QUESTIONS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Q: "How do you ensure the AI doesn't make false claims about products?"
A: "The prompt explicitly says 'based on the product name and specifications
   only.' We don't ask it to make claims about durability or performance
   that aren't in the product data. For critical specs, we pull from the
   structured attribute database, not from LLM generation."

Q: "Why streaming instead of waiting for full response?"
A: "User experience. A 5-second blank screen feels broken. Streaming shows
   the response building in real-time — users start reading immediately.
   It also gives perceived performance improvement."

Q: "How do you handle multiple languages?"
A: "The prompt can include a language instruction: 'Respond in German' or
   'Respond in Dutch.' Since VidaXL operates in 20+ countries, this is
   critical. The same prompt template works for all languages."
"""


# =================================================================================
# FEATURE 3: CONVERSATIONAL QUOTATION AGENT (MCP Server)
# =================================================================================
"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WHAT IT DOES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

A sales rep can CREATE A FULL QUOTE through natural language conversation:

    User: "Create a quote for tap extensions and hacksaws for customer ABC Corp"

    Agent: (internally)
        1. Resolves "ABC Corp" → customer ID 12345
        2. Searches products matching "tap extensions" and "hacksaws"
        3. Fetches current pricing for customer 12345
        4. Creates quote with line items
        5. Returns: "Quote #Q-7890 created with 2 line items. Total: $450.
                     Tap Extension 8-inch: $120 x 2 = $240
                     Hacksaw 12-inch: $105 x 2 = $210"

    User: "Can we get a 10% discount?"

    Agent: (calls repricing tool)
        "Applied 10% discount. New total: $405. Shall I submit?"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ARCHITECTURE — MCP (Model Context Protocol)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    ┌─────────────────────────────────────────────────┐
    │  LLM (Claude/GPT) — The Brain                   │
    │  Reads user message, decides which tool to call  │
    └──────────────────────┬──────────────────────────┘
                           │ (tool calls)
                           ▼
    ┌─────────────────────────────────────────────────┐
    │  MCP SERVER (Python FastMCP)                     │
    │  Exposes ERP operations as LLM-callable tools:   │
    │                                                  │
    │  @mcp.tool() get_customer_info(name)             │
    │  @mcp.tool() search_products(query)              │
    │  @mcp.tool() get_pricing(customer_id, product_id)│
    │  @mcp.tool() create_quote(customer_id, items)    │
    │  @mcp.tool() apply_discount(quote_id, percent)   │
    │  @mcp.tool() submit_order(quote_id)              │
    │  ... 17+ tools total                             │
    └──────────────────────┬──────────────────────────┘
                           │ (API calls)
                           ▼
    ┌─────────────────────────────────────────────────┐
    │  ERP BACKEND (SXE/CSD APIs)                      │
    │  Real business data: customers, products, prices │
    └─────────────────────────────────────────────────┘

KEY CONCEPTS:
    - MCP = Model Context Protocol (open standard for connecting LLMs to tools)
    - FastMCP = Python framework for building MCP servers
    - Each @mcp.tool() has a name, description, and parameters
    - The LLM reads tool descriptions and DECIDES which to call
    - OAuth authentication for secure API access
    - Session management for multi-turn conversations

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
THE 17+ TOOLS IN THE MCP SERVER
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    CATEGORY            TOOLS
    Customer            get_customer_info, get_customer_credit, get_customer_orders
    Products            search_products, get_product_details, get_product_pricing
    Quotes              create_quote, add_line_item, apply_discount, submit_quote
    Orders              get_order_status, track_shipment
    Pricing             get_price_list, calculate_discount, reprice_quote
    Credit              check_credit_hold, get_credit_score
    Recommendations     get_ai_recommendations (cross-sell, upsell)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TOKEN OPTIMIZATION IN MCP (Your key contribution)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Problem: ERP APIs return HUGE JSON responses (50+ fields per object).
         Sending all of that to the LLM wastes tokens and money.

Your solution: FIELD-LEVEL RELEVANCE FILTERING
    - Before sending API response to LLM, filter to only relevant fields
    - Use embeddings to determine which fields are relevant to the user's question
    - Store field relevance mappings in a .pkl file for fast lookup

    Example:
        API returns 50 fields for a customer (address, phone, fax, tax ID, ...)
        User asked: "What's the credit limit for ABC Corp?"
        Filter: Only send {customer_name, credit_limit, credit_used, credit_available}
        Result: 90% fewer tokens sent to LLM

Also: YAML Agent Instructions
    - Each agent (QuotationAgent, InvoiceAgent) has a YAML config file
    - Defines: role, tools available, output format, constraints
    - Keeps system prompts structured and maintainable

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
HOW TO EXPLAIN IN INTERVIEW (45-second version)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

"We also built conversational AI agents using the MCP pattern. Our Quotation
Agent lets a sales rep create a full quote through natural language — 'create
a quote for tap extensions for customer ABC Corp.' The agent resolves the
customer, searches products, fetches pricing, and creates the quote — all
conversationally.

The MCP server is built with Python FastMCP and exposes 17+ ERP operations
as LLM-callable tools. The LLM reads tool descriptions and decides which
to call at each step.

My key contribution was token optimization — ERP APIs return 50+ fields per
object, but the LLM only needs 3-4 relevant fields. I implemented field-level
filtering using embeddings to reduce input tokens by 60-70%, which significantly
cut API costs at scale."

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FOLLOW-UP QUESTIONS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Q: "What is MCP?"
A: "Model Context Protocol — an open standard for connecting LLMs to external
   tools and data sources. Think of it like a USB-C port for AI. The MCP server
   exposes tools with names, descriptions, and parameters. The LLM reads those
   descriptions and decides which tool to call. It's the same pattern as
   LangChain's @tool but standardized across platforms."

Q: "How does the agent decide which tool to call?"
A: "Each tool has a description that explains what it does. The LLM reads ALL
   tool descriptions and matches the user's intent to the right tool. If the
   user says 'check credit for ABC', the LLM picks get_customer_credit because
   its description says 'Check credit limit and usage for a customer.'"

Q: "How do you handle authentication?"
A: "OAuth with service-aware authentication. The MCP server authenticates with
   the ERP backend using OAuth tokens. Session management ensures the token
   is refreshed automatically. The user doesn't need to re-authenticate."

Q: "What if the agent makes a mistake?"
A: "We have guardrails: (1) Confirmation before destructive actions — 'Shall I
   submit this quote?' (2) Validation of inputs before API calls. (3) Error
   handling with user-friendly messages. (4) Audit logging of all agent actions."
"""


# =================================================================================
# SECTION 4: COMPARISON TABLE — WHY ATTRIBUTE GENERATION IS BEST FOR INTERVIEW
# =================================================================================
"""
    ASPECT              SELL IT/SUPPORT IT       ATTRIBUTE GENERATION        QUOTATION AGENT
    ─────────────────────────────────────────────────────────────────────────────────────────
    LLM calls           1 call, display text     2 calls (gen + mapping)     Multiple (per tool)
    Output              Free-form HTML           Structured JSON             Conversational
    Architecture        Simple prompt→display    Two-pass AI pipeline        Agent + MCP tools
    Data persistence    None (just shows)        Writes to DynamoDB          Creates real orders
    Batch processing    No                       Yes (5 per batch)           No
    Integration depth   Widget-level             Full stack (UI→LLM→DB)      Full stack
    eCommerce relevance Medium (sales tool)      HIGH (catalog enrichment)   Medium (B2B sales)
    Technical depth     Low                      HIGH                        HIGH

    LEAD WITH: Attribute Generation (most technically deep + most eCommerce relevant)
    FOLLOW UP: Quotation Agent (shows conversational AI + MCP)
    MENTION BRIEFLY: Sell It/Support It (shows same infra reused across features)
"""


# =================================================================================
# SECTION 5: TECHNICAL KEYWORDS TO DROP IN INTERVIEW
# =================================================================================
"""
Drop these naturally in your answers:

    - Two-pass LLM architecture (generation + mapping)
    - Prompt template with variable substitution
    - Structured JSON output parsing
    - Human-in-the-loop validation
    - Batch processing with configurable batch sizes
    - Configurable prompts in database (no code changes)
    - Model versioning (Claude 3.7 Sonnet → Claude Haiku 4.5)
    - UOM validation against system master data
    - Attribute deduplication via semantic mapping
    - Field-level relevance filtering (token optimization)
    - MCP (Model Context Protocol)
    - Feature flags / tenant-level activation
    - Streaming responses (typewriter effect)
    - OAuth authentication + session management
    - YAML agent instructions
"""


# =================================================================================
# SECTION 6: ADDITIONAL INTERVIEW Q&A
# =================================================================================
"""
Q1: "Have you integrated Gen AI into a real application?"
A: "Yes. In our distribution ERP, I worked on 8+ Gen AI features. The most
   impactful was AI-powered product attribute generation using a two-pass
   LLM architecture — first generating attributes, then mapping them to
   existing system attributes to prevent duplication. We also built
   conversational agents using MCP for quote creation and order management."

Q2: "How do you manage prompts in production?"
A: "Prompts are stored in a configuration table with a promptId key. Business
   users can update prompt text without code deployments. Model name and version
   are also configurable. This separation of prompt from code is critical for
   iteration speed — we can A/B test prompts without releases."

Q3: "How do you handle multi-tenant AI features?"
A: "Feature flags at the tenant level. Each customer can enable/disable Gen AI
   features independently. The feature check happens before any LLM call —
   if disabled, the UI doesn't even show the AI buttons."

Q4: "What model do you use and why?"
A: "Claude Haiku 4.5 for high-volume features (attribute generation, product info)
   because it's fast and cost-effective. Claude 3.7 Sonnet for complex reasoning
   tasks (quotation agent, credit analysis). We match model capability to task
   complexity — don't use expensive models for simple tasks."

Q5: "How do you measure success of AI features?"
A: "Three metrics: (1) Time saved — attribute generation reduced from 30 min/product
   to 2 min. (2) Cost — $0.01/product vs $4/product manual. (3) Quality — human
   review acceptance rate (what % of AI-generated attributes are approved without
   edits). We track all three and report monthly."

Q6: "What challenges did you face?"
A: "Three main challenges: (1) Hallucination — LLM inventing product specs that
   don't exist. Solved with human-in-the-loop and constrained prompts.
   (2) Token cost at scale — solved with field-level filtering and model selection.
   (3) Attribute duplication — solved with the two-pass mapping architecture."

Q7: "How is this relevant to eCommerce?"
A: "Product data quality is the foundation of eCommerce. Better attributes mean
   better search, better filtering, better product comparisons, and ultimately
   better conversion rates. Every eCommerce platform needs structured product
   data — and AI can generate it 100x faster than manual entry."

Q8: "What would you do differently if starting over?"
A: "I'd add automated evaluation from day one — a test set of products with
   known correct attributes, and run every prompt change against it to catch
   regressions. I'd also add semantic caching to avoid re-generating attributes
   for similar products."
"""

print("=" * 60)
print("Gen AI Features in Real Project — Interview Revision")
print("=" * 60)
print()
print("3 Features (in order of interview priority):")
print("  1. AI-Powered Product Attribute Generation (LEAD)")
print("  2. Sell It / Support It — Product Intelligence")
print("  3. Conversational Quotation Agent (MCP Server)")
print()
print("Key sections:")
print("  - Complete technical flow (step by step)")
print("  - Two-pass LLM architecture explanation")
print("  - Business impact with numbers")
print("  - 60-second interview explanation for each")
print("  - Follow-up Q&A for each feature")
print("  - Technical keywords to drop")
print("  - Comparison table (why Attribute Gen is best)")
print("=" * 60)
