"""
===================================================================================
VIDAXL AI PROMPTER — COMPLETE INTERVIEW PREPARATION
===================================================================================

Company: VidaXL (Dutch eCommerce, home & garden products, 20+ countries)
Role: AI Prompter (Prompt Engineering + AI Orchestration + API Integration)
Interview: L1 Discussion with Adrianna Gacka (Product Owner AI) & Arjan van Helden (Manager UX)
Duration: 60 minutes

THIS LESSON COVERS:
    1. Company Overview & What They Do
    2. Prompt Engineering — Techniques (Zero-shot, Few-shot, CoT, Role, etc.)
    3. Token Optimization — How to Reduce Cost
    4. AI Orchestration & API Integration
    5. eCommerce AI Use Cases (Product Descriptions, Support, SEO, etc.)
    6. Prompt Testing & Evaluation
    7. AI Tools & Platforms You Should Know
    8. How to Explain AI to Non-Technical People
    9. Your Experience Mapped to This Role
    10. 40+ Interview Q&A with Ready Answers

EXISTING LESSONS TO REVISE:
    - 06_docsage_project_deep_dive.py (Sections 5, 6, 13 — LLM Factory, Tools, Design Decisions)
    - 07_rag_complete_guide.py (Section 1-3 only — What is RAG, Architecture)
    - Your MCP project experience (17+ tools, token optimization, YAML instructions)
===================================================================================
"""


# =================================================================================
# SECTION 1: COMPANY OVERVIEW — VIDAXL
# =================================================================================
"""
WHAT IS VIDAXL?
    - Dutch eCommerce company, founded 2006, headquartered in Venlo, Netherlands
    - Global online retailer: furniture, home & garden, DIY, pet supplies, sports
    - Ships to 20+ countries (Europe, US, Australia)
    - Started on eBay, grew into independent platform (vidaxl.com)
    - Also runs DropXL (dropshipping platform for other sellers)
    - Office in Kondapur, Hyderabad (India tech team)
    - Product-based company (sells own-brand products)

WHY THEY NEED AI:
    - 100,000+ products need descriptions in multiple languages
    - Customer support across 20+ countries
    - SEO optimization for product listings
    - Pricing optimization against competitors
    - Content generation at scale (marketing, emails, social media)
    - Workflow automation (reduce manual work)

YOUR INTERVIEW PANEL:
    - Adrianna Gacka — Product Owner AI and Automation (she decides what AI projects to build)
    - Arjan van Helden — Manager User Experience (he cares about how AI improves user experience)

    They will ask: "Can this person help us implement AI in our eCommerce operations?"
    They are NOT looking for a hardcore coder. They want someone who can:
    1. Write excellent prompts that produce quality output
    2. Optimize those prompts to save money (token cost)
    3. Integrate AI into their existing systems (API knowledge)
    4. Research and recommend new AI tools/approaches
    5. Communicate clearly with both tech and non-tech teams
"""


# =================================================================================
# SECTION 2: PROMPT ENGINEERING — ALL TECHNIQUES
# =================================================================================
"""
WHAT IS PROMPT ENGINEERING?
    The practice of designing effective inputs (prompts) for LLMs to produce
    accurate, relevant, and useful outputs. It's about COMMUNICATING with AI
    in a way that gets the best results.

    INTERVIEW ANSWER:
    "Prompt engineering is the art and science of crafting inputs for LLMs
    to get reliable, high-quality outputs. It involves choosing the right
    technique (zero-shot, few-shot, chain-of-thought), structuring the prompt
    with clear instructions, constraints, and output format, and iteratively
    testing and refining until the output meets quality standards."

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

TECHNIQUE 1: ZERO-SHOT PROMPTING
    Ask the model directly without any examples.
    Works for simple, well-defined tasks.

    Example:
        "Write a product description for a wooden garden bench.
         Max 100 words. Tone: friendly and professional."

    When to use: Simple tasks where the model already knows the format.
    When NOT to use: Complex tasks where output format matters a lot.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

TECHNIQUE 2: FEW-SHOT PROMPTING
    Give 2-3 examples of input-output pairs, then ask for a new one.
    The model learns the PATTERN from your examples.

    Example:
        "Here are examples of good product descriptions:

         Product: Oak Dining Table
         Description: Elevate your dining space with this solid oak table.
         Seats 6 comfortably. Natural wood grain finish adds warmth to any room.

         Product: Velvet Sofa
         Description: Sink into luxury with this plush velvet sofa.
         Available in 5 colors. Perfect for modern living rooms.

         Now write a description for:
         Product: Rattan Garden Chair"

    When to use: When you need consistent style/format across many outputs.
    When NOT to use: When examples add too many tokens (cost issue).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

TECHNIQUE 3: CHAIN-OF-THOUGHT (CoT) PROMPTING
    Ask the model to think step by step before giving the final answer.
    Dramatically improves reasoning and accuracy.

    Example:
        "Think step by step:
         1. What are the key features of this product?
         2. Who is the target customer?
         3. What tone should the description have?
         4. Now write the product description."

    When to use: Complex tasks requiring reasoning (pricing, analysis, comparisons).
    When NOT to use: Simple tasks (adds unnecessary tokens).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

TECHNIQUE 4: ROLE PROMPTING (Persona)
    Assign a specific role/expertise to the model.
    Changes the tone, vocabulary, and depth of the response.

    Example:
        "You are a senior eCommerce copywriter with 10 years of experience
         writing product descriptions that convert browsers into buyers.
         Write a description for: Bamboo Standing Desk"

    When to use: When you need domain-specific expertise or a specific tone.
    Pro tip: The more specific the role, the better the output.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

TECHNIQUE 5: OUTPUT FORMATTING / STRUCTURED OUTPUT
    Tell the model EXACTLY what format you want.
    Prevents verbose, unstructured responses.

    Example:
        "Generate product data in this JSON format:
         {
           'title': 'max 60 chars, include main keyword',
           'description': 'max 150 words, 3 paragraphs',
           'bullet_points': ['feature 1', 'feature 2', 'feature 3'],
           'seo_keywords': ['keyword1', 'keyword2', 'keyword3']
         }"

    When to use: When output needs to be parsed by code (API integration).
    This is CRITICAL for VidaXL — they need structured data for their platform.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

TECHNIQUE 6: CONSTRAINTS & GUARDRAILS
    Set explicit boundaries on what the model should/shouldn't do.

    Example:
        "Rules:
         - Max 100 words
         - Do NOT mention competitor brands
         - Do NOT make claims about durability without evidence
         - Include at least 2 SEO keywords from this list: [garden, outdoor, wooden]
         - Tone: warm, inviting, professional
         - Language: British English (not American)"

    When to use: ALWAYS in production. Constraints prevent bad outputs.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

TECHNIQUE 7: PROMPT CHAINING (Multi-step)
    Break a complex task into multiple smaller prompts.
    Output of prompt 1 becomes input of prompt 2.

    Example (product listing pipeline):
        Prompt 1: "Extract key features from this raw product data: [raw data]"
        Prompt 2: "Write a title using these features: [features from step 1]"
        Prompt 3: "Write a description using these features: [features from step 1]"
        Prompt 4: "Generate SEO keywords for this product: [title + description]"

    When to use: Complex workflows where one prompt can't do everything well.
    This is AI ORCHESTRATION — exactly what VidaXL wants.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

TECHNIQUE 8: SELF-CONSISTENCY / SELF-CRITIQUE
    Ask the model to generate multiple answers, then pick the best one.
    Or ask it to critique its own output and improve it.

    Example:
        "Write a product description. Then review it for:
         - Is it under 100 words?
         - Does it include the key features?
         - Is the tone appropriate?
         If any check fails, rewrite it."

    When to use: High-quality outputs where accuracy matters.
"""


# =================================================================================
# SECTION 3: TOKEN OPTIMIZATION — HOW TO REDUCE COST
# =================================================================================
"""
WHY THIS MATTERS:
    VidaXL has 100,000+ products. If each product description costs $0.05 in tokens,
    that's $5,000 just for descriptions. Optimizing tokens = saving real money.

    The JD specifically says: "Optimize token spent to reduce cost."
    This is YOUR strength — you did this in your MCP project!

TOKEN BASICS:
    - 1 token ~ 4 characters in English (roughly 3/4 of a word)
    - "Hello world" = 2 tokens
    - GPT-4: $10/million input tokens, $30/million output tokens
    - GPT-3.5: $0.50/million input tokens, $1.50/million output tokens
    - Groq (Llama): FREE or very cheap

OPTIMIZATION TECHNIQUES:

    TECHNIQUE 1: SHORTER SYSTEM PROMPTS
        BAD (45 tokens):
            "You are a helpful assistant that writes product descriptions
             for an eCommerce website. Please make sure the descriptions
             are engaging and informative."

        GOOD (20 tokens):
            "Write engaging eCommerce product descriptions. Max 100 words."

        Savings: 55% fewer input tokens on EVERY call.

    TECHNIQUE 2: REMOVE UNNECESSARY EXAMPLES (Few-shot → Zero-shot)
        If the model produces good output without examples, remove them.
        Each example adds 50-100 tokens. With 3 examples = 150-300 extra tokens.
        At 100,000 products = 15-30 million wasted tokens.

    TECHNIQUE 3: CONSTRAIN OUTPUT LENGTH
        "Max 100 words" or "Max 3 sentences" prevents the model from
        generating 500-word essays when you only need 100 words.
        Directly reduces output tokens.

    TECHNIQUE 4: STRUCTURED OUTPUT (JSON)
        Instead of: "Write a description and also give me the title and keywords"
        Use: "Return JSON: {title, description, keywords}"
        JSON is more compact than prose. No filler words.

    TECHNIQUE 5: BATCH PROCESSING
        BAD: 1 API call per product (100,000 calls)
        GOOD: 5 products per API call (20,000 calls)

        "Generate descriptions for these 5 products:
         1. Oak Table - dimensions: 120x80cm
         2. Garden Chair - material: rattan
         3. ..."

        Saves on per-call overhead and system prompt repetition.

    TECHNIQUE 6: MODEL SELECTION PER TASK
        Simple tasks (title generation) → GPT-3.5 or Llama-8b (cheap)
        Complex tasks (full description with SEO) → GPT-4 (expensive but better)
        Don't use GPT-4 for everything — use the cheapest model that works.

    TECHNIQUE 7: CACHING
        If the same product is queried twice, return the cached response.
        No API call needed = zero tokens = zero cost.

    TECHNIQUE 8: FIELD-LEVEL FILTERING (Your MCP experience!)
        Don't send the ENTIRE product data to the LLM.
        Filter to only the fields the LLM needs.

        BAD: Send all 50 fields of product data (2000 tokens)
        GOOD: Send only name, category, dimensions, material (200 tokens)

        This is EXACTLY what you did in your MCP project with field-level
        relevance filtering. MENTION THIS IN THE INTERVIEW.

INTERVIEW ANSWER:
    "In my MCP project, I reduced token consumption by implementing field-level
    relevance filtering — instead of sending entire API payloads to the LLM,
    I filtered to only the relevant fields. I also used constrained output
    formatting with YAML agent instructions and few-shot examples only when
    needed. For batch operations, I process multiple items per API call to
    reduce system prompt repetition. The key principle is: send the minimum
    context needed for the task, and constrain the output to the minimum
    format needed for the result."
"""


# =================================================================================
# SECTION 4: AI ORCHESTRATION & API INTEGRATION
# =================================================================================
"""
WHAT IS AI ORCHESTRATION?
    Connecting multiple AI calls, tools, and systems into a WORKFLOW.
    Instead of one prompt → one answer, you have a PIPELINE:
    Input → Step 1 (AI) → Step 2 (AI) → Step 3 (validation) → Output

    Think of it like a factory assembly line, but for AI tasks.

INTERVIEW ANSWER:
    "AI orchestration is chaining multiple AI operations into a workflow.
    In my DocSage project, I built a LangGraph workflow with 5 nodes:
    agent (decides which tool to call), tools (executes retrieval),
    grade (checks document relevance), generate (produces answer), and
    validate (checks for hallucination). Each node is a separate AI call
    with its own prompt and model. The graph orchestrates the flow with
    conditional routing — if grading fails, it rewrites and retries."

ORCHESTRATION TOOLS:
    - LangChain (chains, LCEL pipe operator)
    - LangGraph (stateful graphs with conditional routing)
    - LlamaIndex (workflows for RAG)
    - n8n / Make.com (no-code orchestration)
    - Custom Python scripts with API calls

API INTEGRATION BASICS:
    An API (Application Programming Interface) is how two systems talk to each other.

    How you call an AI API:
        1. Send HTTP POST request to the API endpoint
        2. Include your API key in the header (authentication)
        3. Send the prompt in the request body (JSON)
        4. Receive the response (JSON with the generated text)

    Example (OpenAI API):
        POST https://api.openai.com/v1/chat/completions
        Headers: Authorization: Bearer sk-xxx
        Body: {"model": "gpt-4", "messages": [{"role": "user", "content": "..."}]}
        Response: {"choices": [{"message": {"content": "generated text"}}]}

    In Python (what you know):
        from langchain_groq import ChatGroq
        llm = ChatGroq(api_key="gsk_...", model="llama-3.1-8b-instant")
        response = llm.invoke("Write a product description")

YOUR EXPERIENCE TO MENTION:
    - MCP server: 17+ tools connected via API (OAuth authentication, session management)
    - DocSage: Groq API, Tavily API, Gmail SMTP API — all integrated
    - LangGraph orchestration: 5 nodes, conditional routing, self-correction loops
    - You understand REST APIs, JSON, authentication, error handling
"""


# =================================================================================
# SECTION 5: eCOMMERCE AI USE CASES
# =================================================================================
"""
These are the SPECIFIC use cases VidaXL likely has. Prepare examples for each.

USE CASE 1: PRODUCT DESCRIPTION GENERATION
    Problem: 100,000+ products need unique, SEO-optimized descriptions
    Solution: AI generates descriptions from raw product data

    Prompt example:
        "You are an eCommerce copywriter for a home & garden store.
         Generate a product description.

         Product: Wooden Garden Bench
         Material: Teak wood
         Dimensions: 150cm x 60cm x 90cm
         Color: Natural brown
         Weight capacity: 200kg

         Rules:
         - Max 120 words
         - Include dimensions and material
         - Tone: warm, inviting
         - Include 2 SEO keywords: 'garden bench', 'outdoor seating'
         - End with a call-to-action"

USE CASE 2: MULTI-LANGUAGE TRANSLATION
    Problem: Products sold in 20+ countries, need descriptions in Dutch, German, French, etc.
    Solution: AI translates while maintaining marketing tone

    Prompt example:
        "Translate this product description to German.
         Maintain the marketing tone and SEO keywords.
         Adapt measurements to metric system if needed.
         Keep the same structure (title, description, bullet points)."

USE CASE 3: CUSTOMER SUPPORT AUTOMATION
    Problem: Thousands of customer queries daily across multiple countries
    Solution: AI chatbot handles FAQs, escalates complex issues

    Prompt example:
        "You are a customer support agent for VidaXL.
         Answer the customer's question using ONLY the provided FAQ context.
         If you cannot answer from the FAQ, say: 'Let me connect you with a human agent.'
         Tone: helpful, empathetic, professional.
         Never make promises about delivery dates unless stated in the FAQ."

USE CASE 4: SEO TITLE OPTIMIZATION
    Problem: Product titles need to rank on Google
    Solution: AI generates keyword-rich titles within character limits

    Prompt example:
        "Generate an SEO-optimized product title.
         Rules:
         - Max 60 characters
         - Include primary keyword first
         - Include brand name
         - Include key differentiator (material, size, or color)

         Product: A large outdoor dining table made of acacia wood, seats 8
         Primary keyword: outdoor dining table"

USE CASE 5: REVIEW SUMMARIZATION
    Problem: Products have 100s of reviews, customers don't read all
    Solution: AI summarizes key themes (pros, cons, common issues)

USE CASE 6: EMAIL MARKETING
    Problem: Need personalized email campaigns for different customer segments
    Solution: AI generates email copy based on customer behavior and product data

USE CASE 7: PRICING INTELLIGENCE
    Problem: Need to monitor competitor prices and suggest optimal pricing
    Solution: AI analyzes competitor data and recommends price adjustments
"""


# =================================================================================
# SECTION 6: PROMPT TESTING & EVALUATION
# =================================================================================
"""
HOW DO YOU KNOW IF A PROMPT IS GOOD?

    The JD says: "testing many different prompts and documenting their
    qualitative and monetary outcome"

EVALUATION FRAMEWORK:

    STEP 1: Define success criteria BEFORE testing
        - Quality: Is the output accurate, relevant, well-written?
        - Consistency: Does it produce similar quality every time?
        - Cost: How many tokens does it use?
        - Speed: How fast is the response?

    STEP 2: A/B test prompts
        - Write 2-3 versions of the same prompt
        - Run each on 20-50 test inputs
        - Compare outputs on quality, cost, consistency

    STEP 3: Document results
        | Prompt Version | Avg Quality (1-5) | Avg Tokens | Cost/1000 items | Notes |
        |---|---|---|---|---|
        | V1 (zero-shot) | 3.5 | 180 | $0.90 | Fast but inconsistent |
        | V2 (few-shot) | 4.2 | 320 | $1.60 | Better quality, more expensive |
        | V3 (CoT) | 4.5 | 250 | $1.25 | Best balance of quality and cost |

    STEP 4: Iterate
        - Take the best version, tweak it further
        - Test edge cases (very short products, products with missing data)
        - Document what works and what doesn't

METRICS TO TRACK:
    - Token count (input + output) per call
    - Cost per item (tokens x price per token)
    - Quality score (human rating 1-5)
    - Hallucination rate (% of outputs with incorrect info)
    - Consistency (% of outputs that match expected format)
    - Latency (seconds per response)

INTERVIEW ANSWER:
    "I test prompts systematically. I define success criteria first (quality,
    cost, consistency), then A/B test 2-3 prompt versions on a sample of
    inputs. I document the results in a comparison table showing average
    quality score, token usage, and cost per item. Then I iterate on the
    best version. I also test edge cases — what happens with missing data,
    very short inputs, or unusual products. The goal is finding the sweet
    spot between quality and cost."
"""


# =================================================================================
# SECTION 7: AI TOOLS & PLATFORMS YOU SHOULD KNOW
# =================================================================================
"""
TOOL                 WHAT IT IS                              YOUR EXPERIENCE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ChatGPT (OpenAI)     Most popular LLM, GPT-4/4o              Used via API
Claude (Anthropic)   Best for long documents, safety          Know about it
Gemini (Google)      Multimodal, good for images+text         Know about it
Groq                 Fast inference, free tier                 Used daily (DocSage)
LangChain            Framework for building LLM apps           Used (chains, tools, LCEL)
LangGraph            Stateful agent workflows                  Used (DocSage, 10 agents)
LangSmith            Tracing and evaluation                    Know about it
Streamlit            Frontend for AI apps                      Used (DocSage UI)
FAISS                Vector database (in-memory)               Used (DocSage)
Pinecone             Managed vector database                   Know about it
Tavily               Web search API for LLMs                   Used (DocSage fallback)
n8n / Make.com       No-code AI workflow automation            Know about it

INTERVIEW TIP: When they ask "What AI tools have you used?", say:
    "I've worked with Groq API for LLM inference, LangChain and LangGraph
    for orchestration, FAISS for vector storage, HuggingFace for embeddings,
    Tavily for web search, and Streamlit for frontend. I'm also familiar
    with OpenAI's API, Claude, and tools like LangSmith for evaluation."
"""


# =================================================================================
# SECTION 8: EXPLAINING AI TO NON-TECHNICAL PEOPLE
# =================================================================================
"""
The JD says: "communicate complex technical concepts to non-technical stakeholders"
Arjan van Helden is Manager UX — he may NOT be deeply technical.

RULES FOR EXPLAINING AI SIMPLY:
    1. Use ANALOGIES (not jargon)
    2. Focus on WHAT it does (not HOW it works)
    3. Talk about BUSINESS IMPACT (time saved, cost reduced, quality improved)
    4. Use EXAMPLES they can relate to

EXAMPLE EXPLANATIONS:

    "What is an LLM?"
    TECHNICAL: "A transformer-based neural network trained on billions of tokens..."
    SIMPLE: "It's like a very smart autocomplete. It reads your question and
             writes the most likely good answer, word by word."

    "What is RAG?"
    SIMPLE: "Instead of the AI guessing from memory, we give it a reference book
             to look up answers. It's like an open-book exam."

    "What is prompt engineering?"
    SIMPLE: "It's writing clear instructions for the AI. Like briefing a new
             employee — the better your brief, the better their work."

    "What is token optimization?"
    SIMPLE: "Every word we send to the AI costs money. Token optimization is
             like editing a telegram — say the same thing in fewer words to
             save money. At our scale of 100,000 products, even small savings
             per product add up to thousands of euros."

    "What is AI orchestration?"
    SIMPLE: "Instead of one AI doing everything, we have a pipeline — like a
             factory assembly line. Step 1 extracts features, Step 2 writes
             the title, Step 3 writes the description, Step 4 checks quality.
             Each step is specialized and the whole pipeline is automated."
"""


# =================================================================================
# SECTION 9: YOUR EXPERIENCE MAPPED TO THIS ROLE
# =================================================================================
"""
Map EVERY JD requirement to YOUR experience:

JD REQUIREMENT                          YOUR EXPERIENCE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Craft and optimize AI prompts           MCP project: wrote YAML agent instructions,
                                        system prompts for 17+ tools. DocSage: wrote
                                        prompts for agent, grading, generation, rewriting.

Optimize token spent                    MCP project: field-level relevance filtering
                                        reduced token consumption. Used constrained
                                        output formatting. Different models for different tasks.

API integrations                        Integrated Groq API, Tavily API, Gmail SMTP API,
                                        HuggingFace API. Built MCP server with OAuth
                                        authentication and session management.

AI orchestration                        Built LangGraph workflow with 5 nodes, conditional
                                        routing, self-correction loops. This IS orchestration.

Conduct AI research                     Continuously learning new patterns (Agentic RAG,
                                        Corrective RAG, Adaptive RAG). Applied latest
                                        findings to DocSage project.

Collaborate with dev teams              4 years in software development teams. Used JIRA,
                                        Confluence, Git, sprint planning.

Document and present findings           Created comprehensive documentation for all projects.
                                        Can explain complex AI concepts simply.

Ensure stability of AI workflows        DocSage has error handling, retry logic, fallbacks,
                                        recursion limits, and streaming status updates.

Experience with LLMs                    Daily use of Groq (Llama), familiar with GPT, Claude.
                                        Built production systems with LLMs.

IT development projects                 4 years: Java Spring Boot microservices, React.js
                                        frontend, Python GenAI development.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

YOUR INTRO (tailored for this role):
    "Hi, I'm Tirupathi Naidu. I have 4 years of experience in software
    development, with the last 1.5 years focused on Generative AI.

    I've built production AI systems where prompt optimization was critical.
    In my MCP project, I designed prompts for 17+ LLM-callable tools and
    reduced token consumption using field-level filtering and constrained
    output formatting with YAML agent instructions.

    I also built DocSage, a document intelligence agent using LangGraph —
    that's AI orchestration with multiple steps: retrieval, grading,
    generation, and validation, all connected in an automated pipeline.

    I work daily with Python, LangChain, LangGraph, and various LLM APIs.
    I'm passionate about finding the right balance between output quality
    and token cost — which I understand is exactly what this role focuses on."
"""


# =================================================================================
# SECTION 10: INTERVIEW Q&A (Continued)
# =================================================================================
"""
TOKEN OPTIMIZATION QUESTIONS:

Q9: "How do you reduce token cost?"
A: "Six techniques: (1) Shorter system prompts. (2) Constrain output length.
   (3) Structured output (JSON). (4) Batch multiple items per call.
   (5) Cheaper models for simple tasks. (6) Cache repeated responses.
   In my MCP project, I used field-level filtering to send only relevant
   data to the LLM, reducing input tokens by 60-70%."

Q10: "What is a token?"
A: "A token is roughly 4 characters or 3/4 of a word. 'Hello world' = 2 tokens.
    LLM APIs charge per token — both input (what you send) and output (what you get).
    Reducing tokens = reducing cost."

Q11: "If generating descriptions for 100,000 products costs too much, what do you do?"
A: "First, use the cheapest model that produces acceptable quality (GPT-3.5 instead
   of GPT-4 for simple descriptions). Second, batch 5 products per call to reduce
   system prompt repetition. Third, cache results so regeneration isn't needed.
   Fourth, use zero-shot instead of few-shot if quality is still good. Fifth,
   constrain output to exactly what's needed (JSON, max words)."

AI ORCHESTRATION QUESTIONS:

Q12: "What is AI orchestration?"
A: "Chaining multiple AI operations into an automated workflow. Instead of one
   prompt doing everything, you have a pipeline: extract features → write title →
   write description → validate quality. Each step is specialized."

Q13: "Have you built an AI orchestration system?"
A: "Yes. DocSage uses LangGraph with 5 nodes connected by conditional edges.
   The agent decides which tool to call, the grader checks quality, and if
   quality is low, the system self-corrects by rewriting and retrying.
   I also built an MCP server that orchestrates 17+ tools for ERP integration."

Q14: "What tools do you use for orchestration?"
A: "LangChain for simple chains (prompt | llm | parser), LangGraph for complex
   workflows with conditional routing and state management. For simpler automation,
   tools like n8n or Make.com work well for non-coding teams."

eCOMMERCE QUESTIONS:

Q15: "How can AI help VidaXL specifically?"
A: "Five immediate use cases: (1) Generate product descriptions at scale for
   100,000+ products. (2) Translate descriptions to 20+ languages while maintaining
   marketing tone. (3) Automate customer support for common queries. (4) Generate
   SEO-optimized titles and meta descriptions. (5) Summarize customer reviews
   to identify product issues and highlights."

Q16: "How would you automate product description generation?"
A: "I'd build a pipeline: (1) Take raw product data (name, material, dimensions).
   (2) Use a prompt template with role, constraints, and output format.
   (3) Generate descriptions in batches of 5-10 products per API call.
   (4) Validate output (check length, keywords, format). (5) Store results.
   (6) For quality control, sample 5% for human review."

Q17: "How do you ensure AI-generated content is brand-consistent?"
A: "Three ways: (1) Strong system prompt with brand voice guidelines (tone, words
   to use/avoid, style). (2) Few-shot examples of approved descriptions.
   (3) Output validation — check against brand rules before publishing."

GENERAL AI QUESTIONS:

Q18: "What is hallucination?"
A: "When the AI generates information that sounds plausible but is factually wrong.
   Example: inventing a product feature that doesn't exist. Prevention: provide
   all data in the prompt, constrain to 'answer ONLY from provided data',
   and add validation steps."

Q19: "What's the difference between GPT-4 and GPT-3.5?"
A: "GPT-4 is more capable (better reasoning, fewer errors, follows instructions
   better) but 10-20x more expensive. GPT-3.5 is faster and cheaper but less
   accurate for complex tasks. For simple product descriptions, GPT-3.5 is
   often sufficient. For complex analysis or reasoning, GPT-4 is worth the cost."

Q20: "How do you stay updated on AI trends?"
A: "I follow AI research papers, LinkedIn AI communities, and channels like
   Shina San's Gen Academy. I experiment with new models and techniques weekly.
   I recently studied 10 RAG patterns and implemented 3 of them (Agentic,
   Corrective, Adaptive) in my DocSage project."

Q21: "What is the difference between AI and GenAI?"
A: "AI is the broad field (includes ML, computer vision, NLP, robotics).
   GenAI specifically refers to models that GENERATE new content — text, images,
   code, audio. LLMs like GPT and Claude are GenAI. A spam filter is AI but
   not GenAI."

Q22: "How do you handle edge cases in AI outputs?"
A: "Test with edge cases during development: empty inputs, very long inputs,
   inputs in wrong language, products with missing data. Add fallback logic —
   if the AI output doesn't match expected format, retry or use a default template."

SOFT SKILL QUESTIONS:

Q23: "How do you prioritize multiple AI projects?"
A: "I use an impact-effort matrix. High impact + low effort = do first.
   I estimate token cost savings and quality improvement for each project,
   then prioritize by business value. I break large projects into 2-week sprints
   with clear deliverables."

Q24: "How do you explain AI to a non-technical stakeholder?"
A: "I use analogies. RAG = open book exam. Prompt engineering = briefing a new
   employee. Token optimization = editing a telegram to save money. I focus on
   business impact (time saved, cost reduced) not technical details."

Q25: "Tell me about a time you solved a difficult AI problem."
A: "In DocSage, the agent was losing context in multi-turn conversations because
   ToolMessages (large document chunks) were overflowing the 8K token limit.
   I solved it by building a compact context window — keeping only the last 10
   human/AI messages and skipping ToolMessages. This preserved follow-up context
   while staying within token limits. The fix was about understanding the tradeoff
   between context richness and token constraints."

SCENARIO QUESTIONS:

Q26: "A product description AI generates incorrect dimensions. What do you do?"
A: "Root cause: the AI is hallucinating because dimensions weren't in the prompt.
   Fix: (1) Always include raw product data (dimensions, material, weight) in the
   prompt. (2) Add constraint: 'Use ONLY the provided data. Do NOT invent specs.'
   (3) Add validation: compare output dimensions against input data."

Q27: "The AI generates descriptions that are too generic. How do you improve?"
A: "Three fixes: (1) Add more specific product data to the prompt (unique features,
   use cases, target audience). (2) Use few-shot examples of GOOD descriptions.
   (3) Add constraints: 'Mention at least 2 unique features. Avoid generic phrases
   like high-quality or premium.' (4) Use role prompting: 'You are a specialist
   in garden furniture who highlights what makes each piece unique.'"

Q28: "How would you handle a situation where the AI API goes down?"
A: "Implement fallback logic: (1) Retry with exponential backoff (wait 1s, 2s, 4s).
   (2) If primary API (GPT-4) is down, fall back to secondary (GPT-3.5 or Groq).
   (3) If all APIs are down, queue the request and process later.
   (4) For critical paths, have cached/pre-generated content as emergency fallback."

Q29: "How do you measure the ROI of AI implementation?"
A: "Compare before and after: (1) Time: How long did it take manually vs with AI?
   (2) Cost: Manual copywriter cost vs API cost per description.
   (3) Quality: Human rating of AI output vs manual output.
   (4) Scale: Can now generate 10,000 descriptions/day vs 50/day manually.
   Example: If a copywriter writes 50 descriptions/day at $200/day = $4/description.
   AI generates 10,000/day at $500 API cost = $0.05/description. 80x cheaper."

Q30: "What would you do in your first 30 days in this role?"
A: "Week 1: Understand current AI workflows, tools, and pain points. Review existing
   prompts and their performance. Week 2: Identify quick wins — prompts that can be
   optimized for cost or quality immediately. Week 3: Implement 2-3 optimizations,
   document results with before/after metrics. Week 4: Present findings to the team,
   propose a roadmap for the next quarter."
"""

print("=" * 60)
print("VidaXL AI Prompter — Interview Preparation")
print("=" * 60)
print()
print("10 Sections:")
print("  1.  Company Overview (VidaXL)")
print("  2.  Prompt Engineering Techniques (8 techniques)")
print("  3.  Token Optimization (8 techniques)")
print("  4.  AI Orchestration & API Integration")
print("  5.  eCommerce AI Use Cases (7 use cases)")
print("  6.  Prompt Testing & Evaluation")
print("  7.  AI Tools & Platforms")
print("  8.  Explaining AI to Non-Technical People")
print("  9.  Your Experience Mapped to This Role")
print("  10. 30 Interview Q&A")
print()
print("ALSO REVISE:")
print("  - 06_docsage_project_deep_dive.py (Sections 5, 6, 13)")
print("  - 07_rag_complete_guide.py (Sections 1-3 only)")
print("=" * 60)
