"""
UNIVERSAL RAG SKELETON — Memorize this pattern.
Every RAG interview question follows these 6 steps.
Upload a file, ask a question, get a grounded answer.
"""

# ============================================
# STEP 1: IMPORTS (memorize these 8 lines)
# ============================================
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# ============================================
# STEP 2: LOAD ENV + CREATE LLM
# ============================================
load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY", "")

llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.3)

# ============================================
# STEP 3: LOAD DOCUMENT
# ============================================
loader = PyPDFLoader("my_document.pdf")
pages = loader.load()
print(f"Loaded {len(pages)} pages")

# ============================================
# STEP 4: CHUNK (Split into smaller pieces)
# ============================================
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)
chunks = splitter.split_documents(pages)
print(f"Created {len(chunks)} chunks")

# ============================================
# STEP 5: EMBED + STORE (FAISS vector store)
# ============================================
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(chunks, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# ============================================
# STEP 6: RAG CHAIN (Prompt + LLM + Parser)
# ============================================
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer the question using ONLY the provided context. "
               "If the context doesn't have the answer, say 'I don't know'."),
    ("human", "Context:\n{context}\n\nQuestion: {question}"),
])

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# ============================================
# STEP 7: ASK A QUESTION
# ============================================
answer = rag_chain.invoke("What is this document about?")
print(answer)
