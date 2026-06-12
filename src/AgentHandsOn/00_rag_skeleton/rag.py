# RAG SKELETON — Typed by hand, corrected version
# ============================================
# STEP 1: IMPORTS
# ============================================
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq                                    # FIX 1: langchain_groq, not langchain_core
from langchain_community.document_loaders import TextLoader            # Using TextLoader for .txt file
from langchain_text_splitters import RecursiveCharacterTextSplitter     # FIX 3: Character before Text
from langchain_huggingface import HuggingFaceEmbeddings                # FIX 4: HuggingFace (not HUgging)
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser              # FIX 5: output_parsers (plural), StrOutputParser
from langchain_core.runnables import RunnablePassthrough               # FIX 6: lowercase 't' in through

# ============================================
# STEP 2: LOAD ENV + CREATE LLM
# ============================================
load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY", "")            # FIX 7: add "" default

llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.3)

# ============================================
# STEP 3: LOAD DOCUMENT
# ============================================
file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample.txt")  # FIX 8: sample.txt
loader = TextLoader(file_path, encoding="utf-8")
pages = loader.load()
print(f"Loaded {len(pages)} pages")

# ============================================
# STEP 4: CHUNK
# ============================================
splitter = RecursiveCharacterTextSplitter(                             # FIX 9: chunk_size, chunk_overlap (underscores)
    chunk_size=1000,
    chunk_overlap=200,
)
chunks = splitter.split_documents(pages)                               # FIX 10: split (not spilit)
print(f"Created {len(chunks)} chunks")

# ============================================
# STEP 5: EMBED + STORE
# ============================================
embeddings = HuggingFaceEmbeddings(                                    # FIX 11: model_name (underscore)
    model_name="sentence-transformers/all-MiniLM-L6-v2"                # FIX 12: sentence-transformers (lowercase, plural)
)
vectorstore = FAISS.from_documents(chunks, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})           # FIX 13: as_retriever, search_kwargs={"k": 4}

# ============================================
# STEP 6: RAG CHAIN
# ============================================
def format_docs(docs):                                                 # FIX 14: no type hint needed here
    return "\n\n".join(doc.page_content for doc in docs)

prompt = ChatPromptTemplate.from_messages([                            # FIX 15: from_messages (plural) + list []
    ("system", "Answer the question using ONLY the provided context. "
               "If the context doesn't have the answer, say 'I don't know'."),
    ("human", "Context:\n{context}\n\nQuestion: {question}"),
])

rag_chain = (                                                          # FIX 17: use {} dict with string keys
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()                                                # FIX 18: need () to instantiate
)

# ============================================
# STEP 7: ASK A QUESTION
# ============================================
answer = rag_chain.invoke("What RAG patterns does DocSage use?")
print(answer)                                                          # FIX 19: invoke returns string directly