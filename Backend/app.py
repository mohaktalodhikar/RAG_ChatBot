import os
from fastapi import FastAPI
from pydantic import BaseModel
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline
from fastapi.middleware.cors import CORSMiddleware

# =========================
# CREATE FASTAPI APP
# =========================

app = FastAPI(title="RAG Chatbot API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development only
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Question(BaseModel):
    query: str

# =========================
# STEP 1: Load PDF
# =========================

pdf_path = "yourfile.pdf"

loader = PyPDFLoader(pdf_path)
documents = loader.load()

print("✅ PDF Loaded Successfully")

# =========================
# STEP 2: Split Text
# =========================

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

docs = text_splitter.split_documents(documents)

print(f"✅ Text Split into {len(docs)} chunks")

# =========================
# STEP 3: Create Embeddings
# =========================

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("✅ Embeddings Model Loaded")

# =========================
# STEP 4: Store in FAISS
# =========================

vectorstore = FAISS.from_documents(docs, embeddings)

print("✅ Vector Database Created")

# =========================
# STEP 5: Create Retriever
# =========================

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# =========================
# STEP 6: Load LLM
# =========================

pipe = pipeline(
    "text2text-generation",
    model="google/flan-t5-small",
    max_new_tokens=512
)

llm = HuggingFacePipeline(pipeline=pipe)

print("✅ LLM Loaded")

# =========================
# STEP 7: Create RAG Chain
# =========================

prompt = PromptTemplate.from_template("""
Answer the following question based only on the provided context:
Context: {context}
Question: {question}
Answer:
""")

qa_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
)

print("✅ RAG API Ready!")

# =========================
# API ENDPOINT
# =========================

@app.post("/ask")
def ask_question(data: Question):
    response = qa_chain.invoke(data.query)
    return {"answer": response}