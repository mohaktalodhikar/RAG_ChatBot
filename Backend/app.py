import os
import re
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

prompt = PromptTemplate.from_template("""You are an expert AI assistant. Provide a clear, well-structured answer to the question based on the provided context.

Guidelines:
1. Give a direct, concise answer without repetition
2. Use proper formatting with paragraphs and bullet points where appropriate
3. If the context doesn't contain the answer, say so clearly
4. Do not repeat the same information multiple times
5. Keep the response focused and relevant

Context: {context}

Question: {question}

Provide a clear, structured answer:""")

def clean_response(text):
    """Clean and format the response to be clear and perfect"""
    if not text:
        return "No response generated."
    
    # Step 1: Extract page_content if present (remove metadata)
    if "page_content='" in text:
        try:
            # Extract content between page_content=' and the last '
            start = text.find("page_content='") + len("page_content='")
            end = text.rfind("'") if text.rfind("'") > start else len(text)
            text = text[start:end]
        except:
            pass
    
    # Step 2: Replace literal escape sequences
    text = text.replace('\\n', ' ').replace('\\t', ' ')
    text = text.replace("\'n", " ").replace("'n", " ")
    
    # Step 3: Fix common escape issues and "nWord" artifacts
    text = text.replace(' n ', ' ')
    # Fix words split by 'n' or '-n' (e.g., long-ncherished -> long-cherished, result nof -> result of)
    text = re.sub(r'(\w+)-n([a-z])', r'\1-\2', text)
    text = re.sub(r'(\w+)\s+no([a-z])', r'\1 o\2', text)
    
    # Remove 'n' prefix from words (e.g., nMan -> Man, nthings -> things)
    # Matches 'n' at the start of a word or after whitespace/punctuation
    text = re.sub(r'(^|[\s.,!?;:])n([A-Za-z])', r'\1\2', text)
    
    # Clean up any remaining standalone 'n' characters or artifacts
    text = re.sub(r'\s+n\s+', ' ', text)
    text = text.replace(' nof ', ' of ')
    
    # Step 4: Remove metadata keys and clean up
    text = re.sub(r"\w+='[^']*',?\s*", "", text)
    text = re.sub(r'\w+=[^\s,]+,?\s*', "", text)
    
    # Step 5: Clean whitespace and formatting
    text = re.sub(r'\s+', ' ', text)
    text = text.replace(' .', '.').replace(' ,', ',')
    
    # Step 6: Proper sentence structure
    sentences = text.split('. ')
    unique_sentences = []
    seen = set()
    
    for sentence in sentences:
        sentence = sentence.strip()
        if sentence and sentence not in seen and len(sentence) > 10:
            seen.add(sentence)
            unique_sentences.append(sentence)
    
    # Join sentences properly
    cleaned = '. '.join(unique_sentences)
    cleaned = cleaned.strip()
    
    # Ensure proper ending
    if cleaned and not cleaned[-1] in '.!?':
        cleaned += '.'
    
    return cleaned

# Create RAG chain
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
    raw_response = qa_chain.invoke(data.query)
    cleaned_response = clean_response(raw_response)
    return {"answer": cleaned_response, "raw_length": len(raw_response), "cleaned_length": len(cleaned_response)}
