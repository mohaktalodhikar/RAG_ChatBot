# 📘 RAG Chatbot using FAISS & HuggingFace

An end-to-end Retrieval-Augmented Generation (RAG) chatbot that processes PDF documents and answers user questions using semantic search and a local LLM.

## **👨‍💻 About This Project**

This project was built as part of my hands-on learning in Retrieval-Augmented Generation (RAG) systems. It demonstrates my understanding of vector databases, embeddings, and LLM integration.

## **🚀 Project Overview**

This project implements a complete RAG pipeline:

📄 Load and extract text from PDF

✂ Split text into smaller chunks

🔢 Convert text into vector embeddings

🗄 Store vectors in FAISS vector database

🔎 Retrieve relevant chunks using similarity search

🤖 Generate contextual answers using a HuggingFace LLM

The chatbot answers questions strictly based on the uploaded PDF content.


## **🧠 Architecture**

>User Query
     
>Retriever (FAISS Similarity Search)
     
>Top-k Relevant Chunks
     
>Prompt Template (Context + Question)
     
>HuggingFace LLM (FLAN-T5)
     
>Final Answer


## **🛠 Tech Stack**

>Python

>LangChain

>FAISS (Vector Store)

>Sentence Transformers (Embeddings)

>HuggingFace Transformers

>FLAN-T5 (Small/Base)

>FastAPI


## **⚙️ Installation**     

1️⃣ Clone Repository

```bash
git clone https://github.com/mohaktalodhikar/RAG_ChatBot.git
cd RAG-chatbot
```

2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

or manually:

```bash
pip install langchain langchain-community langchain-huggingface \
faiss-cpu transformers sentence-transformers pypdf
```

