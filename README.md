# AI Research Assistant (CPU-Friendly, Local)

This is a local AI Research Assistant built with **FastAPI**, **LangChain**, **FAISS**, and **HuggingFace models**.  
It can answer questions based on your documents using **retrieval-augmented generation (RAG)**. The app also includes a **simple HTML GUI** for interactive usage.

---

## Features

- Upload your text documents and create embeddings using **MiniLM**.
- Search relevant chunks using **FAISS vector store**.
- Generate answers using **FLAN-T5-small** (CPU-friendly).
- Interactive **HTML GUI** for asking questions.
- Returns the answer along with source documents.

---

## Tech Stack

- **Python 3.10+**
- **FastAPI** for backend API
- **LangChain** for document retrieval and chaining
- **FAISS** for vector search
- **HuggingFace Transformers** for text generation
- **HTML + JavaScript** for the frontend

---

## Installation

1. Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/ai_research_assistant.git
cd ai_research_assist
