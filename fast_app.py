import os
import faiss
import pickle
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from typing import List

# ----------------------------
# FastAPI App
# ----------------------------
app = FastAPI(title="AI Research Assistant (CPU-Friendly)")

# ----------------------------
# Request Model
# ----------------------------
class QueryRequest(BaseModel):
    question: str

# ----------------------------
# Load & Split Documents
# ----------------------------
def load_documents(file_path: str) -> List[Document]:
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    return [Document(page_content=text, metadata={"source": file_path})]

def split_documents(documents: List[Document]) -> List[Document]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50
    )
    return text_splitter.split_documents(documents)

# ----------------------------
# Build or Load FAISS Index
# ----------------------------
def build_or_load_faiss(docs: List[Document], embeddings, index_path="faiss_index") -> FAISS:
    if os.path.exists(index_path):
        print("✅ Loading existing FAISS index...")
        with open(f"{index_path}/faiss.pkl", "rb") as f:
            return pickle.load(f)
    else:
        print("✅ Creating new FAISS index...")
        vectorstore = FAISS.from_documents(docs, embeddings)
        os.makedirs(index_path, exist_ok=True)
        faiss.write_index(vectorstore.index, f"{index_path}/faiss.index")
        with open(f"{index_path}/faiss.pkl", "wb") as f:
            pickle.dump(vectorstore, f)
        print("✅ FAISS index created and saved locally.")
        return vectorstore

# ----------------------------
# Load resources on startup
# ----------------------------
print("🔹 Loading documents and embeddings...")
file_path = "sample_docs.txt"
documents = load_documents(file_path)
split_docs = split_documents(documents)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = build_or_load_faiss(split_docs, embeddings)

print("🔹 Loading model and tokenizer...")
model_name = "google/flan-t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

pipe = pipeline(
    "text2text-generation",
    model=model,
    tokenizer=tokenizer,
    max_length=512,
    max_new_tokens=150,
    do_sample=True,
    top_k=50,
    temperature=0.7
)

print("✅ AI Research Assistant ready!")

# ----------------------------
# API Endpoint
# ----------------------------
@app.post("/ask")
def ask_question(request: QueryRequest):
    query = request.question

    # Retrieve top-1 relevant document chunk
    retriever = vectorstore.as_retriever(search_kwargs={"k": 1})
    docs = retriever.get_relevant_documents(query)

    context = docs[0].page_content if docs else "No context found."

    prompt = f"""
You are an AI research assistant.
Answer the following question concisely in 3-5 sentences using only the provided context.

Context:
{context}

Question: {query}
Answer:
"""

    try:
        result = pipe(prompt)[0]['generated_text']
        sources = [doc.metadata for doc in docs]
        return {"answer": result, "sources": sources}
    except Exception as e:
        return {"error": str(e)}
