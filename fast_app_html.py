import os
import pickle
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from typing import List

# ----------------------------
# FastAPI App
# ----------------------------
app = FastAPI(title="AI Research Assistant (CPU-Friendly)")

# Mount static folder (optional)
if not os.path.exists("static"):
    os.makedirs("static")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Templates folder
templates = Jinja2Templates(directory="templates")

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
    if os.path.exists(f"{index_path}/faiss.pkl"):
        with open(f"{index_path}/faiss.pkl", "rb") as f:
            return pickle.load(f)
    else:
        vectorstore = FAISS.from_documents(docs, embeddings)
        os.makedirs(index_path, exist_ok=True)
        faiss_index_path = os.path.join(index_path, "faiss.index")
        FAISS.save_local(vectorstore, faiss_index_path)
        with open(f"{index_path}/faiss.pkl", "wb") as f:
            pickle.dump(vectorstore, f)
        return vectorstore

# ----------------------------
# Load documents and embeddings
# ----------------------------
file_path = "sample_docs.txt"
documents = load_documents(file_path)
split_docs = split_documents(documents)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = build_or_load_faiss(split_docs, embeddings)

# ----------------------------
# Load CPU-friendly LLM
# ----------------------------
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

# ----------------------------
# Serve HTML GUI
# ----------------------------
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# ----------------------------
# Ask endpoint
# ----------------------------
class QueryRequest(BaseModel):
    question: str

@app.post("/ask")
def ask_question(request: QueryRequest):
    query = request.question

    # Retrieve top-3 relevant document chunks
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    docs = retriever.get_relevant_documents(query)
    context = "\n\n".join([doc.page_content for doc in docs]) if docs else "No context found."

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
