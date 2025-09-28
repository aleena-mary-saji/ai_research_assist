import os
import faiss
import pickle
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

# ----------------------------
# Load & Split Documents
# ----------------------------
def load_documents(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    return [Document(page_content=text, metadata={"source": file_path})]

def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,  # smaller chunks → avoids overflow
        chunk_overlap=50
    )
    return text_splitter.split_documents(documents)

# ----------------------------
# Build or Load FAISS Index
# ----------------------------
def build_or_load_faiss(docs, embeddings, index_path="faiss_index"):
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
# Main AI Assistant
# ----------------------------
def main():
    print("\n🔹 AI Research Assistant (CPU-Friendly, Local) 🔹")
    print("Type 'exit' or 'quit' to leave.\n")

    # Load documents
    file_path = "sample_docs.txt"
    documents = load_documents(file_path)
    split_docs = split_documents(documents)

    # Load embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Build/load FAISS
    vectorstore = build_or_load_faiss(split_docs, embeddings)

    # Load model + tokenizer
    model_name = "google/flan-t5-small"  # CPU-friendly
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    # Generation pipeline
    pipe = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=512,      # limit input+output
        max_new_tokens=150,  # short answers
        do_sample=True,
        top_k=50,
        temperature=0.7
    )

    # Interactive Q&A loop
    while True:
        query = input("Ask a question: ")
        if query.lower() in ["exit", "quit"]:
            print("👋 Exiting AI Research Assistant. Goodbye!")
            break

        # Retrieve top-1 relevant document chunk
        retriever = vectorstore.as_retriever(search_kwargs={"k": 1})
        docs = retriever.get_relevant_documents(query)

        context = docs[0].page_content if docs else "No context found."

        # Focused, concise prompt
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
            print("\n🔹 Answer:\n", result)
            print("\n🔹 Sources Metadata:")
            for doc in docs:
                print(doc.metadata)
        except Exception as e:
            print("⚠️ Error during generation:", e)

if __name__ == "__main__":
    main()
