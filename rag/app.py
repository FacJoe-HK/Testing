import os
import tempfile
from pathlib import Path

import chromadb
import streamlit as st
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
from openai import AzureOpenAI
from pypdf import PdfReader

# Load environment
load_dotenv()

AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "")
AZURE_KEY = os.getenv("AZURE_OPENAI_KEY", "")
AZURE_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
CHAT_DEPLOYMENT = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-5.1")
EMBED_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBED_DEPLOYMENT", "text-embedding-3-large")

if not AZURE_ENDPOINT:
    st.warning("Set Azure OpenAI env vars in .env")

client = AzureOpenAI(
    api_version=AZURE_API_VERSION,
    azure_endpoint=AZURE_ENDPOINT,
    api_key=AZURE_KEY,
)

# Chroma setup
PERSIST_DIR = str(Path(__file__).parent / ".chroma")
chroma_client = chromadb.PersistentClient(path=PERSIST_DIR)

# Embedding function using Azure OpenAI

def azure_embed(texts):
    if isinstance(texts, str):
        texts = [texts]
    resp = client.embeddings.create(model=EMBED_DEPLOYMENT, input=texts)
    return [d.embedding for d in resp.data]


embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction("all-MiniLM-L6-v2")
# Override with Azure embed
class AzureEmbeddingFunction:
    def __call__(self, texts):
        return azure_embed(texts)

azure_embedding_fn = AzureEmbeddingFunction()
collection = chroma_client.get_or_create_collection("kb", embedding_function=azure_embedding_fn)


def read_file(uploaded_file):
    if uploaded_file.name.lower().endswith(".pdf"):
        reader = PdfReader(uploaded_file)
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
    else:
        text = uploaded_file.read().decode("utf-8", errors="ignore")
    return text


def chunk_text(text, chunk_size=800, overlap=200):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i : i + chunk_size]
        chunks.append(" ".join(chunk))
        i += chunk_size - overlap
    return chunks


def ingest(uploaded_files):
    docs = []
    for f in uploaded_files:
        text = read_file(f)
        for i, chunk in enumerate(chunk_text(text)):
            docs.append((f"{f.name}-{i}", chunk, {"source": f.name}))
    if docs:
        ids, texts, metas = zip(*docs)
        collection.upsert(ids=ids, documents=texts, metadatas=metas)


def search(query, k=5):
    return collection.query(query_texts=[query], n_results=k)


def generate_checklist(query, passages):
    messages = [
        {
            "role": "system",
            "content": "You are a cybersecurity audit assistant. Use provided context to draft a concise checklist (bullets)."
        },
        {
            "role": "user",
            "content": f"Context:\n{passages}\n\nTask: {query}\nReturn bullet checklist only."
        },
    ]
    resp = client.chat.completions.create(
        model=CHAT_DEPLOYMENT,
        messages=messages,
        max_completion_tokens=800,
        temperature=0.2,
    )
    return resp.choices[0].message.content


st.set_page_config(page_title="Audit Checklist RAG", layout="wide")
st.title("Audit Checklist RAG")

st.sidebar.header("Ingest docs")
uploaded = st.sidebar.file_uploader("Upload PDFs or text", type=["pdf", "txt", "md"], accept_multiple_files=True)
if st.sidebar.button("Ingest uploaded") and uploaded:
    ingest(uploaded)
    st.sidebar.success("Ingested")

query = st.text_area("Describe the checklist you need", height=120, placeholder="e.g., ISO27001 Annex A gap assessment for access control")
if st.button("Generate checklist"):
    results = search(query, k=5)
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    context = "\n---\n".join(f"{meta.get('source','')}\n{doc}" for doc, meta in zip(docs, metas))
    checklist = generate_checklist(query, context)
    st.subheader("Checklist")
    st.write(checklist)

    with st.expander("Context used"):
        st.write(context)

st.sidebar.header("Status")
st.sidebar.write(f"Endpoint set: {'yes' if AZURE_ENDPOINT else 'no'}")
st.sidebar.write(f"Chat deployment: {CHAT_DEPLOYMENT}")
st.sidebar.write(f"Embed deployment: {EMBED_DEPLOYMENT}")
