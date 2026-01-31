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

# Defaults from env (optional)
DEFAULT_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "")
DEFAULT_KEY = os.getenv("AZURE_OPENAI_KEY", "")
DEFAULT_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
DEFAULT_CHAT_DEPLOY = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-5.1")
DEFAULT_EMBED_DEPLOY = os.getenv("AZURE_OPENAI_EMBED_DEPLOYMENT", "text-embedding-3-large")

# Session config helpers
if "cfg" not in st.session_state:
    st.session_state.cfg = {
        "endpoint": DEFAULT_ENDPOINT,
        "key": DEFAULT_KEY,
        "api_version": DEFAULT_API_VERSION,
        "chat": DEFAULT_CHAT_DEPLOY,
        "embed": DEFAULT_EMBED_DEPLOY,
    }
if "client" not in st.session_state:
    st.session_state.client = None


def set_client():
    cfg = st.session_state.cfg
    if not cfg["endpoint"] or not cfg["key"]:
        st.session_state.client = None
        return
    st.session_state.client = AzureOpenAI(
        api_version=cfg["api_version"],
        azure_endpoint=cfg["endpoint"],
        api_key=cfg["key"],
    )


def current_client():
    return st.session_state.client


# Chroma setup
PERSIST_DIR = str(Path(__file__).parent / ".chroma")
chroma_client = chromadb.PersistentClient(path=PERSIST_DIR)


def azure_embed(texts):
    client = current_client()
    if client is None:
        raise RuntimeError("No Azure client configured. Set endpoint/key in sidebar.")
    if isinstance(texts, str):
        texts = [texts]
    resp = client.embeddings.create(model=st.session_state.cfg["embed"], input=texts)
    return [d.embedding for d in resp.data]


class AzureEmbeddingFunction:
    def __call__(self, texts):
        return azure_embed(texts)


aure_emb_fn = AzureEmbeddingFunction()
collection = chroma_client.get_or_create_collection("kb", embedding_function=aure_emb_fn)


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
    client = current_client()
    if client is None:
        raise RuntimeError("No Azure client configured. Set endpoint/key in sidebar.")
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
        model=st.session_state.cfg["chat"],
        messages=messages,
        max_completion_tokens=800,
        temperature=0.2,
    )
    return resp.choices[0].message.content


st.set_page_config(page_title="Audit Checklist RAG", layout="wide")
st.title("Audit Checklist RAG")

# Sidebar config inputs
st.sidebar.header("Azure OpenAI config")
endpoint = st.sidebar.text_input("Endpoint", value=st.session_state.cfg["endpoint"], placeholder="https://<your-endpoint>.cognitiveservices.azure.com/")
key = st.sidebar.text_input("API Key", value=st.session_state.cfg["key"], type="password")
api_version = st.sidebar.text_input("API Version", value=st.session_state.cfg["api_version"])
chat_dep = st.sidebar.text_input("Chat deployment", value=st.session_state.cfg["chat"])
embed_dep = st.sidebar.text_input("Embed deployment", value=st.session_state.cfg["embed"])
if st.sidebar.button("Save config"):
    st.session_state.cfg = {
        "endpoint": endpoint.strip(),
        "key": key.strip(),
        "api_version": api_version.strip(),
        "chat": chat_dep.strip(),
        "embed": embed_dep.strip(),
    }
    set_client()
    st.sidebar.success("Config applied (session only)")

if st.session_state.client is None:
    set_client()

st.sidebar.header("Ingest docs")
uploaded = st.sidebar.file_uploader("Upload PDFs or text", type=["pdf", "txt", "md"], accept_multiple_files=True)
if st.sidebar.button("Ingest uploaded") and uploaded:
    try:
        ingest(uploaded)
        st.sidebar.success("Ingested")
    except Exception as e:
        st.sidebar.error(f"Ingest failed: {e}")

query = st.text_area("Describe the checklist you need", height=120, placeholder="e.g., ISO27001 Annex A gap assessment for access control")
if st.button("Generate checklist"):
    try:
        results = search(query, k=5)
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        context = "\n---\n".join(f"{meta.get('source','')}\n{doc}" for doc, meta in zip(docs, metas))
        checklist = generate_checklist(query, context)
        st.subheader("Checklist")
        st.write(checklist)

        with st.expander("Context used"):
            st.write(context)
    except Exception as e:
        st.error(f"Generation failed: {e}")

st.sidebar.header("Status")
st.sidebar.write(f"Endpoint set: {'yes' if st.session_state.cfg['endpoint'] else 'no'}")
st.sidebar.write(f"Chat deployment: {st.session_state.cfg['chat']}")
st.sidebar.write(f"Embed deployment: {st.session_state.cfg['embed']}")
