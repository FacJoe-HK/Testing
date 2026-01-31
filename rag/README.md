# Audit Checklist RAG (Azure OpenAI + Streamlit)

Minimal RAG app for cybersecurity audit/gap-analysis checklists. Upload internal standards/legal requirements, then generate a checklist via Azure OpenAI.

## Setup
1) Copy `.env.example` to `.env` and fill:
```
AZURE_OPENAI_ENDPOINT=https://<your-endpoint>.cognitiveservices.azure.com/
AZURE_OPENAI_KEY=<your-api-key>
AZURE_OPENAI_API_VERSION=2024-12-01-preview
AZURE_OPENAI_CHAT_DEPLOYMENT=gpt-5.1
AZURE_OPENAI_EMBED_DEPLOYMENT=text-embedding-3-large
```

2) Install deps (recommended in venv):
```
cd rag
pip install -r requirements.txt
```

3) Run Streamlit:
```
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
```

## Usage
- Ingest: Upload PDFs or text/markdown via sidebar and click “Ingest uploaded”.
- Query: Describe the checklist you need (e.g., "ISO27001 Annex A access control gap assessment").
- Generate: Click “Generate checklist” to see bullets; context shown in expander.

## Notes
- Uses Chroma for local vector store (persisted at `rag/.chroma`).
- Embeddings via Azure OpenAI (set deployments in env).
- Temperature fixed low (0.2) for consistency; adjust in `app.py` if needed.
