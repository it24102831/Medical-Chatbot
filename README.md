# Medical Chatbot (Render 512MB-safe)

This version removes local embedding-model inference from the web process. Pinecone integrated embeddings handle both ingestion and query embedding.

## 1) Create virtual environment
```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

## 2) Install web dependencies
```bash
pip install -r requirements.txt
```

## 3) Install ingestion dependencies
```bash
pip install -r requirements-ingest.txt
```

## 4) Create environment variables
```bash
cp .env.example .env
# then edit .env with real keys
```

Required:
- `PINECONE_API_KEY`
- `PINECONE_INDEX_NAME` (default `medical-chatbot-v2`)
- `PINECONE_NAMESPACE` (default `medical-book`)
- `PINECONE_EMBED_MODEL` (default `llama-text-embed-v2`)
- `OPENROUTER_API_KEY`
- `OPENROUTER_MODEL` (default `openai/gpt-4o-mini`)
- `MAX_CONTEXT_CHARS` (default `12000`)
- `MAX_INPUT_CHARS` (default `2000`)
- `PYTHON_VERSION` (default `3.11.9`)

## 5) Pinecone migration (required)
Because embeddings moved from local `all-MiniLM-L6-v2` to hosted embeddings, **do not reuse the old 384-dim index**.
Use a new index name (example: `medical-chatbot-v2`).

## 6) Create/populate the integrated-embedding index
Place PDFs in `data/`, then run:
```bash
python store_index.py
```
Optional full namespace refresh:
```bash
CLEAR_NAMESPACE=true python store_index.py
```

## 7) Verify uploaded records
- Check script summary output (`records uploaded`, `index`, `namespace`, `embedding model`).
- In Pinecone console, confirm records in the configured namespace.

## 8) Run Flask locally
```bash
python app.py
```

## 9) Run Gunicorn locally
```bash
PORT=8000 gunicorn --bind 0.0.0.0:$PORT --workers 1 --threads 2 --timeout 120 --access-logfile - --error-logfile - app:app
```

## 10) Deploy to Render (Web Service)
- Build command:
```bash
python -m pip install --upgrade pip && python -m pip install --no-cache-dir -r requirements.txt
```
- Start command:
```bash
gunicorn --bind 0.0.0.0:$PORT --workers 1 --threads 2 --timeout 120 --access-logfile - --error-logfile - app:app
```
- Health check path: `/health`

## 11) Configure Render env vars
Set:
- `PYTHON_VERSION=3.11.9`
- `WEB_CONCURRENCY=1`
- `PYTHONUNBUFFERED=1`
- `MALLOC_ARENA_MAX=2`
- `PINECONE_API_KEY`
- `PINECONE_INDEX_NAME=medical-chatbot-v2`
- `PINECONE_NAMESPACE=medical-book`
- `PINECONE_EMBED_MODEL=llama-text-embed-v2`
- `OPENROUTER_API_KEY`
- `OPENROUTER_MODEL=openai/gpt-4o-mini`
- `MAX_CONTEXT_CHARS=12000`
- `MAX_INPUT_CHARS=2000`
- `TOP_K=3`

## 12) Test health and readiness
```bash
curl -s http://127.0.0.1:8000/health
curl -s http://127.0.0.1:8000/ready
curl -s -X POST http://127.0.0.1:8000/get -H 'Content-Type: application/json' -d '{"msg":"What is hypertension?"}'
```

## 13) Troubleshooting
- **Memory-limit restarts**: verify one worker, runtime uses `requirements.txt` only, and no torch/sentence-transformers installed.
- **502 errors**: check OpenRouter/Pinecone auth, provider status, and app logs.
- **Empty index**: rerun `python store_index.py` and verify records uploaded.
- **Wrong namespace**: ensure `PINECONE_NAMESPACE` matches ingestion namespace.
- **Invalid API keys**: update `PINECONE_API_KEY`/`OPENROUTER_API_KEY`.
- **Pinecone rate limits**: reduce request rate, keep `TOP_K` low.
- **OpenRouter failures**: verify model name, retries/timeouts, and provider availability.
