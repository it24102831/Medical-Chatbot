# Deploy this chatbot to Google Cloud Run

This project uses Flask, OpenRouter and Pinecone. It does not use a direct `OPENAI_API_KEY`.

## Before deployment

1. Keep your real `.env` only on your computer.
2. Populate Pinecone once from your local project:

```bash
python store_index.py
```

3. Verify the chatbot locally:

```bash
docker build -t medical-chatbot .
docker run --env-file .env -e PORT=8080 -p 8080:8080 medical-chatbot
```

Open `http://localhost:8080` and `http://localhost:8080/health`, then stop with Ctrl+C.

## Google Cloud setup

Use Google Cloud Shell or a terminal with the Google Cloud CLI.

```bash
gcloud auth login
gcloud config set project YOUR_PROJECT_ID
export PROJECT_ID="YOUR_PROJECT_ID"
export REGION="asia-south1"
export SERVICE_NAME="medical-chatbot"
```

Enable services:

```bash
gcloud services enable \
  run.googleapis.com \
  cloudbuild.googleapis.com \
  artifactregistry.googleapis.com \
  secretmanager.googleapis.com
```

Create secrets:

```bash
printf '%s' 'YOUR_PINECONE_API_KEY' | \
  gcloud secrets create pinecone-api-key --data-file=- --replication-policy=automatic

printf '%s' 'YOUR_OPENROUTER_API_KEY' | \
  gcloud secrets create openrouter-api-key --data-file=- --replication-policy=automatic
```

Create the runtime service account:

```bash
gcloud iam service-accounts create medical-chatbot-sa \
  --display-name="Medical Chatbot Cloud Run Service"

export SERVICE_ACCOUNT="medical-chatbot-sa@${PROJECT_ID}.iam.gserviceaccount.com"
```

Grant access to the secrets:

```bash
gcloud secrets add-iam-policy-binding pinecone-api-key \
  --member="serviceAccount:${SERVICE_ACCOUNT}" \
  --role="roles/secretmanager.secretAccessor"

gcloud secrets add-iam-policy-binding openrouter-api-key \
  --member="serviceAccount:${SERVICE_ACCOUNT}" \
  --role="roles/secretmanager.secretAccessor"
```

Deploy from the folder containing `app.py` and `Dockerfile`:

```bash
gcloud run deploy "$SERVICE_NAME" \
  --source . \
  --region "$REGION" \
  --service-account "$SERVICE_ACCOUNT" \
  --allow-unauthenticated \
  --set-secrets "PINECONE_API_KEY=pinecone-api-key:latest,OPENROUTER_API_KEY=openrouter-api-key:latest" \
  --set-env-vars "INDEX_NAME=medical-chatbot" \
  --memory 2Gi \
  --cpu 2 \
  --concurrency 8 \
  --timeout 300 \
  --max-instances 3 \
  --port 8080
```

Get the URL:

```bash
gcloud run services describe "$SERVICE_NAME" \
  --region "$REGION" \
  --format="value(status.url)"
```

Read logs if deployment fails:

```bash
gcloud run services logs read "$SERVICE_NAME" \
  --region "$REGION" \
  --limit 100
```

## Notes

- `store_index.py` should not run at every Cloud Run startup.
- The `data/` folder is excluded from the deployed image because the PDFs are already stored as vectors in Pinecone.
- The first request may be slow because the Hugging Face embedding model is initialized lazily.
- `--allow-unauthenticated` makes the service public and can create OpenRouter costs. Add authentication and rate limiting before a real public launch.
