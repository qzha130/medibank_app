# Publish the Medibank chatbot

The easiest free hosting option is [Streamlit Community Cloud](https://docs.streamlit.io/deploy/streamlit-community-cloud). This project has a dedicated public entry point, `public_app.py`. Keep `app.py` for the full local app with Vector Database and Logs pages.

## Deploy from GitHub

1. Sign in at [share.streamlit.io](https://share.streamlit.io/) with the GitHub account that owns `qzha130/medibank_app`.
2. Create an app. Select that repository, the pushed branch, and **`public_app.py`** as the main file. Choose **Python 3.12** in Advanced settings.
3. Create a [Gemini API key in Google AI Studio](https://ai.google.dev/gemini-api/docs/api-key). Paste the following into Advanced settings → Secrets, replacing only the key value:

   ```toml
   CHAT_PROVIDER = "gemini"
   EMBEDDING_PROVIDER = "gemini"
   GEMINI_CHAT_MODEL = "gemini-3.8-flash"
   GEMINI_EMBEDDING_MODEL = "gemini-embedding-001"
   GEMINI_API_KEY = "your-key"
   PDF_DIRECTORY = "medibank_data"
   CHROMA_DIRECTORY = "data/chroma"
   ```

4. Click Deploy. The first visit builds the Chroma index from the two bundled documents. Subsequent visitors reuse it. Streamlit provides a public `https://…streamlit.app` URL; choose a memorable available subdomain if desired.

The safe example is also in `.streamlit/secrets.toml.example`. Store a real key only in Streamlit's Secrets settings or a local ignored `.streamlit/secrets.toml`. Do not put it in GitHub. See [Streamlit deployment steps](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy) and [secrets management](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/secrets-management).

## Free hosting and model limits

Community Cloud hosting is free. Google offers a free tier for eligible models with project-specific usage limits. Its current [pricing page](https://ai.google.dev/gemini-api/docs/pricing) lists free use for `gemini-3.8-flash`, but does not list separate pricing for `gemini-embedding-001`. That embedding model remains supported; verify its available quota and cost in your Google AI Studio project before assuming the whole app is free. See [rate limits](https://ai.google.dev/gemini-api/docs/rate-limits). Free-tier input can be used to improve Google's products; this demo asks visitors to avoid personal account and medical details. Paid-tier use has different terms and charges. Free hosting does not make a paid model API free.

Ollama running at `localhost` on your PC cannot be used by Community Cloud: that address refers to the cloud machine. Use Gemini for both chat and embeddings for this deployment. The local app continues to support Ollama. OpenAI is also supported in public mode using server secrets, but requires API billing.

## Chroma and the public interface

Chroma runs inside the app process and stores its index in `data/chroma`; a separate database service is unnecessary for this small demo. Cloud startup rebuilds a missing index using the bundled documents and the configured embedding model. It does not upload your existing Ollama vectors or reuse them with Gemini. The cached Chroma resource is shared, while conversations remain in each browser session.

The public interface has fixed provider settings and does not render API keys, endpoint controls, file paths, uploads, database exports, or shared logs. It registers only the chat page; the local administration pages also reject direct access in public mode. Optional agent activity shows only the current conversation's operation summaries.

An in-memory budget permits at most 60 submitted questions per hour across the app, with a five-second cooldown per browser session. One question can trigger multiple provider requests. This limit resets when the process restarts and is not a durable billing limit or complete abuse protection. Use provider quotas for a firm usage boundary.

Public human fallback directs visitors to Medibank's contact page. It does not submit a case to Medibank or save an unmonitored review ticket. The local app retains its review queue and Logs page. Logs still exist on the host for debugging; they can contain question and answer text. Host files and caches should be treated as disposable, so this deployment is for a demo rather than durable case management.

## Files included in Git

Include application/backend code, local pages and launch scripts, locked runtime dependencies, test sources, documentation, safe configuration examples, and exactly the two files in `medibank_data` required to rebuild the index. Keep these documents current when policy information changes.

Exclude `.env`, real Streamlit secrets, `.venv`, Chroma data, logs, human review records, screenshots, caches, and duplicate PDFs in the project root. `requirements.txt` installs the tested runtime lock; development tools are installed separately with `requirements-dev.txt`. The lock is resolved for Windows and Linux.

## Test the public entry point locally

Copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml`, replace its placeholder key, then run:

```powershell
.\.venv\Scripts\python.exe -m streamlit run public_app.py --server.address 127.0.0.1 --server.port 8503
```

Use a separate process from the local app. For the normal local administration UI, keep using `./run.ps1` on port 8502.

If startup fails, check Community Cloud's owner-only runtime logs and the deployment secret names. Common causes are a missing key, unavailable model or exhausted quota, missing documents, or incompatible Python dependencies. Public visitors see a generic startup message rather than exception details or credentials. Backend module edits require a server restart to load new code.
