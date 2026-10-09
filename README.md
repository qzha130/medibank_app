# Medibank Chatbot

A Streamlit chatbot that answers questions using **two PDFs**. It extracts page text, splits it into chunks, and stores embeddings in a persistent Chroma database. A LangChain agent searches those chunks and includes the PDF filename and page in its answers.

For a public website, deploy **`public_app.py`** using the [free hosting guide](DEPLOYMENT.md). It prepares Chroma automatically and keeps server credentials and administrative pages out of the public chat interface. Use **`app.py`** for the full local interface.

Choose the chat model and embedding model independently: use local Ollama, online OpenAI, online Gemini, or mix providers. Local defaults are `llama3.1:8b` for chat and `nomic-embed-text` for embeddings. OpenAI defaults are `gpt-4.1-mini` and `text-embedding-3-small`. Gemini defaults are the stable [`gemini-3.8-flash`](https://ai.google.dev/gemini-api/docs/models/gemini-3.8-flash) chat model and `gemini-embedding-001` for embeddings.

## Run on Windows

Open PowerShell in this project and run setup once. It creates a Python 3.12 environment, installs the tested dependencies, and copies `.env.example` to `.env` only when that file does not already exist:

```powershell
.\setup.ps1
```

For local models, install [Ollama](https://ollama.com/download/windows) and download both models:

```powershell
ollama pull llama3.1:8b
ollama pull nomic-embed-text
```

Ollama must be running. If its desktop app has not started the service, run `ollama serve` in a separate terminal. The default address is `http://localhost:11434`.

Use `ollama list` to inspect installed models. Running the two `ollama pull` commands again refreshes those tags. If embedding weights change, use a fresh `CHROMA_DIRECTORY` and build the index again so stored vectors and query embeddings use matching weights.

For OpenAI, set `OPENAI_API_KEY` in `.env` or enter it in the app's sidebar. API billing must be configured. Keep `.env` private.

For Gemini, [create an API key in Google AI Studio](https://ai.google.dev/gemini-api/docs/api-key), then set `GEMINI_API_KEY` in `.env` or enter it in the sidebar. `GOOGLE_API_KEY` is also accepted as a fallback. Select **Gemini · online** as the chat provider and keep **Ollama · local** as the embedding provider to use Gemini answers with your existing local Chroma index:

```dotenv
CHAT_PROVIDER=gemini
GEMINI_API_KEY=your-key
GEMINI_CHAT_MODEL=gemini-3.8-flash
EMBEDDING_PROVIDER=ollama
OLLAMA_EMBEDDING_MODEL=nomic-embed-text
```

To use Gemini embeddings as well, open **Vector Database**, select **Gemini · online** as the index embedding provider, use `gemini-embedding-001`, and build a new index. Select matching embedding settings on the chatbot page to query that index. This integration uses the Gemini Developer API through Google AI Studio. The current LangChain adapter supports `gemini-embedding-001` and rejects Gemini Embedding 2. Keys stay with their selected online provider.

If Gemini reports a Vertex AI or Enterprise configuration conflict, unset `GOOGLE_GENAI_USE_VERTEXAI` and `GOOGLE_GENAI_USE_ENTERPRISE`, or set them to `false`, then restart Streamlit.

The default PDF folder is `medibank_data`. Git includes one copy of the two knowledge documents so the public app can rebuild Chroma. Root-level duplicate PDFs and generated indexes are excluded. Manage PDF folders, uploads, and indexing in the local **Vector Database** page. Each PDF can be up to 50 MB and must have a different filename. Start the UI:

```powershell
.\run.ps1
```

Open [http://127.0.0.1:8502](http://127.0.0.1:8502) in your browser. Use `.\run.ps1 -Port 8503` to choose another port. The chatbot automatically loads a compatible saved index using your embedding settings; choose your chat model and ask a question. Use PDFs with selectable text; scanned pages need OCR before importing.

The chat screen presents general Medibank questions with model and connection settings collapsed by default. Turn on **Show agent activity** in the sidebar to see each new reply's scope checks, knowledge searches, source checks, and response timing; turn it off to hide them. Activity uses operation logs scoped to the current request and browser session. Replies retain their activity summaries while you navigate between pages, and failed attempts can also be inspected. The separate **Logs** page keeps the full application events available.

After editing backend Python modules, restart Streamlit to ensure the running process loads the updated code.

## Build the index from PowerShell

You can index the two PDFs without opening Streamlit. The command reads the same `.env` settings and reuses an existing complete index when its documents and embedding settings match:

```powershell
.\.venv\Scripts\python.exe scripts\build_index.py
```

Use `--pdf-directory` and `--chroma-directory` to change folders. `--embedding-provider`, `--embedding-model`, and `--embedding-url` select the embedding connection; API keys come from `.env` or environment variables. `--chunk-size` and `--chunk-overlap` default to 1000 and 200 characters. Run with `--help` for all options.

## Review the vector database

Open **Vector Database** from Streamlit's navigation sidebar. In **Add or update PDFs**, choose a local folder or upload the two PDFs, select the index embedding provider/model/server and chunk settings, then click **Build / load knowledge base**. These controls are available before any index exists. Indexing starts only when you click the button. A successful build selects that index for chat.

The page also reads existing Chroma indexes directly: browsing does not need an API key, an embedding service, or a chat model. Browsing another saved index leaves the chat selection unchanged; click **Use in chat** to select it explicitly, then use matching embedding provider/model/server settings on the chatbot page.

Choose a saved index to review its document counts, physical pages, and stored passages. Filter records by document and physical page; page `0` includes every page. Paginate through the chunks, then select a record to see its full text and metadata. **Load vector values** displays the vector's dimensions, coordinates, norm, and plot. Click **Prepare filtered export** to download all matching records as JSON or CSV, up to 10,000 passages.

Semantic search embeds your query and compares it with the saved vectors. Select the same embedding provider, model, and server used to build that index; the page disables search when their identity differs or the required API key is missing. Search results show the stored text and similarity distance. This search uses the embedding model and Chroma without calling an LLM.

## Logs and human review

Open **Logs** from the navigation sidebar to inspect application events and handle local human review tickets. Events are saved in `data/logs/events.jsonl`, with automatic rotation and API credentials redacted. Review tickets are saved in `data/handoffs/requests.jsonl`; mark them pending, reviewed, or resolved and record notes. `LOG_DIRECTORY` and `HANDOFF_DIRECTORY` can change these locations.

The chatbot checks question scope and retrieved evidence before calling the LLM. Off-topic questions and requests to override its instructions receive a scoped fallback. Explicit human requests, account or clinical questions, and questions with weak or unavailable evidence create local review tickets.

Set `GUARDRAIL_MAX_DISTANCE` in `.env` to adjust the relevance gate; the default is `0.35`. It is a cosine-distance threshold: lower values require closer retrieved evidence. Human fallback creates a ticket in this app's local queue for you to review.

## How it works

```text
2 PDFs → page text → chunks → embedding model → persistent Chroma
Question → scope / evidence guardrail → LangChain agent → PDF search tool → answer + sources
Human request / insufficient evidence → local human review queue
```

The index is reused for the same documents and embedding settings. Changing either creates a separate index so vectors from different embedding models are not mixed. Switching only the chat model does not require re-embedding the PDFs. Retrieved chunks are expanded with surrounding text from the same physical page, within the context limit, to preserve nearby conditions and exceptions. Saved indexes contain extracted PDF text in `data/chroma`; set `CHROMA_DIRECTORY` in `.env` to use another location.

The supplied pair of PDFs has **88 pages**, indexed as **303 passages** with the default chunk settings. Chroma runs as an embedded persistent database in this app; no separate Chroma server is needed. It stores both vectors and their source text. Document questions search it before the chat model answers; off-topic questions and human requests are handled before calling the LLM. Gemini chat with Ollama embeddings reuses this index. Changing the embedding provider or model requires building its matching index.

Informative answers require a retrieved filename/page citation. If a model omits citations, it gets one revision with search disabled; an unsuccessful revision produces a clear error. Citation checks validate source labels rather than the meaning of every claim. Conversation and evidence limits keep local model prompts manageable.

With online embeddings, PDF chunks are sent to the selected provider, OpenAI or Google. With online chat, questions, conversation history, and retrieved excerpts are sent to the selected chat provider. Local mode sends this text to the configured Ollama server; use a local address to keep inference on your computer. Source references show retrieved evidence, and do not guarantee an answer is correct. Check the cited pages for consequential decisions.

The public entry point is an anonymous demo with fixed server settings, session conversations, a small shared request budget, and contact-based human fallback. It disables administration and shared log access. Durable account services or case management need additional authentication and storage; see [deployment limitations](DEPLOYMENT.md).

## Checks

```powershell
.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.venv\Scripts\python.exe -m pytest -q
```

The tests generate small searchable PDFs and use deterministic fake embeddings/models. They do not require Ollama, an API key, or network access.

`requirements.txt` loads `requirements.lock.txt`, the pinned runtime dependency set resolved for Windows and Linux. `requirements.in` lists its direct packages; development tools are separate in `requirements-dev.txt`.

Official references: [LangChain agents](https://docs.langchain.com/oss/python/langchain/agents), [LangChain Chroma integration](https://docs.langchain.com/oss/python/integrations/vectorstores/chroma), [Ollama quickstart](https://docs.ollama.com/quickstart), [OpenAI embeddings](https://developers.openai.com/api/docs/guides/embeddings), and [Gemini API keys](https://ai.google.dev/gemini-api/docs/api-key).
