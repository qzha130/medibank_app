# Medibank Chatbot

A Streamlit chatbot that answers questions using **two PDFs**. It extracts page text, splits it into chunks, and stores embeddings in a persistent Chroma database. A LangChain agent searches those chunks and includes the PDF filename and page in its answers.

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

The chat screen presents general Medibank questions with model and connection settings collapsed by default. Turn on **Show processing steps** before asking a question to watch a scrolling window update while the LLM works. It shows checks, searches, model progress, and reasoning text as the model returns it. Turn the option off to hide the window. Saved steps can be shown later without calling the model again. Supported Gemini models return thinking summaries; models that do not supply reasoning still show their real processing activity. The separate Sources panel is hidden; answers retain their inline references. Reasoning stays separate from answers, downloaded conversations, application logs, and later conversation turns. Processing windows are scoped to the current request and browser session and show the latest 8,000 characters. Failed attempts can also be inspected. The separate **Logs** page keeps the full application events available.

After editing backend Python modules, restart Streamlit to ensure the running process loads the updated code.

## Build the index from PowerShell

You can index the two PDFs without opening Streamlit. The command reads the same `.env` settings and reuses an existing complete index when its documents and embedding settings match:

```powershell
.\.venv\Scripts\python.exe scripts\build_index.py
```

Use `--pdf-directory` and `--chroma-directory` to change folders. `--embedding-provider`, `--embedding-model`, and `--embedding-url` select the embedding connection; API keys come from `.env` or environment variables. `--chunk-size` and `--chunk-overlap` default to 1000 and 200 characters.

    