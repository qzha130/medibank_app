# Medibank Chatbot

A Streamlit chatbot that answers questions using **two Public Information **. It extracts page text, splits it into chunks, and stores embeddings in a persistent Chroma database. A LangChain agent searches those chunks and includes the PDF filename and page in its answers.

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

```dotenv
CHAT_PROVIDER=gemini
GEMINI_API_KEY=your-key
GEMINI_CHAT_MODEL=gemini-3.8-flash
EMBEDDING_PROVIDER=ollama
OLLAMA_EMBEDDING_MODEL=nomic-embed-text
```



```powershell
.\run.ps1
```


## Build the index from PowerShell


```powershell
.\.venv\Scripts\python.exe scripts\build_index.py
```

Use `--pdf-directory` and `--chroma-directory` to change folders. `--embedding-provider`, `--embedding-model`, and `--embedding-url` select the embedding connection; API keys come from `.env` or environment variables. `--chunk-size` and `--chunk-overlap` default to 1000 and 200 characters.

    