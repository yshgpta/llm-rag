# Gemini Document RAG

A Streamlit application for uploading a PDF or image, building a Chroma-backed RAG index with Gemini embeddings, chatting with the uploaded document, tracing the workflow in Langfuse, and evaluating responses with RAGAS.

## What This App Does

1. Upload a PDF, JPG, JPEG, or PNG.
2. Extract text from the document.
3. Split the text into smaller chunks.
4. Create Gemini embeddings for each chunk.
5. Store the chunks and vectors in Chroma.
6. Ask questions in a chat UI.
7. Retrieve the most relevant chunks from Chroma.
8. Ask Gemini to answer using only the retrieved document context.
9. Trace indexing, retrieval, and chat operations in Langfuse.
10. Optionally run RAGAS metrics over completed chat turns.

## Main Technologies

### Streamlit

Streamlit powers the web UI in `app.py`.

It provides:

- Sidebar configuration for Gemini, Langfuse, model names, and Chroma path
- Document upload
- Document indexing button
- Chat tab
- Evaluation tab
- Status messages, errors, source expanders, and dataframes

### Gemini

Gemini is the LLM provider.

The app uses Gemini for:

- Chat answers through `ChatGoogleGenerativeAI`
- Embeddings through `GoogleGenerativeAIEmbeddings`
- RAGAS evaluator calls through Gemini's OpenAI-compatible endpoint

Default models:

- Chat: `gemini-3.5-flash`
- Embeddings: `models/gemini-embedding-001`

### LangChain

LangChain provides the model wrappers and document utilities.

Used for:

- Gemini chat model wrapper
- Gemini embeddings wrapper
- `Document` objects
- Recursive text splitting
- Callback integration with Langfuse

Main file: `src/rag.py`

### RAG

RAG means Retrieval-Augmented Generation.

In this project:

- The uploaded document is split into chunks.
- Each chunk is embedded into a vector.
- Chroma stores the chunk vectors.
- A user question is embedded and matched against the stored chunks.
- Gemini receives the retrieved chunks as context.
- The answer is generated from the uploaded document context only.

### Chroma

Chroma is the local vector database.

Used for:

- Storing document chunks and embeddings
- Running similarity search for each user question

By default, Chroma data is stored in `chroma_db/`, which is ignored by git.

### Langfuse

Langfuse provides observability and tracing.

The app traces:

- `document-index`: document upload, text extraction, chunk creation, and vector indexing
- `document-chat`: each chat turn
- `retrieve-document-context`: retrieval step inside a chat turn
- Gemini/LangChain LLM calls via Langfuse callback handler

Trace metadata includes:

- Document name
- Chat model
- Embedding model
- Session id
- Tags such as `rag`, `chat`, `index`, and `streamlit`
- Input question and output answer
- Source count, latency, and cost when available

Langfuse regions:

- EU: `https://cloud.langfuse.com`
- US: `https://us.cloud.langfuse.com`
- Japan: `https://jp.cloud.langfuse.com`
- HIPAA: `https://hipaa.cloud.langfuse.com`

Set `LANGFUSE_HOST` to the region where your project keys were created. If the dashboard is empty, check that you are logged into the same region as your keys.

### RAGAS

RAGAS evaluates RAG quality.

The app currently runs:

- `faithfulness`: whether the answer is supported by retrieved context
- `answer_relevancy`: whether the answer addresses the user question

The evaluator uses:

- Gemini through an async OpenAI-compatible client
- Google Gemini embeddings
- trimmed retrieved contexts to avoid evaluator output truncation

Main file: `src/evaluation.py`

### PDF and Image Loading

Document loading lives in `src/document_loader.py`.

Supported inputs:

- PDF: text extraction with `pypdf`
- JPG/JPEG/PNG: OCR with `pytesseract` and image handling with `Pillow`

For scanned PDFs, convert pages to images or OCR them before upload.

## Project Structure

```text
.
├── app.py
├── requirements.txt
├── packages.txt
├── runtime.txt
├── README.md
├── .env.example
├── .gitignore
├── .streamlit
│   ├── config.toml
│   └── secrets.toml.example
├── src
│   ├── config.py
│   ├── document_loader.py
│   ├── evaluation.py
│   ├── monitoring.py
│   └── rag.py
└── tests
    ├── test_config.py
    ├── test_rag_helpers.py
    └── test_retryable_embeddings.py
```

## Setup

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Copy the environment template:

```bash
cp .env.example .env
```

Add your keys to `.env`:

```bash
GOOGLE_API_KEY=your_gemini_api_key

LANGFUSE_PUBLIC_KEY=your_langfuse_public_key
LANGFUSE_SECRET_KEY=your_langfuse_secret_key
LANGFUSE_HOST=https://jp.cloud.langfuse.com
```

Use the correct Langfuse region for your project. The example above uses Japan.

For OCR support on macOS:

```bash
brew install tesseract
```

## Local SSL Note

Some local machines with corporate or self-signed TLS interception can fail Python certificate validation when exporting traces to Langfuse.

For local development only, set:

```bash
LANGFUSE_VERIFY_SSL=false
```

Keep SSL verification enabled in production.

## Run

```bash
streamlit run app.py
```

Open:

```text
http://localhost:8501
```

## Streamlit Cloud Deployment

This repo includes the files Streamlit Cloud expects:

- `requirements.txt` for Python dependencies
- `packages.txt` for the Tesseract OCR system package
- `runtime.txt` to pin Python 3.11
- `.streamlit/config.toml` for Streamlit server defaults
- `.streamlit/secrets.toml.example` as a safe template for deployment secrets

Deploy settings:

```text
Repository: yshgpta/llm-rag
Branch: main
Main file path: app.py
```

In Streamlit Cloud, add these app secrets:

```toml
GOOGLE_API_KEY = "your_gemini_api_key"

LANGFUSE_PUBLIC_KEY = "your_langfuse_public_key"
LANGFUSE_SECRET_KEY = "your_langfuse_secret_key"
LANGFUSE_HOST = "https://cloud.langfuse.com"
LANGFUSE_VERIFY_SSL = true

GEMINI_CHAT_MODEL = "gemini-3.5-flash"
GEMINI_EMBEDDING_MODEL = "models/gemini-embedding-001"
CHROMA_DIR = "/tmp/chroma_db"
```

Use the Langfuse host for the region where your project keys were created. For example, `https://cloud.langfuse.com` is EU and `https://jp.cloud.langfuse.com` is Japan.

Streamlit Cloud storage is ephemeral. Uploaded files and Chroma indexes are created at runtime and can disappear when the app restarts, so upload and index the document again after a restart.

## Usage

1. Open the app.
2. Confirm Gemini and Langfuse settings in the sidebar.
3. Upload a PDF or image.
4. Click `Index document`.
5. Open the `Chat` tab.
6. Ask questions about the uploaded document.
7. Open the `Evaluate` tab.
8. Click `Run RAGAS evaluation`.

## Configuration

| Variable | Required | Purpose |
| --- | --- | --- |
| `GOOGLE_API_KEY` | Yes | Gemini chat, embeddings, and evaluator calls |
| `LANGFUSE_PUBLIC_KEY` | Optional | Langfuse tracing |
| `LANGFUSE_SECRET_KEY` | Optional | Langfuse tracing |
| `LANGFUSE_HOST` | Optional | Langfuse region URL |
| `LANGFUSE_VERIFY_SSL` | Optional | Local TLS troubleshooting |
| `GEMINI_CHAT_MODEL` | Optional | Gemini chat model override |
| `GEMINI_EMBEDDING_MODEL` | Optional | Gemini embedding model override |
| `CHROMA_DIR` | Optional | Local Chroma storage directory |

Configuration is loaded in this order:

1. Values entered in the Streamlit sidebar
2. Environment variables or local `.env`
3. Streamlit Cloud secrets
4. Built-in defaults

## Tests

Run:

```bash
python -m pytest -q
```

Current tests cover:

- Config defaults
- Document splitting metadata
- Chroma collection naming
- Chat response text extraction
- Gemini embedding retry behavior
- RAGAS evaluator construction
- Evaluation context trimming

## Troubleshooting

### No Langfuse Traces

Check:

- `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` are set.
- `LANGFUSE_HOST` matches the region where the keys were created.
- You are viewing the same region in the Langfuse dashboard.
- The app has run an indexed chat turn after the key was configured.

Region examples:

- EU dashboard: `https://cloud.langfuse.com`
- Japan dashboard: `https://jp.cloud.langfuse.com`

### Gemini Model Not Found

If Gemini returns a model not found error, select another model in the sidebar or set:

```bash
GEMINI_CHAT_MODEL=gemini-3.5-flash
GEMINI_EMBEDDING_MODEL=models/gemini-embedding-001
```

### RAGAS Output Truncated

The evaluator uses a larger token budget and trims evaluation contexts. If this still happens, reduce the number or size of retrieved contexts or use a Gemini model with a larger reliable structured-output budget.

### OCR Not Working

Install Tesseract:

```bash
brew install tesseract
```

Then restart Streamlit.

## Security Notes

- Do not commit `.env`.
- Do not hardcode API keys.
- `chroma_db/` is ignored because it contains generated local vector data.
- `LANGFUSE_VERIFY_SSL=false` is only for local development in environments with TLS interception.
