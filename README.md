# DocAI Pro

DocAI Pro is a Streamlit-based RAG chatbot that lets users upload PDF, TXT, and DOCX documents, then ask questions that are answered strictly from the uploaded content. It combines local sentence-transformer embeddings, FAISS retrieval, and the Anthropic Messages API into a single-file application suitable for demos, internal tools, or client deliverables in the AI automation space.

## Features

- Multi-file upload: PDF (pdfplumber + pypdf), TXT, and DOCX (python-docx).
- Grounded answers: the system prompt directs the model to answer using ONLY the retrieved document context and to say it does not know when the answer is not present.
- Retrieval pipeline: documents are chunked at 800 characters with 150 characters of overlap, embedded locally with `all-MiniLM-L6-v2`, and indexed in FAISS (`IndexFlatL2`). The top 4 chunks are retrieved per question.
- Streaming answers from the Anthropic Messages API, with `max_tokens=1024`, using the official `anthropic` SDK pointed at an OpenAI/Anthropic-compatible gateway via a configurable `base_url`.
- Editable system prompt with four presets: Default, Formal, Casual, and Expert.
- Model selector; the available list can be extended with the `CLAUDE_MODEL` environment variable.
- Chat history kept across turns in the same session, with a reset button.
- Privacy: embeddings are computed locally, so document text is never sent to a third party for indexing; only the small retrieved context is sent to the model per question.

## How it works

```
                         Streamlit UI
                              |
                              v
upload PDF/TXT/DOCX  -->  text extraction  -->  chunking (800/150)
                              |
                              v
              local embeddings (all-MiniLM-L6-v2)
                              |
                              v
                       FAISS IndexFlatL2
                              |
                              v
                    retrieve top 4 chunks
                              |
                              v
                  prompt with context  -->  streaming answer
```

## Tech stack

- Python 3.10+
- Streamlit
- Anthropic Python SDK
- sentence-transformers
- FAISS (faiss-cpu)
- pdfplumber, pypdf
- python-docx
- python-dotenv
- NumPy

## Quick start

1. Install dependencies. The first run downloads a ~90 MB embedding model once.
   ```bash
   pip install -r requirements.txt
   ```

2. Copy the example environment file and add your Anthropic-compatible API key.
   ```bash
   cp .env.example .env
   ```
   Then set `ANTHROPIC_API_KEY` in `.env`.

3. Run the app.
   ```bash
   streamlit run app.py
   ```
   The browser opens at `http://localhost:8501`.

## Configuration

| Variable | Required | Description |
| --- | --- | --- |
| `ANTHROPIC_API_KEY` | Yes | API key for the Anthropic-compatible gateway. |
| `CLAUDE_MODEL` | No | Extra model ID appended to the model selector. |

## Customization

| What to change | Where |
| --- | --- |
| System prompt / tone | Sidebar text area or `DEFAULT_SYSTEM_PROMPT` in `app.py`. |
| Available models | `MODEL_OPTIONS` in `app.py` or the `CLAUDE_MODEL` env variable. |
| App title and icon | `APP_TITLE` and `APP_ICON` in `app.py`. |
| Chunk size and retrieval depth | `CHUNK_SIZE`, `CHUNK_OVERLAP`, and `TOP_K` in `app.py`. |
| Answer length | `MAX_TOKENS` in `app.py`. |

## Try it

A sample document, `data.txt` (a fictional ACME Cloud FAQ), is included. Upload it and ask questions such as:

- "How much does the Business plan cost?"
- "What encryption do you use?"
- "How long are deleted files kept on the Free plan?"

## Deployment

Run locally with `streamlit run app.py`. For hosted deployment on Streamlit Community Cloud or any VPS, set the API key as an environment variable or secret rather than relying on a `.env` file.

## Project structure

| File | Purpose |
| --- | --- |
| `app.py` | Complete Streamlit application: document loading, chunking, embeddings, FAISS retrieval, and streaming chat UI. |
| `requirements.txt` | Python dependencies. |
| `.env.example` | Template for environment variables. |
| `data.txt` | Sample document for testing. |
| `README.md` | This file. |

## Notes and limitations

- Scanned or image-only PDFs contain no extractable text and require OCR first.
- The FAISS index is held in memory and rebuilt per session, so nothing persists between runs.
- Keep the API key out of version control; `.env` should be gitignored.
