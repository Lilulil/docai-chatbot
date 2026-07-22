# 📄 AI Document Chatbot

A ready-to-sell, web-based AI chatbot that answers questions **grounded in the documents your client uploads**. Upload PDF / TXT / DOCX files and chat with them — powered by Claude and a simple built-in RAG (retrieval-augmented generation) pipeline.

Built with **Streamlit** so it runs in the browser with zero front-end work.

---

## ✨ Features

- **Multi-file upload** — PDF, TXT, and DOCX (multiple files at once)
- **Chat grounded in the documents** — answers come from the uploaded content, not made-up
- **Simple RAG** — documents are split into chunks, embedded locally, and the most relevant pieces are retrieved per question (handles large files)
- **Customizable system prompt** — set the assistant's persona (e.g. *"You are a friendly customer-service assistant"*)
- **Model choice** — switch between Claude Haiku (fast & cheap) and Claude Sonnet (smarter)
- **Chat history** — the conversation is shown in the main window and remembered across turns
- **Reset button** — start a fresh conversation anytime
- **Clean, professional UI** — logo, title, and a tidy sidebar

> **Note on privacy:** Embeddings are computed **locally** on the machine running the app (using `sentence-transformers`), so document text is only sent to Claude as the small retrieved context needed to answer each question.

---

## 🚀 Quick Start

### 1. Install Python
Make sure you have **Python 3.10 or newer** installed. Check with:
```bash
python --version
```

### 2. Install the dependencies
From the project folder:
```bash
pip install -r requirements.txt
```
> The first run downloads a small embedding model (~90 MB). This happens once.

### 3. Add your Claude API key
1. Get a key from the [Anthropic Console](https://console.anthropic.com/) → **Settings → API Keys**.
2. Copy the example env file and paste your key in:
```bash
cp .env.example .env
```
Then open `.env` and set:
```
ANTHROPIC_API_KEY=sk-ant-your-real-key-here
```

### 4. Run the app
```bash
streamlit run app.py
```
Your browser opens automatically at `http://localhost:8501`.

---

## 🧪 Testing it

A sample document, `data.txt`, is included (an FAQ for a fictional "ACME Cloud" service).

1. Start the app.
2. In the sidebar, upload `data.txt`.
3. Ask questions like:
   - *"How much does the Business plan cost?"*
   - *"How long are deleted files kept on the free plan?"*
   - *"What encryption do you use?"*
   - *"Can I get a refund?"*

The chatbot will answer using only the document's content.

---

## 🎨 Customizing for a client

Everything a client typically wants to change is easy to reach:

| What to change | Where |
|---|---|
| Assistant persona / tone | **System prompt** box in the sidebar (or `DEFAULT_SYSTEM_PROMPT` in `app.py`) |
| Available models | `MODEL_OPTIONS` in `app.py` |
| App title / logo emoji | `APP_TITLE` and `APP_ICON` in `app.py` |
| Chunk size / retrieval count | `CHUNK_SIZE`, `TOP_K` in `app.py` |
| Answer length | `MAX_TOKENS` in `app.py` |

---

## 💰 Cost notes (for reselling)

- **Embeddings are free** — they run locally, no per-use cost.
- **Claude API usage is pay-as-you-go.** Haiku is the cheapest option and is a great default for a "Basic" package. You (or your client) pay Anthropic only for the questions asked.
- Set spending limits in the Anthropic Console to stay in control.

---

## 📦 Deployment options

- **Local / demo:** `streamlit run app.py` (what this README covers).
- **Share online:** deploy free on [Streamlit Community Cloud](https://streamlit.io/cloud), or on any VPS. Set `ANTHROPIC_API_KEY` as a secret/environment variable there instead of using a `.env` file.

---

## 🛠️ Troubleshooting

| Problem | Fix |
|---|---|
| *"No ANTHROPIC_API_KEY found"* | Make sure `.env` exists and contains a valid key, then restart the app. |
| Upload says "No readable text found" | Scanned/image-only PDFs have no extractable text. Use a text-based PDF or run OCR first. |
| First run is slow | The embedding model downloads once (~90 MB). Later runs are fast. |
| `faiss` install issues | Ensure you're on 64-bit Python 3.10+. Reinstall with `pip install faiss-cpu`. |

---

## 📁 Project structure

```
ai-document-chatbot/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
├── .env.example        # Template for your API key
├── data.txt            # Sample document for testing
└── README.md           # This file
```

---

## 📄 License

You are free to sell and modify this project as part of your services.
