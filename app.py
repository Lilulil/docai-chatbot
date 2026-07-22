"""
AI Document Chatbot — Premium Dark UI
Run:  streamlit run app.py
"""

import os
import hashlib
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv

import anthropic
from pypdf import PdfReader
import docx
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
load_dotenv()

APP_TITLE = "DocAI Pro"
APP_ICON = "📄"

MODEL_OPTIONS = {
    "Claude Opus 4.6": "claude-opus-4-6",
    "Claude Opus 4.7": "claude-opus-4-7",
    "Claude Opus 4.8": "claude-opus-4-8",
    "GPT-5.5": "gpt-5.5",
}

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the user's questions using ONLY the "
    "information in the provided document context. If the answer is not "
    "contained in the context, say you don't know based on the documents "
    "provided. Be concise, accurate, and cite the relevant details."
)

SYSTEM_PROMPT_PRESETS = {
    "📝 Default": DEFAULT_SYSTEM_PROMPT,
    "💼 Formal": "You are a professional corporate assistant. Answer questions formally and professionally using ONLY the information in the provided documents. Use clear business language.",
    "😊 Casual": "You are a friendly and approachable assistant. Answer questions in a warm, conversational tone using ONLY the information in the provided documents.",
    "🎯 Expert": "You are a technical expert. Answer questions with precision and depth using ONLY the information in the provided documents."
}

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
TOP_K = 4
MAX_TOKENS = 1024
MAX_FILE_SIZE_MB = 10

st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)


# --------------------------------------------------------------------------- #
# Custom CSS — Premium Dark UI
# --------------------------------------------------------------------------- #
def inject_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

    /* ============================================================ */
    /* ROOT VARIABLES — PREMIUM DARK THEME                          */
    /* ============================================================ */
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        box-sizing: border-box;
    }

    .stApp {
        background: #0b0b10 !important;
    }

    .stApp > div:first-child {
        background: #0b0b10 !important;
    }

    /* subtle grid overlay biar keliatan mahal */
    .stApp::before {
        content: "";
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        pointer-events: none;
        background-image: 
            radial-gradient(circle at 20% 50%, rgba(4, 86, 197, 0.03) 0%, transparent 50%),
            radial-gradient(circle at 80% 20%, rgba(70, 72, 212, 0.03) 0%, transparent 50%),
            radial-gradient(circle at 50% 80%, rgba(4, 86, 197, 0.02) 0%, transparent 50%);
        z-index: 0;
    }

    .block-container {
        padding-top: 0.6rem !important;
        padding-bottom: 4rem !important;
        max-width: 1100px !important;
        position: relative;
        z-index: 1;
    }

    /* ============================================================ */
    /* SIDEBAR — PREMIUM GLASS                                      */
    /* ============================================================ */
    section[data-testid="stSidebar"] {
        background: rgba(15, 15, 25, 0.92) !important;
        backdrop-filter: blur(40px) !important;
        -webkit-backdrop-filter: blur(40px) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.04) !important;
        padding: 1.5rem 1rem !important;
        box-shadow: 4px 0 60px rgba(0, 0, 0, 0.5) !important;
    }

    section[data-testid="stSidebar"] .block-container {
        padding: 0 !important;
    }

    .sidebar-brand {
        display: flex;
        align-items: center;
        gap: 14px;
        padding: 10px 14px;
        margin-bottom: 20px;
        border-radius: 16px;
        background: linear-gradient(135deg, rgba(4, 86, 197, 0.15), rgba(70, 72, 212, 0.08));
        border: 1px solid rgba(255, 255, 255, 0.04);
        transition: all 0.3s ease;
    }
    .sidebar-brand:hover {
        border-color: rgba(4, 86, 197, 0.15);
    }
    .sidebar-brand .icon {
        width: 40px;
        height: 40px;
        border-radius: 12px;
        background: linear-gradient(135deg, #0456c5, #4648d4);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 20px;
        box-shadow: 0 4px 20px rgba(4, 86, 197, 0.3);
    }
    .sidebar-brand .text {
        font-weight: 800;
        font-size: 1.15rem;
        color: white;
        letter-spacing: -0.02em;
    }
    .sidebar-brand .sub {
        font-size: 0.55rem;
        color: rgba(255, 255, 255, 0.25);
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-top: -1px;
    }

    .sidebar-label {
        font-size: 0.6rem;
        font-weight: 700;
        color: rgba(255, 255, 255, 0.2);
        text-transform: uppercase;
        letter-spacing: 0.1em;
        padding: 10px 6px 4px 6px;
    }

    /* Sidebar inputs premium */
    section[data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] > div {
        border-radius: 12px !important;
        border: 1px solid rgba(255, 255, 255, 0.06) !important;
        background: rgba(255, 255, 255, 0.03) !important;
        color: white !important;
        transition: all 0.2s ease !important;
    }
    section[data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] > div:hover {
        border-color: rgba(4, 86, 197, 0.3) !important;
    }
    section[data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] > div * {
        color: white !important;
    }

    section[data-testid="stSidebar"] textarea {
        border-radius: 12px !important;
        border: 1px solid rgba(255, 255, 255, 0.06) !important;
        background: rgba(255, 255, 255, 0.03) !important;
        color: rgba(255, 255, 255, 0.8) !important;
        font-size: 0.82rem !important;
        transition: all 0.2s ease !important;
        line-height: 1.5 !important;
    }
    section[data-testid="stSidebar"] textarea:focus {
        border-color: rgba(4, 86, 197, 0.4) !important;
        box-shadow: 0 0 0 3px rgba(4, 86, 197, 0.06) !important;
    }

    /* File uploader premium */
    section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] {
        border-radius: 14px !important;
        border: 1px dashed rgba(255, 255, 255, 0.06) !important;
        background: rgba(255, 255, 255, 0.02) !important;
        transition: all 0.3s ease !important;
    }
    section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"]:hover {
        border-color: rgba(4, 86, 197, 0.3) !important;
        background: rgba(4, 86, 197, 0.04) !important;
    }
    section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] * {
        color: rgba(255, 255, 255, 0.3) !important;
    }

    /* Buttons premium */
    .stButton > button {
        border-radius: 12px !important;
        background: linear-gradient(135deg, #0456c5, #4648d4) !important;
        color: white !important;
        border: none !important;
        font-weight: 600 !important;
        font-size: 0.85rem !important;
        padding: 0.5rem 1.2rem !important;
        box-shadow: 0 4px 24px rgba(4, 86, 197, 0.2) !important;
        transition: all 0.25s ease !important;
        letter-spacing: 0.01em;
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 40px rgba(4, 86, 197, 0.35) !important;
    }

    /* ============================================================ */
    /* HEADER — PREMIUM                                              */
    /* ============================================================ */
    .app-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 6px 0 16px 0;
        border-bottom: 1px solid rgba(255, 255, 255, 0.04);
        margin-bottom: 20px;
    }
    .app-header-left {
        display: flex;
        align-items: center;
        gap: 14px;
    }
    .app-header-title {
        font-size: 1.15rem;
        font-weight: 800;
        color: white;
        letter-spacing: -0.02em;
    }
    .app-header-title span {
        background: linear-gradient(135deg, #0456c5, #4648d4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .app-header-badge {
        font-size: 0.5rem;
        padding: 2px 10px;
        border-radius: 999px;
        background: rgba(255, 255, 255, 0.04);
        color: rgba(255, 255, 255, 0.2);
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        border: 1px solid rgba(255, 255, 255, 0.04);
    }
    .app-header-avatar {
        width: 34px;
        height: 34px;
        border-radius: 50%;
        background: linear-gradient(135deg, rgba(4, 86, 197, 0.2), rgba(70, 72, 212, 0.1));
        border: 1px solid rgba(255, 255, 255, 0.06);
        color: rgba(255, 255, 255, 0.6);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 12px;
        font-weight: 700;
        transition: all 0.3s ease;
    }
    .app-header-avatar:hover {
        border-color: rgba(4, 86, 197, 0.2);
    }

    /* ============================================================ */
    /* CHAT BUBBLES — PREMIUM                                        */
    /* ============================================================ */
    [data-testid="stChatMessage"] {
        padding: 0 !important;
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        border-radius: 0 !important;
        margin-bottom: 20px !important;
        animation: fadeUp 0.4s ease both;
    }

    .chat-bubble {
        display: flex;
        gap: 14px;
        align-items: flex-start;
        max-width: 82%;
    }
    .chat-bubble.user {
        margin-left: auto;
        flex-direction: row-reverse;
    }
    .chat-bubble.assistant {
        margin-right: auto;
    }

    .chat-avatar {
        width: 36px;
        height: 36px;
        border-radius: 50%;
        flex-shrink: 0;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 15px;
        transition: all 0.3s ease;
    }
    .chat-avatar.user {
        background: linear-gradient(135deg, #0456c5, #4648d4);
        color: white;
        box-shadow: 0 4px 20px rgba(4, 86, 197, 0.2);
    }
    .chat-avatar.assistant {
        background: rgba(255, 255, 255, 0.04);
        color: rgba(255, 255, 255, 0.4);
        border: 1px solid rgba(255, 255, 255, 0.04);
    }

    .chat-content {
        padding: 14px 20px;
        border-radius: 20px;
        font-size: 0.92rem;
        line-height: 1.7;
        transition: all 0.3s ease;
    }
    .chat-bubble.user .chat-content {
        background: linear-gradient(135deg, #0456c5, #4648d4);
        color: white;
        border-radius: 20px 4px 20px 20px;
        box-shadow: 0 4px 30px rgba(4, 86, 197, 0.15);
    }
    .chat-bubble.assistant .chat-content {
        background: rgba(255, 255, 255, 0.03);
        color: rgba(255, 255, 255, 0.85);
        border-radius: 4px 20px 20px 20px;
        border: 1px solid rgba(255, 255, 255, 0.04);
        backdrop-filter: blur(4px);
    }

    .chat-bubble.assistant .chat-content p {
        color: rgba(255, 255, 255, 0.85) !important;
        margin-bottom: 8px;
    }
    .chat-bubble.assistant .chat-content strong {
        color: white !important;
    }
    .chat-bubble.assistant .chat-content ul,
    .chat-bubble.assistant .chat-content ol {
        padding-left: 20px;
        color: rgba(255, 255, 255, 0.7);
    }
    .chat-bubble.assistant .chat-content li {
        margin-bottom: 4px;
    }

    /* ============================================================ */
    /* CHAT INPUT — PREMIUM                                          */
    /* ============================================================ */
    [data-testid="stChatInput"] {
        border-radius: 24px !important;
        border: 1px solid rgba(255, 255, 255, 0.04) !important;
        background: rgba(255, 255, 255, 0.02) !important;
        box-shadow: 0 4px 40px rgba(0, 0, 0, 0.3) !important;
        transition: all 0.3s ease !important;
        padding: 4px !important;
    }
    [data-testid="stChatInput"]:hover {
        border-color: rgba(255, 255, 255, 0.08) !important;
    }
    [data-testid="stChatInput"]:focus-within {
        border-color: rgba(4, 86, 197, 0.3) !important;
        box-shadow: 0 4px 40px rgba(4, 86, 197, 0.06) !important;
    }
    [data-testid="stChatInput"] input {
        color: white !important;
        font-size: 0.9rem !important;
        padding: 8px 16px !important;
    }
    [data-testid="stChatInput"] input::placeholder {
        color: rgba(255, 255, 255, 0.15) !important;
    }

    /* ============================================================ */
    /* EMPTY STATE — PREMIUM                                         */
    /* ============================================================ */
    .empty-state {
        text-align: center;
        padding: 60px 30px;
        background: rgba(255, 255, 255, 0.01);
        border-radius: 28px;
        border: 1px solid rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(4px);
        margin-top: 40px;
    }
    .empty-state .emoji {
        font-size: 52px;
        display: block;
        margin-bottom: 12px;
        opacity: 0.6;
    }
    .empty-state h3 {
        font-size: 1.2rem;
        font-weight: 700;
        color: rgba(255, 255, 255, 0.6);
        margin: 8px 0 4px;
        letter-spacing: -0.01em;
    }
    .empty-state p {
        color: rgba(255, 255, 255, 0.15);
        font-size: 0.9rem;
        margin: 0;
    }

    /* ============================================================ */
    /* TYPING INDICATOR — PREMIUM                                    */
    /* ============================================================ */
    .typing-indicator {
        display: flex;
        gap: 6px;
        padding: 12px 20px;
        background: rgba(255, 255, 255, 0.02);
        border-radius: 20px 20px 20px 4px;
        border: 1px solid rgba(255, 255, 255, 0.03);
        width: fit-content;
        backdrop-filter: blur(4px);
    }
    .typing-indicator .dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: #0456c5;
        opacity: 0.4;
        animation: typingBounce 1.2s infinite;
    }
    .typing-indicator .dot:nth-child(2) { animation-delay: 0.2s; opacity: 0.6; }
    .typing-indicator .dot:nth-child(3) { animation-delay: 0.4s; opacity: 0.8; }

    @keyframes typingBounce {
        0%, 60%, 100% { transform: translateY(0); }
        30% { transform: translateY(-6px); }
    }

    @keyframes fadeUp {
        from { opacity: 0; transform: translateY(16px); }
        to { opacity: 1; transform: translateY(0); }
    }

    /* ============================================================ */
    /* CHIP — PREMIUM                                                */
    /* ============================================================ */
    .chip {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        font-size: 0.65rem;
        font-weight: 600;
        color: rgba(255, 255, 255, 0.3);
        background: rgba(255, 255, 255, 0.02);
        border: 1px solid rgba(255, 255, 255, 0.04);
        padding: 4px 16px;
        border-radius: 999px;
    }
    .chip .pulse {
        width: 6px;
        height: 6px;
        border-radius: 50%;
        background: #10b981;
        display: inline-block;
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0% { opacity: 0.3; }
        50% { opacity: 1; }
        100% { opacity: 0.3; }
    }

    /* ============================================================ */
    /* HIDE STREAMLIT DEFAULT                                        */
    /* ============================================================ */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    [data-testid="stHeader"] { background: transparent; }

    /* ============================================================ */
    /* SCROLLBAR — PREMIUM                                           */
    /* ============================================================ */
    ::-webkit-scrollbar {
        width: 4px;
    }
    ::-webkit-scrollbar-track {
        background: transparent;
    }
    ::-webkit-scrollbar-thumb {
        background: rgba(255, 255, 255, 0.04);
        border-radius: 10px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: rgba(4, 86, 197, 0.2);
    }

    /* ============================================================ */
    /* RESPONSIVE                                                    */
    /* ============================================================ */
    @media (max-width: 640px) {
        .chat-bubble { max-width: 94%; }
        .app-header-title { font-size: 0.95rem; }
        .block-container { padding: 0.3rem !important; }
        .empty-state { padding: 30px 16px; }
        .chat-content { font-size: 0.85rem; padding: 10px 14px; }
    }
    </style>
    """, unsafe_allow_html=True)

# --------------------------------------------------------------------------- #
# Cached resources
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner=False)
def get_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


@st.cache_resource(show_spinner=False)
def get_anthropic_client():
    return anthropic.Anthropic(
        base_url="https://agentrouter.org/v1"
    )


# --------------------------------------------------------------------------- #
# Document reading
# --------------------------------------------------------------------------- #
def read_pdf(file):
    try:
        reader = PdfReader(file)
        return "\n".join((page.extract_text() or "") for page in reader.pages)
    except Exception:
        return "[Error reading PDF]"


def read_docx(file):
    try:
        document = docx.Document(file)
        return "\n".join(paragraph.text for paragraph in document.paragraphs)
    except Exception:
        return "[Error reading DOCX]"


def read_txt(file):
    try:
        return file.read().decode("utf-8", errors="ignore")
    except Exception:
        return "[Error reading TXT]"


def extract_text(uploaded_file):
    name = uploaded_file.name.lower()
    if name.endswith(".pdf"):
        return read_pdf(uploaded_file)
    if name.endswith(".docx"):
        return read_docx(uploaded_file)
    if name.endswith(".txt"):
        return read_txt(uploaded_file)
    return ""


def validate_file_size(uploaded_files):
    oversized = []
    for f in uploaded_files:
        size_mb = f.size / (1024 * 1024)
        if size_mb > MAX_FILE_SIZE_MB:
            oversized.append((f.name, size_mb))
    return oversized


# --------------------------------------------------------------------------- #
# Chunking + indexing
# --------------------------------------------------------------------------- #
def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    text = " ".join(text.split())
    if not text:
        return []
    chunks = []
    start = 0
    step = max(1, chunk_size - overlap)
    while start < len(text):
        chunks.append(text[start:start + chunk_size])
        start += step
    return chunks


def build_index(chunks):
    model = get_embedding_model()
    embeddings = model.encode(chunks, convert_to_numpy=True, show_progress_bar=False).astype("float32")
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index


def retrieve(query, index, chunks, top_k=TOP_K):
    model = get_embedding_model()
    query_vec = model.encode([query], convert_to_numpy=True, show_progress_bar=False).astype("float32")
    top_k = min(top_k, len(chunks))
    _distances, indices = index.search(query_vec, top_k)
    return [chunks[i] for i in indices[0]]


def files_fingerprint(uploaded_files):
    hasher = hashlib.md5()
    for f in uploaded_files:
        hasher.update(f.name.encode())
        hasher.update(str(f.size).encode())
    return hasher.hexdigest()


# --------------------------------------------------------------------------- #
# Session state
# --------------------------------------------------------------------------- #
def init_state():
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("chunks", [])
    st.session_state.setdefault("index", None)
    st.session_state.setdefault("doc_fingerprint", None)


def reset_chat():
    st.session_state["messages"] = []


def export_chat():
    if not st.session_state["messages"]:
        return "", "chat_export.txt"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    filename = f"chat_export_{timestamp}.txt"
    content = f"DocAI Pro — Chat Export\nExported: {timestamp}\n" + "="*60 + "\n\n"
    for msg in st.session_state["messages"]:
        role = "👤 User" if msg["role"] == "user" else "🤖 Assistant"
        content += f"{role}:\n{msg['content']}\n\n"
    return content, filename


def answer_question(client, model, system_prompt, context, history, question):
    grounded_question = (
        f"Use the following document context to answer the question.\n\n"
        f"<document_context>\n{context}\n</document_context>\n\n"
        f"Question: {question}"
    )
    api_messages = history + [{"role": "user", "content": grounded_question}]
    try:
        with client.messages.stream(
            model=model,
            max_tokens=MAX_TOKENS,
            system=system_prompt,
            messages=api_messages,
        ) as stream:
            for text in stream.text_stream:
                yield text
    except Exception as e:
        yield f"[Error: {str(e)}]"


# --------------------------------------------------------------------------- #
# UI
# --------------------------------------------------------------------------- #
def sidebar():
    with st.sidebar:
        st.markdown("""
        <div class="sidebar-brand">
            <div class="icon">📄</div>
            <div>
                <div class="text">DocAI Pro</div>
                <div class="sub">Kecerdasan Terfokus</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="sidebar-label">⚙️ Model</div>', unsafe_allow_html=True)
        model_label = st.selectbox("Model", list(MODEL_OPTIONS.keys()), label_visibility="collapsed")
        model = MODEL_OPTIONS[model_label]

        st.markdown('<div class="sidebar-label">🧠 Asisten</div>', unsafe_allow_html=True)
        preset = st.selectbox("Preset", list(SYSTEM_PROMPT_PRESETS.keys()), label_visibility="collapsed")
        system_prompt = st.text_area(
            "System prompt",
            value=SYSTEM_PROMPT_PRESETS.get(preset, DEFAULT_SYSTEM_PROMPT),
            height=100,
            label_visibility="collapsed",
        )

        st.markdown('<div class="sidebar-label">📎 Upload</div>', unsafe_allow_html=True)
        st.markdown("""
        <div style="border:1.5px dashed rgba(4,86,197,0.25);border-radius:14px;padding:18px;text-align:center;background:rgba(4,86,197,0.04);margin-bottom:8px;">
            <div style="font-size:28px;">☁️</div>
            <div style="font-size:0.8rem;color:rgba(255,255,255,0.5);">Seret file atau <span style="color:#0456c5;font-weight:600;">Pilih</span></div>
            <div style="font-size:0.6rem;color:rgba(255,255,255,0.2);margin-top:4px;">Max 10MB per file</div>
        </div>
        """, unsafe_allow_html=True)

        uploaded_files = st.file_uploader(
            "PDF, TXT, DOCX",
            type=["pdf", "txt", "docx"],
            accept_multiple_files=True,
            label_visibility="collapsed",
        )

        if uploaded_files:
            oversized = validate_file_size(uploaded_files)
            if oversized:
                for name, size in oversized:
                    st.warning(f"⚠️ {name} ({size:.1f}MB) > 10MB")
                uploaded_files = [f for f in uploaded_files if f.size / (1024 * 1024) <= MAX_FILE_SIZE_MB]

            if uploaded_files:
                fingerprint = files_fingerprint(uploaded_files)
                if fingerprint != st.session_state["doc_fingerprint"]:
                    with st.spinner("📖 Indexing..."):
                        all_chunks = []
                        bar = st.progress(0)
                        for i, f in enumerate(uploaded_files):
                            text = extract_text(f)
                            all_chunks.extend(chunk_text(text))
                            bar.progress((i + 1) / len(uploaded_files))
                        bar.empty()
                        if all_chunks:
                            st.session_state["chunks"] = all_chunks
                            st.session_state["index"] = build_index(all_chunks)
                            st.session_state["doc_fingerprint"] = fingerprint
                            st.success(f"✅ {len(all_chunks)} chunks ready")
                        else:
                            st.warning("⚠️ No text found")

        if st.session_state["chunks"]:
            st.markdown(f"""
            <div class="chip">
                <span class="pulse"></span>
                {len(st.session_state["chunks"])} chunks ready
            </div>
            """, unsafe_allow_html=True)

        st.divider()

        if st.session_state["messages"]:
            content, filename = export_chat()
            st.download_button("📥 Download chat", data=content, file_name=filename, mime="text/plain", use_container_width=True)

        st.button("🔄 Reset chat", on_click=reset_chat, use_container_width=True)

    return model, system_prompt


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def render_chat_message(msg):
    role = msg["role"]
    is_user = role == "user"
    avatar = "👤" if is_user else "🤖"
    bubble_class = "user" if is_user else "assistant"
    st.markdown(f"""
    <div class="chat-bubble {bubble_class}">
        <div class="chat-avatar {bubble_class}">{avatar}</div>
        <div class="chat-content">{msg["content"]}</div>
    </div>
    """, unsafe_allow_html=True)


def main():
    inject_css()
    init_state()
    model, system_prompt = sidebar()

    # Header
    st.markdown("""
    <div class="app-header">
        <div class="app-header-left">
            <span class="app-header-title">DocAI Assistant</span>
            <span class="app-header-badge">v2.4</span>
        </div>
        <div class="app-header-avatar">UA</div>
    </div>
    """, unsafe_allow_html=True)

    if not os.getenv("ANTHROPIC_API_KEY"):
        st.error("⚠️ No `ANTHROPIC_API_KEY` found. Please add it to `.env` and restart.")
        st.stop()

    client = get_anthropic_client()

    # Empty state
    if not st.session_state["messages"]:
        ready = bool(st.session_state["chunks"])
        st.markdown(f"""
        <div class="empty-state">
            <div class="emoji">{'💬' if ready else '📥'}</div>
            <h3>{'Ask about your documents' if ready else 'Upload a document to start'}</h3>
            <p>{'Type your question below' if ready else 'Use the sidebar to upload a PDF, TXT, or DOCX file.'}</p>
        </div>
        """, unsafe_allow_html=True)

    # Chat history
    for msg in st.session_state["messages"]:
        render_chat_message(msg)

    # New question
    if question := st.chat_input("Ask about your documents..."):
        if not st.session_state["chunks"]:
            st.warning("⚠️ Please upload a document first.")
            st.stop()

        st.session_state["messages"].append({"role": "user", "content": question})
        render_chat_message({"role": "user", "content": question})

        context_chunks = retrieve(question, st.session_state["index"], st.session_state["chunks"])
        context = "\n\n---\n\n".join(context_chunks)

        with st.chat_message("assistant"):
            typing = st.empty()
            typing.markdown("""
            <div class="typing-indicator">
                <span class="dot"></span><span class="dot"></span><span class="dot"></span>
            </div>
            """, unsafe_allow_html=True)

            try:
                response = st.write_stream(answer_question(
                    client=client,
                    model=model,
                    system_prompt=system_prompt,
                    context=context,
                    history=st.session_state["messages"][:-1],
                    question=question,
                ))
                typing.empty()
            except Exception as e:
                typing.empty()
                st.error(f"❌ {str(e)}")
                response = f"[Error: {str(e)}]"

        st.session_state["messages"].append({"role": "assistant", "content": response})
        st.rerun()


if __name__ == "__main__":
    main()
