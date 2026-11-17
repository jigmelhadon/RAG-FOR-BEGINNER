import html
import os
import re
from collections import Counter

import httpx
import streamlit as st
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ----------------------------- Configuration -----------------------------
DB_DIR = "db/chroma_db"
DOCS_DIR = "docs"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "llama3.2:3b"
ALLOWED_TYPES = ["txt", "md", "pdf"]
PAGE_CHAT, PAGE_LIB = "💬 Chat", "📚 Library"

SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the question using only the context provided. "
    "If the answer is not in the context, say you don't know."
)
STYLES = {
    "Concise": "Keep the answer short: two to four sentences.",
    "Detailed": "Give a thorough, well-organised answer.",
    "Bullet points": "Answer as a short bulleted list.",
}

os.makedirs(DOCS_DIR, exist_ok=True)
st.set_page_config(page_title="Knowledge Assistant", page_icon="📚", layout="centered")

# --------------------------------- Styling --------------------------------
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
#MainMenu, footer, [data-testid="stToolbar"] { visibility: hidden; }
header[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 2rem; padding-bottom: 6rem; max-width: 860px; }
[data-testid="stSidebar"] { border-right: 1px solid #E5E7EB; }
[data-testid="stColumn"], [data-testid="column"] { min-width: 0; }

/* Headings */
.hero { text-align: center; padding: 0.6rem 0 0.2rem 0; }
.hero h1 {
    font-size: 2.2rem; font-weight: 700; margin-bottom: 0.2rem;
    background: linear-gradient(90deg, #4F46E5, #9333EA);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.hero p { color: #6B7280; font-size: 1rem; margin: 0; }
.page-title { font-size: 1.7rem; font-weight: 700; color: #111827; margin-bottom: 0.1rem; }
.page-sub { color: #6B7280; font-size: 0.95rem; margin-bottom: 1.1rem; }
.section-title { font-size: 1.05rem; font-weight: 600; color: #111827; margin: 1.4rem 0 0.6rem 0; }
.scope-pill {
    display: table; margin: 0.8rem auto 1rem auto; background: #EEF2FF; color: #4338CA;
    border-radius: 999px; padding: 0.25rem 0.85rem; font-size: 0.8rem; font-weight: 500;
}

/* Sidebar */
.brand { font-size: 1.15rem; font-weight: 700; color: #111827; margin-bottom: 0.8rem; }
.side-label {
    font-size: 0.72rem; font-weight: 600; letter-spacing: 0.06em; text-transform: uppercase;
    color: #9CA3AF; margin: 1.1rem 0 0.4rem 0;
}
.status-row { display: flex; align-items: center; gap: 0.5rem; font-size: 0.85rem; color: #374151; }
.dot { width: 9px; height: 9px; border-radius: 50%; display: inline-block; }
.dot-on { background: #10B981; } .dot-off { background: #EF4444; }
.privacy { font-size: 0.78rem; color: #9CA3AF; margin-top: 0.6rem; }

/* Cards */
.card {
    background: white; border: 1px solid #E5E7EB; border-radius: 16px;
    padding: 1.2rem 1.4rem; margin: 0.6rem 0 1rem 0; box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}
.card h4 { margin: 0 0 0.4rem 0; font-weight: 600; color: #111827; }
.card p { margin: 0; color: #6B7280; font-size: 0.95rem; }
.stat {
    background: white; border: 1px solid #E5E7EB; border-radius: 14px;
    padding: 0.9rem 1rem; text-align: center;
}
.stat-num { font-size: 1.6rem; font-weight: 700; color: #4F46E5; line-height: 1.2; }
.stat-label { font-size: 0.78rem; color: #6B7280; }

/* Document rows */
.badge {
    display: block; text-align: center; border-radius: 8px; padding: 0.3rem 0;
    font-size: 0.72rem; font-weight: 700; letter-spacing: 0.04em;
}
.badge-pdf { background: #FEE2E2; color: #B91C1C; }
.badge-txt { background: #DBEAFE; color: #1D4ED8; }
.badge-md  { background: #EDE9FE; color: #6D28D9; }
.doc-name {
    font-size: 0.92rem; font-weight: 600; color: #111827;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}
.doc-meta { font-size: 0.78rem; color: #6B7280; }
.doc-warn { font-size: 0.78rem; color: #D97706; }

/* Sources */
.source-card {
    background: #F8F9FC; border: 1px solid #E5E7EB; border-left: 4px solid #4F46E5;
    border-radius: 10px; padding: 0.7rem 0.9rem; margin-bottom: 0.6rem;
}
.source-head { display: flex; justify-content: space-between; gap: 0.6rem; margin-bottom: 0.3rem; }
.source-title {
    font-weight: 600; font-size: 0.85rem; color: #4F46E5;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}
.match {
    flex-shrink: 0; background: #DCFCE7; color: #15803D; font-size: 0.72rem;
    font-weight: 600; border-radius: 999px; padding: 0.1rem 0.55rem;
}
.source-text { font-size: 0.85rem; color: #4B5563; line-height: 1.5; white-space: pre-wrap; }

/* Buttons */
div.stButton > button, div.stDownloadButton > button { border-radius: 12px; transition: all 0.15s ease; }
div.stButton > button:hover, div.stDownloadButton > button:hover { transform: translateY(-1px); }
button[kind="primary"], [data-testid="stBaseButton-primary"] {
    background: linear-gradient(135deg, #16A34A, #22C55E) !important;
    border: none !important; color: #FFFFFF !important; font-weight: 600 !important;
    box-shadow: 0 2px 6px rgba(22, 163, 74, 0.25);
}
button[kind="primary"]:hover, [data-testid="stBaseButton-primary"]:hover {
    background: linear-gradient(135deg, #15803D, #16A34A) !important;
}
button[kind="primary"] p, [data-testid="stBaseButton-primary"] p { color: #FFFFFF !important; }
button[kind="primary"]:disabled, [data-testid="stBaseButton-primary"]:disabled {
    background: #D1D5DB !important; box-shadow: none; opacity: 0.8;
}
</style>
""",
    unsafe_allow_html=True,
)


# ------------------------------- Core helpers ------------------------------
@st.cache_resource(show_spinner="Loading knowledge base...")
def load_db():
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    return Chroma(
        persist_directory=DB_DIR,
        embedding_function=embeddings,
        collection_metadata={"hnsw:space": "cosine"},
    )


@st.cache_resource
def load_model():
    return ChatOllama(model=LLM_MODEL, temperature=0.2, num_ctx=4096)


db = load_db()
model = load_model()
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)


def ollama_running():
    try:
        httpx.get("http://localhost:11434", timeout=1.0)
        return True
    except Exception:
        return False


def safe_name(name):
    return re.sub(r"[^A-Za-z0-9._ -]", "_", os.path.basename(name))


def doc_path(filename):
    return os.path.join(DOCS_DIR, filename)


def short_name(name, limit=30):
    return name if len(name) <= limit else name[: limit - 1] + "…"


def human_size(path):
    n = os.path.getsize(path) if os.path.exists(path) else 0
    if n < 1024:
        return f"{n} B"
    if n < 1024**2:
        return f"{n / 1024:.0f} KB"
    return f"{n / 1024**2:.1f} MB"


def list_files():
    return sorted(
        f for f in os.listdir(DOCS_DIR) if f.lower().rsplit(".", 1)[-1] in ALLOWED_TYPES
    )


def indexed_counts():
    res = db.get(include=["metadatas"])
    return Counter(m.get("source") for m in (res.get("metadatas") or []) if m)


def load_file(path):
    if path.lower().endswith(".pdf"):
        from langchain_community.document_loaders import PyPDFLoader

        return PyPDFLoader(path).load()
    return TextLoader(path, encoding="utf-8").load()


def remove_from_index(path):
    res = db.get(where={"source": path})
    if res["ids"]:
        db.delete(ids=res["ids"])


def index_file(filename):
    path = doc_path(filename)
    chunks = [c for c in splitter.split_documents(load_file(path)) if c.page_content.strip()]
    if not chunks:
        raise ValueError("no readable text found (scanned PDFs are not supported)")
    remove_from_index(path)
    db.add_documents(chunks)
    return len(chunks)


def stream_answer(messages):
    for chunk in model.stream(messages):
        yield chunk.content


# ------------------------------- Callbacks ---------------------------------
def delete_document(filename):
    path = doc_path(filename)
    remove_from_index(path)
    if os.path.exists(path):
        os.remove(path)
    st.session_state.scope = [s for s in st.session_state.get("scope", []) if s != filename]
    st.session_state.flash = [("success", f"Removed {filename}")]


def sync_unindexed():
    counts = indexed_counts()
    flash = []
    for f in list_files():
        if counts.get(doc_path(f), 0) == 0:
            try:
                flash.append(("success", f"Indexed {f} ({index_file(f)} passages)"))
            except Exception as e:
                flash.append(("error", f"{f}: {e}"))
    st.session_state.flash = flash or [("info", "Everything is already indexed.")]


def set_pending(question):
    st.session_state.pending_query = question
    st.session_state.page = PAGE_CHAT


def summarize_doc(filename):
    st.session_state.scope = [filename]
    st.session_state.pending_query = f'Give me a clear summary of the document "{filename}".'
    st.session_state.pending_k = 6
    st.session_state.page = PAGE_CHAT


def go_library():
    st.session_state.page = PAGE_LIB


def clear_chat():
    st.session_state.messages = []


def export_markdown():
    lines = ["# Knowledge Assistant: conversation", ""]
    for m in st.session_state.messages:
        who = "You" if m["role"] == "user" else "Assistant"
        lines += [f"**{who}:** {m['content']}", ""]
    return "\n".join(lines)


# ------------------------------- Session state -----------------------------
st.session_state.setdefault("messages", [])
st.session_state.setdefault("uploader_key", 0)
st.session_state.setdefault("page", PAGE_CHAT)

files = list_files()
counts = indexed_counts()
total_chunks = sum(counts.get(doc_path(f), 0) for f in files)
st.session_state.scope = [s for s in st.session_state.get("scope", []) if s in files]

for kind, msg in st.session_state.pop("flash", []):
    st.toast(msg, icon={"success": "✅", "error": "⚠️", "info": "ℹ️"}[kind])

# --------------------------------- Sidebar ---------------------------------
with st.sidebar:
    st.markdown("<div class='brand'>📚 Knowledge Assistant</div>", unsafe_allow_html=True)
    st.radio("Navigation", [PAGE_CHAT, PAGE_LIB], key="page", label_visibility="collapsed")

    st.markdown("<div class='side-label'>Chat options</div>", unsafe_allow_html=True)
    scope = st.multiselect(
        "Search in",
        options=files,
        key="scope",
        placeholder="All documents",
        format_func=lambda f: short_name(f, 32),
        help="Pick one or more documents to limit the search. Leave empty to search everything.",
    )
    style = st.selectbox("Answer style", list(STYLES))
    k = st.slider("Passages to search", 1, 8, 3, help="More passages give broader answers but slower replies.")
    show_sources = st.toggle("Show sources", value=True)

    st.markdown("<div class='side-label'>Conversation</div>", unsafe_allow_html=True)
    st.download_button(
        "⬇️ Export chat",
        data=export_markdown(),
        file_name="conversation.md",
        mime="text/markdown",
        use_container_width=True,
        disabled=not st.session_state.messages,
    )
    st.button("🗑️ Clear chat", on_click=clear_chat, use_container_width=True)

    st.markdown("<div class='side-label'>Status</div>", unsafe_allow_html=True)
    running = ollama_running()
    st.markdown(
        f"<div class='status-row'><span class='dot {'dot-on' if running else 'dot-off'}'></span>"
        f"AI model {'ready' if running else 'offline: open the Ollama app'}</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<div class='privacy'>🔒 Runs fully on your computer. Model: {LLM_MODEL}</div>",
        unsafe_allow_html=True,
    )


# ---------------------------------- Pages ----------------------------------
def render_sources(sources):
    best = max((s.get("score", 0) for s in sources), default=0)
    if sources and best < 0.2:
        st.caption("⚠️ These passages are only loosely related to your question, so the answer may be unreliable.")
    with st.expander(f"📎 Sources ({len(sources)})"):
        for i, s in enumerate(sources, 1):
            pct = int(round(s.get("score", 0) * 100))
            st.markdown(
                f"<div class='source-card'><div class='source-head'>"
                f"<span class='source-title'>{i}. {html.escape(s['file'])}</span>"
                f"<span class='match'>{pct}% match</span></div>"
                f"<div class='source-text'>{html.escape(s['text'])}</div></div>",
                unsafe_allow_html=True,
            )


def render_library():
    st.markdown(
        "<div class='page-title'>Document library</div>"
        "<div class='page-sub'>Upload files and manage what the assistant knows.</div>",
        unsafe_allow_html=True,
    )

    questions = sum(1 for m in st.session_state.messages if m["role"] == "user")
    c1, c2, c3 = st.columns(3)
    for col, num, label in [(c1, len(files), "Documents"), (c2, total_chunks, "Passages"), (c3, questions, "Questions asked")]:
        col.markdown(
            f"<div class='stat'><div class='stat-num'>{num}</div><div class='stat-label'>{label}</div></div>",
            unsafe_allow_html=True,
        )

    st.markdown("<div class='section-title'>Add documents</div>", unsafe_allow_html=True)
    with st.container(border=True):
        uploaded = st.file_uploader(
            "Upload files",
            type=ALLOWED_TYPES,
            accept_multiple_files=True,
            key=f"uploader_{st.session_state.uploader_key}",
            label_visibility="collapsed",
        )
        st.caption("Supported: TXT, MD, PDF. Files are saved in your docs folder.")
        if st.button("➕ Add to knowledge base", type="primary", use_container_width=True, disabled=not uploaded):
            flash = []
            with st.spinner("Adding documents..."):
                for f in uploaded:
                    name = safe_name(f.name)
                    with open(doc_path(name), "wb") as out:
                        out.write(f.getbuffer())
                    try:
                        flash.append(("success", f"Added {name} ({index_file(name)} passages)"))
                    except Exception as e:
                        os.remove(doc_path(name))
                        flash.append(("error", f"{name}: {e}"))
            st.session_state.flash = flash
            st.session_state.uploader_key += 1
            st.rerun()

    st.markdown("<div class='section-title'>Your documents</div>", unsafe_allow_html=True)
    if not files:
        st.info("No documents yet. Upload one above to get started.")
        return

    search = st.text_input("Filter", placeholder="🔍 Filter documents by name", label_visibility="collapsed")
    shown = [f for f in files if search.lower() in f.lower()]
    if not shown:
        st.caption("No documents match your filter.")

    for f in shown:
        n = counts.get(doc_path(f), 0)
        ext = f.rsplit(".", 1)[-1].lower()
        meta = (
            f"<div class='doc-meta'>{human_size(doc_path(f))} · {n} passages</div>"
            if n
            else "<div class='doc-warn'>Not indexed yet</div>"
        )
        with st.container(border=True):
            a, b, c, d = st.columns([1.1, 6, 0.9, 0.9], vertical_alignment="center")
            a.markdown(f"<span class='badge badge-{ext}'>{ext.upper()}</span>", unsafe_allow_html=True)
            b.markdown(
                f"<div class='doc-name' title='{html.escape(f)}'>{html.escape(f)}</div>{meta}",
                unsafe_allow_html=True,
            )
            c.button("📝", key=f"sum_{f}", on_click=summarize_doc, args=(f,), help="Summarize this document", use_container_width=True, disabled=not n)
            d.button("🗑️", key=f"del_{f}", on_click=delete_document, args=(f,), help="Remove this document", use_container_width=True)

    if any(counts.get(doc_path(f), 0) == 0 for f in files):
        st.button("🔄 Index missing documents", on_click=sync_unindexed, use_container_width=True)


def render_chat():
    st.markdown(
        "<div class='hero'><h1>Knowledge Assistant</h1>"
        "<p>Ask questions and get answers grounded in your own documents.</p></div>",
        unsafe_allow_html=True,
    )
    pill = (
        "Searching in: " + ", ".join(short_name(s, 28) for s in scope)
        if scope
        else f"Searching in all {len(files)} document(s)"
    )
    st.markdown(f"<div class='scope-pill'>🔎 {html.escape(pill)}</div>", unsafe_allow_html=True)

    typed = st.chat_input("Ask a question about your documents...", disabled=total_chunks == 0)
    pending = st.session_state.pop("pending_query", None)
    pending_k = st.session_state.pop("pending_k", None)
    query = typed or pending
    k_use = pending_k or k

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"], avatar="🙂" if msg["role"] == "user" else "📚"):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and show_sources and msg.get("sources"):
                render_sources(msg["sources"])

    if not st.session_state.messages and not query:
        if total_chunks == 0:
            st.markdown(
                "<div class='card'><h4>👋 Let's get started</h4>"
                "<p>Upload your first document in the Library, then come back here to ask questions.</p></div>",
                unsafe_allow_html=True,
            )
            st.button("📚 Go to Library", type="primary", on_click=go_library, use_container_width=True)
        else:
            st.markdown(
                f"<div class='card'><h4>👋 Welcome</h4><p>{len(files)} document(s) are ready. "
                "I answer only from your documents. Pick a starter below or type your own question.</p></div>",
                unsafe_allow_html=True,
            )
            starters = [("🔑 Key points across my documents", set_pending, "What are the key points across my documents?")]
            starters += [(f"📝 Summarize: {short_name(f, 24)}", summarize_doc, f) for f in files[:3]]
            cols = st.columns(2)
            for i, (label, cb, arg) in enumerate(starters):
                cols[i % 2].button(label, key=f"starter_{i}", on_click=cb, args=(arg,), use_container_width=True)

    if not query:
        return

    history = list(st.session_state.messages)
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user", avatar="🙂"):
        st.markdown(query)

    with st.chat_message("assistant", avatar="📚"):
        try:
            # Short follow-ups like "tell me more" borrow the previous question for searching
            search_q = query
            prev_users = [m["content"] for m in history if m["role"] == "user"]
            if len(query.split()) < 5 and prev_users:
                search_q = prev_users[-1] + " " + query

            flt = {"source": {"$in": [doc_path(f) for f in scope]}} if scope else None
            with st.spinner("Searching your documents..."):
                results = db.similarity_search_with_score(search_q, k=k_use, filter=flt)
            if not results:
                raise ValueError("No passages found in the selected documents.")

            context = "\n\n".join(d.page_content for d, _ in results)
            messages = [SystemMessage(content=f"{SYSTEM_PROMPT} {STYLES[style]}")]
            for m in history[-6:]:
                messages.append(HumanMessage(content=m["content"]) if m["role"] == "user" else AIMessage(content=m["content"]))
            messages.append(HumanMessage(content=f"Context:\n{context}\n\nQuestion: {query}"))

            answer = st.write_stream(stream_answer(messages))

            sources = []
            for d, dist in results:
                label = os.path.basename(d.metadata.get("source", "unknown"))
                if "page" in d.metadata:
                    label += f" (page {int(d.metadata['page']) + 1})"
                sources.append({"file": label, "text": d.page_content, "score": max(0.0, 1.0 - float(dist))})
            if show_sources:
                render_sources(sources)

            st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
        except Exception as e:
            st.error(
                "Could not get an answer. Make sure the Ollama app is open and the model is downloaded "
                f"(`ollama pull {LLM_MODEL}`).\n\nDetails: {e}"
            )


if st.session_state.page == PAGE_LIB:
    render_library()
else:
    render_chat()