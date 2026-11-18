import html
import io
import json
import os
import re
import time
from collections import Counter
from datetime import datetime

import httpx
import pandas as pd
import streamlit as st
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ----------------------------- Configuration -----------------------------
DB_DIR = "db/chroma_db"
DOCS_DIR = "docs"  # uploaded documents are saved here, inside your VS Code project
GEN_DIR = "generated"  # graphs and images you create are saved here too
DOCS_ABS = os.path.abspath(DOCS_DIR)
GEN_ABS = os.path.abspath(GEN_DIR)
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "llama3.2:3b"
IMAGE_MODEL = "stabilityai/sd-turbo"  # local Stable Diffusion model used for images
ALLOWED_TYPES = ["txt", "md", "pdf"]
MAX_MB = 25
PAGE_CHAT, PAGE_LIB, PAGE_STUDIO = "Chat", "Documents", "Studio"

SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the question using only the context provided. "
    "If the answer is not in the context, say you don't know."
)
STYLES = {
    "Concise": "Keep the answer short: two to four sentences.",
    "Detailed": "Give a thorough, well-organised answer.",
    "Bullet points": "Answer as a short bulleted list.",
}
SORTS = ["Name", "Newest first", "Largest first"]

MODE_HYBRID, MODE_DOCS, MODE_GENERAL = "Smart", "Documents", "General AI"
MODES = [MODE_HYBRID, MODE_DOCS, MODE_GENERAL]
MODE_HELP = {
    MODE_HYBRID: "Uses your documents when they are relevant, otherwise answers from the model's general knowledge.",
    MODE_DOCS: "Answers only from your uploaded documents.",
    MODE_GENERAL: "Chats like a normal AI assistant and ignores your documents.",
}
GENERAL_PROMPT = (
    "You are a helpful, knowledgeable AI assistant. Answer clearly and accurately. "
    "If you are not sure about something, say so instead of guessing."
)
HYBRID_PROMPT = (
    "You are a helpful assistant. Context from the user's documents is provided below. "
    "Use it when it is relevant to the question and say the answer comes from the documents. "
    "If the context does not cover the question, answer from your general knowledge and say the documents did not cover it."
)
RELEVANCE_MIN = 0.2  # below this match score, Smart mode ignores the documents

# Studio
TOOL_GRAPHS, TOOL_IMAGES = "Graphs", "Images"
SRC_DOCS, SRC_AI, SRC_DATA, SRC_CHAT, SRC_DESC = "My documents", "AI knowledge", "Pasted data", "Last chat answer", "My description"
CHART_TYPES = ["Bar", "Horizontal bar", "Line", "Pie"]
PALETTE = ["#0F5C4D", "#E3A008", "#3B82A0", "#B4532A", "#7A8F3B", "#6B4E9B", "#8C9B96", "#C2410C"]
INK = "#14211F"
CONTEXT_CHARS = 6000
GRAPH_RULES = (
    "You turn text into chartable data and reply with JSON only, in exactly this shape: "
    '{"title": string, "x_label": string, "y_label": string, '
    '"series": [{"name": string, "points": [{"label": string, "value": number}]}]}. '
    "Keep labels short (at most four words). Use plain numbers without currency symbols or commas. "
)
GRAPH_FROM_TEXT = (
    GRAPH_RULES
    + "Use only numbers that appear in the context and never invent values. "
    'If the context has no usable numbers, reply {"error": "no numeric data found"}.'
)
GRAPH_FROM_KNOWLEDGE = (
    GRAPH_RULES + "Use your general knowledge. Values may be approximate, so round them sensibly. "
    'If you do not know reliable figures, reply {"error": "no reliable data available"}.'
)
IMAGE_PROMPT_RULES = (
    "You write prompts for an AI image generator. Describe one concrete visual scene: subject, setting, colours and mood. "
    "The image must contain no words, letters, numbers or logos. Use at most 40 words. Reply with the prompt only."
)
IMAGE_STYLES = {
    "Illustration": "clean flat vector illustration, soft colours",
    "Photo-realistic": "realistic photograph, natural light, sharp detail",
    "Watercolor": "soft watercolor painting, gentle textures",
    "Minimal": "minimalist design, simple shapes, plenty of white space",
    "Cinematic": "cinematic lighting, dramatic atmosphere, detailed",
}
IMAGE_SIZES = {"Square": (512, 512), "Landscape": (768, 512), "Portrait": (512, 768)}

os.makedirs(DOCS_DIR, exist_ok=True)
os.makedirs(GEN_DIR, exist_ok=True)
st.set_page_config(page_title="Knowledge Assistant", page_icon="📚", layout="centered", initial_sidebar_state="collapsed")

# --------------------------------- Styling --------------------------------
# Palette: paper #F5F7F4 · ink #14211F · pine #0F5C4D · saffron #E3A008 · hairline #DDE4DF
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600&family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');

html, body, [class*="css"], .stApp { font-family: 'Plus Jakarta Sans', -apple-system, 'Segoe UI', sans-serif; color: #14211F; }
.stApp { background: #F5F7F4; }
#MainMenu, footer, [data-testid="stToolbar"], [data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"] { display: none !important; }
header[data-testid="stHeader"] { background: transparent; height: 0; }
.block-container { padding-top: 1.4rem; padding-bottom: 7rem; max-width: 900px; }
[data-testid="stColumn"], [data-testid="column"] { min-width: 0; }
[data-testid="stBottom"] > div, [data-testid="stBottomBlockContainer"] { background: #F5F7F4; }

/* Top bar */
.brand { display: flex; align-items: center; gap: 0.65rem; }
.brand-mark { width: 36px; height: 36px; border-radius: 10px; background: #0F5C4D; color: #FFFFFF; font-family: 'Fraunces', serif;
              font-size: 1.15rem; font-weight: 600; display: inline-flex; align-items: center; justify-content: center; flex-shrink: 0; }
.brand-name { font-family: 'Fraunces', serif; font-size: 1.15rem; font-weight: 600; line-height: 1.1; white-space: nowrap; }
.brand-sub { font-size: 0.72rem; color: #6B7A75; }
.status-chip { display: inline-flex; align-items: center; gap: 0.5rem; background: #FFFFFF; border: 1px solid #DDE4DF;
               border-radius: 999px; padding: 0.3rem 0.8rem; font-size: 0.78rem; color: #33423E; float: right; white-space: nowrap; }
.dot { width: 8px; height: 8px; border-radius: 50%; display: inline-block; }
.dot-on { background: #16A34A; box-shadow: 0 0 0 3px #16A34A26; }
.dot-off { background: #DC2626; box-shadow: 0 0 0 3px #DC262626; }
.topbar-rule { height: 1px; background: #DDE4DF; margin: 0.9rem 0 1.8rem 0; }

/* Segmented control (every horizontal radio). Left aligned by default, centred in the top bar. */
div[data-testid="stRadio"] { display: flex; justify-content: flex-start; }
.st-key-page div[data-testid="stRadio"] { justify-content: center; }
div[data-testid="stRadio"] > div[role="radiogroup"] { display: inline-flex; flex-wrap: wrap; gap: 2px; background: #E8EEEB; padding: 4px; border-radius: 12px; }
div[data-testid="stRadio"] label[data-baseweb="radio"] { padding: 0.38rem 1rem; border-radius: 9px; margin: 0; cursor: pointer; }
div[data-testid="stRadio"] label[data-baseweb="radio"] > div:first-child { display: none; }
div[data-testid="stRadio"] label[data-baseweb="radio"]:has(input:checked) { background: #FFFFFF; box-shadow: 0 1px 3px rgba(20, 33, 31, 0.12); }
div[data-testid="stRadio"] label p { font-weight: 600; font-size: 0.88rem; color: #14211F; white-space: nowrap; }

/* Headings */
.hero h1 { font-family: 'Fraunces', serif; font-size: 2.5rem; font-weight: 600; line-height: 1.1; margin: 0 0 0.45rem 0; letter-spacing: -0.01em; }
.hero p { color: #5C6B66; font-size: 1rem; margin: 0; max-width: 36rem; line-height: 1.55; }
.page-title { font-family: 'Fraunces', serif; font-size: 2.2rem; font-weight: 600; line-height: 1.12; margin-bottom: 0.35rem; }
.page-sub { color: #5C6B66; font-size: 0.98rem; margin-bottom: 1.4rem; max-width: 38rem; line-height: 1.55; }
.section-title { font-size: 1.05rem; font-weight: 600; margin: 2rem 0 0.8rem 0; }
.panel-title { font-family: 'Fraunces', serif; font-size: 1.3rem; font-weight: 600; margin: 0 0 0.2rem 0; }
.panel-sub { color: #5C6B66; font-size: 0.9rem; margin: 0 0 1rem 0; line-height: 1.5; }
.mode-help { color: #6B7A75; font-size: 0.84rem; margin: 0.5rem 0 0 0.15rem; }
.scope-pill { display: inline-block; margin: 0.9rem 0 1rem 0; background: #E4EFEB; color: #0F5C4D; border-radius: 999px;
              padding: 0.28rem 0.85rem; font-size: 0.82rem; font-weight: 500; }
.empty { text-align: center; color: #6B7A75; padding: 3.2rem 1rem; font-size: 0.95rem; line-height: 1.6; }
.empty b { display: block; font-family: 'Fraunces', serif; font-size: 1.3rem; color: #14211F; margin-bottom: 0.3rem; font-weight: 600; }
.note { color: #6B7A75; font-size: 0.82rem; line-height: 1.5; margin-top: 0.5rem; }

/* Form fields */
[data-testid="stWidgetLabel"] p { font-size: 0.84rem; font-weight: 600; color: #33423E; }
[data-baseweb="select"] > div, [data-baseweb="input"], [data-baseweb="textarea"], textarea, input { border-radius: 10px !important; }

/* Stats */
.stat { background: #FFFFFF; border: 1px solid #DDE4DF; border-radius: 14px; padding: 0.9rem 1rem; }
.stat-num { font-family: 'Fraunces', serif; font-size: 1.8rem; font-weight: 600; color: #0F5C4D; line-height: 1.15; }
.stat-label { font-size: 0.8rem; color: #5C6B66; }

/* Upload area */
.upload-head h3 { font-family: 'Fraunces', serif; font-size: 1.35rem; font-weight: 600; margin: 0 0 0.25rem 0; }
.upload-head p { color: #5C6B66; font-size: 0.92rem; margin: 0 0 0.8rem 0; line-height: 1.5; }
.path-box { display: inline-block; max-width: 100%; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; background: #F0F5F3;
            border: 1px solid #DDE4DF; border-radius: 8px; padding: 0.15rem 0.55rem; margin-bottom: 0.8rem;
            font-family: ui-monospace, Menlo, Consolas, monospace; font-size: 0.78rem; color: #0F5C4D; }
[data-testid="stFileUploaderDropzone"] { background: #F0F5F3; border: 2px dashed #0F5C4D66; border-radius: 18px; padding: 2rem 1rem;
                                         transition: background 0.15s ease, border-color 0.15s ease; }
[data-testid="stFileUploaderDropzone"]:hover { background: #E4EFEB; border-color: #0F5C4D; }
[data-testid="stFileUploaderDropzone"] button { border-radius: 10px; border: 1px solid #0F5C4D; color: #0F5C4D; background: #FFFFFF; }
.chip { display: inline-flex; align-items: center; gap: 0.45rem; background: #FFFFFF; border: 1px solid #DDE4DF; border-radius: 999px;
        padding: 0.2rem 0.75rem; margin: 0.15rem 0.3rem 0.15rem 0; font-size: 0.82rem; }
.chip em { font-style: normal; color: #6B7A75; font-size: 0.75rem; }
.chip-replace { border-color: #E3A008; background: #FFF8E6; }

/* Cards & containers */
.card { background: #FFFFFF; border: 1px solid #DDE4DF; border-radius: 16px; padding: 1.2rem 1.4rem; margin: 0.6rem 0 1rem 0; }
[data-testid="stVerticalBlockBorderWrapper"] { border-color: #DDE4DF !important; border-radius: 16px !important; background: #FFFFFF; }
[data-testid="stExpander"] { background: #FFFFFF; border: 1px solid #DDE4DF; border-radius: 16px; }
[data-testid="stExpander"] details { border: none; }
[data-testid="stImage"] img { border-radius: 12px; }
.quote { background: #F5F7F4; border-left: 4px solid #E3A008; border-radius: 8px; padding: 0.6rem 0.85rem; font-size: 0.86rem;
         color: #44524E; line-height: 1.55; margin-bottom: 0.7rem; }

/* Document rows */
.badge { display: block; text-align: center; border-radius: 8px; padding: 0.35rem 0; font-size: 0.72rem; font-weight: 700; }
.badge-pdf { background: #FDE8E8; color: #B42318; }
.badge-txt { background: #E0ECFB; color: #1A5DB4; }
.badge-md  { background: #ECE8FA; color: #5B3CC4; }
.doc-name { font-size: 0.93rem; font-weight: 600; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.doc-meta { font-size: 0.78rem; color: #6B7A75; }
.doc-warn { font-size: 0.78rem; color: #B45309; }
.preview { font-size: 0.82rem; color: #44524E; line-height: 1.55; white-space: pre-wrap; max-width: 420px; }

/* Chat */
[data-testid="stChatMessage"] { background: #FFFFFF; border: 1px solid #DDE4DF; border-radius: 16px; padding: 1rem 1.1rem; margin-bottom: 0.7rem; }
[data-testid="stChatInput"] { border-radius: 14px; background: #FFFFFF; }
[data-testid="stChatInput"] > div { border: 1px solid #C9D3CE !important; border-radius: 14px; }
[data-testid="stChatInput"] > div:focus-within { border-color: #0F5C4D !important; box-shadow: 0 0 0 1px #0F5C4D; }
.meta-line { font-size: 0.76rem; color: #6B7A75; margin-top: 0.3rem; }

/* Sources */
.source-card { background: #F5F7F4; border: 1px solid #DDE4DF; border-left: 4px solid #0F5C4D; border-radius: 10px; padding: 0.7rem 0.9rem; margin-bottom: 0.6rem; }
.source-head { display: flex; justify-content: space-between; gap: 0.6rem; margin-bottom: 0.3rem; }
.source-title { font-weight: 600; font-size: 0.85rem; color: #0F5C4D; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.match { flex-shrink: 0; background: #FBEFC9; color: #8A5A00; font-size: 0.72rem; font-weight: 600; border-radius: 999px; padding: 0.1rem 0.55rem; }
.source-text { font-size: 0.85rem; color: #44524E; line-height: 1.55; white-space: pre-wrap; }

/* Buttons */
div.stButton > button, div.stDownloadButton > button, [data-testid="stPopover"] > div > button {
    border-radius: 12px; border: 1px solid #C9D3CE; background: #FFFFFF; font-weight: 500; min-height: 2.5rem; transition: all 0.15s ease; }
div.stButton > button:hover, div.stDownloadButton > button:hover, [data-testid="stPopover"] > div > button:hover { border-color: #0F5C4D; color: #0F5C4D; }
button[kind="primary"], [data-testid="stBaseButton-primary"] { background: #0F5C4D !important; border: 1px solid #0F5C4D !important; color: #FFFFFF !important; font-weight: 600 !important; }
button[kind="primary"]:hover, [data-testid="stBaseButton-primary"]:hover { background: #0B4A3E !important; color: #FFFFFF !important; }
button[kind="primary"] p, [data-testid="stBaseButton-primary"] p { color: #FFFFFF !important; }
button[kind="primary"]:disabled, [data-testid="stBaseButton-primary"]:disabled { background: #D5DCD8 !important; border-color: #D5DCD8 !important; opacity: 1; }
button:focus-visible { outline: 2px solid #E3A008 !important; outline-offset: 2px; }
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


@st.cache_resource
def load_json_model():
    # Same model, but forced to reply in JSON. Used to pull chart data out of text.
    return ChatOllama(model=LLM_MODEL, temperature=0, num_ctx=4096, format="json")


@st.cache_resource(show_spinner="Loading the image model. The first run downloads it, so this can take a few minutes...")
def load_image_pipe():
    import torch
    from diffusers import AutoPipelineForText2Image

    if torch.cuda.is_available():
        device, dtype = "cuda", torch.float16
    elif torch.backends.mps.is_available():
        device, dtype = "mps", torch.float32  # float32 avoids black images on Apple chips
    else:
        device, dtype = "cpu", torch.float32
    pipe = AutoPipelineForText2Image.from_pretrained(IMAGE_MODEL, torch_dtype=dtype, variant="fp16")
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    return pipe


db = load_db()
model = load_model()
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)


@st.cache_data(ttl=10)
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


def format_bytes(n):
    if n < 1024:
        return f"{n} B"
    if n < 1024**2:
        return f"{n / 1024:.0f} KB"
    return f"{n / 1024**2:.1f} MB"


def file_size(path):
    return os.path.getsize(path) if os.path.exists(path) else 0


def modified_ts(path):
    try:
        return os.path.getmtime(path)
    except OSError:
        return 0


def modified_date(path):
    ts = modified_ts(path)
    return datetime.fromtimestamp(ts).strftime("%d %b %Y") if ts else ""


def list_files():
    return sorted(f for f in os.listdir(DOCS_DIR) if f.lower().rsplit(".", 1)[-1] in ALLOWED_TYPES)


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


def text_preview(filename, limit=1200):
    try:
        with open(doc_path(filename), encoding="utf-8", errors="ignore") as fh:
            text = fh.read(limit + 1)
        return text[:limit] + ("…" if len(text) > limit else "")
    except OSError:
        return ""


def citation_counts():
    cited = Counter()
    for m in st.session_state.messages:
        for s in m.get("sources", []) if m["role"] == "assistant" else []:
            cited[re.sub(r" \(page \d+\)$", "", s["file"])] += 1
    return cited


# ------------------------------ Studio helpers -----------------------------
def to_number(v):
    """Turn '3,329,445', 'Nu. 240,000' or '12%' into a float. Returns None if there is no number."""
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        m = re.search(r"-?\d[\d,]*\.?\d*", v.replace(" ", ""))
        if m:
            try:
                return float(m.group(0).replace(",", ""))
            except ValueError:
                return None
    return None


def fmt_num(v):
    return f"{v:,.0f}" if abs(v) >= 100 or float(v).is_integer() else f"{v:,.2f}"


def parse_json(text):
    try:
        return json.loads(text)
    except Exception:
        match = re.search(r"\{.*\}", text, re.S)
        if match:
            try:
                return json.loads(match.group(0))
            except Exception:
                pass
    raise ValueError("The AI did not return usable data. Try rephrasing your request, or use pasted data.")


def chart_frame(raw):
    """Turn the AI's JSON into a DataFrame: one row per label, one column per series."""
    if raw.get("error"):
        raise ValueError(f"No chart could be made: {raw['error']}.")
    series_raw = raw.get("series")
    if not series_raw and raw.get("labels") and raw.get("values"):
        series_raw = [{"name": raw.get("name") or "Value", "points": [{"label": a, "value": b} for a, b in zip(raw["labels"], raw["values"])]}]
    if not isinstance(series_raw, list):
        series_raw = []

    labels = []
    for s in series_raw:
        for p in s.get("points", []) if isinstance(s, dict) else []:
            label = str(p.get("label", "")).strip()
            if label and label not in labels:
                labels.append(label)

    columns = {}
    for s in series_raw:
        if not isinstance(s, dict):
            continue
        lookup = {}
        for p in s.get("points", []):
            value = to_number(p.get("value"))
            if value is not None:
                lookup[str(p.get("label", "")).strip()] = value
        if lookup:
            name = str(s.get("name") or "Value")
            while name in columns:
                name += " (2)"
            columns[name] = [lookup.get(label) for label in labels]

    if not labels or not columns:
        raise ValueError("No numbers were found to chart. Try describing the data more specifically, or use pasted data.")
    df = pd.DataFrame(columns, index=labels).head(20)
    df.index.name = str(raw.get("x_label") or "")
    return df


def parse_pasted_data(text):
    df = pd.read_csv(io.StringIO(text.strip()), sep=None, engine="python")
    if df.shape[1] < 2:
        raise ValueError("Use at least two columns: labels first, then numbers. Put the column names in the first row.")
    df = df.set_index(df.columns[0])
    df = df.apply(lambda col: col.map(to_number)).dropna(axis=1, how="all")
    if df.empty:
        raise ValueError("No numeric columns were found in the pasted data.")
    df.index = df.index.astype(str)
    return df.head(20)


def draw_chart(df, kind, title, x_label="", y_label=""):
    """Draw the chart in the app's palette and return PNG bytes."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(8.2, 4.8), dpi=170)
    fig.patch.set_facecolor("white")
    labels = [short_name(str(i), 22) for i in df.index]
    n = len(df.columns)
    colors = [PALETTE[i % len(PALETTE)] for i in range(max(n, len(labels)))]

    if kind == "Pie":
        values = df.iloc[:, 0].fillna(0).clip(lower=0)
        if values.sum() <= 0:
            raise ValueError("A pie chart needs positive numbers.")
        ax.pie(
            values, labels=labels, colors=colors, startangle=90, counterclock=False,
            autopct=lambda p: f"{p:.0f}%" if p >= 4 else "",
            wedgeprops={"linewidth": 1.5, "edgecolor": "white"}, textprops={"fontsize": 9, "color": INK},
        )
        ax.axis("equal")
    elif kind == "Horizontal bar":
        h = 0.8 / n
        y = np.arange(len(labels))
        for j, col in enumerate(df.columns):
            vals = df[col].fillna(0)
            bars = ax.barh(y + (j - (n - 1) / 2) * h, vals, height=h * 0.92, color=PALETTE[j % len(PALETTE)], label=str(col))
            if n == 1:
                ax.bar_label(bars, labels=[fmt_num(v) for v in vals], padding=3, fontsize=8, color=INK)
        ax.set_yticks(y, labels)
        ax.invert_yaxis()
        ax.grid(axis="x", color="#E5EAE7", linewidth=0.8)
    elif kind == "Line":
        for j, col in enumerate(df.columns):
            ax.plot(labels, df[col], marker="o", linewidth=2.2, markersize=5, color=PALETTE[j % len(PALETTE)], label=str(col))
        ax.grid(axis="y", color="#E5EAE7", linewidth=0.8)
    else:  # Bar
        w = 0.8 / n
        x = np.arange(len(labels))
        for j, col in enumerate(df.columns):
            vals = df[col].fillna(0)
            bars = ax.bar(x + (j - (n - 1) / 2) * w, vals, width=w * 0.92, color=PALETTE[j % len(PALETTE)], label=str(col))
            if n == 1 and len(labels) <= 12:
                ax.bar_label(bars, labels=[fmt_num(v) for v in vals], padding=3, fontsize=8, color=INK)
        ax.set_xticks(x, labels)
        ax.grid(axis="y", color="#E5EAE7", linewidth=0.8)

    if kind != "Pie":
        from matplotlib.ticker import FuncFormatter

        value_axis = ax.xaxis if kind == "Horizontal bar" else ax.yaxis
        value_axis.set_major_formatter(FuncFormatter(lambda v, _: fmt_num(v)))  # 1,100,000 instead of 1e6
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color("#C9D3CE")
        ax.tick_params(colors="#44524E", labelsize=8.5, length=0)
        if kind != "Horizontal bar" and (len(labels) > 5 or max(len(label) for label in labels) > 10):
            plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
        label_axis, value_title = (ax.set_ylabel, ax.set_xlabel) if kind == "Horizontal bar" else (ax.set_xlabel, ax.set_ylabel)
        if x_label:
            label_axis(x_label, fontsize=9, color="#44524E")
        if y_label:
            value_title(y_label, fontsize=9, color="#44524E")
        if n > 1:
            ax.legend(frameon=False, fontsize=8.5)
    ax.set_title(title, loc="left", fontsize=13, fontweight="bold", color=INK, pad=14)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return buf.getvalue()


def collect_context(query, doc_choice):
    flt = {"source": {"$in": [doc_path(doc_choice)]}} if doc_choice and doc_choice != "All documents" else None
    found = db.similarity_search(query, k=6, filter=flt)
    if not found:
        raise ValueError("No passages were found in your documents.")
    return "\n\n".join(d.page_content for d in found)[:CONTEXT_CHARS]


def build_graph(source, description, doc_choice, pasted, chat_text):
    """Return (dataframe, title, x_label, y_label, note) for the chosen data source."""
    if source == SRC_DATA:
        df = parse_pasted_data(pasted)
        return df, description or "Chart from pasted data", df.index.name or "", "", "Built from the data you pasted."

    if source == SRC_DOCS:
        context = collect_context(description or "budget cost totals figures numbers percentages table", doc_choice)
        rules, note = GRAPH_FROM_TEXT, "Numbers were read from your documents by the AI. Check them against the source."
    elif source == SRC_CHAT:
        context, rules, note = chat_text[:CONTEXT_CHARS], GRAPH_FROM_TEXT, "Numbers were read from the last chat answer."
    else:
        context, rules, note = "", GRAPH_FROM_KNOWLEDGE, "These figures come from the AI's own knowledge and may be approximate. Verify them before use."

    task = f"Chart request: {description or 'the most important numeric data'}"
    body = f"Context:\n{context}\n\n{task}" if context else task
    reply = load_json_model().invoke([SystemMessage(content=rules), HumanMessage(content=body)])
    raw = parse_json(reply.content)
    df = chart_frame(raw)
    return df, str(raw.get("title") or description or "Chart"), df.index.name or "", str(raw.get("y_label") or ""), note


def build_image_prompt(idea, context, style):
    """Ask the local AI for a short visual prompt, then add the chosen style."""
    material = f"Report excerpt:\n{context}\n\nFocus: {idea or 'the main theme'}" if context else f"Idea: {idea}"
    reply = model.invoke([SystemMessage(content=IMAGE_PROMPT_RULES), HumanMessage(content=material)])
    text = " ".join(reply.content.strip().strip('"').split()[:45])
    return f"{IMAGE_STYLES[style]}, {text}"


def generate_image(prompt, size, steps, seed):
    import torch

    pipe = load_image_pipe()
    width, height = IMAGE_SIZES[size]
    generator = torch.Generator(device="cpu").manual_seed(int(seed)) if seed else None
    result = pipe(prompt=prompt, num_inference_steps=int(steps), guidance_scale=0.0, width=width, height=height, generator=generator)
    buf = io.BytesIO()
    result.images[0].save(buf, format="PNG")
    return buf.getvalue()


def save_generated(png, prefix, title, source):
    path = os.path.join(GEN_DIR, f"{prefix}-{datetime.now().strftime('%Y%m%d-%H%M%S')}.png")
    with open(path, "wb") as out:
        out.write(png)
    st.session_state.gallery.insert(0, {"kind": prefix, "title": title, "png": png, "source": source, "path": path})
    del st.session_state.gallery[12:]
    return path


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


def reindex_all():
    flash = []
    for f in list_files():
        try:
            flash.append(("success", f"Re-indexed {f} ({index_file(f)} passages)"))
        except Exception as e:
            flash.append(("error", f"{f}: {e}"))
    st.session_state.flash = flash or [("info", "No documents to index.")]


def summarize_doc(filename):
    st.session_state.scope = [filename]
    st.session_state.pending_query = f'Give me a clear summary of the document "{filename}".'
    st.session_state.pending_k = 6
    st.session_state.page = PAGE_CHAT


def send_to_studio(kind, text):
    """Open the Studio with the last chat answer ready to use as the source."""
    st.session_state.chat_seed = text
    st.session_state.studio_tool = TOOL_GRAPHS if kind == "graph" else TOOL_IMAGES
    st.session_state["graph_source" if kind == "graph" else "image_source"] = SRC_CHAT
    st.session_state.page = PAGE_STUDIO


def clear_seed():
    st.session_state.chat_seed = ""


def clear_chat():
    st.session_state.messages = []


def clear_gallery():
    st.session_state.gallery = []


def export_markdown():
    lines = ["# Knowledge Assistant: conversation", ""]
    for m in st.session_state.messages:
        who = "You" if m["role"] == "user" else "Assistant"
        lines += [f"**{who}:** {m['content']}", ""]
        if m["role"] == "assistant" and m.get("sources"):
            names = sorted({s["file"] for s in m["sources"]})
            lines += [f"_Sources: {', '.join(names)}_", ""]
    return "\n".join(lines)


# ------------------------------- Session state -----------------------------
st.session_state.setdefault("messages", [])
st.session_state.setdefault("uploader_key", 0)
st.session_state.setdefault("page", PAGE_CHAT)
st.session_state.setdefault("gallery", [])
st.session_state.setdefault("chat_seed", "")

# Keep settings alive when their widgets are not on screen (for example while on another page)
for _key, _default in {
    "scope": [], "answer_style": "Concise", "top_k": 3, "show_src": True, "chat_mode": MODE_HYBRID,
    "studio_tool": TOOL_GRAPHS, "graph_source": SRC_DOCS, "image_source": SRC_DOCS,
    "chart_type": CHART_TYPES[0], "image_style": "Illustration", "image_size": "Square",
}.items():
    st.session_state[_key] = st.session_state.get(_key, _default)

files = list_files()
counts = indexed_counts()
total_chunks = sum(counts.get(doc_path(f), 0) for f in files)
st.session_state.scope = [s for s in st.session_state.scope if s in files]

for kind, msg in st.session_state.pop("flash", []):
    st.toast(msg, icon={"success": "✅", "error": "⚠️", "info": "ℹ️"}[kind])

# --------------------------------- Top bar ---------------------------------
c_brand, c_nav, c_status = st.columns([2.2, 3.4, 2.0], vertical_alignment="center")
c_brand.markdown(
    "<div class='brand'><span class='brand-mark'>K</span><div>"
    "<div class='brand-name'>Knowledge Assistant</div>"
    "<div class='brand-sub'>Private answers from your own files</div></div></div>",
    unsafe_allow_html=True,
)
c_nav.radio("Navigation", [PAGE_CHAT, PAGE_LIB, PAGE_STUDIO], key="page", horizontal=True, label_visibility="collapsed")
running = ollama_running()
c_status.markdown(
    f"<span class='status-chip'><span class='dot {'dot-on' if running else 'dot-off'}'></span>"
    f"{LLM_MODEL + ' ready' if running else 'Offline · open Ollama'}</span>",
    unsafe_allow_html=True,
)
st.markdown("<div class='topbar-rule'></div>", unsafe_allow_html=True)


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


def render_meta(msg):
    parts = ["From your documents" if msg.get("mode", MODE_DOCS) == MODE_DOCS else "General knowledge, not from your files"]
    if msg.get("elapsed"):
        parts.append(f"{msg['elapsed']:.1f} s")
    st.markdown(f"<div class='meta-line'>{' · '.join(parts)}</div>", unsafe_allow_html=True)


def render_answer_actions(text, key):
    """Shortcuts under an answer: send it to the Studio as a graph or an image."""
    a, b, _ = st.columns([1.3, 1.3, 3])
    a.button("📊 Make a graph", key=f"g_{key}", on_click=send_to_studio, args=("graph", text), use_container_width=True)
    b.button("🖼️ Make an image", key=f"i_{key}", on_click=send_to_studio, args=("image", text), use_container_width=True)


def save_and_index(uploaded):
    """Save each uploaded file into the docs folder, then index it. Returns toast messages."""
    flash = []
    bar = st.progress(0.0, text="Starting...")
    for i, f in enumerate(uploaded):
        name = safe_name(f.name)
        bar.progress(i / len(uploaded), text=f"Saving and indexing {short_name(name, 40)}...")
        if f.size > MAX_MB * 1024**2:
            flash.append(("error", f"{name}: larger than {MAX_MB} MB, so it was skipped"))
            continue
        existed = os.path.exists(doc_path(name))
        with open(doc_path(name), "wb") as out:
            out.write(f.getbuffer())
        try:
            n = index_file(name)
            flash.append(("success", f"{'Updated' if existed else 'Added'} {name} ({n} passages), saved in {DOCS_DIR}/"))
        except Exception as e:
            os.remove(doc_path(name))
            flash.append(("error", f"{name}: {e}"))
    bar.progress(1.0, text="Done")
    return flash


def render_uploader(show_header=True):
    """The upload box. Used on the Documents page and inside the Chat page."""
    if show_header:
        st.markdown(
            "<div class='upload-head'><h3>Drop your files here</h3>"
            "<p>Each file is saved into your project's docs folder, so it also appears in VS Code, then indexed for chat.</p>"
            f"<span class='path-box' title='{html.escape(DOCS_ABS)}'>{html.escape(DOCS_ABS)}</span></div>",
            unsafe_allow_html=True,
        )
    uploaded = st.file_uploader(
        "Upload files",
        type=ALLOWED_TYPES,
        accept_multiple_files=True,
        key=f"uploader_{st.session_state.uploader_key}",
        label_visibility="collapsed",
    )
    st.caption(f"TXT, MD or PDF · up to {MAX_MB} MB each · files with the same name are replaced")
    if uploaded:
        chips = "".join(
            f"<span class='chip{' chip-replace' if os.path.exists(doc_path(safe_name(f.name))) else ''}'>"
            f"{html.escape(short_name(f.name, 28))}<em>{format_bytes(f.size)}"
            f"{' · replaces existing' if os.path.exists(doc_path(safe_name(f.name))) else ''}</em></span>"
            for f in uploaded
        )
        st.markdown(chips, unsafe_allow_html=True)
    label = f"Add {len(uploaded)} file(s) to knowledge base" if uploaded else "Add to knowledge base"
    if st.button(label, type="primary", use_container_width=True, disabled=not uploaded, key="add_btn"):
        st.session_state.flash = save_and_index(uploaded)
        st.session_state.uploader_key += 1
        st.rerun()


def render_library():
    st.markdown(
        "<div class='page-title'>Documents</div>"
        "<div class='page-sub'>Upload files and manage what the assistant can answer from.</div>",
        unsafe_allow_html=True,
    )

    questions = sum(1 for m in st.session_state.messages if m["role"] == "user")
    storage = format_bytes(sum(file_size(doc_path(f)) for f in files))
    stats = [(len(files), "Documents"), (total_chunks, "Passages"), (questions, "Questions asked"), (storage, "Storage used")]
    for col, (num, label) in zip(st.columns(4), stats):
        col.markdown(
            f"<div class='stat'><div class='stat-num'>{num}</div><div class='stat-label'>{label}</div></div>",
            unsafe_allow_html=True,
        )

    st.markdown("<div class='section-title'>Add documents</div>", unsafe_allow_html=True)
    with st.container(border=True):
        render_uploader()

    st.markdown("<div class='section-title'>Your documents</div>", unsafe_allow_html=True)
    if not files:
        st.markdown(
            "<div class='empty'><b>No documents yet</b>Drop a file in the box above and it will be ready to chat with in seconds.</div>",
            unsafe_allow_html=True,
        )
        return

    f1, f2 = st.columns([3, 1.3])
    search = f1.text_input("Filter", placeholder="🔍 Filter documents by name", label_visibility="collapsed")
    sort_by = f2.selectbox("Sort", SORTS, label_visibility="collapsed")
    shown = [f for f in files if search.lower() in f.lower()]
    if sort_by == "Newest first":
        shown.sort(key=lambda f: modified_ts(doc_path(f)), reverse=True)
    elif sort_by == "Largest first":
        shown.sort(key=lambda f: file_size(doc_path(f)), reverse=True)
    if not shown:
        st.caption("No documents match your filter.")

    cited = citation_counts()
    for f in shown:
        n = counts.get(doc_path(f), 0)
        ext = f.rsplit(".", 1)[-1].lower()
        if n:
            extra = f" · cited {cited[f]}×" if cited.get(f) else ""
            meta = (
                f"<div class='doc-meta'>{format_bytes(file_size(doc_path(f)))} · {n} passages · "
                f"{modified_date(doc_path(f))}{extra}</div>"
            )
        else:
            meta = "<div class='doc-warn'>Not indexed yet</div>"
        with st.container(border=True):
            a, b, c, d, e = st.columns([1, 5, 0.8, 0.8, 0.8], vertical_alignment="center")
            a.markdown(f"<span class='badge badge-{ext}'>{ext.upper()}</span>", unsafe_allow_html=True)
            b.markdown(f"<div class='doc-name' title='{html.escape(f)}'>{html.escape(f)}</div>{meta}", unsafe_allow_html=True)
            with c.popover("👁", help="Preview", use_container_width=True):
                if ext == "pdf":
                    st.caption("PDF preview is not available. Use Summarize to see what is inside.")
                else:
                    st.markdown(f"<div class='preview'>{html.escape(text_preview(f))}</div>", unsafe_allow_html=True)
            d.button("📝", key=f"sum_{f}", on_click=summarize_doc, args=(f,), help="Summarize this document", use_container_width=True, disabled=not n)
            e.button("🗑️", key=f"del_{f}", on_click=delete_document, args=(f,), help="Remove this document", use_container_width=True)

    st.markdown("<div style='height:0.4rem'></div>", unsafe_allow_html=True)
    b1, b2 = st.columns(2)
    b1.button("🔄 Re-index all documents", on_click=reindex_all, use_container_width=True)
    if any(counts.get(doc_path(f), 0) == 0 for f in files):
        b2.button("➕ Index missing documents", on_click=sync_unindexed, use_container_width=True)


def render_chat():
    st.markdown(
        "<div class='hero'><h1>Ask your documents</h1>"
        "<p>Answers come from your files or from the AI's general knowledge, and every answer says which.</p></div>",
        unsafe_allow_html=True,
    )

    # Toolbar: settings, export, clear
    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
    t1, t2, t3 = st.columns(3)
    with t1.popover("⚙️ Chat settings", use_container_width=True):
        st.multiselect(
            "Search in",
            options=files,
            key="scope",
            placeholder="All documents",
            format_func=lambda f: short_name(f, 32),
            help="Pick one or more documents to limit the search. Leave empty to search everything.",
        )
        st.selectbox("Answer style", list(STYLES), key="answer_style")
        st.slider("Passages to search", 1, 8, key="top_k", help="More passages give broader answers but slower replies.")
        st.toggle("Show sources", key="show_src")
    t2.download_button(
        "⬇️ Export chat",
        data=export_markdown(),
        file_name="conversation.md",
        mime="text/markdown",
        use_container_width=True,
        disabled=not st.session_state.messages,
    )
    t3.button("🗑️ Clear chat", on_click=clear_chat, use_container_width=True, disabled=not st.session_state.messages)

    st.markdown("<div style='height:0.9rem'></div>", unsafe_allow_html=True)
    st.radio("Answer mode", MODES, key="chat_mode", horizontal=True, label_visibility="collapsed")
    st.markdown(f"<div class='mode-help'>{html.escape(MODE_HELP[st.session_state.chat_mode])}</div>", unsafe_allow_html=True)

    scope, style, k, show_sources = (st.session_state.scope, st.session_state.answer_style, st.session_state.top_k, st.session_state.show_src)
    mode = st.session_state.chat_mode

    if mode == MODE_GENERAL:
        pill = "🌐 General AI: your documents are not used"
    else:
        pill = "🔎 " + (
            "Searching in: " + ", ".join(short_name(s, 28) for s in scope)
            if scope
            else f"Searching in all {len(files)} document(s)"
        )
    st.markdown(f"<div class='scope-pill'>{html.escape(pill)}</div>", unsafe_allow_html=True)

    # Upload box right on the chat page, open by default until the first document is added
    with st.expander("📎 Add documents", expanded=len(files) == 0):
        render_uploader(show_header=False)

    needs_docs = mode == MODE_DOCS and total_chunks == 0
    if needs_docs:
        placeholder = "Add a document above, or switch to Smart or General AI mode"
    elif mode == MODE_GENERAL:
        placeholder = "Ask me anything..."
    else:
        placeholder = "Ask a question..."
    typed = st.chat_input(placeholder, disabled=needs_docs)
    pending = st.session_state.pop("pending_query", None)
    pending_k = st.session_state.pop("pending_k", None)
    query = typed or pending
    k_use = pending_k or k

    last = len(st.session_state.messages) - 1
    for i, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"], avatar="🙂" if msg["role"] == "user" else "📚"):
            st.markdown(msg["content"])
            if msg["role"] == "assistant":
                render_meta(msg)
                if show_sources and msg.get("sources"):
                    render_sources(msg["sources"])
                if i == last and not query and msg["content"].strip():
                    render_answer_actions(msg["content"], f"h{i}")

    if not st.session_state.messages and not query:
        if needs_docs:
            st.markdown(
                "<div class='empty'><b>Your library is empty</b>Upload a TXT, MD or PDF file above, "
                "or switch to Smart or General AI mode to chat without documents.</div>",
                unsafe_allow_html=True,
            )
        elif mode == MODE_GENERAL or total_chunks == 0:
            st.markdown(
                "<div class='empty'><b>Ask me anything</b>Type your message in the box below.</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"<div class='empty'><b>{len(files)} document(s) ready</b>Type your question in the box below.</div>",
                unsafe_allow_html=True,
            )

    if not query:
        return

    history = list(st.session_state.messages)
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user", avatar="🙂"):
        st.markdown(query)

    with st.chat_message("assistant", avatar="📚"):
        try:
            started = time.perf_counter()
            results, used = [], MODE_GENERAL
            if mode != MODE_GENERAL and total_chunks:
                # Short follow-ups like "tell me more" borrow the previous question for searching
                search_q = query
                prev_users = [m["content"] for m in history if m["role"] == "user"]
                if len(query.split()) < 5 and prev_users:
                    search_q = prev_users[-1] + " " + query

                flt = {"source": {"$in": [doc_path(f) for f in scope]}} if scope else None
                with st.spinner("Searching your documents..."):
                    results = db.similarity_search_with_score(search_q, k=k_use, filter=flt)
                if mode == MODE_DOCS and not results:
                    raise ValueError("No passages found in the selected documents.")
                best = max((1.0 - float(dist) for _, dist in results), default=0.0)
                if mode == MODE_DOCS or best >= RELEVANCE_MIN:
                    used = MODE_DOCS
                else:
                    results = []  # Smart mode: documents do not cover this, fall back to general knowledge

            if used == MODE_DOCS:
                context = "\n\n".join(d.page_content for d, _ in results)
                prompt = SYSTEM_PROMPT if mode == MODE_DOCS else HYBRID_PROMPT
                final_question = f"Context:\n{context}\n\nQuestion: {query}"
            else:
                prompt = GENERAL_PROMPT
                final_question = query

            messages = [SystemMessage(content=f"{prompt} {STYLES[style]}")]
            for m in history[-6:]:
                messages.append(HumanMessage(content=m["content"]) if m["role"] == "user" else AIMessage(content=m["content"]))
            messages.append(HumanMessage(content=final_question))

            answer = st.write_stream(stream_answer(messages))
            elapsed = time.perf_counter() - started
            render_meta({"mode": used, "elapsed": elapsed})

            sources = []
            for d, dist in results:
                label = os.path.basename(d.metadata.get("source", "unknown"))
                if "page" in d.metadata:
                    label += f" (page {int(d.metadata['page']) + 1})"
                sources.append({"file": label, "text": d.page_content, "score": max(0.0, 1.0 - float(dist))})
            if show_sources and sources:
                render_sources(sources)
            if answer.strip():
                render_answer_actions(answer, "new")

            st.session_state.messages.append(
                {"role": "assistant", "content": answer, "sources": sources, "elapsed": elapsed, "mode": used}
            )
        except Exception as e:
            st.error(
                "Could not get an answer. Make sure the Ollama app is open and the model is downloaded "
                f"(`ollama pull {LLM_MODEL}`).\n\nDetails: {e}"
            )


# ---------------------------------- Studio ---------------------------------
def render_seed_card():
    """Shows the chat answer that is being used as the source, with a button to drop it."""
    seed = st.session_state.chat_seed
    st.markdown(f"<div class='quote'>{html.escape(short_name(seed.strip(), 420))}</div>", unsafe_allow_html=True)
    st.button("Remove this answer", key="seed_clear", on_click=clear_seed)


def render_graph_tool():
    seed = st.session_state.chat_seed
    sources = [SRC_DOCS, SRC_AI, SRC_DATA] + ([SRC_CHAT] if seed else [])
    if st.session_state.graph_source not in sources:
        st.session_state.graph_source = SRC_DOCS

    with st.container(border=True):
        st.markdown(
            "<div class='panel-title'>Create a graph</div>"
            "<div class='panel-sub'>Pick where the numbers come from, choose a chart style, then generate.</div>",
            unsafe_allow_html=True,
        )
        st.radio("Data source", sources, key="graph_source", horizontal=True, label_visibility="collapsed")
        source = st.session_state.graph_source
        st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

        doc_choice = "All documents"
        c1, c2 = st.columns(2)
        c1.selectbox("Chart type", CHART_TYPES, key="chart_type")
        if source == SRC_DOCS:
            options = ["All documents"] + files
            if st.session_state.get("graph_doc") not in options:
                st.session_state.graph_doc = "All documents"
            doc_choice = c2.selectbox("Report to read", options, key="graph_doc", format_func=lambda f: short_name(f, 34))
        elif source == SRC_DATA:
            c2.text_input("Chart title", key="graph_desc", placeholder="Optional title")
        else:
            c2.markdown("")

        if source == SRC_CHAT:
            render_seed_card()
        if source == SRC_DATA:
            st.text_area(
                "Paste your data",
                key="graph_data",
                height=140,
                placeholder="Block,Cost\nB1,239800\nB2,162500\nB3,1101500",
                help="First row is the column names. First column is the labels. Separate columns with commas, tabs or semicolons.",
            )
        else:
            st.text_input(
                "What should the graph show?",
                key="graph_desc",
                placeholder="For example: cost of each work block" if source != SRC_AI else "For example: population of the largest districts of Bhutan",
            )

        description = (st.session_state.get("graph_desc") or "").strip()
        pasted = (st.session_state.get("graph_data") or "").strip()
        blocked = (
            (source == SRC_DOCS and total_chunks == 0)
            or (source == SRC_AI and not description)
            or (source == SRC_DATA and not pasted)
        )
        if source == SRC_DOCS and total_chunks == 0:
            st.markdown("<div class='note'>Add a document on the Documents page first.</div>", unsafe_allow_html=True)
        if source == SRC_AI:
            st.markdown("<div class='note'>Figures come from the AI's own knowledge, so they may be approximate.</div>", unsafe_allow_html=True)

        st.markdown("<div style='height:0.4rem'></div>", unsafe_allow_html=True)
        clicked = st.button("Generate graph", type="primary", use_container_width=True, disabled=blocked, key="graph_go")

    if clicked:
        try:
            with st.spinner("Reading the data and drawing your graph..."):
                df, title, x_label, y_label, note = build_graph(source, description, doc_choice, pasted, seed)
                png = draw_chart(df, st.session_state.chart_type, title, x_label, y_label)
            save_generated(png, "graph", title, source)
            st.session_state.graph_result = {"png": png, "df": df, "title": title, "note": note}
        except Exception as e:
            st.session_state.graph_result = None
            st.error(f"Could not create the graph. {e}")

    result = st.session_state.get("graph_result")
    if result:
        st.markdown("<div class='section-title'>Your graph</div>", unsafe_allow_html=True)
        with st.container(border=True):
            st.image(result["png"])
            st.markdown(f"<div class='note'>{html.escape(result['note'])} Saved in <code>{GEN_DIR}/</code>.</div>", unsafe_allow_html=True)
            d1, d2 = st.columns(2)
            d1.download_button("⬇️ Download PNG", result["png"], file_name="graph.png", mime="image/png", use_container_width=True, key="dl_graph_png")
            d2.download_button("⬇️ Download data (CSV)", result["df"].to_csv().encode("utf-8"), file_name="graph-data.csv", mime="text/csv", use_container_width=True, key="dl_graph_csv")
            with st.expander("View the data behind the graph"):
                st.dataframe(result["df"], use_container_width=True)


def render_image_tool():
    seed = st.session_state.chat_seed
    sources = [SRC_DOCS, SRC_DESC] + ([SRC_CHAT] if seed else [])
    if st.session_state.image_source not in sources:
        st.session_state.image_source = SRC_DOCS

    with st.container(border=True):
        st.markdown(
            "<div class='panel-title'>Create an image</div>"
            "<div class='panel-sub'>Generate an illustration from a report, a chat answer, or your own idea. "
            "It runs on your computer.</div>",
            unsafe_allow_html=True,
        )
        st.radio("Image source", sources, key="image_source", horizontal=True, label_visibility="collapsed")
        source = st.session_state.image_source
        st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

        doc_choice = "All documents"
        c1, c2, c3 = st.columns(3)
        c1.selectbox("Style", list(IMAGE_STYLES), key="image_style")
        c2.selectbox("Shape", list(IMAGE_SIZES), key="image_size")
        if source == SRC_DOCS:
            options = ["All documents"] + files
            if st.session_state.get("image_doc") not in options:
                st.session_state.image_doc = "All documents"
            doc_choice = c3.selectbox("Report to read", options, key="image_doc", format_func=lambda f: short_name(f, 22))
        else:
            c3.markdown("")

        if source == SRC_CHAT:
            render_seed_card()
        label = "Describe the image" if source == SRC_DESC else "Focus (optional)"
        st.text_area(
            label,
            key="image_desc",
            height=90,
            placeholder="For example: a dzong beside a river in autumn" if source == SRC_DESC else "For example: the drone survey of the heritage site",
        )
        improve = True
        if source == SRC_DESC:
            improve = st.toggle("Let the AI improve my description", value=True, key="image_improve")

        with st.expander("Advanced"):
            a1, a2 = st.columns(2)
            a1.slider("Quality steps", 1, 8, 4, key="image_steps", help="More steps can add detail but take longer. 4 is a good balance.")
            a2.number_input("Seed", min_value=0, value=0, step=1, key="image_seed", help="Use the same number to repeat a result. 0 means random.")

        idea = (st.session_state.get("image_desc") or "").strip()
        blocked = (source == SRC_DOCS and total_chunks == 0) or (source == SRC_DESC and not idea)
        if source == SRC_DOCS and total_chunks == 0:
            st.markdown("<div class='note'>Add a document on the Documents page first.</div>", unsafe_allow_html=True)
        st.markdown(
            "<div class='note'>Images are illustrations. They cannot show exact diagrams, text or real people. "
            "For data, use Graphs. The first image downloads the model, which can take a few minutes.</div>",
            unsafe_allow_html=True,
        )
        st.markdown("<div style='height:0.4rem'></div>", unsafe_allow_html=True)
        clicked = st.button("Generate image", type="primary", use_container_width=True, disabled=blocked, key="image_go")

    if clicked:
        try:
            style = st.session_state.image_style
            with st.spinner("Writing the prompt..."):
                if source == SRC_DESC and not improve:
                    prompt = f"{IMAGE_STYLES[style]}, {idea}"
                elif source == SRC_DESC:
                    prompt = build_image_prompt(idea, "", style)
                elif source == SRC_CHAT:
                    prompt = build_image_prompt(idea, seed[:CONTEXT_CHARS], style)
                else:
                    prompt = build_image_prompt(idea, collect_context(idea or "main theme and key findings", doc_choice), style)
            with st.spinner("Generating the image..."):
                png = generate_image(prompt, st.session_state.image_size, st.session_state.image_steps, st.session_state.image_seed)
            title = short_name(idea or prompt.split(", ", 1)[-1], 60)
            save_generated(png, "image", title, source)
            st.session_state.image_result = {"png": png, "prompt": prompt}
        except ImportError:
            st.session_state.image_result = None
            st.error(
                "The image tools are not installed yet. Run this in your terminal, then restart the app:\n\n"
                "`pip install torch diffusers transformers accelerate safetensors`"
            )
        except Exception as e:
            st.session_state.image_result = None
            st.error(f"Could not create the image. {e}")

    result = st.session_state.get("image_result")
    if result:
        st.markdown("<div class='section-title'>Your image</div>", unsafe_allow_html=True)
        with st.container(border=True):
            st.image(result["png"])
            st.markdown(f"<div class='note'>Saved in <code>{GEN_DIR}/</code>.</div>", unsafe_allow_html=True)
            st.download_button("⬇️ Download PNG", result["png"], file_name="image.png", mime="image/png", use_container_width=True, key="dl_image_png")
            with st.expander("Prompt used"):
                st.write(result["prompt"])


def render_gallery():
    items = st.session_state.gallery
    if not items:
        return
    head, clear = st.columns([4, 1.2], vertical_alignment="bottom")
    head.markdown("<div class='section-title' style='margin-bottom:0.2rem'>Recent creations</div>", unsafe_allow_html=True)
    clear.button("Clear list", on_click=clear_gallery, use_container_width=True, key="gal_clear")
    st.markdown(f"<div class='note' style='margin:0 0 0.8rem 0'>Files are kept in <code>{html.escape(GEN_ABS)}</code></div>", unsafe_allow_html=True)
    for row in range(0, len(items), 3):
        cols = st.columns(3)
        for col, (idx, item) in zip(cols, list(enumerate(items))[row : row + 3]):
            with col.container(border=True):
                st.image(item["png"])
                st.markdown(
                    f"<div class='doc-name' title='{html.escape(item['title'])}'>{html.escape(item['title'])}</div>"
                    f"<div class='doc-meta'>{'Graph' if item['kind'] == 'graph' else 'Image'} · {html.escape(item['source'])}</div>",
                    unsafe_allow_html=True,
                )
                st.download_button("⬇️ Download", item["png"], file_name=os.path.basename(item["path"]), mime="image/png",
                                   use_container_width=True, key=f"gal_dl_{idx}")


def render_studio():
    st.markdown(
        "<div class='page-title'>Studio</div>"
        "<div class='page-sub'>Turn your reports, chat answers or ideas into graphs and images. "
        "Everything is created on your computer and saved in your project.</div>",
        unsafe_allow_html=True,
    )
    st.radio("Studio tool", [TOOL_GRAPHS, TOOL_IMAGES], key="studio_tool", horizontal=True, label_visibility="collapsed")
    st.markdown("<div style='height:0.9rem'></div>", unsafe_allow_html=True)
    if st.session_state.studio_tool == TOOL_GRAPHS:
        render_graph_tool()
    else:
        render_image_tool()
    render_gallery()


if st.session_state.page == PAGE_LIB:
    render_library()
elif st.session_state.page == PAGE_STUDIO:
    render_studio()
else:
    render_chat()