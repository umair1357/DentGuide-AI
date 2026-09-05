import os
import hashlib
from typing import List, Dict, Any

import numpy as np
import streamlit as st
import fitz  # PyMuPDF
from google import genai
from google.genai import types


# -----------------------------
# Configuration
# -----------------------------
APP_TITLE = "GuidelineGPT — Evidence-Based Clinical Q&A"
CHAT_MODEL = os.getenv("GEMINI_CHAT_MODEL", "gemini-2.5-flash")
EMBED_MODEL = os.getenv("GEMINI_EMBED_MODEL", "text-embedding-004")

# Approximate character-based chunking. Keeping chunks page-aware makes
# source/page citations much easier to verify.
CHUNK_SIZE = 1800
CHUNK_OVERLAP = 250
TOP_K = 8
EMBED_BATCH_SIZE = 64


# -----------------------------
# Page configuration / styling
# -----------------------------
st.set_page_config(
    page_title=APP_TITLE,
    page_icon="📚",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container {max-width: 1200px; padding-top: 2rem;}
    .source-card {
        border: 1px solid #d9dee7;
        border-radius: 10px;
        padding: 12px 14px;
        margin: 8px 0;
        background: #fafbfc;
    }
    .small-muted {color: #667085; font-size: 0.9rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------
# Secrets / client
# -----------------------------
def get_api_key() -> str:
    """Read the Gemini key from Streamlit secrets or environment."""
    try:
        key = st.secrets.get("GEMINI_API_KEY", "") or st.secrets.get("GOOGLE_API_KEY", "")
    except Exception:
        key = ""

    return key or os.getenv("GEMINI_API_KEY", "") or os.getenv("GOOGLE_API_KEY", "")


@st.cache_resource(show_spinner=False)
def get_gemini_client(api_key: str) -> genai.Client:
    return genai.Client(api_key=api_key)


# -----------------------------
# PDF processing
# -----------------------------
def file_hash(uploaded_file) -> str:
    return hashlib.sha256(uploaded_file.getvalue()).hexdigest()


def extract_pdf_pages(uploaded_file) -> List[Dict[str, Any]]:
    """
    Extract text page-by-page. Each returned item keeps the original PDF
    page number so answers can cite exact source pages.
    """
    pdf_bytes = uploaded_file.getvalue()
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")

    pages = []
    try:
        for page_index in range(len(doc)):
            page = doc.load_page(page_index)
            text = page.get_text("text").strip()
            if text:
                pages.append(
                    {
                        "document": uploaded_file.name,
                        "page": page_index + 1,
                        "text": text,
                    }
                )
    finally:
        doc.close()

    return pages


def split_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    """Character chunking with overlap; avoids splitting at every page."""
    text = " ".join(text.split())
    if not text:
        return []

    chunks = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = min(start + size, text_len)

        # Prefer a natural boundary near the end of the chunk.
        if end < text_len:
            candidates = [
                text.rfind(". ", start + int(size * 0.55), end),
                text.rfind("; ", start + int(size * 0.55), end),
                text.rfind(" ", start + int(size * 0.55), end),
            ]
            boundary = max(candidates)
            if boundary > start:
                end = boundary + 1

        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)

        if end >= text_len:
            break

        next_start = max(end - overlap, start + 1)
        start = next_start

    return chunks


def build_chunks(pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    chunks = []
    for page in pages:
        for i, chunk in enumerate(split_text(page["text"])):
            chunks.append(
                {
                    "document": page["document"],
                    "page": page["page"],
                    "chunk": i + 1,
                    "text": chunk,
                }
            )
    return chunks


# -----------------------------
# Embeddings / retrieval
# -----------------------------
def embed_texts(client: genai.Client, texts: List[str]) -> np.ndarray:
    vectors = []

    for start in range(0, len(texts), EMBED_BATCH_SIZE):
        batch = texts[start : start + EMBED_BATCH_SIZE]
        response = client.models.embed_content(
            model=EMBED_MODEL,
            contents=batch,
        )
        for emb in response.embeddings:
            vectors.append(emb.values)

    matrix = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    matrix = matrix / np.clip(norms, 1e-12, None)
    return matrix


def retrieve(
    client: genai.Client,
    query: str,
    chunks: List[Dict[str, Any]],
    matrix: np.ndarray,
    top_k: int = TOP_K,
):
    q_response = client.models.embed_content(
        model=EMBED_MODEL,
        contents=query,
    )
    q = np.asarray(q_response.embeddings[0].values, dtype=np.float32)
    q /= max(float(np.linalg.norm(q)), 1e-12)

    scores = matrix @ q
    indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in indices:
        item = dict(chunks[int(idx)])
        item["score"] = float(scores[int(idx)])
        results.append(item)

    return results


# -----------------------------
# Answer generation
# -----------------------------
SYSTEM_PROMPT = """
You are GuidelineGPT, an evidence-grounded clinical guideline assistant.

Your job is to answer questions ONLY from the supplied excerpts from the
uploaded therapeutic and infection-control guideline PDFs.

Rules:
1. Do not use outside medical knowledge, memory, or assumptions.
2. If the supplied excerpts do not contain enough information to answer,
   explicitly say: "I could not find enough information in the uploaded
   guidelines to answer this reliably."
3. Never invent a recommendation, dose, duration, contraindication, or
   citation.
4. Preserve important qualifiers such as "recommended", "consider",
   "avoid", "contraindicated", "first-line", age restrictions, and exceptions.
5. Give a concise, clinically useful answer first.
6. When possible, cite the source immediately after the relevant statement
   using [Document — p. X].
7. If sources disagree, clearly state the disagreement and cite both.
8. Distinguish what the guidelines explicitly state from any inference.
9. This tool is for reference/education and does not replace professional
   clinical judgment or local policy.
""".strip()


def answer_question(client: genai.Client, question: str, results: List[Dict[str, Any]]) -> str:
    context_parts = []

    for i, item in enumerate(results, start=1):
        context_parts.append(
            f"""SOURCE {i}
Document: {item['document']}
PDF page: {item['page']}
Relevance score: {item['score']:.3f}
Text:
{item['text']}
"""
        )

    context = "\n\n".join(context_parts)

    user_prompt = f"""
Question:
{question}

Retrieved excerpts from the uploaded guidelines:
{context}

Answer the question using only these excerpts. Include source/page citations
for the key recommendations. If the excerpts are insufficient, say so rather
than filling gaps from general medical knowledge.
""".strip()

    response = client.models.generate_content(
        model=CHAT_MODEL,
        contents=user_prompt,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.2,
        ),
    )

    return response.text.strip()


# -----------------------------
# Session-state helpers
# -----------------------------
def clear_index():
    for key in ("index_hash", "chunks", "matrix", "documents", "messages"):
        st.session_state.pop(key, None)


# -----------------------------
# UI
# -----------------------------
st.title("📚 GuidelineGPT")
st.caption(
    "Ask questions against your uploaded therapeutic and infection-control "
    "guidelines. Answers are grounded in retrieved PDF passages."
)

with st.sidebar:
    st.header("1. Upload guidelines")
    uploads = st.file_uploader(
        "Upload one or more PDF guideline books",
        type=["pdf"],
        accept_multiple_files=True,
        max_upload_size=200,
        help="For example: therapeutic guidelines + infection-control guidelines.",
    )

    st.divider()
    st.header("2. Model")
    st.write(f"**Answer model:** `{CHAT_MODEL}`")
    st.write(f"**Embedding model:** `{EMBED_MODEL}`")
    st.caption("You can override model names with GEMINI_CHAT_MODEL and GEMINI_EMBED_MODEL.")

    st.divider()
    st.warning(
        "Clinical safety: this app answers from uploaded documents only. "
        "Verify important decisions against the original guideline and "
        "current local policy."
    )

api_key = get_api_key()

if not api_key:
    st.error(
        "Gemini API key not found. Add GEMINI_API_KEY (or GOOGLE_API_KEY) to Streamlit Secrets "
        "or set it as an environment variable."
    )
    st.stop()

client = get_gemini_client(api_key)

if uploads:
    current_hash = hashlib.sha256(
        b"".join(
            f.name.encode("utf-8") + f.getvalue()
            for f in uploads
        )
    ).hexdigest()

    if st.session_state.get("index_hash") != current_hash:
        with st.status("Processing your guideline PDFs…", expanded=True) as status:
            all_pages = []

            for uploaded_file in uploads:
                st.write(f"Reading **{uploaded_file.name}**…")
                try:
                    pages = extract_pdf_pages(uploaded_file)
                    all_pages.extend(pages)
                    st.write(f"✓ Extracted text from {len(pages)} pages")
                except Exception as exc:
                    st.error(f"Could not read {uploaded_file.name}: {exc}")
                    st.stop()

            if not all_pages:
                status.update(label="No searchable text found", state="error")
                st.error(
                    "These PDFs appear to contain no selectable text. "
                    "If they are scanned/image-only PDFs, OCR is required."
                )
                st.stop()

            chunks = build_chunks(all_pages)
            st.write(f"✓ Created {len(chunks):,} searchable text chunks")

            with st.spinner("Creating semantic search index…"):
                matrix = embed_texts(
                    client,
                    [item["text"] for item in chunks],
                )

            st.session_state.index_hash = current_hash
            st.session_state.chunks = chunks
            st.session_state.matrix = matrix
            st.session_state.documents = [f.name for f in uploads]
            st.session_state.messages = []

            status.update(
                label="Guidelines are ready for questions",
                state="complete",
            )

if "chunks" not in st.session_state:
    st.info(
        "Upload your therapeutic and infection-control guideline PDFs in the "
        "sidebar to build the searchable knowledge base."
    )
    st.stop()

st.success(
    f"Knowledge base ready: {len(st.session_state['documents'])} document(s), "
    f"{len(st.session_state['chunks']):,} searchable chunks."
)

# Chat history
for message in st.session_state.get("messages", []):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

question = st.chat_input(
    "Ask a question, e.g. 'What does the guideline recommend for prophylaxis?'"
)

if question:
    st.session_state.messages.append(
        {"role": "user", "content": question}
    )
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching the guidelines and generating an answer…"):
                results = retrieve(
                    client,
                    question,
                    st.session_state["chunks"],
                    st.session_state["matrix"],
                )
                answer = answer_question(client, question, results)

            st.markdown(answer)

            with st.expander("Retrieved sources"):
                for item in results:
                    st.markdown(
                        f"""
                        <div class="source-card">
                        <strong>{item['document']}</strong> — PDF page {item['page']}
                        <br>
                        <span class="small-muted">
                        Retrieval score: {item['score']:.3f}
                        </span>
                        <br><br>
                        {item['text']}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            st.session_state.messages.append(
                {"role": "assistant", "content": answer}
            )

        except Exception as exc:
            st.error(
                "The question could not be processed. "
                "Check your API key, model access, and deployment logs."
            )
            st.exception(exc)

st.divider()
st.caption(
    "GuidelineGPT uses retrieval-augmented generation (RAG): PDF → page-aware "
    "chunks → embeddings → relevant passages → LLM answer."
)
