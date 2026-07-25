"""
Athena — chat with your PDFs.

Everything lives in memory for the duration of the browser session. No files are
written to disk, no database, no login. Close the tab and it is gone.

Run with:
    pip install -r requirements.txt
    streamlit run app.py
"""

from __future__ import annotations

import os
import re
import time
import traceback

import numpy as np
import streamlit as st
from google import genai
from google.genai import types
from pypdf import PdfReader

# --- Configuration ---------------------------------------------------------

CHAT_MODEL = "gemini-flash-latest"   # alias -> current GA Flash. Pin e.g.
                                     # "gemini-3.6-flash" for reproducibility,
                                     # or "gemini-2.5-flash-lite" if you hit
                                     # free-tier quota limits.
EMBED_MODEL = "gemini-embedding-001"
EMBED_DIM = 768        # gemini-embedding-001 defaults to 3072; 768 is plenty
                       # here and keeps the in-memory matrix small.
CHUNK_CHARS = 1800
CHUNK_OVERLAP = 400
TOP_K = 6
EMBED_BATCH = 16
MAX_RETRIES = 5
HISTORY_TURNS = 4      # prior exchanges fed back into the prompt

SYSTEM_PROMPT = """You are a careful research assistant answering questions about \
a set of PDFs the user has uploaded.

Rules:
- Answer only from the excerpts provided. They are the entire source of truth.
- If the excerpts do not contain the answer, say so plainly. Do not guess or fall \
back on general knowledge.
- Cite the source of each claim inline as [filename, p.N].
- Quote sparingly and briefly; prefer your own words.
- If the question is ambiguous given the documents, say what is ambiguous."""

st.set_page_config(page_title="Athena", page_icon="📚", layout="wide")


# --- Text extraction and chunking ------------------------------------------

def read_pdf(uploaded) -> list[tuple[int, str]]:
    """Return [(page_number, text)] for one uploaded PDF, skipping blank pages."""
    reader = PdfReader(uploaded)
    pages = []
    for page_no, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            pages.append((page_no, text))
    return pages


def _paragraphs(text: str) -> list[str]:
    return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]


def _hard_split(text: str, limit: int) -> list[str]:
    """Break a single oversized paragraph on sentence boundaries."""
    if len(text) <= limit:
        return [text]
    sentences = re.split(r"(?<=[.!?])\s+", text)
    out, current = [], ""
    for sentence in sentences:
        while len(sentence) > limit:          # pathological run-on / table dump
            out.append(sentence[:limit])
            sentence = sentence[limit:]
        if len(current) + len(sentence) + 1 > limit and current:
            out.append(current)
            current = sentence
        else:
            current = f"{current} {sentence}".strip()
    if current:
        out.append(current)
    return out


def build_chunks(docs: dict[str, list[tuple[int, str]]]) -> list[dict]:
    """Group paragraphs into overlapping chunks that remember their page range."""
    chunks: list[dict] = []

    for file_name, pages in docs.items():
        units: list[tuple[int, str]] = []
        for page_no, text in pages:
            for para in _paragraphs(text):
                for piece in _hard_split(para, CHUNK_CHARS):
                    units.append((page_no, piece))

        buffer: list[tuple[int, str]] = []
        length = 0
        fresh = 0          # units added since the last flush

        def flush() -> list[tuple[int, str]]:
            """Emit the buffered units as a chunk, return the overlap tail."""
            if not buffer:
                return []
            chunks.append({
                "text": "\n\n".join(t for _, t in buffer),
                "file": file_name,
                "first_page": buffer[0][0],
                "last_page": buffer[-1][0],
            })
            # Carry trailing units back as overlap. A typical paragraph is
            # bigger than the overlap budget, so always allow the first one
            # through — otherwise overlap never happens on real prose. Cap it
            # at half a chunk so an oversized unit is not duplicated whole.
            tail, tail_len = [], 0
            for unit in reversed(buffer):
                size = len(unit[1])
                if not tail:
                    if size > CHUNK_CHARS // 2:
                        break
                elif tail_len + size > CHUNK_OVERLAP:
                    break
                tail.insert(0, unit)
                tail_len += size
            return tail

        for page_no, piece in units:
            # `fresh` guards against flushing a buffer holding only the carried
            # overlap, which would emit a chunk duplicating the previous tail.
            if length + len(piece) > CHUNK_CHARS and fresh:
                buffer = flush()
                length = sum(len(t) for _, t in buffer)
                fresh = 0
            buffer.append((page_no, piece))
            length += len(piece)
            fresh += 1

        if fresh:
            flush()

    return chunks


# --- Embeddings and retrieval ----------------------------------------------

TRANSIENT = ("429", "resource_exhausted", "rate limit", "quota",
             "503", "unavailable", "500", "internal", "deadline")


def _embed_request(client: genai.Client, contents, task_type: str) -> list[list[float]]:
    """One embed call, retrying transient failures with exponential backoff."""
    config = types.EmbedContentConfig(
        task_type=task_type,
        output_dimensionality=EMBED_DIM,
    )
    last_error: Exception | None = None

    for attempt in range(MAX_RETRIES):
        try:
            response = client.models.embed_content(
                model=EMBED_MODEL, contents=contents, config=config
            )
            return [e.values for e in response.embeddings]
        except Exception as exc:
            last_error = exc
            if not any(flag in str(exc).lower() for flag in TRANSIENT):
                raise
            if attempt < MAX_RETRIES - 1:
                time.sleep(min(2 ** attempt, 30))

    raise last_error  # type: ignore[misc]


def embed(client: genai.Client, texts: list[str], task_type: str,
          progress=None) -> np.ndarray:
    """Embed texts and L2-normalise, so cosine similarity is a plain dot product."""
    vectors: list[list[float]] = []
    batch_size = EMBED_BATCH
    index = 0

    while index < len(texts):
        group = texts[index:index + batch_size]
        try:
            vectors.extend(_embed_request(client, group, task_type))
        except Exception:
            if batch_size > 1:
                # This endpoint rejects multi-input requests. Drop to one per
                # call permanently rather than retrying the batch every time.
                batch_size = 1
                continue
            raise
        index += len(group)
        if progress:
            progress(min(index / len(texts), 1.0))

    matrix = np.asarray(vectors, dtype=np.float32)
    # Truncated MRL embeddings are not unit-length, so normalise explicitly.
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.clip(norms, 1e-8, None)


def retrieve(query_vector: np.ndarray, matrix: np.ndarray, k: int) -> list[int]:
    scores = matrix @ query_vector
    k = min(k, len(scores))
    top = np.argpartition(-scores, k - 1)[:k]
    return top[np.argsort(-scores[top])].tolist()


def format_context(chunks: list[dict], indices: list[int]) -> str:
    blocks = []
    for i in indices:
        chunk = chunks[i]
        pages = (
            f"p.{chunk['first_page']}"
            if chunk["first_page"] == chunk["last_page"]
            else f"pp.{chunk['first_page']}-{chunk['last_page']}"
        )
        blocks.append(f"[{chunk['file']}, {pages}]\n{chunk['text']}")
    return "\n\n---\n\n".join(blocks)


# --- Session state ---------------------------------------------------------

def init_state() -> None:
    st.session_state.setdefault("chunks", [])
    st.session_state.setdefault("matrix", None)
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("doc_signature", None)
    st.session_state.setdefault("sources", {})


def clear_documents() -> None:
    st.session_state.chunks = []
    st.session_state.matrix = None
    st.session_state.messages = []
    st.session_state.doc_signature = None
    st.session_state.sources = {}


init_state()


# --- Sidebar ---------------------------------------------------------------

with st.sidebar:
    st.title("📚 Athena")
    st.caption("Ask questions about your PDFs. Nothing is saved.")

    api_key = st.text_input(
        "Gemini API key",
        type="password",
        value=os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or "",
        help="Kept in memory for this session only. Get one at aistudio.google.com/apikey",
    )

    uploads = st.file_uploader(
        "PDFs", type="pdf", accept_multiple_files=True, label_visibility="collapsed"
    )

    if st.session_state.chunks:
        st.divider()
        st.caption(
            f"{len(st.session_state.sources)} document(s), "
            f"{len(st.session_state.chunks)} chunks indexed"
        )
        for name, page_count in st.session_state.sources.items():
            st.caption(f"• {name} — {page_count} pages with text")
        if st.button("Clear everything", use_container_width=True):
            clear_documents()
            st.rerun()

    st.divider()
    st.caption(
        "Documents live in memory only. Closing this tab discards them. "
        "Text is sent to Google's Gemini API to answer questions."
    )


# --- Indexing --------------------------------------------------------------

if not api_key:
    st.info("Enter your Gemini API key in the sidebar to begin.")
    st.stop()

client = genai.Client(api_key=api_key)

if not uploads:
    if st.session_state.chunks:
        clear_documents()
    st.info("Upload one or more PDFs in the sidebar.")
    st.stop()

signature = tuple(sorted((f.name, f.size) for f in uploads))

if signature != st.session_state.doc_signature:
    with st.status("Indexing your PDFs…", expanded=True) as status:
        st.write("Extracting text…")
        docs, sources = {}, {}
        for uploaded in uploads:
            pages = read_pdf(uploaded)
            if pages:
                docs[uploaded.name] = pages
                sources[uploaded.name] = len(pages)

        if not docs:
            status.update(label="No text found", state="error")
            st.warning(
                "These look like scanned images with no text layer. "
                "Run them through OCR (e.g. `ocrmypdf`) and upload again."
            )
            st.stop()

        st.write("Splitting into chunks…")
        chunks = build_chunks(docs)

        st.write(f"Embedding {len(chunks)} chunks…")
        bar = st.progress(0.0)
        try:
            matrix = embed(
                client, [c["text"] for c in chunks], "RETRIEVAL_DOCUMENT",
                progress=bar.progress,
            )
        except Exception as exc:
            status.update(label="Embedding failed", state="error")
            st.error(f"Could not embed the documents:\n\n{type(exc).__name__}: {exc}")
            with st.expander("Full traceback"):
                st.code(traceback.format_exc())
            st.stop()

        st.session_state.chunks = chunks
        st.session_state.matrix = matrix
        st.session_state.sources = sources
        st.session_state.doc_signature = signature
        st.session_state.messages = []
        status.update(label=f"Ready — {len(chunks)} chunks indexed", state="complete")

    st.rerun()


# --- Chat ------------------------------------------------------------------

st.title("Athena")
st.caption("Answers are grounded in your uploaded documents.")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("context"):
            with st.expander("Sources used"):
                st.text(message["context"])

question = st.chat_input("Ask a question about your PDFs…")

if question:
    with st.chat_message("user"):
        st.markdown(question)
    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("assistant"):
        try:
            # Fold in the previous question so follow-ups like "and the second one?"
            # still retrieve something sensible.
            previous = [m for m in st.session_state.messages[:-1] if m["role"] == "user"]
            search_text = f"{previous[-1]['content']}\n{question}" if previous else question

            query_vector = embed(client, [search_text], "RETRIEVAL_QUERY")[0]
            indices = retrieve(query_vector, st.session_state.matrix, TOP_K)
            context = format_context(st.session_state.chunks, indices)

            history = ""
            recent = st.session_state.messages[:-1][-HISTORY_TURNS * 2:]
            if recent:
                history = "Earlier in this conversation:\n" + "\n".join(
                    f"{m['role'].capitalize()}: {m['content']}" for m in recent
                ) + "\n\n"

            prompt = (
                f"{history}"
                f"Excerpts from the user's documents:\n\n{context}\n\n"
                f"---\n\nQuestion: {question}"
            )

            stream = client.models.generate_content_stream(
                model=CHAT_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    temperature=0.2,
                ),
            )
            answer = st.write_stream(chunk.text or "" for chunk in stream)

            with st.expander("Sources used"):
                st.text(context)

            st.session_state.messages.append(
                {"role": "assistant", "content": answer, "context": context}
            )

        except Exception as exc:
            st.error(f"Something went wrong: {exc}")
            st.session_state.messages.pop()