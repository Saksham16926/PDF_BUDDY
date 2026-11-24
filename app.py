import hashlib
import os
import re
import shutil
import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_unstructured import UnstructuredLoader

load_dotenv()

# One sub-folder per uploaded PDF (named by the file's hash), so documents never mix.
DB_ROOT = Path("chroma_store")
DB_ROOT.mkdir(exist_ok=True)

EMBED_MODEL = "nvidia/nemotron-3.5-embed-1b"
CHAT_MODEL = "nvidia/nemotron-3.5-bolt-e4b"

PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a knowledgeable assistant helping the user understand a document they uploaded.
You are given excerpts from that document along with the user's question.

How to answer:
1. Start from the excerpts. Combine information from several excerpts, reason over it,
   and give complete answers. For "how many" or "list all" questions, count and list every
   item the excerpts support.
2. If the excerpts only partly answer the question, give the part they support and say
   what seems to be missing.
3. If the document does not cover the question but you know the answer from general
   knowledge, answer it and say clearly that this comes from general knowledge and not
   from the document.
4. Never say you cannot find an answer when the excerpts contain relevant information.
5. Be clear and concise. Use short lists when they help. Do not mention "excerpts" or
   "context" unless needed, and do not list sources unless asked.""",
        ),
        MessagesPlaceholder("history"),
        (
            "human",
            """Document excerpts:
{context}

Question:
{question}""",
        ),
    ]
)

SOURCE_REQUEST = re.compile(
    r"\b(sources?|references?|citations?|which page|what page|"
    r"where (did|do) you (get|find|read)|show (me )?(the )?(passages|excerpts))\b",
    re.IGNORECASE,
)

# ---------- page setup & styling ----------

st.set_page_config(page_title="Ask your PDF", page_icon="📄", layout="centered")

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:wght@500;700&family=DM+Sans:wght@400;500&display=swap');

html, body, [class*="css"], .stMarkdown, .stChatMessage {
    font-family: 'DM Sans', sans-serif;
}
h1, h2, h3 {
    font-family: 'Bricolage Grotesque', sans-serif !important;
    letter-spacing: -0.02em;
}
.block-container { padding-top: 2.5rem; max-width: 760px; }

/* sidebar */
section[data-testid="stSidebar"] { border-right: 1px solid #DCE3D6; }
section[data-testid="stSidebar"] h2 { font-size: 1.15rem; }

/* document status chip */
.doc-chip {
    border-left: 4px solid #2F6B3F;
    background: #EAF0E5;
    padding: 0.6rem 0.9rem;
    border-radius: 4px;
    font-size: 0.92rem;
    margin: 0.5rem 0 1rem 0;
}
.doc-chip b { display: block; font-weight: 500; word-break: break-word; }
.doc-chip span { color: #4B5A4F; }

/* chat bubbles */
.stChatMessage { background: transparent; padding: 0.4rem 0; }
[data-testid="stChatMessageAvatarUser"] { background: #2F6B3F; }
[data-testid="stChatMessageAvatarAssistant"] { background: #A4262C; }

/* source cards */
.src { font-size: 0.85rem; color: #34423A; border-left: 3px solid #DCE3D6;
       padding-left: 0.7rem; margin: 0.6rem 0; white-space: pre-wrap; }
.src small { color: #6B7A6F; display: block; margin-bottom: 0.15rem; }

#MainMenu, footer { visibility: hidden; }
</style>
""",
    unsafe_allow_html=True,
)

# ---------- cached models ----------


@st.cache_resource(show_spinner=False)
def get_embeddings():
    return NVIDIAEmbeddings(model=EMBED_MODEL)


@st.cache_resource(show_spinner=False)
def get_llm():
    return ChatNVIDIA(model=CHAT_MODEL)


# ---------- indexing ----------


def file_hash(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()[:16]


def build_or_load_store(data: bytes, filename: str, status) -> tuple[Chroma, int]:
    """Create a Chroma store for this PDF, or reuse it if it was indexed before."""
    h = file_hash(data)
    persist_dir = DB_ROOT / h
    embeddings = get_embeddings()

    if persist_dir.exists() and any(persist_dir.iterdir()):
        status.update(label="Loading saved index…")
        store = Chroma(persist_directory=str(persist_dir), embedding_function=embeddings)
        return store, store._collection.count()

    status.update(label="Reading PDF…")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(data)
        tmp_path = tmp.name
    try:
        docs = UnstructuredLoader(tmp_path).load()
    finally:
        os.remove(tmp_path)

    for d in docs:
        d.metadata["source"] = filename  # show the real name, not the temp path

    status.update(label="Splitting into chunks…")
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = filter_complex_metadata(splitter.split_documents(docs))

    if not chunks:
        raise ValueError("No readable text found in this PDF (it may be a scanned image).")

    status.update(label=f"Embedding {len(chunks)} chunks…")
    store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(persist_dir),
    )
    return store, len(chunks)


def make_search_queries(question: str, history: list[dict]) -> list[str]:
    """Ask the LLM for a few alternative search queries so retrieval covers more ground."""
    convo = "\n".join(f"{m['role']}: {m['content'][:300]}" for m in history[-4:])
    prompt = (
        "Write 3 short search queries that would find passages in a document that help "
        "answer the question below. Use different wording and related terms. "
        "If the question is a follow-up, use the conversation to make each query "
        "self-contained. Output only the queries, one per line, no numbering.\n\n"
        f"Conversation:\n{convo or '(none)'}\n\nQuestion: {question}"
    )
    try:
        out = get_llm().invoke(prompt).content
    except Exception:
        return [question]
    extra = []
    for line in out.splitlines():
        line = re.sub(r"^\s*(\d+[.)]|[-•*])\s*", "", line).strip()
        if line:
            extra.append(line)
    return [question] + extra[:3]


def retrieve(store: Chroma, queries: list[str], k: int):
    """Run MMR search for every query, merge and de-duplicate the results."""
    retriever = store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": k, "fetch_k": max(15, k * 3), "lambda_mult": 0.5},
    )
    seen, merged = set(), []
    for q in queries:
        for d in retriever.invoke(q):
            if d.page_content not in seen:
                seen.add(d.page_content)
                merged.append(d)
    return merged[: k * 2]


def to_sources(docs):
    return [
        {
            "page": d.metadata.get("page_number"),
            "text": d.page_content.strip()[:400] + ("…" if len(d.page_content) > 400 else ""),
        }
        for d in docs
    ]


def reset_document():
    h = st.session_state.get("doc_hash")
    if h:
        shutil.rmtree(DB_ROOT / h, ignore_errors=True)
    for key in ("store", "doc_name", "doc_hash", "chunk_count", "messages"):
        st.session_state.pop(key, None)


# ---------- session state ----------

st.session_state.setdefault("messages", [])

# ---------- sidebar ----------

with st.sidebar:
    st.header("Your document")
    uploaded = st.file_uploader("Upload a PDF", type=["pdf"], label_visibility="collapsed")

    if uploaded is not None:
        data = uploaded.getvalue()
        new_hash = file_hash(data)

        if st.session_state.get("doc_hash") != new_hash:
            # A different file than the one currently loaded: index it.
            with st.status("Preparing document…", expanded=False) as status:
                try:
                    store, n = build_or_load_store(data, uploaded.name, status)
                except Exception as e:
                    status.update(label="Could not process this PDF", state="error")
                    st.error(str(e))
                else:
                    st.session_state.update(
                        store=store,
                        doc_name=uploaded.name,
                        doc_hash=new_hash,
                        chunk_count=n,
                        messages=[],  # new document, fresh conversation
                    )
                    status.update(label="Ready", state="complete")

    if st.session_state.get("store"):
        st.markdown(
            f"""<div class="doc-chip"><b>{st.session_state.doc_name}</b>
            <span>{st.session_state.chunk_count} passages indexed</span></div>""",
            unsafe_allow_html=True,
        )

        st.subheader("Settings")
        k = st.slider(
            "Passages per search",
            3, 12, 6,
            help="Higher values help with questions that span many parts of the document.",
        )
        always_sources = st.toggle(
            "Always show sources",
            value=False,
            help="Off by default. You can also just ask, e.g. \"show sources\".",
        )

        c1, c2 = st.columns(2)
        if c1.button("Clear chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
        if c2.button("Remove PDF", use_container_width=True):
            reset_document()
            st.rerun()
    else:
        k = 6
        always_sources = False

# ---------- main area ----------

st.title("Ask your PDF")

store = st.session_state.get("store")

if not store:
    st.markdown(
        "Upload a PDF in the sidebar. Once it's indexed, ask questions in plain "
        "language and answers come only from that document."
    )
    st.stop()

st.caption(f"Answering from **{st.session_state.doc_name}**")


def render_sources(sources):
    with st.expander(f"Sources ({len(sources)})"):
        for s in sources:
            page = s["page"]
            label = f"Page {page}" if page else "Passage"
            st.markdown(
                f'<div class="src"><small>{label}</small>{s["text"]}</div>',
                unsafe_allow_html=True,
            )


# replay history
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("sources") and (always_sources or m.get("show_sources")):
            render_sources(m["sources"])

# new question
if query := st.chat_input("Ask something about the document"):
    history = list(st.session_state.messages)  # everything before this question
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    last_with_sources = next(
        (m for m in reversed(history) if m["role"] == "assistant" and m.get("sources")),
        None,
    )

    with st.chat_message("assistant"):
        # The user is asking to see sources for the previous answer: no new search needed.
        if last_with_sources and len(query.split()) <= 12 and SOURCE_REQUEST.search(query):
            text = "Here are the passages behind my last answer."
            st.markdown(text)
            render_sources(last_with_sources["sources"])
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": text,
                    "sources": last_with_sources["sources"],
                    "show_sources": True,
                }
            )
        else:
            try:
                with st.spinner("Searching the document…"):
                    queries = make_search_queries(query, history)
                    docs = retrieve(store, queries, k)

                context = "\n\n".join(d.page_content for d in docs)
                chat_history = [
                    HumanMessage(m["content"]) if m["role"] == "user" else AIMessage(m["content"])
                    for m in history[-6:]
                ]
                final_prompt = PROMPT.invoke(
                    {"context": context, "question": query, "history": chat_history}
                )

                def token_stream():
                    for chunk in get_llm().stream(final_prompt):
                        yield chunk.content

                answer = st.write_stream(token_stream())
                sources = to_sources(docs)
                if always_sources:
                    render_sources(sources)
                st.session_state.messages.append(
                    {"role": "assistant", "content": answer, "sources": sources}
                )
            except Exception as e:
                st.error(f"Something went wrong while answering: {e}")