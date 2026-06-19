"""
BioRAG — Streamlit UI
Query interface only. Ingestion is a backend batch operation.
"""

import httpx
import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="BioRAG",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Session state ─────────────────────────────────────────────────────────────

if "query_history" not in st.session_state:
    st.session_state.query_history = []

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("🧬 BioRAG")
    st.caption("Biomedical literature query interface")
    st.divider()

    api_url = st.text_input("API base URL", value="http://100.108.66.35:8000")

    try:
        r = httpx.get(f"{api_url}/health", timeout=3.0)
        st.success("API online", icon="✅") if r.status_code == 200 else st.error(f"HTTP {r.status_code}", icon="⚠️")
    except Exception:
        st.error("API unreachable", icon="🔴")

    st.divider()
    namespace = st.text_input(
        "Namespace",
        value="default",
        help="Restricts retrieval to documents in this namespace. Maps to a study, project, or access group.",
    )
    top_k = st.slider("Chunks to retrieve (k)", min_value=1, max_value=10, value=5)
    st.divider()
    st.caption("Phase 1 — semantic retrieval only. Hybrid search + reranker coming in Phase 2.")

# ── Main area ─────────────────────────────────────────────────────────────────

st.header("Ask a question")
st.caption(
    "Retrieves the most relevant passages from ingested literature using PubMedBERT embeddings, "
    "then generates a cited answer via Llama-3.3-70B (NVIDIA NIM)."
)

question = st.text_input(
    "Question",
    placeholder="What is the mechanism of action of daratumumab?",
    label_visibility="collapsed",
)

if st.button("Ask", type="primary", disabled=not question.strip()):
    with st.spinner("Retrieving chunks and generating answer…"):
        try:
            response = httpx.post(
                f"{api_url}/query",
                data={"question": question, "namespace": namespace, "top_k": top_k},
                timeout=60.0,
            )
            if response.status_code == 200:
                result = response.json()
                st.session_state.query_history.insert(
                    0,
                    {
                        "question": question,
                        "namespace": namespace,
                        "answer": result["answer"],
                        "sources": result["sources"],
                        "chunks_retrieved": result["chunks_retrieved"],
                    },
                )
            else:
                st.error(f"Query failed: HTTP {response.status_code}\n\n```\n{response.text}\n```")
        except httpx.TimeoutException:
            st.error("LLM call timed out. Check NIM API status or try a shorter question.")
        except Exception as e:
            st.error(f"Unexpected error: {e}")

# ── Render history ────────────────────────────────────────────────────────────

for entry in st.session_state.query_history[:5]:
    with st.container(border=True):
        st.markdown(f"**Q:** {entry['question']}")
        st.caption(f"namespace: `{entry['namespace']}` · {entry['chunks_retrieved']} chunks retrieved")
        st.markdown(entry["answer"])

        sources = entry["sources"]
        if sources:
            st.divider()
            st.caption(f"**Retrieved sources** ({len(sources)} chunks)")
            for i, src in enumerate(sources, start=1):
                label = (
                    f"[{i}] {src['filename']}"
                    + (f" — p.{src['page_number']}" if src.get("page_number") is not None else "")
                    + f"  (similarity: {src['score']:.4f})"
                )
                with st.expander(label):
                    st.markdown(src["text_preview"])
                    st.caption("Preview truncated to 200 chars. Full text passed to LLM.")
        else:
            st.caption("No sources returned.")