"""
streamlit_app.py — Web UI for the Local Second Brain.
Run with: streamlit run streamlit_app.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import streamlit as st
st.set_page_config(page_title="Second Brain", page_icon="🧠", layout="wide")

import config

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stSidebar"] { background: #0f0f0f; }
.stTextInput > div > input { background: #1a1a1a; color: #e8e8e0; border: 1px solid #333; }
.source-tag { background: #1a1a1a; border-left: 3px solid #4a9eff;
              padding: 4px 10px; margin: 4px 0; border-radius: 0 4px 4px 0;
              font-size: 12px; color: #888; font-family: monospace; }
.brain-response { background: #111; border: 1px solid #1e1e1e;
                  border-radius: 8px; padding: 16px; margin: 8px 0; }
</style>
""", unsafe_allow_html=True)


# ── state ─────────────────────────────────────────────────────────────────────
def _index_ready():
    return config.FAISS_INDEX_PATH.exists() and config.METADATA_PATH.exists()


@st.cache_resource(show_spinner="Loading brain…")
def get_agent():
    from agent import SecondBrainAgent
    return SecondBrainAgent()


# ── sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🧠 Second Brain")
    st.caption(f"Model: `{config.LLM_MODEL}`")
    st.caption(f"Embeddings: `{config.EMBEDDING_MODEL}`")
    st.divider()

    if st.button("🔄 Re-index knowledge base", use_container_width=True):
        with st.spinner("Running pipeline…"):
            import numpy as np
            from ingestion import ingest, save_chunks
            from embedder import embed_texts
            from vector_store import build_index, save_index
            chunks = ingest(config.DATA_DIR)
            if chunks:
                save_chunks(chunks)
                vecs = embed_texts([c.text for c in chunks])
                np.save(str(config.EMBEDDINGS_DIR / "vectors.npy"), vecs)
                idx = build_index(vecs)
                save_index(idx)
                st.cache_resource.clear()
                st.success(f"Indexed {len(chunks)} chunks")
            else:
                st.warning(f"No files found in {config.DATA_DIR}")

    st.divider()
    mode = st.radio("Mode", ["💬 Chat", "💡 Ideas", "🔗 Connections", "🪞 Reflect", "❓ Questions"])

    st.divider()
    if _index_ready():
        from ingestion import load_chunks
        try:
            chunks = load_chunks()
            st.metric("Chunks indexed", len(chunks))
            sources = sorted({c.source for c in chunks})
            with st.expander(f"{len(sources)} files indexed"):
                for s in sources:
                    st.caption(s)
        except Exception:
            pass
    else:
        st.warning("Index not built yet.\nAdd files to /data and click Re-index.")


# ── main area ─────────────────────────────────────────────────────────────────
if not _index_ready():
    st.info("👋 Welcome! Add your notes/PDFs/code to the `/data` folder, then click **Re-index** in the sidebar.")
    st.stop()

agent = get_agent()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


if mode == "💬 Chat":
    st.header("Chat with your knowledge base")

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask anything about your notes…"):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            from vector_store import search, format_context
            from llm import stream_response

            results = search(prompt)
            context = format_context(results)

            with st.expander("📎 Sources used", expanded=False):
                for chunk, score in results:
                    st.markdown(f'<div class="source-tag">{chunk.source} · chunk {chunk.chunk_index} · {score:.3f}</div>', unsafe_allow_html=True)

            placeholder = st.empty()
            full = []
            for token in stream_response(prompt, context, agent.history[-6:]):
                full.append(token)
                placeholder.markdown("".join(full) + "▌")
            placeholder.markdown("".join(full))
            response = "".join(full)

        st.session_state.chat_history.append({"role": "assistant", "content": response})
        agent.history.append({"role": "user", "content": prompt})
        agent.history.append({"role": "assistant", "content": response})

elif mode == "💡 Ideas":
    st.header("Project Ideas")
    if st.button("Generate ideas from my notes"):
        with st.spinner("Thinking…"):
            st.markdown(agent.generate_ideas())

elif mode == "🔗 Connections":
    st.header("Hidden Connections")
    if st.button("Find connections between my notes"):
        with st.spinner("Connecting dots…"):
            st.markdown(agent.find_connections())

elif mode == "🪞 Reflect":
    st.header("Daily Reflection")

    if st.button("Generate reflection"):
        with st.spinner("Reflecting on your day…"):
            from reflection import run_reflection_streamlit

            result = run_reflection_streamlit()

            if result["status"] == "no_data":
                st.warning("No activity found for today.")
            else:
                st.success("Reflection saved ✅")

                st.subheader("✨ Summary")
                st.markdown(result["summary"])

                with st.expander("🔍 Raw Work Context"):
                    st.text(result["raw_context"])

                st.caption(f"Saved to: {result['file']}")

elif mode == "❓ Questions":
    st.header("Curiosity Engine")
    if st.button("Ask me questions about my notes"):
        with st.spinner("Formulating questions…"):
            st.markdown(agent.ask_curiosity_questions())


