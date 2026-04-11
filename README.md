# 🧠 Local Second Brain (Powered by Gemma)

A fully offline AI system that understands, connects, and reasons over your personal knowledge base — including notes, PDFs, chats, and code.

This project uses **Gemma (local LLM)** + **vector search (RAG)** to build a system that doesn’t just retrieve information, but *thinks with you*.

---

## 🚀 Features

### 🧠 Knowledge Understanding

* Ingests:

  * Markdown notes
  * PDFs
  * Code files
  * Chat logs
* Splits content into semantic chunks
* Stores embeddings in a vector database

### 🔍 Smart Retrieval (RAG)

* Context-aware search over your knowledge
* Combines multiple sources dynamically
* Returns grounded, relevant answers

### 💡 Idea Generation Engine

* Finds hidden connections between unrelated notes
* Suggests project ideas based on your past work
* Surfaces “unfinished thoughts”

### 🔁 Reflection Loop (Daily Intelligence)

* Summarizes your daily activity
* Extracts insights
* Feeds them back into long-term memory

### ❓ Curiosity Engine

* Asks *you* meaningful questions:

  * “You’ve explored EEG twice. Is this a core interest?”
  * “Why did you abandon this project?”

---

## 🏗️ Architecture Overview

```
                ┌────────────────────┐
                │   Local Files       │
                │ (PDFs, Notes, Code)│
                └─────────┬──────────┘
                          ↓
                ┌────────────────────┐
                │   Data Ingestion    │
                │  (Chunk + Clean)    │
                └─────────┬──────────┘
                          ↓
                ┌────────────────────┐
                │   Embeddings        │
                │ (SentenceTransform) │
                └─────────┬──────────┘
                          ↓
                ┌────────────────────┐
                │   Vector DB (FAISS)│
                └─────────┬──────────┘
                          ↓
                ┌────────────────────┐
                │   Retriever         │
                └─────────┬──────────┘
                          ↓
                ┌────────────────────┐
                │   Gemma (Local LLM)│
                └─────────┬──────────┘
                          ↓
                ┌────────────────────┐
                │   Final Response    │
                └────────────────────┘
```

---

## 🧰 Tech Stack

| Component     | Tool                              |
| ------------- | --------------------------------- |
| LLM           | Gemma (2B / 7B, quantized)        |
| Embeddings    | Sentence Transformers             |
| Vector DB     | FAISS / Chroma                    |
| Backend       | Python                            |
| Orchestration | LangChain / LlamaIndex (optional) |
| Interface     | CLI / Web (Streamlit or FastAPI)  |

---

## 📦 Installation

### 1. Clone the repo

```bash
git clone https://github.com/yourusername/local-second-brain.git
cd local-second-brain
```

### 2. Create virtual environment

```bash
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 📁 Project Structure

```
local-second-brain/
│
├── data/                  # Raw user data
├── embeddings/           # Stored vector index
├── models/               # Gemma model files
│
├── src/
│   ├── ingestion.py      # Load + chunk data
│   ├── embedder.py       # Generate embeddings
│   ├── vector_store.py   # FAISS integration
│   ├── retriever.py      # Query logic
│   ├── llm.py            # Gemma interface
│   ├── agent.py          # Main reasoning pipeline
│   └── reflection.py     # Daily summary engine
│
├── app.py                # CLI / Web interface
├── config.py             # Settings
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup Instructions

### 🔹 Step 1: Add your data

Place files in:

```
/data/
```

Supported:

* `.txt`, `.md`
* `.pdf`
* `.py`, `.java`, etc.

---

### 🔹 Step 2: Run ingestion pipeline

```bash
python src/ingestion.py
```

This will:

* Load files
* Chunk text
* Clean formatting

---

### 🔹 Step 3: Generate embeddings

```bash
python src/embedder.py
```

---

### 🔹 Step 4: Build vector index

```bash
python src/vector_store.py
```

---

### 🔹 Step 5: Start the system

```bash
python app.py
```

---

## 🧠 Example Queries

```
> What ideas can I build using my past projects?

> Connect my OpenCV work with EEG data

> What topics do I revisit the most?

> Summarize my learning patterns
```

---

## 🔥 Advanced Features (Optional)

### 1. 🧬 Personal Fine-Tuning (LoRA)

* Train on your writing style
* Improves tone + personalization

### 2. 🧠 Multi-Layer Memory

* Short-term: recent chats
* Long-term: vector DB
* Insights: AI-generated summaries

### 3. ⏳ Recency Weighting

* Boost recent notes in retrieval

### 4. 🔁 Reflection Scheduler

Run daily:

```bash
python src/reflection.py
```

---

## ⚡ Performance Tips

* Use quantized models (GGUF / 4-bit)
* Keep chunk size: **200–500 tokens**
* Use top-k retrieval (k=3–5)
* Cache embeddings

---

## 🚧 Limitations

* Limited reasoning vs larger cloud models
* Context window constraints
* Requires good chunking + prompts

---

## 🛣️ Future Improvements

* Voice interface
* Browser plugin integration
* Multi-agent reasoning
* Visual knowledge graph
* Real-time note ingestion

---

## 🎯 Why This Project Matters

This is not just a chatbot.

It’s:

* A thinking system
* A memory extension
* A creativity engine

You are essentially building a **local AI that evolves with you**.

---

## 🤝 Contributing

PRs welcome! Ideas:

* Better retrieval algorithms
* UI improvements
* New memory layers

---

## 📜 License

MIT License

---

## 👨‍💻 Author

Built by someone who refuses to let their ideas die in scattered notes.

---
