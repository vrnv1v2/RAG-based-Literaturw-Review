
# 🎓 Academic RAG State Graph
An enterprise-grade, stateful, and autonomous multi-engine Retrieval-Augmented Generation (RAG) pipeline orchestrated via **LangGraph** for structured academic literature review.

---

## 🏛️ System Architecture

This system transitions beyond basic, linear RAG pipelines into a state-machine topology. It couples a dual-engine hybrid retrieval stack with an automated query-reformulation and reflection boundary loop to enforce strict factual alignment and eliminate hallucination.



[User Query] ──► [Hybrid Search Node] ──► [Neural Reranker] ──► [Guardrail Router]
(CLI Entry)      • Vector (FAISS)         • MS-Marco MiniLM     • Logit Score Pass?
• Lexical (BM25Okapi)                              │
┌──────────────────────┴──────────────┐
⚠️ FAIL (Score < -1.5)                 ✅ PASS
│                                     │
[Query Reformer Node]                [Generation Node]
• TinyLlama Alignment                • Context-Bounded
• Iteration Loop Ceiling Max=2       • Strict Fact-Only

```

---

## ⚡ Core Engineering Features

* **Stateful Optimization Graph:** Built natively on **LangGraph**. If the retrieved contexts do not meet strict information density thresholds, the graph dynamically reroutes execution to an autonomous query optimization node to reformulate the search space up to 2 times.
* **Dual-Engine Hybrid Retrieval:** Combines semantic dense vector layouts (**FAISS** using Maximum Inner Product L2-normalized embeddings) with deep keyword sparse matching (**BM25Okapi** lexical matrices).
* **Reciprocal Rank Fusion (RRF):** Blends separate dense and sparse candidate pools algorithmically using an interleaved ranking formula ($1 / (c + \text{rank})$) to ensure optimized document recall.
* **Cross-Encoder Neural Reranking:** Implements an `ms-marco-MiniLM-L6-v2` cross-attention transformer layer to calculate exact query-context cross-logits, cleanly pruning irrelevant context windows before generation.
* **Layout-Aware Ingestion Engine:** Implements geometric block-reading rules via `PyMuPDF` to preserve reading boundaries in multi-column academic PDF layouts, eliminating the text-scrambling bugs common in standard PDF parsers.

---

## 📂 Repository Layout

```text
academic-rag-state-graph/
├── .gitignore                  # Prevents VRAM cache and binary database pollution
├── README.md                   # System documentation and execution guide
├── requirements.txt            # Python production package dependencies
├── config.yaml                 # Unified configuration deck for paths and hyperparameters
├── main.py                     # Primary CLI application entry point
├── data/                       # Local volume directory for source academic PDFs
│   ├── Kusner_et_al_P1.pdf
│   ├── Kozodoi_et_Al_P1.pdf
│   └── [Remaining 11 papers...]
└── src/
    └── academic_rag/           # Isolated application package namespace
        ├── __init__.py         # Compiles directory as an importable module
        ├── state.py            # LangGraph production state schemas
        ├── ingestion.py        # PDF structural data loading and extraction logic
        ├── chunking.py         # Sentence segmentation & sliding context windows
        ├── models.py           # Embeddings, Vector Stores, and Local Inference Wrappers
        └── graph.py            # State Graph loop orchestrations, nodes, and edges

```

---

## 🛠️ Installation & Setup

This repository is designed to run locally using on-device compute.

```bash
# 1. Clone the repository
git clone [https://github.com/vrnv1v2/RAG-based-Literaturw-Review.git](https://github.com/vrnv1v2/RAG-based-Literaturw-Review.git)
cd academic-rag-state-graph

# 2. Build isolated virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 3. Install production dependencies
pip install --upgrade pip
pip install -r requirements.txt

```

---

## 📊 Dataset Setup

This pipeline evaluates reasoning performance against a corpus of 13 public academic research papers focusing on machine learning policy, credit scoring risk anomalies, and algorithmic fairness.

To run the pipeline out of the box:

1. Create a folder named `data/` in the root repository path (if it wasn't generated automatically).
2. Place your target academic PDFs directly into that directory.

---

## 🚀 Running the Production Pipeline

Trigger the runtime pipeline via the CLI entry point using the `--query` string parameter. The system will handle document reading, vector space initialization, graph token execution, and output generation entirely within the active terminal session.

```bash
python main.py --query "1 sentence Explanation of paper Kozodoi et al"

```

### Example Logs & Runtime Pipeline Flow

```text
🚀 Initialising Models...
📂 Processing Documents...
📂 Ingesting and parsing 13 local academic papers...
✅ Structural text extraction complete for 13 source layouts.
✂️ Slicing texts into sliding sentence context windows...
🧬 Sending 4271 text blocks to the embedding engine...
✅ FAISS store successfully populated with 4271 tracked vectors.
🔤 Tokenizing 4271 academic chunks for BM25...
✅ BM25 indexing complete.

⚡ Executing LangGraph for Query: '1 sentence Explanation of paper Kozodoi et al'

⚡ [NODE 1] Running Hybrid Search & Reranking for query: '1 sentence Explanation of paper Kozodoi et al'
✅ Extracted 3 highly relevant paper chunks across threshold barriers.

⚡ [NODE 2] Compiling your context prompt layout and executing local inference...

================ OUTPUT ================
Kozodoi et al. (2019) investigated the effects of social support structures on the mental health metrics of individuals experiencing chronic pain conditions, finding that high-affinity support directly correlated with a reduction in anxiety and depression symptoms.
========================================

```

---

## ⚙️ Configuration Adjustments

To alter model setups, update embedding instructions, shift retrieval similarity filters, or increase maximum self-correction retry cycles, modify the values directly inside the centralized **`config.yaml`** file:

```yaml
paths:
  data_dir: "./data"
  vector_store_path: "./local_faiss_store"

models:
  embedding: "BAAI/bge-large-en-v1.5"
  reranker: "cross-encoder/ms-marco-MiniLM-L6-v2"
  llm: "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

parameters:
  sentences_per_chunk: 4
  overlap: 1
  dense_threshold: 0.35
  rerank_threshold: -1.5        # Baseline MS-MARCO logit barrier limit
  max_loops: 2                  # Number of self-correction loops allowed before fallback

```

```

```
