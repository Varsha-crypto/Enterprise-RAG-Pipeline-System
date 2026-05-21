# RAG Pipeline — Semantic Search Backend

A full-stack RAG (Retrieval-Augmented Generation) pipeline with a FastAPI backend and React frontend. Supports ingesting data from uploaded files or existing PostgreSQL tables, building FAISS vector indexes, and querying the indexed data via semantic search or LLM-powered summaries.

---

## Features

- **Three ingestion routes:**
  - Upload a `.txt` file (supports `.csv`, `.json`, `.pdf`, `.parquet`, and `.docx` formats)
  - Connect to an existing PostgreSQL table as a source
  - Import a previously exported config ZIP
- **Configurable pipeline:** choice of embedding model, chunking strategy (fixed-size or sentence-based), and FAISS index type (HNSW, Flat, IVF)
- **Batch/incremental processing:** large files and tables are processed in batches; partial results are searchable after the first batch completes
- **Semantic search:** query indexed data by similarity using the `/unified-search` endpoint
- **LLM summaries:** query indexed data with a natural language question and receive an LLM-generated answer via Groq API
- **Pipeline cancellation:** cancel a running pipeline mid-execution; partial DB data and index files are automatically cleaned up
- **Config export/import:** download a config ZIP containing the JSON and index files for reuse

---

## Repo Structure

```
├── backend/
│   ├── app/
│   │   ├── main.py               # FastAPI app and all endpoints
│   │   ├── config.py             # Environment config and model registries
│   │   ├── pipeline/
│   │   │   ├── orchestrator.py   # Pipeline execution and cancellation logic
│   │   │   ├── ingestion.py      # File and DB data sources
│   │   │   ├── chunking.py       # Fixed-size and sentence-based chunking
│   │   │   ├── embedding.py      # Embedding generation
│   │   │   └── indexing.py       # FAISS index building and incremental merge
│   │   ├── services/
│   │   │   ├── embedding_service.py  # SentenceTransformer model management
│   │   │   ├── faiss_manager.py      # FAISS index load/search/refresh
│   │   │   └── llm_service.py        # Groq API LLM summaries
│   │   ├── db/
│   │   │   └── dynamic_connection.py # Generic PostgreSQL connection wrapper
│   │   ├── workflows/
│   │   │   ├── import_config.py      # Config import validation
│   │   │   └── build_pipeline.py     # Pipeline config validation utilities
│   │   └── utils/
│   │       └── progress_stream.py    # SSE pub/sub queue manager
│   ├── configs/                  # Runtime-generated pipeline config JSONs (gitignored)
│   ├── faiss_indexes/            # Runtime-generated FAISS index files (gitignored)
│   ├── initdb/                   # SQL run on first Postgres container startup
│   ├── scripts/                  # Standalone utility scripts
│   ├── docker-compose.yml        # PostgreSQL container setup
│   └── requirements.txt
└── frontend/
    ├── src/
    │   ├── App.jsx                         # Root component and wizard state
    │   ├── services/api.js                 # All backend API calls and SSE logic
    │   └── components/wizard/steps/        # One component per wizard step
    └── package.json
```

---

## Local Setup

### Prerequisites

- Python 3.11+
- Node.js 18+
- Docker (for PostgreSQL)
- A [Groq API key](https://console.groq.com/)

### 1. Start PostgreSQL

```bash
cd backend
docker-compose up -d
```

This starts PostgreSQL on port `5433` with user/password `postgres` and database `appdb`.

### 2. Backend

```bash
cd backend
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Create a `.env` file in `backend/`:

```env
GROQ_API_KEY=your_groq_api_key_here
DB_HOST=localhost
DB_PORT=5433
DB_USER=postgres
DB_PASSWORD=postgres
DB_NAME=appdb
```

Start the server:

```bash
cd backend
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

API docs available at `http://localhost:8000/docs`.

### 3. Frontend

```bash
cd frontend
npm install
npm run dev
```

Frontend runs at `http://localhost:5173`.

> **macOS note:** Set `OMP_NUM_THREADS=1` and `KMP_DUPLICATE_LIB_OK=TRUE` before starting uvicorn to avoid OpenMP conflicts between PyTorch and FAISS.

---

## Usage

The frontend is a step-by-step wizard:

1. **Data Ingestion** — choose to upload a file or connect to an existing PostgreSQL source table
2. **Database Configuration** — configure the target database where embeddings will be stored
3. **Pipeline Design** — select embedding model, chunking strategy, index type, and LLM model
4. **Pipeline Execution** — monitor real-time progress via the SSE progress bar
5. **Query Playground** — run semantic searches against the indexed data
6. **Summary** — ask natural language questions and receive LLM-generated answers
=======
# Enterprise RAG Pipeline System

## Overview

Developed a full-stack Retrieval-Augmented Generation (RAG) platform enabling **semantic search**, intelligent document retrieval, and AI-powered summarization over custom enterprise datasets.
The system combines vector embeddings, similarity search, local LLM inference, and scalable vector storage to support efficient knowledge retrieval workflows.



## Features

* Semantic document search and retrieval
* Retrieval-Augmented Generation (RAG) pipeline
* Vector embedding and similarity search
* AI-powered summarization using local LLMs
* Search history tracking
* Duplicate document detection
* Chunk preview and retrieval visualization
* AI confidence scoring
* User feedback collection system
* Backend log monitoring
* Offline AI inference without cloud dependency


## Tech Stack

* Python
* FastAPI
* React
* PostgreSQL
* pgvector
* FAISS
* Docker
* Docker Compose



## Workflow

1. Document ingestion and preprocessing
2. Text chunking and embedding generation
3. Vector storage using PostgreSQL + pgvector
4. Similarity search with FAISS
5. Retrieval of relevant document chunks
6. Local LLM inference for summarization
7. Result scoring, feedback, and monitoring

---

## Core Functionalities

### RAG Pipeline Architecture

* Built a scalable Retrieval-Augmented Generation pipeline
* Enabled semantic search over custom enterprise datasets
* Combined vector retrieval with local LLM summarization

### Embedding & Similarity Search

* Implemented embedding pipelines using BAAI/bge-m3 models
* Integrated FAISS for high-performance similarity search
* Optimized chunk retrieval for contextual relevance

### Advanced Platform Features

Developed enterprise-grade platform capabilities including:

* Search History tracking
* Duplicate Detection
* Chunk Preview visualization
* AI Confidence Scoring
* User Feedback collection
* Backend Log Monitoring

### Local LLM Integration

* Integrated Qwen models for offline AI inference
* Enabled private and cloud-independent summarization workflows
* Reduced external API dependency for secure deployments

### Scalable Deployment

* Containerized the full application using Docker Compose
* Configured PostgreSQL + pgvector for vector database storage
* Designed scalable deployment-ready architecture



## Key Contributions

* Built an end-to-end enterprise RAG pipeline
* Developed semantic search and vector retrieval systems
* Implemented advanced document intelligence features
* Integrated local LLM inference for offline summarization
* Containerized the platform for scalable deployment
* Designed modular full-stack architecture using FastAPI and React



## Future Improvements

* Hybrid search using BM25 + vector retrieval
* Multi-modal document support
* Streaming LLM responses
* Role-based access control (RBAC)
* Distributed vector indexing
* Real-time analytics dashboard



## Use Cases

* Enterprise knowledge management
* Intelligent document retrieval
* AI-powered internal search systems
* Research assistant platforms
* Offline enterprise AI systems
* Secure document intelligence solutions
>>>>>>> 13f1ed1b8ca441e045c829f2a890772bc1b80d9f
