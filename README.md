# Documents RAG

A Retrieval-Augmented Generation (RAG) system that allows users to upload documents and chat with them using AI. Built with Streamlit, FastAPI, LangChain, and LangGraph, following Domain-Driven Design (DDD) architecture.

## Features

- **Multi-Format Document Processing**: Support for PDF, Markdown, and plain-text files with automatic text extraction
- **Batch Upload from Local Filesystem**: Scan directories and glob patterns on the host machine
- **Image Analysis**: Built-in image captioning using vision LLM for processing images within PDFs
- **Advanced RAG Architecture**: LangGraph state machine with intelligent routing and tool calling
- **Conversation Memory**: Persistent chat sessions using Redis checkpointing
- **Vector Search**: Semantic search with MMR and re-ranking for accurate document retrieval
- **REST API**: FastAPI backend with Swagger docs at `/docs`

## Architecture

```
Host                          Docker
┌─────────────┐    HTTP    ┌──────────────────┐
│  Streamlit   │ ────────> │  FastAPI          │
│  (UI + file  │           │  (use cases,      │
│   discovery) │           │   infra, domain)  │
└─────────────┘           └──────────────────┘
                               │  │  │
                          Qdrant Ollama Redis
```

Streamlit runs on the host so it can access the local filesystem for batch uploads. FastAPI runs in Docker alongside the infrastructure services.

### DDD Layer Structure

```
src/
├── api/              # FastAPI REST layer
│   ├── app.py        # Application factory, lifespan, exception handlers
│   ├── schemas.py    # Pydantic request/response models
│   └── routes/       # Endpoint handlers (chat, documents)
│
├── domain/           # Core business logic (framework-agnostic)
│   ├── entities/     # Document, Metadata
│   ├── ports/        # Protocol-based interfaces
│   └── value_objects/# ChatMessage, SessionId, FileInfo
│
├── infrastructure/   # External implementations (grouped by capability)
│   ├── llm/          # OllamaLLM
│   ├── storage/      # Qdrant, Redis, Filesystem
│   ├── processing/   # TextSplitter, PdfProcessor, MarkdownProcessor
│   └── agent/        # LanggraphAgent
│
├── application/      # Orchestration (depends only on ports)
│   ├── use_cases/    # ProcessDocument, ChatWithDocuments, BatchProcessDocuments
│   └── dto/          # Request/Response objects
│
├── presentation/     # Streamlit UI (runs on host)
│   ├── api_client.py # HTTP client for the FastAPI backend
│   ├── components/   # Chat, BatchFileUploader, Sidebar
│   └── state/        # Session management
│
└── container.py      # Dependency injection
```

## Technology Stack

- **UI**: Streamlit
- **API**: FastAPI + Uvicorn
- **AI Framework**: LangChain + LangGraph
- **LLM**: Ollama (qwen3, llama3.2, gemma3)
- **Embeddings**: all-minilm via Ollama
- **Vector Database**: Qdrant
- **Session Management**: Redis
- **PDF Processing**: PyPDF + Pillow
- **Package Manager**: uv

## Quick Start

### Prerequisites

- Docker and Docker Compose
- Python 3.12+
- uv package manager

### 1. Start the backend and infrastructure

```bash
# Start all services (FastAPI + Qdrant + Ollama + Redis)
docker compose up -d
```

The API will be available at `http://localhost:8000`. Swagger docs at `http://localhost:8000/docs`.

### 2. Start the Streamlit UI

```bash
# Install dependencies
uv sync

# Run the UI on the host
uv run streamlit run main.py
```

The UI will be available at `http://localhost:8501`.

## API Endpoints

All routes are under the `/api/v1` prefix.

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/v1/documents/batch` | Upload and process documents (multipart file upload) |
| `POST` | `/api/v1/chat` | Chat with indexed documents (JSON) |
| `GET` | `/health` | Health check |

## Configuration

Environment variables (`.env` file):

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL` | `qwen3:1.7b` | Main conversation model |
| `SUMMARY_MODEL` | `llama3.2:3b` | Summarization model |
| `EMBEDDINGS_MODEL` | `all-minilm` | Text embedding model |
| `IMAGE_CAPTIONING_MODEL` | `gemma3:4b` | Image analysis model |
| `QDRANT_URL` | `http://qdrant:6333` | Qdrant server URL |
| `QDRANT_COLLECTION_NAME` | `documents` | Vector collection name |
| `OLLAMA_URL` | `http://ollama:11434` | Ollama server URL |
| `REDIS_URL` | `redis://redis:6379` | Redis server URL |
| `API_URL` | `http://localhost:8000` | FastAPI backend URL (used by Streamlit) |
