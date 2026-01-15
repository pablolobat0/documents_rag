# Documents RAG

A Retrieval-Augmented Generation (RAG) system that allows users to upload documents and chat with them using AI. Built with Streamlit, LangChain, and LangGraph, following Domain-Driven Design (DDD) architecture.

## Features

- **Multi-Format Document Processing**: Support for text and PDF files with automatic text extraction
- **Image Analysis**: Built-in image captioning using vision LLM for processing images within PDFs
- **Advanced RAG Architecture**: LangGraph state machine with intelligent routing and tool calling
- **Conversation Memory**: Persistent chat sessions using Redis checkpointing
- **Document Classification**: Automatic classification of CVs, receipts, and general documents
- **Metadata Extraction**: Structured extraction of document information using LLM
- **Vector Search**: Semantic search with MMR and re-ranking for accurate document retrieval

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Streamlit UI   │    │     Ollama      │    │     Qdrant      │
│  (Presentation) │◄──►│   (LLM Service) │◄──►│ (Vector Store)  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                                              │
         ▼                                              │
┌─────────────────┐    ┌─────────────────┐              │
│   Application   │    │      Redis      │◄─────────────┘
│   (Use Cases)   │    │ (Checkpointing) │
└─────────────────┘    └─────────────────┘
         │
         ▼
┌─────────────────┐
│     Domain      │
│ (Ports/Entities)│
└─────────────────┘
```

### DDD Layer Structure

```
src/
├── domain/           # Core business logic (framework-agnostic)
│   ├── entities/     # Document, Metadata, Classification
│   ├── ports/        # Protocol-based interfaces
│   └── value_objects/# ChatMessage, SessionId, FileInfo
│
├── infrastructure/   # External implementations (grouped by capability)
│   ├── llm/          # OllamaLLM
│   ├── storage/      # Qdrant, Redis, Filesystem
│   ├── processing/   # TextSplitter, PdfProcessor, Classifier
│   └── agent/        # LanggraphAgent
│
├── application/      # Orchestration (depends only on ports)
│   ├── use_cases/    # ProcessDocument, ChatWithDocuments
│   └── dto/          # Request/Response objects
│
├── presentation/     # Streamlit UI
│   ├── components/   # Chat, FileUploader, Sidebar
│   └── state/        # Session management
│
└── container.py      # Dependency injection
```

## Technology Stack

- **UI**: Streamlit
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

### Using Docker Compose (Recommended)

```bash
# Start all services
docker compose up -d

# View logs
docker compose logs -f

# Stop services
docker compose down
```

The application will be available at `http://localhost:8501`

### Local Development

```bash
# Install dependencies
uv sync

# Start infrastructure services
docker compose up -d qdrant ollama redis

# Run the app
uv run streamlit run main.py
```

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
