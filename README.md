# Documents RAG

A Retrieval-Augmented Generation (RAG) system that allows users to upload documents and chat with them using AI. Built with FastAPI, LangChain, and LangGraph, featuring intelligent document processing, image analysis, and conversational AI capabilities.

## 🚀 Features

- **Multi-Format Document Processing**: Support for text and PDF files with automatic text extraction
- **Image Analysis**: Built-in image captioning using a LLM for processing images within PDFs
- **Advanced RAG Architecture**: LangGraph state machine with intelligent routing and tool calling
- **Conversation Memory**: Persistent chat sessions with automatic summarization
- **Document Classification**: Automatic classification of CVs, receipts, and general documents
- **Metadata Extraction**: Structured extraction of document information
- **Vector Search**: Semantic search with re-ranking for accurate document retrieval

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   FastAPI App   │    │     Ollama      │    │     Qdrant      │
│   (Backend)     │◄──►│   (LLM Service) │◄──►│ (Vector Store)  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│      Redis      │    │   Image Proc.   │    │   Document      │
│ (Chat Sessions) │    │                 │    │   Processing    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🛠️ Technology Stack

- **Backend**: FastAPI with Python 3.12+
- **AI Framework**: LangChain + LangGraph for agent workflows
- **LLM**: Ollama 
- **Vector Database**: Qdrant for semantic search
- **Document Processing**: PyPDF for PDF extraction, text chunking
- **Session Management**: Redis for conversation persistence
- **Containerization**: Docker + Docker Compose

## 🚀 Quick Start

### Prerequisites

- Docker and Docker Compose
- Python 3.12+ (for development)
- uv package manager (recommended)

### 1. Clone the Repository

```bash
git clone https://github.com/pablolobat0/documents_rag.git
cd documents-rag
```

### 2. Using Docker Compose (Recommended)

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

The application will be available at `http://localhost:8000`
API documentation at `http://localhost:8000/docs`

## 📚 API Usage

### Chat with Documents

Send messages to the AI assistant that can answer questions based on your uploaded documents.

```bash
curl -X 'POST' \
  'http://localhost:8000/chat/' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "session_id": "user123",
  "messages": [
    {
      "role": "user",
      "content": "What information do you have about quarterly reports?"
    }
  ]
}'
```

**Response:**
```json
{
  "response": "Based on the uploaded documents, I found information about quarterly reports in several documents...",
  "session_id": "user123",
  "timestamp": "2024-01-15T10:30:00",
  "sources_used": ["doc1.pdf", "doc2.txt"],
  "tool_calls": null
}
```

### Upload Documents

Upload text or PDF documents to make them available for chat.

```bash
curl -X 'POST' \
  'http://localhost:8000/document/upload' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@/path/to/your/document.pdf'
```

**Response:**
```json
{
  "message": "Documento insertado con éxito"
}
```

## ⚙️ Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL` | `qwen3:1.7b` | Main conversation model |
| `SUMMARY_MODEL` | `llama3.2:3b` | Conversation summarization model |
| `EMBEDDINGS_MODEL` | `all-minilm` | Text embedding model |
| `IMAGE_CAPTIONING_MODEL` | `gemma3:4b` | Image analysis model |
| `QDRANT_URL` | `http://qdrant:6333` | Qdrant server URL |
| `QDRANT_COLLECTION_NAME` | `documents` | Vector collection name |
| `OLLAMA_URL` | `http://ollama:11434` | Ollama server URL |
| `REDIS_URL` | `redis://redis:6379` | Redis server URL |
| `MIN_IMAGE_WIDTH` | `100` | Minimum image width for processing |
| `MIN_IMAGE_HEIGHT` | `100` | Minimum image height for processing |


## 📁 Project Structure

```
.
├── app/                        # FastAPI application
│   ├── main.py                 # Application entry point
│   ├── dependencies.py         # Dependency injection
│   ├── routers/                # API endpoints
│   │   ├── chat.py             # Chat endpoints
│   │   └── vector_storage.py   # Document upload endpoints
│   ├── services/               # Business logic
│   │   ├── chat.py             # Chat service
│   │   ├── vector_storage.py   # Document processing
│   │   ├── metadata.py         # Metadata extraction
│   │   ├── image_captioning.py # Image analysis
│   │   └── metadata_storage.py # Metadata persistence
│   └── schemas/                # Pydantic models
│       ├── chat.py             # Chat schemas
│       └── metadata.py         # Metadata schemas
├── src/                        # Core AI logic
│   ├── agent.py                # LangGraph agent
│   ├── prompts.py              # System prompts
│   └── schemas.py              # Agent schemas
├── compose.yaml                # Docker Compose configuration
├── Dockerfile                  # Backend Docker image
├── pyproject.toml              # Python dependencies
└─ .env                         # Environment variables
```
