# Documents RAG

This project hosts a RAG (Retrieval-Augmented Generation) chatbot as a FastAPI application. It uses Docker for containerization and includes services like Qdrant for vector storage, Ollama for running large language models, and Redis for caching.

## About the Project

The chatbot is built using LangChain and LangGraph, creating a state machine that orchestrates the conversation flow. It can retrieve relevant document snippets from a Qdrant vector store and use a language model from Ollama to generate responses. You can upload your own documents (text or PDF) to the vector store to customize the chatbot's knowledge base.

## Getting Started

To get started with this project, you need to have Docker and Docker Compose installed on your machine.

1. **Clone the repository:**

   ```bash
   git clone https://github.com/your-username/documents-rag.git
   cd documents-rag
   ```

2. **Run the application:**

   ```bash
   docker-compose up -d
   ```

This will start all the services defined in the `compose.yaml` file, including the FastAPI application, Qdrant, Ollama, and Redis.

## Usage

The API is documented using Swagger UI, which you can access at `http://localhost:8000/docs`.

### Chat

To chat with the chatbot, send a POST request to the `/chat/` endpoint with a `ChatConversation` object.

**Request:**

```bash
curl -X 'POST' \
  'http://localhost:8000/chat/' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "messages": [
    {
      "role": "user",
      "content": "What is the capital of France?"
    }
  ]
}'
```

### Upload Documents

To upload a document to the vector store, send a POST request to the `/document/upload` endpoint with the file to upload. The supported file types are `text/plain` and `application/pdf`.

**Request:**

```bash
curl -X 'POST' \
  'http://localhost:8000/document/upload' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@/path/to/your/document.txt'
```

## Project Structure

```
.
├── app/                # FastAPI application
│   ├── routers/        # API routers
│   ├── services/       # Business logic
│   └── ...
├── src/                # Chatbot source code
│   ├── agent.py        # Core agent logic
│   └── ...
├── compose.yaml        # Docker Compose configuration
├── Dockerfile          # Dockerfile for the backend
└── pyproject.toml      # Python dependencies
```
