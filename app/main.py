import logging
from fastapi import FastAPI
from app.routers.chat import chat_router
from app.routers.vector_storage import vector_storage_router

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Documents RAG",
    description="A Retrieval-Augmented Generation (RAG) system that allows users to upload documents and chat with them using AI. Supports text and PDF files with image processing capabilities.",
    version="0.0.1",
)

app.include_router(chat_router)
app.include_router(vector_storage_router)


@app.get("/")
def root():
    logger.info("Root endpoint accessed")
    return "Hola mundo"
