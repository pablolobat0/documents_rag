import logging
from contextlib import asynccontextmanager

from fastapi import APIRouter, FastAPI

from config.settings import settings
from src.api.routes.chat import router as chat_router
from src.api.routes.documents import router as documents_router
from src.container import get_container

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.basicConfig(level=logging.INFO)
    logger.info("Starting up — initializing container")
    get_container()
    yield
    logger.info("Shutting down")


app = FastAPI(title="Documents RAG API", description="", lifespan=lifespan)

api_router = APIRouter()

api_router.include_router(documents_router)
api_router.include_router(chat_router)

app.include_router(api_router, prefix=settings.api_prefix)


@app.get("/health")
async def health():
    return {"status": "ok"}
