from fastapi import FastAPI
from app.routers.chat import chat_router
from app.routers.vector_storage import vector_storage_router


app = FastAPI(
    title="Documents RAG",
    description="Hosts a RAG Chatbot",
    version="0.0.1",
)

app.include_router(chat_router)
app.include_router(vector_storage_router)


@app.get("/")
def root():
    return "Hola mundo"
