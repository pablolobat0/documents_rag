from fastapi import FastAPI
from app.routers.chat import chat_router


app = FastAPI(
    title="Documents RAG",
    description="Hosts a RAG Chatbot",
    version="0.0.1",
)

app.include_router(chat_router)


@app.get("/")
def root():
    return "Hola mundo"
