import os
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional

from agent.agent import build_agent_executor, format_history

app = FastAPI(title="Genomics AI Agent")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve generated reports
os.makedirs("/app/reports", exist_ok=True)
app.mount("/reports", StaticFiles(directory="/app/reports"), name="reports")

agent_executor = build_agent_executor()


class ChatRequest(BaseModel):
    message: str
    history: Optional[list[dict]] = []


class ChatResponse(BaseModel):
    response: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    result = agent_executor.invoke({
        "input": req.message,
        "chat_history": format_history(req.history or []),
    })
    return ChatResponse(response=result["output"])
