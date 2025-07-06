from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import Optional
from app.services.answer_builder import build_answer
import time

app = FastAPI()

class AskRequest(BaseModel):
    question: str
    top_k: Optional[int] = 3

@app.get("/")
def read_root():
    return {"message": "QA Assistant is running!"}

@app.get("/ping")
def ping():
    return {"status": "ok"}

@app.post("/ask")
def ask_question(req: AskRequest, request: Request):
    try:
        start = time.time()
        result = build_answer(req.question, top_k=req.top_k)
        end = time.time()
        print(f"[{request.client.host}] ⏱ Xử lý câu hỏi mất {end - start:.2f} giây")
        return result
    except Exception as e:
        return {"error": str(e)}