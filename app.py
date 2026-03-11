import os
import uuid
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

app = FastAPI()

HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    raise RuntimeError("HF_TOKEN is not set. Export it before running the app.")

HF_MODEL = os.environ.get("HF_MODEL", "mistralai/Mistral-7B-Instruct-v0.2")

# FIXED: Correct OpenAI-compatible chat completions endpoint
HF_API_URL = f"https://router.huggingface.co/hf-inference/models/{HF_MODEL}/v1/chat/completions"
HF_HEADERS = {
    "Authorization": f"Bearer {HF_TOKEN}",
    "Content-Type": "application/json",
}

sessions = {}
SYSTEM_PROMPT = "You are a helpful CS teaching assistant. Give concise explanations."

class ChatRequest(BaseModel):
    session_id: str
    message: str = Field(..., min_length=1, max_length=1000)

    @field_validator("message")
    @classmethod
    def strip_message(cls, v: str) -> str:
        return v.strip()


@app.get("/")
def read_root():
    return {"message": "Conversational LLM API is running"}


@app.post("/session")
def create_session():
    session_id = str(uuid.uuid4())
    sessions[session_id] = [{"role": "system", "content": SYSTEM_PROMPT}]
    return {"session_id": session_id}


def call_hf_inference(messages: list[dict]) -> str:
    payload = {
        "model": HF_MODEL,
        "messages": messages,
        "max_tokens": 256,
        "temperature": 0.7,
        "stream": False,
    }

    try:
        r = requests.post(HF_API_URL, headers=HF_HEADERS, json=payload, timeout=60)
    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Hugging Face request failed: {e}")

    if r.status_code in (401, 403):
        raise HTTPException(status_code=r.status_code, detail="HF auth failed or model is gated.")
    if r.status_code == 429:
        raise HTTPException(status_code=429, detail="HF quota/rate limit exceeded.")
    if r.status_code == 503:
        raise HTTPException(status_code=503, detail="Model is loading. Retry in a minute.")
    if not r.ok:
        raise HTTPException(status_code=500, detail=f"HF error: {r.status_code} {r.text}")

    try:
        data = r.json()
    except ValueError:
        raise HTTPException(status_code=500, detail=f"HF returned non-JSON response: {r.text[:500]}")
    try:
        return data["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError):
        raise HTTPException(status_code=500, detail=f"Unexpected response format: {data}")


@app.post("/chat")
def chat(request: ChatRequest):
    if request.session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    sessions[request.session_id].append({"role": "user", "content": request.message})

    try:
        llm_output = call_hf_inference(sessions[request.session_id])
        if not llm_output:
            raise HTTPException(status_code=500, detail="Empty response from model.")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Something went wrong: {e}")

    sessions[request.session_id].append({"role": "assistant", "content": llm_output})
    turn_count = len([m for m in sessions[request.session_id] if m["role"] == "user"])

    return {"response": llm_output, "turn_count": turn_count, "model": HF_MODEL}
