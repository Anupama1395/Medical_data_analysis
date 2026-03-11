import logging
import os
import uuid
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# ---- Hugging Face settings ----
HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    raise RuntimeError("HF_TOKEN is not set. Export it before running the app.")

HF_MODEL = os.environ.get("HF_MODEL", "mistralai/Mistral-7B-Instruct-v0.2")
HF_API_URL = f"https://router.huggingface.co/hf-inference/models/{HF_MODEL}"
HF_HEADERS = {
    "Authorization": f"Bearer {HF_TOKEN}",
    "Content-Type": "application/json",
}

# ---- In-memory session storage ----
# NOTE: This is a simple in-memory storage for demonstration purposes.
# Sessions are lost on application restart and this approach doesn't scale
# across multiple server instances. For production use, consider using
# a persistent storage solution like Redis.
sessions = {}

SYSTEM_PROMPT = "You are a helpful CS teaching assistant. Give concise explanations."

class ChatRequest(BaseModel):
    session_id: str
    message: str = Field(..., min_length=1, max_length=1000)

    @field_validator("message")
    @classmethod
    def strip_message(cls, v):
        return v.strip()

@app.get("/")
def read_root():
    return {"message": "Conversational LLM API is running"}

@app.post("/session")
def create_session():
    session_id = str(uuid.uuid4())
    sessions[session_id] = [{"role": "system", "content": SYSTEM_PROMPT}]
    return {"session_id": session_id}

def build_prompt(history):
    # Convert chat history to a single text prompt
    sys = next((m["content"] for m in history if m["role"] == "system"), SYSTEM_PROMPT)

    lines = [f"System: {sys}", ""]
    for m in history:
        if m["role"] == "user":
            lines.append(f"User: {m['content']}")
        elif m["role"] == "assistant":
            lines.append(f"Assistant: {m['content']}")
    lines.append("Assistant:")
    return "\n".join(lines)

def hf_generate(prompt: str) -> str:
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 200,
            "temperature": 0.7,
            "return_full_text": False
        }
    }

    try:
        r = requests.post(HF_API_URL, headers=HF_HEADERS, json=payload, timeout=60)
    except requests.exceptions.Timeout:
        logger.error("Timeout while connecting to Hugging Face API")
        raise HTTPException(status_code=504, detail="Request to language model timed out")
    except requests.exceptions.ConnectionError as e:
        logger.error("Connection error to Hugging Face API: %s", str(e))
        raise HTTPException(status_code=503, detail="Unable to connect to language model service")

    # Common HF errors include 401 (bad token), 429 (rate limit), 503 (model loading)
    if r.status_code != 200:
        logger.error("Hugging Face API returned status %d: %s", r.status_code, r.text)
        raise HTTPException(status_code=r.status_code, detail=r.text)

    data = r.json()

    # HF can return either list[{"generated_text":...}] or dict with error
    if isinstance(data, dict) and "error" in data:
        logger.error("Hugging Face API error: %s", data["error"])
        raise HTTPException(status_code=500, detail=data["error"])

    if isinstance(data, list) and len(data) > 0 and "generated_text" in data[0]:
        return data[0]["generated_text"].strip()

    # Fallback
    return str(data)

@app.post("/chat")
def chat(request: ChatRequest):
    logger.debug("Processing chat request for session: %s", request.session_id)
    if request.session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    sessions[request.session_id].append({"role": "user", "content": request.message})

    try:
        prompt = build_prompt(sessions[request.session_id])
        llm_output = hf_generate(prompt)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Unexpected error during chat processing: %s", str(e))
        raise HTTPException(status_code=500, detail="An unexpected error occurred while processing your request.")

    sessions[request.session_id].append({"role": "assistant", "content": llm_output})

    turn_count = len([m for m in sessions[request.session_id] if m["role"] == "user"])

    return {"response": llm_output, "turn_count": turn_count}
