from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import requests, os

# --- Config ---
HF_TOKEN = os.getenv("HF_TOKEN")  # Make sure to set in Render Dashboard
HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"}

app = FastAPI()

# Allow all origins for frontend (your GitHub Pages app)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Request Model ---
class ProcessRequest(BaseModel):
    text: str
    mode: str  # summarize | rewrite | correct | expand | paraphrase

# --- Hugging Face Helper ---
def hf_request(api_url: str, payload: dict, field: str):
    try:
        response = requests.post(api_url, headers=HEADERS, json=payload, timeout=60)

        # Parse JSON safely
        try:
            data = response.json()
        except ValueError:
            raise HTTPException(status_code=502, detail=f"Non-JSON response: {response.text[:200]}")

        # Handle Hugging Face errors
        if response.status_code != 200 or "error" in data:
            raise HTTPException(status_code=502, detail=data)

        # Handle list[dict] or dict response formats
        if isinstance(data, list) and len(data) > 0 and field in data[0]:
            return data[0][field]
        elif isinstance(data, dict) and field in data:
            return data[field]

        raise HTTPException(status_code=500, detail=f"Unexpected response format: {data}")

    except requests.RequestException as e:
        raise HTTPException(status_code=500, detail=f"HuggingFace request failed: {str(e)}")


# --- Routes ---

@app.get("/")
def home():
    return {"status": "AIWord backend running"}

@app.post("/process")
async def process(req: ProcessRequest):
    text = req.text.strip()
    mode = req.mode.lower()

    if not text:
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    if mode == "summarize":
        API_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-cnn"
        result = hf_request(API_URL, {"inputs": text}, "summary_text")

    elif mode == "rewrite":
        API_URL = "https://api-inference.huggingface.co/models/google/flan-t5-large"
        result = hf_request(API_URL, {"inputs": f"Rewrite this in clearer English:\n{text}"}, "generated_text")

    elif mode == "correct":
        API_URL = "https://api-inference.huggingface.co/models/prithivida/grammar_error_correcter_v1"
        result = hf_request(API_URL, {"inputs": text}, "generated_text")

    elif mode == "expand":
        API_URL = "https://api-inference.huggingface.co/models/google/flan-t5-large"
        result = hf_request(API_URL, {"inputs": f"Expand this idea in more detail:\n{text}"}, "generated_text")

    elif mode == "paraphrase":
        API_URL = "https://api-inference.huggingface.co/models/google/flan-t5-large"
        result = hf_request(API_URL, {"inputs": f"Paraphrase this while keeping the same meaning:\n{text}"}, "generated_text")

    else:
        raise HTTPException(status_code=400, detail="Invalid mode. Use: summarize, rewrite, correct, expand, paraphrase")

    return {"mode": mode, "output": result}
