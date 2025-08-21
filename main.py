from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import requests, os

HF_TOKEN = os.getenv("HF_TOKEN")
HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"}

app = FastAPI()

# Allow all origins (for dev). Restrict later if needed.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TextRequest(BaseModel):
    text: str


# --- Utility for safe HF calls ---
def hf_request(api_url: str, payload: dict, field: str):
    try:
        response = requests.post(api_url, headers=HEADERS, json=payload, timeout=60)

        # Try to parse JSON
        try:
            data = response.json()
        except ValueError:
            raise HTTPException(status_code=502, detail=f"Non-JSON response: {response.text[:200]}")

        # Handle Hugging Face errors
        if response.status_code != 200 or "error" in data:
            raise HTTPException(status_code=502, detail=data)

        # Some models return list[dict], others return dict
        if isinstance(data, list) and len(data) > 0 and field in data[0]:
            return data[0][field]
        elif isinstance(data, dict) and field in data:
            return data[field]

        raise HTTPException(status_code=500, detail=f"Unexpected response format: {data}")

    except requests.RequestException as e:
        raise HTTPException(status_code=500, detail=f"HuggingFace request failed: {str(e)}")


# --- Routes ---
@app.post("/summarize")
async def summarize(req: TextRequest):
    API_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-cnn"
    summary = hf_request(API_URL, {"inputs": req.text}, "summary_text")
    return {"summary": summary}


@app.post("/rewrite")
async def rewrite(req: TextRequest):
    API_URL = "https://api-inference.huggingface.co/models/google/flan-t5-large"
    rewritten = hf_request(API_URL, {"inputs": f"Rewrite this in clearer English:\n{req.text}"}, "generated_text")
    return {"rewritten": rewritten}


@app.post("/correct")
async def correct(req: TextRequest):
    API_URL = "https://api-inference.huggingface.co/models/prithivida/grammar_error_correcter_v1"
    corrected = hf_request(API_URL, {"inputs": req.text}, "generated_text")
    return {"corrected": corrected}


@app.post("/expand")
async def expand(req: TextRequest):
    API_URL = "https://api-inference.huggingface.co/models/google/flan-t5-large"
    expanded = hf_request(API_URL, {"inputs": f"Expand this idea in more detail:\n{req.text}"}, "generated_text")
    return {"expanded": expanded}
