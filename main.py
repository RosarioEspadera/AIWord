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

class ProcessRequest(BaseModel):
    text: str
    mode: str  # summarize | rewrite | correct | expand | paraphrase


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


# --- Single Route ---
@app.post("/process")
async def process(req: ProcessRequest):
    text = req.text
    mode = req.mode.lower()

    if mode == "summarize":
        API_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-cnn"
        result = hf_request(API_URL, {"inputs": text}, "summary_text")
        return {"mode": mode, "output": result}

    elif mode == "rewrite":
        API_URL = "https://api-inference.huggingface.co/models/google/flan-t5-large"
        result = hf_request(API_URL, {"inputs": f"Rewrite this in clearer English:\n{text}"}, "generated_text")
        return {"mode": mode, "output": result}

    elif mode == "correct":
        API_URL = "https://api-inference.huggingface.co/models/prithivida/grammar_error_correcter_v1"
        result = hf_request(API_URL, {"inputs": text}, "generated_text")
        return {"mode": mode, "output": result}

    elif mode == "expand":
        API_URL = "https://api-inference.huggingface.co/models/google/flan-t5-large"
        result = hf_request(API_URL, {"inputs": f"Expand this idea in more detail:\n{text}"}, "generated_text")
        return {"mode": mode, "output": result}

    elif mode == "paraphrase":
        API_URL = "https://api-inference.huggingface.co/models/google/flan-t5-large"
        result = hf_request(API_URL, {"inputs": f"Paraphrase this while keeping the same meaning:\n{text}"}, "generated_text")
        return {"mode": mode, "output": result}

    else:
        raise HTTPException(status_code=400, detail="Invalid mode. Choose from: summarize, rewrite, correct, expand, paraphrase")
