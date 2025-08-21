from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import requests, os

HF_TOKEN = os.getenv("HF_TOKEN")
HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"}

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ProcessRequest(BaseModel):
    text: str
    mode: str


def hf_request(api_url: str, payload: dict):
    try:
        response = requests.post(api_url, headers=HEADERS, json=payload, timeout=60)
        try:
            data = response.json()
        except ValueError:
            raise HTTPException(status_code=502, detail=f"Non-JSON response: {response.text[:200]}")

        if response.status_code != 200 or "error" in data:
            raise HTTPException(status_code=502, detail=data)

        # Normalize HuggingFace outputs
        if isinstance(data, list):
            if len(data) > 0:
                if "summary_text" in data[0]:
                    return data[0]["summary_text"]
                if "generated_text" in data[0]:
                    return data[0]["generated_text"]
        elif isinstance(data, dict):
            if "summary_text" in data:
                return data["summary_text"]
            if "generated_text" in data:
                return data["generated_text"]

        return str(data)

    except requests.RequestException as e:
        raise HTTPException(status_code=500, detail=f"HuggingFace request failed: {str(e)}")


@app.post("/process")
async def process(req: ProcessRequest):
    text = req.text
    mode = req.mode.lower()

    if mode == "summarize":
        API_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-cnn"
        result = hf_request(API_URL, {"inputs": text})
    elif mode == "rewrite":
        API_URL = "https://api-inference.huggingface.co/models/google/flan-t5-large"
        result = hf_request(API_URL, {"inputs": f"Rewrite this in clearer English:\n{text}"})
    elif mode == "correct":
        API_URL = "https://api-inference.huggingface.co/models/prithivida/grammar_error_correcter_v1"
        result = hf_request(API_URL, {"inputs": text})
    elif mode == "expand":
        API_URL = "https://api-inference.huggingface.co/models/google/flan-t5-large"
        result = hf_request(API_URL, {"inputs": f"Expand this idea in more detail:\n{text}"})
    elif mode == "paraphrase":
        API_URL = "https://api-inference.huggingface.co/models/google/flan-t5-large"
        result = hf_request(API_URL, {"inputs": f"Paraphrase this while keeping the same meaning:\n{text}"})
    else:
        raise HTTPException(status_code=400, detail="Invalid mode. Choose from: summarize, rewrite, correct, expand, paraphrase")

    return {"mode": mode, "output": result}
