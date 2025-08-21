from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import requests, os

HF_TOKEN = os.getenv("HF_TOKEN")
HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"}

app = FastAPI()

# Allow all origins (for dev). Restrict later to your frontend domain.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TextRequest(BaseModel):
    text: str

@app.post("/summarize")
async def summarize(req: TextRequest):
    API_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-cnn"
    response = requests.post(API_URL, headers=HEADERS, json={"inputs": req.text})
    return {"summary": response.json()}

@app.post("/rewrite")
async def rewrite(req: TextRequest):
    API_URL = "https://api-inference.huggingface.co/models/google/flan-t5-large"
    payload = {"inputs": f"Rewrite this in clearer English:\n{req.text}"}

    response = requests.post(API_URL, headers=HEADERS, json=payload)

    try:
        data = response.json()
    except Exception:
        raise HTTPException(status_code=500, detail="Invalid response from Hugging Face")

    if "error" in data:
        raise HTTPException(status_code=500, detail=data["error"])

    return {"rewritten": data[0]["generated_text"]}

@app.post("/correct")
async def correct(req: TextRequest):
    API_URL = "https://api-inference.huggingface.co/models/prithivida/grammar_error_correcter_v1"
    response = requests.post(API_URL, headers=HEADERS, json={"inputs": req.text})
    return {"corrected": response.json()}

# ✅ New Expand Endpoint
@app.post("/expand")
async def expand(req: TextRequest):
    API_URL = "https://api-inference.huggingface.co/models/google/flan-t5-large"
    payload = {"inputs": f"Expand this short text into a longer, detailed version:\n{req.text}"}

    response = requests.post(API_URL, headers=HEADERS, json=payload)

    try:
        data = response.json()
    except Exception:
        raise HTTPException(status_code=500, detail="Invalid response from Hugging Face")

    if "error" in data:
        raise HTTPException(status_code=500, detail=data["error"])

    return {"expanded": data[0]["generated_text"]}
