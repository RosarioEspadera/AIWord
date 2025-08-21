from fastapi import FastAPI
from pydantic import BaseModel
import requests, os

HF_TOKEN = os.getenv("HF_TOKEN")
HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"}

app = FastAPI()

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

    # Debug logging
    print("HF status:", response.status_code)
    print("HF response text:", response.text)

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
