from fastapi import FastAPI
from pydantic import BaseModel
import requests
import os

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
    prompt = f"Rewrite this in clearer English:\n{req.text}"
    response = requests.post(API_URL, headers=HEADERS, json={"inputs": prompt})
    return {"rewritten": response.json()}
