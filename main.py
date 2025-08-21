from fastapi import FastAPI
from pydantic import BaseModel
from huggingface_hub import InferenceClient
import os
from dotenv import load_dotenv

load_dotenv()

# Get your HF token from environment variable (Render dashboard > Environment)
HF_TOKEN = os.getenv("HF_TOKEN")
# You can use free models like "facebook/bart-large-cnn" (summarization)
# or "gpt2" for text generation
client = InferenceClient(token=HF_TOKEN)

app = FastAPI()

class TextRequest(BaseModel):
    text: str

@app.post("/summarize")
async def summarize(req: TextRequest):
    result = client.summarization(
        model="facebook/bart-large-cnn",
        inputs=req.text
    )
    return {"summary": result[0]["summary_text"]}

@app.post("/rewrite")
async def rewrite(req: TextRequest):
    # Using a text generation model
    prompt = f"Rewrite this in clearer English:\n{req.text}"
    result = client.text_generation(
        model="gpt2",
        prompt=prompt,
        max_new_tokens=200
    )
    return {"rewritten": result}
