from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq
import os

# Load Groq API key (make sure it's set in Render dashboard)
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

app = FastAPI()

# Allow frontend (GitHub Pages, etc.)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to your domain if you want stricter security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request schema
class ProcessRequest(BaseModel):
    text: str
    mode: str

@app.get("/")
async def root():
    return {"status": "AIWord backend running with Groq 🚀"}

def groq_request(prompt: str):
    try:
        completion = client.chat.completions.create(
            model="llama3-8b-8192",  # Free + fast Groq model
            messages=[{"role": "user", "content": prompt}],
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Groq request failed: {str(e)}")

@app.post("/process")
async def process(req: ProcessRequest):
    text = req.text.strip()
    mode = req.mode.lower()

    if not text:
        raise HTTPException(status_code=400, detail="No text provided")

    if mode == "summarize":
        result = groq_request(f"Summarize this text:\n\n{text}")
    elif mode == "rewrite":
        result = groq_request(f"Rewrite this text to make it clearer:\n\n{text}")
    elif mode == "correct":
        result = groq_request(f"Correct the grammar of this text:\n\n{text}")
    elif mode == "expand":
        result = groq_request(f"Expand this text with more detail:\n\n{text}")
    elif mode == "paraphrase":
        result = groq_request(f"Paraphrase this text while keeping the meaning:\n\n{text}")
    else:
        raise HTTPException(status_code=400, detail="Invalid mode")

    return {"mode": mode, "output": result}
