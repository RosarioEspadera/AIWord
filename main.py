from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq
import os

# Load API key
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError("Missing GROQ_API_KEY in environment variables")

client = Groq(api_key=GROQ_API_KEY)

app = FastAPI()

# Allow all origins (for dev)
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

@app.post("/process")
async def process(req: ProcessRequest):
    text = req.text
    mode = req.mode.lower()

    # Prompt template for different actions
    if mode == "summarize":
        prompt = f"Summarize this text in 2-3 sentences:\n\n{text}"
    elif mode == "rewrite":
        prompt = f"Rewrite this text in clearer English:\n\n{text}"
    elif mode == "correct":
        prompt = f"Correct any grammar/spelling mistakes:\n\n{text}"
    elif mode == "expand":
        prompt = f"Expand this idea with more detail:\n\n{text}"
    elif mode == "paraphrase":
        prompt = f"Paraphrase this text, keeping the same meaning:\n\n{text}"
    else:
        raise HTTPException(status_code=400, detail="Invalid mode")

    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",  # fast + free
            messages=[
                {"role": "system", "content": "You are a helpful AI writing assistant."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=512,
        )
        result = response.choices[0].message["content"]
        return {"mode": mode, "output": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Groq request failed: {str(e)}")
