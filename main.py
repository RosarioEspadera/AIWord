from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import os
from openai import OpenAI

# Get OpenAI API key from environment
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("Missing OPENAI_API_KEY environment variable.")

client = OpenAI(api_key=OPENAI_API_KEY)

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


@app.get("/")
async def root():
    return {"status": "AIWord backend running (OpenAI)"}


@app.post("/process")
async def process(req: ProcessRequest):
    text = req.text
    mode = req.mode.lower()

    # Prompt templates for each mode
    prompts = {
        "summarize": f"Summarize this text in 2-3 sentences:\n\n{text}",
        "rewrite": f"Rewrite this text in clearer English:\n\n{text}",
        "correct": f"Correct the grammar and spelling in this text:\n\n{text}",
        "expand": f"Expand this idea with more detail:\n\n{text}",
        "paraphrase": f"Paraphrase this text while keeping the same meaning:\n\n{text}",
    }

    if mode not in prompts:
        raise HTTPException(status_code=400, detail="Invalid mode. Choose: summarize, rewrite, correct, expand, paraphrase")

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",  # or "gpt-4o-mini" for cheaper/faster
            messages=[
                {"role": "system", "content": "You are an assistant that processes text."},
                {"role": "user", "content": prompts[mode]}
            ],
            max_tokens=300
        )

        output = response.choices[0].message.content.strip()
        return {"mode": mode, "output": output}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OpenAI request failed: {str(e)}")
