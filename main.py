from fastapi import FastAPI, HTTPException
from app.schemas import GenerateRequest, GenerateResponse
from app.ollama_client import generate_with_ollama

app = FastAPI(
    title="LLM API via Ollama Desktop",
    version="0.1.0",
)


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/generate", response_model=GenerateResponse)
async def generate_text(request: GenerateRequest):
    try:
        output_text = await generate_with_ollama(
            model=request.model,
            prompt=request.prompt,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return GenerateResponse(
        model=request.model,
        prompt=request.prompt,
        output=output_text,
    )
