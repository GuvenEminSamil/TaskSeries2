from pydantic import BaseModel, Field

class GenerateRequest(BaseModel):
    prompt: str = Field(..., description="User prompt from the LLM")
    model: str = Field(
        "gemma3:4b",
        description="Ollama model name, e.g. 'gemma3:4b' or 'qwen3:8b'",
    )
    temperature: float = Field(
        0.2,
        ge=0.0,
        le=1.0,
        description="Sampling temperature (0 = more deterministic)",
    )
    max_tokens: int = Field(
        256,
        gt=0,
        description="Maximum number of tokens to generate"
    )

class GenerateResponse(BaseModel):
    model: str
    prompt: str
    output: str