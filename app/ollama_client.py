import httpx

OLLAMA_URL = "http://localhost:11434/api/generate"


async def generate_with_ollama(
    model: str,
    prompt: str,
    temperature: float = 0.2,
    max_tokens: int = 256,
) -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        },
        "stream": False,
    }

    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(OLLAMA_URL, json=payload)
        response.raise_for_status()
        data = response.json()

    if "response" not in data:
        raise RuntimeError(f"Unexpected Ollama response: {data}")

    return data["response"]
