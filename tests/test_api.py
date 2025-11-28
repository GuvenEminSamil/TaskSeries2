import pytest
from fastapi.testclient import TestClient

from app import main

client = TestClient(main.app)

def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}

@pytest.mark.asyncio
async def test_generate_success(monkeypatch):
    async def fake_generate_with_ollama(model: str, prompt: str, temperature: float = 0.2, max_tokens: int = 256):
        return "MOCK_OUTPUT"

    monkeypatch.setattr(main, "generate_with_ollama", fake_generate_with_ollama)

    r = client.post("/generate", json={"prompt": "hi", "model": "gemma3:4b", "temperature": 0.2, "max_tokens": 16})
    assert r.status_code == 200
    data = r.json()
    assert data["output"] == "MOCK_OUTPUT"
    assert data["prompt"] == "hi"
    assert data["model"] == "gemma3:4b"

def test_generate_validation_error():
    r = client.post("/generate", json={"model": "gemma3:4b"})
    assert r.status_code == 422

@pytest.mark.asyncio
async def test_generate_handles_ollama_error(monkeypatch):
    async def fake_generate_with_ollama(*args, **kwargs):
        raise RuntimeError("Ollama down")

    monkeypatch.setattr(main, "generate_with_ollama", fake_generate_with_ollama)

    r = client.post("/generate", json={"prompt": "hi"})
    assert r.status_code == 500
