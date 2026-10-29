import asyncio

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import app
from app.providers import ClaudeProvider, GeminiProvider, GrokProvider, GroqProvider, NvidiaProvider, OllamaProvider, OpenAIProvider, ProviderError, ProviderResult
from app.schemas import ChatMessage, ChatRequest
from app.service import build_prompt


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    monkeypatch.setenv("LLM_MODE", "mock")
    monkeypatch.setenv("MIDDLEWARE_API_KEY", "")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setenv("DEFAULT_PROVIDER", "claude")
    monkeypatch.setenv("XAI_API_KEY", "")
    monkeypatch.setenv("OLLAMA_API_KEY", "")
    monkeypatch.setenv("OLLAMA_BASE_URL", "https://ollama.com/v1")
    monkeypatch.setenv("OLLAMA_DEFAULT_MODEL", "gemma4:31b")
    monkeypatch.setenv("GROQ_API_KEY", "")
    monkeypatch.setenv("GROQ_DEFAULT_MODEL", "llama-3.3-70b-versatile")
    monkeypatch.setenv("GEMINI_API_KEY", "")
    monkeypatch.setenv("GEMINI_DEFAULT_MODEL", "gemini-3.5-flash-lite")
    monkeypatch.setenv("GEMINI_IMAGE_MODEL", "gemini-nano-banana-2.1")
    monkeypatch.setenv("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai")
    monkeypatch.setenv("AIHORDE_API_KEY", "")
    monkeypatch.setenv("AIHORDE_IMAGE_MODEL", "stable_diffusion")
    monkeypatch.setenv("AIHORDE_BASE_URL", "https://aihorde.net/api/v2")
    monkeypatch.setenv("HF_CREDENTIALS", "")
    monkeypatch.setenv("HF_VIDEO_MODEL", "bytedance/seedance-2.5/text-to-video")
    monkeypatch.setenv("HF_BASE_URL", "https://api.higgsfield.ai")
    monkeypatch.setenv("NVIDIA_API_KEY", "")
    monkeypatch.setenv("NVIDIA_DEFAULT_MODEL", "nvidia/nemotron-3.5-lightning-30b-a3b")
    monkeypatch.setenv("GROK_DEFAULT_MODEL", "grok-4.7")
    monkeypatch.setenv("CLAUDE_DEFAULT_MODEL", "claude-sonnet-4-5")
    monkeypatch.setenv("OPENAI_DEFAULT_MODEL", "gpt-4o-mini")


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_home_page_has_the_call_form(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "Call API" in response.text
    assert "Query" in response.text
    assert "Context" in response.text
    assert "Stream the answer" in response.text
    assert "Generate an image" in response.text
    assert "Generate a video" not in response.text
    assert 'href="/swagger"' in response.text


def test_swagger_ui_and_openapi(client):
    swagger = client.get("/swagger")
    assert swagger.status_code == 200
    assert "swagger-ui" in swagger.text
    index = client.get("/swagger/index.html")
    assert index.status_code == 200
    docs = client.get("/docs")
    assert docs.status_code == 200
    assert "swagger-ui" in docs.text

    spec = client.get("/openapi.json")
    assert spec.status_code == 200
    body = spec.json()
    assert set(body["paths"]) == {"/api/v1/chat", "/api/v1/images", "/v1/chat/completions"}
    examples = body["paths"]["/api/v1/chat"]["post"]["requestBody"]["content"]["application/json"]["examples"]
    assert examples["claude"]["value"]["type"] == "claude"
    assert examples["chatgpt"]["value"]["type"] == "chatgpt"
    assert examples["grok"]["value"]["type"] == "grok"
    assert examples["nvidia"]["value"]["type"] == "nvidia"
    assert examples["groq"]["value"]["type"] == "groq"
    assert examples["ollama"]["value"]["type"] == "ollama"
    assert examples["gemini"]["value"]["type"] == "gemini"


def test_providers_lists_claude_and_chatgpt(client):
    response = client.get("/api/v1/providers")
    assert response.status_code == 200
    body = response.json()
    assert body["mode"] == "mock"
    assert body["endpoint"] == "/api/v1/chat"
    types = {item["type"] for item in body["providers"]}
    assert types == {"claude", "chatgpt", "grok", "groq", "nvidia", "ollama", "gemini", "horde"}


def test_application_one_calls_claude_with_auto_model(client):
    response = client.post(
        "/api/v1/chat",
        json={
            "app": "application-one",
            "model": "auto",
            "type": "claude",
            "query": "Say hello",
            "context": "Be brief.",
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["app"] == "application-one"
    assert body["type"] == "claude"
    assert body["provider"] == "claude"
    assert body["model"] == "claude-sonnet-4-5"
    assert body["requested_model"] == "auto"
    assert body["query"] == "Say hello"
    assert body["context"] == "Be brief."
    assert body["mock"] is True
    assert "type=claude" in body["answer"]
    assert "query=Say hello" in body["answer"]


def test_clude_alias_goes_to_claude(client):
    response = client.post(
        "/api/v1/chat",
        json={"model": "auto", "type": "clude", "query": "Hello", "app": "application-one"},
    )
    assert response.status_code == 200
    assert response.json()["type"] == "claude"
    assert response.json()["provider"] == "claude"


def test_chatgpt_application_uses_the_same_api(client):
    response = client.post(
        "/api/v1/chat",
        json={
            "app": "chatgpt-app",
            "model": "auto",
            "type": "chatgpt",
            "query": "Write a haiku",
            "context": "Keep it short.",
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["app"] == "chatgpt-app"
    assert body["type"] == "chatgpt"
    assert body["provider"] == "openai"
    assert body["model"] == "gpt-4o-mini"
    assert body["context"] == "Keep it short."


def test_auto_type_follows_the_model_name(client):
    response = client.post(
        "/api/v1/chat",
        json={"model": "gpt-4o", "type": "auto", "query": "Hello", "app": "second-app"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["type"] == "chatgpt"
    assert body["model"] == "gpt-4o"
    assert body["app"] == "second-app"


def test_auto_type_and_auto_model_use_the_default_provider(client):
    response = client.post("/api/v1/chat", json={"model": "auto", "type": "auto", "query": "Hello"})
    assert response.status_code == 200
    body = response.json()
    assert body["type"] == "claude"
    assert body["model"] == "claude-sonnet-4-5"
    assert body["app"] == "default"


def test_app_name_header_is_used_when_body_app_is_omitted(client):
    response = client.post(
        "/api/v1/chat",
        headers={"X-App-Name": "billing-app"},
        json={"model": "auto", "type": "claude", "query": "Hello"},
    )
    assert response.status_code == 200
    assert response.json()["app"] == "billing-app"


def test_query_is_required(client):
    response = client.post("/api/v1/chat", json={"model": "auto", "type": "claude"})
    assert response.status_code == 400
    assert "query" in response.json()["error"]["message"]


def test_unknown_type_is_rejected(client):
    response = client.post("/api/v1/chat", json={"type": "mistral", "query": "Hello"})
    assert response.status_code == 400
    assert "claude" in response.json()["error"]["message"]


def test_gemini_uses_the_flash_model(client):
    response = client.post(
        "/api/v1/chat",
        json={"model": "auto", "type": "gemini", "query": "Hello", "app": "gemini-app"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["type"] == "gemini"
    assert body["provider"] == "gemini"
    assert body["model"] == "gemini-3.5-flash-lite"


def test_stream_returns_server_sent_events(client):
    response = client.post(
        "/api/v1/chat",
        json={"type": "claude", "query": "Hello", "stream": True, "app": "application-one"},
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    text = response.text
    assert "event: start" in text
    assert "event: delta" in text
    assert "event: done" in text
    assert "Hello" in text
    assert '"answer"' in text
    assert text.rstrip().endswith("data: [DONE]")


def test_stream_without_a_key_stays_json(client, monkeypatch):
    monkeypatch.setenv("LLM_MODE", "live")
    response = client.post(
        "/api/v1/chat",
        json={"type": "ollama", "model": "auto", "query": "Hello", "stream": True, "app": "ollama-app"},
    )
    assert response.status_code == 503
    assert response.headers["content-type"].startswith("application/json")
    assert "OLLAMA_API_KEY" in response.json()["error"]["message"]


def test_invalid_app_name_is_rejected(client):
    response = client.post(
        "/api/v1/chat",
        json={"app": "bad/name", "type": "claude", "query": "Hello"},
    )
    assert response.status_code == 400


def test_metadata_is_returned(client):
    response = client.post(
        "/api/v1/chat",
        json={"type": "claude", "query": "Hello", "metadata": {"trace": "abc"}},
    )
    assert response.status_code == 200
    assert response.json()["metadata"] == {"trace": "abc"}


def test_api_key_is_required_when_configured(client, monkeypatch):
    monkeypatch.setenv("MIDDLEWARE_API_KEY", "secret-key")
    payload = {"type": "claude", "query": "Hello", "app": "application-one"}

    rejected = client.post("/api/v1/chat", json=payload)
    assert rejected.status_code == 401

    allowed = client.post("/api/v1/chat", json=payload, headers={"X-API-Key": "secret-key"})
    assert allowed.status_code == 200

    bearer = client.post(
        "/api/v1/chat",
        json=payload,
        headers={"Authorization": "Bearer secret-key"},
    )
    assert bearer.status_code == 200


def test_chatgpt_style_completions_endpoint(client):
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "auto",
            "messages": [
                {"role": "system", "content": "Be brief."},
                {"role": "user", "content": "Hello from chatgpt"},
            ],
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "chat.completion"
    assert body["model"] == "gpt-4o-mini"
    assert body["choices"][0]["message"]["role"] == "assistant"
    assert "type=chatgpt" in body["choices"][0]["message"]["content"]
    assert "query=Hello from chatgpt" in body["choices"][0]["message"]["content"]


def test_openai_stream_returns_chunks(client):
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "auto",
            "stream": True,
            "messages": [{"role": "user", "content": "Hello from chatgpt"}],
        },
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    text = response.text
    assert "chat.completion.chunk" in text
    assert "query=Hello" in text
    assert "chatgpt" in text
    assert text.rstrip().endswith("data: [DONE]")


def test_missing_claude_key_returns_503(client, monkeypatch):
    monkeypatch.setenv("LLM_MODE", "live")
    response = client.post(
        "/api/v1/chat",
        json={"type": "claude", "model": "auto", "query": "Hello", "app": "application-one"},
    )
    assert response.status_code == 503
    assert "ANTHROPIC_API_KEY" in response.json()["error"]["message"]


def test_missing_openai_key_returns_503(client, monkeypatch):
    monkeypatch.setenv("LLM_MODE", "live")
    response = client.post(
        "/api/v1/chat",
        json={"type": "chatgpt", "model": "auto", "query": "Hello", "app": "chatgpt-app"},
    )
    assert response.status_code == 503
    assert "OPENAI_API_KEY" in response.json()["error"]["message"]


def test_live_claude_request_keeps_context_query_and_history(client, monkeypatch):
    monkeypatch.setenv("LLM_MODE", "live")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    captured = {}

    async def fake_complete(self, model, system, messages, temperature, max_tokens, top_p):
        captured.update(
            {
                "model": model,
                "system": system,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "top_p": top_p,
            }
        )
        return ProviderResult(text="Answer from Claude", model=model, input_tokens=5, output_tokens=3)

    monkeypatch.setattr("app.service.ClaudeProvider.complete", fake_complete)
    response = client.post(
        "/api/v1/chat",
        json={
            "app": "application-one",
            "model": "auto",
            "type": "claude",
            "query": "What did I ask?",
            "context": "The user is building middleware.",
            "temperature": 0.2,
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi"},
            ],
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "Answer from Claude"
    assert body["mock"] is False
    assert body["usage"]["input_tokens"] == 5
    assert body["usage"]["output_tokens"] == 3
    assert body["usage"]["total_tokens"] == 8
    assert captured["model"] == "claude-sonnet-4-5"
    assert captured["system"] == "The user is building middleware."
    assert captured["messages"] == [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi"},
        {"role": "user", "content": "What did I ask?"},
    ]
    assert captured["temperature"] == 0.2


def test_build_prompt_does_not_duplicate_the_query():
    request = ChatRequest(
        query="Hello",
        context="Be brief",
        messages=[
            ChatMessage(role="system", content="Extra"),
            ChatMessage(role="user", content="Hello"),
        ],
    )
    system, messages, query = build_prompt(request)
    assert system == "Be brief\n\nExtra"
    assert messages == [{"role": "user", "content": "Hello"}]
    assert query == "Hello"


class FakeResponse:
    def __init__(self, status_code, payload):
        self.status_code = status_code
        self._payload = payload
        self.text = str(payload)

    def json(self):
        return self._payload


class FakeClient:
    def __init__(self, response):
        self.response = response
        self.url = None
        self.headers = None
        self.payload = None

    async def post(self, url, headers, json):
        self.url = url
        self.headers = headers
        self.payload = json
        return self.response


def test_claude_provider_posts_context_as_system_and_merges_user_turns():
    client = FakeClient(
        FakeResponse(
            200,
            {
                "model": "claude-sonnet-4-5",
                "stop_reason": "end_turn",
                "content": [{"type": "text", "text": "Hello back"}],
                "usage": {"input_tokens": 4, "output_tokens": 2},
            },
        )
    )
    settings = Settings(anthropic_api_key="sk-ant-test", anthropic_base_url="https://api.anthropic.com")

    async def scenario():
        return await ClaudeProvider(settings, client).complete(
            model="claude-sonnet-4-5",
            system="Be brief",
            messages=[
                {"role": "user", "content": "One"},
                {"role": "user", "content": "Two"},
            ],
            temperature=1.4,
            max_tokens=128,
            top_p=0.5,
        )

    result = asyncio.run(scenario())
    assert result.text == "Hello back"
    assert client.url == "https://api.anthropic.com/v1/messages"
    assert client.headers["x-api-key"] == "sk-ant-test"
    assert client.headers["anthropic-version"] == "2023-06-01"
    assert client.payload["system"] == "Be brief"
    assert client.payload["messages"] == [{"role": "user", "content": "One\n\nTwo"}]
    assert client.payload["temperature"] == 1
    assert "top_p" not in client.payload
    assert "sk-ant-test" not in str(client.payload)


def test_grouk_alias_calls_grok(client):
    response = client.post(
        "/api/v1/chat",
        json={"model": "auto", "type": "grouk", "query": "Hello", "app": "grok-app"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["type"] == "grok"
    assert body["provider"] == "grok"
    assert body["model"] == "grok-4.7"


def test_grok_model_name_selects_grok_when_type_is_auto(client):
    response = client.post(
        "/api/v1/chat",
        json={"model": "grok-4", "type": "auto", "query": "Hello", "app": "grok-app"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["type"] == "grok"
    assert body["model"] == "grok-4"


def test_missing_grok_key_returns_503(client, monkeypatch):
    monkeypatch.setenv("LLM_MODE", "live")
    response = client.post(
        "/api/v1/chat",
        json={"type": "grok", "model": "auto", "query": "Hello", "app": "grok-app"},
    )
    assert response.status_code == 503
    assert "XAI_API_KEY" in response.json()["error"]["message"]


def test_nim_alias_calls_nvidia_and_keeps_the_model_id(client):
    response = client.post(
        "/api/v1/chat",
        json={"model": "auto", "type": "nim", "query": "Hello", "app": "nvidia-app"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["type"] == "nvidia"
    assert body["provider"] == "nvidia"
    assert body["model"] == "nvidia/nemotron-3.5-lightning-30b-a3b"

    named = client.post(
        "/api/v1/chat",
        json={
            "model": "nvidia/llama-3.1-70b-instruct",
            "type": "auto",
            "query": "Hello",
            "app": "nvidia-app",
        },
    )
    assert named.status_code == 200
    assert named.json()["type"] == "nvidia"
    assert named.json()["model"] == "nvidia/llama-3.1-70b-instruct"


def test_missing_nvidia_key_returns_503(client, monkeypatch):
    monkeypatch.setenv("LLM_MODE", "live")
    response = client.post(
        "/api/v1/chat",
        json={"type": "nvidia", "model": "auto", "query": "Hello", "app": "nvidia-app"},
    )
    assert response.status_code == 503
    assert "NVIDIA_API_KEY" in response.json()["error"]["message"]


def test_groq_is_not_sent_to_grok(client):
    response = client.post(
        "/api/v1/chat",
        json={"model": "auto", "type": "groq", "query": "Hello", "app": "groq-app"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["type"] == "groq"
    assert body["provider"] == "groq"
    assert body["model"] == "llama-3.3-70b-versatile"

    grok = client.post(
        "/api/v1/chat",
        json={"model": "auto", "type": "grok", "query": "Hello", "app": "grok-app"},
    )
    assert grok.status_code == 200
    assert grok.json()["provider"] == "grok"
    assert grok.json()["model"] == "grok-4.7"


def test_missing_groq_key_returns_503(client, monkeypatch):
    monkeypatch.setenv("LLM_MODE", "live")
    response = client.post(
        "/api/v1/chat",
        json={"type": "groq", "model": "auto", "query": "Hello", "app": "groq-app"},
    )
    assert response.status_code == 503
    assert "GROQ_API_KEY" in response.json()["error"]["message"]


def test_ollama_uses_the_cloud_model(client):
    response = client.post(
        "/api/v1/chat",
        json={"model": "auto", "type": "ollama", "query": "Hello", "app": "ollama-app"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["type"] == "ollama"
    assert body["provider"] == "ollama"
    assert body["model"] == "gemma4:31b"


def test_missing_ollama_key_returns_503(client, monkeypatch):
    monkeypatch.setenv("LLM_MODE", "live")
    response = client.post(
        "/api/v1/chat",
        json={"type": "ollama", "model": "auto", "query": "Hello", "app": "ollama-app"},
    )
    assert response.status_code == 503
    assert "OLLAMA_API_KEY" in response.json()["error"]["message"]


def test_ollama_provider_posts_to_local_server_without_a_key():
    client = FakeClient(
        FakeResponse(
            200,
            {
                "model": "llama3.2",
                "choices": [{"message": {"role": "assistant", "content": "Hello from Ollama"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 2},
            },
        )
    )
    settings = Settings(ollama_api_key="", ollama_base_url="http://127.0.0.1:11434")

    async def scenario():
        return await OllamaProvider(settings, client).complete(
            model="llama3.2",
            system="Be brief",
            messages=[{"role": "user", "content": "Hello"}],
            temperature=None,
            max_tokens=64,
            top_p=None,
        )

    result = asyncio.run(scenario())
    assert result.text == "Hello from Ollama"
    assert client.url == "http://127.0.0.1:11434/v1/chat/completions"
    assert "Authorization" not in client.headers
    assert client.payload["model"] == "llama3.2"
    assert client.payload["stream"] is False


def test_ollama_provider_posts_to_ollama_cloud():
    client = FakeClient(
        FakeResponse(
            200,
            {
                "model": "gemma4:31b",
                "choices": [{"message": {"role": "assistant", "content": "Hello from Ollama Cloud"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 2},
            },
        )
    )
    settings = Settings(ollama_api_key="ollama-test-key", ollama_base_url="https://ollama.com/v1")

    async def scenario():
        return await OllamaProvider(settings, client).complete(
            model="gemma4:31b",
            system="Be brief",
            messages=[{"role": "user", "content": "Hello"}],
            temperature=None,
            max_tokens=64,
            top_p=None,
        )

    result = asyncio.run(scenario())
    assert result.text == "Hello from Ollama Cloud"
    assert client.url == "https://ollama.com/v1/chat/completions"
    assert client.headers["Authorization"] == "Bearer ollama-test-key"
    assert client.payload["model"] == "gemma4:31b"


def test_ollama_stream_reads_sse_deltas():
    lines = [
        'data: {"model":"gemma4:31b","choices":[{"delta":{"role":"assistant"},"finish_reason":null}]}',
        "",
        'data: {"model":"gemma4:31b","choices":[{"delta":{"content":"Hello"},"finish_reason":null}]}',
        "",
        'data: {"model":"gemma4:31b","choices":[{"delta":{"content":" there"},"finish_reason":null}]}',
        "",
        'data: {"model":"gemma4:31b","choices":[{"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":3,"completion_tokens":2}}',
        "",
        "data: [DONE]",
        "",
    ]

    class FakeStreamResponse:
        status_code = 200

        def __init__(self, body):
            self.body = body

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return False

        async def aiter_lines(self):
            for line in self.body:
                yield line

        async def aread(self):
            return b""

    class FakeStreamClient:
        def __init__(self, body):
            self.body = body
            self.url = None
            self.headers = None
            self.payload = None

        def stream(self, _method, url, headers=None, json=None):
            self.url = url
            self.headers = headers
            self.payload = json
            return FakeStreamResponse(self.body)

    client = FakeStreamClient(lines)
    settings = Settings(ollama_api_key="ollama-test-key", ollama_base_url="https://ollama.com/v1")

    async def scenario():
        pieces = []
        async for piece in OllamaProvider(settings, client).stream(
            model="gemma4:31b",
            system="Be brief",
            messages=[{"role": "user", "content": "Hello"}],
            temperature=None,
            max_tokens=64,
            top_p=None,
        ):
            pieces.append(piece)
        return pieces

    pieces = asyncio.run(scenario())
    assert "".join(piece.text for piece in pieces) == "Hello there"
    assert pieces[-1].finish_reason == "stop"
    assert pieces[-1].output_tokens == 2
    assert client.url == "https://ollama.com/v1/chat/completions"
    assert client.payload["stream"] is True
    assert client.headers["Authorization"] == "Bearer ollama-test-key"


def test_groq_provider_posts_to_groq():
    client = FakeClient(
        FakeResponse(
            200,
            {
                "model": "llama-3.3-70b-versatile",
                "choices": [{"message": {"role": "assistant", "content": "Hello from Groq"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 7, "completion_tokens": 3},
            },
        )
    )
    settings = Settings(groq_api_key="gsk_test_key", groq_base_url="https://api.groq.com/openai/v1")

    async def scenario():
        return await GroqProvider(settings, client).complete(
            model="llama-3.3-70b-versatile",
            system="Be brief",
            messages=[{"role": "user", "content": "Hello"}],
            temperature=None,
            max_tokens=64,
            top_p=None,
        )

    result = asyncio.run(scenario())
    assert result.text == "Hello from Groq"
    assert client.url == "https://api.groq.com/openai/v1/chat/completions"
    assert client.headers["Authorization"] == "Bearer gsk_test_key"
    assert client.payload["model"] == "llama-3.3-70b-versatile"
    assert "gsk_test_key" not in str(client.payload)


def test_nvidia_provider_posts_to_nim():
    client = FakeClient(
        FakeResponse(
            200,
            {
                "model": "nvidia/nemotron-3.5-lightning-30b-a3b",
                "choices": [{"message": {"role": "assistant", "content": "Hello from NIM"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 9, "completion_tokens": 4},
            },
        )
    )
    settings = Settings(
        nvidia_api_key="nvapi-test-key",
        nvidia_base_url="https://integrate.api.nvidia.com/v1",
    )

    async def scenario():
        return await NvidiaProvider(settings, client).complete(
            model="nvidia/nemotron-3.5-lightning-30b-a3b",
            system="Be brief",
            messages=[{"role": "user", "content": "Hello"}],
            temperature=None,
            max_tokens=64,
            top_p=None,
        )

    result = asyncio.run(scenario())
    assert result.text == "Hello from NIM"
    assert client.url == "https://integrate.api.nvidia.com/v1/chat/completions"
    assert client.headers["Authorization"] == "Bearer nvapi-test-key"
    assert client.payload["model"] == "nvidia/nemotron-3.5-lightning-30b-a3b"
    assert client.payload["max_tokens"] == 64
    assert client.payload["stream"] is False
    assert "nvapi-test-key" not in str(client.payload)


def test_grok_provider_posts_to_xai():
    client = FakeClient(
        FakeResponse(
            200,
            {
                "model": "grok-4.7",
                "choices": [{"message": {"role": "assistant", "content": "Hello from Grok"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 8, "completion_tokens": 4},
            },
        )
    )
    settings = Settings(xai_api_key="xai-test-key", xai_base_url="https://api.x.ai/v1")

    async def scenario():
        return await GrokProvider(settings, client).complete(
            model="grok-4.7",
            system="Be brief",
            messages=[{"role": "user", "content": "Hello"}],
            temperature=None,
            max_tokens=64,
            top_p=None,
        )

    result = asyncio.run(scenario())
    assert result.text == "Hello from Grok"
    assert client.url == "https://api.x.ai/v1/chat/completions"
    assert client.headers["Authorization"] == "Bearer xai-test-key"
    assert client.payload["model"] == "grok-4.7"
    assert client.payload["max_completion_tokens"] == 64
    assert client.payload["messages"][0] == {"role": "system", "content": "Be brief"}
    assert "xai-test-key" not in str(client.payload)


def test_openai_provider_posts_chatgpt_messages():
    client = FakeClient(
        FakeResponse(
            200,
            {
                "model": "gpt-4o-mini",
                "choices": [{"message": {"role": "assistant", "content": "Hi"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 6, "completion_tokens": 1},
            },
        )
    )
    settings = Settings(openai_api_key="sk-openai-test", openai_base_url="https://api.openai.com/v1")

    async def scenario():
        return await OpenAIProvider(settings, client).complete(
            model="gpt-4o-mini",
            system="Be brief",
            messages=[{"role": "user", "content": "Hello"}],
            temperature=0.4,
            max_tokens=64,
            top_p=None,
        )

    result = asyncio.run(scenario())
    assert result.text == "Hi"
    assert result.input_tokens == 6
    assert client.url == "https://api.openai.com/v1/chat/completions"
    assert client.headers["Authorization"] == "Bearer sk-openai-test"
    assert client.payload["messages"][0] == {"role": "system", "content": "Be brief"}
    assert client.payload["messages"][1] == {"role": "user", "content": "Hello"}
    assert "sk-openai-test" not in str(client.payload)


def test_provider_auth_failure_does_not_look_like_a_middleware_key_error():
    client = FakeClient(FakeResponse(401, {"error": {"message": "invalid x-api-key"}}))
    settings = Settings(anthropic_api_key="sk-ant-test")

    async def scenario():
        await ClaudeProvider(settings, client).complete(
            model="claude-sonnet-4-5",
            system="",
            messages=[{"role": "user", "content": "Hi"}],
            temperature=None,
            max_tokens=32,
            top_p=None,
        )

    with pytest.raises(ProviderError) as caught:
        asyncio.run(scenario())
    assert caught.value.status == 502
    assert "ANTHROPIC_API_KEY" in caught.value.message


def test_image_auto_uses_grok_and_returns_a_picture(client):
    response = client.post(
        "/api/v1/images",
        json={"type": "auto", "model": "auto", "prompt": "A red bicycle", "app": "image-app"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["type"] == "grok"
    assert body["provider"] == "grok"
    assert body["model"] == "grok-imagine-image-2.0"
    assert body["prompt"] == "A red bicycle"
    assert body["size"] == "1024x1024"
    assert body["mock"] is True
    assert body["images"][0]["b64_json"]


def test_image_chatgpt_uses_the_openai_image_model(client):
    response = client.post(
        "/api/v1/images",
        json={"type": "chatgpt", "prompt": "A blue door", "app": "image-app"},
    )
    assert response.status_code == 200
    assert response.json()["model"] == "gpt-image-2.5-flare"
    assert response.json()["provider"] == "openai"


def test_image_prompt_is_required(client):
    response = client.post("/api/v1/images", json={"type": "grok", "app": "image-app"})
    assert response.status_code == 400
    assert "prompt" in response.json()["error"]["message"]


def test_image_ollama_uses_z_image_turbo(client):
    response = client.post(
        "/api/v1/images",
        json={"type": "ollama", "model": "auto", "prompt": "A red bicycle", "app": "image-app"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["type"] == "ollama"
    assert body["provider"] == "ollama"
    assert body["model"] == "x/z-image-turbo"
    assert body["images"][0]["b64_json"]


def test_nvidia_image_rejects_other_sizes(client):
    response = client.post(
        "/api/v1/images",
        json={"type": "nvidia", "prompt": "A cat", "size": "512x512", "app": "image-app"},
    )
    assert response.status_code == 400
    assert "1024x1024" in response.json()["error"]["message"]


def test_image_nvidia_uses_flux_dev(client):
    response = client.post(
        "/api/v1/images",
        json={"type": "nvidia", "model": "auto", "prompt": "A red bicycle", "app": "image-app"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["provider"] == "nvidia"
    assert body["model"] == "black-forest-labs/flux.1-dev"
    assert body["images"][0]["b64_json"]


def test_nvidia_image_posts_flux_dev_steps():
    from app.images import _generate_live

    client = FakeClient(FakeResponse(200, {"artifacts": [{"base64": "abc123", "finishReason": "SUCCESS"}]}))
    settings = Settings(
        nvidia_api_key="nv-test-key",
        nvidia_image_model="black-forest-labs/flux.1-dev",
        nvidia_image_url="https://ai.api.nvidia.com/v1/genai/black-forest-labs/flux.1-dev",
    )

    async def scenario():
        return await _generate_live(
            settings, client, "nvidia", "black-forest-labs/flux.1-dev", "A red bicycle", "1024x1024", 1
        )

    images = asyncio.run(scenario())
    assert images[0].b64_json == "abc123"
    assert client.url.endswith("/black-forest-labs/flux.1-dev")
    assert client.payload["steps"] == 10
    assert client.payload["width"] == 1024
    assert "nv-test-key" not in str(client.payload)


def test_missing_grok_image_key_returns_503(client, monkeypatch):
    monkeypatch.setenv("LLM_MODE", "live")
    response = client.post(
        "/api/v1/images",
        json={"type": "grok", "prompt": "A cat", "app": "image-app"},
    )
    assert response.status_code == 503
    assert "XAI_API_KEY" in response.json()["error"]["message"]


def test_grok_image_posts_to_xai():
    from app.images import _generate_live

    client = FakeClient(
        FakeResponse(200, {"data": [{"url": "https://example.test/cat.png"}]})
    )
    settings = Settings(xai_api_key="xai-test-key", xai_base_url="https://api.x.ai/v1", grok_image_model="grok-imagine-image-2.0")

    async def scenario():
        return await _generate_live(settings, client, "grok", "grok-imagine-image-2.0", "A cat", "1024x1024", 1)

    images = asyncio.run(scenario())
    assert images[0].url == "https://example.test/cat.png"
    assert client.url == "https://api.x.ai/v1/images/generations"
    assert client.headers["Authorization"] == "Bearer xai-test-key"
    assert client.payload["prompt"] == "A cat"
    assert client.payload["aspect_ratio"] == "1:1"
    assert "xai-test-key" not in str(client.payload)


def test_ollama_image_posts_to_generate():
    from app.images import _generate_live

    client = FakeClient(FakeResponse(200, {"image": "abc123", "done": True}))
    settings = Settings(
        ollama_api_key="ollama-test-key",
        ollama_base_url="https://ollama.com/v1",
        ollama_image_model="x/z-image-turbo",
    )

    async def scenario():
        return await _generate_live(
            settings, client, "ollama", "x/z-image-turbo", "A red bicycle", "1024x768", 1
        )

    images = asyncio.run(scenario())
    assert images[0].b64_json == "abc123"
    assert client.url == "https://ollama.com/api/generate"
    assert client.headers["Authorization"] == "Bearer ollama-test-key"
    assert client.payload["model"] == "x/z-image-turbo"
    assert client.payload["prompt"] == "A red bicycle"
    assert client.payload["width"] == 1024
    assert client.payload["height"] == 768
    assert client.payload["stream"] is False
    assert "ollama-test-key" not in str(client.payload)


def test_missing_gemini_key_returns_503(client, monkeypatch):
    monkeypatch.setenv("LLM_MODE", "live")
    response = client.post(
        "/api/v1/chat",
        json={"type": "gemini", "model": "auto", "query": "Hello", "app": "gemini-app"},
    )
    assert response.status_code == 503
    assert "GEMINI_API_KEY" in response.json()["error"]["message"]


def test_gemini_provider_posts_to_google():
    client = FakeClient(
        FakeResponse(
            200,
            {
                "model": "gemini-3.5-flash-lite",
                "choices": [{"message": {"role": "assistant", "content": "Hello from Gemini"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 4, "completion_tokens": 3},
            },
        )
    )
    settings = Settings(
        gemini_api_key="gemini-test-key",
        gemini_base_url="https://generativelanguage.googleapis.com/v1beta/openai",
    )

    async def scenario():
        return await GeminiProvider(settings, client).complete(
            model="gemini-3.5-flash-lite",
            system="Be brief",
            messages=[{"role": "user", "content": "Hello"}],
            temperature=None,
            max_tokens=64,
            top_p=None,
        )

    result = asyncio.run(scenario())
    assert result.text == "Hello from Gemini"
    assert client.url == "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
    assert client.headers["Authorization"] == "Bearer gemini-test-key"
    assert client.payload["model"] == "gemini-3.5-flash-lite"
    assert "gemini-test-key" not in str(client.payload)


def test_image_gemini_uses_nano_banana(client):
    response = client.post(
        "/api/v1/images",
        json={"type": "gemini", "model": "auto", "prompt": "A red bicycle", "app": "image-app"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["type"] == "gemini"
    assert body["provider"] == "gemini"
    assert body["model"] == "gemini-nano-banana-2.1"
    assert body["images"][0]["b64_json"]


def test_gemini_image_posts_to_generate_content():
    from app.images import _generate_live

    client = FakeClient(
        FakeResponse(
            200,
            {
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {"inlineData": {"mimeType": "image/png", "data": "abc123"}}
                            ]
                        }
                    }
                ]
            },
        )
    )
    settings = Settings(
        gemini_api_key="gemini-test-key",
        gemini_base_url="https://generativelanguage.googleapis.com/v1beta/openai",
        gemini_image_model="gemini-nano-banana-2.1",
    )

    async def scenario():
        return await _generate_live(
            settings, client, "gemini", "gemini-nano-banana-2.1", "A red bicycle", "1024x1024", 1
        )

    images = asyncio.run(scenario())
    assert images[0].b64_json == "abc123"
    assert client.url == "https://generativelanguage.googleapis.com/v1beta/models/gemini-nano-banana-2.1:generateContent"
    assert client.headers["x-goog-api-key"] == "gemini-test-key"
    assert client.payload["contents"][0]["parts"][0]["text"] == "A red bicycle"
    assert client.payload["generationConfig"]["imageConfig"]["aspectRatio"] == "1:1"
    assert "gemini-test-key" not in str(client.payload)


def test_image_horde_uses_stable_diffusion(client):
    response = client.post(
        "/api/v1/images",
        json={"type": "horde", "model": "auto", "prompt": "A red bicycle", "size": "512x512", "app": "image-app"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["type"] == "horde"
    assert body["provider"] == "horde"
    assert body["model"] == "stable_diffusion"
    assert body["images"][0]["b64_json"]


def test_image_horde_rejects_size_not_multiple_of_64(client):
    response = client.post(
        "/api/v1/images",
        json={"type": "horde", "prompt": "A red bicycle", "size": "1000x1000"},
    )
    assert response.status_code == 400


def test_missing_horde_key_returns_503(client, monkeypatch):
    monkeypatch.setenv("LLM_MODE", "live")
    response = client.post(
        "/api/v1/images",
        json={"type": "horde", "model": "auto", "prompt": "A red bicycle", "size": "512x512"},
    )
    assert response.status_code == 503
    assert "AIHORDE_API_KEY" in response.json()["error"]["message"]


def test_horde_image_posts_then_polls(monkeypatch):
    from app.images import _generate_live

    class HordeClient:
        def __init__(self):
            self.url = None
            self.headers = None
            self.payload = None
            self.gets = []

        async def post(self, url, headers, json):
            self.url = url
            self.headers = headers
            self.payload = json
            return FakeResponse(202, {"id": "job-1", "kudos": 8})

        async def get(self, url, headers):
            self.gets.append(url)
            if url.endswith("/check/job-1"):
                return FakeResponse(200, {"done": True, "faulted": False})
            return FakeResponse(200, {"done": True, "generations": [{"img": "https://example.com/bike.webp"}]})

    async def instant(_seconds):
        return None

    monkeypatch.setattr("app.images.asyncio.sleep", instant)
    client = HordeClient()
    settings = Settings(
        aihorde_api_key="horde-test-key",
        aihorde_base_url="https://aihorde.net/api/v2",
        aihorde_image_model="stable_diffusion",
        aihorde_wait_seconds=30,
    )

    async def scenario():
        return await _generate_live(
            settings, client, "horde", "stable_diffusion", "A red bicycle", "512x512", 1
        )

    images = asyncio.run(scenario())
    assert images[0].url == "https://example.com/bike.webp"
    assert client.url == "https://aihorde.net/api/v2/generate/async"
    assert client.headers["apikey"] == "horde-test-key"
    assert client.headers["Client-Agent"] == "llm-middleware:1.0:local"
    assert client.payload["models"] == ["stable_diffusion"]
    assert client.payload["params"]["width"] == 512
    assert client.gets == [
        "https://aihorde.net/api/v2/generate/check/job-1",
        "https://aihorde.net/api/v2/generate/status/job-1",
    ]
    assert "horde-test-key" not in str(client.payload)


def test_free_video_quota_message_asks_for_a_token():
    from app.videos import _free_error

    message = _free_error("You have exceeded your ZeroGPU quota (120s requested vs. -60s left).", has_token=False)
    assert "HUGGINGFACE_TOKEN" in message
    assert "hf_" not in message
    assert "quota" not in _free_error("null", has_token=True).lower()


def test_free_video_sends_huggingface_token(monkeypatch):
    from app.videos import VideoJob, _JOBS, _run_free

    seen = {}

    class FakeStream:
        status_code = 200

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def aiter_lines(self):
            yield 'data: {"msg":"process_completed","output":{"error":"You have exceeded your ZeroGPU quota"}}'

    class FakeClient:
        def __init__(self, *args, **kwargs):
            return None

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def post(self, url, json=None, headers=None):
            seen["headers"] = headers
            seen["url"] = url
            return FakeResponse(200, {})

        def stream(self, method, url, params=None, headers=None):
            seen["stream_headers"] = headers
            return FakeStream()

    monkeypatch.setenv("HUGGINGFACE_TOKEN", "hf-test-token")
    monkeypatch.setattr("app.videos.httpx.AsyncClient", FakeClient)
    job = VideoJob(
        id="mw_vid_token",
        created=1,
        app="video-app",
        type="free",
        provider="free",
        model="Lightricks/ltx-video-distilled",
        prompt="A red bicycle rolling",
        status="queued",
        mock=False,
    )
    _JOBS[job.id] = job
    asyncio.run(_run_free(job.id))
    assert seen["headers"]["Authorization"] == "Bearer hf-test-token"
    assert seen["stream_headers"]["Authorization"] == "Bearer hf-test-token"
    assert "hf-test-token" not in (job.error or "")
    assert job.status == "failed"
    _JOBS.pop(job.id, None)


def test_extract_video_reads_b64_and_url():
    from app.videos import _extract_video

    url, raw = _extract_video({"b64_video": "abc123"})
    assert url is None
    assert raw == "abc123"
    url, raw = _extract_video({"video_url": "https://cdn.example/clip.mp4"})
    assert url == "https://cdn.example/clip.mp4"
    assert raw is None
    url, raw = _extract_video({"video": {"url": "https://cdn.example/seedance.mp4"}})
    assert url == "https://cdn.example/seedance.mp4"


def test_higgsfield_submit_status_and_cancel(monkeypatch):
    from app.schemas import VideoRequest
    from app.videos import cancel_video, create_video, refresh_video

    class HiggsfieldClient:
        def __init__(self):
            self.posts = []
            self.headers = None
            self.payload = None

        async def post(self, url, headers, json=None):
            self.posts.append(url)
            self.headers = headers
            self.payload = json
            if isinstance(json, dict) and "prompt" in json:
                self.generation = json
            if str(url).endswith("/cancel"):
                return FakeResponse(202, {})
            return FakeResponse(
                200,
                {
                    "status": "queued",
                    "request_id": "req-1",
                    "status_url": "https://api.higgsfield.ai/requests/req-1/status",
                    "cancel_url": "https://api.higgsfield.ai/requests/req-1/cancel",
                },
            )

        async def get(self, url, headers):
            return FakeResponse(
                200,
                {"status": "completed", "request_id": "req-1", "video": {"url": "https://cdn.example/clip.mp4"}},
            )

    monkeypatch.setenv("LLM_MODE", "live")
    monkeypatch.setenv("HF_CREDENTIALS", "key-id:key-secret")
    client = HiggsfieldClient()

    async def scenario():
        started = await create_video(
            VideoRequest(type="higgsfield", prompt="A cinematic scene at sunset", duration=5),
            "video-app",
            client,
        )
        finished = await refresh_video(started, client)
        queued = await create_video(
            VideoRequest(type="higgsfield", prompt="Another scene", duration=5),
            "video-app",
            client,
        )
        canceled = await cancel_video(queued.id, client)
        return finished, canceled

    finished, canceled = asyncio.run(scenario())
    assert finished.id == "req-1"
    assert finished.status == "completed"
    assert finished.remote_url == "https://cdn.example/clip.mp4"
    assert canceled.status == "canceled"
    assert client.headers["Authorization"] == "Key key-id:key-secret"
    assert client.generation["prompt"] == "Another scene"
    assert client.generation["duration"] == 5
    assert client.generation["resolution"] == "720p"
    assert client.posts[-1].endswith("/cancel")
    assert "key-secret" not in str(client.generation)
