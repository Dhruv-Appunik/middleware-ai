import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass

import httpx

from app.config import Settings


class BadRequest(Exception):
    def __init__(self, message: str):
        self.message = message
        super().__init__(message)


class ProviderError(Exception):
    def __init__(self, message: str, status: int = 502, provider: str = ""):
        self.message = message
        self.status = status
        self.provider = provider
        super().__init__(message)


@dataclass
class ProviderResult:
    text: str
    model: str
    input_tokens: int | None
    output_tokens: int | None
    finish_reason: str = "stop"


@dataclass
class StreamEvent:
    text: str = ""
    model: str = ""
    finish_reason: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None


def _error_text(response: httpx.Response) -> str:
    try:
        data = response.json()
    except Exception:
        text = (response.text or "").strip()
        return text[:500] or f"Provider returned HTTP {response.status_code}"
    if isinstance(data, dict):
        err = data.get("error")
        if isinstance(err, dict) and err.get("message"):
            return str(err["message"])
        if isinstance(err, str):
            return err
        if data.get("detail"):
            return str(data["detail"])
        if data.get("message"):
            return str(data["message"])
    return f"Provider returned HTTP {response.status_code}"


def _raise_for_status(response: httpx.Response, provider: str, key_name: str) -> None:
    if response.status_code < 400:
        return
    if response.status_code in (401, 403):
        raise ProviderError(
            f"The provider rejected the API key. Check {key_name} in the .env file.",
            status=502,
            provider=provider,
        )
    status = response.status_code if response.status_code < 500 else 502
    raise ProviderError(_error_text(response), status=status, provider=provider)


def prepare_claude_messages(messages: list[dict]) -> list[dict]:
    """Claude requires alternating user and assistant turns, starting with user."""
    merged: list[dict] = []
    for message in messages:
        if merged and merged[-1]["role"] == message["role"]:
            merged[-1] = {
                "role": message["role"],
                "content": merged[-1]["content"] + "\n\n" + message["content"],
            }
        else:
            merged.append({"role": message["role"], "content": message["content"]})
    if merged and merged[0]["role"] == "assistant":
        merged.insert(0, {"role": "user", "content": "Continue the conversation."})
    return merged


class ClaudeProvider:
    name = "claude"

    def __init__(self, settings: Settings, client: httpx.AsyncClient | None = None):
        self.settings = settings
        self.client = client

    async def complete(
        self,
        model: str,
        system: str,
        messages: list[dict],
        temperature: float | None,
        max_tokens: int,
        top_p: float | None,
    ) -> ProviderResult:
        if not self.settings.anthropic_api_key.strip():
            raise ProviderError(
                "ANTHROPIC_API_KEY is not set. Add it to the .env file to use type claude.",
                status=503,
                provider=self.name,
            )
        payload: dict = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": prepare_claude_messages(messages),
        }
        if system:
            payload["system"] = system
        if temperature is not None:
            payload["temperature"] = min(temperature, 1)
        elif top_p is not None:
            payload["top_p"] = top_p

        data = await self._post(payload)
        parts: list[str] = []
        for block in data.get("content") or []:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text") or ""))
        text = "".join(parts).strip()
        if not text:
            raise ProviderError("Claude returned an empty answer.", status=502, provider=self.name)
        usage = data.get("usage") or {}
        return ProviderResult(
            text=text,
            model=str(data.get("model") or model),
            input_tokens=_as_int(usage.get("input_tokens")),
            output_tokens=_as_int(usage.get("output_tokens")),
            finish_reason=str(data.get("stop_reason") or "stop"),
        )

    async def _post(self, payload: dict) -> dict:
        headers = {
            "x-api-key": self.settings.anthropic_api_key.strip(),
            "anthropic-version": "2023-06-01",
        }
        url = self.settings.anthropic_base_url.rstrip("/") + "/v1/messages"
        return await _send(self.client, self.settings, url, headers, payload, self.name, "ANTHROPIC_API_KEY")

    async def stream(
        self,
        model: str,
        system: str,
        messages: list[dict],
        temperature: float | None,
        max_tokens: int,
        top_p: float | None,
    ) -> AsyncIterator[StreamEvent]:
        if not self.settings.anthropic_api_key.strip():
            raise ProviderError(
                "ANTHROPIC_API_KEY is not set. Add it to the .env file to use type claude.",
                status=503,
                provider=self.name,
            )
        payload: dict = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": prepare_claude_messages(messages),
            "stream": True,
        }
        if system:
            payload["system"] = system
        if temperature is not None:
            payload["temperature"] = min(temperature, 1)
        elif top_p is not None:
            payload["top_p"] = top_p
        headers = {
            "x-api-key": self.settings.anthropic_api_key.strip(),
            "anthropic-version": "2023-06-01",
            "accept": "text/event-stream",
        }
        url = self.settings.anthropic_base_url.rstrip("/") + "/v1/messages"
        async for event in _iter_claude_stream(
            self.client, self.settings, url, headers, payload, self.name, "ANTHROPIC_API_KEY"
        ):
            yield event


class OpenAIProvider:
    name = "openai"

    def __init__(self, settings: Settings, client: httpx.AsyncClient | None = None):
        self.settings = settings
        self.client = client

    async def complete(
        self,
        model: str,
        system: str,
        messages: list[dict],
        temperature: float | None,
        max_tokens: int,
        top_p: float | None,
    ) -> ProviderResult:
        if not self.settings.openai_api_key.strip():
            raise ProviderError(
                "OPENAI_API_KEY is not set. Add it to the .env file to use type chatgpt.",
                status=503,
                provider=self.name,
            )
        chat_messages: list[dict] = []
        if system:
            chat_messages.append({"role": "system", "content": system})
        chat_messages.extend(messages)
        payload: dict = {
            "model": model,
            "messages": chat_messages,
            "max_tokens": max_tokens,
        }
        if temperature is not None:
            payload["temperature"] = temperature
        if top_p is not None:
            payload["top_p"] = top_p

        data = await self._post(payload)
        choices = data.get("choices") or []
        if not choices:
            raise ProviderError("ChatGPT returned no choices.", status=502, provider=self.name)
        message = choices[0].get("message") or {}
        text = _content_text(message.get("content")).strip()
        if not text:
            raise ProviderError("ChatGPT returned an empty answer.", status=502, provider=self.name)
        usage = data.get("usage") or {}
        return ProviderResult(
            text=text,
            model=str(data.get("model") or model),
            input_tokens=_as_int(usage.get("prompt_tokens")),
            output_tokens=_as_int(usage.get("completion_tokens")),
            finish_reason=str(choices[0].get("finish_reason") or "stop"),
        )

    async def _post(self, payload: dict) -> dict:
        headers = {"Authorization": f"Bearer {self.settings.openai_api_key.strip()}"}
        url = self.settings.openai_base_url.rstrip("/") + "/chat/completions"
        return await _send(self.client, self.settings, url, headers, payload, self.name, "OPENAI_API_KEY")

    async def stream(
        self,
        model: str,
        system: str,
        messages: list[dict],
        temperature: float | None,
        max_tokens: int,
        top_p: float | None,
    ) -> AsyncIterator[StreamEvent]:
        if not self.settings.openai_api_key.strip():
            raise ProviderError(
                "OPENAI_API_KEY is not set. Add it to the .env file to use type chatgpt.",
                status=503,
                provider=self.name,
            )
        payload = _openai_payload(model, system, messages, temperature, max_tokens, top_p, token_field="max_tokens")
        headers = {"Authorization": f"Bearer {self.settings.openai_api_key.strip()}"}
        url = self.settings.openai_base_url.rstrip("/") + "/chat/completions"
        async for event in _iter_openai_stream(
            self.client, self.settings, url, headers, payload, self.name, "OPENAI_API_KEY"
        ):
            yield event


class GrokProvider:
    name = "grok"

    def __init__(self, settings: Settings, client: httpx.AsyncClient | None = None):
        self.settings = settings
        self.client = client

    async def complete(
        self,
        model: str,
        system: str,
        messages: list[dict],
        temperature: float | None,
        max_tokens: int,
        top_p: float | None,
    ) -> ProviderResult:
        if not self.settings.xai_api_key.strip():
            raise ProviderError(
                "XAI_API_KEY is not set. Add it to the .env file to use type grok.",
                status=503,
                provider=self.name,
            )
        chat_messages: list[dict] = []
        if system:
            chat_messages.append({"role": "system", "content": system})
        chat_messages.extend(messages)
        payload: dict = {
            "model": model,
            "messages": chat_messages,
            "max_completion_tokens": max_tokens,
        }
        if temperature is not None:
            payload["temperature"] = temperature
        if top_p is not None:
            payload["top_p"] = top_p

        headers = {"Authorization": f"Bearer {self.settings.xai_api_key.strip()}"}
        url = self.settings.xai_base_url.rstrip("/") + "/chat/completions"
        data = await _send(self.client, self.settings, url, headers, payload, self.name, "XAI_API_KEY")
        choices = data.get("choices") or []
        if not choices:
            raise ProviderError("Grok returned no choices.", status=502, provider=self.name)
        message = choices[0].get("message") or {}
        text = _content_text(message.get("content")).strip()
        if not text:
            raise ProviderError("Grok returned an empty answer.", status=502, provider=self.name)
        usage = data.get("usage") or {}
        return ProviderResult(
            text=text,
            model=str(data.get("model") or model),
            input_tokens=_as_int(usage.get("prompt_tokens")),
            output_tokens=_as_int(usage.get("completion_tokens")),
            finish_reason=str(choices[0].get("finish_reason") or "stop"),
        )

    async def stream(
        self,
        model: str,
        system: str,
        messages: list[dict],
        temperature: float | None,
        max_tokens: int,
        top_p: float | None,
    ) -> AsyncIterator[StreamEvent]:
        if not self.settings.xai_api_key.strip():
            raise ProviderError(
                "XAI_API_KEY is not set. Add it to the .env file to use type grok.",
                status=503,
                provider=self.name,
            )
        payload = _openai_payload(
            model, system, messages, temperature, max_tokens, top_p, token_field="max_completion_tokens"
        )
        headers = {"Authorization": f"Bearer {self.settings.xai_api_key.strip()}"}
        url = self.settings.xai_base_url.rstrip("/") + "/chat/completions"
        async for event in _iter_openai_stream(
            self.client, self.settings, url, headers, payload, self.name, "XAI_API_KEY"
        ):
            yield event


class NvidiaProvider:
    name = "nvidia"

    def __init__(self, settings: Settings, client: httpx.AsyncClient | None = None):
        self.settings = settings
        self.client = client

    async def complete(
        self,
        model: str,
        system: str,
        messages: list[dict],
        temperature: float | None,
        max_tokens: int,
        top_p: float | None,
    ) -> ProviderResult:
        if not self.settings.nvidia_api_key.strip():
            raise ProviderError(
                "NVIDIA_API_KEY is not set. Add it to the .env file to use type nvidia. "
                "Create a key at https://build.nvidia.com/settings/api-keys",
                status=503,
                provider=self.name,
            )
        chat_messages: list[dict] = []
        if system:
            chat_messages.append({"role": "system", "content": system})
        chat_messages.extend(messages)
        payload: dict = {
            "model": model,
            "messages": chat_messages,
            "max_tokens": max_tokens,
            "stream": False,
        }
        if temperature is not None:
            payload["temperature"] = temperature
        if top_p is not None:
            payload["top_p"] = top_p

        headers = {"Authorization": f"Bearer {self.settings.nvidia_api_key.strip()}"}
        url = self.settings.nvidia_base_url.rstrip("/") + "/chat/completions"
        data = await _send(self.client, self.settings, url, headers, payload, self.name, "NVIDIA_API_KEY")
        choices = data.get("choices") or []
        if not choices:
            raise ProviderError("NVIDIA NIM returned no choices.", status=502, provider=self.name)
        message = choices[0].get("message") or {}
        text = _content_text(message.get("content")).strip()
        if not text:
            text = _content_text(message.get("reasoning_content")).strip()
        if not text:
            raise ProviderError("NVIDIA NIM returned an empty answer.", status=502, provider=self.name)
        usage = data.get("usage") or {}
        return ProviderResult(
            text=text,
            model=str(data.get("model") or model),
            input_tokens=_as_int(usage.get("prompt_tokens")),
            output_tokens=_as_int(usage.get("completion_tokens")),
            finish_reason=str(choices[0].get("finish_reason") or "stop"),
        )

    async def stream(
        self,
        model: str,
        system: str,
        messages: list[dict],
        temperature: float | None,
        max_tokens: int,
        top_p: float | None,
    ) -> AsyncIterator[StreamEvent]:
        if not self.settings.nvidia_api_key.strip():
            raise ProviderError(
                "NVIDIA_API_KEY is not set. Add it to the .env file to use type nvidia. "
                "Create a key at https://build.nvidia.com/settings/api-keys",
                status=503,
                provider=self.name,
            )
        payload = _openai_payload(model, system, messages, temperature, max_tokens, top_p, token_field="max_tokens")
        headers = {"Authorization": f"Bearer {self.settings.nvidia_api_key.strip()}"}
        url = self.settings.nvidia_base_url.rstrip("/") + "/chat/completions"
        async for event in _iter_openai_stream(
            self.client, self.settings, url, headers, payload, self.name, "NVIDIA_API_KEY"
        ):
            yield event


class GroqProvider:
    name = "groq"

    def __init__(self, settings: Settings, client: httpx.AsyncClient | None = None):
        self.settings = settings
        self.client = client

    async def complete(
        self,
        model: str,
        system: str,
        messages: list[dict],
        temperature: float | None,
        max_tokens: int,
        top_p: float | None,
    ) -> ProviderResult:
        if not self.settings.groq_api_key.strip():
            raise ProviderError(
                "GROQ_API_KEY is not set. Add it to the .env file to use type groq. "
                "Create a key at https://console.groq.com/keys",
                status=503,
                provider=self.name,
            )
        chat_messages: list[dict] = []
        if system:
            chat_messages.append({"role": "system", "content": system})
        chat_messages.extend(messages)
        payload: dict = {
            "model": model,
            "messages": chat_messages,
            "max_tokens": max_tokens,
            "stream": False,
        }
        if temperature is not None:
            payload["temperature"] = temperature
        if top_p is not None:
            payload["top_p"] = top_p

        headers = {"Authorization": f"Bearer {self.settings.groq_api_key.strip()}"}
        url = self.settings.groq_base_url.rstrip("/") + "/chat/completions"
        data = await _send(self.client, self.settings, url, headers, payload, self.name, "GROQ_API_KEY")
        choices = data.get("choices") or []
        if not choices:
            raise ProviderError("Groq returned no choices.", status=502, provider=self.name)
        message = choices[0].get("message") or {}
        text = _content_text(message.get("content")).strip()
        if not text:
            raise ProviderError("Groq returned an empty answer.", status=502, provider=self.name)
        usage = data.get("usage") or {}
        return ProviderResult(
            text=text,
            model=str(data.get("model") or model),
            input_tokens=_as_int(usage.get("prompt_tokens")),
            output_tokens=_as_int(usage.get("completion_tokens")),
            finish_reason=str(choices[0].get("finish_reason") or "stop"),
        )

    async def stream(
        self,
        model: str,
        system: str,
        messages: list[dict],
        temperature: float | None,
        max_tokens: int,
        top_p: float | None,
    ) -> AsyncIterator[StreamEvent]:
        if not self.settings.groq_api_key.strip():
            raise ProviderError(
                "GROQ_API_KEY is not set. Add it to the .env file to use type groq. "
                "Create a key at https://console.groq.com/keys",
                status=503,
                provider=self.name,
            )
        payload = _openai_payload(model, system, messages, temperature, max_tokens, top_p, token_field="max_tokens")
        headers = {"Authorization": f"Bearer {self.settings.groq_api_key.strip()}"}
        url = self.settings.groq_base_url.rstrip("/") + "/chat/completions"
        async for event in _iter_openai_stream(
            self.client, self.settings, url, headers, payload, self.name, "GROQ_API_KEY"
        ):
            yield event


class GeminiProvider:
    name = "gemini"

    def __init__(self, settings: Settings, client: httpx.AsyncClient | None = None):
        self.settings = settings
        self.client = client

    async def complete(
        self,
        model: str,
        system: str,
        messages: list[dict],
        temperature: float | None,
        max_tokens: int,
        top_p: float | None,
    ) -> ProviderResult:
        if not self.settings.gemini_api_key.strip():
            raise ProviderError(
                "GEMINI_API_KEY is not set. Add it to the .env file to use type gemini. "
                "Create a key at https://aistudio.google.com/apikey",
                status=503,
                provider=self.name,
            )
        chat_messages: list[dict] = []
        if system:
            chat_messages.append({"role": "system", "content": system})
        chat_messages.extend(messages)
        payload: dict = {
            "model": model,
            "messages": chat_messages,
            "max_tokens": max_tokens,
            "stream": False,
        }
        if temperature is not None:
            payload["temperature"] = temperature
        if top_p is not None:
            payload["top_p"] = top_p
        headers = {"Authorization": f"Bearer {self.settings.gemini_api_key.strip()}"}
        url = self.settings.gemini_base_url.rstrip("/") + "/chat/completions"
        data = await _send(self.client, self.settings, url, headers, payload, self.name, "GEMINI_API_KEY")
        choices = data.get("choices") or []
        if not choices:
            raise ProviderError("Gemini returned no choices.", status=502, provider=self.name)
        message = choices[0].get("message") or {}
        text = _content_text(message.get("content")).strip()
        if not text:
            raise ProviderError("Gemini returned an empty answer.", status=502, provider=self.name)
        usage = data.get("usage") or {}
        return ProviderResult(
            text=text,
            model=str(data.get("model") or model),
            input_tokens=_as_int(usage.get("prompt_tokens")),
            output_tokens=_as_int(usage.get("completion_tokens")),
            finish_reason=str(choices[0].get("finish_reason") or "stop"),
        )

    async def stream(
        self,
        model: str,
        system: str,
        messages: list[dict],
        temperature: float | None,
        max_tokens: int,
        top_p: float | None,
    ) -> AsyncIterator[StreamEvent]:
        if not self.settings.gemini_api_key.strip():
            raise ProviderError(
                "GEMINI_API_KEY is not set. Add it to the .env file to use type gemini. "
                "Create a key at https://aistudio.google.com/apikey",
                status=503,
                provider=self.name,
            )
        payload = _openai_payload(model, system, messages, temperature, max_tokens, top_p, token_field="max_tokens")
        headers = {"Authorization": f"Bearer {self.settings.gemini_api_key.strip()}"}
        url = self.settings.gemini_base_url.rstrip("/") + "/chat/completions"
        async for event in _iter_openai_stream(
            self.client, self.settings, url, headers, payload, self.name, "GEMINI_API_KEY"
        ):
            yield event


class OllamaProvider:
    name = "ollama"

    def __init__(self, settings: Settings, client: httpx.AsyncClient | None = None):
        self.settings = settings
        self.client = client

    async def complete(
        self,
        model: str,
        system: str,
        messages: list[dict],
        temperature: float | None,
        max_tokens: int,
        top_p: float | None,
    ) -> ProviderResult:
        chat_messages: list[dict] = []
        if system:
            chat_messages.append({"role": "system", "content": system})
        chat_messages.extend(messages)
        payload: dict = {
            "model": model,
            "messages": chat_messages,
            "max_tokens": max_tokens,
            "stream": False,
        }
        if temperature is not None:
            payload["temperature"] = temperature
        if top_p is not None:
            payload["top_p"] = top_p

        api_key = self.settings.ollama_api_key.strip()
        base = self.settings.ollama_base_url.rstrip("/")
        cloud = "ollama.com" in base.lower()
        if cloud and not api_key:
            raise ProviderError(
                "OLLAMA_API_KEY is not set. Add it to the .env file to use online Ollama. "
                "Create a key at https://ollama.com/settings/keys",
                status=503,
                provider=self.name,
            )
        headers = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        if not base.endswith("/v1"):
            base += "/v1"
        url = base + "/chat/completions"
        data = await _send(
            self.client,
            self.settings,
            url,
            headers,
            payload,
            self.name,
            "OLLAMA_API_KEY",
            unreachable_message=(
                "Could not reach Ollama Cloud."
                if cloud
                else "Ollama is not running. Start it with ollama serve, then pull a model with ollama pull llama3.2."
            ),
        )
        choices = data.get("choices") or []
        if not choices:
            raise ProviderError("Ollama returned no choices.", status=502, provider=self.name)
        message = choices[0].get("message") or {}
        text = _content_text(message.get("content")).strip()
        if not text:
            raise ProviderError("Ollama returned an empty answer.", status=502, provider=self.name)
        usage = data.get("usage") or {}
        return ProviderResult(
            text=text,
            model=str(data.get("model") or model),
            input_tokens=_as_int(usage.get("prompt_tokens")),
            output_tokens=_as_int(usage.get("completion_tokens")),
            finish_reason=str(choices[0].get("finish_reason") or "stop"),
        )

    async def stream(
        self,
        model: str,
        system: str,
        messages: list[dict],
        temperature: float | None,
        max_tokens: int,
        top_p: float | None,
    ) -> AsyncIterator[StreamEvent]:
        api_key = self.settings.ollama_api_key.strip()
        base = self.settings.ollama_base_url.rstrip("/")
        cloud = "ollama.com" in base.lower()
        if cloud and not api_key:
            raise ProviderError(
                "OLLAMA_API_KEY is not set. Add it to the .env file to use online Ollama. "
                "Create a key at https://ollama.com/settings/keys",
                status=503,
                provider=self.name,
            )
        payload = _openai_payload(model, system, messages, temperature, max_tokens, top_p, token_field="max_tokens")
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        if not base.endswith("/v1"):
            base += "/v1"
        url = base + "/chat/completions"
        async for event in _iter_openai_stream(
            self.client,
            self.settings,
            url,
            headers,
            payload,
            self.name,
            "OLLAMA_API_KEY",
            unreachable_message=(
                "Could not reach Ollama Cloud."
                if cloud
                else "Ollama is not running. Start it with ollama serve, then pull a model with ollama pull llama3.2."
            ),
        ):
            yield event


async def _send(
    client: httpx.AsyncClient | None,
    settings: Settings,
    url: str,
    headers: dict,
    payload: dict,
    provider: str,
    key_name: str,
    unreachable_message: str | None = None,
) -> dict:
    owns_client = client is None
    http = client or httpx.AsyncClient(timeout=settings.request_timeout_seconds)
    try:
        try:
            response = await http.post(url, headers=headers, json=payload)
        except httpx.TimeoutException as exc:
            raise ProviderError(
                "The model provider took too long to answer.",
                status=504,
                provider=provider,
            ) from exc
        except httpx.HTTPError as exc:
            raise ProviderError(
                unreachable_message or "Could not reach the model provider.",
                status=502,
                provider=provider,
            ) from exc
        _raise_for_status(response, provider, key_name)
        try:
            data = response.json()
        except Exception as exc:
            raise ProviderError("Provider returned invalid JSON.", status=502, provider=provider) from exc
        if not isinstance(data, dict):
            raise ProviderError("Provider returned an unexpected response.", status=502, provider=provider)
        return data
    finally:
        if owns_client:
            await http.aclose()


def _content_text(content) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks: list[str] = []
        for part in content:
            if isinstance(part, dict):
                chunks.append(str(part.get("text") or ""))
            else:
                chunks.append(str(part))
        return "".join(chunks)
    return str(content)


def _as_int(value) -> int | None:
    return value if isinstance(value, int) else None


def _openai_payload(
    model: str,
    system: str,
    messages: list[dict],
    temperature: float | None,
    max_tokens: int,
    top_p: float | None,
    token_field: str,
) -> dict:
    chat_messages: list[dict] = []
    if system:
        chat_messages.append({"role": "system", "content": system})
    chat_messages.extend(messages)
    payload: dict = {
        "model": model,
        "messages": chat_messages,
        token_field: max_tokens,
        "stream": True,
    }
    if temperature is not None:
        payload["temperature"] = temperature
    if top_p is not None:
        payload["top_p"] = top_p
    return payload


@asynccontextmanager
async def _open_stream(
    client: httpx.AsyncClient | None,
    settings: Settings,
    url: str,
    headers: dict,
    payload: dict,
    provider: str,
    key_name: str,
    unreachable_message: str | None = None,
):
    owns_client = client is None
    http = client or httpx.AsyncClient(timeout=settings.request_timeout_seconds)
    try:
        try:
            async with http.stream("POST", url, headers=headers, json=payload) as response:
                if response.status_code >= 400:
                    raw = await response.aread()
                    failed = httpx.Response(status_code=response.status_code, content=raw)
                    _raise_for_status(failed, provider, key_name)
                yield response
        except ProviderError:
            raise
        except httpx.TimeoutException as exc:
            raise ProviderError(
                "The model provider took too long to answer.",
                status=504,
                provider=provider,
            ) from exc
        except httpx.HTTPError as exc:
            raise ProviderError(
                unreachable_message or "Could not reach the model provider.",
                status=502,
                provider=provider,
            ) from exc
    finally:
        if owns_client:
            await http.aclose()


async def _iter_sse_data(response) -> AsyncIterator[str]:
    async for raw in response.aiter_lines():
        line = raw.strip()
        if not line or line.startswith(":") or line.startswith("event:"):
            continue
        if line.startswith("data:"):
            yield line[5:].strip()


def _error_from_payload(data: dict, provider: str) -> ProviderError | None:
    err = data.get("error")
    if isinstance(err, dict) and err.get("message"):
        return ProviderError(str(err["message"]), status=502, provider=provider)
    if isinstance(err, str) and err.strip():
        return ProviderError(err, status=502, provider=provider)
    return None


def _openai_piece(data: dict) -> StreamEvent | None:
    usage = data.get("usage") or {}
    choices = data.get("choices") or []
    text = ""
    finish = None
    if choices and isinstance(choices[0], dict):
        delta = choices[0].get("delta") or {}
        text = _content_text(delta.get("content"))
        if not text:
            text = _content_text(delta.get("reasoning_content"))
        finish_raw = choices[0].get("finish_reason")
        finish = str(finish_raw) if finish_raw else None
    model = str(data.get("model") or "")
    input_tokens = _as_int(usage.get("prompt_tokens"))
    output_tokens = _as_int(usage.get("completion_tokens"))
    if not text and not finish and not model and input_tokens is None and output_tokens is None:
        return None
    return StreamEvent(
        text=text,
        model=model,
        finish_reason=finish,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )


async def _iter_openai_stream(
    client,
    settings: Settings,
    url: str,
    headers: dict,
    payload: dict,
    provider: str,
    key_name: str,
    unreachable_message: str | None = None,
) -> AsyncIterator[StreamEvent]:
    async with _open_stream(
        client, settings, url, headers, payload, provider, key_name, unreachable_message
    ) as response:
        async for data in _iter_sse_data(response):
            if data == "[DONE]":
                break
            try:
                parsed = json.loads(data)
            except json.JSONDecodeError:
                continue
            if not isinstance(parsed, dict):
                continue
            error = _error_from_payload(parsed, provider)
            if error:
                raise error
            piece = _openai_piece(parsed)
            if piece is not None:
                yield piece


def _claude_piece(data: dict) -> StreamEvent | None:
    kind = data.get("type")
    if kind == "error":
        err = data.get("error") or {}
        message = err.get("message") if isinstance(err, dict) else err
        raise ProviderError(str(message or "Claude stream failed."), status=502, provider="claude")
    if kind == "message_start":
        message = data.get("message") or {}
        usage = message.get("usage") or {}
        return StreamEvent(
            model=str(message.get("model") or ""),
            input_tokens=_as_int(usage.get("input_tokens")),
        )
    if kind == "content_block_delta":
        delta = data.get("delta") or {}
        if delta.get("type") == "text_delta":
            return StreamEvent(text=str(delta.get("text") or ""))
        return None
    if kind == "message_delta":
        delta = data.get("delta") or {}
        usage = data.get("usage") or {}
        finish = delta.get("stop_reason")
        return StreamEvent(
            finish_reason=str(finish) if finish else None,
            output_tokens=_as_int(usage.get("output_tokens")),
        )
    return None


async def _iter_claude_stream(
    client,
    settings: Settings,
    url: str,
    headers: dict,
    payload: dict,
    provider: str,
    key_name: str,
) -> AsyncIterator[StreamEvent]:
    async with _open_stream(client, settings, url, headers, payload, provider, key_name) as response:
        async for data in _iter_sse_data(response):
            if data == "[DONE]":
                break
            try:
                parsed = json.loads(data)
            except json.JSONDecodeError:
                continue
            if not isinstance(parsed, dict):
                continue
            piece = _claude_piece(parsed)
            if piece is not None:
                yield piece
