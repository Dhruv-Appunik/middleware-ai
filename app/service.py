import json
import re
import time
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass

from app.config import Settings, get_settings
from app.providers import (
    BadRequest,
    ClaudeProvider,
    GrokProvider,
    GroqProvider,
    GeminiProvider,
    NvidiaProvider,
    OllamaProvider,
    OpenAIProvider,
    ProviderError,
    ProviderResult,
    StreamEvent,
)
from app.schemas import ChatRequest, ChatResponse, Usage

_CLAUDE_ALIASES = {"claude", "clude", "claud", "cluade", "anthropic"}
_CHATGPT_ALIASES = {"chatgpt", "gpt"}
_OPENAI_ALIASES = {"openai"}
_GROK_ALIASES = {"grok", "grouk", "xai"}
_GROQ_ALIASES = {"groq"}
_GEMINI_ALIASES = {"gemini", "google"}
_OLLAMA_ALIASES = {"ollama"}
_NVIDIA_ALIASES = {"nvidia", "nim", "nvidianim"}


@dataclass
class Resolved:
    provider: str
    type_name: str
    model: str
    requested_type: str
    requested_model: str


def _token(value: str) -> str:
    return re.sub(r"[\s_\-]+", "", value.strip().lower())


def _prefix_provider(prefix: str) -> str | None:
    token = _token(prefix)
    if token in _CLAUDE_ALIASES:
        return "claude"
    if token in _CHATGPT_ALIASES or token in _OPENAI_ALIASES:
        return "openai"
    if token in _GROK_ALIASES:
        return "grok"
    if token in _GROQ_ALIASES:
        return "groq"
    if token in _GEMINI_ALIASES:
        return "gemini"
    if token in _OLLAMA_ALIASES:
        return "ollama"
    return None


def _infer_provider(model_name: str) -> str | None:
    token = _token(model_name)
    if token in {"", "auto"}:
        return None
    if token.startswith("claude") or token.startswith("anthropic"):
        return "claude"
    if token.startswith(("gpt", "chatgpt", "o1", "o3", "o4")):
        return "openai"
    if token.startswith("groq"):
        return "groq"
    if token.startswith("gemini") or token.startswith("google"):
        return "gemini"
    if token.startswith(("grok", "grouk")):
        return "grok"
    if token.startswith("nvidia") or token.startswith("nim"):
        return "nvidia"
    if token.startswith("ollama"):
        return "ollama"
    return None


def _split_model(raw_model: str) -> tuple[str | None, str]:
    if "/" not in raw_model:
        return None, raw_model
    prefix, rest = raw_model.split("/", 1)
    provider = _prefix_provider(prefix)
    if provider is None:
        return None, raw_model
    return provider, rest.strip() or "auto"


def _provider_from_type(type_token: str, original_type: str) -> tuple[str, str, str | None]:
    """Return provider, public type name, and a model override when type is a model id."""
    if type_token in _CLAUDE_ALIASES:
        return "claude", "claude", None
    if type_token in _CHATGPT_ALIASES:
        return "openai", "chatgpt", None
    if type_token in _OPENAI_ALIASES:
        return "openai", "openai", None
    if type_token in _GROK_ALIASES:
        return "grok", "grok", None
    if type_token.startswith("claude") or type_token.startswith("anthropic"):
        return "claude", "claude", original_type.strip()
    if type_token.startswith(("gpt", "chatgpt", "o1", "o3", "o4")):
        return "openai", "chatgpt", original_type.strip()
    if type_token.startswith("groq"):
        return "groq", "groq", original_type.strip() if type_token not in _GROQ_ALIASES else None
    if type_token in _GEMINI_ALIASES:
        return "gemini", "gemini", None
    if type_token.startswith("gemini") or type_token.startswith("google"):
        return "gemini", "gemini", original_type.strip()
    if type_token.startswith(("grok", "grouk")):
        return "grok", "grok", original_type.strip()
    if type_token in _NVIDIA_ALIASES:
        return "nvidia", "nvidia", None
    if type_token.startswith("nvidia") or type_token.startswith("nim"):
        return "nvidia", "nvidia", original_type.strip()
    if type_token in _OLLAMA_ALIASES:
        return "ollama", "ollama", None
    if type_token.startswith("ollama"):
        return "ollama", "ollama", original_type.strip()
    raise BadRequest('type must be "auto", "claude", "chatgpt", "grok", "groq", "nvidia", "ollama", or "gemini".')


def resolve(
    req: ChatRequest,
    settings: Settings,
    fallback_provider: str | None = None,
) -> Resolved:
    requested_type = (req.type or "auto").strip() or "auto"
    requested_model = (req.model or "auto").strip() or "auto"
    type_token = _token(requested_type)
    hint, model_name = _split_model(requested_model)

    model_override = None
    if type_token in {"", "auto"}:
        provider = hint or _infer_provider(model_name)
        if provider is None:
            fallback = _token(fallback_provider or settings.default_provider or "claude")
            if fallback in _OPENAI_ALIASES or fallback in _CHATGPT_ALIASES or fallback.startswith(("gpt", "o1", "o3", "o4")):
                provider = "openai"
            elif fallback in _GROQ_ALIASES or fallback.startswith("groq"):
                provider = "groq"
            elif fallback in _GROK_ALIASES or fallback.startswith(("grok", "grouk")):
                provider = "grok"
            elif fallback in _NVIDIA_ALIASES or fallback.startswith(("nvidia", "nim")):
                provider = "nvidia"
            elif fallback in _OLLAMA_ALIASES or fallback.startswith("ollama"):
                provider = "ollama"
            elif fallback in _GEMINI_ALIASES or fallback.startswith(("gemini", "google")):
                provider = "gemini"
            else:
                provider = "claude"
        if provider == "claude":
            type_name = "claude"
        elif provider == "groq":
            type_name = "groq"
        elif provider == "grok":
            type_name = "grok"
        elif provider == "nvidia":
            type_name = "nvidia"
        elif provider == "ollama":
            type_name = "ollama"
        elif provider == "gemini":
            type_name = "gemini"
        else:
            type_name = "chatgpt"
    else:
        provider, type_name, model_override = _provider_from_type(type_token, requested_type)

    if hint and hint != provider:
        model_name = requested_model

    if model_override and _token(model_name) in {"", "auto"}:
        model_name = model_override

    if _token(model_name) in {"", "auto"}:
        if provider == "claude":
            model_name = settings.claude_default_model
        elif provider == "groq":
            model_name = settings.groq_default_model
        elif provider == "grok":
            model_name = settings.grok_default_model
        elif provider == "nvidia":
            model_name = settings.nvidia_default_model
        elif provider == "ollama":
            model_name = settings.ollama_default_model
        elif provider == "gemini":
            model_name = settings.gemini_default_model
        else:
            model_name = settings.openai_default_model

    return Resolved(
        provider=provider,
        type_name=type_name,
        model=model_name,
        requested_type=requested_type,
        requested_model=requested_model,
    )


def build_prompt(req: ChatRequest) -> tuple[str, list[dict], str]:
    system_parts: list[str] = []
    messages: list[dict] = []
    if req.context.strip():
        system_parts.append(req.context.strip())

    for message in req.messages:
        content = message.content.strip()
        if not content:
            continue
        role = message.normalized_role()
        if role == "system":
            system_parts.append(content)
            continue
        if role not in {"user", "assistant"}:
            raise BadRequest('Each message role must be "system", "user", or "assistant".')
        messages.append({"role": role, "content": content})

    query = req.query.strip()
    if query and not (
        messages and messages[-1]["role"] == "user" and messages[-1]["content"] == query
    ):
        messages.append({"role": "user", "content": query})

    if not any(message["role"] == "user" for message in messages):
        raise BadRequest("Send a query, or at least one user message.")

    echo_query = query
    if not echo_query:
        for message in reversed(messages):
            if message["role"] == "user":
                echo_query = message["content"]
                break
    return "\n\n".join(system_parts), messages, echo_query


def _mock_result(resolved: Resolved, system: str, messages: list[dict], query: str, app_name: str) -> ProviderResult:
    answer = (
        f"[mock] app={app_name} type={resolved.type_name} "
        f"provider={resolved.provider} model={resolved.model} query={query}"
    )
    source = system + "".join(message["content"] for message in messages)
    input_tokens = max(1, len(source) // 4)
    output_tokens = max(1, len(answer) // 4)
    return ProviderResult(
        text=answer,
        model=resolved.model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        finish_reason="stop",
    )


def _provider(settings: Settings, name: str):
    if name == "claude":
        return ClaudeProvider(settings)
    if name == "grok":
        return GrokProvider(settings)
    if name == "nvidia":
        return NvidiaProvider(settings)
    if name == "groq":
        return GroqProvider(settings)
    if name == "ollama":
        return OllamaProvider(settings)
    if name == "gemini":
        return GeminiProvider(settings)
    return OpenAIProvider(settings)


def _usage(input_tokens: int | None, output_tokens: int | None) -> Usage:
    total = None
    if input_tokens is not None or output_tokens is not None:
        total = (input_tokens or 0) + (output_tokens or 0)
    return Usage(input_tokens=input_tokens, output_tokens=output_tokens, total_tokens=total)


async def handle_chat(
    req: ChatRequest,
    app_name: str,
    fallback_provider: str | None = None,
) -> ChatResponse:
    settings = get_settings()
    resolved = resolve(req, settings, fallback_provider=fallback_provider)
    system, messages, query = build_prompt(req)

    if settings.mock_mode:
        result = _mock_result(resolved, system, messages, query, app_name)
    else:
        result = await _provider(settings, resolved.provider).complete(
            model=resolved.model,
            system=system,
            messages=messages,
            temperature=req.temperature,
            max_tokens=req.max_tokens or 1024,
            top_p=req.top_p,
        )

    return ChatResponse(
        id="mw_" + uuid.uuid4().hex[:24],
        created=int(time.time()),
        app=app_name,
        type=resolved.type_name,
        provider=resolved.provider,
        model=result.model,
        requested_type=resolved.requested_type,
        requested_model=resolved.requested_model,
        query=query,
        context=system,
        answer=result.text,
        finish_reason=result.finish_reason,
        mock=settings.mock_mode,
        usage=_usage(result.input_tokens, result.output_tokens),
        metadata=req.metadata,
    )


@dataclass
class StreamSession:
    id: str
    created: int
    app: str
    type_name: str
    provider: str
    model: str
    requested_type: str
    requested_model: str
    query: str
    context: str
    mock: bool
    metadata: dict
    events: AsyncIterator[StreamEvent]


async def _mock_events(result: ProviderResult) -> AsyncIterator[StreamEvent]:
    words = result.text.split(" ")
    for index, word in enumerate(words):
        last = index == len(words) - 1
        yield StreamEvent(
            text=word if index == 0 else " " + word,
            model=result.model if index == 0 else "",
            finish_reason=result.finish_reason if last else None,
            input_tokens=result.input_tokens if last else None,
            output_tokens=result.output_tokens if last else None,
        )


async def _buffered_events(events: AsyncIterator[StreamEvent], provider: str) -> AsyncIterator[StreamEvent]:
    buffered: list[StreamEvent] = []
    text = ""
    try:
        async for event in events:
            buffered.append(event)
            text += event.text
            if text.strip():
                break
        else:
            raise ProviderError("The model returned an empty answer.", status=502, provider=provider)
    except Exception:
        await events.aclose()
        raise

    async def replay() -> AsyncIterator[StreamEvent]:
        try:
            for event in buffered:
                yield event
            async for event in events:
                yield event
        finally:
            await events.aclose()

    return replay()


async def open_chat_stream(
    req: ChatRequest,
    app_name: str,
    fallback_provider: str | None = None,
) -> StreamSession:
    settings = get_settings()
    resolved = resolve(req, settings, fallback_provider=fallback_provider)
    system, messages, query = build_prompt(req)
    if settings.mock_mode:
        source = _mock_events(_mock_result(resolved, system, messages, query, app_name))
    else:
        source = _provider(settings, resolved.provider).stream(
            model=resolved.model,
            system=system,
            messages=messages,
            temperature=req.temperature,
            max_tokens=req.max_tokens or 1024,
            top_p=req.top_p,
        )
    events = await _buffered_events(source, resolved.provider)
    return StreamSession(
        id="mw_" + uuid.uuid4().hex[:24],
        created=int(time.time()),
        app=app_name,
        type_name=resolved.type_name,
        provider=resolved.provider,
        model=resolved.model,
        requested_type=resolved.requested_type,
        requested_model=resolved.requested_model,
        query=query,
        context=system,
        mock=settings.mock_mode,
        metadata=req.metadata,
        events=events,
    )


def _sse(event: str, payload: dict) -> bytes:
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n".encode()


def _data(payload: dict) -> bytes:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n".encode()


async def middleware_events(session: StreamSession) -> AsyncIterator[bytes]:
    yield _sse(
        "start",
        {
            "id": session.id,
            "app": session.app,
            "type": session.type_name,
            "provider": session.provider,
            "model": session.model,
            "mock": session.mock,
        },
    )
    parts: list[str] = []
    finish = "stop"
    input_tokens = None
    output_tokens = None
    model = session.model
    async for event in session.events:
        if event.model:
            model = event.model
        if event.text:
            parts.append(event.text)
            yield _sse("delta", {"text": event.text})
        if event.finish_reason:
            finish = event.finish_reason
        if event.input_tokens is not None:
            input_tokens = event.input_tokens
        if event.output_tokens is not None:
            output_tokens = event.output_tokens
    yield _sse(
        "done",
        {
            "id": session.id,
            "created": session.created,
            "app": session.app,
            "type": session.type_name,
            "provider": session.provider,
            "model": model,
            "requested_type": session.requested_type,
            "requested_model": session.requested_model,
            "query": session.query,
            "context": session.context,
            "answer": "".join(parts),
            "finish_reason": finish,
            "mock": session.mock,
            "usage": _usage(input_tokens, output_tokens).model_dump(),
            "metadata": session.metadata,
        },
    )
    yield b"data: [DONE]\n\n"


async def openai_events(session: StreamSession) -> AsyncIterator[bytes]:
    completion_id = "chatcmpl_" + session.id.removeprefix("mw_")
    model = session.model
    yield _data(
        {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": session.created,
            "model": model,
            "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
        }
    )
    finish = "stop"
    input_tokens = None
    output_tokens = None
    async for event in session.events:
        if event.model:
            model = event.model
        if event.text:
            yield _data(
                {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": session.created,
                    "model": model,
                    "choices": [{"index": 0, "delta": {"content": event.text}, "finish_reason": None}],
                }
            )
        if event.finish_reason:
            finish = event.finish_reason
        if event.input_tokens is not None:
            input_tokens = event.input_tokens
        if event.output_tokens is not None:
            output_tokens = event.output_tokens
    usage = None
    if input_tokens is not None or output_tokens is not None:
        prompt_tokens = input_tokens or 0
        completion_tokens = output_tokens or 0
        usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }
    final = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": session.created,
        "model": model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": finish}],
    }
    if usage is not None:
        final["usage"] = usage
    yield _data(final)
    yield b"data: [DONE]\n\n"
