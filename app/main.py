import hmac
import logging
import re
import time
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Body, Depends, FastAPI, Request, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.security import APIKeyHeader, HTTPAuthorizationCredentials, HTTPBearer

from app.config import get_settings
from app.images import handle_image
from app.page import PAGE
from app.providers import BadRequest, ProviderError
from app.schemas import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
    ErrorResponse,
    ImageRequest,
    ImageResponse,
    OpenAIChatRequest,
    OpenAIChatResponse,
)
from app.service import handle_chat, middleware_events, openai_events, open_chat_stream

logger = logging.getLogger("llm_middleware")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
bearer_scheme = HTTPBearer(auto_error=False)
APP_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 _.\-]{0,79}$")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    settings = get_settings()
    logger.info(
        "LLM middleware ready mode=%s auth=%s claude_key=%s openai_key=%s grok_key=%s nvidia_key=%s groq_key=%s ollama_key=%s gemini_key=%s horde_key=%s",
        "mock" if settings.mock_mode else "live",
        "on" if settings.auth_required else "off",
        "set" if settings.anthropic_api_key.strip() else "missing",
        "set" if settings.openai_api_key.strip() else "missing",
        "set" if settings.xai_api_key.strip() else "missing",
        "set" if settings.nvidia_api_key.strip() else "missing",
        "set" if settings.groq_api_key.strip() else "missing",
        "set" if settings.ollama_api_key.strip() else "missing",
        "set" if settings.gemini_api_key.strip() else "missing",
        "set" if settings.aihorde_api_key.strip() else "missing",
    )
    yield


app = FastAPI(
    title="LLM Middleware",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    swagger_ui_parameters={"persistAuthorization": True},
    description=(
        "One API for every application. "
        "Send model, type, query, and context. "
        "type claude calls Claude. type chatgpt calls ChatGPT. type grok calls xAI Grok. type groq calls Groq. type nvidia calls NVIDIA NIM. type ollama calls online Ollama. type gemini calls Google Gemini. "
        "POST /api/v1/images generates a picture with type chatgpt, grok, nvidia, ollama, gemini, or horde. "
        "Swagger UI is at /swagger."
    ),
    openapi_tags=[
        {"name": "chat", "description": "Call Claude, ChatGPT, xAI Grok, Groq, NVIDIA NIM, Gemini, or Ollama through this middleware."},
        {"name": "images", "description": "Generate an image with ChatGPT, xAI Grok, NVIDIA, Ollama, Gemini, or AI Horde."},
    ],
)

CHAT_EXAMPLES = Body(
    openapi_examples={
        "claude": {
            "summary": "Application one calls Claude",
            "value": {
                "app": "application-one",
                "model": "auto",
                "type": "claude",
                "query": "Say hello from application one",
                "context": "You are a helpful assistant.",
            },
        },
        "chatgpt": {
            "summary": "ChatGPT application",
            "value": {
                "app": "chatgpt-app",
                "model": "auto",
                "type": "chatgpt",
                "query": "Say hello from the ChatGPT application",
                "context": "You are a helpful assistant.",
            },
        },
        "grok": {
            "summary": "Grok",
            "value": {
                "app": "grok-app",
                "model": "auto",
                "type": "grok",
                "query": "Say hello from the Grok application",
                "context": "You are a helpful assistant.",
            },
        },
        "nvidia": {
            "summary": "NVIDIA NIM",
            "value": {
                "app": "nvidia-app",
                "model": "auto",
                "type": "nvidia",
                "query": "Say hello from the NVIDIA application",
                "context": "You are a helpful assistant.",
            },
        },
        "groq": {
            "summary": "Groq",
            "value": {
                "app": "groq-app",
                "model": "auto",
                "type": "groq",
                "query": "Say hello from the Groq application",
                "context": "You are a helpful assistant.",
            },
        },
        "ollama": {
            "summary": "Ollama",
            "value": {
                "app": "ollama-app",
                "model": "auto",
                "type": "ollama",
                "query": "Say hello from the Ollama application",
                "context": "You are a helpful assistant.",
            },
        },
        "gemini": {
            "summary": "Gemini",
            "value": {
                "app": "gemini-app",
                "model": "auto",
                "type": "gemini",
                "query": "Say hello from the Gemini application",
                "context": "You are a helpful assistant.",
            },
        },
    }
)

ERROR_RESPONSES = {
    400: {"model": ErrorResponse, "description": "Missing query, unknown type, or invalid app name."},
    401: {"model": ErrorResponse, "description": "Middleware API key is missing or wrong."},
    502: {"model": ErrorResponse, "description": "The model provider rejected the call."},
    503: {"model": ErrorResponse, "description": "The provider API key is not configured."},
    504: {"model": ErrorResponse, "description": "The model provider took too long."},
}

_settings = get_settings()
_origins = [item.strip() for item in _settings.cors_origins.split(",") if item.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins or ["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class AuthError(Exception):
    pass


def require_api_key_checked(
    key: str | None = Security(api_key_header),
    bearer: HTTPAuthorizationCredentials | None = Security(bearer_scheme),
) -> None:
    expected = get_settings().middleware_api_key.strip()
    if not expected:
        return
    supplied = (key or "").strip()
    if not supplied and bearer is not None:
        supplied = bearer.credentials.strip()
    ok = bool(supplied) and hmac.compare_digest(supplied.encode(), expected.encode())
    if not ok:
        raise AuthError()


@app.exception_handler(AuthError)
async def auth_error_handler(_request: Request, _exc: AuthError):
    return JSONResponse(
        status_code=401,
        content={
            "error": {
                "message": "Invalid API key. Send header X-API-Key or Authorization: Bearer <key>.",
                "type": "auth_error",
            }
        },
    )


@app.exception_handler(BadRequest)
async def bad_request_handler(_request: Request, exc: BadRequest):
    return JSONResponse(
        status_code=400,
        content={"error": {"message": exc.message, "type": "bad_request"}},
    )


@app.exception_handler(ProviderError)
async def provider_error_handler(_request: Request, exc: ProviderError):
    logger.info("provider_error provider=%s status=%s", exc.provider or "-", exc.status)
    body = {"error": {"message": exc.message, "type": "provider_error"}}
    if exc.provider:
        body["error"]["provider"] = exc.provider
    return JSONResponse(status_code=exc.status, content=body)


def application_name(body_app: str, request: Request, fallback: str) -> str:
    chosen = body_app.strip()
    if not chosen or chosen == "default":
        chosen = (request.headers.get("x-app-name") or fallback).strip() or fallback
    if not APP_NAME.fullmatch(chosen):
        raise BadRequest(
            "app may contain letters, numbers, spaces, dots, underscores, and hyphens."
        )
    return chosen


def _swagger_page() -> HTMLResponse:
    return get_swagger_ui_html(
        openapi_url=app.openapi_url,
        title="LLM Middleware - Swagger",
        swagger_ui_parameters={"persistAuthorization": True},
    )


@app.get("/swagger", include_in_schema=False)
@app.get("/swagger/index.html", include_in_schema=False)
async def swagger_ui():
    return _swagger_page()


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def home():
    return PAGE


@app.get("/health", include_in_schema=False)
@app.get("/api/v1/health", include_in_schema=False)
async def health():
    return {"status": "ok", "service": "llm-middleware"}


@app.get("/api/v1/providers", include_in_schema=False)
async def providers():
    settings = get_settings()
    return {
        "mode": "mock" if settings.mock_mode else "live",
        "auth_required": settings.auth_required,
        "default_provider": settings.default_provider,
        "endpoint": "/api/v1/chat",
        "image_endpoint": "/api/v1/images",
        "image_default_provider": settings.image_default_provider,
        "providers": [
            {
                "type": "claude",
                "provider": "claude",
                "configured": bool(settings.anthropic_api_key.strip()),
                "default_model": settings.claude_default_model,
            },
            {
                "type": "chatgpt",
                "provider": "openai",
                "configured": bool(settings.openai_api_key.strip()),
                "default_model": settings.openai_default_model,
            },
            {
                "type": "grok",
                "provider": "grok",
                "configured": bool(settings.xai_api_key.strip()),
                "default_model": settings.grok_default_model,
            },
            {
                "type": "nvidia",
                "provider": "nvidia",
                "configured": bool(settings.nvidia_api_key.strip()),
                "default_model": settings.nvidia_default_model,
            },
            {
                "type": "groq",
                "provider": "groq",
                "configured": bool(settings.groq_api_key.strip()),
                "default_model": settings.groq_default_model,
            },
            {
                "type": "ollama",
                "provider": "ollama",
                "configured": "ollama.com" not in settings.ollama_base_url.lower()
                or bool(settings.ollama_api_key.strip()),
                "default_model": settings.ollama_default_model,
                "base_url": settings.ollama_base_url,
            },
            {
                "type": "gemini",
                "provider": "gemini",
                "configured": bool(settings.gemini_api_key.strip()),
                "default_model": settings.gemini_default_model,
            },
            {
                "type": "horde",
                "provider": "horde",
                "configured": bool(settings.aihorde_api_key.strip()),
                "default_model": settings.aihorde_image_model,
            },
        ],
    }


@app.post(
    "/api/v1/chat",
    response_model=ChatResponse,
    tags=["chat"],
    summary="Call the middleware",
    description="Every application uses this endpoint. Set type to claude, chatgpt, grok, groq, nvidia, or ollama. model auto selects the default model. Set stream true to receive the answer as server-sent events.",
    responses=ERROR_RESPONSES,
)
async def chat(
    request: Request,
    body: Annotated[ChatRequest, CHAT_EXAMPLES],
    _: None = Depends(require_api_key_checked),
):
    app_name = application_name(body.app, request, "default")
    started = time.perf_counter()
    if body.stream:
        session = await open_chat_stream(body, app_name)
        logger.info("chat stream app=%s type=%s model=%s", session.app, session.type_name, session.model)
        return StreamingResponse(
            middleware_events(session),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
    result = await handle_chat(body, app_name)
    logger.info(
        "chat app=%s type=%s model=%s mock=%s ms=%d",
        result.app,
        result.type,
        result.model,
        result.mock,
        int((time.perf_counter() - started) * 1000),
    )
    return result


IMAGE_EXAMPLES = Body(
    openapi_examples={
        "grok": {
            "summary": "Grok image",
            "value": {
                "app": "image-app",
                "model": "auto",
                "type": "grok",
                "prompt": "A red bicycle on a quiet street at sunset",
                "size": "1024x1024",
                "n": 1,
            },
        },
        "chatgpt": {
            "summary": "ChatGPT image",
            "value": {
                "app": "image-app",
                "model": "auto",
                "type": "chatgpt",
                "prompt": "A red bicycle on a quiet street at sunset",
                "size": "1024x1024",
            },
        },
        "nvidia": {
            "summary": "NVIDIA image",
            "value": {
                "app": "image-app",
                "model": "auto",
                "type": "nvidia",
                "prompt": "A red bicycle on a quiet street at sunset",
                "size": "1024x1024",
            },
        },
        "ollama": {
            "summary": "Ollama image",
            "value": {
                "app": "image-app",
                "model": "auto",
                "type": "ollama",
                "prompt": "A red bicycle on a quiet street at sunset",
                "size": "1024x1024",
            },
        },
        "gemini": {
            "summary": "Gemini image",
            "value": {
                "app": "image-app",
                "model": "auto",
                "type": "gemini",
                "prompt": "A red bicycle on a quiet street at sunset",
                "size": "1024x1024",
            },
        },
        "horde": {
            "summary": "AI Horde image",
            "value": {
                "app": "image-app",
                "model": "auto",
                "type": "horde",
                "prompt": "A red bicycle on a quiet street at sunset",
                "size": "512x512",
                "n": 1,
            },
        },
    }
)


@app.post(
    "/api/v1/images",
    response_model=ImageResponse,
    tags=["images"],
    summary="Generate an image",
    description="Create a picture from a text prompt. type auto uses Grok. type chatgpt uses OpenAI. type nvidia uses FLUX.1-dev. type ollama uses x/z-image-turbo. type gemini uses Gemini image generation. type horde uses AI Horde and waits for a shared worker. The answer includes a url or base64 image.",
    responses=ERROR_RESPONSES,
)
async def images(
    request: Request,
    body: Annotated[ImageRequest, IMAGE_EXAMPLES],
    _: None = Depends(require_api_key_checked),
):
    app_name = application_name(body.app, request, "image-app")
    started = time.perf_counter()
    result = await handle_image(body, app_name)
    logger.info(
        "image app=%s type=%s model=%s count=%d mock=%s ms=%d",
        result.app,
        result.type,
        result.model,
        len(result.images),
        result.mock,
        int((time.perf_counter() - started) * 1000),
    )
    return result


@app.post(
    "/v1/chat/completions",
    response_model=OpenAIChatResponse,
    tags=["chat"],
    summary="OpenAI-compatible chat",
    description="Same middleware for a client that already sends OpenAI chat messages. model auto uses ChatGPT. Set stream true for a token stream.",
    responses=ERROR_RESPONSES,
)
async def openai_completions(
    request: Request,
    body: OpenAIChatRequest,
    _: None = Depends(require_api_key_checked),
):
    """Same middleware, in the shape a ChatGPT-style client already sends."""
    context_parts: list[str] = []
    history: list[ChatMessage] = []
    for message in body.messages:
        if message.normalized_role() == "system":
            if message.content.strip():
                context_parts.append(message.content.strip())
            continue
        history.append(message)

    query = ""
    if history and history[-1].normalized_role() == "user":
        query = history[-1].content
        history = history[:-1]

    incoming = ChatRequest(
        model=body.model,
        type=body.type or "auto",
        query=query,
        context="\n\n".join(context_parts),
        messages=history,
        temperature=body.temperature,
        max_tokens=body.max_tokens if body.max_tokens is not None else (body.max_completion_tokens or 1024),
        top_p=body.top_p,
        stream=body.stream,
    )
    app_name = application_name(body.user or "", request, "chatgpt-app")
    if incoming.stream:
        session = await open_chat_stream(incoming, app_name, fallback_provider="openai")
        return StreamingResponse(
            openai_events(session),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
    result = await handle_chat(incoming, app_name, fallback_provider="openai")
    prompt_tokens = result.usage.input_tokens or 0
    completion_tokens = result.usage.output_tokens or 0
    return {
        "id": "chatcmpl_" + result.id.removeprefix("mw_"),
        "object": "chat.completion",
        "created": result.created,
        "model": result.model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": result.answer},
                "finish_reason": result.finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run("app.main:app", host=settings.host, port=settings.port, reload=True)
