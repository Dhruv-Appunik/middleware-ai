import asyncio
import math
import re
import time
import uuid

import httpx

from app.config import Settings, get_settings
from app.providers import BadRequest, ProviderError, _raise_for_status, _send
from app.schemas import GeneratedImage, ImageRequest, ImageResponse

_SIZE = re.compile(r"^\d{2,4}x\d{2,4}$")
_OPENAI_ALIASES = {"chatgpt", "gpt", "openai"}
_GROK_ALIASES = {"grok", "grouk", "xai"}
_NVIDIA_ALIASES = {"nvidia", "nim", "nvidianim"}
_OLLAMA_ALIASES = {"ollama"}
_GEMINI_ALIASES = {"gemini", "google"}
_HORDE_ALIASES = {"horde", "aihorde", "stablehorde"}
_MOCK_PNG = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)


def _token(value: str) -> str:
    return re.sub(r"[\s_\-]+", "", value.strip().lower())


def _parse_size(size: str) -> tuple[int, int]:
    cleaned = size.strip().lower()
    if not _SIZE.fullmatch(cleaned):
        raise BadRequest('size must look like "1024x1024".')
    width, height = cleaned.split("x")
    return int(width), int(height)


def _clean_b64(value: str) -> str:
    text = value.strip()
    if text.startswith("data:") and "," in text:
        return text.split(",", 1)[1]
    return text


def _aspect_ratio(width: int, height: int) -> str:
    divisor = math.gcd(width, height)
    return f"{width // divisor}:{height // divisor}"


def resolve_image(req: ImageRequest, settings: Settings) -> tuple[str, str, str, str, str]:
    requested_type = (req.type or "auto").strip() or "auto"
    requested_model = (req.model or "auto").strip() or "auto"
    token = _token(requested_type)
    if token in {"", "auto"}:
        token = _token(settings.image_default_provider)
    if token in _OPENAI_ALIASES:
        provider, type_name = "openai", "chatgpt" if token != "openai" else "openai"
        default_model = settings.openai_image_model
    elif token in _GROK_ALIASES:
        provider, type_name = "grok", "grok"
        default_model = settings.grok_image_model
    elif token in _NVIDIA_ALIASES:
        provider, type_name = "nvidia", "nvidia"
        default_model = settings.nvidia_image_model
    elif token in _OLLAMA_ALIASES or token.startswith("ollama"):
        provider, type_name = "ollama", "ollama"
        default_model = settings.ollama_image_model
    elif token in _GEMINI_ALIASES or token.startswith("gemini") or token.startswith("google"):
        provider, type_name = "gemini", "gemini"
        default_model = settings.gemini_image_model
    elif token in _HORDE_ALIASES or token.startswith("horde") or token.startswith("aihorde"):
        provider, type_name = "horde", "horde"
        default_model = settings.aihorde_image_model
    else:
        raise BadRequest(
            'type must be "auto", "chatgpt", "grok", "nvidia", "ollama", "gemini", or "horde".'
        )
    model_name = default_model if _token(requested_model) in {"", "auto"} else requested_model
    return provider, type_name, model_name, requested_type, requested_model


def _collect_images(data: dict) -> list[GeneratedImage]:
    images: list[GeneratedImage] = []
    direct = data.get("image")
    if isinstance(direct, str) and direct.strip():
        images.append(GeneratedImage(b64_json=_clean_b64(direct)))
        return images
    rows = data.get("data")
    if not isinstance(rows, list):
        rows = data.get("artifacts") if isinstance(data.get("artifacts"), list) else []
    for row in rows:
        if not isinstance(row, dict):
            continue
        url = row.get("url")
        raw = row.get("b64_json") or row.get("base64") or row.get("image") or ""
        b64 = _clean_b64(str(raw))
        if not url and not b64:
            continue
        images.append(GeneratedImage(url=str(url) if url else None, b64_json=b64 or None))
    return images


def _gemini_images(data: dict) -> list[GeneratedImage]:
    images: list[GeneratedImage] = []
    for candidate in data.get("candidates") or []:
        if not isinstance(candidate, dict):
            continue
        content = candidate.get("content") or {}
        for part in content.get("parts") or []:
            if not isinstance(part, dict):
                continue
            inline = part.get("inlineData") or part.get("inline_data") or {}
            if not isinstance(inline, dict):
                continue
            raw = str(inline.get("data") or "").strip()
            if raw:
                images.append(GeneratedImage(b64_json=_clean_b64(raw)))
    return images


def _horde_picture(value: str) -> GeneratedImage | None:
    text = value.strip()
    if not text:
        return None
    if text.startswith("http://") or text.startswith("https://"):
        return GeneratedImage(url=text)
    return GeneratedImage(b64_json=_clean_b64(text))


async def _horde_get(http: httpx.AsyncClient, url: str, headers: dict) -> dict:
    try:
        response = await http.get(url, headers=headers)
    except httpx.TimeoutException as exc:
        raise ProviderError("AI Horde took too long to answer.", status=504, provider="horde") from exc
    except httpx.HTTPError as exc:
        raise ProviderError("Could not reach AI Horde.", status=502, provider="horde") from exc
    _raise_for_status(response, "horde", "AIHORDE_API_KEY")
    try:
        data = response.json()
    except Exception as exc:
        raise ProviderError("AI Horde returned invalid JSON.", status=502, provider="horde") from exc
    if not isinstance(data, dict):
        raise ProviderError("AI Horde returned an unexpected response.", status=502, provider="horde")
    return data


async def _horde_images(
    settings: Settings,
    client: httpx.AsyncClient | None,
    model: str,
    prompt: str,
    width: int,
    height: int,
    count: int,
) -> list[GeneratedImage]:
    if not settings.aihorde_api_key.strip():
        raise ProviderError(
            "AIHORDE_API_KEY is not set. Add it to the .env file to use type horde for images. "
            "Create a key at https://aihorde.net/register",
            status=503,
            provider="horde",
        )
    if width % 64 or height % 64:
        raise BadRequest('AI Horde size must be a multiple of 64, such as "512x512" or "1024x1024".')
    base = settings.aihorde_base_url.rstrip("/")
    headers = {
        "apikey": settings.aihorde_api_key.strip(),
        "Client-Agent": "llm-middleware:1.0:local",
        "User-Agent": "llm-middleware/1.0",
        "Accept": "application/json",
    }
    accepted = await _send(
        client,
        settings,
        f"{base}/generate/async",
        headers,
        {
            "prompt": prompt,
            "params": {
                "width": width,
                "height": height,
                "steps": 20,
                "n": count,
                "sampler_name": "k_euler",
                "cfg_scale": 7,
            },
            "models": [model],
            "nsfw": False,
            "censor_nsfw": True,
            "replacement_filter": True,
            "r2": True,
            "shared": False,
            "slow_workers": True,
        },
        "horde",
        "AIHORDE_API_KEY",
        unreachable_message="Could not reach AI Horde.",
    )
    job_id = str(accepted.get("id") or "").strip()
    if not job_id:
        raise ProviderError("AI Horde did not accept the image request.", status=502, provider="horde")
    owns_client = client is None
    http = client or httpx.AsyncClient(timeout=settings.request_timeout_seconds)
    deadline = time.monotonic() + settings.aihorde_wait_seconds
    try:
        await asyncio.sleep(1)
        while time.monotonic() < deadline:
            check = await _horde_get(http, f"{base}/generate/check/{job_id}", headers)
            if check.get("faulted"):
                raise ProviderError("AI Horde could not finish this image.", status=502, provider="horde")
            if check.get("done"):
                break
            await asyncio.sleep(3)
        else:
            raise ProviderError(
                "AI Horde is still drawing this image. Try again in a minute.",
                status=504,
                provider="horde",
            )
        status = await _horde_get(http, f"{base}/generate/status/{job_id}", headers)
    finally:
        if owns_client:
            await http.aclose()
    if status.get("faulted"):
        raise ProviderError("AI Horde could not finish this image.", status=502, provider="horde")
    images: list[GeneratedImage] = []
    for row in status.get("generations") or []:
        if not isinstance(row, dict):
            continue
        picture = _horde_picture(str(row.get("img") or ""))
        if picture is not None:
            images.append(picture)
    if not images:
        raise ProviderError("AI Horde returned no images.", status=502, provider="horde")
    return images


async def _generate_live(
    settings: Settings,
    client: httpx.AsyncClient | None,
    provider: str,
    model: str,
    prompt: str,
    size: str,
    count: int,
) -> list[GeneratedImage]:
    width, height = _parse_size(size)
    if provider == "openai":
        if not settings.openai_api_key.strip():
            raise ProviderError(
                "OPENAI_API_KEY is not set. Add it to the .env file to use type chatgpt for images.",
                status=503,
                provider="openai",
            )
        data = await _send(
            client,
            settings,
            settings.openai_base_url.rstrip("/") + "/images/generations",
            {"Authorization": f"Bearer {settings.openai_api_key.strip()}"},
            {"model": model, "prompt": prompt, "n": count, "size": f"{width}x{height}"},
            "openai",
            "OPENAI_API_KEY",
        )
    elif provider == "grok":
        if not settings.xai_api_key.strip():
            raise ProviderError(
                "XAI_API_KEY is not set. Add it to the .env file to use type grok for images. "
                "Create a key at https://console.x.ai/",
                status=503,
                provider="grok",
            )
        data = await _send(
            client,
            settings,
            settings.xai_base_url.rstrip("/") + "/images/generations",
            {"Authorization": f"Bearer {settings.xai_api_key.strip()}"},
            {
                "model": model,
                "prompt": prompt,
                "n": count,
                "aspect_ratio": _aspect_ratio(width, height),
            },
            "grok",
            "XAI_API_KEY",
        )
    elif provider == "ollama":
        if count != 1:
            raise BadRequest("Ollama image generation accepts n 1.")
        base = settings.ollama_base_url.rstrip("/")
        if base.endswith("/v1"):
            base = base[:-3]
        cloud = "ollama.com" in base.lower()
        api_key = settings.ollama_api_key.strip()
        if cloud and not api_key:
            raise ProviderError(
                "OLLAMA_API_KEY is not set. Add it to the .env file to use type ollama for images. "
                "Create a key at https://ollama.com/settings/keys",
                status=503,
                provider="ollama",
            )
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        data = await _send(
            client,
            settings,
            base + "/api/generate",
            headers,
            {
                "model": model,
                "prompt": prompt,
                "width": width,
                "height": height,
                "stream": False,
            },
            "ollama",
            "OLLAMA_API_KEY",
            unreachable_message=(
                "Could not reach Ollama Cloud."
                if cloud
                else "Ollama is not running. Start it with ollama serve, then pull an image model with ollama pull x/z-image-turbo."
            ),
        )
    elif provider == "gemini":
        if not settings.gemini_api_key.strip():
            raise ProviderError(
                "GEMINI_API_KEY is not set. Add it to the .env file to use type gemini for images. "
                "Create a key at https://aistudio.google.com/apikey",
                status=503,
                provider="gemini",
            )
        if count != 1:
            raise BadRequest("Gemini image generation accepts n 1.")
        base = settings.gemini_base_url.rstrip("/")
        if base.endswith("/openai"):
            base = base[: -len("/openai")]
        data = await _send(
            client,
            settings,
            f"{base}/models/{model}:generateContent",
            {"x-goog-api-key": settings.gemini_api_key.strip()},
            {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {
                    "responseModalities": ["IMAGE"],
                    "imageConfig": {"aspectRatio": _aspect_ratio(width, height)},
                },
            },
            "gemini",
            "GEMINI_API_KEY",
        )
        images = _gemini_images(data)
        if not images:
            raise ProviderError("Gemini returned no images.", status=502, provider="gemini")
        return images
    elif provider == "horde":
        return await _horde_images(settings, client, model, prompt, width, height, count)
    else:
        if not settings.nvidia_api_key.strip():
            raise ProviderError(
                "NVIDIA_API_KEY is not set. Add it to the .env file to use type nvidia for images. "
                "Create a key at https://build.nvidia.com/settings/api-keys",
                status=503,
                provider="nvidia",
            )
        if count != 1:
            raise BadRequest("NVIDIA image generation accepts n 1.")
        if (width, height) != (1024, 1024):
            raise BadRequest('NVIDIA image generation accepts size "1024x1024".')
        schnell = "schnell" in model
        owns_client = client is None
        http = client or httpx.AsyncClient(timeout=180)
        try:
            data = await _send(
                http,
                settings,
                settings.nvidia_image_url,
                {
                    "Authorization": f"Bearer {settings.nvidia_api_key.strip()}",
                    "Accept": "application/json",
                },
                {
                    "prompt": prompt,
                    "width": width,
                    "height": height,
                    "steps": 4 if schnell else 10,
                    "cfg_scale": 0 if schnell else 3.5,
                    "seed": 0,
                    "mode": "base",
                    "samples": 1,
                },
                "nvidia",
                "NVIDIA_API_KEY",
            )
        finally:
            if owns_client:
                await http.aclose()
    images = _collect_images(data)
    if not images:
        raise ProviderError("The image provider returned no images.", status=502, provider=provider)
    return images


async def handle_image(
    req: ImageRequest,
    app_name: str,
    client: httpx.AsyncClient | None = None,
) -> ImageResponse:
    prompt = req.prompt.strip()
    if not prompt:
        raise BadRequest("Send a prompt.")
    if len(prompt) > 4000:
        raise BadRequest("prompt must be 4000 characters or fewer.")
    settings = get_settings()
    provider, type_name, model_name, requested_type, requested_model = resolve_image(req, settings)
    size = req.size.strip().lower()
    width, height = _parse_size(size)
    if provider == "nvidia" and req.n != 1:
        raise BadRequest("NVIDIA image generation accepts n 1.")
    if provider == "nvidia" and (width, height) != (1024, 1024):
        raise BadRequest('NVIDIA image generation accepts size "1024x1024".')
    if provider == "ollama" and req.n != 1:
        raise BadRequest("Ollama image generation accepts n 1.")
    if provider == "gemini" and req.n != 1:
        raise BadRequest("Gemini image generation accepts n 1.")
    if provider == "horde" and (width % 64 or height % 64):
        raise BadRequest('AI Horde size must be a multiple of 64, such as "512x512" or "1024x1024".')
    if settings.mock_mode:
        images = [GeneratedImage(b64_json=_MOCK_PNG) for _ in range(req.n)]
    else:
        images = await _generate_live(settings, client, provider, model_name, prompt, size, req.n)
    return ImageResponse(
        id="mw_img_" + uuid.uuid4().hex[:24],
        created=int(time.time()),
        app=app_name,
        type=type_name,
        provider=provider,
        model=model_name,
        requested_type=requested_type,
        requested_model=requested_model,
        prompt=prompt,
        size=size,
        mock=settings.mock_mode,
        images=images,
        metadata=req.metadata,
    )
