import asyncio
import base64
import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import httpx

from app.config import Settings, get_settings
from app.providers import BadRequest, ProviderError, _error_text
from app.schemas import VideoJobResponse, VideoRequest

_VIDEO_DIR = Path(__file__).resolve().parent.parent / "data" / "videos"
_JOBS: dict[str, "VideoJob"] = {}
_TASKS: set[asyncio.Task] = set()


@dataclass
class VideoJob:
    id: str
    created: int
    app: str
    type: str
    provider: str
    model: str
    prompt: str
    status: str
    mock: bool
    remote_url: str | None = None
    file_name: str | None = None
    error: str | None = None
    status_url: str = ""
    cancel_url: str = ""
    duration: int = 5
    resolution: str = "720p"
    aspect_ratio: str = "16:9"
    output_format: str = "mp4"
    generate_audio: bool = True
    metadata: dict = field(default_factory=dict)


def _public_url(job: VideoJob, base_url: str) -> str | None:
    if job.remote_url:
        return job.remote_url
    if job.file_name:
        return base_url.rstrip("/") + f"/api/v1/videos/{job.id}/file"
    return None


def job_response(job: VideoJob, base_url: str) -> VideoJobResponse:
    return VideoJobResponse(
        id=job.id,
        created=job.created,
        app=job.app,
        type=job.type,
        provider=job.provider,
        model=job.model,
        prompt=job.prompt,
        status=job.status,
        video_url=_public_url(job, base_url),
        error=job.error,
        mock=job.mock,
        metadata=job.metadata,
    )


def find_job(job_id: str) -> VideoJob | None:
    return _JOBS.get(job_id)


def video_path(job: VideoJob) -> Path | None:
    if not job.file_name:
        return None
    return _VIDEO_DIR / job.file_name


def _extract_video(data: dict) -> tuple[str | None, str | None]:
    video = data.get("video")
    if isinstance(video, dict):
        nested = video.get("url")
        if isinstance(nested, str) and nested.startswith(("http://", "https://")):
            return nested, None
    raw_video = data.get("b64_video")
    if isinstance(raw_video, str) and raw_video.strip():
        text = raw_video.strip()
        if text.startswith("http://") or text.startswith("https://"):
            return text, None
        return None, text.split(",", 1)[1] if text.startswith("data:") and "," in text else text
    for key in ("video_url", "url"):
        value = data.get(key)
        if isinstance(value, str) and value.startswith(("http://", "https://")):
            return value, None
    video = data.get("video")
    if isinstance(video, str) and video.startswith(("http://", "https://")):
        return video, None
    if isinstance(video, str) and len(video.strip()) > 80:
        text = video.strip()
        return None, text.split(",", 1)[1] if text.startswith("data:") and "," in text else text
    rows = data.get("artifacts") if isinstance(data.get("artifacts"), list) else data.get("data")
    if isinstance(rows, list):
        for row in rows:
            if isinstance(row, dict):
                found = _extract_video(row)
                if found[0] or found[1]:
                    return found
    return None, None


_RESOLUTIONS = {"480p", "720p", "1080p"}
_ASPECTS = {"16:9", "4:3", "1:1", "3:4", "9:16", "21:9"}
_FORMATS = {"mp4", "mov"}
_HF_DONE = {"completed", "failed", "nsfw", "canceled"}


def _resolve(req: VideoRequest, settings: Settings) -> tuple[str, str, str]:
    token = (req.type or "auto").strip().lower().replace("_", "").replace("-", "").replace(" ", "")
    requested = (req.model or "auto").strip() or "auto"
    if token in {"", "auto", "free", "ltx", "huggingface"}:
        model = settings.free_video_model if requested.lower() in {"", "auto"} else requested
        return "free", "free", model
    if token in {"nvidia", "nim"}:
        model = settings.nvidia_video_model if requested.lower() in {"", "auto"} else requested
        return "nvidia", "nvidia", model
    if token in {"higgsfield", "hf", "seedance"}:
        model = settings.hf_video_model if requested.lower() in {"", "auto"} else requested
        return "higgsfield", "higgsfield", model
    raise BadRequest('type must be "auto", "free", "higgsfield", or "nvidia".')


async def create_video(
    req: VideoRequest,
    app_name: str,
    client: httpx.AsyncClient | None = None,
) -> VideoJob:
    prompt = req.prompt.strip()
    if not prompt:
        raise BadRequest("Send a prompt.")
    if len(prompt) > 4000:
        raise BadRequest("prompt must be 4000 characters or fewer.")
    if req.resolution not in _RESOLUTIONS:
        raise BadRequest('resolution must be "480p", "720p", or "1080p".')
    if req.aspect_ratio not in _ASPECTS:
        raise BadRequest('aspect_ratio must be "16:9", "4:3", "1:1", "3:4", "9:16", or "21:9".')
    if req.output_format not in _FORMATS:
        raise BadRequest('output_format must be "mp4" or "mov".')
    settings = get_settings()
    provider, type_name, model = _resolve(req, settings)
    if not settings.mock_mode and provider == "nvidia" and not settings.nvidia_api_key.strip():
        raise ProviderError(
            "NVIDIA_API_KEY is not set. Add it to the .env file to generate video. "
            "Create a key at https://build.nvidia.com/settings/api-keys",
            status=503,
            provider="nvidia",
        )
    if not settings.mock_mode and provider == "higgsfield" and not settings.hf_credentials.strip():
        raise ProviderError(
            "HF_CREDENTIALS is not set. Add key_id:key_secret to the .env file to use Higgsfield video. "
            "Create a key in the Higgsfield dashboard.",
            status=503,
            provider="higgsfield",
        )
    job = VideoJob(
        id="mw_vid_" + uuid.uuid4().hex[:24],
        created=int(time.time()),
        app=app_name,
        type=type_name,
        provider=provider,
        model=model,
        prompt=prompt,
        status="queued",
        mock=settings.mock_mode,
        duration=req.duration,
        resolution=req.resolution,
        aspect_ratio=req.aspect_ratio,
        output_format=req.output_format,
        generate_audio=req.generate_audio,
        metadata=req.metadata,
    )
    _JOBS[job.id] = job
    if settings.mock_mode:
        job.status = "completed"
        job.remote_url = "https://example.com/mock-video.mp4"
        return job
    if provider == "higgsfield":
        try:
            await _submit_higgsfield(job, settings, client)
        except Exception:
            _JOBS.pop(job.id, None)
            raise
        return job
    task = asyncio.create_task(_run_free(job.id) if provider == "free" else _run_nvidia(job.id))
    _TASKS.add(task)
    task.add_done_callback(_TASKS.discard)
    return job


def _hf_headers(settings: Settings) -> dict:
    return {
        "Authorization": f"Key {settings.hf_credentials.strip()}",
        "Accept": "application/json",
        "Content-Type": "application/json",
    }


async def _hf_send(
    method: str,
    url: str,
    settings: Settings,
    payload: dict | None = None,
    client: httpx.AsyncClient | None = None,
) -> httpx.Response:
    owns_client = client is None
    http = client or httpx.AsyncClient(timeout=60)
    headers = _hf_headers(settings)
    try:
        try:
            if method == "POST":
                return await http.post(url, headers=headers, json=payload)
            return await http.get(url, headers=headers)
        except httpx.TimeoutException as exc:
            raise ProviderError("Higgsfield took too long to answer.", status=504, provider="higgsfield") from exc
        except httpx.HTTPError as exc:
            raise ProviderError("Could not reach Higgsfield.", status=502, provider="higgsfield") from exc
    finally:
        if owns_client:
            await http.aclose()


def _apply_higgsfield(job: VideoJob, data: dict) -> None:
    if data.get("status_url"):
        job.status_url = str(data["status_url"])
    if data.get("cancel_url"):
        job.cancel_url = str(data["cancel_url"])
    status = str(data.get("status") or "").strip()
    if status:
        job.status = status
    url, _raw = _extract_video(data)
    if url:
        job.remote_url = url
        job.status = "completed"
    if job.status == "nsfw":
        job.error = "Higgsfield rejected this prompt."
    elif job.status == "failed":
        err = data.get("error")
        if isinstance(err, dict):
            job.error = str(err.get("message") or "Higgsfield could not finish this video.")
        elif isinstance(err, str) and err.strip():
            job.error = err
        else:
            job.error = "Higgsfield could not finish this video."


async def _submit_higgsfield(
    job: VideoJob,
    settings: Settings,
    client: httpx.AsyncClient | None,
) -> None:
    endpoint = settings.hf_video_model.strip().strip("/")
    url = settings.hf_base_url.rstrip("/") + "/" + endpoint
    response = await _hf_send(
        "POST",
        url,
        settings,
        {
            "prompt": job.prompt,
            "duration": job.duration,
            "resolution": job.resolution,
            "aspect_ratio": job.aspect_ratio,
            "output_format": job.output_format,
            "generate_audio": job.generate_audio,
        },
        client,
    )
    data = _body(response)
    if response.status_code >= 400:
        if response.status_code in (401, 403):
            raise ProviderError(
                "Higgsfield rejected the API key. Check HF_CREDENTIALS in the .env file.",
                status=502,
                provider="higgsfield",
            )
        raise ProviderError(_error_text(response), status=502 if response.status_code >= 500 else response.status_code, provider="higgsfield")
    request_id = str(data.get("request_id") or "").strip()
    if not request_id:
        raise ProviderError("Higgsfield did not return a request id.", status=502, provider="higgsfield")
    _JOBS.pop(job.id, None)
    job.id = request_id
    _JOBS[job.id] = job
    _apply_higgsfield(job, data)
    if job.status not in _HF_DONE:
        job.status = str(data.get("status") or "queued")


async def refresh_video(job: VideoJob, client: httpx.AsyncClient | None = None) -> VideoJob:
    if job.provider != "higgsfield" or job.mock or job.status in _HF_DONE:
        return job
    settings = get_settings()
    status_url = job.status_url or (
        settings.hf_base_url.rstrip("/") + "/requests/" + job.id + "/status"
    )
    response = await _hf_send("GET", status_url, settings, client=client)
    if response.status_code >= 400:
        if response.status_code in (401, 403):
            _fail(job, "Higgsfield rejected the API key. Check HF_CREDENTIALS in the .env file.")
            return job
        _fail(job, _error_text(response))
        return job
    _apply_higgsfield(job, _body(response))
    return job


async def cancel_video(job_id: str, client: httpx.AsyncClient | None = None) -> VideoJob | None:
    job = _JOBS.get(job_id)
    if job is None:
        return None
    if job.mock or job.provider != "higgsfield":
        if job.status in _HF_DONE and job.status != "queued":
            raise BadRequest("Only a queued video can be canceled.")
        job.status = "canceled"
        return job
    if job.status != "queued":
        raise BadRequest("Only a queued video can be canceled.")
    settings = get_settings()
    cancel_url = job.cancel_url or (settings.hf_base_url.rstrip("/") + "/requests/" + job.id + "/cancel")
    response = await _hf_send("POST", cancel_url, settings, {}, client)
    if response.status_code in (200, 202):
        job.status = "canceled"
        return job
    if response.status_code == 400:
        raise BadRequest("This video already started and cannot be canceled.")
    raise ProviderError(_error_text(response), status=502, provider="higgsfield")


def _save_video(job_id: str, raw_b64: str) -> str:
    _VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    name = f"{job_id}.mp4"
    (_VIDEO_DIR / name).write_bytes(base64.b64decode(raw_b64))
    return name


def _fail(job: VideoJob, message: str) -> None:
    job.status = "failed"
    job.error = message


async def _nvidia_request(method: str, url: str, settings: Settings, payload: dict | None = None) -> httpx.Response:
    headers = {
        "Authorization": f"Bearer {settings.nvidia_api_key.strip()}",
        "Accept": "application/json",
        "Content-Type": "application/json",
    }
    timeout = httpx.Timeout(30.0, read=settings.nvidia_video_wait_seconds)
    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            if method == "POST":
                return await client.post(url, headers=headers, json=payload)
            return await client.get(url, headers=headers)
        except httpx.TimeoutException as exc:
            raise ProviderError("NVIDIA video took too long to answer.", status=504, provider="nvidia") from exc
        except httpx.HTTPError as exc:
            raise ProviderError("Could not reach NVIDIA video.", status=502, provider="nvidia") from exc


def _body(response: httpx.Response) -> dict:
    raw = getattr(response, "content", None)
    if raw in {b"", ""}:
        return {}
    try:
        data = response.json()
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _store_result(job: VideoJob, data: dict) -> None:
    url, raw = _extract_video(data)
    if url:
        job.remote_url = url
        job.status = "completed"
        return
    if raw:
        job.file_name = _save_video(job.id, raw)
        job.status = "completed"
        return
    _fail(job, "NVIDIA returned no video.")


async def _run_free(job_id: str) -> None:
    job = _JOBS[job_id]
    settings = get_settings()
    job.status = "in_progress"
    base = settings.free_video_url.rstrip("/")
    payload = {
        "data": [
            job.prompt,
            "worst quality, inconsistent motion, blurry, jittery, distorted",
            None,
            None,
            512,
            704,
            "text-to-video",
            2,
            9,
            42,
            True,
            1,
            True,
        ]
    }
    timeout = httpx.Timeout(30.0, read=settings.nvidia_video_wait_seconds)
    session = uuid.uuid4().hex
    headers = _free_headers(settings)
    payload["fn_index"] = 4
    payload["session_hash"] = session
    payload["trigger_id"] = 4
    try:
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            started = await client.post(base + "/gradio_api/queue/join", json=payload, headers=headers)
            if started.status_code >= 400:
                _fail(job, "The free video service did not accept this prompt.")
                return
            async with client.stream(
                "GET",
                base + "/gradio_api/queue/data",
                params={"session_hash": session},
                headers=headers,
            ) as stream:
                if stream.status_code >= 400:
                    _fail(job, "The free video service could not be reached.")
                    return
                async for line in stream.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    message = json.loads(line.split(":", 1)[1].strip() or "null")
                    if not isinstance(message, dict):
                        continue
                    kind = str(message.get("msg") or "")
                    if kind != "process_completed":
                        continue
                    output = message.get("output") if isinstance(message.get("output"), dict) else {}
                    if output.get("error"):
                        _fail(job, _free_error(json.dumps(output.get("error")), has_token=bool(headers)))
                        return
                    await _store_free_video(job, client, base, json.dumps(output.get("data")), headers)
                    return
        if job.status == "in_progress":
            _fail(job, "The free video service closed before the clip was ready.")
    except ProviderError as exc:
        _fail(job, exc.message)
    except Exception:
        _fail(job, "The free video service failed.")


def _free_headers(settings: Settings) -> dict[str, str]:
    token = settings.huggingface_token.strip()
    if not token:
        return {}
    return {"Authorization": f"Bearer {token}"}


def _free_error(raw: str, *, has_token: bool) -> str:
    text = raw.strip()
    try:
        parsed = json.loads(text)
        if isinstance(parsed, str) and parsed.strip():
            text = parsed.strip()
    except Exception:
        pass
    text = " ".join(text.split())
    empty = not text or text.lower() in {"null", "none"}
    quota = empty or "quota" in text.lower() or "zerogpu" in text.lower()
    if quota and not has_token:
        return (
            "The free video GPU time for this network is used up. "
            "Add a free Hugging Face token as HUGGINGFACE_TOKEN in the .env file, then send one new job. "
            "Create the token at https://huggingface.co/settings/tokens"
        )
    if empty:
        return "The free video service did not return a clip."
    return text[:300]


async def _store_free_video(
    job: VideoJob,
    client: httpx.AsyncClient,
    base: str,
    raw: str,
    headers: dict[str, str],
) -> None:
    parsed = json.loads(raw)
    video = parsed[0].get("video") if isinstance(parsed, list) and parsed and isinstance(parsed[0], dict) else None
    if not isinstance(video, dict):
        _fail(job, "The free video service returned no video.")
        return
    url = str(video.get("url") or "").strip()
    if url.startswith("/"):
        url = base + url
    if not url.startswith(("http://", "https://")):
        _fail(job, "The free video service returned no video.")
        return
    downloaded = await client.get(url, headers=headers)
    if downloaded.status_code >= 400 or len(downloaded.content) < 1000:
        _fail(job, "The free video service returned an empty clip.")
        return
    _VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    name = f"{job.id}.mp4"
    (_VIDEO_DIR / name).write_bytes(downloaded.content)
    job.file_name = name
    job.status = "completed"


async def _run_nvidia(job_id: str) -> None:
    job = _JOBS[job_id]
    settings = get_settings()
    job.status = "in_progress"
    try:
        response = await _nvidia_request(
            "POST",
            settings.nvidia_video_url,
            settings,
            {
                "model_mode": "text2video",
                "prompt": job.prompt,
                "resolution": "480_16_9",
                "num_frames": 25,
                "num_inference_steps": 35,
                "fps": 24,
                "seed": 0,
            },
        )
        if response.status_code == 200:
            _store_result(job, _body(response))
            return
        if response.status_code != 202:
            detail = _error_text(response)
            if response.status_code == 404 or "Not found for account" in detail:
                _fail(
                    job,
                    "NVIDIA's public video API returned 404 for this key. "
                    "Cosmos3 Nano is listed at https://build.nvidia.com/nvidia/cosmos3-nano, "
                    "but https://ai.api.nvidia.com/v1/cosmos/nvidia/cosmos3-nano is not serving requests.",
                )
            elif response.status_code in (401, 403):
                _fail(job, "The provider rejected the API key. Check NVIDIA_API_KEY in the .env file.")
            else:
                _fail(job, detail)
            return
        request_id = response.headers.get("NVCF-REQID", "").strip()
        if not request_id:
            inline = _body(response)
            if inline:
                _store_result(job, inline)
                return
            _fail(job, "NVIDIA accepted the video but did not return a job id.")
            return
        deadline = time.monotonic() + settings.nvidia_video_wait_seconds
        while time.monotonic() < deadline:
            await asyncio.sleep(5)
            status = await _nvidia_request(
                "GET",
                settings.nvidia_video_status_url.rstrip("/") + "/" + request_id,
                settings,
            )
            if status.status_code == 202:
                continue
            if status.status_code == 200:
                _store_result(job, _body(status))
                return
            _fail(job, _error_text(status))
            return
        _fail(job, "NVIDIA is still rendering this video. Try the job again later.")
    except ProviderError as exc:
        _fail(job, exc.message)
    except Exception:
        _fail(job, "NVIDIA video failed.")

