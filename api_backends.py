"""
api_backends.py

Cloud VLM APIs for grounding eval (OpenAI GPT, Google Gemini, Anthropic Claude).
Uses HTTP + base64 JPEG image parts (stdlib only).

Parallel requests: run_api_jobs_parallel(..., workers=N) with ThreadPoolExecutor.
"""

from __future__ import annotations

import base64
import json
import os
import re
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Any, Callable

from PIL import Image

_PKG_ROOT = Path(__file__).resolve().parent

# Map normalized backend name -> API provider
_PROVIDER_BY_BACKEND: dict[str, str] = {
    "openai": "openai",
    "gpt": "openai",
    "chatgpt": "openai",
    "gemini": "gemini",
    "google": "gemini",
    "claude": "anthropic",
    "anthropic": "anthropic",
}

_DEFAULT_KEY_FILES: dict[str, tuple[str, ...]] = {
    "openai": (".openai_api_key",),
    "gemini": (".gemini_api_key",),
    "anthropic": (".anthropic_api_key",),
}

_ENV_KEYS: dict[str, tuple[str, ...]] = {
    "openai": ("OPENAI_API_KEY",),
    "gemini": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
    "anthropic": ("ANTHROPIC_API_KEY",),
}

_IMAGE_CACHE_LOCK = threading.Lock()
_IMAGE_B64_CACHE: dict[str, tuple[str, str]] = {}


def api_provider_for_backend(backend: str) -> str:
    key = (backend or "").strip().lower().replace("_", "-")
    if key not in _PROVIDER_BY_BACKEND:
        raise ValueError(f"Not an API backend: {backend!r}")
    return _PROVIDER_BY_BACKEND[key]


def resolve_api_key(
    backend: str,
    *,
    explicit: str | None = None,
    key_file: Path | str | None = None,
) -> str:
    """Resolve API key from explicit arg, key file, package dotfile, or env."""
    if explicit and explicit.strip():
        return explicit.strip()

    provider = api_provider_for_backend(backend)

    if key_file is not None:
        path = Path(key_file).expanduser()
        if path.is_file():
            text = path.read_text(encoding="utf-8").strip()
            if text:
                return text

    for name in _DEFAULT_KEY_FILES.get(provider, ()):
        path = _PKG_ROOT / name
        if path.is_file():
            text = path.read_text(encoding="utf-8").strip()
            if text:
                return text

    for env_name in _ENV_KEYS.get(provider, ()):
        val = os.environ.get(env_name, "").strip()
        if val:
            return val

    files = ", ".join(_DEFAULT_KEY_FILES.get(provider, ()))
    envs = ", ".join(_ENV_KEYS.get(provider, ()))
    raise FileNotFoundError(
        f"No API key for provider {provider!r} (backend={backend!r}). "
        f"Set one of env [{envs}] or create {files} under {_PKG_ROOT}"
    )


def clear_image_encode_cache() -> None:
    with _IMAGE_CACHE_LOCK:
        _IMAGE_B64_CACHE.clear()


def _pil_to_b64_jpeg(
    image: Image.Image,
    *,
    max_side: int | None = None,
    cache_key: str | None = None,
    jpeg_quality: int = 90,
) -> tuple[str, str]:
    if cache_key:
        with _IMAGE_CACHE_LOCK:
            hit = _IMAGE_B64_CACHE.get(cache_key)
        if hit is not None:
            return hit

    img = image.convert("RGB")
    if max_side is not None and max_side > 0:
        img.thumbnail((max_side, max_side), Image.Resampling.BICUBIC)
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=jpeg_quality)
    raw = buf.getvalue()
    encoded = base64.b64encode(raw).decode("ascii"), "image/jpeg"

    if cache_key:
        with _IMAGE_CACHE_LOCK:
            _IMAGE_B64_CACHE[cache_key] = encoded
    return encoded


def _http_json_post(
    url: str,
    payload: dict[str, Any],
    headers: dict[str, str],
    *,
    timeout_sec: int,
    max_retries: int = 6,
) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    last_err: Exception | None = None
    for attempt in range(max_retries):
        req = urllib.request.Request(
            url=url,
            data=data,
            method="POST",
            headers={**headers, "Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
                body = resp.read().decode("utf-8")
            return json.loads(body)
        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8", errors="replace")
            last_err = RuntimeError(f"API HTTP {e.code}: {err_body[:2000]}")
            if e.code in (429, 500, 502, 503, 504) and attempt < max_retries - 1:
                delay = min(2.0 ** attempt, 60.0)
                if e.code == 429:
                    m = re.search(r"retry(?:_after| in)?[:\s]+(\d+(?:\.\d+)?)", err_body, re.I)
                    if m:
                        delay = max(delay, float(m.group(1)))
                time.sleep(delay)
                continue
            raise last_err from e
        except urllib.error.URLError as e:
            last_err = RuntimeError(f"API request failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(min(2.0 ** attempt, 30.0))
                continue
            raise last_err from e
    raise last_err or RuntimeError("API request failed")


def _openai_vision_call(
    *,
    model: str,
    prompt: str,
    image_b64: str,
    mime: str,
    api_key: str,
    temperature: float,
    max_tokens: int,
    timeout_sec: int,
) -> str:
    url = "https://api.openai.com/v1/chat/completions"
    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime};base64,{image_b64}"},
                    },
                ],
            }
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    obj = _http_json_post(
        url,
        payload,
        {"Authorization": f"Bearer {api_key}"},
        timeout_sec=timeout_sec,
    )
    choice0 = (obj.get("choices") or [{}])[0]
    content = (choice0.get("message") or {}).get("content") or ""
    return content.strip()


def _gemini_vision_call(
    *,
    model: str,
    prompt: str,
    image_b64: str,
    mime: str,
    api_key: str,
    temperature: float,
    max_tokens: int,
    timeout_sec: int,
) -> str:
    model_path = model if model.startswith("models/") else f"models/{model}"
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/{model_path}"
        f":generateContent?key={api_key}"
    )
    payload: dict[str, Any] = {
        "contents": [
            {
                "parts": [
                    {"text": prompt},
                    {"inline_data": {"mime_type": mime, "data": image_b64}},
                ],
            }
        ],
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_tokens,
        },
    }
    obj = _http_json_post(url, payload, {}, timeout_sec=timeout_sec)
    candidates = obj.get("candidates") or []
    if not candidates:
        return ""
    parts = (((candidates[0].get("content") or {}).get("parts")) or [])
    return "".join(
        (p.get("text") or "") for p in parts if isinstance(p, dict)
    ).strip()


def _anthropic_vision_call(
    *,
    model: str,
    prompt: str,
    image_b64: str,
    mime: str,
    api_key: str,
    temperature: float,
    max_tokens: int,
    timeout_sec: int,
) -> str:
    url = "https://api.anthropic.com/v1/messages"
    payload: dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": mime,
                            "data": image_b64,
                        },
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ],
    }
    obj = _http_json_post(
        url,
        payload,
        {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        },
        timeout_sec=timeout_sec,
    )
    parts = obj.get("content") or []
    texts = [p.get("text", "") for p in parts if isinstance(p, dict) and p.get("type") == "text"]
    return "".join(texts).strip()


def api_vision_generate(
    *,
    provider: str,
    model: str,
    image: Image.Image,
    prompt: str,
    api_key: str,
    image_max_side: int = 384,
    temperature: float = 0.0,
    max_tokens: int = 512,
    timeout_sec: int = 120,
    image_cache_key: str | None = None,
) -> str:
    image_b64, mime = _pil_to_b64_jpeg(
        image,
        max_side=image_max_side,
        cache_key=image_cache_key,
    )
    if provider == "openai":
        return _openai_vision_call(
            model=model,
            prompt=prompt,
            image_b64=image_b64,
            mime=mime,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout_sec=timeout_sec,
        )
    if provider == "gemini":
        return _gemini_vision_call(
            model=model,
            prompt=prompt,
            image_b64=image_b64,
            mime=mime,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout_sec=timeout_sec,
        )
    if provider == "anthropic":
        return _anthropic_vision_call(
            model=model,
            prompt=prompt,
            image_b64=image_b64,
            mime=mime,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout_sec=timeout_sec,
        )
    raise ValueError(f"Unknown API provider: {provider!r}")


@dataclass
class ApiVisionJob:
    """One vision API request (parallel-safe when jobs are independent)."""

    job_id: Any
    prompt: str
    image: Image.Image | None = None
    image_path: Path | str | None = None
    pil_side: int | None = None
    gen_kw: dict[str, Any] = field(default_factory=dict)


@dataclass
class ApiVisionResult:
    job_id: Any
    text: str | None = None
    error: str | None = None


def _load_job_image(job: ApiVisionJob) -> Image.Image:
    if job.image is not None:
        img = job.image.convert("RGB")
    elif job.image_path is not None:
        img = Image.open(job.image_path).convert("RGB")
    else:
        raise ValueError(f"ApiVisionJob {job.job_id!r} needs image or image_path")
    if job.pil_side is not None and job.pil_side > 0:
        side = int(job.pil_side)
        img = img.resize((side, side), Image.Resampling.BICUBIC)
    return img


def run_api_job(backend: "ApiVLMBackend", job: ApiVisionJob) -> ApiVisionResult:
    try:
        image = _load_job_image(job)
        text = backend.generate(image, job.prompt, **job.gen_kw)
        return ApiVisionResult(job_id=job.job_id, text=text)
    except Exception as e:
        return ApiVisionResult(job_id=job.job_id, error=str(e))


def run_api_jobs_parallel(
    backend: "ApiVLMBackend",
    jobs: list[ApiVisionJob],
    *,
    workers: int = 1,
    on_result: Callable[[ApiVisionResult], None] | None = None,
) -> list[ApiVisionResult]:
    """
    Run independent vision API jobs with up to *workers* concurrent HTTP calls.

    Returns results in the same order as *jobs* (not completion order).
    Sequential multi-step chains (verb depends on instrument) must stay serial per sample.
    """
    if not jobs:
        return []
    n_workers = max(1, min(int(workers), len(jobs)))
    if n_workers == 1:
        out = [run_api_job(backend, job) for job in jobs]
        if on_result:
            for r in out:
                on_result(r)
        return out

    by_id: dict[Any, ApiVisionResult] = {}
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(run_api_job, backend, job): job for job in jobs}
        for fut in as_completed(futures):
            result = fut.result()
            by_id[result.job_id] = result
            if on_result:
                on_result(result)
    return [by_id[job.job_id] for job in jobs]


def api_parallel_enabled(backend: Any, workers: int) -> bool:
    return isinstance(backend, ApiVLMBackend) and max(1, int(workers)) > 1


class ApiVLMBackend:
    """VLMBackend-compatible wrapper for cloud vision APIs."""

    def __init__(
        self,
        *,
        provider: str,
        model: str,
        api_key: str,
        image_size: int = 384,
        timeout_sec: int = 120,
        api_workers: int = 1,
    ):
        self.provider = provider
        self.model = model
        self.api_key = api_key
        self.image_size = image_size
        self.timeout_sec = timeout_sec
        self.api_workers = max(1, int(api_workers))

    def to(self, device: str, dtype: Any = None) -> None:
        return None

    def get_prompt_builder(self, system_prompt: str | None = None) -> Any:
        return None

    def generate(self, image: Image.Image, prompt: str, **gen_kw: Any) -> str:
        temperature = float(gen_kw.get("temperature", 0.0))
        if not gen_kw.get("do_sample", temperature > 0):
            temperature = 0.0
        max_tokens = int(gen_kw.get("max_new_tokens", 512))
        timeout_sec = int(gen_kw.get("request_timeout_sec", self.timeout_sec))
        cache_key = gen_kw.get("image_cache_key")
        if cache_key is not None:
            cache_key = str(cache_key)
        return api_vision_generate(
            provider=self.provider,
            model=self.model,
            image=image,
            prompt=prompt.strip(),
            api_key=self.api_key,
            image_max_side=self.image_size,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout_sec=timeout_sec,
            image_cache_key=cache_key,
        )

    def generate_parallel(
        self,
        jobs: list[ApiVisionJob],
        *,
        workers: int | None = None,
        on_result: Callable[[ApiVisionResult], None] | None = None,
    ) -> list[ApiVisionResult]:
        return run_api_jobs_parallel(
            self,
            jobs,
            workers=workers if workers is not None else self.api_workers,
            on_result=on_result,
        )


def load_api_backend(
    backend: str,
    model_id: str,
    *,
    api_key: str | None = None,
    api_key_file: Path | str | None = None,
    image_size: int = 384,
    timeout_sec: int = 120,
    api_workers: int = 1,
) -> tuple[ApiVLMBackend, dict[str, Any]]:
    provider = api_provider_for_backend(backend)
    key = resolve_api_key(backend, explicit=api_key, key_file=api_key_file)
    workers = max(1, int(api_workers))
    backend_obj = ApiVLMBackend(
        provider=provider,
        model=model_id,
        api_key=key,
        image_size=image_size,
        timeout_sec=timeout_sec,
        api_workers=workers,
    )
    meta = {
        "source": "cloud_api",
        "provider": provider,
        "model_id": model_id,
        "hub_model_id": model_id,
        "image_side": image_size,
        "bbox_coord_space": "normalized_01",
        "api_workers": workers,
    }
    return backend_obj, meta
