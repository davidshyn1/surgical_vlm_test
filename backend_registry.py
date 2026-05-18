"""Backend names, defaults, and helpers for grounding eval scripts."""

from __future__ import annotations

import re
from pathlib import Path

# CLI --backend choices (prismatic = TRI-ML loader; others = HF AutoProcessor)
BACKEND_CHOICES: tuple[str, ...] = (
    "prismatic",
    "hf",
    # Cosmos-Reason2 (size-specific)
    "cosmos",  # default 2B
    "cosmos-2b",
    "cosmos-32b",
    # Qwen3-VL (size-specific)
    "qwen3",  # default 4B Instruct
    "qwen3-4b",
    "qwen3-32b",
    # other HF families
    "groot",
    "qwen",
    "qwen2.5",
    "internvl",
    "internvl3.5",
    "paligemma",
    "paligemma2",
    # Cloud APIs (vision)
    "openai",
    "gpt",
    "chatgpt",
    "gemini",
    "claude",
    "anthropic",
)

# Hub model_id per --backend (override with --model-id)
DEFAULT_MODEL_IDS: dict[str, str] = {
    "prismatic": "prism-dinosiglip+7b",
    "hf": "Qwen/Qwen3-VL-4B-Instruct",
    # Cosmos-Reason2
    "cosmos": "nvidia/Cosmos-Reason2-2B",
    "cosmos-2b": "nvidia/Cosmos-Reason2-2B",
    "cosmos-32b": "nvidia/Cosmos-Reason2-32B",
    # Qwen3-VL
    "qwen": "Qwen/Qwen3-VL-4B-Instruct",
    "qwen3": "Qwen/Qwen3-VL-4B-Instruct",
    "qwen3-4b": "Qwen/Qwen3-VL-4B-Instruct",
    "qwen3-32b": "Qwen/Qwen3-VL-32B-Instruct",
    "qwen2.5": "Qwen/Qwen2.5-VL-32B-Instruct",
    # InternVL (use *-HF for transformers AutoProcessor; non-HF is OpenGVLab custom format)
    "internvl": "OpenGVLab/InternVL3_5-38B-HF",
    "internvl3.5": "OpenGVLab/InternVL3_5-38B-HF",
    "paligemma": "google/paligemma2-28b-pt-224",
    "paligemma2": "google/paligemma2-28b-pt-224",
    "groot": "nvidia/GR00T-H",
    # Cloud APIs (--model-id = API model name)
    "openai": "gpt-4o",
    "gpt": "gpt-4o",
    "chatgpt": "gpt-4o-mini",
    "gemini": "gemini-2.0-flash",
    "claude": "claude-sonnet-4-20250514",
    "anthropic": "claude-sonnet-4-20250514",
}

# Default output folder slug when --model-name is omitted (or "original")
BACKEND_OUTPUT_SLUGS: dict[str, str] = {
    "prismatic": "prismatic-7b",
    "hf": "qwen3-vl-4b",
    "cosmos": "cosmos-reason2-2b",
    "cosmos-2b": "cosmos-reason2-2b",
    "cosmos-32b": "cosmos-reason2-32b",
    "qwen": "qwen3-vl-4b",
    "qwen3": "qwen3-vl-4b",
    "qwen3-4b": "qwen3-vl-4b",
    "qwen3-32b": "qwen3-vl-32b",
    "qwen2.5": "qwen2.5-vl-7b",
    "internvl": "internvl3.5-38b",
    "internvl3.5": "internvl3.5-38b",
    "paligemma": "paligemma2-28b",
    "paligemma2": "paligemma2-28b",
    "groot": "groot-h",
    "openai": "gpt-4o",
    "gpt": "gpt-4o",
    "chatgpt": "gpt-4o-mini",
    "gemini": "gemini-2.0-flash",
    "claude": "claude-sonnet-4",
    "anthropic": "claude-sonnet-4",
}

# Accept cosmos2b, qwen3_32b, etc. on CLI / BACKEND= env
_BACKEND_KEY_ALIASES: dict[str, str] = {
    "cosmos2b": "cosmos-2b",
    "cosmos_2b": "cosmos-2b",
    "cosmos32b": "cosmos-32b",
    "cosmos_32b": "cosmos-32b",
    "qwen3_4b": "qwen3-4b",
    "qwen3vl4b": "qwen3-4b",
    "qwen3-vl-4b": "qwen3-4b",
    "qwen3_32b": "qwen3-32b",
    "qwen3vl32b": "qwen3-32b",
    "qwen3-vl-32b": "qwen3-32b",
}

_API_ALIASES = frozenset(
    {
        "openai",
        "gpt",
        "chatgpt",
        "gemini",
        "google",
        "claude",
        "anthropic",
    }
)

_HF_ALIASES = frozenset(
    {
        "hf",
        "cosmos",
        "cosmos-2b",
        "cosmos-32b",
        "groot",
        "qwen",
        "qwen2.5",
        "qwen3",
        "qwen3-4b",
        "qwen3-32b",
        "internvl",
        "internvl3.5",
        "paligemma",
        "paligemma2",
    }
)


def normalize_backend_key(backend: str) -> str:
    key = (backend or "").strip().lower().replace("_", "-")
    if key in _BACKEND_KEY_ALIASES:
        return _BACKEND_KEY_ALIASES[key]
    compact = key.replace("-", "")
    if compact in _BACKEND_KEY_ALIASES:
        return _BACKEND_KEY_ALIASES[compact]
    return key


def is_prismatic_backend(backend: str) -> bool:
    return normalize_backend_key(backend) == "prismatic"


def is_hf_backend(backend: str) -> bool:
    return normalize_backend_key(backend) in _HF_ALIASES


def is_api_backend(backend: str) -> bool:
    return normalize_backend_key(backend) in _API_ALIASES


def resolve_hf_token(backend: str, token_path: Path) -> str | None:
    """Return HF hub token text, or None for cloud API backends (no Hub download)."""
    if is_api_backend(backend):
        return None
    return token_path.resolve().read_text(encoding="utf-8").strip()


def resolve_model_id(backend: str, model_id: str | None) -> str:
    if model_id:
        return model_id
    key = normalize_backend_key(backend)
    if key not in DEFAULT_MODEL_IDS:
        raise ValueError(
            f"No default model_id for backend {backend!r}. Pass --model-id explicitly. "
            f"Known backends: {', '.join(BACKEND_CHOICES)}"
        )
    return DEFAULT_MODEL_IDS[key]


def hub_id_to_output_slug(model_id: str) -> str:
    tail = model_id.rsplit("/", 1)[-1].strip()
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "_", tail).strip("_").lower()
    return slug or "model"


def resolve_output_model_name(
    backend: str,
    model_id: str,
    user_name: str | None = None,
) -> str:
    """
    Folder slug for outputs/.../{backend}_{model_name}/.

    Uses --model-name when set (and not the legacy placeholder 'original').
    Otherwise uses BACKEND_OUTPUT_SLUGS or the Hub repo tail.
    """
    name = (user_name or "").strip()
    if name and name.lower() != "original":
        return re.sub(r"[^a-zA-Z0-9._-]+", "_", name).strip("_") or "model"
    key = normalize_backend_key(backend)
    if key in BACKEND_OUTPUT_SLUGS:
        return BACKEND_OUTPUT_SLUGS[key]
    return hub_id_to_output_slug(model_id)


def bbox_parse_mode(backend: str, meta: dict | None = None) -> str:
    """Return 'cosmos' for Qwen-VL 0–1000 scale parsing, else 'prismatic' style [0,1]."""
    if is_prismatic_backend(backend):
        return "prismatic"
    if meta and str(meta.get("bbox_coord_space") or "") == "qwen_1000":
        return "cosmos"
    mt = str((meta or {}).get("model_type") or "").lower()
    if "qwen" in mt and "vl" in mt:
        return "cosmos"
    return "prismatic"
