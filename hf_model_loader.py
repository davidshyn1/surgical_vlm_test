"""
hf_model_loader.py

Load Hugging Face VLMs via transformers AutoProcessor / AutoModel*.
Used by grounding tasks for InternVL, Qwen, PaliGemma, Cosmos-Reason2, etc.

Prismatic checkpoints use backends.load_backend(..., backend="prismatic") only.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoProcessor,
    AutoTokenizer,
)

_REPO_ROOT = Path(__file__).resolve().parent.parent
# Fixed local Hub snapshot root (override with HF_HUB_CACHE env if needed).
DEFAULT_HF_HOME = _REPO_ROOT / ".cache" / "huggingface"
DEFAULT_HF_HUB_CACHE = DEFAULT_HF_HOME / "hub"

PRISMATIC_VLMS_REPO = "TRI-ML/prismatic-vlms"
PRISMATIC_VLMS_SUBDIRS = (
    "prism-dinosiglip-224px+7b",
    "prism-dinosiglip+7b",
)

# Optional: GR00T-H Eagle bundle fix when loading nvidia/GR00T-H via HF path
_EAGLE_ASSET_NAMES = (
    "added_tokens.json",
    "chat_template.json",
    "config.json",
    "generation_config.json",
    "merges.txt",
    "preprocessor_config.json",
    "processor_config.json",
    "special_tokens_map.json",
    "tokenizer_config.json",
    "vocab.json",
)
_EAGLE_UPSTREAM_BASE = (
    "https://raw.githubusercontent.com/NVIDIA-Medtech/GR00T-H/main/"
    "gr00t/model/modules/nvidia/Eagle-Block2A-2B-v2"
)


def configure_hf_cache(hub_cache: str | Path | None = None) -> Path:
    """
  Pin Hugging Face caches under ``<surgical>/.cache/huggingface/``.

  Hub weights resolve from ``.../hub`` (``HF_HUB_CACHE``).
  Safe to call repeatedly; only fills unset env vars unless *hub_cache* is passed.
  """
    hub = Path(hub_cache or os.environ.get("HF_HUB_CACHE") or DEFAULT_HF_HUB_CACHE).resolve()
    home = hub.parent
    transformers_cache = home / "transformers"

    if hub_cache is not None:
        os.environ["HF_HUB_CACHE"] = str(hub)
        os.environ["HUGGINGFACE_HUB_CACHE"] = str(hub)
        os.environ["HF_HOME"] = str(home)
        os.environ["TRANSFORMERS_CACHE"] = str(transformers_cache)
    else:
        os.environ.setdefault("HF_HUB_CACHE", str(hub))
        os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(hub))
        os.environ.setdefault("HF_HOME", str(home))
        os.environ.setdefault("TRANSFORMERS_CACHE", str(transformers_cache))
    return hub


def _hub_cache_dir(cache_dir: str | Path | None) -> str:
    """Directory passed to ``from_pretrained(..., cache_dir=...)`` (Hub snapshots)."""
    if cache_dir is not None:
        return str(Path(cache_dir).resolve())
    return str(configure_hf_cache())


# Apply repo-default cache before any Hub download when this module is imported.
configure_hf_cache()


def _as_torch_device(device: str | torch.device) -> torch.device:
    if isinstance(device, torch.device):
        return device
    return torch.device(device or "cuda")


def _device_map_arg(device: str | torch.device) -> str:
    """String for transformers ``device_map`` (from_pretrained)."""
    dev = _as_torch_device(device)
    if dev.type == "cpu":
        return "cpu"
    if dev.index is not None:
        return f"cuda:{dev.index}"
    return "cuda"


def _dtype_for_device(device: str | torch.device) -> torch.dtype:
    dev = _as_torch_device(device)
    return torch.bfloat16 if dev.type == "cuda" and torch.cuda.is_available() else torch.float32


def ensure_eagle_block2a_bundle() -> None:
    import urllib.error
    import urllib.request

    try:
        import gr00t.model.modules.eagle_backbone as eagle_backbone
    except ImportError:
        return
    bundle = Path(eagle_backbone.__file__).resolve().parent / "nvidia" / "Eagle-Block2A-2B-v2"
    bundle.mkdir(parents=True, exist_ok=True)
    for name in _EAGLE_ASSET_NAMES:
        dest = bundle / name
        if dest.is_file() and dest.stat().st_size > 0:
            continue
        url = f"{_EAGLE_UPSTREAM_BASE}/{name}"
        req = urllib.request.Request(url, headers={"User-Agent": "surgical-hf-loader/1.0"})
        with urllib.request.urlopen(req, timeout=120) as resp:
            dest.write_bytes(resp.read())


def _is_qwen_vl_model_type(model_type: str) -> bool:
    mt = (model_type or "").lower()
    return "qwen" in mt and "vl" in mt


_INTERNVL_PYTHON_DEPS = ("timm", "einops")


def _ensure_internvl_deps(model_id: str) -> None:
    if "internvl" not in model_id.lower():
        return
    missing: list[str] = []
    for pkg in _INTERNVL_PYTHON_DEPS:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        raise ImportError(
            f"InternVL load requires: {', '.join(missing)}. "
            f"In your HF_PYTHON env run: pip install {' '.join(missing)}"
        ) from None


def _infer_default_image_side(processor: Any, config: Any) -> int:
    for attr in ("image_processor", "video_processor"):
        proc = getattr(processor, attr, None)
        if proc is None:
            continue
        size = getattr(proc, "size", None)
        if isinstance(size, dict):
            if "shortest_edge" in size:
                return int(size["shortest_edge"])
            if "height" in size and "width" in size:
                return int(size["height"])
        if isinstance(size, (list, tuple)) and size:
            return int(size[0])
    mt = str(getattr(config, "model_type", "") or "").lower()
    if "paligemma" in mt:
        return 224
    if "internvl" in mt:
        return 448
    return 384


def load_hf_vlm(
    model_id: str,
    *,
    hf_token: str | None = None,
    device: str | torch.device = "cuda",
    cache_dir: str | Path | None = None,
    trust_remote_code: bool = True,
) -> tuple[Any, Any | None, dict[str, Any]]:
    """
    Load a HF Hub VLM with AutoProcessor + best-effort AutoModel class.

    Returns (model, processor, meta).
    """
    cache = _hub_cache_dir(cache_dir)
    token = hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    dtype = _dtype_for_device(device)
    _ensure_internvl_deps(model_id)

    if "GR00T" in model_id.upper() or model_id.strip() in ("nvidia/GR00T-H",):
        ensure_eagle_block2a_bundle()
        try:
            import gr00t.model.gr00t_n1d6.gr00t_n1d6  # noqa: F401
        except ImportError:
            try:
                import gr00t.model  # noqa: F401
            except ImportError as e:
                raise ImportError(
                    "GR00T-H requires the gr00t package (pip install -e ../backend/GR00T-H)."
                ) from e

    config = AutoConfig.from_pretrained(
        model_id,
        cache_dir=cache,
        trust_remote_code=trust_remote_code,
        token=token,
    )
    model_type = str(getattr(config, "model_type", "") or "")

    processor: Any | None = None
    try:
        processor = AutoProcessor.from_pretrained(
            model_id,
            cache_dir=cache,
            trust_remote_code=trust_remote_code,
            token=token,
        )
    except Exception:
        try:
            processor = AutoTokenizer.from_pretrained(
                model_id,
                cache_dir=cache,
                trust_remote_code=trust_remote_code,
                token=token,
            )
        except Exception:
            processor = None

    model: Any | None = None
    loader_order: list[type] = []
    if _is_qwen_vl_model_type(model_type):
        import transformers

        if hasattr(transformers, "Qwen3VLForConditionalGeneration"):
            loader_order.append(transformers.Qwen3VLForConditionalGeneration)
        if hasattr(transformers, "Qwen2VLForConditionalGeneration"):
            loader_order.append(transformers.Qwen2VLForConditionalGeneration)
    loader_order.extend(
        [
            AutoModelForImageTextToText,
            AutoModelForCausalLM,
            AutoModel,
        ]
    )
    seen: set[type] = set()
    last_err: Exception | None = None
    for loader_cls in loader_order:
        if loader_cls in seen:
            continue
        seen.add(loader_cls)
        try:
            load_kw: dict[str, Any] = {
                "cache_dir": cache,
                "trust_remote_code": trust_remote_code,
                "token": token,
            }
            load_kw["dtype"] = dtype
            load_kw["torch_dtype"] = dtype  # older transformers
            if device != "auto":
                load_kw["device_map"] = _device_map_arg(device)
            if _is_qwen_vl_model_type(model_type):
                load_kw["attn_implementation"] = "sdpa"
            model = loader_cls.from_pretrained(model_id, **load_kw)
            break
        except Exception as e:
            last_err = e
            model = None

    if model is None:
        raise RuntimeError(f"Failed to load model {model_id!r}: {last_err}") from last_err

    image_side = _infer_default_image_side(processor, config)
    meta = {
        "source": "hf_autoprocessor",
        "hub_model_id": model_id,
        "model_id": model_id,
        "model_type": model_type,
        "loader": type(model).__name__,
        "processor": type(processor).__name__ if processor is not None else None,
        "image_side": image_side,
        "bbox_coord_space": "qwen_1000" if _is_qwen_vl_model_type(model_type) else "normalized_01",
    }
    return model, processor, meta


def is_prismatic_hub_id(model_id: str) -> bool:
    return model_id.startswith(f"{PRISMATIC_VLMS_REPO}/")
