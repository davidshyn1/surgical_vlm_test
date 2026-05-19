"""
hf_model_loader.py

Load Hugging Face VLMs via transformers AutoProcessor / AutoModel*.
Used by grounding tasks for InternVL, Qwen, PaliGemma, Cosmos-Reason2, etc.

Prismatic checkpoints use backends.load_backend(..., backend="prismatic") only.
"""

from __future__ import annotations

import json
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


def _set_eval_mode(model: Any) -> Any:
    """Inference: disable dropout / batchnorm train behavior after load."""
    if hasattr(model, "eval"):
        model.eval()
    return model


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


def _split_hub_model_id(model_id: str) -> tuple[str, str | None]:
    """
    ``org/repo/subfolder`` → (``org/repo``, ``subfolder``).
    ``org/repo`` → (``org/repo``, None).
    """
    parts = model_id.strip().strip("/").split("/")
    if len(parts) <= 2:
        return model_id.strip(), None
    return "/".join(parts[:2]), "/".join(parts[2:])


def _hf_download_file(
    repo_id: str,
    filename: str,
    *,
    cache_dir: str,
    token: str | None,
) -> Path | None:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        return None
    try:
        path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            cache_dir=cache_dir,
            token=token,
        )
        return Path(path)
    except Exception:
        return None


def resolve_peft_adapter_dir(
    model_id: str,
    *,
    cache_dir: str,
    token: str | None = None,
) -> Path | None:
    """
    Return a local directory containing ``adapter_config.json``, or None.
    Supports a local path or Hub ids like ``khtks/Qwen3-VL/surgsigma_qwen3vl_full``.
    """
    local = Path(model_id).expanduser()
    if local.is_dir() and (local / "adapter_config.json").is_file():
        return local.resolve()

    repo_id, subfolder = _split_hub_model_id(model_id)

    if subfolder is not None:
        try:
            from huggingface_hub import snapshot_download
        except ImportError:
            pass
        else:
            try:
                root = snapshot_download(
                    repo_id=repo_id,
                    allow_patterns=[f"{subfolder}/*"],
                    cache_dir=cache_dir,
                    token=token,
                )
                adapter_dir = Path(root) / subfolder
                if (adapter_dir / "adapter_config.json").is_file():
                    return adapter_dir.resolve()
            except Exception:
                pass

    rel_cfg = (
        f"{subfolder}/adapter_config.json" if subfolder else "adapter_config.json"
    )
    cfg_path = _hf_download_file(
        repo_id,
        rel_cfg,
        cache_dir=cache_dir,
        token=token,
    )
    if cfg_path is not None and cfg_path.is_file():
        return cfg_path.parent.resolve()
    return None


def read_peft_adapter_config(adapter_dir: Path) -> dict[str, Any]:
    path = adapter_dir / "adapter_config.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _peft_adapter_has_weights(adapter_dir: Path) -> bool:
    return (adapter_dir / "adapter_model.safetensors").is_file() or (
        adapter_dir / "adapter_model.bin"
    ).is_file()


def _ensure_peft_adapter_weights(
    adapter_dir: Path,
    adapter_model_id: str,
    *,
    cache_dir: str,
    token: str | None,
) -> Path:
    """Download ``adapter_model.*`` when only ``adapter_config.json`` is cached."""
    if _peft_adapter_has_weights(adapter_dir):
        return adapter_dir

    repo_id, subfolder = _split_hub_model_id(adapter_model_id)
    if subfolder is None:
        raise FileNotFoundError(
            f"PEFT adapter weights missing under {adapter_dir}; "
            "expected adapter_model.safetensors or adapter_model.bin."
        )

    try:
        from huggingface_hub import hf_hub_download, snapshot_download
    except ImportError as e:
        raise ImportError(
            "huggingface_hub is required to download PEFT adapter weights."
        ) from e

    last_err: Exception | None = None
    for fname in ("adapter_model.safetensors", "adapter_model.bin"):
        try:
            hf_hub_download(
                repo_id=repo_id,
                filename=f"{subfolder}/{fname}",
                cache_dir=cache_dir,
                token=token,
            )
            break
        except Exception as e:
            last_err = e
    else:
        try:
            snapshot_download(
                repo_id=repo_id,
                allow_patterns=[f"{subfolder}/adapter_model.*"],
                cache_dir=cache_dir,
                token=token,
            )
        except Exception as e:
            raise FileNotFoundError(
                f"Could not download PEFT weights for {adapter_model_id!r}."
            ) from (last_err or e)

    refreshed = resolve_peft_adapter_dir(
        adapter_model_id,
        cache_dir=cache_dir,
        token=token,
    )
    if refreshed is not None and _peft_adapter_has_weights(refreshed):
        return refreshed
    if _peft_adapter_has_weights(adapter_dir):
        return adapter_dir
    raise FileNotFoundError(
        f"PEFT adapter weights still missing for {adapter_model_id!r} under {adapter_dir}."
    )


def _peft_torch_device(device: str | torch.device) -> str:
    dev = str(device)
    if dev == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return dev


def _load_peft_adapter_on_base(
    base_model: Any,
    adapter_dir: Path,
    *,
    device: str | torch.device,
) -> Any:
    """
    Attach a local PEFT adapter directory to ``base_model``.

    Avoids ``PeftModel.from_pretrained`` Hub paths that break when only
    ``adapter_config.json`` is cached (``HFValidationError``) or when ``peft``
    passes deprecated ``use_auth_token`` to ``huggingface_hub>=1.14``.
    """
    from peft import PeftModel
    from peft.mapping import PEFT_TYPE_TO_CONFIG_MAPPING
    from peft.utils.save_and_load import load_peft_weights, set_peft_model_state_dict

    adapter_cfg = read_peft_adapter_config(adapter_dir)
    peft_type = str(adapter_cfg.get("peft_type") or "")
    if peft_type not in PEFT_TYPE_TO_CONFIG_MAPPING:
        raise ValueError(f"Unsupported peft_type {peft_type!r} in {adapter_dir}")

    config_cls = PEFT_TYPE_TO_CONFIG_MAPPING[peft_type]
    peft_config = config_cls.from_pretrained(str(adapter_dir))
    peft_config.inference_mode = True
    model = PeftModel(base_model, peft_config, adapter_name="default")
    weights = load_peft_weights(str(adapter_dir), device=_peft_torch_device(device))
    set_peft_model_state_dict(model, weights, adapter_name="default")
    return model


def _load_processor(
    model_id: str,
    *,
    cache_dir: str,
    token: str | None,
    trust_remote_code: bool,
) -> Any | None:
    for mid in (model_id,):
        try:
            return AutoProcessor.from_pretrained(
                mid,
                cache_dir=cache_dir,
                trust_remote_code=trust_remote_code,
                token=token,
            )
        except Exception:
            pass
        try:
            return AutoTokenizer.from_pretrained(
                mid,
                cache_dir=cache_dir,
                trust_remote_code=trust_remote_code,
                token=token,
            )
        except Exception:
            pass
    return None


def _build_model_loader_order(model_type: str) -> list[type]:
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
    return loader_order


def _load_pretrained_model(
    model_id: str,
    *,
    model_type: str,
    cache_dir: str,
    token: str | None,
    trust_remote_code: bool,
    dtype: torch.dtype,
    device: str | torch.device,
) -> Any:
    loader_order = _build_model_loader_order(model_type)
    seen: set[type] = set()
    last_err: Exception | None = None
    for loader_cls in loader_order:
        if loader_cls in seen:
            continue
        seen.add(loader_cls)
        try:
            load_kw: dict[str, Any] = {
                "cache_dir": cache_dir,
                "trust_remote_code": trust_remote_code,
                "token": token,
            }
            load_kw["dtype"] = dtype
            load_kw["torch_dtype"] = dtype
            if device != "auto":
                load_kw["device_map"] = _device_map_arg(device)
            if _is_qwen_vl_model_type(model_type):
                load_kw["attn_implementation"] = "sdpa"
            return loader_cls.from_pretrained(model_id, **load_kw)
        except Exception as e:
            last_err = e
    raise RuntimeError(f"Failed to load model {model_id!r}: {last_err}") from last_err


def _should_merge_peft() -> bool:
    return os.environ.get("HF_PEFT_MERGE", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def load_hf_vlm_peft_adapter(
    adapter_model_id: str,
    *,
    hf_token: str | None = None,
    device: str | torch.device = "cuda",
    cache_dir: str | Path | None = None,
    trust_remote_code: bool = True,
    base_model_id: str | None = None,
) -> tuple[Any, Any | None, dict[str, Any]]:
    """Load base VLM + PEFT LoRA adapter (e.g. surgsigma_qwen3vl_full on Qwen3-VL-4B)."""
    cache = _hub_cache_dir(cache_dir)
    token = hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    dtype = _dtype_for_device(device)

    adapter_dir = resolve_peft_adapter_dir(
        adapter_model_id,
        cache_dir=cache,
        token=token,
    )
    if adapter_dir is None:
        raise FileNotFoundError(
            f"PEFT adapter not found for {adapter_model_id!r} "
            "(expected adapter_config.json)."
        )

    adapter_cfg = read_peft_adapter_config(adapter_dir)
    base_id = (
        base_model_id
        or os.environ.get("HF_BASE_MODEL_ID", "").strip()
        or str(adapter_cfg.get("base_model_name_or_path") or "").strip()
    )
    if not base_id:
        raise ValueError(
            f"adapter_config.json under {adapter_dir} has no base_model_name_or_path; "
            "set HF_BASE_MODEL_ID."
        )

    _ensure_internvl_deps(base_id)

    config = AutoConfig.from_pretrained(
        base_id,
        cache_dir=cache,
        trust_remote_code=trust_remote_code,
        token=token,
    )
    model_type = str(getattr(config, "model_type", "") or "")

    processor = _load_processor(
        str(adapter_dir),
        cache_dir=cache,
        token=token,
        trust_remote_code=trust_remote_code,
    )
    if processor is None:
        processor = _load_processor(
            base_id,
            cache_dir=cache,
            token=token,
            trust_remote_code=trust_remote_code,
        )

    base_model = _load_pretrained_model(
        base_id,
        model_type=model_type,
        cache_dir=cache,
        token=token,
        trust_remote_code=trust_remote_code,
        dtype=dtype,
        device=device,
    )

    try:
        import peft  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "PEFT adapter load requires the `peft` package. "
            "In your HF_PYTHON env run: pip install peft"
        ) from e

    adapter_dir = _ensure_peft_adapter_weights(
        adapter_dir,
        adapter_model_id,
        cache_dir=cache,
        token=token,
    )
    model = _load_peft_adapter_on_base(
        base_model,
        adapter_dir,
        device=device,
    )
    merged = False
    if _should_merge_peft():
        model = model.merge_and_unload()
        merged = True

    model = _set_eval_mode(model)

    image_side = _infer_default_image_side(processor, config)
    meta = {
        "source": "hf_peft_adapter",
        "hub_model_id": adapter_model_id,
        "model_id": adapter_model_id,
        "peft_adapter_dir": str(adapter_dir),
        "base_model_id": base_id,
        "peft_type": adapter_cfg.get("peft_type"),
        "peft_merged": merged,
        "model_type": model_type,
        "loader": type(model).__name__,
        "processor": type(processor).__name__ if processor is not None else None,
        "image_side": image_side,
        "bbox_coord_space": "qwen_1000" if _is_qwen_vl_model_type(model_type) else "normalized_01",
    }
    return model, processor, meta


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
    if looks_like_prismatic_model_id(model_id):
        raise ValueError(
            f"Model id {model_id!r} is a TRI-ML Prismatic checkpoint. "
            "Use backends.load_backend(backend='prismatic') or "
            "BACKEND=prismatic bash grounding_task.sh <task> (not qwen3/hf AutoProcessor)."
        )

    cache = _hub_cache_dir(cache_dir)
    token = hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    dtype = _dtype_for_device(device)

    adapter_dir = resolve_peft_adapter_dir(
        model_id,
        cache_dir=cache,
        token=token,
    )
    if adapter_dir is not None:
        return load_hf_vlm_peft_adapter(
            model_id,
            hf_token=hf_token,
            device=device,
            cache_dir=cache_dir,
            trust_remote_code=trust_remote_code,
        )

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

    processor = _load_processor(
        model_id,
        cache_dir=cache,
        token=token,
        trust_remote_code=trust_remote_code,
    )

    model = _load_pretrained_model(
        model_id,
        model_type=model_type,
        cache_dir=cache,
        token=token,
        trust_remote_code=trust_remote_code,
        dtype=dtype,
        device=device,
    )
    model = _set_eval_mode(model)

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


def looks_like_prismatic_model_id(model_id: str) -> bool:
    """True for TRI-ML Prismatic ids (must use ``backend='prismatic'``, not HF AutoProcessor)."""
    mid = (model_id or "").strip()
    if not mid:
        return False
    if is_prismatic_hub_id(mid):
        return True
    # Registry ids, e.g. prism-dinosiglip+7b
    if mid.startswith("prism-") and "+" in mid:
        return True
    return mid in PRISMATIC_VLMS_SUBDIRS
