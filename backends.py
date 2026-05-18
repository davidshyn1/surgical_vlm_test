"""
backends.py

Unified VLM backends for surgical grounding eval.

- prismatic: TRI-ML/prismatic-vlms via local backend package or HF checkpoint (.pt + config.json)
- hf (+ aliases qwen3, internvl, paligemma2, cosmos, groot, …): transformers AutoProcessor path
"""

from __future__ import annotations

import json
import os
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from api_backends import load_api_backend
from backend_registry import (
    is_api_backend,
    is_hf_backend,
    is_prismatic_backend,
    normalize_backend_key,
    resolve_model_id,
)
from hf_model_loader import load_hf_vlm

_REPO_ROOT = Path(__file__).resolve().parent.parent
_BACKEND_ROOT = _REPO_ROOT / "backend"
_PRISMATIC_PKG = _BACKEND_ROOT / "prismatic" / "prismatic"


class VLMBackend(ABC):
    @abstractmethod
    def generate(self, image: Image.Image, prompt: str, **gen_kw: Any) -> str:
        ...

    @abstractmethod
    def to(self, device: str, dtype: torch.dtype | None = None) -> None:
        ...

    def get_prompt_builder(self, system_prompt: str | None = None) -> Any:
        return None


class PrismaticBackend(VLMBackend):
    def __init__(self, vlm: Any, image_size: int = 384):
        self.vlm = vlm
        self.image_size = image_size

    def to(self, device: str, dtype: torch.dtype | None = None) -> None:
        self.vlm.to(device, dtype=dtype)

    def get_prompt_builder(self, system_prompt: str | None = None) -> Any:
        from prismatic.models.backbones.llm.prompting import PurePromptBuilder

        return PurePromptBuilder("prism-dinosiglip+7b", model_family="prism")

    def generate(self, image: Image.Image, prompt: str, **gen_kw: Any) -> str:
        from prismatic import load as prismatic_load

        return prismatic_load.generate(
            self.vlm,
            image,
            prompt,
            temperature=gen_kw.get("temperature", 0.2),
            max_new_tokens=gen_kw.get("max_new_tokens", 512),
        )


def _processor_has_chat_template(processor: Any) -> bool:
    for obj in (processor, getattr(processor, "tokenizer", None)):
        if obj is None:
            continue
        if getattr(obj, "chat_template", None):
            return True
    return False


def _hf_model_device(model: Any) -> torch.device:
    dev = getattr(model, "device", None)
    if dev is not None:
        return dev
    return next(model.parameters()).device


def _move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, val in batch.items():
        out[key] = val.to(device) if hasattr(val, "to") else val
    return out


def _is_paligemma_processor(
    processor: Any,
    *,
    model_type: str = "",
    model_id: str = "",
) -> bool:
    mt = (model_type or "").lower()
    mid = (model_id or "").lower()
    proc_name = type(processor).__name__.lower()
    return "paligemma" in mt or "paligemma" in mid or "paligemma" in proc_name


def _format_processor_text_prompt(
    prompt: str,
    *,
    processor: Any,
    model_type: str = "",
    model_id: str = "",
    num_images: int = 1,
) -> str:
    """PaliGemma expects leading ``<image>`` token(s), one per image."""
    text = prompt.strip()
    if not _is_paligemma_processor(processor, model_type=model_type, model_id=model_id):
        return text
    if text.lstrip().startswith("<image>"):
        return text
    n = max(1, int(num_images))
    prefix = "".join("<image>" for _ in range(n))
    return f"{prefix}\n{text}" if text else prefix


def _prepare_hf_vlm_inputs(
    processor: Any,
    model: Any,
    image: Image.Image,
    prompt: str,
    *,
    model_type: str = "",
    model_id: str = "",
) -> dict[str, Any]:
    """
    Build model inputs for HF VLMs.

    - Qwen-VL / InternVL / Cosmos: ``apply_chat_template`` (when template exists)
    - PaliGemma and others without template: ``processor(text=..., images=...)``
    """
    device = _hf_model_device(model)
    text_prompt = _format_processor_text_prompt(
        prompt,
        processor=processor,
        model_type=model_type,
        model_id=model_id,
        num_images=1,
    )
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": text_prompt},
            ],
        }
    ]

    if _processor_has_chat_template(processor) and hasattr(processor, "apply_chat_template"):
        try:
            inputs = processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
            inputs = _move_batch_to_device(inputs, device)
            if "pixel_values" not in inputs and hasattr(processor, "__call__"):
                try:
                    extra = processor(
                        text=text_prompt, images=image, return_tensors="pt",
                    )
                    for key, val in extra.items():
                        if key not in inputs and hasattr(val, "to"):
                            inputs[key] = val.to(device)
                except Exception:
                    pass
            return inputs
        except (ValueError, TypeError) as err:
            err_msg = str(err).lower()
            if "chat template" not in err_msg and "jinja" not in err_msg:
                raise

    if not hasattr(processor, "__call__"):
        raise RuntimeError(
            "HF processor has no chat template and is not callable with text+images."
        )
    inputs = processor(text=text_prompt, images=image, return_tensors="pt")
    return _move_batch_to_device(inputs, device)


def _decode_hf_generation(
    processor: Any,
    inputs: dict[str, Any],
    gen_ids: Any,
) -> str:
    input_ids = inputs.get("input_ids")
    if input_ids is not None:
        trimmed = [out[len(inp) :] for inp, out in zip(input_ids, gen_ids)]
    else:
        trimmed = gen_ids

    if hasattr(processor, "batch_decode"):
        return processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()
    tok = getattr(processor, "tokenizer", None)
    if tok is not None and hasattr(tok, "batch_decode"):
        return tok.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()
    if hasattr(processor, "decode"):
        row = trimmed[0] if hasattr(trimmed, "__getitem__") else trimmed
        return processor.decode(row, skip_special_tokens=True).strip()
    raise RuntimeError("Cannot decode model output: no batch_decode on processor/tokenizer.")


class HfAutoBackend(VLMBackend):
    """Hugging Face VLM via AutoProcessor (chat template or text+images fallback)."""

    def __init__(
        self,
        model: Any,
        processor: Any,
        *,
        image_size: int = 384,
        model_id: str = "",
        model_type: str = "",
    ):
        self.model = model
        self.processor = processor
        self.image_size = image_size
        self.model_id = model_id
        self.model_type = model_type

    def to(self, device: str, dtype: torch.dtype | None = None) -> None:
        if getattr(self.model, "hf_device_map", None) is not None:
            return
        self.model.to(device, dtype=dtype)

    def generate(self, image: Image.Image, prompt: str, **gen_kw: Any) -> str:
        proc = self.processor
        if proc is None:
            raise RuntimeError("HF backend has no processor/tokenizer.")

        inputs = _prepare_hf_vlm_inputs(
            proc,
            self.model,
            image,
            prompt,
            model_type=self.model_type,
            model_id=self.model_id,
        )
        temperature = gen_kw.get("temperature", 0.0)
        do_sample = gen_kw.get("do_sample", temperature > 0)

        gen_ids = self.model.generate(
            **inputs,
            max_new_tokens=gen_kw.get("max_new_tokens", 512),
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
        )
        return _decode_hf_generation(proc, inputs, gen_ids)


def _ensure_prismatic_on_path() -> None:
    if str(_PRISMATIC_PKG) not in sys.path:
        sys.path.insert(0, str(_PRISMATIC_PKG.parent.parent))


def _load_prismatic(
    model_id: str,
    hf_token: str | None,
    vlm_checkpoint: Path | None,
    vlm_config: Path | None,
    device: str,
) -> tuple[VLMBackend, dict[str, Any]]:
    _ensure_prismatic_on_path()
    from prismatic import load as prismatic_load

    if vlm_checkpoint is not None and vlm_config is not None:
        vlm = prismatic_load.load_vlm(
            str(vlm_checkpoint),
            hf_token=hf_token,
            device=device,
            config_path=str(vlm_config),
        )
        meta = {
            "source": "prismatic_checkpoint",
            "checkpoint": str(vlm_checkpoint),
            "config": str(vlm_config),
            "model_id": model_id,
            "bbox_coord_space": "normalized_01",
        }
    else:
        vlm = prismatic_load.load(model_id, hf_token=hf_token, device=device)
        meta = {
            "source": "prismatic_hub",
            "model_id": model_id,
            "bbox_coord_space": "normalized_01",
        }
    return PrismaticBackend(vlm, image_size=384), meta


def _load_hf(
    model_id: str,
    hf_token: str | None,
    device: str | torch.device,
) -> tuple[VLMBackend, dict[str, Any]]:
    model, processor, meta = load_hf_vlm(
        model_id,
        hf_token=hf_token,
        device=device,
    )
    image_side = int(meta.get("image_side") or 384)
    backend = HfAutoBackend(
        model,
        processor,
        image_size=image_side,
        model_id=model_id,
        model_type=str(meta.get("model_type") or ""),
    )
    return backend, meta


def build_vlm_user_prompt(
    backend: VLMBackend,
    user_prompt: str,
    *,
    wrap: Any = None,
) -> str:
    """Prismatic uses PurePromptBuilder; HF backends use the user prompt as-is."""
    msg = wrap(user_prompt) if wrap is not None else user_prompt.strip()
    pb = backend.get_prompt_builder()
    if pb is None:
        return msg
    pb.add_turn(role="human", message=msg)
    return pb.get_prompt()


def load_backend(
    backend: str,
    *,
    model_id: str | None = None,
    hf_token: str | None = None,
    api_key: str | None = None,
    api_key_file: Path | None = None,
    vlm_checkpoint: Path | None = None,
    vlm_config: Path | None = None,
    device: str | torch.device = "cuda",
    api_timeout_sec: int = 120,
) -> tuple[VLMBackend, dict[str, Any]]:
    """
    Load a VLM backend.

    - backend="prismatic": TRI-ML prismatic-vlms only (optional local .pt + config.json)
    - backend in {hf, qwen3, internvl3.5, …}: transformers AutoProcessor
    - backend in {openai, gpt, gemini, claude, …}: cloud vision API (OpenAI / Gemini / Anthropic)
    """
    name = normalize_backend_key(backend)
    mid = resolve_model_id(name, model_id)

    if is_prismatic_backend(name):
        return _load_prismatic(mid, hf_token, vlm_checkpoint, vlm_config, device)
    if is_api_backend(name):
        api_backend, meta = load_api_backend(
            name,
            mid,
            api_key=api_key,
            api_key_file=api_key_file,
            timeout_sec=api_timeout_sec,
        )
        return api_backend, meta  # type: ignore[return-value]
    if is_hf_backend(name):
        return _load_hf(mid, hf_token, device)
    raise ValueError(
        f"Unknown backend {backend!r}. Use one of: prismatic, hf, qwen3, internvl3.5, "
        "paligemma2, cosmos, openai, gemini, claude, …"
    )
