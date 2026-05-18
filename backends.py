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

from backend_registry import (
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


class HfAutoBackend(VLMBackend):
    """Hugging Face VLM via AutoProcessor + chat template (Qwen-VL, InternVL, PaliGemma, …)."""

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

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        if hasattr(proc, "apply_chat_template"):
            inputs = proc.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
            inputs = {k: v.to(self.model.device) if hasattr(v, "to") else v for k, v in inputs.items()}
            if "pixel_values" not in inputs and hasattr(proc, "__call__"):
                try:
                    extra = proc(text=prompt, images=image, return_tensors="pt")
                    for k, v in extra.items():
                        if k not in inputs and hasattr(v, "to"):
                            inputs[k] = v.to(self.model.device)
                except Exception:
                    pass
        else:
            inputs = proc(text=prompt, images=image, return_tensors="pt")
            inputs = {k: v.to(self.model.device) if hasattr(v, "to") else v for k, v in inputs.items()}

        gen_ids = self.model.generate(
            **inputs,
            max_new_tokens=gen_kw.get("max_new_tokens", 512),
            do_sample=gen_kw.get("temperature", 0.0) > 0,
            temperature=gen_kw.get("temperature", 0.2) if gen_kw.get("temperature", 0) > 0 else None,
        )
        trimmed = [out[len(inp) :] for inp, out in zip(inputs["input_ids"], gen_ids)]
        return proc.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()


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
    vlm_checkpoint: Path | None = None,
    vlm_config: Path | None = None,
    device: str | torch.device = "cuda",
) -> tuple[VLMBackend, dict[str, Any]]:
    """
    Load a VLM backend.

    - backend="prismatic": TRI-ML prismatic-vlms only (optional local .pt + config.json)
    - backend in {hf, qwen3, internvl3.5, paligemma2, cosmos, groot, …}: transformers AutoProcessor
    """
    name = normalize_backend_key(backend)
    mid = resolve_model_id(name, model_id)

    if is_prismatic_backend(name):
        return _load_prismatic(mid, hf_token, vlm_checkpoint, vlm_config, device)
    if is_hf_backend(name):
        return _load_hf(mid, hf_token, device)
    raise ValueError(
        f"Unknown backend {backend!r}. Use one of: prismatic, hf, qwen3, internvl3.5, "
        "paligemma2, cosmos, groot, …"
    )
