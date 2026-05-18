"""
backends.py

Unified VLM backend adapters for grounding tasks.

All supported backends (Prismatic, Cosmos, GR00T-H) expose the same interface:
  backend.generate(image, prompt, **gen_kw)  -> str   (visual inference)
  backend.generate_text(prompt, **gen_kw)    -> str   (language-only)
  backend.get_prompt_builder()               -> PromptBuilder
  backend.image_size                         -> int
  backend.meta                               -> dict

Usage:
  from backends import load_backend
  backend, meta = load_backend("prismatic", model_id="prism-dinosiglip+7b",
                                hf_token="...", device=torch.device("cuda:0"))
  text = backend.generate(image, prompt, max_new_tokens=256)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import torch
from PIL import Image

_PKG_ROOT = Path(__file__).resolve().parent
_REPO_ROOT = _PKG_ROOT.parent
_DEFAULT_HF = _REPO_ROOT / ".cache" / "huggingface"
HF_CACHE_ROOT = os.environ.setdefault("HF_HOME", str(_DEFAULT_HF))
os.environ.setdefault("HF_HUB_CACHE", f"{HF_CACHE_ROOT}/hub")
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", os.environ["HF_HUB_CACHE"])
os.environ.setdefault("TRANSFORMERS_CACHE", f"{HF_CACHE_ROOT}/transformers")


# ── Prompt Builder ────────────────────────────────────────────────────────────

class SimplePromptBuilder:
    """Fallback prompt builder (pass-through for models that handle wrapping internally)."""

    def __init__(self) -> None:
        self._parts: list[str] = []

    def add_turn(self, role: str, message: str) -> None:
        if role == "human":
            self._parts.append(message)

    def get_prompt(self) -> str:
        return "\n\n".join(self._parts).strip()


# ── Base ──────────────────────────────────────────────────────────────────────

class VLMBackend:
    meta: dict

    def generate(self, image: Image.Image, prompt: str, **gen_kw: Any) -> str:
        raise NotImplementedError

    def generate_text(self, prompt: str, **gen_kw: Any) -> str:
        raise NotImplementedError

    def get_prompt_builder(self) -> SimplePromptBuilder:
        return SimplePromptBuilder()

    @property
    def image_size(self) -> int:
        return 384

    def to(self, device: torch.device, dtype=None) -> "VLMBackend":
        return self


# ── Prismatic ─────────────────────────────────────────────────────────────────

class PrismaticBackend(VLMBackend):
    def __init__(self, vlm, meta: dict) -> None:
        self.vlm = vlm
        self.meta = meta

    def generate(self, image: Image.Image, prompt: str, **gen_kw: Any) -> str:
        return self.vlm.generate(image, prompt, **gen_kw)

    def generate_text(self, prompt: str, **gen_kw: Any) -> str:
        from contextlib import nullcontext
        from prismatic.models.vlms.prismatic import PrismaticVLM

        vlm = self.vlm
        tokenizer = vlm.llm_backbone.tokenizer
        input_ids = tokenizer(prompt, truncation=True, return_tensors="pt").input_ids.to(vlm.device)
        mm = torch.tensor([], dtype=torch.long, device=vlm.device)
        autocast_dtype = vlm.llm_backbone.half_precision_dtype
        ctx: Any = (
            torch.autocast("cuda", dtype=autocast_dtype, enabled=vlm.enable_mixed_precision_training)
            if torch.cuda.is_available()
            else nullcontext()
        )
        with torch.inference_mode(), ctx:
            generated_ids = super(PrismaticVLM, vlm).generate(
                input_ids=input_ids,
                pixel_values=None,
                multimodal_indices=mm,
                **gen_kw,
            )
        return tokenizer.decode(generated_ids[0, input_ids.shape[1]:], skip_special_tokens=True).strip()

    def get_prompt_builder(self):
        return self.vlm.get_prompt_builder()

    @property
    def image_size(self) -> int:
        if self.meta.get("source") != "local_checkpoint":
            return 384
        vid = self.meta.get("vision_backbone_id") or ""
        return 224 if "224px" in vid else 384

    def to(self, device: torch.device, dtype=None) -> "PrismaticBackend":
        if dtype is not None:
            self.vlm.to(device, dtype=dtype)
        else:
            self.vlm.to(device)
        return self


# ── Cosmos ────────────────────────────────────────────────────────────────────

class CosmosBackend(VLMBackend):
    def __init__(self, model, processor, meta: dict) -> None:
        self.model = model
        self.processor = processor
        self.meta = meta

    def generate(self, image: Image.Image, prompt: str, **gen_kw: Any) -> str:
        conversation = [
            {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
            {
                "role": "user",
                "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}],
            },
        ]
        return self._run_conversation(conversation, **gen_kw)

    def generate_text(self, prompt: str, **gen_kw: Any) -> str:
        conversation = [
            {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
            {"role": "user", "content": [{"type": "text", "text": prompt}]},
        ]
        return self._run_conversation(conversation, **gen_kw)

    def _run_conversation(self, conversation: list, **gen_kw: Any) -> str:
        inputs = self.processor.apply_chat_template(
            conversation,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)
        max_new_tokens = int(gen_kw.get("max_new_tokens", 256))
        do_sample = bool(gen_kw.get("do_sample", False))
        temperature = float(gen_kw.get("temperature", 0.1))
        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
        )
        trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids, strict=False)
        ]
        return self.processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()

    @property
    def image_size(self) -> int:
        return 384

    def to(self, device: torch.device, dtype=None) -> "CosmosBackend":
        # Cosmos is loaded with device_map="auto"; keep API compatibility.
        return self


# ── GR00T-H VLM-only (Eagle/Cosmos backbone) ─────────────────────────────────

class Gr00TBackend(VLMBackend):
    """
    GR00T-H VLM-only backend.

    This backend intentionally uses only the vision-language generation path and
    does not load or use GR00T action head modules.
    """

    def __init__(self, model, processor, meta: dict) -> None:
        self.model = model
        self.processor = processor
        self.meta = meta

    def generate(self, image: Image.Image, prompt: str, **gen_kw: Any) -> str:
        conversation = [
            {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
            {
                "role": "user",
                "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}],
            },
        ]
        return self._run_conversation(conversation, **gen_kw)

    def generate_text(self, prompt: str, **gen_kw: Any) -> str:
        conversation = [
            {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
            {"role": "user", "content": [{"type": "text", "text": prompt}]},
        ]
        return self._run_conversation(conversation, **gen_kw)

    def _run_conversation(self, conversation: list, **gen_kw: Any) -> str:
        inputs = self.processor.apply_chat_template(
            conversation,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)
        max_new_tokens = int(gen_kw.get("max_new_tokens", 256))
        do_sample = bool(gen_kw.get("do_sample", False))
        temperature = float(gen_kw.get("temperature", 0.1))
        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
        )
        trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids, strict=False)
        ]
        return self.processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()

    @property
    def image_size(self) -> int:
        return 384

    def to(self, device: torch.device, dtype=None) -> "Gr00TBackend":
        # Model may be loaded with device_map="auto"; no-op for compatibility.
        return self


# ── Factory ───────────────────────────────────────────────────────────────────

def load_backend(
    backend: str,
    *,
    model_id: str,
    hf_token: str,
    vlm_checkpoint: Path | None = None,
    vlm_config: Path | None = None,
    device: torch.device | None = None,
) -> tuple[VLMBackend, dict]:
    """
    Load a VLM backend by name.

    Args:
        backend: One of "prismatic", "cosmos", "groot".
        model_id: HF Hub model id or local model identifier.
        hf_token: Hugging Face access token string.
        vlm_checkpoint: Path to local checkpoint file (prismatic .pt).
        vlm_config: Path to config.json for prismatic fine-tune checkpoints.
        device: Target device (Cosmos ignores this; loaded with device_map="auto").

    Returns:
        (backend_instance, meta_dict)
    """
    if device is None:
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    if backend == "prismatic":
        return _load_prismatic(
            model_id=model_id,
            hf_token=hf_token,
            vlm_checkpoint=vlm_checkpoint,
            vlm_config=vlm_config,
        )
    elif backend == "cosmos":
        return _load_cosmos(model_id=model_id, hf_token=hf_token)
    elif backend == "groot":
        return _load_groot(model_id=model_id, hf_token=hf_token)
    else:
        raise ValueError(f"Unknown backend {backend!r}. Choose from: prismatic, cosmos, groot.")


def _load_prismatic(
    *,
    model_id: str,
    hf_token: str,
    vlm_checkpoint: Path | None,
    vlm_config: Path | None,
) -> tuple[PrismaticBackend, dict]:
    from prismatic import load
    from prismatic.models.materialize import get_llm_backbone_and_tokenizer, get_vision_backbone_and_transform
    from prismatic.models.vlms import PrismaticVLM

    if vlm_checkpoint is None:
        vlm = load(model_id, hf_token=hf_token)
        meta = {"source": "hf_hub", "hub_model_id": model_id}
        return PrismaticBackend(vlm, meta), meta

    ckpt = Path(vlm_checkpoint).resolve()
    if not ckpt.is_file():
        raise FileNotFoundError(f"--vlm-checkpoint not a file: {ckpt}")

    if vlm_config is not None:
        cfg_path = Path(vlm_config).resolve()
    else:
        d = ckpt.parent
        cfg_path = (d.parent if d.name == "checkpoints" else d) / "config.json"
    if not cfg_path.is_file():
        raise FileNotFoundError(f"config.json not found (tried {cfg_path}). Set --vlm-config explicitly.")

    import json
    with cfg_path.open("r", encoding="utf-8") as f:
        model_cfg = json.load(f)["model"]

    vision_backbone, _ = get_vision_backbone_and_transform(
        model_cfg["vision_backbone_id"],
        model_cfg["image_resize_strategy"],
    )
    llm_backbone, _ = get_llm_backbone_and_tokenizer(
        model_cfg["llm_backbone_id"],
        llm_max_length=model_cfg.get("llm_max_length", 2048),
        hf_token=hf_token,
        inference_mode=True,
    )
    vlm = PrismaticVLM.from_pretrained(
        ckpt,
        model_cfg["model_id"],
        vision_backbone,
        llm_backbone,
        arch_specifier=model_cfg["arch_specifier"],
    )
    meta = {
        "source": "local_checkpoint",
        "model_id": model_cfg["model_id"],
        "vlm_checkpoint": str(ckpt),
        "vlm_config": str(cfg_path),
        "vision_backbone_id": model_cfg["vision_backbone_id"],
        "llm_backbone_id": model_cfg["llm_backbone_id"],
        "arch_specifier": model_cfg["arch_specifier"],
        "image_resize_strategy": model_cfg.get("image_resize_strategy"),
    }
    return PrismaticBackend(vlm, meta), meta


def _load_cosmos(
    *,
    model_id: str,
    hf_token: str,
) -> tuple[CosmosBackend, dict]:
    import transformers

    model = transformers.Qwen3VLForConditionalGeneration.from_pretrained(
        model_id,
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        token=hf_token,
        attn_implementation="sdpa",
    )
    processor = transformers.Qwen3VLProcessor.from_pretrained(model_id, token=hf_token)
    meta = {"source": "hf_hub", "hub_model_id": model_id, "model_id": model_id}
    return CosmosBackend(model, processor, meta), meta


def _load_groot(
    *,
    model_id: str,
    hf_token: str,
) -> tuple[Gr00TBackend, dict]:
    """
    Load GR00T-H VLM-only backend.

    Expected model id examples:
      - nvidia/GR00T-H
      - local fine-tuned checkpoint path
    """
    # Gr00tN1d6 is registered on import (AutoConfig/AutoModel/AutoProcessor), not in stock Transformers.
    try:
        import gr00t.model  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "GR00T backend requires the local `gr00t` package (backend/GR00T-H). "
            "Install it (e.g. pip install -e path/to/GR00T-H) or add GR00T-H to PYTHONPATH "
            "(grounding_task.sh does this when BACKEND=groot)."
        ) from e

    import transformers

    resolved_vlm_model_id = model_id
    source_vla_model_id: str | None = None

    # If model_id points to a GR00T VLA checkpoint/config, load only its VLM backbone.
    # Gr00tN1d6Config stores backbone id in `model_name` (e.g. nvidia/Eagle-Block2A-2B-v2).
    try:
        cfg = transformers.AutoConfig.from_pretrained(
            model_id,
            trust_remote_code=True,
            token=hf_token,
        )
        cfg_model_type = str(getattr(cfg, "model_type", "")).lower()
        cfg_arches = [str(x).lower() for x in (getattr(cfg, "architectures", None) or [])]
        is_gr00t_vla = (
            "gr00t" in cfg_model_type
            or any("gr00t" in a for a in cfg_arches)
            or hasattr(cfg, "model_name")
        )
        backbone_from_cfg = str(getattr(cfg, "model_name", "")).strip()
        if is_gr00t_vla and backbone_from_cfg:
            source_vla_model_id = model_id
            resolved_vlm_model_id = backbone_from_cfg
    except Exception:
        # Keep backward-compatible behavior when config probing fails.
        pass

    model = transformers.AutoModel.from_pretrained(
        resolved_vlm_model_id,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True,
        token=hf_token,
    )
    processor = transformers.AutoProcessor.from_pretrained(
        resolved_vlm_model_id,
        trust_remote_code=True,
        token=hf_token,
    )
    meta = {
        "source": "hf_hub",
        "hub_model_id": resolved_vlm_model_id,
        "model_id": resolved_vlm_model_id,
        "mode": "vlm_only",
        "note": "GR00T action head is not loaded/used.",
    }
    if source_vla_model_id is not None:
        meta["source_vla_model_id"] = source_vla_model_id
        meta["resolved_from_vla_backbone"] = True
        meta["note"] = "Loaded VLM backbone only from GR00T VLA config; action head is not loaded/used."
    return Gr00TBackend(model, processor, meta), meta
