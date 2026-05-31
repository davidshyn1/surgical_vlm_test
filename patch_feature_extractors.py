"""
patch_feature_extractors.py

Extract patch-level vision features for cross-attention visual grounding.

Backends:
  - timm: shared DINOv2 + SigLIP (reference)
  - prismatic: Prismatic VLM vision_backbone (DINO + SigLIP)
  - hf: each HF VLM's own vision tower via AutoProcessor + model hooks
"""

from __future__ import annotations

import math
import sys
from contextlib import nullcontext
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import timm
from PIL import Image
from torchvision.transforms import Compose, Resize
from torchvision.transforms.functional import pil_to_tensor

_RGB_PROJ_CACHE: dict[tuple[int, str], torch.Tensor] = {}


def _rgb_to_feat_projection(feat_dim: int, device: torch.device) -> torch.Tensor:
    """Fixed 3→D projection matrix [3, feat_dim] for query RGB patch tokens."""
    key = (feat_dim, str(device))
    if key not in _RGB_PROJ_CACHE:
        g = torch.Generator(device="cpu").manual_seed(42 + feat_dim)
        proj = torch.randn(3, feat_dim, generator=g)
        # Row-normalize (do not use qr on [3, D]: reduced QR returns [3, 3] only).
        proj = proj / proj.norm(dim=1, keepdim=True).clamp(min=1e-8)
        _RGB_PROJ_CACHE[key] = proj.to(device)
    out = _RGB_PROJ_CACHE[key]
    if out.shape != (3, feat_dim):
        del _RGB_PROJ_CACHE[key]
        return _rgb_to_feat_projection(feat_dim, device)
    return out


def image_as_pixel_patch_tokens(
    image: Image.Image,
    grid_h: int,
    grid_w: int,
    feat_dim: int,
    device: torch.device,
    *,
    mode: str = "pixel_grid",
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Represent the query image at patch-token level without a vision backbone.

    - pixel_grid: resize query to (grid_w, grid_h); each cell is one query token.
    - single: whole query image → one query token (reference-instrument style).
    """
    proj = _rgb_to_feat_projection(feat_dim, device)
    if mode == "single":
        t = pil_to_tensor(image.convert("RGB")).float().div_(255.0).to(device)
        rgb = F.adaptive_avg_pool2d(t.unsqueeze(0), 1).squeeze(0).reshape(1, 3)
        patches = F.normalize(rgb @ proj, dim=-1)
        cls = patches.squeeze(0)
        return patches, cls

    img = image.convert("RGB").resize((grid_w, grid_h), Image.Resampling.BILINEAR)
    t = pil_to_tensor(img).float().div_(255.0)
    rgb = t.permute(1, 2, 0).reshape(grid_h * grid_w, 3).to(device)
    patches = F.normalize(rgb @ proj, dim=-1)
    cls = patches.mean(dim=0)
    return patches, cls


# Back-compat alias
query_image_as_patch_tokens = image_as_pixel_patch_tokens

DINO_TIMM_ID = "vit_large_patch14_reg4_dinov2.lvd142m"
SIGLIP_TIMM_ID = "vit_so400m_patch14_siglip_224"
PATCH_SIZE = 14


def _model_device(model: Any) -> torch.device:
    dev = getattr(model, "device", None)
    if dev is not None:
        return dev
    return next(model.parameters()).device


def _move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in batch.items():
        out[k] = v.to(device) if hasattr(v, "to") else v
    return out


def _split_seq_tokens(t: torch.Tensor, has_cls: bool) -> tuple[torch.Tensor, torch.Tensor | None]:
    if not has_cls:
        return t, None
    if t.dim() != 3 or t.shape[1] < 1:
        return t, None
    if t.shape[1] == 1:
        return t[:, :0, :], t[:, 0, :]
    return t[:, 1:, :], t[:, 0, :]


def _extract_timm_patches_and_cls(
    model,
    pixel_values: torch.Tensor,
    layer_idx: int,
    has_cls: bool = True,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    result = model.get_intermediate_layers(
        pixel_values, n={layer_idx}, return_prefix_tokens=True
    )
    if not result:
        raise RuntimeError("get_intermediate_layers returned empty list")
    layer_out = result[0]
    if isinstance(layer_out, tuple):
        if (
            len(layer_out) >= 2
            and isinstance(layer_out[0], torch.Tensor)
            and isinstance(layer_out[1], torch.Tensor)
        ):
            patch_tokens, prefix_tokens = layer_out[0], layer_out[1]
            cls_token = (
                prefix_tokens[:, 0, :]
                if (has_cls and prefix_tokens.dim() == 3 and prefix_tokens.shape[1] > 0)
                else None
            )
            return patch_tokens, cls_token
        if len(layer_out) == 1 and isinstance(layer_out[0], torch.Tensor):
            return _split_seq_tokens(layer_out[0], has_cls)
    if isinstance(layer_out, torch.Tensor):
        return _split_seq_tokens(layer_out, has_cls)
    raise TypeError(f"Unexpected layer_out type={type(layer_out).__name__}")


def _infer_grid_from_token_count(n_tokens: int, grid_thw: torch.Tensor | None) -> tuple[int, int]:
    if grid_thw is not None and grid_thw.numel() >= 3:
        dims = grid_thw.reshape(-1, 3)[0].tolist()
        t, h, w = int(dims[0]), int(dims[1]), int(dims[2])
        if t >= 1 and h >= 1 and w >= 1 and t * h * w == n_tokens:
            return h, w
    side = int(round(math.sqrt(n_tokens)))
    if side * side == n_tokens:
        return side, side
    raise ValueError(
        f"Cannot infer 2D grid for {n_tokens} tokens (grid_thw={grid_thw})."
    )


def _flatten_patch_tensor(t: torch.Tensor) -> torch.Tensor:
    """Return [N, D] float patch tokens."""
    if t.dim() == 3:
        t = t[0]
    if t.dim() != 2:
        raise ValueError(f"Expected patch tensor [N,D], got shape {tuple(t.shape)}")
    return t.float()


def _vision_output_to_tensor(out: Any) -> torch.Tensor:
    """Unwrap HF vision / get_image_features outputs to a patch token tensor."""
    if isinstance(out, torch.Tensor):
        return out
    if isinstance(out, (list, tuple)):
        for item in out:
            try:
                return _vision_output_to_tensor(item)
            except (TypeError, ValueError):
                continue
        raise TypeError(f"Could not extract tensor from sequence type={type(out).__name__}")
    for attr in (
        "last_hidden_state",
        "hidden_states",
        "image_embeds",
        "pooler_output",
    ):
        if hasattr(out, attr):
            val = getattr(out, attr)
            if val is None:
                continue
            if attr == "hidden_states" and isinstance(val, (list, tuple)) and val:
                return _vision_output_to_tensor(val[-1])
            return _vision_output_to_tensor(val)
    raise TypeError(
        f"Could not extract patch tensor from vision output type={type(out).__name__}"
    )


@dataclass
class PatchFeatureBackbone:
    """Unified patch feature API for cross-attention."""

    grid_h: int
    grid_w: int
    device: torch.device
    source_names: list[str]
    source_labels: dict[str, str]
    meta: dict[str, Any] = field(default_factory=dict)
    autocast_dtype: torch.dtype | None = None

    def extract(self, image: Image.Image) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        raise NotImplementedError


@dataclass
class _TimmDualBackbone(PatchFeatureBackbone):
    dino_model: Any = None
    dino_transform: Any = None
    siglip_model: Any = None
    siglip_transform: Any = None
    dino_layer_idx: int = 0
    siglip_layer_idx: int = 0

    def extract(self, image: Image.Image) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        dino_in = self.dino_transform(image).unsqueeze(0).to(self.device)
        siglip_in = self.siglip_transform(image).unsqueeze(0).to(self.device)
        dino_in = dino_in.to(dtype=next(self.dino_model.parameters()).dtype)
        siglip_in = siglip_in.to(dtype=next(self.siglip_model.parameters()).dtype)
        ctx = (
            torch.autocast("cuda", dtype=self.autocast_dtype)
            if (self.autocast_dtype is not None and torch.cuda.is_available())
            else nullcontext()
        )
        with torch.inference_mode(), ctx:
            dino_patches, dino_cls = _extract_timm_patches_and_cls(
                self.dino_model, dino_in, self.dino_layer_idx, has_cls=True
            )
            siglip_patches, _ = _extract_timm_patches_and_cls(
                self.siglip_model, siglip_in, self.siglip_layer_idx, has_cls=False
            )
        dino_patches = dino_patches[0]
        dino_cls = dino_cls[0] if dino_cls is not None else dino_patches.mean(dim=0)
        siglip_patches = siglip_patches[0]
        concat_patches = torch.cat([dino_patches, siglip_patches], dim=-1)
        concat_cls = torch.cat([dino_cls, siglip_patches.mean(dim=0)], dim=-1)
        return {
            "dino": (dino_patches, dino_cls),
            "concat": (concat_patches, concat_cls),
        }


@dataclass
class _HfVisionBackbone(PatchFeatureBackbone):
    model: Any = None
    processor: Any = None
    model_type: str = ""

    def _processor_image_batch(self, image: Image.Image) -> dict[str, Any]:
        proc = self.processor
        mt = (self.model_type or "").lower()
        mid = str(getattr(self.model, "name_or_path", "") or "").lower()

        if "paligemma" in mt or "paligemma" in mid:
            text = "<image>\n"
            batch = proc(text=text, images=image, return_tensors="pt")
            return _move_batch(batch, self.device)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": "Describe."},
                ],
            }
        ]
        if hasattr(proc, "apply_chat_template"):
            batch = proc.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=False,
                return_dict=True,
                return_tensors="pt",
            )
            return _move_batch(batch, self.device)
        batch = proc(text="Describe.", images=image, return_tensors="pt")
        return _move_batch(batch, self.device)

    def _run_vision(self, batch: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        model = self.model
        mt = (self.model_type or "").lower()
        grid_thw = batch.get("image_grid_thw")

        ctx = (
            torch.autocast("cuda", dtype=self.autocast_dtype)
            if (self.autocast_dtype is not None and torch.cuda.is_available())
            else nullcontext()
        )

        with torch.inference_mode(), ctx:
            # Qwen2/3-VL family
            get_img_feat = getattr(model, "get_image_features", None)
            if get_img_feat is None and hasattr(model, "model"):
                get_img_feat = getattr(model.model, "get_image_features", None)
            if "qwen" in mt and callable(get_img_feat):
                pixel_values = batch.get("pixel_values")
                image_grid_thw = batch.get("image_grid_thw")
                if pixel_values is None:
                    raise ValueError("Qwen-VL batch missing pixel_values")
                out = get_img_feat(pixel_values, image_grid_thw=image_grid_thw)
                patches = _flatten_patch_tensor(_vision_output_to_tensor(out))
                cls = patches.mean(dim=0)
                return patches, cls, image_grid_thw

            visual = getattr(model, "visual", None)
            if visual is None and hasattr(model, "model"):
                visual = getattr(model.model, "visual", None)

            if visual is not None and batch.get("pixel_values") is not None:
                pixel_values = batch["pixel_values"]
                kwargs: dict[str, Any] = {}
                if batch.get("image_grid_thw") is not None:
                    kwargs["grid_thw"] = batch["image_grid_thw"]
                out = visual(pixel_values, **kwargs)
                patches = _flatten_patch_tensor(_vision_output_to_tensor(out))
                cls = patches.mean(dim=0)
                return patches, cls, batch.get("image_grid_thw")

            # PaliGemma
            vision_tower = getattr(model, "vision_tower", None)
            if vision_tower is None and hasattr(model, "model"):
                vision_tower = getattr(model.model, "vision_tower", None)
            if vision_tower is not None and batch.get("pixel_values") is not None:
                out = vision_tower(batch["pixel_values"])
                if isinstance(out, tuple):
                    out = out[0]
                if out.dim() == 3 and out.shape[1] > 1:
                    patches = out[0]
                    cls = patches[0] if patches.shape[0] > 0 else patches.mean(dim=0)
                    patches = patches[1:] if patches.shape[0] > 1 else patches
                else:
                    patches = _flatten_patch_tensor(out)
                    cls = patches.mean(dim=0)
                return patches, cls, None

            vision_model = getattr(model, "vision_model", None)
            if vision_model is not None and batch.get("pixel_values") is not None:
                out = vision_model(batch["pixel_values"])
                if isinstance(out, tuple):
                    out = out[0]
                if out.dim() == 3:
                    out = out[0]
                patches = _flatten_patch_tensor(out)
                cls = patches.mean(dim=0)
                return patches, cls, batch.get("image_grid_thw")

            # InternVL / generic
            if hasattr(model, "extract_feature") and batch.get("pixel_values") is not None:
                out = model.extract_feature(batch["pixel_values"])
                patches = _flatten_patch_tensor(out)
                cls = patches.mean(dim=0)
                return patches, cls, None

            if hasattr(model, "get_image_features"):
                kwargs = {k: batch[k] for k in ("pixel_values", "image_grid_thw") if k in batch}
                out = model.get_image_features(**kwargs)
                patches = _flatten_patch_tensor(_vision_output_to_tensor(out))
                cls = patches.mean(dim=0)
                return patches, cls, batch.get("image_grid_thw")

        raise RuntimeError(
            f"No vision patch extraction path for model_type={self.model_type!r} "
            f"({type(model).__name__})."
        )

    def extract(self, image: Image.Image) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        batch = self._processor_image_batch(image)
        patches, cls, grid_thw = self._run_vision(batch)
        if cls is None:
            cls = patches.mean(dim=0)
        gh, gw = _infer_grid_from_token_count(int(patches.shape[0]), grid_thw)
        if gh != self.grid_h or gw != self.grid_w:
            # Processor may yield different effective grid per image; trust this sample.
            self.grid_h, self.grid_w = gh, gw
        return {"vision": (patches, cls)}


def _make_timm_model(timm_id: str, image_size: int):
    m = timm.create_model(timm_id, pretrained=True, num_classes=0, img_size=image_size)
    m.eval()
    return m


def _make_timm_transform(model, image_size: int):
    cfg = timm.data.resolve_model_data_config(model)
    cfg["input_size"] = (3, image_size, image_size)
    t = timm.data.create_transform(**cfg, is_training=False)
    if isinstance(t, Compose) and isinstance(t.transforms[0], Resize):
        t = Compose([Resize(image_size, interpolation=t.transforms[0].interpolation), *t.transforms[1:]])
    return t


def load_timm_patch_backbone(image_size: int, device: torch.device) -> PatchFeatureBackbone:
    print("[INFO] Patch features: timm DINOv2 + SigLIP", file=sys.stderr)
    dino_model = _make_timm_model(DINO_TIMM_ID, image_size)
    siglip_model = _make_timm_model(SIGLIP_TIMM_ID, image_size)
    dino_transform = _make_timm_transform(dino_model, image_size)
    siglip_transform = _make_timm_transform(siglip_model, image_size)
    autocast_dtype = torch.float16
    try:
        dino_model = dino_model.to(device).half()
        siglip_model = siglip_model.to(device).half()
    except (torch.cuda.OutOfMemoryError, RuntimeError):
        print("[WARN] timm OOM — CPU fp32", file=sys.stderr)
        device = torch.device("cpu")
        dino_model = dino_model.to(device).float()
        siglip_model = siglip_model.to(device).float()
        autocast_dtype = None
    grid_side = image_size // PATCH_SIZE
    return _TimmDualBackbone(
        grid_h=grid_side,
        grid_w=grid_side,
        device=device,
        source_names=["dino", "concat"],
        source_labels={"dino": "DINO (1024)", "concat": "DINO+SigLIP (2176)"},
        meta={"source": "timm_direct", "dino_id": DINO_TIMM_ID, "siglip_id": SIGLIP_TIMM_ID},
        autocast_dtype=autocast_dtype,
        dino_model=dino_model,
        dino_transform=dino_transform,
        siglip_model=siglip_model,
        siglip_transform=siglip_transform,
        dino_layer_idx=len(dino_model.blocks) - 2,
        siglip_layer_idx=len(siglip_model.blocks) - 2,
    )


def load_prismatic_patch_backbone(
    *,
    hf_token: str,
    hub_model_id: str,
    vlm_checkpoint: Path | None,
    vlm_config: Path | None,
    device: torch.device,
) -> PatchFeatureBackbone:
    from backends import load_backend

    print("[INFO] Patch features: Prismatic vision_backbone", file=sys.stderr)
    backend, meta = load_backend(
        "prismatic",
        model_id=hub_model_id,
        hf_token=hf_token,
        vlm_checkpoint=vlm_checkpoint,
        vlm_config=vlm_config,
        device=device,
    )
    backend.to(device, dtype=torch.bfloat16)
    vb = backend.vlm.vision_backbone
    grid_side = int(vb.num_patches ** 0.5)
    timm_bp = _TimmDualBackbone(
        grid_h=grid_side,
        grid_w=grid_side,
        device=device,
        source_names=["dino", "concat"],
        source_labels={"dino": "DINO (1024)", "concat": "DINO+SigLIP (2176)"},
        meta=meta,
        autocast_dtype=torch.bfloat16,
        dino_model=vb.dino_featurizer,
        dino_transform=lambda img: vb.image_transform(img)["dino"],
        siglip_model=vb.siglip_featurizer,
        siglip_transform=lambda img: vb.image_transform(img)["siglip"],
        dino_layer_idx=len(vb.dino_featurizer.blocks) - 2,
        siglip_layer_idx=len(vb.siglip_featurizer.blocks) - 2,
    )
    return timm_bp


def load_hf_patch_backbone(
    *,
    backend: str,
    model_id: str,
    hf_token: str | None,
    device: torch.device,
) -> PatchFeatureBackbone:
    from hf_model_loader import load_hf_vlm

    print(f"[INFO] Patch features: HF vision ({backend} / {model_id})", file=sys.stderr)
    model, processor, meta = load_hf_vlm(model_id, hf_token=hf_token, device=device)
    model_type = str(meta.get("model_type") or "")
    dev = _model_device(model)
    autocast_dtype = torch.bfloat16 if dev.type == "cuda" and torch.cuda.is_available() else None

    probe = Image.new("RGB", (224, 224), color=(128, 64, 32))
    bp = _HfVisionBackbone(
        grid_h=16,
        grid_w=16,
        device=dev,
        source_names=["vision"],
        source_labels={"vision": f"HF vision ({model_type or model_id})"},
        meta={**meta, "patch_extraction": "hf_vision_tower"},
        autocast_dtype=autocast_dtype,
        model=model,
        processor=processor,
        model_type=model_type,
    )
    try:
        feats = bp.extract(probe)
        patches, _ = feats["vision"]
        gh, gw = _infer_grid_from_token_count(int(patches.shape[0]), None)
        bp.grid_h, bp.grid_w = gh, gw
        print(f"[INFO] HF vision grid probe: {gh}x{gw} (N={patches.shape[0]}, D={patches.shape[1]})", file=sys.stderr)
    except Exception as exc:
        print(f"[WARN] HF grid probe failed ({exc}); using 16x16 placeholder grid.", file=sys.stderr)
    return bp


def load_patch_feature_backbone(
    feature_backbone: str,
    *,
    backend: str,
    model_id: str,
    hf_token: str | None,
    device: torch.device,
    image_size: int = 224,
    vlm_checkpoint: Path | None = None,
    vlm_config: Path | None = None,
) -> PatchFeatureBackbone:
    fb = (feature_backbone or "timm").strip().lower()
    if fb == "timm":
        return load_timm_patch_backbone(image_size, device)
    if fb == "prismatic":
        return load_prismatic_patch_backbone(
            hf_token=hf_token or "",
            hub_model_id=model_id,
            vlm_checkpoint=vlm_checkpoint,
            vlm_config=vlm_config,
            device=device,
        )
    if fb == "hf":
        return load_hf_patch_backbone(
            backend=backend,
            model_id=model_id,
            hf_token=hf_token,
            device=device,
        )
    raise ValueError(f"Unknown feature_backbone={feature_backbone!r}; use timm, prismatic, or hf.")
