"""
visual_cross_attention_cholect50.py

CholecT50 challenge-val: query × test frame cross-attention on patch features.

Query encoding (default ``vision_ref``): query and test both use ``--feature-backbone``
vision patches (same embedding space → localization works). ``pixel_grid`` uses RGB patch
tokens on **both** sides (appearance baseline, no VLM semantics).

Feature backbones:
  - timm: shared DINOv2 + SigLIP (dino + concat sources)
  - prismatic: Prismatic vision_backbone (dino + concat)
  - hf: each VLM's own vision tower (--backend / --model-id)

Methods (per source):
  - cls_softmax: query CLS → test patch softmax
  - mean_cosine: mean query patches → test patches cosine
  - patch_cross_mean / patch_cross_max: full query→test cross-attention

Writes comparison figures (query + test + patch_cross_max only) and per-method
heatmap PNGs under ``outputs/visual_cross_attention_cholect50/``.

Usage:
  python visual_cross_attention_cholect50.py --video VID68 --frame 837
  python visual_cross_attention_cholect50.py --video VID68 --frame 837 --query-from-gt-crop
  python visual_cross_attention_cholect50.py --samples-per-instrument 3 --seed 42
  BACKEND=qwen3-4b bash grounding_task.sh visual_cross_attention_cholect50 \\
    --feature-backbone hf --video VID68 --frame 837 --query-from-gt-crop
  BACKEND=prismatic bash grounding_task.sh visual_cross_attention_cholect50 \\
    --feature-backbone prismatic --vlm-checkpoint /path/to/step-....pt
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from difflib import get_close_matches
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from backend_registry import (
    BACKEND_CHOICES,
    resolve_hf_token,
    resolve_model_id,
    resolve_output_model_name,
)
from patch_feature_extractors import (
    PatchFeatureBackbone,
    image_as_pixel_patch_tokens,
    load_patch_feature_backbone,
)
from cholect50_data import (
    CHALLENGE_VAL_ROOT,
    collect_instrument_annotations,
    discover_video_roots,
    infer_pil_side,
    resolve_frame_path,
    sample_by_instrument,
)
from cholect_query_match import canonical_label_key, query_matches_frame_labels
from utils import CHOLECT_ROOT, iou_xyxy, load_label_json, normalize_instrument_name, parse_annotation_row, resolve_device

_SCRIPT_ROOT = Path(__file__).resolve().parent
_DEFAULT_OUTPUT_ROOT = _SCRIPT_ROOT / "outputs" / "visual_cross_attention_cholect50"
_DEFAULT_QUERY_DIR = _SCRIPT_ROOT / "assets" / "cholect50_query"
_DEFAULT_HF_TOKEN = _SCRIPT_ROOT / ".hf_token"

DEFAULT_IMAGE_SIZE = 224
METHODS = ["cls_softmax", "mean_cosine", "patch_cross_mean", "patch_cross_max"]
COMPARISON_METHODS = ["patch_cross_max"]
SUMMARY_ATTENTION_METHOD = "patch_cross_max"
ATTENTION_BBOX_PERCENTILE = 75.0
# Scale normalized cosine logits before softmax (higher → peakier; lower → broader blobs).
CLS_SOFTMAX_LOGIT_SCALE = 5.0
PATCH_CROSS_MEAN_LOGIT_SCALE = 12.0
# patch_cross_max (tighter peaks)
# PATCH_CROSS_MAX_LOGIT_SCALE = 10.0
# PATCH_CROSS_MAX_POST_SCALE = 2.5
# Softer query/test softmax → high-attention mass spreads over more patches.
PATCH_CROSS_MAX_LOGIT_SCALE = 7.5
PATCH_CROSS_MAX_POST_SCALE = 1.9
# Viz (patch_cross_max): lower floor + gamma≈1 + slightly more blur → wider visible peak region.
# HEATMAP_TOP_PERCENTILE = 50.0
# HEATMAP_GAMMA = 1.15
# PATCH_CROSS_MAX_GAUSS_SIGMA = 1.0
HEATMAP_TOP_PERCENTILE = 40.0
HEATMAP_GAMMA = 1.0
CLS_HEATMAP_GAUSS_SIGMA = 0.7
PATCH_CROSS_MAX_GAUSS_SIGMA = 1.45
_QUERY_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")


# ── Labels ────────────────────────────────────────────────────────────────────

def get_frame_labels(vid_name: str, frame_idx: int, labels_dir: Path) -> set[str]:
    try:
        label_data = load_label_json(labels_dir, vid_name)
    except FileNotFoundError:
        return set()
    categories = label_data.get("categories") or {}
    rows = (label_data.get("annotations") or {}).get(str(frame_idx), [])
    labels: set[str] = set()
    for row in rows:
        parsed = parse_annotation_row(list(row), categories)
        if parsed is None or parsed["visibility"] < 0.5:
            continue
        labels.add(parsed["instrument_name"])
        tgt = parsed.get("target_name") or ""
        if tgt and tgt not in ("null-target", "null_target"):
            labels.add(tgt)
    return labels


def get_full_frame_annotations(vid_name: str, frame_idx: int, labels_dir: Path) -> list[dict[str, Any]]:
    try:
        label_data = load_label_json(labels_dir, vid_name)
    except FileNotFoundError:
        return []
    categories = label_data.get("categories") or {}
    rows = (label_data.get("annotations") or {}).get(str(frame_idx), [])
    out: list[dict[str, Any]] = []
    for row in rows:
        parsed = parse_annotation_row(list(row), categories)
        if parsed is None or parsed["visibility"] < 0.5:
            continue
        bbox = parsed.get("bbox_xyxy")
        if bbox is None:
            continue
        out.append(
            {
                "instrument": parsed["instrument_name"],
                "triplet": parsed.get("triplet_str", ""),
                "verb": parsed.get("verb_name", ""),
                "target": parsed.get("target_name", ""),
                "phase": parsed.get("phase_name", ""),
                "visibility": parsed["visibility"],
                "bbox_xyxy": bbox,
            }
        )
    return out


# ── Query matching ────────────────────────────────────────────────────────────

def build_query_index(query_dir: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    if not query_dir.is_dir():
        return index
    for p in sorted(query_dir.rglob("*")):
        if p.is_file() and p.suffix.lower() in _QUERY_EXTS:
            index[canonical_label_key(p.stem)] = p
    return index


def find_query_for_instrument(
    instrument_name: str,
    query_index: dict[str, Path],
    fuzzy_cutoff: float = 0.4,
) -> tuple[Path | None, str]:
    key = canonical_label_key(instrument_name)
    if not key:
        return None, "empty_key"
    if key in query_index:
        return query_index[key], "exact"
    for qkey, path in query_index.items():
        if key in qkey or qkey in key:
            return path, f"contains({qkey})"
    matches = get_close_matches(key, list(query_index.keys()), n=1, cutoff=fuzzy_cutoff)
    if matches:
        return query_index[matches[0]], f"fuzzy({matches[0]})"
    return None, "no_match"


def crop_gt_query_image(test_image: Image.Image, bbox_xyxy: tuple[float, float, float, float]) -> Image.Image:
    w, h = test_image.size
    xmin, ymin, xmax, ymax = bbox_xyxy
    left = int(max(0, min(w - 1, xmin * w)))
    top = int(max(0, min(h - 1, ymin * h)))
    right = int(max(left + 1, min(w, xmax * w)))
    bottom = int(max(top + 1, min(h, ymax * h)))
    return test_image.crop((left, top, right, bottom))


def resolve_query_image(
    *,
    instrument: str,
    test_image: Image.Image,
    bbox_xyxy: tuple[float, float, float, float],
    query_index: dict[str, Path],
    query_from_gt_crop: bool,
    fuzzy_cutoff: float,
) -> tuple[Image.Image | None, str, str]:
    qpath, qmatch = find_query_for_instrument(instrument, query_index, fuzzy_cutoff=fuzzy_cutoff)
    if qpath is not None:
        return Image.open(qpath).convert("RGB"), qpath.stem, qmatch
    if query_from_gt_crop:
        return crop_gt_query_image(test_image, bbox_xyxy), f"{instrument}_gt_crop", "gt_bbox_crop"
    return None, "", "no_match"


# ── Patch mask / stats ────────────────────────────────────────────────────────

def bbox_to_patch_mask(
    bbox_xyxy: tuple[float, float, float, float],
    grid_h: int,
    grid_w: int,
) -> np.ndarray:
    xmin, ymin, xmax, ymax = bbox_xyxy
    mask = np.zeros(grid_h * grid_w, dtype=bool)
    for r in range(grid_h):
        for c in range(grid_w):
            p_x0, p_x1 = c / grid_w, (c + 1) / grid_w
            p_y0, p_y1 = r / grid_h, (r + 1) / grid_h
            if p_x1 > xmin and p_x0 < xmax and p_y1 > ymin and p_y0 < ymax:
                mask[r * grid_w + c] = True
    return mask


def attention_bbox_stats(
    attn_map: np.ndarray,
    bbox_xyxy: tuple[float, float, float, float],
    grid_h: int,
    grid_w: int,
) -> dict[str, float]:
    inst_mask = bbox_to_patch_mask(bbox_xyxy, grid_h, grid_w)
    flat = attn_map.flatten()
    inst_attn = float(flat[inst_mask].mean()) if inst_mask.any() else 0.0
    bg_attn = float(flat[~inst_mask].mean()) if (~inst_mask).any() else 0.0
    ratio = inst_attn / (bg_attn + 1e-8)
    n_top = max(1, int(0.1 * len(flat)))
    top_idx = np.argsort(flat)[-n_top:]
    prec_top = float(inst_mask[top_idx].sum() / n_top)
    return {
        "instrument_mean_attn": round(inst_attn, 6),
        "background_mean_attn": round(bg_attn, 6),
        "instrument_attn_ratio": round(ratio, 6),
        "precision_at_top10pct": round(prec_top, 6),
        "n_instrument_patches": int(inst_mask.sum()),
        "n_background_patches": int((~inst_mask).sum()),
    }


def instrument_key_for_record(instrument: str, ann: dict[str, Any]) -> str:
    """Stable per-instrument slug for aggregation (EndoVis id or CholecT50 norm)."""
    iid = str(ann.get("instrument_id") or "").strip().lower()
    if iid:
        return iid
    key = canonical_label_key(instrument)
    if key:
        return key.replace("-", "_")
    return normalize_instrument_name(instrument).replace("-", "_")


def bbox_from_attention_map(
    attn_map: np.ndarray,
    grid_h: int,
    grid_w: int,
    *,
    percentile: float = ATTENTION_BBOX_PERCENTILE,
) -> tuple[float, float, float, float] | None:
    """Tight normalized xyxy bbox over high-attention patches (fallback: argmax patch)."""
    grid = attn_map.reshape(grid_h, grid_w).astype(np.float32)
    thr = float(np.percentile(grid, percentile))
    mask = grid >= thr
    if not mask.any():
        r, c = np.unravel_index(int(grid.argmax()), grid.shape)
        mask = np.zeros_like(grid, dtype=bool)
        mask[r, c] = True
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    if len(rows) == 0 or len(cols) == 0:
        return None
    r0, r1 = int(rows[0]), int(rows[-1])
    c0, c1 = int(cols[0]), int(cols[-1])
    return (c0 / grid_w, r0 / grid_h, (c1 + 1) / grid_w, (r1 + 1) / grid_h)


def attention_bbox_miou(
    attn_map: np.ndarray,
    bbox_xyxy: tuple[float, float, float, float],
    grid_h: int,
    grid_w: int,
) -> tuple[float, tuple[float, float, float, float] | None]:
    """IoU between GT bbox and bbox from thresholded attention map (patch grid)."""
    pred = bbox_from_attention_map(attn_map, grid_h, grid_w)
    if pred is None:
        return 0.0, None
    return float(iou_xyxy(pred, bbox_xyxy)), pred


def _metric_values_from_records(
    records: list[dict[str, Any]],
) -> tuple[list[float], list[float]]:
    ratios = [
        float(r["instrument_attn_ratio"])
        for r in records
        if r.get("instrument_attn_ratio") is not None
    ]
    mious = [float(r["mIoU"]) for r in records if r.get("mIoU") is not None]
    return ratios, mious


def _avg_block(
    ratios: list[float],
    mious: list[float],
    *,
    n_samples: int | None = None,
) -> dict[str, Any]:
    n = n_samples if n_samples is not None else max(len(ratios), len(mious))
    ratio_avg = (sum(ratios) / len(ratios)) if ratios else None
    miou_avg = (sum(mious) / len(mious)) if mious else None
    return {
        "n_samples": n,
        "instrument_attn_ratio_avg": (
            round(ratio_avg, 6) if ratio_avg is not None else None
        ),
        "mIoU_avg": round(miou_avg, 6) if miou_avg is not None else None,
    }


def overall_avg_from_summary(summary: dict[str, Any] | None) -> dict[str, Any] | None:
    """Top-level final averages (micro + macro) for ``cross_attention.json``."""
    if not summary:
        return None
    overall = summary.get("overall") or {}
    micro = overall.get("micro") or {}
    macro = overall.get("macro") or summary.get("macro") or {}
    final = overall.get("final") or micro
    return {
        "primary_source": summary.get("primary_source"),
        "method": summary.get("method"),
        "n_records": micro.get("n_samples"),
        "n_instruments": macro.get("n_instruments"),
        "instrument_attn_ratio_avg": final.get("instrument_attn_ratio_avg"),
        "mIoU_avg": final.get("mIoU_avg"),
        "micro": {
            "instrument_attn_ratio_avg": micro.get("instrument_attn_ratio_avg"),
            "mIoU_avg": micro.get("mIoU_avg"),
            "n_samples": micro.get("n_samples"),
        },
        "macro": {
            "instrument_attn_ratio_avg": macro.get("instrument_attn_ratio_avg"),
            "mIoU_avg": macro.get("mIoU_avg"),
            "n_instruments": macro.get("n_instruments"),
        },
    }


def build_per_instrument_summary(
    records: list[dict[str, Any]],
    *,
    primary_source: str,
    method: str = SUMMARY_ATTENTION_METHOD,
    samples_per_instrument: int | None = None,
    eval_all: bool = False,
) -> dict[str, Any] | None:
    """Per-instrument means + overall micro/macro/final averages."""
    if not records:
        return None

    all_ratios, all_mious = _metric_values_from_records(records)
    micro = _avg_block(all_ratios, all_mious, n_samples=len(records))

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rec in records:
        key = str(rec.get("instrument_key") or "").strip()
        if not key:
            key = instrument_key_for_record(
                str(rec.get("instrument") or ""),
                rec.get("annotation") or rec,
            )
        grouped[key].append(rec)

    instruments: dict[str, Any] = {}
    ratio_macro: list[float] = []
    miou_macro: list[float] = []

    for key in sorted(grouped):
        recs = grouped[key]
        ratios = [
            float(r["instrument_attn_ratio"])
            for r in recs
            if r.get("instrument_attn_ratio") is not None
        ]
        mious = [float(r["mIoU"]) for r in recs if r.get("mIoU") is not None]
        ratio_avg = (sum(ratios) / len(ratios)) if ratios else None
        miou_avg = (sum(mious) / len(mious)) if mious else None
        block: dict[str, Any] = {
            "n_samples": len(recs),
            "instrument_attn_ratio_avg": (
                round(ratio_avg, 6) if ratio_avg is not None else None
            ),
            "mIoU_avg": round(miou_avg, 6) if miou_avg is not None else None,
            "instrument_attn_ratio_values": [round(v, 6) for v in ratios],
            "mIoU_values": [round(v, 6) for v in mious],
        }
        if ratio_avg is not None:
            ratio_macro.append(ratio_avg)
        if miou_avg is not None:
            miou_macro.append(miou_avg)
        instruments[key] = block

    macro = {
        "n_instruments": len(instruments),
        "n_records_with_bbox": len(all_ratios),
        "instrument_attn_ratio_avg": (
            round(sum(ratio_macro) / len(ratio_macro), 6) if ratio_macro else None
        ),
        "mIoU_avg": round(sum(miou_macro) / len(miou_macro), 6) if miou_macro else None,
    }

    return {
        "primary_source": primary_source,
        "method": method,
        "attention_bbox_percentile": ATTENTION_BBOX_PERCENTILE,
        "samples_per_instrument": samples_per_instrument,
        "eval_all": bool(eval_all),
        "instruments": instruments,
        "macro": macro,
        "overall": {
            "micro": micro,
            "macro": macro,
            "final": micro,
        },
    }


def print_per_instrument_summary(summary: dict[str, Any] | None) -> None:
    if not summary:
        return
    overall = summary.get("overall") or {}
    micro = overall.get("micro") or {}
    macro = overall.get("macro") or summary.get("macro") or {}
    final = overall.get("final") or micro
    print(
        f"[SUMMARY] FINAL (micro, all records) "
        f"instrument_attn_ratio_avg={final.get('instrument_attn_ratio_avg')} "
        f"mIoU_avg={final.get('mIoU_avg')} n={micro.get('n_samples')}",
        file=sys.stderr,
    )
    print(
        f"[SUMMARY] macro (mean of per-instrument avgs) "
        f"instrument_attn_ratio_avg={macro.get('instrument_attn_ratio_avg')} "
        f"mIoU_avg={macro.get('mIoU_avg')} "
        f"n_instruments={macro.get('n_instruments')}",
        file=sys.stderr,
    )
    for key, block in (summary.get("instruments") or {}).items():
        print(
            f"  {key}: n={block.get('n_samples')} "
            f"inst_ratio_avg={block.get('instrument_attn_ratio_avg')} "
            f"mIoU_avg={block.get('mIoU_avg')}",
            file=sys.stderr,
        )


# ── Cross-attention maps ──────────────────────────────────────────────────────

def _emphasize_peak_grid(
    grid: np.ndarray,
    *,
    top_percentile: float,
    gamma: float,
) -> np.ndarray:
    """Min–max + optional soft floor + mild gamma."""
    g = grid.astype(np.float32, copy=True)
    if g.size == 0:
        return g
    lo, hi = float(g.min()), float(g.max())
    if hi > lo:
        g = (g - lo) / (hi - lo)
    if top_percentile > 0.0:
        floor = float(np.percentile(g, top_percentile))
        g = np.clip((g - floor) / (float(g.max()) - floor + 1e-8), 0.0, 1.0)
    if gamma != 1.0:
        g = np.power(g, gamma, dtype=np.float32)
    lo, hi = float(g.min()), float(g.max())
    if hi > lo:
        g = (g - lo) / (hi - lo)
    return g.astype(np.float32)


def _compute_maps_for_source(
    q_patches: torch.Tensor,
    q_cls: torch.Tensor,
    t_patches: torch.Tensor,
    grid_h: int,
    grid_w: int,
) -> dict[str, np.ndarray]:
    def _norm01(a: np.ndarray) -> np.ndarray:
        lo, hi = a.min(), a.max()
        return ((a - lo) / (hi - lo)).astype(np.float32) if hi > lo else a.astype(np.float32)

    def _smooth_grid(grid: np.ndarray, sigma: float) -> np.ndarray:
        if sigma <= 0:
            return grid
        try:
            from scipy.ndimage import gaussian_filter
        except ImportError:
            return grid
        return gaussian_filter(grid.astype(np.float32), sigma=sigma, mode="nearest")

    q_n = F.normalize(q_cls.float().unsqueeze(0), dim=-1)
    t_n = F.normalize(t_patches.float(), dim=-1)
    cls_logits = (q_n @ t_n.T).squeeze(0)
    cls_softmax = F.softmax(cls_logits * CLS_SOFTMAX_LOGIT_SCALE, dim=0)
    mean_q = F.normalize(q_patches.float().mean(0, keepdim=True), dim=-1)
    norm_k = F.normalize(t_patches.float(), dim=-1)
    mean_cosine_raw = (mean_q @ norm_k.T).squeeze(0)
    # Sharpen for heatmap: raw cosine is often flat across the frame.
    mean_cosine = F.softmax(mean_cosine_raw * max(4.0, mean_cosine_raw.shape[-1] ** 0.25), dim=0)
    q_pn = F.normalize(q_patches.float(), dim=-1)
    patch_logits = q_pn @ t_n.T
    patch_cross_mean = F.softmax(patch_logits * PATCH_CROSS_MEAN_LOGIT_SCALE, dim=-1).mean(dim=0)
    patch_cross_max = F.softmax(patch_logits * PATCH_CROSS_MAX_LOGIT_SCALE, dim=-1).max(dim=0).values
    patch_cross_max = F.softmax(patch_cross_max * PATCH_CROSS_MAX_POST_SCALE, dim=0)

    def _to_grid(
        vec: torch.Tensor,
        *,
        smooth_sigma: float = 0.0,
        emphasize_peaks: bool = False,
    ) -> np.ndarray:
        flat = vec.detach().float().cpu().numpy().reshape(-1)
        n = grid_h * grid_w
        if flat.size != n:
            flat = (
                F.interpolate(
                    torch.tensor(flat).view(1, 1, -1),
                    size=(n,),
                    mode="linear",
                    align_corners=False,
                )
                .squeeze()
                .numpy()
            )
        grid = flat.reshape(grid_h, grid_w)
        if emphasize_peaks:
            grid = _emphasize_peak_grid(
                grid,
                top_percentile=HEATMAP_TOP_PERCENTILE,
                gamma=HEATMAP_GAMMA,
            )
            grid = _smooth_grid(grid, smooth_sigma)
            lo, hi = float(grid.min()), float(grid.max())
            if hi > lo:
                grid = (grid - lo) / (hi - lo)
            return grid.astype(np.float32)
        grid = _smooth_grid(grid, smooth_sigma)
        return _norm01(grid)

    return {
        "cls_softmax": _to_grid(cls_softmax, smooth_sigma=CLS_HEATMAP_GAUSS_SIGMA),
        "mean_cosine": _to_grid(mean_cosine),
        "patch_cross_mean": _to_grid(patch_cross_mean),
        "patch_cross_max": _to_grid(
            patch_cross_max,
            smooth_sigma=PATCH_CROSS_MAX_GAUSS_SIGMA,
            emphasize_peaks=True,
        ),
    }


def compute_attention_all_sources(
    bp: PatchFeatureBackbone,
    query_image: Image.Image,
    test_image: Image.Image,
    *,
    query_encoding: str = "vision_ref",
) -> dict[str, dict[str, np.ndarray]]:
    enc = (query_encoding or "vision_ref").strip().lower()
    if enc == "backbone":
        enc = "vision_ref"

    use_vision_pairs = enc in ("vision_ref",)
    use_pixel_pairs = enc in ("pixel_grid", "single")

    if use_vision_pairs:
        q_feats = bp.extract(query_image)
        t_feats = bp.extract(test_image)
    elif use_pixel_pairs:
        # RGB patches on both sides (same random projection); do not mix with vision tokens.
        probe = bp.extract(test_image)
        q_mode = "single" if enc == "single" else "pixel_grid"
        q_feats = {}
        t_feats = {}
        for src in bp.source_names:
            t_patches, _ = probe[src]
            feat_dim = int(t_patches.shape[-1])
            t_patches, t_cls = image_as_pixel_patch_tokens(
                test_image, bp.grid_h, bp.grid_w, feat_dim, bp.device, mode="pixel_grid"
            )
            q_patches, q_cls = image_as_pixel_patch_tokens(
                query_image, bp.grid_h, bp.grid_w, feat_dim, bp.device, mode=q_mode
            )
            q_feats[src] = (q_patches, q_cls)
            t_feats[src] = (t_patches, t_cls)
    else:
        raise ValueError(
            f"Unknown query_encoding={query_encoding!r}; "
            "use vision_ref, pixel_grid, single, or backbone."
        )

    out: dict[str, dict[str, np.ndarray]] = {}
    for src in bp.source_names:
        q_patches, q_cls = q_feats[src]
        t_patches, _ = t_feats[src]
        out[src] = _compute_maps_for_source(q_patches, q_cls, t_patches, bp.grid_h, bp.grid_w)
    return out


def map_stats(attn_map: np.ndarray) -> dict[str, float]:
    flat = attn_map.flatten()
    return {"max": float(flat.max()), "mean": float(flat.mean()), "top10pct": float(np.percentile(flat, 90))}


# ── Visualization ─────────────────────────────────────────────────────────────

def overlay_heatmap(image: Image.Image, attn_map: np.ndarray, alpha: float = 0.55) -> Image.Image:
    import matplotlib.cm as cm

    H, W = image.size[1], image.size[0]
    attn_up = F.interpolate(
        torch.tensor(attn_map).unsqueeze(0).unsqueeze(0),
        size=(H, W),
        mode="bilinear",
        align_corners=False,
    ).squeeze().numpy()
    heat = Image.fromarray((cm.get_cmap("jet")(attn_up)[:, :, :3] * 255).astype(np.uint8)).convert("RGBA")
    return Image.blend(image.convert("RGBA"), heat, alpha=alpha).convert("RGB")


def make_comparison_figure(
    query_image: Image.Image,
    query_name: str,
    test_image: Image.Image,
    test_label: str,
    all_maps: dict[str, dict[str, np.ndarray]],
    source_labels: dict[str, str],
    frame_labels: set[str],
    alpha: float,
    bbox_xyxy: tuple[float, float, float, float] | None,
    annotation: dict | None,
    query_match_method: str,
    run_title: str,
    display_size: int = 220,
) -> Image.Image:
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    method_short = {
        "cls_softmax": "CLS softmax",
        "mean_cosine": "Mean cosine",
        "patch_cross_mean": "Patch cross mean",
        "patch_cross_max": "Patch cross max",
    }
    matched = query_matches_frame_labels(query_name, frame_labels)
    source_names = list(source_labels.keys())
    n_rows = max(1, len(source_names))
    n_cols = 2 + len(COMPARISON_METHODS)  # query | test original | heatmap(s)

    fig = plt.figure(figsize=(n_cols * 3.0, n_rows * 3.4), facecolor="#12121f")
    gs = GridSpec(n_rows, n_cols, figure=fig, wspace=0.15, hspace=0.25)

    ann_bit = ""
    if annotation:
        ann_bit = f" | {annotation.get('triplet', '')} | {annotation.get('phase', '')}"
    fig.suptitle(
        f"{run_title} | query: {query_name}{' ✓' if matched else ''} [{query_match_method}] | "
        f"test: {test_label}{ann_bit}",
        fontsize=8,
        color="white",
        y=1.02,
    )

    def _add_bbox(ax, *, edgecolor: str = "lime") -> None:
        if bbox_xyxy is None:
            return
        xmin, ymin, xmax, ymax = bbox_xyxy
        ax.add_patch(
            mpatches.Rectangle(
                (xmin * display_size, ymin * display_size),
                (xmax - xmin) * display_size,
                (ymax - ymin) * display_size,
                linewidth=2,
                edgecolor=edgecolor,
                facecolor="none",
            )
        )

    def _show(ax, img: Image.Image, title: str, *, title_color: str = "white") -> None:
        ax.imshow(img.resize((display_size, display_size)))
        ax.set_title(title, fontsize=7, color=title_color)
        ax.axis("off")

    heatmap_title_suffix = "\n+ GT bbox" if bbox_xyxy is not None else ""

    # Query (once, full height) — reference image only, no per-source duplicate
    ax_q = fig.add_subplot(gs[:, 0])
    _show(ax_q, query_image, f"Query\n{query_name}", title_color="lightyellow")

    # Test frame original (once, full height)
    ax_t = fig.add_subplot(gs[:, 1])
    test_disp = test_image.resize((display_size, display_size))
    ax_t.imshow(test_disp)
    test_title = f"Test (original)\n{test_label}"
    if bbox_xyxy is not None:
        test_title += "\nGT bbox"
    ax_t.set_title(test_title, fontsize=7, color="lightgreen")
    ax_t.axis("off")
    _add_bbox(ax_t)

    for row_i, src in enumerate(source_names):
        for col_i, method in enumerate(COMPARISON_METHODS):
            ax = fig.add_subplot(gs[row_i, 2 + col_i])
            ov = overlay_heatmap(test_image, all_maps[src][method], alpha=alpha)
            title = f"{method_short[method]}{heatmap_title_suffix}\n[{source_labels[src]}]"
            _show(ax, ov, title, title_color="lightcyan")
            _add_bbox(ax)

    plt.tight_layout(pad=0.5)
    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).copy()


# ── Per-frame run ─────────────────────────────────────────────────────────────

def run_cross_attention_for_frame(
    *,
    bp: PatchFeatureBackbone,
    test_path: Path,
    test_label: str,
    labels_dir: Path | None = None,
    vid_name: str | None = None,
    frame_idx: int | None = None,
    query_index: dict[str, Path],
    query_from_gt_crop: bool,
    fuzzy_cutoff: float,
    output_dir: Path,
    alpha: float,
    run_title: str,
    instrument_filter: str | None = None,
    query_encoding: str = "vision_ref",
    annotations: list[dict[str, Any]] | None = None,
    frame_labels: set[str] | None = None,
) -> list[dict[str, Any]]:
    test_image = Image.open(test_path).convert("RGB")
    resolved_labels: set[str] = set(frame_labels or ())
    resolved_annotations: list[dict[str, Any]] = list(annotations or [])

    if not resolved_annotations:
        if vid_name is not None and frame_idx is not None and labels_dir is not None:
            resolved_labels = get_frame_labels(vid_name, frame_idx, labels_dir)
            resolved_annotations = get_full_frame_annotations(vid_name, frame_idx, labels_dir)
        elif instrument_filter:
            resolved_annotations = [{"instrument": instrument_filter, "bbox_xyxy": None}]
        elif query_index:
            resolved_annotations = [
                {"instrument": p.stem, "bbox_xyxy": None} for p in query_index.values()
            ]

    if vid_name is not None and frame_idx is not None and labels_dir is not None and not resolved_annotations:
        print(
            f"[WARN] No visible bbox annotations on {vid_name} frame {frame_idx}"
            + (f" (filtered: {instrument_filter})" if instrument_filter else ""),
            file=sys.stderr,
        )

    seen_inst: set[str] = set()
    records: list[dict[str, Any]] = []

    for ann in resolved_annotations:
        inst = ann["instrument"]
        if inst in seen_inst:
            continue
        seen_inst.add(inst)
        if instrument_filter and inst != instrument_filter:
            continue

        bbox = ann.get("bbox_xyxy")
        query_img, query_name, qmatch = resolve_query_image(
            instrument=inst,
            test_image=test_image,
            bbox_xyxy=bbox or (0.0, 0.0, 1.0, 1.0),
            query_index=query_index,
            query_from_gt_crop=query_from_gt_crop,
            fuzzy_cutoff=fuzzy_cutoff,
        )
        if query_img is None:
            print(f"  [SKIP] no query for {inst}", file=sys.stderr)
            continue

        print(f"  [{query_name}] {inst} ...", file=sys.stderr, end="")
        try:
            all_maps = compute_attention_all_sources(
                bp, query_img, test_image, query_encoding=query_encoding
            )
        except Exception as exc:
            print(f" FAIL: {exc}", file=sys.stderr)
            continue

        fig = make_comparison_figure(
            query_img,
            query_name,
            test_image,
            test_label,
            all_maps,
            bp.source_labels,
            resolved_labels,
            alpha,
            bbox,
            ann if bbox else None,
            qmatch,
            run_title,
        )
        stem = f"{test_label}_query_{query_name}"
        fig.save(output_dir / f"{stem}_comparison.png")
        for src in bp.source_names:
            for method, amap in all_maps[src].items():
                overlay_heatmap(test_image, amap, alpha=alpha).save(
                    output_dir / f"{stem}_{src}_{method}.png"
                )

        stats: dict[str, Any] = {
            s: {m: map_stats(all_maps[s][m]) for m in METHODS} for s in bp.source_names
        }
        if bbox is not None:
            for src in bp.source_names:
                for method in METHODS:
                    bbox_stats = attention_bbox_stats(
                        all_maps[src][method], bbox, bp.grid_h, bp.grid_w
                    )
                    miou_v, pred_bbox = attention_bbox_miou(
                        all_maps[src][method], bbox, bp.grid_h, bp.grid_w
                    )
                    bbox_stats["mIoU"] = round(miou_v, 6)
                    if pred_bbox is not None:
                        bbox_stats["pred_bbox_xyxy"] = [round(v, 6) for v in pred_bbox]
                    stats[src][method]["bbox"] = bbox_stats

        primary = bp.source_names[0]
        bbox_primary = stats[primary][SUMMARY_ATTENTION_METHOD].get("bbox", {})
        ratio = bbox_primary.get("instrument_attn_ratio", float("nan"))
        miou_v = bbox_primary.get("mIoU", float("nan"))
        print(
            f" done  {primary}_inst_ratio={ratio:.3f} mIoU={miou_v:.3f}",
            file=sys.stderr,
        )

        rec: dict[str, Any] = {
            "test_label": test_label,
            "test_path": str(test_path),
            "query": query_name,
            "query_encoding": query_encoding,
            "instrument": inst,
            "instrument_key": instrument_key_for_record(inst, ann),
            "query_match_method": qmatch,
            "matched_label": query_matches_frame_labels(query_name, resolved_labels),
            "stats": stats,
        }
        if bbox is not None:
            rec["instrument_attn_ratio"] = bbox_primary.get("instrument_attn_ratio")
            rec["mIoU"] = bbox_primary.get("mIoU")
            rec["pred_bbox_xyxy"] = bbox_primary.get("pred_bbox_xyxy")
        if vid_name:
            rec["video"] = vid_name
        if frame_idx is not None:
            rec["frame_index"] = frame_idx
        if bbox:
            rec["bbox_xyxy"] = list(bbox)
            rec["annotation"] = {k: ann[k] for k in ("triplet", "verb", "target", "phase") if ann.get(k)}
        records.append(rec)

    return records


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CholecT50 cross-attention visual grounding (DINO / concat).")
    p.add_argument("--backend", choices=BACKEND_CHOICES, default="prismatic")
    p.add_argument(
        "--feature-backbone",
        choices=("timm", "prismatic", "hf"),
        default="timm",
        help="timm/prismatic: DINO+SigLIP; hf: --backend model's vision tower (AutoProcessor).",
    )
    p.add_argument(
        "--query-encoding",
        choices=("vision_ref", "pixel_grid", "single", "backbone"),
        default="vision_ref",
        help=(
            "vision_ref (default): query+test via --feature-backbone (best for tool localization); "
            "pixel_grid/single: RGB patch tokens on both sides; backbone: alias for vision_ref."
        ),
    )
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--video", type=str, default=None, help="VID name + --frame for single frame.")
    mode.add_argument("--test-image", type=Path, default=None)
    mode.add_argument("--eval-all", action="store_true", help="All bbox annotations (no per-instrument cap).")
    p.add_argument("--frame", type=int, default=None)
    p.add_argument("--dataset-root", type=Path, default=CHALLENGE_VAL_ROOT)
    p.add_argument("--videos-root", type=Path, default=None)
    p.add_argument("--cholect-root-fallback", type=Path, default=CHOLECT_ROOT)
    p.add_argument("--instrument", type=str, default=None)
    p.add_argument("--samples-per-instrument", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit", type=int, default=None, help="Max frames in batch mode.")
    p.add_argument("--query-dir", type=Path, default=_DEFAULT_QUERY_DIR)
    p.add_argument("--query-from-gt-crop", action="store_true")
    p.add_argument("--fuzzy-cutoff", type=float, default=0.4)
    p.add_argument("--model-id", type=str, default=None)
    p.add_argument("--model-name", type=str, default=None)
    p.add_argument("--vlm-checkpoint", type=Path, default=None)
    p.add_argument("--vlm-config", type=Path, default=None)
    p.add_argument("--hf-token", type=Path, default=_DEFAULT_HF_TOKEN)
    p.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_SIZE)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--device", type=str, default="0")
    p.add_argument("--alpha", type=float, default=0.55)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    backend = args.backend
    model_id = resolve_model_id(backend, args.model_id)
    model_name = resolve_output_model_name(backend, args.model_name, model_id)

    out_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else (_DEFAULT_OUTPUT_ROOT / f"{model_name}").resolve()
    )
    viz_dir = out_dir / "viz"
    viz_dir.mkdir(parents=True, exist_ok=True)

    image_size = args.image_size
    if args.feature_backbone == "prismatic":
        image_size = infer_pil_side(args)

    hf_token = resolve_hf_token(backend, args.hf_token)
    bp = load_patch_feature_backbone(
        args.feature_backbone,
        backend=backend,
        model_id=model_id,
        hf_token=hf_token,
        device=device,
        image_size=image_size,
        vlm_checkpoint=args.vlm_checkpoint,
        vlm_config=args.vlm_config,
    )

    args.dataset_root = args.dataset_root.resolve()
    labels_dir = args.dataset_root / "labels"
    if not labels_dir.is_dir():
        raise FileNotFoundError(f"labels not found: {labels_dir}")

    query_dir = args.query_dir.resolve()
    query_index = build_query_index(query_dir)
    if not query_index and not args.query_from_gt_crop:
        print(
            f"[WARN] Empty query index at {query_dir}. Use --query-from-gt-crop or add PNGs.",
            file=sys.stderr,
        )

    run_title = f"{args.feature_backbone} | query={args.query_encoding} | {model_name}"
    print(
        f"[INFO] out={out_dir}  grid={bp.grid_h}x{bp.grid_w}  query_encoding={args.query_encoding}  "
        f"device={bp.device}  queries={len(query_index)}",
        file=sys.stderr,
    )

    all_records: list[dict[str, Any]] = []

    # ── Single test image ─────────────────────────────────────────────────────
    if args.test_image is not None:
        test_path = args.test_image.resolve()
        test_label = test_path.stem
        all_records = run_cross_attention_for_frame(
            bp=bp,
            test_path=test_path,
            test_label=test_label,
            labels_dir=labels_dir,
            vid_name=None,
            frame_idx=None,
            query_index=query_index,
            query_from_gt_crop=args.query_from_gt_crop,
            fuzzy_cutoff=args.fuzzy_cutoff,
            output_dir=viz_dir,
            alpha=args.alpha,
            run_title=run_title,
            instrument_filter=args.instrument,
            query_encoding=args.query_encoding,
        )

    # ── Single video frame ────────────────────────────────────────────────────
    elif args.video is not None:
        if args.frame is None:
            raise SystemExit("--video requires --frame")
        vid = args.video if args.video.upper().startswith("VID") else f"VID{args.video}"
        video_roots = discover_video_roots(args)
        if not video_roots:
            raise FileNotFoundError("No video roots; set --videos-root or CHOLECT50_VIDEOS_ROOT")
        test_path = resolve_frame_path(vid, args.frame, video_roots)
        if test_path is None:
            raise FileNotFoundError(f"Frame not found: {vid} f{args.frame:06d}")
        test_label = f"{vid}_f{args.frame:06d}"
        all_records = run_cross_attention_for_frame(
            bp=bp,
            test_path=test_path,
            test_label=test_label,
            labels_dir=labels_dir,
            vid_name=vid,
            frame_idx=args.frame,
            query_index=query_index,
            query_from_gt_crop=args.query_from_gt_crop,
            fuzzy_cutoff=args.fuzzy_cutoff,
            output_dir=viz_dir,
            alpha=args.alpha,
            run_title=run_title,
            instrument_filter=args.instrument,
            query_encoding=args.query_encoding,
        )

    # ── Batch sample ──────────────────────────────────────────────────────────
    else:
        video_roots = discover_video_roots(args)
        if not video_roots:
            raise FileNotFoundError("No video roots; set --videos-root")
        items = collect_instrument_annotations(
            labels_dir=labels_dir,
            video_roots=video_roots,
            video_filter=None,
            instrument_filter=args.instrument,
        )
        if args.eval_all:
            sampled = items
        else:
            sampled = sample_by_instrument(
                items, cap_per_instrument=max(0, args.samples_per_instrument), seed=args.seed
            )
        if args.limit is not None:
            sampled = sampled[: args.limit]

        by_frame: dict[tuple[str, int], list[dict[str, Any]]] = {}
        for it in sampled:
            by_frame.setdefault((it["vid"], it["frame_index"]), []).append(it)

        print(f"[BATCH] {len(by_frame)} frames from {len(sampled)} annotations", file=sys.stderr)
        for (vid, fi), frame_items in sorted(by_frame.items()):
            test_path = frame_items[0]["img_path"]
            test_label = f"{vid}_f{fi:06d}"
            print(f"\n[{test_label}] {test_path}", file=sys.stderr)
            recs = run_cross_attention_for_frame(
                bp=bp,
                test_path=Path(test_path),
                test_label=test_label,
                labels_dir=labels_dir,
                vid_name=vid,
                frame_idx=fi,
                query_index=query_index,
                query_from_gt_crop=args.query_from_gt_crop,
                fuzzy_cutoff=args.fuzzy_cutoff,
                output_dir=viz_dir,
                alpha=args.alpha,
                run_title=run_title,
                query_encoding=args.query_encoding,
            )
            all_records.extend(recs)

    per_inst = None
    overall_avg = None
    if all_records:
        per_inst = build_per_instrument_summary(
            all_records,
            primary_source=bp.source_names[0],
            samples_per_instrument=(
                None if args.eval_all else max(0, args.samples_per_instrument)
            ),
            eval_all=bool(args.eval_all),
        )
        overall_avg = overall_avg_from_summary(per_inst)
        print_per_instrument_summary(per_inst)

    payload = {
        "task": "visual_cross_attention_cholect50",
        "backend": backend,
        "feature_backbone": args.feature_backbone,
        "query_encoding": args.query_encoding,
        "model_id": model_id,
        "model_name": model_name,
        "dataset_root": str(args.dataset_root),
        "query_dir": str(query_dir),
        "query_from_gt_crop": bool(args.query_from_gt_crop),
        "samples_per_instrument": args.samples_per_instrument,
        "eval_all": bool(args.eval_all),
        "seed": args.seed,
        "grid": f"{bp.grid_h}x{bp.grid_w}",
        "sources": bp.source_names,
        "source_labels": bp.source_labels,
        "methods": METHODS,
        "summary_method": SUMMARY_ATTENTION_METHOD,
        "vision_meta": bp.meta,
        "visualization_dir": str(viz_dir),
        "n_records": len(all_records),
        "per_instrument_summary": per_inst,
        "overall_avg": overall_avg,
        "records": all_records,
    }
    out_json = out_dir / "cross_attention.json"
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"\n[SAVE] {out_json}  viz → {viz_dir}  ({len(all_records)} records)", file=sys.stderr)


if __name__ == "__main__":
    main()
