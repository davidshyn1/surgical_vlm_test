"""CholecT50 challenge-val data loading for surgical_vlm_test."""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from utils import load_label_json, normalize_instrument_name, parse_annotation_row

_PKG_ROOT = Path(__file__).resolve().parent
REPO_ROOT = _PKG_ROOT.parent
CHALLENGE_VAL_ROOT = REPO_ROOT / "eval" / "cholect50-challenge-val"

_DEFAULT_MODEL_IDS = {
    "prismatic": "prism-dinosiglip+7b",
    "cosmos": "nvidia/Cosmos-Reason2-2B",
    "groot": "nvidia/GR00T-H",
}

_IMG_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff")


def infer_pil_side(args: argparse.Namespace) -> int:
    cfg_path = None
    if getattr(args, "vlm_config", None) is not None and args.vlm_config.is_file():
        cfg_path = args.vlm_config
    elif getattr(args, "vlm_checkpoint", None) is not None:
        ckpt = args.vlm_checkpoint
        parent = ckpt.parent
        cand = (parent.parent if parent.name == "checkpoints" else parent) / "config.json"
        cfg_path = cand if cand.is_file() else None
    if cfg_path is not None:
        try:
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
            backbone = str((cfg.get("model") or {}).get("vision_backbone_id", ""))
            if "224" in backbone:
                return 224
        except Exception:
            pass
    if "224" in str(getattr(args, "model_id", None) or ""):
        return 224
    return 384


def _numeric_vid_id(vid_name: str) -> int | None:
    m = re.match(r"^VID(\d+)$", (vid_name or "").strip(), re.IGNORECASE)
    return int(m.group(1)) if m else None


def _video_name_candidates(vid_name: str) -> list[str]:
    vid_id = _numeric_vid_id(vid_name)
    if vid_id is None:
        return [vid_name]
    return [f"VID{vid_id}", f"VID{vid_id:02d}"]


def discover_video_roots(args: argparse.Namespace) -> list[Path]:
    roots: list[Path] = []
    if args.videos_root is not None:
        roots.append(args.videos_root.resolve())
    roots.extend([
        (args.dataset_root / "videos").resolve(),
        (args.cholect_root_fallback / "videos").resolve(),
    ])
    uniq: list[Path] = []
    seen: set[str] = set()
    for r in roots:
        k = str(r)
        if k in seen or not r.is_dir():
            continue
        seen.add(k)
        uniq.append(r)
    return uniq


def resolve_frame_path(vid_name: str, frame_index: int, video_roots: list[Path]) -> Path | None:
    stem = f"{frame_index:06d}"
    for vroot in video_roots:
        for vn in _video_name_candidates(vid_name):
            vdir = vroot / vn
            if not vdir.is_dir():
                continue
            for ext in _IMG_EXTS:
                p = vdir / f"{stem}{ext}"
                if p.is_file():
                    return p.resolve()
    return None


def collect_instrument_annotations(
    *,
    labels_dir: Path,
    video_roots: list[Path],
    video_filter: str | None,
    instrument_filter: str | None,
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    instrument_filter_norm = normalize_instrument_name(instrument_filter) if instrument_filter else ""
    label_files = sorted(labels_dir.glob("VID*.json"))
    if video_filter:
        q = _numeric_vid_id(video_filter)
        if q is not None:
            label_files = [p for p in label_files if _numeric_vid_id(p.stem) == q]
        else:
            label_files = [p for p in label_files if p.stem.upper() == video_filter.upper()]

    for lf in label_files:
        vid = lf.stem
        data = load_label_json(labels_dir, vid)
        categories = data.get("categories") or {}
        for fk, rows in sorted((data.get("annotations") or {}).items(), key=lambda x: int(x[0])):
            fi = int(fk)
            img_path = resolve_frame_path(vid, fi, video_roots)
            if img_path is None:
                continue
            for ann_index, row in enumerate(rows):
                parsed = parse_annotation_row(list(row), categories)
                if parsed is None:
                    continue
                inst_name = str(parsed.get("instrument_name", "")).strip()
                inst_norm = normalize_instrument_name(inst_name)
                if not inst_norm:
                    continue
                if instrument_filter_norm and inst_norm != instrument_filter_norm:
                    continue
                items.append(
                    {
                        "vid": vid,
                        "frame_index": fi,
                        "frame_key": fk,
                        "ann_index": ann_index,
                        "img_path": img_path,
                        "instrument_name": inst_name,
                        "instrument_norm": inst_norm,
                        "parsed": parsed,
                    }
                )
    return items


def sample_by_instrument(items: list[dict[str, Any]], cap_per_instrument: int, seed: int) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for it in items:
        grouped[it["instrument_norm"]].append(it)
    selected: list[dict[str, Any]] = []
    for inst in sorted(grouped.keys()):
        pool = list(grouped[inst])
        random.Random(seed ^ hash(inst)).shuffle(pool)
        selected.extend(pool[: max(0, cap_per_instrument)])
    return selected
