"""Shared utilities for surgical_vlm_test (no dependency on surgical_vlm_grounding)."""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import torch

_PKG_ROOT = Path(__file__).resolve().parent
_REPO_ROOT = _PKG_ROOT.parent

CHOLECT_ROOT = Path(
    os.environ.get("CHOLECT_ROOT", "/home/irteam/data-vol1/surgical/CholecT50")
)

ACTION_OPTIONS_FIXED = [
    "aspirate", "clip", "coagulate", "cut", "dissect",
    "grasp", "irrigate", "null-verb", "pack", "retract",
]

TARGET_OPTIONS_FIXED = [
    "abd-wall/cavity", "adhesion", "blood-vessel", "cystic-artery",
    "cystic-duct", "cystic-pedicle", "cystic-plate", "fluid",
    "gallbladder", "gut", "liver", "null-target", "omentum",
    "peritoneum", "specimen-bag",
]


def resolve_device(device_arg: str) -> torch.device:
    s = (device_arg or "").strip().lower()
    if s == "cpu":
        return torch.device("cpu")
    if s.isdigit():
        if not torch.cuda.is_available():
            print(f"WARN CUDA unavailable; fallback to cpu (requested GPU index {s})", file=sys.stderr)
            return torch.device("cpu")
        return torch.device(f"cuda:{int(s)}")
    if s.startswith("cuda"):
        if not torch.cuda.is_available():
            print(f"WARN CUDA unavailable; fallback to cpu (requested {device_arg})", file=sys.stderr)
            return torch.device("cpu")
        return torch.device(device_arg)
    raise ValueError(f"Invalid --device {device_arg!r}; use digits (e.g. 0), 'cpu', or 'cuda:N'.")


def normalize_instrument_name(s: str | None) -> str:
    t = (s or "").strip().lower()
    t = t.replace("_", "-").replace(" ", "-")
    t = re.sub(r"-+", "-", t)
    return t


def result_lookup_key(rec: dict) -> tuple[str, str] | None:
    inp = rec.get("input") or {}
    path = inp.get("image_path")
    if not path:
        return None
    return (path, inp.get("tool") or "")


def load_results_for_resume(out_path: Path) -> tuple[list[dict], dict[tuple[str, str], int]]:
    if not out_path.exists():
        return [], {}
    try:
        with out_path.open("r", encoding="utf-8") as f:
            old = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"WARN: could not read {out_path}: {e}; starting fresh", file=sys.stderr)
        return [], {}
    results = old.get("results", [])
    key_to_idx: dict[tuple[str, str], int] = {}
    for i, r in enumerate(results):
        k = result_lookup_key(r)
        if k is not None:
            key_to_idx[k] = i
    return results, key_to_idx


def upsert_result(
    results: list[dict],
    key_to_idx: dict[tuple[str, str], int],
    row_key: tuple[str, str],
    entry: dict,
) -> None:
    if row_key in key_to_idx:
        results[key_to_idx[row_key]] = entry
    else:
        results.append(entry)
        key_to_idx[row_key] = len(results) - 1


def category_lookup(categories: dict[str, Any], group: str, idx: int) -> str:
    g = categories.get(group) or {}
    return str(g.get(str(int(idx)), g.get(str(idx), f"?{idx}")))


def parse_annotation_row(row: list[Any], categories: dict[str, Any]) -> dict[str, Any] | None:
    if len(row) < 9:
        return None
    triplet_id = int(row[0])
    instrument_id = int(row[1])
    if triplet_id < 0 or instrument_id < 0:
        return None
    visibility = float(row[2])
    verb_id = int(row[7])
    target_id = int(row[8])
    phase_id = int(row[14]) if len(row) > 14 else None

    inst_name = category_lookup(categories, "instrument", instrument_id)
    verb_name = category_lookup(categories, "verb", verb_id)
    tgt_name = category_lookup(categories, "target", target_id)
    triplet_str = category_lookup(categories, "triplet", triplet_id)
    phase_name = category_lookup(categories, "phase", phase_id) if phase_id is not None else ""

    return {
        "triplet_id": triplet_id,
        "instrument_id": instrument_id,
        "verb_id": verb_id,
        "target_id": target_id,
        "visibility": visibility,
        "instrument_name": inst_name,
        "verb_name": verb_name,
        "target_name": tgt_name,
        "triplet_str": triplet_str,
        "phase_id": phase_id,
        "phase_name": phase_name,
    }


def load_label_json(labels_dir: Path, vid_name: str) -> dict[str, Any]:
    path = labels_dir / f"{vid_name}.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing label file: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)
