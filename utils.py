"""Shared utilities for surgical_vlm_test (no dependency on surgical_vlm_grounding)."""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageDraw, ImageFont

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


_LORA_ANSWER_TAG_BODY_RE = re.compile(
    r"<answer>\s*(.*?)\s*</answer>",
    re.IGNORECASE | re.DOTALL,
)
_LORA_ANSWER_TAG_OPEN_RE = re.compile(
    r"<answer>\s*(.*)\s*$",
    re.IGNORECASE | re.DOTALL,
)


def strip_lora_answer_tags(text: str | None) -> str:
    """
    Extract text inside ``<answer>...</answer>`` from LoRA fine-tuned model outputs.

    If no answer tags are present, returns the stripped input unchanged.
    Multiple tagged spans are joined with ``", "``.
    """
    if text is None:
        return ""
    s = str(text).strip()
    if not s or "<answer" not in s.lower():
        return s
    bodies = [m.strip() for m in _LORA_ANSWER_TAG_BODY_RE.findall(s) if m.strip()]
    if bodies:
        return ", ".join(bodies)
    m = _LORA_ANSWER_TAG_OPEN_RE.search(s)
    if m:
        inner = re.sub(r"</answer>\s*$", "", m.group(1), flags=re.IGNORECASE).strip()
        return inner
    return s


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


def cholect_row_bbox_xyxy(row: list[Any]) -> tuple[float, float, float, float] | None:
    """CholecT50 row bbox: indices 3–6 are normalized [x, y, w, h] (top-left + size)."""
    if len(row) < 7:
        return None
    try:
        x, y, w, h = float(row[3]), float(row[4]), float(row[5]), float(row[6])
    except (TypeError, ValueError):
        return None
    if w <= 0.0 or h <= 0.0:
        return None
    return clamp_bbox_xyxy_01((x, y, x + w, y + h))


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
    bbox_xyxy = cholect_row_bbox_xyxy(row)

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
        "bbox_xyxy": bbox_xyxy,
    }


def load_label_json(labels_dir: Path, vid_name: str) -> dict[str, Any]:
    path = labels_dir / f"{vid_name}.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing label file: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


# ── Bbox parsing / IoU (instrument localization) ─────────────────────────────

_BBOX_BRACKET_PATTERN = re.compile(
    r"\[\s*([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)\s*,\s*"
    r"([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)\s*,\s*"
    r"([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)\s*,\s*"
    r"([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)\s*\]"
)

_BBOX_PAREN_PATTERN = re.compile(
    r"\(\s*([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)\s*,\s*"
    r"([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)\s*,\s*"
    r"([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)\s*,\s*"
    r"([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)\s*\)"
)

QWEN_VL_BBOX_SCALE = 1000.0


def _order_bbox_xyxy(
    xmin: float, ymin: float, xmax: float, ymax: float,
) -> tuple[float, float, float, float] | None:
    if xmax < xmin:
        xmin, xmax = xmax, xmin
    if ymax < ymin:
        ymin, ymax = ymax, ymin
    if xmax <= xmin or ymax <= ymin:
        return None
    return (xmin, ymin, xmax, ymax)


def parse_raw_bbox_xyxy(text: str) -> tuple[float, float, float, float] | None:
    """Extract xyxy from [..], (x1,y1,x2,y2), or (x1,y1), (x2,y2) without normalizing."""
    for pattern in (_BBOX_BRACKET_PATTERN, _BBOX_PAREN_PATTERN):
        for m in pattern.finditer(text or ""):
            try:
                vals = tuple(float(m.group(i)) for i in range(1, 5))
            except ValueError:
                continue
            ordered = _order_bbox_xyxy(*vals)
            if ordered is not None:
                return ordered
    corner_pat = re.compile(
        r"\(\s*([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)\s*,\s*"
        r"([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)\s*\)"
    )
    corners = [
        (float(m.group(1)), float(m.group(2)))
        for m in corner_pat.finditer(text or "")
    ]
    if len(corners) >= 2:
        xs = [c[0] for c in corners[:2]]
        ys = [c[1] for c in corners[:2]]
        return _order_bbox_xyxy(min(xs), min(ys), max(xs), max(ys))
    return None


def clamp_bbox_xyxy_px(
    bbox: tuple[float, float, float, float],
    *,
    image_width: int,
    image_height: int,
) -> tuple[float, float, float, float] | None:
    w = max(1, int(image_width))
    h = max(1, int(image_height))
    xmin, ymin, xmax, ymax = bbox
    xmin = max(0.0, min(float(w), xmin))
    ymin = max(0.0, min(float(h), ymin))
    xmax = max(0.0, min(float(w), xmax))
    ymax = max(0.0, min(float(h), ymax))
    return _order_bbox_xyxy(xmin, ymin, xmax, ymax)


def parse_bbox_pixels_from_model_text(
    text: str,
    *,
    bbox_parse_mode: str | None = None,
    image_width: int,
    image_height: int,
) -> tuple[float, float, float, float] | None:
    """Parse model output as pixel xyxy on the native image (no [0,1] normalization)."""
    if bbox_parse_mode == "cosmos":
        raw = parse_raw_bbox_xyxy(text)
        if raw is None:
            return None
        scaled = bbox_qwen1000_to_normalized_01(raw)
        if scaled is None:
            return None
        w, h = max(1, int(image_width)), max(1, int(image_height))
        xmin, ymin, xmax, ymax = scaled
        return clamp_bbox_xyxy_px(
            (xmin * w, ymin * h, xmax * w, ymax * h),
            image_width=image_width,
            image_height=image_height,
        )
    raw = parse_raw_bbox_xyxy(text)
    if raw is None:
        return None
    if max(abs(v) for v in raw) <= 1.0:
        w, h = max(1, int(image_width)), max(1, int(image_height))
        raw = (raw[0] * w, raw[1] * h, raw[2] * w, raw[3] * h)
    return clamp_bbox_xyxy_px(
        raw, image_width=image_width, image_height=image_height,
    )


def clamp_bbox_xyxy_01(bbox: tuple[float, float, float, float]) -> tuple[float, float, float, float] | None:
    xmin, ymin, xmax, ymax = bbox
    xmin = max(0.0, min(1.0, xmin))
    ymin = max(0.0, min(1.0, ymin))
    xmax = max(0.0, min(1.0, xmax))
    ymax = max(0.0, min(1.0, ymax))
    return _order_bbox_xyxy(xmin, ymin, xmax, ymax)


def bbox_pixels_to_normalized_01(
    bbox: tuple[float, float, float, float],
    *,
    image_width: int,
    image_height: int,
) -> tuple[float, float, float, float] | None:
    m = max(abs(v) for v in bbox)
    if m <= 1.0:
        return clamp_bbox_xyxy_01(bbox)
    w = max(1, int(image_width))
    h = max(1, int(image_height))
    xmin, ymin, xmax, ymax = bbox
    return clamp_bbox_xyxy_01((xmin / w, ymin / h, xmax / w, ymax / h))


def bbox_qwen1000_to_normalized_01(
    bbox: tuple[float, float, float, float],
    *,
    scale: float = QWEN_VL_BBOX_SCALE,
) -> tuple[float, float, float, float] | None:
    m = max(abs(v) for v in bbox)
    if m <= 1.0:
        return clamp_bbox_xyxy_01(bbox)
    s = max(1.0, float(scale))
    return clamp_bbox_xyxy_01(tuple(v / s for v in bbox))


def parse_qwen1000_bbox_to_normalized(text: str) -> tuple[float, float, float, float] | None:
    raw = parse_raw_bbox_xyxy(text)
    if raw is None:
        return None
    return bbox_qwen1000_to_normalized_01(raw)


def parse_bbox_from_model_text(
    text: str,
    *,
    backend: str = "",
    bbox_parse_mode: str | None = None,
    image_width: int | None = None,
    image_height: int | None = None,
) -> tuple[float, float, float, float] | None:
    mode = bbox_parse_mode
    if mode is None:
        mode = "cosmos" if backend == "cosmos" else "prismatic"
    if mode == "cosmos":
        return parse_qwen1000_bbox_to_normalized(text)
    raw = parse_raw_bbox_xyxy(text)
    if raw is None:
        return None
    if image_width and image_height:
        return bbox_pixels_to_normalized_01(
            raw, image_width=image_width, image_height=image_height,
        )
    m = max(abs(v) for v in raw)
    if m <= 1.0:
        return clamp_bbox_xyxy_01(raw)
    return raw


def iou_xyxy(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    aa = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    bb = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    den = aa + bb - inter
    return (inter / den) if den > 0 else 0.0


def _average_precision_from_scores(
    y_true: list[int],
    y_score: list[float],
) -> float:
    if not y_true:
        return 0.0
    pairs = sorted(zip(y_score, y_true), key=lambda x: -x[0])
    tp = fp = 0
    precisions: list[float] = []
    n_pos = sum(y_true)
    for score, label in pairs:
        if label:
            tp += 1
        else:
            fp += 1
        if tp + fp > 0:
            precisions.append(tp / (tp + fp))
    if not precisions or n_pos == 0:
        return 0.0
    out = 0.0
    prev_recall = 0.0
    for i, (_, label) in enumerate(pairs):
        if not label:
            continue
        recall = sum(1 for _, l in pairs[: i + 1] if l) / n_pos
        if recall > prev_recall:
            out += precisions[i] * (recall - prev_recall)
            prev_recall = recall
    return out


def _match_detections_to_gt(
    detections: list[tuple[float, tuple[float, float, float, float]]],
    ground_truths: list[tuple[float, tuple[float, float, float, float]]],
    iou_threshold: float,
) -> list[int]:
    """Greedy IoU matching; returns 0/1 label per detection (sorted by score desc)."""
    dets = sorted(detections, key=lambda x: -x[0])
    gt_used = [False] * len(ground_truths)
    labels: list[int] = []
    for _score, db in dets:
        best_iou = 0.0
        best_j = -1
        for j, (_gscore, gb) in enumerate(ground_truths):
            if gt_used[j]:
                continue
            iou = iou_xyxy(db, gb)
            if iou > best_iou:
                best_iou = iou
                best_j = j
        if best_j >= 0 and best_iou >= iou_threshold:
            gt_used[best_j] = True
            labels.append(1)
        else:
            labels.append(0)
    return labels


def compute_detection_map_metrics(
    records: list[dict[str, Any]],
    *,
    class_key: str = "instrument_id",
    iou_thresholds: list[float] | None = None,
) -> dict[str, Any]:
    """
    COCO-style detection metrics on normalized xyxy boxes.

    Each record needs: class id, image/frame id, gt bbox (norm), pred bbox (norm), score.
    """
    if iou_thresholds is None:
        iou_thresholds = [round(0.5 + 0.05 * i, 2) for i in range(10)]

    by_class: dict[str, list[dict[str, Any]]] = {}
    for rec in records:
        cls = str(rec.get(class_key) or "")
        if not cls:
            continue
        by_class.setdefault(cls, []).append(rec)

    per_class_ap: dict[str, dict[str, float]] = {}
    map_at: dict[str, float | None] = {f"mAP@{int(t * 100)}": None for t in iou_thresholds if abs(t * 100 - round(t * 100)) < 1e-6}
    map_at["mAP@75"] = None
    map_at["mAP@50"] = None

    all_ious: list[float] = []

    for cls, rows in sorted(by_class.items()):
        by_image: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            img = str(row.get("image_id") or row.get("frame_stem") or "")
            by_image.setdefault(img, []).append(row)

        cls_aps: dict[str, float] = {}
        for thr in iou_thresholds:
            y_true_all: list[int] = []
            y_score_all: list[float] = []
            for img_rows in by_image.values():
                dets: list[tuple[float, tuple[float, float, float, float]]] = []
                gts: list[tuple[float, tuple[float, float, float, float]]] = []
                for row in img_rows:
                    gt = row.get("gt_bbox_norm")
                    pred = row.get("pred_bbox_norm")
                    if not (isinstance(gt, (list, tuple)) and len(gt) == 4):
                        continue
                    if not (isinstance(pred, (list, tuple)) and len(pred) == 4):
                        continue
                    gt_t = tuple(float(x) for x in gt)
                    pred_t = tuple(float(x) for x in pred)
                    score = float(row.get("score", 1.0))
                    dets.append((score, pred_t))
                    gts.append((1.0, gt_t))
                    all_ious.append(iou_xyxy(gt_t, pred_t))
                if not dets:
                    continue
                labels = _match_detections_to_gt(dets, gts, thr)
                y_true_all.extend(labels)
                y_score_all.extend([s for s, _ in dets])
            key = f"AP@{int(thr * 100)}" if abs(thr * 100 - round(thr * 100)) < 1e-6 else f"AP@{thr}"
            cls_aps[key] = _average_precision_from_scores(y_true_all, y_score_all)
        per_class_ap[cls] = cls_aps

    def _mean_over_classes(ap_key: str) -> float | None:
        vals = [c.get(ap_key) for c in per_class_ap.values() if ap_key in c]
        vals = [v for v in vals if v is not None]
        return sum(vals) / len(vals) if vals else None

    for thr in iou_thresholds:
        ap_key = f"AP@{int(thr * 100)}" if abs(thr * 100 - round(thr * 100)) < 1e-6 else f"AP@{thr}"
        mean_v = _mean_over_classes(ap_key)
        if abs(thr - 0.5) < 1e-6:
            map_at["mAP@50"] = mean_v
        if abs(thr - 0.75) < 1e-6:
            map_at["mAP@75"] = mean_v

    coco_map_vals = [_mean_over_classes(f"AP@{int(t * 100)}") for t in iou_thresholds]
    coco_map_vals = [v for v in coco_map_vals if v is not None]
    coco_ap = sum(coco_map_vals) / len(coco_map_vals) if coco_map_vals else None

    miou = sum(all_ious) / len(all_ious) if all_ious else None

    return {
        "mIoU": miou,
        "mAP@50": map_at.get("mAP@50"),
        "mAP@75": map_at.get("mAP@75"),
        "COCO_AP": coco_ap,
        "iou_thresholds": iou_thresholds,
        "per_class_ap": per_class_ap,
        "n_scored": len(all_ious),
        "n_classes": len(per_class_ap),
    }


# ── Visualization (bbox overlay) ─────────────────────────────────────────────

def _try_font(size: int) -> ImageFont.ImageFont:
    for name in (
        "DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _multiline_bbox(text: str, font: ImageFont.ImageFont) -> tuple[int, int]:
    lines = text.split("\n")
    xs: list[int] = []
    ys: list[int] = []
    for line in lines:
        if hasattr(font, "getbbox"):
            b = font.getbbox(line)
            xs.append(b[2] - b[0])
            ys.append(b[3] - b[1])
        else:
            xs.append(len(line) * 7)
            ys.append(12)
    return max(xs) if xs else 0, sum(ys) + 2 * max(0, len(lines) - 1)


def english_display(s: str) -> str:
    t = (s or "").strip()
    return t.replace("_", " ") if t else ""


def format_instrument_viz_label(
    instrument_name: str,
    *,
    instrument_id: str | None = None,
) -> str:
    """Instrument display name only (no id / region)."""
    name = (instrument_name or "").strip() or english_display(instrument_id or "")
    return name or "unknown"


def _bbox_top_left_px(
    bbox: tuple[float, float, float, float],
    image_size: tuple[int, int],
) -> tuple[int, int]:
    w, h = image_size
    xmin, ymin, _, _ = bbox
    return int(round(xmin * w)), int(round(ymin * h))


def _draw_text_panel(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    *,
    font: ImageFont.ImageFont,
    fill: str = "white",
    bg: tuple[int, int, int] = (0, 0, 0),
    pad: int = 4,
) -> None:
    x, y = xy
    tw, th = _multiline_bbox(text, font)
    w_img = getattr(draw, "im", None)
    max_x = (w_img.size[0] if w_img is not None else 10_000) - tw - 2 * pad
    max_y = (w_img.size[1] if w_img is not None else 10_000) - th - 2 * pad
    x = max(0, min(x, max_x))
    y = max(0, min(y, max_y))
    draw.rectangle([x, y, x + tw + 2 * pad, y + th + 2 * pad], fill=bg)
    draw.multiline_text((x + pad, y + pad), text, fill=fill, font=font, spacing=2)


def draw_bbox_xyxy_norm(
    image: Image.Image,
    bbox: tuple[float, float, float, float],
    *,
    outline: str,
    line_width: int = 3,
) -> None:
    draw = ImageDraw.Draw(image)
    w, h = image.size
    xmin, ymin, xmax, ymax = bbox
    x0 = int(round(xmin * w))
    y0 = int(round(ymin * h))
    x1 = int(round(xmax * w))
    y1 = int(round(ymax * h))
    draw.rectangle([x0, y0, x1, y1], outline=outline, width=line_width)


def draw_bbox_xyxy_px(
    image: Image.Image,
    bbox: tuple[float, float, float, float],
    *,
    outline: str,
    line_width: int = 3,
) -> None:
    draw = ImageDraw.Draw(image)
    xmin, ymin, xmax, ymax = bbox
    x0 = int(round(xmin))
    y0 = int(round(ymin))
    x1 = int(round(xmax))
    y1 = int(round(ymax))
    draw.rectangle([x0, y0, x1, y1], outline=outline, width=line_width)


def _bbox_top_left_for_draw(
    bbox: tuple[float, float, float, float],
    image_size: tuple[int, int],
    *,
    coordinate_space: str = "norm",
) -> tuple[int, int]:
    if coordinate_space == "pixel":
        xmin, ymin, _, _ = bbox
        return int(round(xmin)), int(round(ymin))
    return _bbox_top_left_px(bbox, image_size)


def draw_gt_pred_comparison_visualization(
    rgb: Image.Image,
    *,
    gt_bbox: tuple[float, float, float, float],
    pred_bbox: tuple[float, float, float, float] | None,
    instrument: str,
    instrument_id: str | None = None,
    iou_value: float | None = None,
    coordinate_space: str = "norm",
) -> Image.Image:
    """GT=green, Pred=red (normalized [0,1] or pixel xyxy per coordinate_space)."""
    out = rgb.copy()
    inst_label = format_instrument_viz_label(instrument, instrument_id=instrument_id)
    draw_fn = draw_bbox_xyxy_px if coordinate_space == "pixel" else draw_bbox_xyxy_norm

    draw_fn(out, gt_bbox, outline="#00ff00", line_width=4)
    if pred_bbox is not None:
        draw_fn(out, pred_bbox, outline="#ff0000", line_width=4)

    draw = ImageDraw.Draw(out)
    small = _try_font(13)
    box_font = _try_font(12)
    img_sz = out.size

    gx, gy = _bbox_top_left_for_draw(
        gt_bbox, img_sz, coordinate_space=coordinate_space,
    )
    _draw_text_panel(
        draw,
        (gx, max(0, gy - 28)),
        inst_label,
        font=box_font,
        bg=(0, 80, 0),
    )
    if pred_bbox is not None:
        px, py = _bbox_top_left_for_draw(
            pred_bbox, img_sz, coordinate_space=coordinate_space,
        )
        _draw_text_panel(
            draw,
            (px, max(0, py - 28)),
            inst_label,
            font=box_font,
            bg=(80, 0, 0),
        )

    iou_text = f"{iou_value:.4f}" if iou_value is not None else "n/a"
    text = f"{inst_label}\nGT: green  Pred: red\nIoU: {iou_text}"
    _draw_text_panel(draw, (6, 6), text, font=small)
    return out


def normalize_label_key(term: str) -> str:
    """Canonical key for matching triplet labels (spaces vs hyphens)."""
    return re.sub(r"[\s\-_]+", "-", (term or "").strip().lower())


OTHERS_LABEL_KEY = "others"


def label_classes_with_others(classes: list[str]) -> list[str]:
    """Vocabulary labels plus synthetic ``others`` (GT always 0)."""
    out: list[str] = []
    seen: set[str] = set()
    for cls in classes:
        key = normalize_label_key(cls)
        if key not in seen:
            seen.add(key)
            out.append(cls)
    if OTHERS_LABEL_KEY not in seen:
        out.append("others")
    return out


def build_pred_label_scores(
    pred_tokens: list[str],
    classes: list[str],
) -> tuple[dict[str, float], list[str], list[str]]:
    """
    Hard 0/1 scores from parsed answer tokens.

    - In-vocab token in answer → that label = 1
    - In-vocab label not in answer → 0
    - Any out-of-vocab token → ``others`` = 1 (GT for others is always 0)
    """
    class_keys = [normalize_label_key(c) for c in classes]
    key_to_display = {normalize_label_key(c): c for c in classes}
    scores = {k: 0.0 for k in class_keys}
    in_vocab: list[str] = []
    oov: list[str] = []
    vocab_set = set(class_keys)

    for tok in pred_tokens:
        if tok in vocab_set:
            scores[tok] = 1.0
            in_vocab.append(key_to_display.get(tok, tok))
        else:
            oov.append(tok)

    scores[OTHERS_LABEL_KEY] = 1.0 if oov else 0.0
    return scores, in_vocab, oov


def build_gold_label_scores(gold_set: set[str], classes: list[str]) -> dict[str, float]:
    """Gold 0/1 vector; ``others`` is always 0."""
    class_keys = [normalize_label_key(c) for c in classes]
    scores = {k: (1.0 if k in gold_set else 0.0) for k in class_keys}
    scores[OTHERS_LABEL_KEY] = 0.0
    return scores


def roc_auc_binary(
    y_true: list[int],
    y_score: list[float],
    *,
    undefined_fallback: float = 0.5,
) -> float:
    """ROC-AUC via Mann–Whitney U; undefined cases use ``undefined_fallback``."""
    pos = [s for s, t in zip(y_score, y_true, strict=True) if t == 1]
    neg = [s for s, t in zip(y_score, y_true, strict=True) if t == 0]
    if not pos or not neg:
        return undefined_fallback
    wins = 0.0
    for ps in pos:
        for ns in neg:
            if ps > ns:
                wins += 1.0
            elif ps == ns:
                wins += 0.5
    return wins / (len(pos) * len(neg))


def average_precision_binary(
    y_true: list[int],
    y_score: list[float],
    *,
    no_positive_fallback: float = 0.0,
    all_positive_fallback: float = 1.0,
) -> float:
    """Average precision for binary labels (score-sorted).

    Ties on ``y_score`` keep input order (stable sort). Positives are not
    ranked above negatives when scores are equal.
    """
    if not y_true:
        return no_positive_fallback
    n_pos = sum(y_true)
    if n_pos == 0:
        return no_positive_fallback
    if n_pos == len(y_true):
        return all_positive_fallback

    pairs = sorted(zip(y_score, y_true), key=lambda x: -x[0])
    tp = 0
    fp = 0
    precisions: list[float] = []
    for _score, yt in pairs:
        if yt == 1:
            tp += 1
            precisions.append(tp / (tp + fp))
        else:
            fp += 1
    return sum(precisions) / n_pos


def _macro_multilabel_classification_metrics(
    gold_sets: list[set[str]],
    pred_scores: list[dict[str, float]],
    classes: list[str],
    *,
    include_others: bool = True,
    undefined_auc_fallback: float = 0.5,
) -> dict[str, Any]:
    """Macro AUROC and macro mAP over ``classes`` (+ optional ``others``)."""
    if not gold_sets:
        return {
            "macro_auroc": None,
            "macro_map": None,
            "per_label": {},
            "n_labels_scored": 0,
            "n_labels_total": 0,
        }

    eval_classes = label_classes_with_others(classes) if include_others else list(classes)
    per_label: dict[str, dict[str, float]] = {}
    aucs: list[float] = []
    aps: list[float] = []

    for cls in eval_classes:
        ckey = normalize_label_key(cls)
        y_true = [1 if ckey in gs else 0 for gs in gold_sets]
        if ckey == OTHERS_LABEL_KEY:
            y_true = [0] * len(gold_sets)
        y_score = [float(ps.get(ckey, 0.0)) for ps in pred_scores]

        auc = roc_auc_binary(y_true, y_score, undefined_fallback=undefined_auc_fallback)
        ap = average_precision_binary(y_true, y_score)
        per_label[cls] = {"auroc": auc, "ap": ap}
        aucs.append(auc)
        aps.append(ap)

    return {
        "macro_auroc": sum(aucs) / len(aucs) if aucs else None,
        "macro_map": sum(aps) / len(aps) if aps else None,
        "per_label": per_label,
        "n_labels_scored": len(aucs),
        "n_labels_total": len(eval_classes),
        "includes_others_class": include_others,
    }


def mean_auroc_multilabel(
    gold_sets: list[set[str]],
    pred_scores: list[dict[str, float]],
    classes: list[str],
) -> dict[str, Any]:
    """Macro-averaged AUROC (always defined via per-label fallback + ``others``)."""
    block = _macro_multilabel_classification_metrics(
        gold_sets, pred_scores, classes, include_others=True,
    )
    per_label = {
        cls: metrics.get("auroc")
        for cls, metrics in block.get("per_label", {}).items()
    }
    return {
        "macro_auroc": block.get("macro_auroc"),
        "per_label": per_label,
        "n_labels_scored": block.get("n_labels_scored"),
        "n_labels_total": block.get("n_labels_total"),
    }


def mean_map_multilabel(
    gold_sets: list[set[str]],
    pred_scores: list[dict[str, float]],
    classes: list[str],
) -> dict[str, Any]:
    """Macro-averaged mAP (``others`` GT always 0)."""
    block = _macro_multilabel_classification_metrics(
        gold_sets, pred_scores, classes, include_others=True,
    )
    per_label = {
        cls: metrics.get("ap")
        for cls, metrics in block.get("per_label", {}).items()
    }
    return {
        "macro_map": block.get("macro_map"),
        "per_label": per_label,
        "n_labels_scored": block.get("n_labels_scored"),
        "n_labels_total": block.get("n_labels_total"),
    }


def multilabel_classification_metrics(
    gold_sets: list[set[str]],
    pred_scores: list[dict[str, float]],
    classes: list[str],
) -> dict[str, Any]:
    """Combined macro AUROC + macro mAP block."""
    return _macro_multilabel_classification_metrics(
        gold_sets, pred_scores, classes, include_others=True,
    )


def draw_single_bbox_visualization(
    rgb: Image.Image,
    bbox: tuple[float, float, float, float],
    *,
    instrument: str,
    instrument_id: str | None = None,
    outline: str,
    panel_bg: tuple[int, int, int] = (0, 0, 0),
    coordinate_space: str = "norm",
) -> Image.Image:
    out = rgb.copy()
    draw_fn = draw_bbox_xyxy_px if coordinate_space == "pixel" else draw_bbox_xyxy_norm
    draw_fn(out, bbox, outline=outline, line_width=4)
    draw = ImageDraw.Draw(out)
    small = _try_font(13)
    box_font = _try_font(12)
    inst_label = format_instrument_viz_label(instrument, instrument_id=instrument_id)

    bx, by = _bbox_top_left_for_draw(
        bbox, out.size, coordinate_space=coordinate_space,
    )
    _draw_text_panel(
        draw,
        (bx, max(0, by - 28)),
        inst_label,
        font=box_font,
        bg=panel_bg,
    )
    _draw_text_panel(draw, (6, 6), inst_label, font=small, bg=panel_bg)
    return out
