"""EndoVis-17-VQLA instrument localization data loading."""

from __future__ import annotations

import random
import re
from pathlib import Path
from typing import Any

_PKG_ROOT = Path(__file__).resolve().parent
REPO_ROOT = _PKG_ROOT.parent
DEFAULT_DATASET_ROOT = REPO_ROOT / "eval" / "EndoVis-17-VQLA"
DEFAULT_FRAMES_ROOT = DEFAULT_DATASET_ROOT / "left_frames"
DEFAULT_ANNOTATIONS_ROOT = DEFAULT_DATASET_ROOT / "vqla"

ENDOVIS17_IMAGE_WIDTH = 1280
ENDOVIS17_IMAGE_HEIGHT = 1024

INSTRUMENT_IDS = (
    "bipolar_forceps",
    "large_needle_driver",
    "monopolar_curved_scissors",
    "prograsp_forceps",
    "ultrasound_probe",
)

INSTRUMENT_DISPLAY: dict[str, str] = {
    "bipolar_forceps": "Bipolar Forceps",
    "large_needle_driver": "Large Needle Driver",
    "monopolar_curved_scissors": "Monopolar Curved Scissors",
    "prograsp_forceps": "Prograsp Forceps",
    "ultrasound_probe": "Ultrasound Probe",
}

REGION_DISPLAY: dict[str, str] = {
    "left-top": "left top",
    "left-bottom": "left bottom",
    "right-top": "right top",
    "right-bottom": "right bottom",
}

_WHERE_RE = re.compile(r"^Where is (.+) located\?$")


def instrument_display_name(instrument_id: str) -> str:
    key = (instrument_id or "").strip().lower()
    return INSTRUMENT_DISPLAY.get(key, key.replace("_", " ").title())


def region_display_name(region_id: str) -> str:
    key = (region_id or "").strip().lower()
    return REGION_DISPLAY.get(key, key.replace("-", " "))


def parse_pixel_bbox_str(bbox_str: str) -> tuple[float, float, float, float] | None:
    parts = [p.strip() for p in (bbox_str or "").split(",")]
    if len(parts) != 4:
        return None
    try:
        xmin, ymin, xmax, ymax = (float(p) for p in parts)
    except ValueError:
        return None
    if xmax < xmin:
        xmin, xmax = xmax, xmin
    if ymax < ymin:
        ymin, ymax = ymax, ymin
    if xmax <= xmin or ymax <= ymin:
        return None
    return (xmin, ymin, xmax, ymax)


def bbox_pixels_to_normalized(
    bbox: tuple[float, float, float, float],
    *,
    image_width: int = ENDOVIS17_IMAGE_WIDTH,
    image_height: int = ENDOVIS17_IMAGE_HEIGHT,
) -> tuple[float, float, float, float]:
    w = max(1, int(image_width))
    h = max(1, int(image_height))
    xmin, ymin, xmax, ymax = bbox
    return (xmin / w, ymin / h, xmax / w, ymax / h)


def build_instrument_localization_prompt(
    *,
    instrument_id: str,
    region_id: str | None = None,
) -> str:
    """Per-sample instrument name is substituted (e.g. Large Needle Driver)."""
    inst = instrument_display_name(instrument_id)
    _ = region_id  # label metadata only; dataset question is instrument-only
    return (
        f"Where is the {inst} located? Answer the question with just a bounding box.\n"
        "Format: [x_min, y_min, x_max, y_max]\n"
        "Use normalized coordinates in [0, 1] relative to the image you see.\n"
        f"If the {inst} is not in the image, answer exactly: not present"
    )


def collect_localization_samples(
    *,
    annotations_root: Path,
    frames_root: Path,
    instrument_filter: str | None = None,
    region_filter: str | None = None,
    frame_stem_filter: str | None = None,
) -> list[dict[str, Any]]:
    ann_root = annotations_root.resolve()
    img_root = frames_root.resolve()
    inst_filter = (instrument_filter or "").strip().lower() or None
    region_f = (region_filter or "").strip().lower() or None
    stem_f = (frame_stem_filter or "").strip() or None

    samples: list[dict[str, Any]] = []
    for ann_path in sorted(ann_root.glob("*.txt")):
        stem = ann_path.stem
        if stem_f and stem != stem_f:
            continue
        img_path = img_root / f"{stem}.jpg"
        if not img_path.is_file():
            for ext in (".png", ".jpeg", ".webp"):
                alt = img_root / f"{stem}{ext}"
                if alt.is_file():
                    img_path = alt
                    break
        if not img_path.is_file():
            continue

        for line_no, line in enumerate(ann_path.read_text(encoding="utf-8").splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            parts = line.split("|")
            if len(parts) < 3:
                continue
            question, region_id, bbox_str = parts[0], parts[1].strip(), parts[2].strip()
            m = _WHERE_RE.match(question.strip())
            if not m:
                continue
            instrument_id = m.group(1).strip().lower()
            region_id = region_id.strip().lower()
            if inst_filter and instrument_id != inst_filter:
                continue
            if region_f and region_id != region_f:
                continue
            bbox_px = parse_pixel_bbox_str(bbox_str)
            if bbox_px is None:
                continue
            bbox_norm = bbox_pixels_to_normalized(bbox_px)
            samples.append(
                {
                    "frame_stem": stem,
                    "line_no": line_no,
                    "img_path": img_path.resolve(),
                    "instrument_id": instrument_id,
                    "instrument_display": instrument_display_name(instrument_id),
                    "region_id": region_id,
                    "region_display": region_display_name(region_id),
                    "bbox_xyxy_px": list(bbox_px),
                    "bbox_xyxy_norm": list(bbox_norm),
                    "image_width": ENDOVIS17_IMAGE_WIDTH,
                    "image_height": ENDOVIS17_IMAGE_HEIGHT,
                    "dataset_question": question,
                }
            )
    return samples


def sample_localization_items(
    items: list[dict[str, Any]],
    *,
    cap: int | None,
    seed: int,
) -> list[dict[str, Any]]:
    if cap is None or cap <= 0 or len(items) <= cap:
        return list(items)
    rng = random.Random(seed)
    pool = list(items)
    rng.shuffle(pool)
    return sorted(pool[:cap], key=lambda s: (s["frame_stem"], s["line_no"]))
