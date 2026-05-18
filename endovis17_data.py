"""EndoVis 2017 instrument localization — mask labels to bbox samples."""

from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

_PKG_ROOT = Path(__file__).resolve().parent
REPO_ROOT = _PKG_ROOT.parent
DEFAULT_DATASET_ROOT = REPO_ROOT / "eval" / "endovis2017"
MAPPING_FILENAME = "instrument_type_mapping.json"

ENDOVIS2017_IMAGE_SIZE = 512
ACTIVE_CLASS_IDS = (1, 2, 3, 4, 6)  # present in val masks; 5,7 unused

INSTRUMENT_DISPLAY: dict[str, str] = {
    "bipolar_forceps": "Bipolar Forceps",
    "prograsp_forceps": "Prograsp Forceps",
    "large_needle_driver": "Large Needle Driver",
    "vessel_sealer": "Vessel Sealer",
    "monopolar_curved_scissors": "Monopolar Curved Scissors",
}

INSTRUMENT_IDS = tuple(INSTRUMENT_DISPLAY.keys())


def display_to_slug(display_name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", (display_name or "").strip().lower()).strip("_")


def load_class_mapping(dataset_root: Path) -> tuple[dict[int, str], dict[int, str]]:
    """
    Returns:
      class_id -> instrument_id (slug)
      class_id -> instrument display name
    """
    path = dataset_root / MAPPING_FILENAME
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    id_to_slug: dict[int, str] = {}
    id_to_display: dict[int, str] = {}
    for display_name, class_id in raw.items():
        cid = int(class_id)
        if cid <= 0:
            continue
        slug = display_to_slug(display_name)
        id_to_slug[cid] = slug
        id_to_display[cid] = display_name.strip()
    return id_to_slug, id_to_display


def instrument_display_name(instrument_id: str) -> str:
    key = (instrument_id or "").strip().lower()
    return INSTRUMENT_DISPLAY.get(key, key.replace("_", " ").title())


def bbox_pixels_to_normalized(
    bbox: tuple[float, float, float, float],
    *,
    image_width: int,
    image_height: int,
) -> tuple[float, float, float, float]:
    w = max(1, int(image_width))
    h = max(1, int(image_height))
    xmin, ymin, xmax, ymax = bbox
    return (xmin / w, ymin / h, xmax / w, ymax / h)


def bbox_from_class_mask(
    mask: np.ndarray,
    class_id: int,
    *,
    min_pixels: int = 1,
) -> tuple[float, float, float, float] | None:
    """Tight axis-aligned bbox (inclusive pixel xyxy) for one semantic class."""
    ys, xs = np.where(mask == class_id)
    if xs.size < min_pixels:
        return None
    xmin = float(xs.min())
    xmax = float(xs.max())
    ymin = float(ys.min())
    ymax = float(ys.max())
    if xmax <= xmin or ymax <= ymin:
        return None
    return (xmin, ymin, xmax, ymax)


def extract_instrument_boxes_from_mask(
    mask_path: Path,
    *,
    class_id_to_slug: dict[int, str],
    class_id_to_display: dict[int, str],
    min_pixels: int = 1,
    active_class_ids: tuple[int, ...] = ACTIVE_CLASS_IDS,
) -> list[dict[str, Any]]:
    mask_img = Image.open(mask_path)
    mask = np.asarray(mask_img)
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    h, w = mask.shape[:2]
    present = {int(v) for v in np.unique(mask).tolist() if int(v) != 0}
    boxes: list[dict[str, Any]] = []
    for cid in sorted(present):
        if cid not in active_class_ids:
            continue
        bbox_px = bbox_from_class_mask(mask, cid, min_pixels=min_pixels)
        if bbox_px is None:
            continue
        slug = class_id_to_slug.get(cid)
        if not slug:
            continue
        display = class_id_to_display.get(cid, instrument_display_name(slug))
        bbox_norm = bbox_pixels_to_normalized(bbox_px, image_width=w, image_height=h)
        boxes.append(
            {
                "mask_class_id": cid,
                "instrument_id": slug,
                "instrument_display": display,
                "bbox_xyxy_px": [float(v) for v in bbox_px],
                "bbox_xyxy_norm": [float(v) for v in bbox_norm],
                "mask_pixel_count": int((mask == cid).sum()),
            }
        )
    return boxes


def build_instrument_localization_prompt(*, instrument_id: str) -> str:
    inst = instrument_display_name(instrument_id)
    return (
        f"Where is the {inst} located? Answer the question with just a bounding box.\n"
        "Format: [x_min, y_min, x_max, y_max]\n"
        "Use normalized coordinates in [0, 1] relative to the image you see.\n"
        f"If the {inst} is not in the image, answer exactly: not present"
    )


def list_val_splits(dataset_root: Path) -> list[str]:
    return sorted(
        p.name
        for p in dataset_root.iterdir()
        if p.is_dir() and re.fullmatch(r"val\d+", p.name)
    )


def collect_localization_samples(
    *,
    dataset_root: Path,
    val_splits: list[str] | None = None,
    instrument_filter: str | None = None,
    frame_stem_filter: str | None = None,
    min_mask_pixels: int = 1,
) -> list[dict[str, Any]]:
    root = dataset_root.resolve()
    class_id_to_slug, class_id_to_display = load_class_mapping(root)
    inst_filter = (instrument_filter or "").strip().lower() or None
    stem_f = (frame_stem_filter or "").strip() or None
    splits = val_splits or list_val_splits(root)
    if not splits:
        raise FileNotFoundError(f"No val* folders under {root}")

    samples: list[dict[str, Any]] = []
    for split_name in splits:
        split_dir = root / split_name
        image_dir = split_dir / "image"
        label_dir = split_dir / "label"
        if not image_dir.is_dir() or not label_dir.is_dir():
            continue
        for label_path in sorted(label_dir.glob("*.bmp")):
            stem = label_path.stem
            if stem_f and stem != stem_f:
                continue
            image_path = image_dir / label_path.name
            if not image_path.is_file():
                continue
            img = Image.open(image_path)
            w, h = img.size

            boxes = extract_instrument_boxes_from_mask(
                label_path,
                class_id_to_slug=class_id_to_slug,
                class_id_to_display=class_id_to_display,
                min_pixels=min_mask_pixels,
            )
            for box in boxes:
                if inst_filter and box["instrument_id"] != inst_filter:
                    continue
                samples.append(
                    {
                        "val_split": split_name,
                        "frame_stem": stem,
                        "line_no": int(box["mask_class_id"]),
                        "img_path": str(image_path.resolve()),
                        "label_mask_path": str(label_path.resolve()),
                        "instrument_id": box["instrument_id"],
                        "instrument_display": box["instrument_display"],
                        "mask_class_id": box["mask_class_id"],
                        "bbox_xyxy_px": box["bbox_xyxy_px"],
                        "bbox_xyxy_norm": box["bbox_xyxy_norm"],
                        "mask_pixel_count": box["mask_pixel_count"],
                        "image_width": int(w),
                        "image_height": int(h),
                        "region_id": "",
                        "region_display": "",
                    }
                )
    return samples


def export_bbox_annotations(
    samples: list[dict[str, Any]],
    out_path: Path,
    *,
    dataset_root: Path,
) -> None:
    """Write derived bbox annotations JSON for inspection / reuse."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset": "endovis2017",
        "dataset_root": str(dataset_root.resolve()),
        "source": "mask-derived tight bbox per instrument class per frame",
        "bbox_format": "xyxy pixel inclusive + xyxy normalized [0,1]",
        "count": len(samples),
        "samples": samples,
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


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
    return sorted(
        pool[:cap],
        key=lambda s: (s["val_split"], s["frame_stem"], s["instrument_id"]),
    )
