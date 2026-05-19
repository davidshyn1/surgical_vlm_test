"""Endoscapes2023 — Critical View of Safety (CVS) samples from COCO JSON."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

_PKG_ROOT = Path(__file__).resolve().parent
REPO_ROOT = _PKG_ROOT.parent
DEFAULT_DATASET_ROOT = REPO_ROOT / "eval" / "endoscapes"

YES_NO_OPTIONS: tuple[str, ...] = ("yes", "no")

EvalProtocol = Literal["joint", "per_criterion"]
SplitName = Literal[
    "train",
    "val",
    "test",
    "train_seg",
    "val_seg",
    "test_seg",
]


@dataclass(frozen=True)
class CvsCriterion:
    index: int
    criterion_id: str
    short_name: str
    question: str


# ``ds[0]``, ``ds[1]``, ``ds[2]`` in Endoscapes COCO map to C1, C2, C3.
CVS_CRITERIA: tuple[CvsCriterion, ...] = (
    CvsCriterion(
        0,
        "C1",
        "two_structures",
        "Only two tubular structures connect to the gallbladder.",
    ),
    CvsCriterion(
        1,
        "C2",
        "triangle_cleared",
        "Hepatocystic triangle cleared for visibility.",
    ),
    CvsCriterion(
        2,
        "C3",
        "lower_gallbladder_detached",
        "Lower gallbladder detached from liver bed.",
    ),
)


def binarize_ds(ds: list[float] | tuple[float, ...], *, threshold: float = 0.5) -> list[int]:
    return [1 if float(x) >= threshold else 0 for x in ds]


def resolve_annotation_path(split_dir: Path, annotation_file: str | None) -> Path:
    if annotation_file:
        path = split_dir / annotation_file
        if not path.is_file():
            raise FileNotFoundError(f"Annotation file not found: {path}")
        return path
    for name in ("annotation_coco.json", "annotation_ds_coco.json"):
        path = split_dir / name
        if path.is_file():
            return path
    raise FileNotFoundError(
        f"No annotation_coco.json or annotation_ds_coco.json under {split_dir}"
    )


def load_frame_records(
    dataset_root: Path,
    split: SplitName,
    *,
    annotation_file: str | None = None,
    gt_threshold: float = 0.5,
    video_filter: int | None = None,
) -> list[dict[str, Any]]:
    """One record per image that has ``ds`` (length 3) and a readable JPEG."""
    dataset_root = dataset_root.resolve()
    split_dir = dataset_root / split
    if not split_dir.is_dir():
        raise FileNotFoundError(f"Split directory not found: {split_dir}")

    ann_path = resolve_annotation_path(split_dir, annotation_file)
    coco = json.loads(ann_path.read_text(encoding="utf-8"))
    frames: list[dict[str, Any]] = []

    for im in coco.get("images") or []:
        ds = im.get("ds")
        if not isinstance(ds, (list, tuple)) or len(ds) != 3:
            continue
        vid = im.get("video_id")
        if video_filter is not None and int(vid) != int(video_filter):
            continue
        file_name = str(im.get("file_name") or "")
        img_path = split_dir / file_name
        if not img_path.is_file():
            continue
        ds_raw = [float(x) for x in ds]
        frames.append({
            "split": split,
            "annotation_path": str(ann_path.resolve()),
            "image_id": im.get("id"),
            "file_name": file_name,
            "img_path": img_path.resolve(),
            "video_id": vid,
            "is_ds_keyframe": im.get("is_ds_keyframe"),
            "is_det_keyframe": im.get("is_det_keyframe"),
            "ds_raw": ds_raw,
            "ds_binary": binarize_ds(ds_raw, threshold=gt_threshold),
        })
    return frames


def expand_samples_for_protocol(
    frames: list[dict[str, Any]],
    *,
    eval_protocol: EvalProtocol,
) -> list[dict[str, Any]]:
    if eval_protocol == "joint":
        return [dict(f) for f in frames]

    samples: list[dict[str, Any]] = []
    for frame in frames:
        for crit in CVS_CRITERIA:
            samples.append({
                **frame,
                "criterion_index": crit.index,
                "criterion_id": crit.criterion_id,
                "criterion_key": crit.short_name,
                "criterion_question": crit.question,
                "gold_binary": int(frame["ds_binary"][crit.index]),
                "gold_label": "yes" if frame["ds_binary"][crit.index] else "no",
            })
    return samples


def collect_cvs_samples(
    dataset_root: Path,
    split: SplitName,
    *,
    eval_protocol: EvalProtocol = "joint",
    annotation_file: str | None = None,
    gt_threshold: float = 0.5,
    video_filter: int | None = None,
) -> list[dict[str, Any]]:
    frames = load_frame_records(
        dataset_root,
        split,
        annotation_file=annotation_file,
        gt_threshold=gt_threshold,
        video_filter=video_filter,
    )
    return expand_samples_for_protocol(frames, eval_protocol=eval_protocol)
