"""EndoVis 2018 VQA — Classification QA samples and image paths."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Literal

_PKG_ROOT = Path(__file__).resolve().parent
REPO_ROOT = _PKG_ROOT.parent
DEFAULT_VQA_ROOT = REPO_ROOT / "eval" / "EndoVis-18-VQA"
DEFAULT_IMAGES_ROOT = REPO_ROOT / "eval" / "endovis2018"

ORGAN_QUESTION = "What organ is being operated?"
ORGAN_OPTIONS: tuple[str, ...] = ("kidney",)

STATE_OPTIONS: tuple[str, ...] = (
    "Idle",
    "Looping",
    "Grasping",
    "Retraction",
    "Tissue_Manipulation",
    "Tool_Manipulation",
    "Suturing",
    "Clipping",
    "Cutting",
    "Cauterization",
    "Ultrasound_Sensing",
    "Suction",
    "Staple",
)

LOCATION_OPTIONS: tuple[str, ...] = (
    "left-top",
    "right-top",
    "left-bottom",
    "right-bottom",
)

GLOBAL_ANSWER_KEYWORDS: tuple[str, ...] = tuple(
    dict.fromkeys([*ORGAN_OPTIONS, *STATE_OPTIONS, *LOCATION_OPTIONS])
)

_STATE_Q_RE = re.compile(r"^What is the state of (.+)\?$", re.IGNORECASE)
_LOC_Q_RE = re.compile(r"^Where is (.+) located\?$", re.IGNORECASE)
_FRAME_Q_RE = re.compile(r"frame(\d+)_QA\.txt$", re.IGNORECASE)

ImageSplit = Literal["val", "train", "both"]


def question_template(question: str) -> str:
    q = (question or "").strip()
    if _STATE_Q_RE.match(q):
        return "What is the state of {instrument}?"
    if _LOC_Q_RE.match(q):
        return "Where is {instrument} located?"
    return q


def options_for_question(question: str) -> list[str]:
    q = (question or "").strip()
    if q == ORGAN_QUESTION:
        return list(ORGAN_OPTIONS)
    if _STATE_Q_RE.match(q):
        return list(STATE_OPTIONS)
    if _LOC_Q_RE.match(q):
        return list(LOCATION_OPTIONS)
    raise ValueError(f"Unsupported classification question: {q!r}")


def question_type(question: str) -> str:
    q = (question or "").strip()
    if q == ORGAN_QUESTION:
        return "organ"
    if _STATE_Q_RE.match(q):
        return "state"
    if _LOC_Q_RE.match(q):
        return "location"
    return "other"


def resolve_frame_image(
    seq_num: str,
    frame_index: int,
    *,
    images_root: Path,
    image_split: ImageSplit,
) -> Path | None:
    """Map VQA frame index to ``seq_{n}_frame{idx}.bmp`` under train/val image dirs."""
    rel_names = (
        f"seq_{seq_num}_frame{frame_index:03d}.bmp",
        f"seq_{seq_num}_frame{frame_index}.bmp",
    )
    splits: tuple[str, ...]
    if image_split == "both":
        splits = ("val", "train")
    else:
        splits = (image_split,)

    for split in splits:
        base = images_root / split / "image"
        for name in rel_names:
            path = base / name
            if path.is_file():
                return path.resolve()
    return None


def _parse_qa_file(path: Path) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or "|" not in line:
            continue
        q, a = line.split("|", 1)
        q, a = q.strip(), a.strip()
        if q and a:
            pairs.append((q, a))
    return pairs


def collect_classification_samples(
    vqa_root: Path,
    *,
    images_root: Path,
    image_split: ImageSplit = "val",
    seq_filter: str | None = None,
) -> list[dict[str, Any]]:
    """One sample per ``question|answer`` line in Classification QA files."""
    vqa_root = vqa_root.resolve()
    images_root = images_root.resolve()
    samples: list[dict[str, Any]] = []

    seq_dirs = sorted(vqa_root.glob("seq_*"))
    if seq_filter is not None:
        key = seq_filter.strip().lower().replace("seq_", "").replace("seq", "")
        seq_dirs = [d for d in seq_dirs if d.name.replace("seq_", "") == key]

    for seq_dir in seq_dirs:
        seq_num = seq_dir.name.replace("seq_", "")
        class_dir = seq_dir / "vqa" / "Classification"
        if not class_dir.is_dir():
            continue
        for qa_path in sorted(class_dir.glob("frame*_QA.txt")):
            m = _FRAME_Q_RE.search(qa_path.name)
            if not m:
                continue
            frame_index = int(m.group(1))
            img_path = resolve_frame_image(
                seq_num,
                frame_index,
                images_root=images_root,
                image_split=image_split,
            )
            if img_path is None:
                continue
            for q_idx, (question, gold_keyword) in enumerate(_parse_qa_file(qa_path)):
                options = options_for_question(question)
                if gold_keyword not in options:
                    raise ValueError(
                        f"Gold {gold_keyword!r} not in options for {question!r} ({qa_path})"
                    )
                samples.append({
                    "seq": seq_num,
                    "seq_dir": seq_dir.name,
                    "frame_index": frame_index,
                    "qa_file": str(qa_path.resolve()),
                    "question_index": q_idx,
                    "question": question,
                    "question_template": question_template(question),
                    "question_type": question_type(question),
                    "gold_keyword": gold_keyword,
                    "options": options,
                    "img_path": img_path,
                })
    return samples
