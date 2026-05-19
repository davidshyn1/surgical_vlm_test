"""EndoVis 2018 VQA — Classification QA samples and image paths.

Per frame, samples are emitted in this order:

  Q1 — organ: ``What organ is being operated?``
  Q2… — state: ``What is the state of {instrument}?`` (one per instrument in Classification)
  Q… — location: ``Where is {instrument} located?`` (one per instrument in Classification)
  Q5 — tools: ``What tools are operating the organ?``
         Gold = instruments labeled in the frame's Classification file (state/location lines).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Literal

_PKG_ROOT = Path(__file__).resolve().parent
REPO_ROOT = _PKG_ROOT.parent
DEFAULT_VQA_ROOT = REPO_ROOT / "eval" / "EndoVis-18-VQA"
DEFAULT_IMAGES_ROOT = REPO_ROOT / "eval" / "endovis2018"

ORGAN_QUESTION = "What organ is being operated?"
TOOLS_QUESTION = "What tools are operating the organ?"

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

# Instruments observed in EndoVis-18-VQA Classification labels (used as Q5 MCQ options).
INSTRUMENT_OPTIONS: tuple[str, ...] = (
    "bipolar_forceps",
    "monopolar_curved_scissors",
    "prograsp_forceps",
    "suction",
    "ultrasound_probe",
    "stapler",
    "large_needle_driver",
    "clip_applier",
)

GLOBAL_ANSWER_KEYWORDS: tuple[str, ...] = tuple(
    dict.fromkeys([*ORGAN_OPTIONS, *STATE_OPTIONS, *LOCATION_OPTIONS, *INSTRUMENT_OPTIONS])
)

_STATE_Q_RE = re.compile(r"^What is the state of (.+)\?$", re.IGNORECASE)
_LOC_Q_RE = re.compile(r"^Where is (.+) located\?$", re.IGNORECASE)
_FRAME_Q_RE = re.compile(r"frame(\d+)_QA\.txt$", re.IGNORECASE)

QuestionType = Literal["organ", "state", "location", "tools", "other"]
ImageSplit = Literal["val", "train", "both"]


def canonical_instrument(name: str) -> str:
    """Normalize instrument names to underscore form (``bipolar_forceps``)."""
    s = (name or "").strip().lower()
    s = s.replace("-", "_").replace(" ", "_")
    s = re.sub(r"_+", "_", s)
    return s


def question_template(question: str) -> str:
    q = (question or "").strip()
    if _STATE_Q_RE.match(q):
        return "What is the state of {instrument}?"
    if _LOC_Q_RE.match(q):
        return "Where is {instrument} located?"
    if q.lower() == TOOLS_QUESTION.lower():
        return TOOLS_QUESTION
    return q


def instrument_from_question(question: str) -> str | None:
    q = (question or "").strip()
    m = _STATE_Q_RE.match(q)
    if m:
        return canonical_instrument(m.group(1))
    m = _LOC_Q_RE.match(q)
    if m:
        return canonical_instrument(m.group(1))
    return None


def options_for_question(question: str, *, instrument_options: list[str] | None = None) -> list[str]:
    q = (question or "").strip()
    if q.lower() == ORGAN_QUESTION.lower():
        return list(ORGAN_OPTIONS)
    if q.lower() == TOOLS_QUESTION.lower():
        opts = instrument_options if instrument_options is not None else list(INSTRUMENT_OPTIONS)
        return list(opts)
    if _STATE_Q_RE.match(q):
        return list(STATE_OPTIONS)
    if _LOC_Q_RE.match(q):
        return list(LOCATION_OPTIONS)
    raise ValueError(f"Unsupported classification question: {q!r}")


def question_type(question: str) -> QuestionType:
    q = (question or "").strip()
    if q.lower() == ORGAN_QUESTION.lower():
        return "organ"
    if q.lower() == TOOLS_QUESTION.lower():
        return "tools"
    if _STATE_Q_RE.match(q):
        return "state"
    if _LOC_Q_RE.match(q):
        return "location"
    return "other"


def discover_instruments(vqa_root: Path) -> tuple[str, ...]:
    """Collect instrument names from all Classification QA files under *vqa_root*."""
    found: set[str] = set(INSTRUMENT_OPTIONS)
    for qa_path in vqa_root.glob("seq_*/vqa/Classification/frame*_QA.txt"):
        for _q, _a, inst, qtype in _iter_classification_rows(qa_path):
            if qtype in ("state", "location") and inst:
                found.add(inst)
    return tuple(sorted(found))


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


def _normalize_organ_question(question: str) -> str:
    if question.strip().lower() == ORGAN_QUESTION.lower():
        return ORGAN_QUESTION
    return question.strip()


def _iter_classification_rows(
    qa_path: Path,
) -> list[tuple[str, str, str | None, QuestionType]]:
    """Parse one Classification file into (question, answer, instrument, type) rows."""
    rows: list[tuple[str, str, str | None, QuestionType]] = []
    for question, answer in _parse_qa_file(qa_path):
        q = _normalize_organ_question(question)
        qtype = question_type(q)
        inst = instrument_from_question(q)
        rows.append((q, answer, inst, qtype))
    return rows


def parse_classification_frame(qa_path: Path) -> dict[str, Any]:
    """Structured view of one Classification ``frame*_QA.txt`` file."""
    organ: tuple[str, str] | None = None
    instruments: dict[str, dict[str, str]] = {}

    for question, answer, inst, qtype in _iter_classification_rows(qa_path):
        if qtype == "organ":
            organ = (question, answer)
            continue
        if qtype == "state" and inst:
            instruments.setdefault(inst, {})["state"] = answer
            continue
        if qtype == "location" and inst:
            instruments.setdefault(inst, {})["location"] = answer
            continue

    tool_names = tuple(sorted(instruments.keys()))
    return {
        "organ": organ,
        "instruments": instruments,
        "tool_names": tool_names,
    }


def _format_tools_gold(tool_names: tuple[str, ...]) -> str:
    return ", ".join(tool_names)


def _append_sample(
    samples: list[dict[str, Any]],
    *,
    seq_num: str,
    seq_dir: str,
    frame_index: int,
    qa_path: Path,
    question_index: int,
    img_path: Path,
    question: str,
    gold_keyword: str,
    instrument_options: list[str],
    instrument: str | None = None,
    gold_keywords: list[str] | None = None,
) -> None:
    options = options_for_question(question, instrument_options=instrument_options)
    if question_type(question) != "tools" and gold_keyword not in options:
        raise ValueError(
            f"Gold {gold_keyword!r} not in options for {question!r} ({qa_path})"
        )
    if question_type(question) == "tools":
        gold_set = {canonical_instrument(k) for k in (gold_keywords or [])}
        bad = [k for k in gold_set if k not in {canonical_instrument(o) for o in options}]
        if bad:
            raise ValueError(
                f"Gold tools {bad!r} not in instrument options for {qa_path}"
            )

    sample: dict[str, Any] = {
        "seq": seq_num,
        "seq_dir": seq_dir,
        "frame_index": frame_index,
        "qa_file": str(qa_path.resolve()),
        "question_index": question_index,
        "question": question,
        "question_template": question_template(question),
        "question_type": question_type(question),
        "gold_keyword": gold_keyword,
        "options": options,
        "img_path": img_path,
    }
    if instrument is not None:
        sample["instrument"] = instrument
    if gold_keywords is not None:
        sample["gold_keywords"] = list(gold_keywords)
    samples.append(sample)


def collect_classification_samples(
    vqa_root: Path,
    *,
    images_root: Path,
    image_split: ImageSplit = "val",
    seq_filter: str | None = None,
) -> list[dict[str, Any]]:
    """Build per-frame QA samples: organ, per-instrument state/location, and tools (Q5)."""
    vqa_root = vqa_root.resolve()
    images_root = images_root.resolve()
    instrument_options = list(discover_instruments(vqa_root))
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

            frame = parse_classification_frame(qa_path)
            q_idx = 0

            organ = frame["organ"]
            if organ is not None:
                question, gold = organ
                _append_sample(
                    samples,
                    seq_num=seq_num,
                    seq_dir=seq_dir.name,
                    frame_index=frame_index,
                    qa_path=qa_path,
                    question_index=q_idx,
                    img_path=img_path,
                    question=question,
                    gold_keyword=gold,
                    instrument_options=instrument_options,
                )
                q_idx += 1

            instruments: dict[str, dict[str, str]] = frame["instruments"]
            for inst in sorted(instruments.keys()):
                state = instruments[inst].get("state")
                if state is None:
                    continue
                question = f"What is the state of {inst}?"
                _append_sample(
                    samples,
                    seq_num=seq_num,
                    seq_dir=seq_dir.name,
                    frame_index=frame_index,
                    qa_path=qa_path,
                    question_index=q_idx,
                    img_path=img_path,
                    question=question,
                    gold_keyword=state,
                    instrument_options=instrument_options,
                    instrument=inst,
                )
                q_idx += 1

            for inst in sorted(instruments.keys()):
                location = instruments[inst].get("location")
                if location is None:
                    continue
                question = f"Where is {inst} located?"
                _append_sample(
                    samples,
                    seq_num=seq_num,
                    seq_dir=seq_dir.name,
                    frame_index=frame_index,
                    qa_path=qa_path,
                    question_index=q_idx,
                    img_path=img_path,
                    question=question,
                    gold_keyword=location,
                    instrument_options=instrument_options,
                    instrument=inst,
                )
                q_idx += 1

            tool_names: tuple[str, ...] = frame["tool_names"]
            if tool_names:
                _append_sample(
                    samples,
                    seq_num=seq_num,
                    seq_dir=seq_dir.name,
                    frame_index=frame_index,
                    qa_path=qa_path,
                    question_index=q_idx,
                    img_path=img_path,
                    question=TOOLS_QUESTION,
                    gold_keyword=_format_tools_gold(tool_names),
                    gold_keywords=list(tool_names),
                    instrument_options=instrument_options,
                )

    return samples
