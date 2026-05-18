"""Cholec80 phase-annotation loading and video frame access."""

from __future__ import annotations

import csv
import io
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Iterator, Literal, TYPE_CHECKING

FrameReader = Literal["auto", "ffmpeg", "opencv"]

if TYPE_CHECKING:
    from PIL import Image

_PKG_ROOT = Path(__file__).resolve().parent
REPO_ROOT = _PKG_ROOT.parent

_DEFAULT_MODEL_IDS = {
    "prismatic": "prism-dinosiglip+7b",
    "cosmos": "nvidia/Cosmos-Reason2-2B",
    "groot": "nvidia/GR00T-H",
}

# Canonical phase ids (7 classes, Cholec80 / EndoNet).
PHASE_CANONICAL_IDS = [
    "preparation",
    "calot_triangle_dissection",
    "clipping_cutting",
    "gallbladder_dissection",
    "gallbladder_packaging",
    "cleaning_coagulation",
    "gallbladder_retraction",
]

# Human-readable labels for prompts (bench figure style).
PHASE_DISPLAY_NAMES = [
    "Preparation",
    "Calot Triangle Dissection",
    "Clipping and Cutting",
    "Gallbladder Dissection",
    "Gallbladder Packaging",
    "Cleaning and Coagulation",
    "Gallbladder Retraction",
]

# Raw strings in videoNN-phase.txt (column Phase).
_ANNOTATION_TO_CANONICAL: dict[str, str] = {
    "preparation": "preparation",
    "calottriangledissection": "calot_triangle_dissection",
    "clippingcutting": "clipping_cutting",
    "gallbladderdissection": "gallbladder_dissection",
    "gallbladderpackaging": "gallbladder_packaging",
    "cleaningcoagulation": "cleaning_coagulation",
    "gallbladderretraction": "gallbladder_retraction",
}

CANONICAL_TO_DISPLAY: dict[str, str] = dict(
    zip(PHASE_CANONICAL_IDS, PHASE_DISPLAY_NAMES, strict=True)
)

_IMG_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")


def resolve_cholec80_root(requested: Path | None) -> Path:
    """Resolve dataset root (supports Cholec80 / cholec80 naming)."""
    candidates: list[Path] = []
    if requested is not None:
        candidates.append(requested.resolve())
    env = __import__("os").environ.get("CHOLEC80_ROOT", "").strip()
    if env:
        candidates.append(Path(env).resolve())
    candidates.extend([
        (REPO_ROOT / "data" / "Cholec80").resolve(),
        (REPO_ROOT / "data" / "cholec80").resolve(),
    ])
    seen: set[str] = set()
    for root in candidates:
        key = str(root)
        if key in seen:
            continue
        seen.add(key)
        if (root / "phase_annotations").is_dir() and (root / "videos").is_dir():
            return root
    tried = ", ".join(str(p) for p in candidates)
    raise FileNotFoundError(
        f"Cholec80 root not found (need phase_annotations/ and videos/). Tried: {tried}"
    )


def normalize_phase_label(raw: str | None) -> str | None:
    key = re.sub(r"[^a-z0-9]+", "", (raw or "").strip().lower())
    if not key:
        return None
    return _ANNOTATION_TO_CANONICAL.get(key)


def parse_video_id(name: str) -> int | None:
    s = (name or "").strip()
    m = re.match(r"^(?:video|vid)?(\d+)$", s, re.IGNORECASE)
    return int(m.group(1)) if m else None


def video_stem(vid_num: int) -> str:
    return f"video{vid_num:02d}"


def video_in_split(vid_num: int, split: str) -> bool:
    sp = (split or "eval").strip().lower()
    if sp in ("eval", "test", "evaluation"):
        return 41 <= vid_num <= 80
    if sp in ("train", "finetune", "finetuning"):
        return 1 <= vid_num <= 40
    if sp in ("all", "full"):
        return 1 <= vid_num <= 80
    raise ValueError(f"Unknown split {split!r}; use eval, train, or all.")


def load_phase_annotation_rows(
    phase_file: Path,
    *,
    frame_stride: int = 1,
    max_frames: int | None = None,
) -> list[tuple[int, str]]:
    """Return (frame_index, canonical_phase_id) rows from a phase file."""
    stride = max(1, int(frame_stride))
    rows: list[tuple[int, str]] = []
    with phase_file.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        if reader.fieldnames is None:
            raise ValueError(f"Invalid phase file (no header): {phase_file}")
        frame_key = reader.fieldnames[0]
        phase_key = reader.fieldnames[1]
        for row in reader:
            try:
                fi = int(row[frame_key])
            except (KeyError, TypeError, ValueError):
                continue
            if fi % stride != 0:
                continue
            phase_id = normalize_phase_label(row.get(phase_key))
            if phase_id is None:
                continue
            rows.append((fi, phase_id))
            if max_frames is not None and len(rows) >= max_frames:
                break
    return rows


def list_phase_annotation_files(
    dataset_root: Path,
    *,
    split: str = "eval",
    video_filter: int | None = None,
) -> list[tuple[int, Path]]:
    ann_dir = dataset_root / "phase_annotations"
    out: list[tuple[int, Path]] = []
    for path in sorted(ann_dir.glob("video*-phase.txt")):
        vid = parse_video_id(path.stem.replace("-phase", ""))
        if vid is None:
            continue
        if video_filter is not None and vid != video_filter:
            continue
        if not video_in_split(vid, split):
            continue
        out.append((vid, path))
    return out


def resolve_frame_image_path(
    vid_stem: str,
    frame_index: int,
    frames_root: Path,
) -> Path | None:
    """Pre-extracted frame: frames_root/video41/000123.png (CholecT50-style layout)."""
    stem = f"{int(frame_index):06d}"
    vid = (vid_stem or "").strip()
    candidates = [vid, vid.lower()]
    m = re.match(r"^video(\d+)$", vid, re.IGNORECASE)
    if m:
        n = int(m.group(1))
        candidates.extend([f"video{n:02d}", f"video{n}", f"VID{n}", f"VID{n:02d}"])
    seen: set[str] = set()
    for name in candidates:
        if name in seen:
            continue
        seen.add(name)
        vdir = frames_root / name
        if not vdir.is_dir():
            continue
        for ext in _IMG_EXTS:
            p = vdir / f"{stem}{ext}"
            if p.is_file():
                return p.resolve()
    return None


def collect_phase_samples(
    dataset_root: Path,
    *,
    split: str = "eval",
    video_filter: int | None = None,
    frame_stride: int = 1,
    max_frames_per_video: int | None = None,
    frames_root: Path | None = None,
) -> list[dict[str, Any]]:
    """Build evaluation items: one row per (video, frame_index)."""
    video_dir = dataset_root / "videos"
    items: list[dict[str, Any]] = []
    for vid_num, phase_path in list_phase_annotation_files(
        dataset_root,
        split=split,
        video_filter=video_filter,
    ):
        stem = video_stem(vid_num)
        video_path = video_dir / f"{stem}.mp4"
        if not video_path.is_file():
            alt = video_dir / f"video{vid_num}.mp4"
            video_path = alt if alt.is_file() else video_path
        if not video_path.is_file():
            print(f"WARN: missing video {video_path}", file=__import__("sys").stderr)
            continue
        for fi, phase_id in load_phase_annotation_rows(
            phase_path,
            frame_stride=frame_stride,
            max_frames=max_frames_per_video,
        ):
            img_path = None
            if frames_root is not None:
                img_path = resolve_frame_image_path(stem, fi, frames_root)
            items.append(
                {
                    "vid_num": vid_num,
                    "vid": stem,
                    "frame_index": fi,
                    "phase_id": phase_id,
                    "phase_display": CANONICAL_TO_DISPLAY.get(phase_id, phase_id),
                    "video_path": video_path.resolve(),
                    "phase_annotation": phase_path.resolve(),
                    "img_path": img_path,
                }
            )
    return items


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def read_video_frame_rgb_ffmpeg(video_path: Path, frame_index: int) -> "Image.Image":
    """Decode one frame via ffmpeg subprocess (no OpenCV; does not change numpy)."""
    from PIL import Image

    if not ffmpeg_available():
        raise RuntimeError(
            "ffmpeg not found on PATH. Install ffmpeg, or pass --frames-root with "
            "pre-extracted PNG/JPG frames."
        )
    n = int(frame_index)
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(video_path),
        "-vf",
        f"select=eq(n\\,{n})",
        "-vsync",
        "vfr",
        "-frames:v",
        "1",
        "-f",
        "image2pipe",
        "-vcodec",
        "png",
        "pipe:1",
    ]
    proc = subprocess.run(cmd, capture_output=True, check=False)
    if proc.returncode != 0 or not proc.stdout:
        err = (proc.stderr or b"").decode("utf-8", errors="replace").strip()
        raise OSError(
            f"ffmpeg could not read frame {n} from {video_path}"
            + (f": {err}" if err else "")
        )
    return Image.open(io.BytesIO(proc.stdout)).convert("RGB")


def read_video_frame_rgb_opencv(video_path: Path, frame_index: int) -> "Image.Image":
    """Optional OpenCV path (only if cv2 is already installed)."""
    from PIL import Image

    try:
        import cv2  # type: ignore
    except ImportError as e:
        raise ImportError(
            "opencv (cv2) is not installed. Use frame_reader=ffmpeg or --frames-root."
        ) from e

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise OSError(f"Could not open video: {video_path}")
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        ok, frame = cap.read()
        if not ok or frame is None:
            raise OSError(f"Could not read frame {frame_index} from {video_path}")
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb)
    finally:
        cap.release()


def load_frame_rgb(
    sample: dict[str, Any],
    *,
    frame_reader: FrameReader = "auto",
) -> "Image.Image":
    """
    Load a frame for eval: pre-extracted image (img_path) > ffmpeg > opencv (auto only).
    """
    from PIL import Image

    img_path = sample.get("img_path")
    if img_path is not None:
        p = Path(img_path)
        if p.is_file():
            return Image.open(p).convert("RGB")

    video_path = Path(sample["video_path"])
    frame_index = int(sample["frame_index"])
    mode = (frame_reader or "auto").strip().lower()

    if mode == "ffmpeg":
        return read_video_frame_rgb_ffmpeg(video_path, frame_index)
    if mode == "opencv":
        return read_video_frame_rgb_opencv(video_path, frame_index)

    # auto: prefer ffmpeg (no extra pip deps)
    if ffmpeg_available():
        return read_video_frame_rgb_ffmpeg(video_path, frame_index)
    try:
        return read_video_frame_rgb_opencv(video_path, frame_index)
    except ImportError:
        raise RuntimeError(
            "No frame source: set --frames-root, install ffmpeg on PATH, or install opencv "
            "(may conflict with pinned numpy)."
        ) from None


def read_video_frame_rgb(video_path: Path, frame_index: int) -> "Image.Image":
    """Backward-compatible alias (ffmpeg by default)."""
    return read_video_frame_rgb_ffmpeg(video_path, frame_index)


def iter_samples_by_video(
    items: list[dict[str, Any]],
) -> Iterator[tuple[Path, list[dict[str, Any]]]]:
    """Group samples by video_path (sorted by frame_index within each video)."""
    from collections import defaultdict

    grouped: dict[Path, list[dict[str, Any]]] = defaultdict(list)
    for it in items:
        grouped[it["video_path"]].append(it)
    for vpath in sorted(grouped.keys(), key=str):
        rows = sorted(grouped[vpath], key=lambda x: int(x["frame_index"]))
        yield vpath, rows
