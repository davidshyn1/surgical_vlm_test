"""SAR-RARP50 dataset helpers (segmentation-indexed frames + action_discrete GT)."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Iterator

from cholec80_data import (
    FrameReader,
    ffmpeg_available,
    load_frame_rgb,
    read_video_frame_rgb_ffmpeg,
    read_video_frame_rgb_opencv,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_ROOT = REPO_ROOT / "eval" / "sarrarp50"
FRAMES_SUBDIR = "frames"
SEGMENTATION_SUBDIR = "segmentation"
VIDEO_FILENAME = "video_left.avi"
ACTION_DISCRETE_FILE = "action_discrete.txt"
MANIFEST_FILENAME = "action_samples.jsonl"

# action_discrete.txt uses 0-based class ids (0–7 in the test split).
ACTION_CANONICAL_IDS: tuple[str, ...] = tuple(f"a{i}" for i in range(8))
ACTION_ID_TO_CANONICAL: dict[int, str] = {i: f"a{i}" for i in range(8)}
ACTION_CANONICAL_TO_ID: dict[str, int] = {v: k for k, v in ACTION_ID_TO_CANONICAL.items()}

ACTION_DISPLAY_NAMES: dict[int, str] = {
    0: "Other",
    1: "Picking-up the needle",
    2: "Positioning the needle tip",
    3: "Pushing the needle through the tissue",
    4: "Pulling the needle out of the tissue",
    5: "Tying a knot",
    6: "Cutting the suture",
    7: "Returning/dropping the needle",
}

CANONICAL_TO_DISPLAY: dict[str, str] = {
    ACTION_ID_TO_CANONICAL[i]: ACTION_DISPLAY_NAMES[i] for i in ACTION_DISPLAY_NAMES
}

_VIDEO_DIR_RE = re.compile(r"^video_(\d+)$", re.IGNORECASE)
_FRAME_STEM_RE = re.compile(r"^\d{9}$")


def resolve_sarrarp50_root(path: Path | None = None) -> Path:
    root = (path or DEFAULT_DATASET_ROOT).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"SAR-RARP50 root not found: {root}")
    return root


def parse_video_dir_name(name: str) -> int | None:
    m = _VIDEO_DIR_RE.match(name.strip())
    return int(m.group(1)) if m else None


def video_stem(vid_num: int) -> str:
    return f"video_{vid_num}"


def list_video_dirs(
    dataset_root: Path,
    *,
    video_filter: int | None = None,
) -> list[Path]:
    out: list[Path] = []
    for p in sorted(dataset_root.iterdir()):
        if not p.is_dir():
            continue
        vid = parse_video_dir_name(p.name)
        if vid is None:
            continue
        if video_filter is not None and vid != video_filter:
            continue
        out.append(p)
    return out


def parse_frame_stem(stem: str) -> int | None:
    if not _FRAME_STEM_RE.match(stem):
        return None
    return int(stem)


def load_action_discrete(path: Path) -> dict[int, int]:
    """Map native video frame index -> action class id."""
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}")
    out: dict[int, int] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or "," not in line:
            continue
        frame_s, action_s = line.split(",", 1)
        out[int(frame_s.strip())] = int(action_s.strip())
    return out


def lookup_action_at_frame(action_map: dict[int, int], frame_index: int) -> int | None:
    """Exact match on action_discrete frame index (10 Hz rows)."""
    return action_map.get(int(frame_index))


def list_segmentation_frame_indices(video_dir: Path) -> list[int]:
    seg_dir = video_dir / SEGMENTATION_SUBDIR
    if not seg_dir.is_dir():
        return []
    indices: list[int] = []
    for png in seg_dir.glob("*.png"):
        idx = parse_frame_stem(png.stem)
        if idx is not None:
            indices.append(idx)
    return sorted(indices)


def frame_png_path(video_dir: Path, frame_index: int) -> Path:
    return video_dir / FRAMES_SUBDIR / f"{int(frame_index):09d}.png"


def manifest_path(video_dir: Path) -> Path:
    return video_dir / FRAMES_SUBDIR / MANIFEST_FILENAME


def save_frame_png(image: Any, dest: Path) -> None:
    from PIL import Image

    dest.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(image, Image.Image):
        image.save(dest)
        return
    Image.fromarray(image).save(dest)


def extract_frame_rgb(
    video_path: Path,
    frame_index: int,
    *,
    frame_reader: FrameReader = "auto",
) -> Any:
    mode = (frame_reader or "auto").strip().lower()
    if mode == "ffmpeg":
        return read_video_frame_rgb_ffmpeg(video_path, frame_index)
    if mode == "opencv":
        return read_video_frame_rgb_opencv(video_path, frame_index)
    if ffmpeg_available():
        return read_video_frame_rgb_ffmpeg(video_path, frame_index)
    return read_video_frame_rgb_opencv(video_path, frame_index)


def write_manifest_row(
    fh: Any,
    *,
    vid: str,
    vid_num: int,
    frame_index: int,
    action_id: int,
    img_path: Path,
) -> None:
    row = {
        "vid": vid,
        "vid_num": vid_num,
        "frame_index": frame_index,
        "action_id": action_id,
        "action_canonical": ACTION_ID_TO_CANONICAL[action_id],
        "action_display": ACTION_DISPLAY_NAMES[action_id],
        "img_path": str(img_path),
    }
    fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def extract_video_frames(
    video_dir: Path,
    *,
    frame_reader: FrameReader = "auto",
    overwrite: bool = False,
) -> tuple[int, int, int]:
    """
    Extract PNGs under video_dir/frames/ for every segmentation index.
    Returns (n_segmentation, n_written, n_skipped_missing_action).
    """
    vid_num = parse_video_dir_name(video_dir.name)
    if vid_num is None:
        raise ValueError(f"Not a video directory: {video_dir}")

    video_path = video_dir / VIDEO_FILENAME
    action_path = video_dir / ACTION_DISCRETE_FILE
    if not video_path.is_file():
        raise FileNotFoundError(f"Missing {video_path}")

    frame_indices = list_segmentation_frame_indices(video_dir)
    if not frame_indices:
        return 0, 0, 0

    action_map = load_action_discrete(action_path)
    frames_dir = video_dir / FRAMES_SUBDIR
    frames_dir.mkdir(parents=True, exist_ok=True)

    n_written = 0
    n_missing_action = 0
    manifest_out = manifest_path(video_dir)

    with manifest_out.open("w", encoding="utf-8") as mf:
        for frame_index in frame_indices:
            action_id = lookup_action_at_frame(action_map, frame_index)
            if action_id is None:
                n_missing_action += 1
                continue

            dest = frame_png_path(video_dir, frame_index)
            if dest.is_file() and not overwrite:
                write_manifest_row(
                    mf,
                    vid=video_dir.name,
                    vid_num=vid_num,
                    frame_index=frame_index,
                    action_id=action_id,
                    img_path=dest,
                )
                n_written += 1
                continue

            image = extract_frame_rgb(
                video_path,
                frame_index,
                frame_reader=frame_reader,
            )
            save_frame_png(image, dest)
            write_manifest_row(
                mf,
                vid=video_dir.name,
                vid_num=vid_num,
                frame_index=frame_index,
                action_id=action_id,
                img_path=dest,
            )
            n_written += 1

    return len(frame_indices), n_written, n_missing_action


def action_id_to_canonical(action_id: int) -> str:
    if action_id not in ACTION_ID_TO_CANONICAL:
        raise ValueError(f"Unknown action id: {action_id}")
    return ACTION_ID_TO_CANONICAL[action_id]


def action_id_to_display(action_id: int) -> str:
    if action_id not in ACTION_DISPLAY_NAMES:
        raise ValueError(f"Unknown action id: {action_id}")
    return ACTION_DISPLAY_NAMES[action_id]


def collect_action_samples(
    dataset_root: Path,
    *,
    video_filter: int | None = None,
    max_samples: int | None = None,
    require_frames: bool = True,
) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for video_dir in list_video_dirs(dataset_root, video_filter=video_filter):
        vid_num = parse_video_dir_name(video_dir.name)
        assert vid_num is not None

        action_map = load_action_discrete(video_dir / ACTION_DISCRETE_FILE)
        frame_indices = list_segmentation_frame_indices(video_dir)
        if require_frames:
            frames_dir = video_dir / FRAMES_SUBDIR
            if frames_dir.is_dir():
                on_disk = {
                    idx
                    for p in frames_dir.glob("*.png")
                    if (idx := parse_frame_stem(p.stem)) is not None
                }
                frame_indices = sorted(on_disk) if on_disk else frame_indices
            else:
                frame_indices = []

        video_path = video_dir / VIDEO_FILENAME
        for frame_index in frame_indices:
            action_id = lookup_action_at_frame(action_map, frame_index)
            if action_id is None:
                continue

            img_path = frame_png_path(video_dir, frame_index)
            if require_frames and not img_path.is_file():
                continue

            canonical = action_id_to_canonical(action_id)
            samples.append(
                {
                    "vid": video_dir.name,
                    "vid_num": vid_num,
                    "frame_index": frame_index,
                    "action_id": action_id,
                    "action_canonical": canonical,
                    "action_display": action_id_to_display(action_id),
                    "video_path": str(video_path),
                    "img_path": str(img_path) if img_path.is_file() else None,
                    "dataset_root": str(dataset_root),
                }
            )
            if max_samples is not None and len(samples) >= max_samples:
                return samples
    return samples


def iter_samples_by_video(
    samples: list[dict[str, Any]],
) -> Iterator[tuple[str, list[dict[str, Any]]]]:
    from itertools import groupby

    ordered = sorted(
        samples,
        key=lambda s: (int(s["vid_num"]), int(s["frame_index"])),
    )
    for vid, group in groupby(ordered, key=lambda s: s["vid"]):
        yield vid, list(group)


def load_sample_frame_rgb(
    sample: dict[str, Any],
    *,
    frame_reader: FrameReader = "auto",
) -> Any:
    return load_frame_rgb(sample, frame_reader=frame_reader)
