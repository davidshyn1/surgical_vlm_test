#!/usr/bin/env python3
"""Extract SAR-RARP50 frames at segmentation indices into video_xx/frames/."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(_SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_ROOT))

from cholec80_data import FrameReader  # noqa: E402
from sarrarp50_data import (  # noqa: E402
    extract_video_frames,
    list_video_dirs,
    parse_video_dir_name,
    resolve_sarrarp50_root,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Extract frames from video_left.avi at segmentation mask indices "
            "into video_xx/frames/ and write action_samples.jsonl manifests."
        ),
    )
    p.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help="SAR-RARP50 root (default: ../eval/sarrarp50).",
    )
    p.add_argument(
        "--video",
        type=str,
        default=None,
        help="Single video directory or id, e.g. 47 or video_47.",
    )
    p.add_argument("--overwrite", action="store_true")
    p.add_argument(
        "--frame-reader",
        choices=("auto", "ffmpeg", "opencv"),
        default="auto",
    )
    return p.parse_args()


def _parse_video_filter(raw: str | None) -> int | None:
    if raw is None:
        return None
    s = raw.strip()
    if s.lower().startswith("video_"):
        s = s.split("_", 1)[1]
    return int(s)


def main() -> None:
    args = parse_args()
    dataset_root = resolve_sarrarp50_root(args.dataset_root)
    video_filter = _parse_video_filter(args.video)

    video_dirs = list_video_dirs(dataset_root, video_filter=video_filter)
    if not video_dirs:
        raise RuntimeError(
            f"No video_* directories under {dataset_root}"
            + (f" (filter={video_filter})" if video_filter is not None else "")
        )

    total_seg = total_written = total_missing = 0
    reader: FrameReader = args.frame_reader  # type: ignore[assignment]

    for video_dir in video_dirs:
        vid = parse_video_dir_name(video_dir.name)
        try:
            n_seg, n_written, n_missing = extract_video_frames(
                video_dir,
                frame_reader=reader,
                overwrite=args.overwrite,
            )
        except FileNotFoundError as e:
            print(f"SKIP {video_dir.name}: {e}", file=sys.stderr)
            continue

        total_seg += n_seg
        total_written += n_written
        total_missing += n_missing
        print(
            f"{video_dir.name}: segmentation={n_seg}, "
            f"frames={n_written}, missing_action={n_missing}",
            file=sys.stderr,
        )

    print(
        f"Done. videos={len(video_dirs)}, segmentation_indices={total_seg}, "
        f"frames_on_disk={total_written}, missing_action_rows={total_missing}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
