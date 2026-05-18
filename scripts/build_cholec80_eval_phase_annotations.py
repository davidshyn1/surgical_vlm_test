#!/usr/bin/env python3
"""Write 0.1 fps phase manifests under frames_0p1fps/ without re-extracting PNGs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(_SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_ROOT))

from cholec80_data import (  # noqa: E402
    CHOLEC80_EVAL_FPS,
    CHOLEC80_EVAL_FRAME_STRIDE,
    CHOLEC80_VIDEO_FPS,
    build_subsampled_phase_annotations_for_split,
    default_eval_frames_root,
    infer_native_phase_frame_stride,
    parse_video_id,
    resolve_cholec80_root,
    video_in_split,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Build eval phase annotation files (0.1 fps) under frames_0p1fps/videoNN/."
        ),
    )
    p.add_argument("--dataset-root", type=Path, default=None)
    p.add_argument(
        "--frames-root",
        type=Path,
        default=None,
        help="Output root (default: <dataset-root>/frames_0p1fps).",
    )
    p.add_argument("--split", choices=("eval", "train", "all"), default="eval")
    p.add_argument("--video", type=str, default=None)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument(
        "--verify-native",
        action="store_true",
        help="Print inferred native phase frame stride for one video.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = resolve_cholec80_root(args.dataset_root)
    frames_root = (
        args.frames_root.resolve()
        if args.frames_root is not None
        else default_eval_frames_root(dataset_root)
    )
    frames_root.mkdir(parents=True, exist_ok=True)

    video_filter = parse_video_id(args.video) if args.video else None
    if video_filter is not None and not video_in_split(video_filter, args.split):
        print(
            f"WARN: video {video_filter} outside split={args.split!r}",
            file=sys.stderr,
        )

    if args.verify_native:
        from cholec80_data import list_phase_annotation_files, video_stem

        for vid, path in list_phase_annotation_files(
            dataset_root,
            split=args.split,
            video_filter=video_filter,
        )[:1]:
            stride = infer_native_phase_frame_stride(path)
            print(
                f"{video_stem(vid)} native phase stride≈{stride} "
                f"({CHOLEC80_VIDEO_FPS} fps video; eval {CHOLEC80_EVAL_FPS} fps "
                f"=> stride {CHOLEC80_EVAL_FRAME_STRIDE})",
                file=sys.stderr,
            )

    rows = build_subsampled_phase_annotations_for_split(
        dataset_root,
        frames_root,
        split=args.split,
        video_filter=video_filter,
        overwrite=args.overwrite,
    )
    total = sum(n for _, _, n in rows)
    print(
        f"Wrote {len(rows)} manifest(s), {total} phase rows total -> {frames_root}",
        file=sys.stderr,
    )
    for vid, path, n in rows[:5]:
        print(f"  video{vid:02d}: {n} rows -> {path}", file=sys.stderr)
    if len(rows) > 5:
        print(f"  ... and {len(rows) - 5} more", file=sys.stderr)


if __name__ == "__main__":
    main()
