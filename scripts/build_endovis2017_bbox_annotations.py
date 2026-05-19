#!/usr/bin/env python3
"""Export EndoVis2017 val mask-derived bbox annotations to JSON."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(_SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_ROOT))

from endovis17_data import (  # noqa: E402
    DEFAULT_BBOX_MODE,
    DEFAULT_DATASET_ROOT,
    DEFAULT_MIN_COMPONENT_PIXELS,
    collect_localization_samples,
    export_bbox_annotations,
    list_val_splits,
)


def main() -> None:
    p = argparse.ArgumentParser(description="Build EndoVis2017 mask→bbox annotation JSON.")
    p.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    p.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_DATASET_ROOT / "derived_bbox_annotations_val.json",
    )
    p.add_argument("--val-split", action="append", default=None, help="e.g. val1 (repeatable)")
    p.add_argument("--min-mask-pixels", type=int, default=1)
    p.add_argument(
        "--bbox-mode",
        choices=("all_pixels", "filtered_union", "largest_component"),
        default=DEFAULT_BBOX_MODE,
    )
    p.add_argument(
        "--min-component-pixels",
        type=int,
        default=DEFAULT_MIN_COMPONENT_PIXELS,
    )
    args = p.parse_args()

    root = args.dataset_root.resolve()
    splits = args.val_split or list_val_splits(root)
    samples = collect_localization_samples(
        dataset_root=root,
        val_splits=splits,
        min_mask_pixels=args.min_mask_pixels,
        bbox_mode=args.bbox_mode,
        min_component_pixels=args.min_component_pixels,
    )
    export_bbox_annotations(
        samples,
        args.output.resolve(),
        dataset_root=root,
        bbox_mode=args.bbox_mode,
        min_component_pixels=args.min_component_pixels,
    )
    print(f"Wrote {len(samples)} samples to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
