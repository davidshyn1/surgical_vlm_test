"""
visual_cross_attention_endovis2017.py

EndoVis 2017 val: query × test frame cross-attention on patch features.
GT bboxes are derived from instrument segmentation masks (same pipeline as
``instrument_localization_endovis17``).

Usage:
  python visual_cross_attention_endovis2017.py --val-split val1 --frame-stem seq_1_frame225
  python visual_cross_attention_endovis2017.py --samples-per-instrument 3 --query-from-gt-crop
  BACKEND=qwen3-4b CUDA_VISIBLE_DEVICES=0 MODEL_ID=khtks/Qwen3-VL/surgsigma_qwen3vl_full \\
    bash grounding_task.sh visual_cross_attention_endovis2017 \\
      --feature-backbone hf --query-from-gt-crop --samples-per-instrument 3
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from PIL import Image

from backend_registry import (
    BACKEND_CHOICES,
    resolve_hf_token,
    resolve_model_id,
    resolve_output_model_name,
)
from cholect50_data import infer_pil_side, sample_by_instrument
from endovis17_data import (
    DEFAULT_BBOX_MODE,
    DEFAULT_DATASET_ROOT,
    DEFAULT_MIN_COMPONENT_PIXELS,
    BboxMode,
    collect_localization_samples,
    instrument_display_name,
    list_val_splits,
)
from patch_feature_extractors import load_patch_feature_backbone
from utils import resolve_device
from visual_cross_attention_cholect50 import (
    DEFAULT_IMAGE_SIZE,
    METHODS,
    SUMMARY_ATTENTION_METHOD,
    _DEFAULT_HF_TOKEN,
    build_per_instrument_summary,
    build_query_index,
    overall_avg_from_summary,
    print_per_instrument_summary,
    run_cross_attention_for_frame,
)

_SCRIPT_ROOT = Path(__file__).resolve().parent
_DEFAULT_OUTPUT_ROOT = _SCRIPT_ROOT / "outputs" / "visual_cross_attention_endovis2017"
_DEFAULT_QUERY_DIR = _SCRIPT_ROOT / "assets" / "endovis2017_query"


def collect_endovis_visual_items(
    *,
    dataset_root: Path,
    val_splits: list[str] | None = None,
    instrument_filter: str | None = None,
    frame_stem_filter: str | None = None,
    bbox_mode: BboxMode = DEFAULT_BBOX_MODE,
    min_component_pixels: int = DEFAULT_MIN_COMPONENT_PIXELS,
) -> list[dict[str, Any]]:
    """Mask-derived bbox items for visual cross-attention (one row per instrument per frame)."""
    samples = collect_localization_samples(
        dataset_root=dataset_root,
        val_splits=val_splits,
        instrument_filter=instrument_filter,
        frame_stem_filter=frame_stem_filter,
        bbox_mode=bbox_mode,
        min_component_pixels=min_component_pixels,
        instrument_taxonomy="endovis17",
    )
    items: list[dict[str, Any]] = []
    for s in samples:
        inst_id = str(s["instrument_id"]).strip().lower()
        bbox = s.get("bbox_xyxy_norm")
        if not bbox or len(bbox) != 4:
            continue
        items.append(
            {
                "val_split": s["val_split"],
                "frame_stem": s["frame_stem"],
                "frame_key": (s["val_split"], s["frame_stem"]),
                "img_path": Path(s["img_path"]),
                "instrument_id": inst_id,
                "instrument_norm": inst_id,
                "instrument_name": str(
                    s.get("instrument_display") or instrument_display_name(inst_id)
                ),
                "bbox_xyxy": tuple(float(v) for v in bbox),
                "bbox_xyxy_px": s.get("bbox_xyxy_px"),
                "mask_class_id": s.get("mask_class_id"),
                "bbox_mode": s.get("bbox_mode", bbox_mode),
            }
        )
    return items


def annotations_for_frame(frame_items: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], set[str]]:
    """Build CholecT50-compatible annotation dicts + frame label set for one EndoVis frame."""
    annotations: list[dict[str, Any]] = []
    frame_labels: set[str] = set()
    for it in frame_items:
        inst = it["instrument_name"]
        inst_id = it["instrument_id"]
        frame_labels.add(inst_id)
        frame_labels.add(inst)
        annotations.append(
            {
                "instrument": inst,
                "instrument_id": inst_id,
                "bbox_xyxy": it["bbox_xyxy"],
                "mask_class_id": it.get("mask_class_id"),
            }
        )
    return annotations, frame_labels


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="EndoVis 2017 cross-attention visual grounding (mask-derived bbox GT)."
    )
    p.add_argument("--backend", choices=BACKEND_CHOICES, default="prismatic")
    p.add_argument(
        "--feature-backbone",
        choices=("timm", "prismatic", "hf"),
        default="timm",
        help="timm/prismatic: DINO+SigLIP; hf: --backend model vision tower.",
    )
    p.add_argument(
        "--query-encoding",
        choices=("vision_ref", "pixel_grid", "single", "backbone"),
        default="vision_ref",
    )
    mode = p.add_mutually_exclusive_group()
    mode.add_argument(
        "--val-split",
        type=str,
        default=None,
        help="EndoVis val split (e.g. val1). Use with --frame-stem.",
    )
    mode.add_argument("--test-image", type=Path, default=None, help="Single BMP/PNG frame path.")
    mode.add_argument(
        "--eval-all",
        action="store_true",
        help="All mask-derived bbox samples (no per-instrument cap).",
    )
    p.add_argument(
        "--frame-stem",
        type=str,
        default=None,
        help="Frame stem matching label/*.bmp (e.g. seq_1_frame225). Requires --val-split.",
    )
    p.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    p.add_argument(
        "--val-splits",
        type=str,
        nargs="*",
        default=None,
        help="Batch mode: subset of val splits (default: all val* under dataset-root).",
    )
    p.add_argument("--instrument", type=str, default=None)
    p.add_argument("--samples-per-instrument", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit", type=int, default=None, help="Max frames in batch mode.")
    p.add_argument(
        "--bbox-mode",
        choices=("all_pixels", "filtered_union", "largest_component"),
        default=DEFAULT_BBOX_MODE,
    )
    p.add_argument("--min-component-pixels", type=int, default=DEFAULT_MIN_COMPONENT_PIXELS)
    p.add_argument("--query-dir", type=Path, default=_DEFAULT_QUERY_DIR)
    p.add_argument("--query-from-gt-crop", action="store_true")
    p.add_argument("--fuzzy-cutoff", type=float, default=0.4)
    p.add_argument("--model-id", type=str, default=None)
    p.add_argument("--model-name", type=str, default=None)
    p.add_argument("--vlm-checkpoint", type=Path, default=None)
    p.add_argument("--vlm-config", type=Path, default=None)
    p.add_argument("--hf-token", type=Path, default=_DEFAULT_HF_TOKEN)
    p.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_SIZE)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--device", type=str, default="0")
    p.add_argument("--alpha", type=float, default=0.55)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    backend = args.backend
    model_id = resolve_model_id(backend, args.model_id)
    model_name = resolve_output_model_name(backend, args.model_name, model_id)

    out_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else (_DEFAULT_OUTPUT_ROOT / f"{model_name}").resolve()
    )
    viz_dir = out_dir / "viz"
    viz_dir.mkdir(parents=True, exist_ok=True)

    image_size = args.image_size
    if args.feature_backbone == "prismatic":
        image_size = infer_pil_side(args)

    hf_token = resolve_hf_token(backend, args.hf_token)
    bp = load_patch_feature_backbone(
        args.feature_backbone,
        backend=backend,
        model_id=model_id,
        hf_token=hf_token,
        device=device,
        image_size=image_size,
        vlm_checkpoint=args.vlm_checkpoint,
        vlm_config=args.vlm_config,
    )

    dataset_root = args.dataset_root.resolve()
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"EndoVis 2017 root not found: {dataset_root}")

    val_splits = args.val_splits or list_val_splits(dataset_root)
    bbox_mode: BboxMode = args.bbox_mode  # type: ignore[assignment]

    query_dir = args.query_dir.resolve()
    query_index = build_query_index(query_dir)
    if not query_index and not args.query_from_gt_crop:
        print(
            f"[WARN] Empty query index at {query_dir}. Use --query-from-gt-crop or add PNGs.",
            file=sys.stderr,
        )

    run_title = f"{args.feature_backbone} | query={args.query_encoding} | {model_name}"
    print(
        f"[INFO] out={out_dir}  dataset={dataset_root}  splits={val_splits}  "
        f"bbox_mode={bbox_mode}  grid={bp.grid_h}x{bp.grid_w}  queries={len(query_index)}",
        file=sys.stderr,
    )

    all_records: list[dict[str, Any]] = []

    def _run_one_frame(
        test_path: Path,
        test_label: str,
        frame_items: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        anns, frame_labels = annotations_for_frame(frame_items)
        if not anns:
            print(f"[WARN] No instruments on {test_label}", file=sys.stderr)
            return []
        return run_cross_attention_for_frame(
            bp=bp,
            test_path=test_path,
            test_label=test_label,
            labels_dir=None,
            query_index=query_index,
            query_from_gt_crop=args.query_from_gt_crop,
            fuzzy_cutoff=args.fuzzy_cutoff,
            output_dir=viz_dir,
            alpha=args.alpha,
            run_title=run_title,
            instrument_filter=args.instrument,
            query_encoding=args.query_encoding,
            annotations=anns,
            frame_labels=frame_labels,
        )

    if args.test_image is not None:
        test_path = args.test_image.resolve()
        if args.val_split and args.frame_stem:
            frame_items = collect_endovis_visual_items(
                dataset_root=dataset_root,
                val_splits=[args.val_split],
                instrument_filter=args.instrument,
                frame_stem_filter=args.frame_stem,
                bbox_mode=bbox_mode,
                min_component_pixels=args.min_component_pixels,
            )
            test_label = f"{args.val_split}_{args.frame_stem}"
        else:
            frame_items = []
            if args.instrument:
                frame_items = [
                    {
                        "instrument_name": args.instrument,
                        "instrument_id": args.instrument.strip().lower(),
                        "bbox_xyxy": (0.0, 0.0, 1.0, 1.0),
                    }
                ]
            test_label = test_path.stem
        all_records = _run_one_frame(test_path, test_label, frame_items)

    elif args.val_split is not None:
        if not args.frame_stem:
            raise SystemExit("--val-split requires --frame-stem")
        frame_items = collect_endovis_visual_items(
            dataset_root=dataset_root,
            val_splits=[args.val_split],
            instrument_filter=args.instrument,
            frame_stem_filter=args.frame_stem,
            bbox_mode=bbox_mode,
            min_component_pixels=args.min_component_pixels,
        )
        if not frame_items:
            raise FileNotFoundError(
                f"No mask bbox for {args.val_split}/{args.frame_stem} under {dataset_root}"
            )
        test_path = frame_items[0]["img_path"]
        test_label = f"{args.val_split}_{args.frame_stem}"
        all_records = _run_one_frame(test_path, test_label, frame_items)

    else:
        items = collect_endovis_visual_items(
            dataset_root=dataset_root,
            val_splits=val_splits,
            instrument_filter=args.instrument,
            bbox_mode=bbox_mode,
            min_component_pixels=args.min_component_pixels,
        )
        if args.eval_all:
            sampled = items
        else:
            sampled = sample_by_instrument(
                items, cap_per_instrument=max(0, args.samples_per_instrument), seed=args.seed
            )
        if args.limit is not None:
            sampled = sampled[: args.limit]

        by_frame: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for it in sampled:
            by_frame.setdefault(it["frame_key"], []).append(it)

        print(
            f"[BATCH] {len(by_frame)} frames from {len(sampled)} instrument boxes",
            file=sys.stderr,
        )
        for (val_split, frame_stem), frame_items in sorted(by_frame.items()):
            test_path = frame_items[0]["img_path"]
            test_label = f"{val_split}_{frame_stem}"
            print(f"\n[{test_label}] {test_path}", file=sys.stderr)
            recs = _run_one_frame(test_path, test_label, frame_items)
            for rec in recs:
                rec["val_split"] = val_split
                rec["frame_stem"] = frame_stem
            all_records.extend(recs)

    per_inst = None
    overall_avg = None
    if all_records:
        per_inst = build_per_instrument_summary(
            all_records,
            primary_source=bp.source_names[0],
            samples_per_instrument=(
                None if args.eval_all else max(0, args.samples_per_instrument)
            ),
            eval_all=bool(args.eval_all),
        )
        overall_avg = overall_avg_from_summary(per_inst)
        print_per_instrument_summary(per_inst)

    payload = {
        "task": "visual_cross_attention_endovis2017",
        "backend": backend,
        "feature_backbone": args.feature_backbone,
        "query_encoding": args.query_encoding,
        "model_id": model_id,
        "model_name": model_name,
        "dataset_root": str(dataset_root),
        "val_splits": val_splits,
        "bbox_mode": bbox_mode,
        "min_component_pixels": int(args.min_component_pixels),
        "query_dir": str(query_dir),
        "query_from_gt_crop": bool(args.query_from_gt_crop),
        "samples_per_instrument": args.samples_per_instrument,
        "eval_all": bool(args.eval_all),
        "seed": args.seed,
        "grid": f"{bp.grid_h}x{bp.grid_w}",
        "sources": bp.source_names,
        "source_labels": bp.source_labels,
        "methods": METHODS,
        "summary_method": SUMMARY_ATTENTION_METHOD,
        "vision_meta": bp.meta,
        "visualization_dir": str(viz_dir),
        "n_records": len(all_records),
        "per_instrument_summary": per_inst,
        "overall_avg": overall_avg,
        "records": all_records,
    }
    out_json = out_dir / "cross_attention.json"
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"\n[SAVE] {out_json}  viz → {viz_dir}  ({len(all_records)} records)", file=sys.stderr)


if __name__ == "__main__":
    main()
