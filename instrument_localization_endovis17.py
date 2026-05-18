"""
instrument_localization_endovis17.py

EndoVis 2017 val instrument localization (mask-derived bbox GT).

  - Data: ../eval/endovis2017 val*/image + label masks (512x512)
  - GT: tight bbox per instrument class from semantic mask
  - Prompt: normalized [x_min, y_min, x_max, y_max] in [0, 1]
  - VLM input and visualizations: same square resize (pil_side, e.g. 384x384)
  - Metrics: mIoU, mAP@50, mAP@75, COCO AP
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from backend_registry import (
    BACKEND_CHOICES,
    bbox_parse_mode,
    resolve_hf_token,
    resolve_model_id,
    resolve_output_model_name,
)
from backends import build_vlm_user_prompt, load_backend
from cholect50_data import infer_pil_side
from endovis17_data import (
    DEFAULT_DATASET_ROOT,
    ENDOVIS2017_IMAGE_SIZE,
    build_instrument_localization_prompt,
    collect_localization_samples,
    export_bbox_annotations,
    instrument_display_name,
    list_val_splits,
    sample_localization_items,
)
from utils import (
    clamp_bbox_xyxy_01,
    compute_detection_map_metrics,
    draw_gt_pred_comparison_visualization,
    draw_single_bbox_visualization,
    iou_xyxy,
    load_results_for_resume,
    parse_bbox_from_model_text,
    resolve_device,
    upsert_result,
)

_SCRIPT_ROOT = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_ROOT.parent
DEFAULT_OUTPUT_ROOT = _SCRIPT_ROOT / "outputs" / "instrument_localization_endovis17"
_DEFAULT_HF_TOKEN = _SCRIPT_ROOT / ".hf_token"

def resize_image_for_vlm(image: Image.Image, pil_side: int) -> Image.Image:
    """Square resize — same tensor geometry as VLM inference and viz overlays."""
    side = max(1, int(pil_side))
    return image.convert("RGB").resize(
        (side, side),
        resample=Image.Resampling.BICUBIC,
    )


def _viz_slug(val_split: str, frame_stem: str, instrument_id: str) -> str:
    split = re.sub(r"[^\w.-]+", "_", val_split.strip().lower())
    inst = re.sub(r"[^\w.-]+", "_", instrument_id.strip().lower())
    return f"{split}_{frame_stem}_{inst}"


def _norm_bbox_from_field(bbox: Any) -> tuple[float, float, float, float] | None:
    if not isinstance(bbox, list) or len(bbox) != 4:
        return None
    try:
        t = tuple(float(x) for x in bbox)
    except (TypeError, ValueError):
        return None
    return clamp_bbox_xyxy_01(t) or t


def save_localization_visualizations(
    entry: dict[str, Any],
    image: Image.Image,
    viz_root: Path,
    *,
    force: bool = False,
) -> dict[str, str | None]:
    """Write GT / Pred / comparison JPEGs; return paths stored on entry."""
    paths: dict[str, str | None] = {
        "visualization_gt_path": None,
        "visualization_pred_path": None,
        "visualization_comparison_path": None,
    }
    inp = entry.get("input") or {}
    lc = inp.get("label_context") or {}
    gt = _norm_bbox_from_field(lc.get("label_bbox_xyxy_norm"))
    if gt is None:
        return paths

    out = entry.get("output") or {}
    parsed = out.get("parsed") or {} if isinstance(out, dict) else {}
    pred = _norm_bbox_from_field(parsed.get("bbox_xyxy_norm"))

    frame_stem = str(lc.get("frame_stem") or inp.get("frame_stem") or "frame")
    val_split = str(lc.get("val_split") or inp.get("val_split") or "")
    instrument_id = str(lc.get("label_instrument_id") or "")
    slug = _viz_slug(val_split, frame_stem, instrument_id)

    ev = entry.get("evaluation") or {}
    iou_v = ev.get("iou")
    if iou_v is None and pred is not None:
        iou_v = iou_xyxy(gt, pred)

    inst_disp = (
        str(lc.get("label_instrument_display") or "").strip()
        or instrument_display_name(instrument_id)
    )
    cmp_dir = viz_root / "comparison"
    gt_dir = viz_root / "gt"
    pred_dir = viz_root / "pred"
    for d in (cmp_dir, gt_dir, pred_dir):
        d.mkdir(parents=True, exist_ok=True)

    gt_path = gt_dir / f"{slug}_gt.jpg"
    cmp_path = cmp_dir / f"{slug}_gt_pred.jpg"
    pred_path = pred_dir / f"{slug}_pred.jpg"

    if force or not gt_path.is_file():
        draw_single_bbox_visualization(
            image,
            gt,
            instrument=inst_disp,
            instrument_id=instrument_id,
            outline="#00ff00",
            panel_bg=(0, 80, 0),
        ).save(gt_path, format="JPEG", quality=95)
    paths["visualization_gt_path"] = str(gt_path.resolve())

    if pred is not None and (force or not pred_path.is_file()):
        draw_single_bbox_visualization(
            image,
            pred,
            instrument=inst_disp,
            instrument_id=instrument_id,
            outline="#ff0000",
            panel_bg=(80, 0, 0),
        ).save(pred_path, format="JPEG", quality=95)
        paths["visualization_pred_path"] = str(pred_path.resolve())

    if force or not cmp_path.is_file():
        draw_gt_pred_comparison_visualization(
            image,
            gt_bbox=gt,
            pred_bbox=pred,
            instrument=inst_disp,
            instrument_id=instrument_id,
            iou_value=float(iou_v) if iou_v is not None else None,
        ).save(cmp_path, format="JPEG", quality=95)
    paths["visualization_comparison_path"] = str(cmp_path.resolve())

    return paths


def _attach_visualization_paths(entry: dict[str, Any], paths: dict[str, str | None]) -> None:
    for key, val in paths.items():
        if val:
            entry[key] = val
    out = entry.get("output")
    if isinstance(out, dict):
        for key, val in paths.items():
            if val:
                out[key] = val


def _row_key(sample: dict[str, Any]) -> tuple[str, str]:
    tool = (
        f"endovis2017-loc|{sample['val_split']}|"
        f"{sample['frame_stem']}|{sample['instrument_id']}"
    )
    return str(sample["img_path"]), tool


def _should_skip_resume(rec: dict, tool: str) -> bool:
    if rec.get("error"):
        return False
    inp = rec.get("input") or {}
    if inp.get("tool") != tool:
        return False
    out = rec.get("output")
    if not isinstance(out, dict):
        return False
    parsed = out.get("parsed") or {}
    if parsed.get("not_present"):
        return True
    bbox = parsed.get("bbox_xyxy_norm")
    return isinstance(bbox, list) and len(bbox) == 4


def parse_localization_response(
    text: str,
    *,
    bbox_parse_mode: str,
    image_width: int,
    image_height: int,
) -> dict[str, Any]:
    raw = (text or "").strip()
    if re.search(r"\bnot\s+present\b", raw, re.IGNORECASE):
        return {"not_present": True, "bbox_xyxy_norm": None, "bbox_xyxy_px": None, "raw": raw}

    bbox_norm = parse_bbox_from_model_text(
        raw,
        bbox_parse_mode=bbox_parse_mode,
        image_width=image_width,
        image_height=image_height,
    )
    if bbox_norm is None:
        return {"not_present": False, "bbox_xyxy_norm": None, "bbox_xyxy_px": None, "raw": raw}

    if max(abs(v) for v in bbox_norm) <= 1.0:
        bbox_norm = clamp_bbox_xyxy_01(bbox_norm) or bbox_norm
    w, h = max(1, image_width), max(1, image_height)
    xmin, ymin, xmax, ymax = bbox_norm
    bbox_px = (xmin * w, ymin * h, xmax * w, ymax * h)
    return {
        "not_present": False,
        "bbox_xyxy_norm": list(bbox_norm),
        "bbox_xyxy_px": [float(v) for v in bbox_px],
        "raw": raw,
    }


def _score_record(rec: dict[str, Any]) -> dict[str, Any] | None:
    inp = rec.get("input") or {}
    lc = inp.get("label_context") or {}
    out = rec.get("output")
    if not isinstance(out, dict):
        return None
    parsed = out.get("parsed") or {}
    gt = lc.get("label_bbox_xyxy_norm")
    pred = parsed.get("bbox_xyxy_norm")
    if not (isinstance(gt, list) and len(gt) == 4):
        return None
    iou_v: float | None = None
    if isinstance(pred, list) and len(pred) == 4:
        iou_v = iou_xyxy(
            tuple(float(x) for x in gt),
            tuple(float(x) for x in pred),
        )
    return {
        "instrument_id": lc.get("label_instrument_id"),
        "region_id": lc.get("label_region_id"),
        "gt_bbox_norm": gt,
        "pred_bbox_norm": pred,
        "iou": iou_v,
        "not_present_pred": bool(parsed.get("not_present")),
    }


def aggregate_metrics(results: list[dict[str, Any]]) -> dict[str, Any]:
    det_records: list[dict[str, Any]] = []
    ious: list[float] = []
    parsed_ok = 0
    not_present = 0

    for rec in results:
        ev = rec.get("evaluation")
        if not ev:
            ev = _score_record(rec)
            if ev:
                rec["evaluation"] = ev
        if not ev:
            continue
        if ev.get("not_present_pred"):
            not_present += 1
            continue
        pred = ev.get("pred_bbox_norm")
        gt = ev.get("gt_bbox_norm")
        if not (isinstance(pred, list) and len(pred) == 4):
            continue
        parsed_ok += 1
        inp = rec.get("input") or {}
        lc = inp.get("label_context") or {}
        iou_v = ev.get("iou")
        if iou_v is not None:
            ious.append(float(iou_v))
        det_records.append(
            {
                "instrument_id": ev.get("instrument_id") or lc.get("label_instrument_id"),
                "image_id": lc.get("frame_stem") or inp.get("frame_stem"),
                "frame_stem": lc.get("frame_stem") or inp.get("frame_stem"),
                "gt_bbox_norm": gt,
                "pred_bbox_norm": pred,
                "score": 1.0,
            }
        )

    det = compute_detection_map_metrics(det_records)
    n_scored = len(det_records)
    return {
        "protocol": "endovis2017_instrument_localization",
        "n_results": len(results),
        "n_scored": n_scored,
        "n_parsed_bbox": parsed_ok,
        "n_not_present": not_present,
        "mIoU": det.get("mIoU"),
        "mAP@50": det.get("mAP@50"),
        "mAP@75": det.get("mAP@75"),
        "COCO_AP": det.get("COCO_AP"),
        "per_class_ap": det.get("per_class_ap"),
        "mean_iou_all_parsed": (sum(ious) / len(ious)) if ious else None,
        "detection_detail": det,
    }


def _run_vlm_on_sample(
    *,
    backend,
    pil_side: int,
    image: Image.Image,
    user_prompt: str,
    sample: dict[str, Any],
    args: argparse.Namespace,
    bbox_parse: str,
) -> dict[str, Any]:
    try:
        _ = pil_side  # caller must pass image already resized to (pil_side, pil_side)
        image = image.convert("RGB")
        gen_kw: dict[str, Any] = {"do_sample": args.do_sample, "min_length": 1}
        if args.do_sample:
            gen_kw["temperature"] = args.temperature

        prompt_text = build_vlm_user_prompt(backend, user_prompt)
        text = backend.generate(
            image,
            prompt_text,
            **{**gen_kw, "max_new_tokens": args.max_new_tokens},
        )
        parsed = parse_localization_response(
            text,
            bbox_parse_mode=bbox_parse,
            image_width=int(sample["image_width"]),
            image_height=int(sample["image_height"]),
        )
        return {"text": text, "parsed": parsed}
    except Exception as e:
        return {"error": str(e)}


def _make_result_entry(
    *,
    sample: dict[str, Any],
    user_prompt: str,
    args: argparse.Namespace,
    frame_output: dict[str, Any] | None,
) -> dict[str, Any]:
    path_str, tool = _row_key(sample)
    entry: dict[str, Any] = {
        "input": {
            "image_path": path_str,
            "tool": tool,
            "frame_stem": sample["frame_stem"],
            "label_context": {
                "val_split": sample["val_split"],
                "frame_stem": sample["frame_stem"],
                "label_instrument_id": sample["instrument_id"],
                "label_instrument_display": sample["instrument_display"],
                "mask_class_id": sample["mask_class_id"],
                "label_mask_path": str(sample.get("label_mask_path", "")),
                "label_bbox_xyxy_px": sample["bbox_xyxy_px"],
                "label_bbox_xyxy_norm": sample["bbox_xyxy_norm"],
                "mask_pixel_count": sample.get("mask_pixel_count"),
            },
            "val_split": sample["val_split"],
            "eval_protocol": "endovis2017_instrument_localization",
            "user_prompt": user_prompt,
            "image_width": sample["image_width"],
            "image_height": sample["image_height"],
        },
        "output": None,
    }
    if frame_output is None:
        return entry
    if frame_output.get("error"):
        entry["error"] = frame_output["error"]
        return entry
    entry["output"] = {
        "text": frame_output.get("text"),
        "parsed": frame_output.get("parsed"),
    }
    return entry


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="EndoVis 2017 val instrument localization (mask-derived bbox GT).",
    )
    p.add_argument("--backend", choices=BACKEND_CHOICES, default="prismatic")
    p.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    p.add_argument(
        "--val-split",
        action="append",
        default=None,
        help="Val folder name (repeatable), e.g. val1. Default: all val*.",
    )
    p.add_argument("--instrument", type=str, default=None, help="Filter by instrument id slug.")
    p.add_argument("--frame", type=str, default=None, help="Single frame stem, e.g. seq_1_frame225.")
    p.add_argument("--min-mask-pixels", type=int, default=1, help="Min mask pixels for GT bbox.")
    p.add_argument(
        "--export-annotations",
        type=Path,
        default=None,
        help="Optional path to write mask-derived bbox JSON and exit (no VLM).",
    )
    p.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Random subsample cap (debug). Omit for full val set.",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--model-id", type=str, default=None)
    p.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Output folder slug (default: size alias, e.g. cosmos-reason2-2b, qwen3-vl-32b).",
    )
    p.add_argument("--vlm-checkpoint", type=Path, default=None)
    p.add_argument("--vlm-config", type=Path, default=None)
    p.add_argument("--hf-token", type=Path, default=_DEFAULT_HF_TOKEN)
    p.add_argument("--api-key-file", type=Path, default=None)
    p.add_argument("--api-timeout-sec", type=int, default=120)
    p.add_argument("--device", type=str, default="0")
    p.add_argument("--do-sample", action="store_true")
    p.add_argument("--temperature", type=float, default=0.4)
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--force", action="store_true")
    p.add_argument(
        "--viz",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save GT/Pred/comparison JPEG overlays (default: on).",
    )
    p.add_argument(
        "--viz-only",
        action="store_true",
        help="Skip VLM; (re)build visualizations from existing results JSON (--output).",
    )
    p.add_argument(
        "--viz-side",
        type=int,
        default=None,
        help="Viz/VLM square side (default: backend image_size or 384). Used for --viz-only.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    val_splits = args.val_split or list_val_splits(dataset_root)

    samples = collect_localization_samples(
        dataset_root=dataset_root,
        val_splits=val_splits,
        instrument_filter=args.instrument,
        frame_stem_filter=args.frame,
        min_mask_pixels=args.min_mask_pixels,
    )
    if not samples:
        raise RuntimeError(
            f"No localization samples under {dataset_root} ({val_splits}). "
            "Check val*/image and val*/label."
        )

    if args.export_annotations is not None:
        export_bbox_annotations(
            samples,
            args.export_annotations.resolve(),
            dataset_root=dataset_root,
        )
        print(
            f"Exported {len(samples)} mask-derived bbox samples to {args.export_annotations}",
            file=sys.stderr,
        )
        return

    samples = sample_localization_items(
        samples, cap=args.max_samples, seed=args.seed,
    )

    model_id = resolve_model_id(args.backend, args.model_id)
    model_name = resolve_output_model_name(args.backend, model_id, args.model_name)
    out_root = args.output_root.resolve()
    out_path = (
        args.output.resolve()
        if args.output is not None
        else (
            out_root
            / f"loc_{args.backend}_{model_name}"
            / "endovis2017_instrument_localization.json"
        ).resolve()
    )
    viz_root = out_path.parent / "visualizations"

    def _resolve_pil_side(payload: dict[str, Any] | None = None) -> int:
        if args.viz_side is not None:
            return max(1, int(args.viz_side))
        if payload is not None and payload.get("vlm_input_side") is not None:
            return max(1, int(payload["vlm_input_side"]))
        return max(1, int(infer_pil_side(args)))

    if args.viz_only:
        if not out_path.is_file():
            raise FileNotFoundError(f"--viz-only requires existing results: {out_path}")
        with out_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        results = payload.get("results") or []
        pil_side = _resolve_pil_side(payload)
        n_viz = 0
        for rec in results:
            inp = rec.get("input") or {}
            img_path = inp.get("image_path")
            if not img_path:
                continue
            try:
                vlm_image = resize_image_for_vlm(
                    Image.open(img_path), pil_side,
                )
            except Exception as e:
                print(f"SKIP viz {img_path}: {e}", file=sys.stderr)
                continue
            if "evaluation" not in rec:
                ev = _score_record(rec)
                if ev:
                    rec["evaluation"] = ev
            paths = save_localization_visualizations(
                rec, vlm_image, viz_root, force=args.force,
            )
            _attach_visualization_paths(rec, paths)
            n_viz += 1
        payload["visualization_root"] = str(viz_root.resolve())
        payload["visualization_count"] = n_viz
        payload["vlm_input_side"] = pil_side
        payload["visualization_image_size"] = [pil_side, pil_side]
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(
            f"Wrote {n_viz} visualizations under {viz_root} "
            f"({pil_side}x{pil_side}, VLM resize space)",
            file=sys.stderr,
        )
        return

    print(
        f"EndoVis 2017 localization: samples={len(samples)}, "
        f"frames={len({s['frame_stem'] for s in samples})}, "
        f"splits={sorted({s['val_split'] for s in samples})}, "
        f"image={ENDOVIS2017_IMAGE_SIZE}x{ENDOVIS2017_IMAGE_SIZE} (native BMP).",
        file=sys.stderr,
    )

    hf_token = resolve_hf_token(args.backend, args.hf_token)
    device = resolve_device(args.device)

    backend, meta = load_backend(
        args.backend,
        model_id=model_id,
        hf_token=hf_token,
        api_key_file=args.api_key_file,
        vlm_checkpoint=args.vlm_checkpoint,
        vlm_config=args.vlm_config,
        device=device,
        api_timeout_sec=args.api_timeout_sec,
    )
    bbox_parse = bbox_parse_mode(args.backend, meta)
    backend.to(device, dtype=torch.bfloat16)
    pil_side = (
        _resolve_pil_side()
        if args.viz_side is not None
        else (getattr(backend, "image_size", None) or infer_pil_side(args))
    )
    pil_side = max(1, int(pil_side))

    results, key_to_idx = load_results_for_resume(out_path)
    vlm_calls = 0
    prompt_cache: dict[str, str] = {}

    def _upsert_scored(
        sample: dict[str, Any],
        frame_output: dict[str, Any] | None,
        *,
        vlm_image: Image.Image | None = None,
    ) -> dict[str, Any]:
        inst = sample["instrument_id"]
        if inst not in prompt_cache:
            prompt_cache[inst] = build_instrument_localization_prompt(
                instrument_id=inst,
            )
        user_prompt = prompt_cache[inst]
        row_key = _row_key(sample)
        entry = _make_result_entry(
            sample=sample,
            user_prompt=user_prompt,
            args=args,
            frame_output=frame_output,
        )
        ev = _score_record(entry)
        if ev:
            entry["evaluation"] = ev
        if args.viz and vlm_image is not None:
            paths = save_localization_visualizations(
                entry, vlm_image, viz_root, force=args.force,
            )
            _attach_visualization_paths(entry, paths)
        upsert_result(results, key_to_idx, row_key, entry)
        return entry

    for sample in samples:
        row_key = _row_key(sample)
        path_str, tool = row_key

        if (
            not args.force
            and row_key in key_to_idx
            and _should_skip_resume(results[key_to_idx[row_key]], tool)
        ):
            rec = results[key_to_idx[row_key]]
            ev = _score_record(rec)
            if ev:
                rec["evaluation"] = ev
            if args.viz:
                try:
                    vlm_image = resize_image_for_vlm(
                        Image.open(sample["img_path"]), pil_side,
                    )
                    paths = save_localization_visualizations(
                        rec, vlm_image, viz_root, force=args.force,
                    )
                    _attach_visualization_paths(rec, paths)
                    upsert_result(results, key_to_idx, row_key, rec)
                except Exception as e:
                    print(
                        f"WARN viz {sample['frame_stem']}: {e}",
                        file=sys.stderr,
                    )
            continue

        try:
            vlm_image = resize_image_for_vlm(
                Image.open(sample["img_path"]), pil_side,
            )
        except Exception as e:
            print(f"SKIP {sample['frame_stem']}: {e}", file=sys.stderr)
            _upsert_scored(sample, {"error": str(e)})
            continue

        inst = sample["instrument_id"]
        if inst not in prompt_cache:
            prompt_cache[inst] = build_instrument_localization_prompt(
                instrument_id=inst,
            )

        frame_output = _run_vlm_on_sample(
            backend=backend,
            pil_side=pil_side,
            image=vlm_image,
            user_prompt=prompt_cache[inst],
            sample=sample,
            args=args,
            bbox_parse=bbox_parse,
        )
        vlm_calls += 1
        _upsert_scored(sample, frame_output, vlm_image=vlm_image)

        if vlm_calls % 25 == 0:
            print(f"  ... {vlm_calls} VLM calls", file=sys.stderr)

    print(f"VLM forward passes this run: {vlm_calls}", file=sys.stderr)

    for rec in results:
        if "evaluation" not in rec:
            ev = _score_record(rec)
            if ev:
                rec["evaluation"] = ev

    metrics = aggregate_metrics(results)
    example_prompt = build_instrument_localization_prompt(
        instrument_id="large_needle_driver",
    )
    payload = {
        "task": "instrument_localization",
        "eval_protocol": "endovis2017_instrument_localization",
        "dataset": "endovis2017",
        "dataset_root": str(dataset_root),
        "val_splits": sorted({s["val_split"] for s in samples}),
        "gt_source": "mask label -> tight bbox per instrument class",
        "native_image_size": [ENDOVIS2017_IMAGE_SIZE, ENDOVIS2017_IMAGE_SIZE],
        "vlm_input_side": pil_side,
        "visualization_image_size": [pil_side, pil_side],
        "backend": args.backend,
        "model_id": meta.get("model_id") if meta.get("source") == "local_checkpoint" else model_id,
        "hub_model_id_cli": model_id,
        "model_name": model_name,
        "vlm_load": meta,
        "user_prompt_template_example": example_prompt,
        "bbox_output_format": "[x_min, y_min, x_max, y_max]",
        "metrics_description": {
            "mIoU": "Mean IoU between predicted and GT boxes (normalized xyxy).",
            "mAP@50": "Mean AP across instrument classes at IoU=0.5.",
            "mAP@75": "Mean AP across instrument classes at IoU=0.75.",
            "COCO_AP": "Mean of mAP at IoU thresholds 0.5:0.05:0.95.",
        },
        "vlm_forward_passes": vlm_calls,
        "visualization_root": str(viz_root.resolve()) if args.viz else None,
        "visualization_layout": {
            "note": "Overlays on VLM square resize (pil_side), normalized bbox coords.",
            "gt": "visualizations/gt/{split}_{frame}_{instrument}_gt.jpg",
            "pred": "visualizations/pred/{split}_{frame}_{instrument}_pred.jpg",
            "comparison": "visualizations/comparison/{split}_{frame}_{instrument}_gt_pred.jpg",
        },
        "metrics": metrics,
        "count": len(results),
        "results": results,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    m = metrics
    print(
        f"Wrote {len(results)} entries to {out_path}\n"
        f"  mIoU={m.get('mIoU')}\n"
        f"  mAP@50={m.get('mAP@50')}  mAP@75={m.get('mAP@75')}  COCO_AP={m.get('COCO_AP')}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
