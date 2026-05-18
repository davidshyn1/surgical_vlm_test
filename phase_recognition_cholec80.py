"""
phase_recognition_cholec80.py

Cholec80 surgical phase recognition (7 classes).

  - Prompt (bench): "In the Cholecystectomy surgical image, what is the current Phase?
    The available phase options are ..."
  - Default split: evaluation videos 41–80 (EndoNet test set)
  - Metrics: Accuracy, per-class Recall / Precision / Jaccard, macro averages
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

from backends import load_backend
from cholec80_data import (
    CHOLEC80_EVAL_FPS,
    CHOLEC80_EVAL_FRAME_STRIDE,
    CHOLEC80_EVAL_DATA_RELPATH,
    CHOLEC80_EVAL_FRAMES_RELPATH,
    CANONICAL_TO_DISPLAY,
    package_eval_data_root,
    package_eval_frames_root,
    PHASE_CANONICAL_IDS,
    PHASE_DISPLAY_NAMES,
    _DEFAULT_MODEL_IDS,
    collect_phase_samples,
    ffmpeg_available,
    iter_samples_by_video,
    load_frame_rgb,
    normalize_phase_label,
    parse_video_id,
    resolve_cholec80_root,
    resolve_eval_frames_root,
    video_in_split,
)
from cholect50_data import infer_pil_side
from utils import load_results_for_resume, resolve_device, upsert_result

_SCRIPT_ROOT = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_ROOT.parent
DEFAULT_DATASET_ROOT = _REPO_ROOT / "data" / "Cholec80"
DEFAULT_OUTPUT_ROOT = _SCRIPT_ROOT / "outputs" / "phase_recognition_cholec80"
DEFAULT_FRAMES_ROOT = package_eval_frames_root()
_DEFAULT_HF_TOKEN = _SCRIPT_ROOT / ".hf_token"


def _format_lettered_phases() -> tuple[str, dict[str, str]]:
    lines = ["The available phase options are:"]
    letter_map: dict[str, str] = {}
    name_map: dict[str, str] = {}
    for i, (cid, disp) in enumerate(zip(PHASE_CANONICAL_IDS, PHASE_DISPLAY_NAMES, strict=True)):
        letter = chr(ord("A") + i)
        lines.append(f"{letter}. {disp}")
        letter_map[letter.upper()] = cid
        letter_map[letter.lower()] = cid
        key = re.sub(r"[^a-z0-9]+", "", disp.lower())
        name_map[key] = cid
        name_map[cid] = cid
        name_map[cid.replace("_", "")] = cid
    return "\n".join(lines), {**letter_map, **name_map}


def build_phase_recognition_prompt() -> tuple[str, dict[str, Any]]:
    options_block, option_map = _format_lettered_phases()
    body = (
        "In the Cholecystectomy surgical image, what is the current Phase?\n"
        f"{options_block}"
    )
    return body, {"option_map": option_map}


def wrap_vlm_prompt(body: str) -> str:
    return body.strip()


def _match_phase_token(token: str, option_map: dict[str, str]) -> str | None:
    t = (token or "").strip()
    if not t:
        return None
    if len(t) == 1 and t.upper() in option_map:
        return option_map[t.upper()]
    key = re.sub(r"[^a-z0-9]+", "", t.lower())
    if key in option_map:
        return option_map[key]
    norm = normalize_phase_label(t)
    if norm:
        return norm
    raw = t.lower()
    for k, cid in option_map.items():
        if len(k) <= 1:
            continue
        disp = CANONICAL_TO_DISPLAY.get(cid, "")
        if disp and disp.lower() in raw:
            return cid
        if k in raw:
            return cid
    return None


def parse_phase_response(text: str, *, option_map: dict[str, str]) -> dict[str, Any]:
    raw = (text or "").strip()
    phase_id: str | None = None

    m = re.search(
        r"phase\s*[:=]\s*([^\n.;]+)",
        raw,
        re.IGNORECASE,
    )
    if m:
        phase_id = _match_phase_token(m.group(1), option_map)

    if phase_id is None:
        for line in raw.splitlines():
            s = line.strip()
            s = re.sub(r"^[-*•]\s*", "", s)
            s = re.sub(r"^\d+[.)]\s*", "", s).strip()
            hit = _match_phase_token(s, option_map)
            if hit:
                phase_id = hit
                break

    if phase_id is None:
        hit = _match_phase_token(raw, option_map)
        if hit:
            phase_id = hit

    if phase_id is None:
        for cid in PHASE_CANONICAL_IDS:
            disp = CANONICAL_TO_DISPLAY[cid]
            if disp.lower() in raw.lower():
                phase_id = cid
                break

    return {
        "phase_id": phase_id,
        "phase_display": CANONICAL_TO_DISPLAY.get(phase_id, "") if phase_id else None,
        "raw": raw,
    }


def _row_key(sample: dict[str, Any]) -> tuple[str, str]:
    tool = f"cholec80-phase|{sample['vid']}|f{int(sample['frame_index']):06d}"
    return str(sample["video_path"]), tool


def _score_record(rec: dict[str, Any]) -> dict[str, Any] | None:
    inp = rec.get("input") or {}
    lc = inp.get("label_context") or {}
    out = rec.get("output")
    if not isinstance(out, dict):
        return None
    parsed = out.get("parsed") or {}
    gold = str(lc.get("label_phase_id") or "")
    pred = str(parsed.get("phase_id") or "")
    if not gold:
        return None
    return {
        "gold_phase_id": gold,
        "gold_phase_display": lc.get("label_phase_display"),
        "pred_phase_id": pred or None,
        "pred_phase_display": parsed.get("phase_display"),
        "correct": bool(pred and pred == gold),
    }


def aggregate_metrics(results: list[dict[str, Any]]) -> dict[str, Any]:
    evs = [r["evaluation"] for r in results if r.get("evaluation")]
    if not evs:
        return {"n_results": len(results), "n_scored": 0}

    y_true = [e["gold_phase_id"] for e in evs]
    y_pred = [e.get("pred_phase_id") or "__none__" for e in evs]

    n = len(evs)
    correct = sum(1 for e in evs if e.get("correct"))
    accuracy = correct / n if n else None

    per_class: dict[str, dict[str, Any]] = {}
    recalls: list[float] = []
    precisions: list[float] = []
    jaccards: list[float] = []

    for cid in PHASE_CANONICAL_IDS:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == cid and p == cid)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != cid and p == cid)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == cid and p != cid)
        support = sum(1 for t in y_true if t == cid)

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        jac = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0

        per_class[cid] = {
            "display_name": CANONICAL_TO_DISPLAY.get(cid, cid),
            "support": support,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": prec,
            "recall": rec,
            "jaccard": jac,
        }
        if support > 0:
            recalls.append(rec)
            precisions.append(prec)
            jaccards.append(jac)

    def _macro(vals: list[float]) -> float | None:
        return sum(vals) / len(vals) if vals else None

    return {
        "protocol": "cholec80_phase_recognition",
        "n_scored": n,
        "n_results": len(results),
        "accuracy": {
            "correct": correct,
            "total": n,
            "value": accuracy,
        },
        "macro_recall": _macro(recalls),
        "macro_precision": _macro(precisions),
        "macro_jaccard": _macro(jaccards),
        "per_class": per_class,
    }


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
    return bool(parsed.get("phase_id"))


def _run_vlm_on_frame(
    *,
    backend,
    pil_side: int,
    image: Image.Image,
    user_prompt: str,
    option_map: dict[str, str],
    args: argparse.Namespace,
) -> dict[str, Any]:
    try:
        image = image.convert("RGB").resize(
            (pil_side, pil_side),
            resample=Image.Resampling.BICUBIC,
        )
        gen_kw: dict[str, Any] = {"do_sample": args.do_sample, "min_length": 1}
        if args.do_sample:
            gen_kw["temperature"] = args.temperature

        pb = backend.get_prompt_builder()
        pb.add_turn(role="human", message=wrap_vlm_prompt(user_prompt))
        text = backend.generate(
            image,
            pb.get_prompt(),
            **{**gen_kw, "max_new_tokens": args.max_new_tokens},
        )
        parsed = parse_phase_response(text, option_map=option_map)
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
            "frame_index": sample["frame_index"],
            "label_context": {
                "label_phase_id": sample["phase_id"],
                "label_phase_display": sample["phase_display"],
                "vid": sample["vid"],
                "vid_num": sample["vid_num"],
            },
            "eval_protocol": "cholec80_phase_recognition",
            "user_prompt": user_prompt,
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
        description="Cholec80 phase recognition (eval videos 41–80 by default).",
    )
    p.add_argument("--backend", choices=("prismatic",
                   "cosmos", "groot"), default="prismatic")
    p.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    p.add_argument(
        "--split",
        choices=("eval", "train", "all"),
        default="eval",
        help="eval = video41–80 (default); train = video01–40; all = 01–80.",
    )
    p.add_argument("--video", type=str, default=None,
                   help="Single video, e.g. 41 or video41.")
    p.add_argument(
        "--frame-stride",
        type=int,
        default=None,
        help=(
            f"Subsample annotated frames (native 25 fps phase uses default "
            f"stride {CHOLEC80_EVAL_FRAME_STRIDE} ≈ {CHOLEC80_EVAL_FPS} fps). "
            "With --frames-root and videoNN/videoNN-phase.txt manifests, stride=1."
        ),
    )
    p.add_argument(
        "--max-frames-per-video",
        type=int,
        default=None,
        help="Cap frames per video (debug / smoke test).",
    )
    p.add_argument(
        "--frames-root",
        type=Path,
        default=DEFAULT_FRAMES_ROOT,
        help=(
            "Pre-extracted 0.1 fps frames + videoNN-phase.txt manifests "
            f"(default: {CHOLEC80_EVAL_FRAMES_RELPATH})."
        ),
    )
    p.add_argument(
        "--frame-reader",
        choices=("auto", "ffmpeg", "opencv"),
        default="auto",
        help="MP4 decode: auto=ffmpeg if on PATH else opencv (default: auto).",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--model-id", type=str, default=None)
    p.add_argument("--model-name", type=str, default="original")
    p.add_argument("--vlm-checkpoint", type=Path, default=None)
    p.add_argument("--vlm-config", type=Path, default=None)
    p.add_argument("--hf-token", type=Path, default=_DEFAULT_HF_TOKEN)
    p.add_argument("--device", type=str, default="0")
    p.add_argument("--do-sample", action="store_true")
    p.add_argument("--temperature", type=float, default=0.4)
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = resolve_cholec80_root(args.dataset_root)
    args.dataset_root = dataset_root

    video_filter: int | None = None
    if args.video:
        video_filter = parse_video_id(args.video)
        if video_filter is None:
            raise ValueError(f"Could not parse --video {args.video!r}")

    if video_filter is not None and not video_in_split(video_filter, args.split):
        print(
            f"WARN: video {video_filter} is outside split={args.split!r}; running anyway.",
            file=sys.stderr,
        )

    user_prompt, prompt_meta = build_phase_recognition_prompt()
    option_map = prompt_meta.get("option_map") or {}

    frames_root = resolve_eval_frames_root(
        args.frames_root,
        dataset_root=dataset_root,
        required=True,
    )

    frame_stride_arg = (
        max(1, int(args.frame_stride)) if args.frame_stride is not None else None
    )

    samples = collect_phase_samples(
        dataset_root,
        split=args.split,
        video_filter=video_filter,
        frame_stride=frame_stride_arg,
        max_frames_per_video=args.max_frames_per_video,
        frames_root=frames_root,
    )
    if not samples:
        raise RuntimeError(
            "No phase samples found. Check --frames-root (eval/cholec80/frames_0p1fps), "
            "--split, and --video."
        )

    n_with_img = sum(1 for s in samples if s.get("img_path"))
    if n_with_img < len(samples):
        raise RuntimeError(
            f"Missing PNGs for {len(samples) - n_with_img}/{len(samples)} samples under "
            f"{frames_root}. Re-run scripts/extract_cholec80_frames.sh."
        )

    model_name = re.sub(r"[^a-zA-Z0-9._-]+", "_",
                        (args.model_name or "original").strip() or "original")
    out_root = args.output_root.resolve()
    split_slug = args.split
    stride_label = (
        str(frame_stride_arg)
        if frame_stride_arg is not None
        else (
            "0p1fps_manifest"
            if samples and samples[0].get("phase_manifest_eval")
            else str(CHOLEC80_EVAL_FRAME_STRIDE)
        )
    )
    out_path = (
        args.output.resolve()
        if args.output is not None
        else (
            out_root
            / f"phase_{args.backend}_{model_name}_{split_slug}"
            / f"cholec80_phase_{stride_label}.json"
        ).resolve()
    )

    eval_data_root = package_eval_data_root()
    reader_note = (
        f"frames_root={frames_root} ({n_with_img}/{len(samples)} on disk)"
        if frames_root
        else f"frame_reader={args.frame_reader}, ffmpeg={ffmpeg_available()}"
    )
    print(
        f"Cholec80 phase recognition: split={args.split}, "
        f"videos={len({s['vid_num'] for s in samples})}, "
        f"frames={len(samples)}, stride={stride_label}, {reader_note}.",
        file=sys.stderr,
    )

    model_id = args.model_id or _DEFAULT_MODEL_IDS[args.backend]
    hf_token = args.hf_token.resolve().read_text(encoding="utf-8").strip()
    device = resolve_device(args.device)

    backend, meta = load_backend(
        args.backend,
        model_id=model_id,
        hf_token=hf_token,
        vlm_checkpoint=args.vlm_checkpoint,
        vlm_config=args.vlm_config,
        device=device,
    )
    backend.to(device, dtype=torch.bfloat16)
    pil_side = getattr(backend, "image_size", None) or infer_pil_side(args)

    results, key_to_idx = load_results_for_resume(out_path)
    vlm_calls = 0

    def _upsert_scored(sample: dict[str, Any], frame_output: dict[str, Any] | None) -> None:
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
        upsert_result(results, key_to_idx, row_key, entry)

    for video_path, video_samples in iter_samples_by_video(samples):
        for sample in video_samples:
            row_key = _row_key(sample)
            path_str, tool = row_key

            if (
                not args.force
                and row_key in key_to_idx
                and _should_skip_resume(results[key_to_idx[row_key]], tool)
            ):
                rec = results[key_to_idx[row_key]]
                inp = rec.setdefault("input", {})
                inp["label_context"] = {
                    "label_phase_id": sample["phase_id"],
                    "label_phase_display": sample["phase_display"],
                    "vid": sample["vid"],
                    "vid_num": sample["vid_num"],
                }
                ev = _score_record(rec)
                if ev:
                    rec["evaluation"] = ev
                continue

            try:
                pil_image = load_frame_rgb(
                    sample,
                    frame_reader=args.frame_reader,
                )
            except Exception as e:
                print(
                    f"SKIP {sample['vid']} f{sample['frame_index']}: {e}", file=sys.stderr)
                _upsert_scored(sample, {"error": str(e)})
                continue

            frame_output = _run_vlm_on_frame(
                backend=backend,
                pil_side=pil_side,
                image=pil_image,
                user_prompt=user_prompt,
                option_map=option_map,
                args=args,
            )
            vlm_calls += 1
            _upsert_scored(sample, frame_output)

            if vlm_calls % 50 == 0:
                print(f"  ... {vlm_calls} VLM calls", file=sys.stderr)

    print(f"VLM forward passes this run: {vlm_calls}", file=sys.stderr)

    for rec in results:
        if "evaluation" not in rec:
            ev = _score_record(rec)
            if ev:
                rec["evaluation"] = ev

    metrics = aggregate_metrics(results)
    payload = {
        "task": "phase_recognition",
        "eval_protocol": "cholec80_phase_recognition",
        "dataset": "cholec80",
        "dataset_root": str(dataset_root),
        "eval_data_root": str(eval_data_root),
        "eval_frames_relpath": str(CHOLEC80_EVAL_FRAMES_RELPATH),
        "split": args.split,
        "eval_video_range": "41-80" if args.split == "eval" else ("1-40" if args.split == "train" else "1-80"),
        "frame_stride": stride_label,
        "cholec80_eval_fps": CHOLEC80_EVAL_FPS,
        "cholec80_video_fps": 25,
        "frames_root": str(frames_root) if frames_root else None,
        "frame_reader": args.frame_reader,
        "backend": args.backend,
        "model_id": meta.get("model_id") if meta.get("source") == "local_checkpoint" else model_id,
        "hub_model_id_cli": model_id,
        "model_name": model_name,
        "vlm_load": meta,
        "user_prompt_template": user_prompt,
        "vlm_forward_passes": vlm_calls,
        "metrics": metrics,
        "count": len(results),
        "results": results,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    m = metrics
    acc = (m.get("accuracy") or {}).get("value")
    print(
        f"Wrote {len(results)} entries to {out_path}\n"
        f"  Accuracy={acc}\n"
        f"  Macro Recall={m.get('macro_recall')}  "
        f"Precision={m.get('macro_precision')}  "
        f"Jaccard={m.get('macro_jaccard')}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
