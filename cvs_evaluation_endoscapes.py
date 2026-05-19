"""
cvs_evaluation_endoscapes.py

Endoscapes2023 Critical View of Safety (CVS) — binary yes/no per criterion.

  --eval-protocol joint: one VLM call, three yes/no answers (C1, C2, C3)
  --eval-protocol per_criterion: one VLM call per criterion (MCQ: yes / no)

  GT: images[].ds (3-expert average); binarized with --gt-threshold (default 0.5)
  Metrics: Average Accuracy, Balanced Accuracy (mean of per-criterion BA)
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
    resolve_hf_token,
    resolve_model_id,
    resolve_output_model_name,
)
from backends import build_vlm_user_prompt, load_backend
from cholect50_data import infer_pil_side
from endoscapes_cvs_data import (
    CVS_CRITERIA,
    DEFAULT_DATASET_ROOT,
    YES_NO_OPTIONS,
    collect_cvs_samples,
)
from utils import load_results_for_resume, resolve_device, strip_lora_answer_tags, upsert_result

_SCRIPT_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_ROOT = _SCRIPT_ROOT / "outputs" / "cvs_evaluation_endoscapes"
_DEFAULT_HF_TOKEN = _SCRIPT_ROOT / ".hf_token"

EVAL_PROTOCOL_JOINT = "joint"
EVAL_PROTOCOL_PER_CRITERION = "per_criterion"
EVAL_PROTOCOLS = (EVAL_PROTOCOL_JOINT, EVAL_PROTOCOL_PER_CRITERION)


def _yes_no_option_map() -> dict[str, str]:
    return {
        "yes": "yes",
        "y": "yes",
        "true": "yes",
        "1": "yes",
        "no": "no",
        "n": "no",
        "false": "no",
        "0": "no",
    }


def parse_yes_no(text: str) -> str | None:
    raw = (text or "").strip().lower()
    if not raw:
        return None
    omap = _yes_no_option_map()
    if raw in omap:
        return omap[raw]
    first = re.sub(r"^[-*•]\s*", "", raw)
    first = re.sub(r"^(c\d+|criterion\s*\d+)\s*[:=]\s*", "", first, flags=re.I)
    first = re.sub(r"^\d+[.)]\s*", "", first).strip()
    if first in omap:
        return omap[first]
    for token in re.split(r"[\s,;]+", first):
        if token in omap:
            return omap[token]
    if "yes" in raw and "no" not in raw:
        return "yes"
    if "no" in raw and "yes" not in raw:
        return "no"
    return None


def yes_no_to_binary(label: str | None) -> int | None:
    if label == "yes":
        return 1
    if label == "no":
        return 0
    return None


def build_joint_cvs_prompt() -> str:
    lines = [
        "For this laparoscopic cholecystectomy image, answer yes or no for each "
        "Critical View of Safety criterion.",
        "",
    ]
    for crit in CVS_CRITERIA:
        lines.append(f"{crit.criterion_id}. {crit.question}")
    lines.extend([
        "",
        "Reply with exactly three lines, one answer per line (yes or no only).",
        "Use this format:",
        "C1: yes",
        "C2: no",
        "C3: no",
    ])
    return "\n".join(lines)


def build_criterion_prompt(question: str) -> str:
    opts = ", ".join(YES_NO_OPTIONS)
    return (
        f"{question.strip()}\n\n"
        "Answer with exactly one keyword: yes or no. "
        "Reply with the keyword only — no extra words.\n\n"
        f"Options: {opts}"
    )


def parse_joint_cvs_response(text: str) -> dict[str, Any]:
    raw = strip_lora_answer_tags(text)
    labels: list[str | None] = [None, None, None]

    for line in raw.splitlines():
        s = line.strip()
        if not s:
            continue
        m = re.match(r"^(C[123])\s*[:=]\s*(.+)$", s, re.I)
        if m:
            idx = {"C1": 0, "C2": 1, "C3": 2}[m.group(1).upper()]
            labels[idx] = parse_yes_no(m.group(2))
            continue
        m = re.match(r"^(\d+)\s*[:=.)]\s*(.+)$", s)
        if m:
            num = int(m.group(1))
            if 1 <= num <= 3:
                labels[num - 1] = parse_yes_no(m.group(2))

    # Fallback: collect yes/no tokens in order
    if any(x is None for x in labels):
        found: list[str] = []
        for line in raw.splitlines():
            hit = parse_yes_no(line)
            if hit:
                found.append(hit)
        for i in range(min(3, len(found))):
            if labels[i] is None:
                labels[i] = found[i]

    binary = [yes_no_to_binary(lb) for lb in labels]
    return {
        "labels": labels,
        "binary": binary,
        "raw": raw,
    }


def wrap_vlm_prompt(body: str) -> str:
    return body.strip()


def _row_key(sample: dict[str, Any], eval_protocol: str) -> tuple[str, str]:
    img = str(sample["img_path"])
    if eval_protocol == EVAL_PROTOCOL_PER_CRITERION:
        tool = f"endoscapes-cvs|{sample['image_id']}|{sample['criterion_id']}"
    else:
        tool = f"endoscapes-cvs|{sample['image_id']}"
    return img, tool


def _score_joint(rec: dict[str, Any]) -> dict[str, Any] | None:
    inp = rec.get("input") or {}
    lc = inp.get("label_context") or {}
    out = rec.get("output")
    if not isinstance(out, dict):
        return None
    parsed = out.get("parsed") or {}
    gold = lc.get("ds_binary")
    pred = parsed.get("binary")
    if not isinstance(gold, list) or len(gold) != 3:
        return None
    pred_bin = pred if isinstance(pred, list) and len(pred) == 3 else [None, None, None]
    per_crit: list[dict[str, Any]] = []
    for i, crit in enumerate(CVS_CRITERIA):
        pb = pred_bin[i] if i < len(pred_bin) else None
        per_crit.append({
            "criterion_id": crit.criterion_id,
            "gold_binary": int(gold[i]),
            "pred_binary": pb,
            "correct": pb is not None and int(pb) == int(gold[i]),
        })
    return {
        "ds_raw": lc.get("ds_raw"),
        "ds_binary_gold": gold,
        "ds_binary_pred": pred_bin,
        "per_criterion": per_crit,
        "n_correct": sum(1 for p in per_crit if p.get("correct")),
    }


def _score_per_criterion(rec: dict[str, Any]) -> dict[str, Any] | None:
    inp = rec.get("input") or {}
    lc = inp.get("label_context") or {}
    out = rec.get("output")
    if not isinstance(out, dict):
        return None
    parsed = out.get("parsed") or {}
    gold = lc.get("gold_binary")
    pred = parsed.get("binary")
    if gold is None:
        return None
    return {
        "criterion_id": lc.get("criterion_id"),
        "gold_binary": int(gold),
        "pred_binary": pred,
        "pred_label": parsed.get("label"),
        "correct": pred is not None and int(pred) == int(gold),
    }


def _flatten_criterion_scores(results: list[dict[str, Any]]) -> tuple[list[int], list[int]]:
    y_true: list[int] = []
    y_pred: list[int] = []
    for rec in results:
        ev = rec.get("evaluation")
        if not ev:
            continue
        if "per_criterion" in ev:
            for p in ev["per_criterion"]:
                pb = p.get("pred_binary")
                if pb is None:
                    continue
                y_true.append(int(p["gold_binary"]))
                y_pred.append(int(pb))
        elif ev.get("pred_binary") is not None:
            y_true.append(int(ev["gold_binary"]))
            y_pred.append(int(ev["pred_binary"]))
    return y_true, y_pred


def _balanced_accuracy_single(y_true: list[int], y_pred: list[int]) -> float | None:
    if not y_true:
        return None
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return (sens + spec) / 2.0


def aggregate_metrics(results: list[dict[str, Any]]) -> dict[str, Any]:
    y_true, y_pred = _flatten_criterion_scores(results)
    if not y_true:
        return {"n_results": len(results), "n_scored": 0}

    n = len(y_true)
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    average_accuracy = correct / n

    bas: list[float] = []
    by_crit: dict[str, dict[str, Any]] = {}
    for crit in CVS_CRITERIA:
        yt_c: list[int] = []
        yp_c: list[int] = []
        for rec in results:
            ev = rec.get("evaluation")
            if not ev:
                continue
            if "per_criterion" in ev:
                p = ev["per_criterion"][crit.index]
                pb = p.get("pred_binary")
                if pb is None:
                    continue
                yt_c.append(int(p["gold_binary"]))
                yp_c.append(int(pb))
            elif ev.get("criterion_id") == crit.criterion_id and ev.get("pred_binary") is not None:
                yt_c.append(int(ev["gold_binary"]))
                yp_c.append(int(ev["pred_binary"]))
        acc_c = (
            sum(1 for t, p in zip(yt_c, yp_c) if t == p) / len(yt_c) if yt_c else None
        )
        ba_c = _balanced_accuracy_single(yt_c, yp_c)
        by_crit[crit.criterion_id] = {
            "accuracy": acc_c,
            "balanced_accuracy": ba_c,
            "n_scored": len(yt_c),
        }
        if ba_c is not None:
            bas.append(ba_c)

    balanced_accuracy = sum(bas) / len(bas) if bas else None

    return {
        "protocol": "endoscapes_cvs_binary",
        "n_results": len(results),
        "n_scored_pairs": n,
        "average_accuracy": average_accuracy,
        "balanced_accuracy": balanced_accuracy,
        "per_criterion": by_crit,
    }


def _should_skip_resume(rec: dict, tool: str, eval_protocol: str) -> bool:
    if rec.get("error"):
        return False
    inp = rec.get("input") or {}
    if inp.get("tool") != tool:
        return False
    out = rec.get("output")
    if not isinstance(out, dict):
        return False
    parsed = out.get("parsed") or {}
    if eval_protocol == EVAL_PROTOCOL_JOINT:
        binary = parsed.get("binary")
        return isinstance(binary, list) and any(b is not None for b in binary)
    return parsed.get("binary") is not None


def _generate_vlm_text(
    *,
    backend,
    pil_side: int,
    image_path: Path,
    user_prompt: str,
    args: argparse.Namespace,
) -> str:
    image = Image.open(image_path).convert("RGB")
    image = image.resize((pil_side, pil_side), resample=Image.Resampling.BICUBIC)
    gen_kw: dict[str, Any] = {"do_sample": args.do_sample, "min_length": 1}
    if args.do_sample:
        gen_kw["temperature"] = args.temperature
    prompt_text = build_vlm_user_prompt(
        backend, user_prompt, wrap=wrap_vlm_prompt,
    )
    return backend.generate(
        image,
        prompt_text,
        **{**gen_kw, "max_new_tokens": args.max_new_tokens},
    )


def _run_joint(
    *,
    backend,
    pil_side: int,
    sample: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    prompt = build_joint_cvs_prompt()
    text = _generate_vlm_text(
        backend=backend,
        pil_side=pil_side,
        image_path=Path(sample["img_path"]),
        user_prompt=prompt,
        args=args,
    )
    parsed = parse_joint_cvs_response(text)
    return {"text": text, "parsed": parsed, "user_prompt": prompt}


def _run_per_criterion(
    *,
    backend,
    pil_side: int,
    sample: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    prompt = build_criterion_prompt(sample["criterion_question"])
    text = _generate_vlm_text(
        backend=backend,
        pil_side=pil_side,
        image_path=Path(sample["img_path"]),
        user_prompt=prompt,
        args=args,
    )
    label = parse_yes_no(text)
    binary = yes_no_to_binary(label)
    return {
        "text": text,
        "parsed": {"label": label, "binary": binary, "raw": text},
        "user_prompt": prompt,
    }


def _make_result_entry(
    *,
    sample: dict[str, Any],
    eval_protocol: str,
    user_prompt: str,
    frame_output: dict[str, Any] | None,
) -> dict[str, Any]:
    path_str, tool = _row_key(sample, eval_protocol)
    if eval_protocol == EVAL_PROTOCOL_PER_CRITERION:
        label_context = {
            "image_id": sample["image_id"],
            "file_name": sample["file_name"],
            "video_id": sample.get("video_id"),
            "criterion_id": sample["criterion_id"],
            "criterion_index": sample["criterion_index"],
            "gold_binary": sample["gold_binary"],
            "gold_label": sample["gold_label"],
            "ds_raw": sample["ds_raw"],
            "ds_binary": sample["ds_binary"],
        }
    else:
        label_context = {
            "image_id": sample["image_id"],
            "file_name": sample["file_name"],
            "video_id": sample.get("video_id"),
            "ds_raw": sample["ds_raw"],
            "ds_binary": sample["ds_binary"],
        }

    entry: dict[str, Any] = {
        "input": {
            "image_path": path_str,
            "tool": tool,
            "label_context": label_context,
            "eval_protocol": eval_protocol,
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
    p = argparse.ArgumentParser(description="Endoscapes CVS evaluation (binary yes/no).")
    p.add_argument("--backend", choices=BACKEND_CHOICES, default="prismatic")
    p.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    p.add_argument(
        "--split",
        choices=("train", "val", "test", "train_seg", "val_seg", "test_seg"),
        default="test",
    )
    p.add_argument(
        "--annotation-file",
        type=str,
        default=None,
        help="COCO JSON under split dir (default: annotation_coco.json then annotation_ds_coco.json).",
    )
    p.add_argument(
        "--eval-protocol",
        choices=EVAL_PROTOCOLS,
        default=EVAL_PROTOCOL_JOINT,
        help="joint: 1 VLM call with 3 answers; per_criterion: 3 calls (yes/no MCQ each).",
    )
    p.add_argument(
        "--gt-threshold",
        type=float,
        default=0.5,
        help="Binarize ds[k] >= threshold as yes (1).",
    )
    p.add_argument("--video", type=int, default=None, help="Filter by video_id.")
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--model-id", type=str, default=None)
    p.add_argument("--model-name", type=str, default=None)
    p.add_argument("--vlm-checkpoint", type=Path, default=None)
    p.add_argument("--vlm-config", type=Path, default=None)
    p.add_argument("--hf-token", type=Path, default=_DEFAULT_HF_TOKEN)
    p.add_argument("--api-key-file", type=Path, default=None)
    p.add_argument("--api-timeout-sec", type=int, default=120)
    p.add_argument("--device", type=str, default="0")
    p.add_argument("--do-sample", action="store_true")
    p.add_argument("--temperature", type=float, default=0.4)
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    samples = collect_cvs_samples(
        dataset_root,
        args.split,
        eval_protocol=args.eval_protocol,
        annotation_file=args.annotation_file,
        gt_threshold=args.gt_threshold,
        video_filter=args.video,
    )
    if not samples:
        raise RuntimeError(
            f"No CVS samples for split={args.split!r}. "
            "Check --dataset-root, --split, and annotation JSON with images[].ds."
        )
    if args.max_samples is not None:
        samples = samples[: max(0, int(args.max_samples))]

    n_frames = len({str(s["img_path"]) for s in samples})
    model_id = resolve_model_id(args.backend, args.model_id)
    model_name = resolve_output_model_name(args.backend, model_id, args.model_name)
    out_root = args.output_root.resolve()
    out_path = (
        args.output.resolve()
        if args.output is not None
        else (
            out_root
            / f"cvs_{args.backend}_{model_name}_{args.eval_protocol}_{args.split}"
            / f"endoscapes_cvs_{args.split}.json"
        ).resolve()
    )

    print(
        f"Endoscapes CVS: split={args.split}, protocol={args.eval_protocol}, "
        f"frames={n_frames}, samples={len(samples)}, gt_threshold={args.gt_threshold}.",
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
    backend.to(device, dtype=torch.bfloat16)
    pil_side = getattr(backend, "image_size", None) or infer_pil_side(args)

    results, key_to_idx = load_results_for_resume(out_path)
    vlm_calls = 0
    score_fn = (
        _score_per_criterion
        if args.eval_protocol == EVAL_PROTOCOL_PER_CRITERION
        else _score_joint
    )

    for sample in samples:
        row_key = _row_key(sample, args.eval_protocol)
        path_str, tool = row_key

        if (
            not args.force
            and row_key in key_to_idx
            and _should_skip_resume(
                results[key_to_idx[row_key]], tool, args.eval_protocol,
            )
        ):
            rec = results[key_to_idx[row_key]]
            ev = score_fn(rec)
            if ev:
                rec["evaluation"] = ev
            continue

        try:
            if args.eval_protocol == EVAL_PROTOCOL_PER_CRITERION:
                frame_output = _run_per_criterion(
                    backend=backend, pil_side=pil_side, sample=sample, args=args,
                )
            else:
                frame_output = _run_joint(
                    backend=backend, pil_side=pil_side, sample=sample, args=args,
                )
            vlm_calls += 1
        except Exception as e:
            print(f"SKIP {path_str} {tool}: {e}", file=sys.stderr)
            frame_output = {"error": str(e)}

        user_prompt = frame_output.get("user_prompt", "") if frame_output else ""
        entry = _make_result_entry(
            sample=sample,
            eval_protocol=args.eval_protocol,
            user_prompt=user_prompt,
            frame_output=frame_output,
        )
        ev = score_fn(entry)
        if ev:
            entry["evaluation"] = ev
        upsert_result(results, key_to_idx, row_key, entry)

        if vlm_calls and vlm_calls % 25 == 0:
            print(f"  ... {vlm_calls} VLM calls", file=sys.stderr)

    print(f"VLM forward passes this run: {vlm_calls}", file=sys.stderr)

    for rec in results:
        if "evaluation" not in rec:
            ev = score_fn(rec)
            if ev:
                rec["evaluation"] = ev

    metrics = aggregate_metrics(results)
    payload = {
        "task": "cvs_evaluation",
        "eval_protocol": args.eval_protocol,
        "dataset": "endoscapes2023",
        "dataset_root": str(dataset_root),
        "split": args.split,
        "gt_threshold": args.gt_threshold,
        "cvs_criteria": [
            {
                "criterion_id": c.criterion_id,
                "index": c.index,
                "short_name": c.short_name,
                "question": c.question,
            }
            for c in CVS_CRITERIA
        ],
        "backend": args.backend,
        "model_id": meta.get("model_id") if meta.get("source") == "local_checkpoint" else model_id,
        "hub_model_id_cli": model_id,
        "model_name": model_name,
        "vlm_load": meta,
        "vlm_forward_passes": vlm_calls,
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
        f"  Average Accuracy={m.get('average_accuracy')}\n"
        f"  Balanced Accuracy={m.get('balanced_accuracy')}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
