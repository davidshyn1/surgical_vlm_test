"""
tissue_instrument_recognition_endovis18.py

EndoVis 2018 VQA — instrument & tissue recognition (Classification QA, MCQ).

  - Questions: all lines in ``vqa/Classification/frame*_QA.txt``
  - Prompt: question + comma-separated keyword options; answer = one keyword only
  - Metrics: Accuracy, macro Recall / Precision / Jaccard (IoU) — same as phase recognition
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
from endovis18_vqa_data import (
    DEFAULT_IMAGES_ROOT,
    DEFAULT_VQA_ROOT,
    GLOBAL_ANSWER_KEYWORDS,
    collect_classification_samples,
    options_for_question,
)
from triplet_recognition_cholect50 import (
    _canonical,
    _match_option_token,
    parse_mcq_terms,
)
from utils import load_results_for_resume, resolve_device, upsert_result

_SCRIPT_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_ROOT = _SCRIPT_ROOT / "outputs" / "tissue_instrument_recognition_endovis18"
_DEFAULT_HF_TOKEN = _SCRIPT_ROOT / ".hf_token"

ANSWER_CLASS_IDS: tuple[str, ...] = tuple(_canonical(k) for k in GLOBAL_ANSWER_KEYWORDS)


def _build_name_option_map(options: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for opt in options:
        out[_canonical(opt)] = opt
    return out


def build_recognition_prompt(question: str, options: list[str]) -> tuple[str, dict[str, Any]]:
    option_map = _build_name_option_map(options)
    opts_line = ", ".join(options)
    body = (
        f"{question.strip()}\n\n"
        "Answer with exactly one keyword from the options below. "
        "Reply with the keyword only — no extra words, labels, or punctuation.\n\n"
        f"Options: {opts_line}"
    )
    return body, {"option_map": option_map, "options": options}


def wrap_vlm_prompt(body: str) -> str:
    return body.strip()


def parse_keyword_response(
    text: str,
    *,
    options: list[str],
    option_map: dict[str, str],
) -> dict[str, Any]:
    raw = (text or "").strip()
    pred: str | None = None

    for line in raw.splitlines():
        s = line.strip()
        s = re.sub(r"^[-*•]\s*", "", s)
        s = re.sub(r"^\d+[.)]\s*", "", s).strip()
        if not s:
            continue
        hit = _match_option_token(s, option_map, options)
        if hit:
            pred = hit
            break

    if pred is None:
        hit = _match_option_token(raw, option_map, options)
        if hit:
            pred = hit

    if pred is None:
        hits = parse_mcq_terms(raw, options)
        if hits:
            pred = hits[0]

    return {"keyword": pred, "raw": raw}


def _row_key(sample: dict[str, Any]) -> tuple[str, str]:
    tool = (
        f"endovis18-tir|seq_{sample['seq']}|f{int(sample['frame_index']):03d}"
        f"|q{int(sample['question_index'])}"
    )
    return str(sample["img_path"]), tool


def _score_record(rec: dict[str, Any]) -> dict[str, Any] | None:
    inp = rec.get("input") or {}
    lc = inp.get("label_context") or {}
    out = rec.get("output")
    if not isinstance(out, dict):
        return None
    parsed = out.get("parsed") or {}
    gold = str(lc.get("gold_keyword") or "")
    pred = parsed.get("keyword")
    if not gold:
        return None
    gold_id = _canonical(gold)
    pred_id = _canonical(pred) if pred else ""
    return {
        "gold_keyword": gold,
        "pred_keyword": pred,
        "gold_keyword_id": gold_id,
        "pred_keyword_id": pred_id or None,
        "correct": bool(pred_id and pred_id == gold_id),
    }


def aggregate_metrics(results: list[dict[str, Any]]) -> dict[str, Any]:
    evs = [r["evaluation"] for r in results if r.get("evaluation")]
    if not evs:
        return {"n_results": len(results), "n_scored": 0}

    y_true = [e["gold_keyword_id"] for e in evs]
    y_pred = [e.get("pred_keyword_id") or "__none__" for e in evs]

    n = len(evs)
    correct = sum(1 for e in evs if e.get("correct"))
    accuracy = correct / n if n else None

    recalls: list[float] = []
    precisions: list[float] = []
    jaccards: list[float] = []

    for cid in ANSWER_CLASS_IDS:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == cid and p == cid)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != cid and p == cid)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == cid and p != cid)
        support = sum(1 for t in y_true if t == cid)
        if support <= 0:
            continue

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        jac = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
        recalls.append(rec)
        precisions.append(prec)
        jaccards.append(jac)

    def _macro(vals: list[float]) -> float | None:
        return sum(vals) / len(vals) if vals else None

    return {
        "protocol": "endovis18_tissue_instrument_recognition_mcq",
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
    return bool((out.get("parsed") or {}).get("keyword"))


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


def _make_result_entry(
    *,
    sample: dict[str, Any],
    user_prompt: str,
    prompt_meta: dict[str, Any],
    args: argparse.Namespace,
    frame_output: dict[str, Any] | None,
) -> dict[str, Any]:
    path_str, tool = _row_key(sample)
    entry: dict[str, Any] = {
        "input": {
            "image_path": path_str,
            "tool": tool,
            "label_context": {
                "seq": sample["seq"],
                "frame_index": sample["frame_index"],
                "question": sample["question"],
                "question_template": sample["question_template"],
                "question_type": sample["question_type"],
                "gold_keyword": sample["gold_keyword"],
                "options": sample["options"],
            },
            "eval_protocol": "mcq",
            "prompt_mode": "mcq",
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
        description="EndoVis 2018 VQA tissue/instrument recognition (Classification MCQ).",
    )
    p.add_argument("--backend", choices=BACKEND_CHOICES, default="prismatic")
    p.add_argument("--vqa-root", type=Path, default=DEFAULT_VQA_ROOT)
    p.add_argument("--images-root", type=Path, default=DEFAULT_IMAGES_ROOT)
    p.add_argument(
        "--image-split",
        choices=("val", "train", "both"),
        default="val",
        help="Which EndoVis2018 image split to use (default: val).",
    )
    p.add_argument("--seq", type=str, default=None, help="Evaluate one sequence only, e.g. 2 or seq_2.")
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
    vqa_root = args.vqa_root.resolve()
    images_root = args.images_root.resolve()
    if not vqa_root.is_dir():
        raise FileNotFoundError(f"VQA root not found: {vqa_root}")
    if not images_root.is_dir():
        raise FileNotFoundError(f"Images root not found: {images_root}")

    samples = collect_classification_samples(
        vqa_root,
        images_root=images_root,
        image_split=args.image_split,
        seq_filter=args.seq,
    )
    if not samples:
        raise RuntimeError(
            "No Classification QA samples with resolvable images. "
            "Check --vqa-root, --images-root, --image-split, and --seq."
        )
    if args.max_samples is not None:
        samples = samples[: max(0, int(args.max_samples))]

    model_id = resolve_model_id(args.backend, args.model_id)
    model_name = resolve_output_model_name(args.backend, model_id, args.model_name)
    out_root = args.output_root.resolve()
    split_slug = args.image_split
    out_path = (
        args.output.resolve()
        if args.output is not None
        else (
            out_root
            / f"tir_{args.backend}_{model_name}_mcq_{split_slug}"
            / f"endovis18_tir_{split_slug}.json"
        ).resolve()
    )

    n_frames = len({(s["seq"], s["frame_index"]) for s in samples})
    n_seq = len({s["seq"] for s in samples})
    print(
        f"EndoVis18 tissue/instrument recognition: seq={n_seq}, frames={n_frames}, "
        f"qa_pairs={len(samples)}, image_split={args.image_split}.",
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

    for sample in samples:
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
                "seq": sample["seq"],
                "frame_index": sample["frame_index"],
                "question": sample["question"],
                "question_template": sample["question_template"],
                "question_type": sample["question_type"],
                "gold_keyword": sample["gold_keyword"],
                "options": sample["options"],
            }
            ev = _score_record(rec)
            if ev:
                rec["evaluation"] = ev
            continue

        options = sample["options"]
        user_prompt, prompt_meta = build_recognition_prompt(sample["question"], options)
        option_map = prompt_meta["option_map"]

        try:
            text = _generate_vlm_text(
                backend=backend,
                pil_side=pil_side,
                image_path=Path(sample["img_path"]),
                user_prompt=user_prompt,
                args=args,
            )
            parsed = parse_keyword_response(
                text,
                options=options,
                option_map=option_map,
            )
            frame_output = {"text": text, "parsed": parsed}
            vlm_calls += 1
        except Exception as e:
            print(f"SKIP {path_str} {tool}: {e}", file=sys.stderr)
            frame_output = {"error": str(e)}

        entry = _make_result_entry(
            sample=sample,
            user_prompt=user_prompt,
            prompt_meta=prompt_meta,
            args=args,
            frame_output=frame_output,
        )
        ev = _score_record(entry)
        if ev:
            entry["evaluation"] = ev
        upsert_result(results, key_to_idx, row_key, entry)

        if vlm_calls and vlm_calls % 50 == 0:
            print(f"  ... {vlm_calls} VLM calls", file=sys.stderr)

    print(f"VLM forward passes this run: {vlm_calls}", file=sys.stderr)

    for rec in results:
        if "evaluation" not in rec:
            ev = _score_record(rec)
            if ev:
                rec["evaluation"] = ev

    metrics = aggregate_metrics(results)
    payload = {
        "task": "tissue_instrument_recognition",
        "eval_protocol": "endovis18_classification_mcq",
        "dataset": "EndoVis-18-VQA",
        "vqa_root": str(vqa_root),
        "images_root": str(images_root),
        "image_split": args.image_split,
        "backend": args.backend,
        "model_id": meta.get("model_id") if meta.get("source") == "local_checkpoint" else model_id,
        "hub_model_id_cli": model_id,
        "model_name": model_name,
        "vlm_load": meta,
        "answer_classes": list(GLOBAL_ANSWER_KEYWORDS),
        "answer_class_ids": list(ANSWER_CLASS_IDS),
        "question_templates": {
            "organ": {"question": "What organ is being operated?", "options": list(options_for_question("What organ is being operated?"))},
            "state": {"question": "What is the state of {instrument}?", "options": list(options_for_question("What is the state of bipolar_forceps?"))},
            "location": {"question": "Where is {instrument} located?", "options": list(options_for_question("Where is bipolar_forceps located?"))},
        },
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
