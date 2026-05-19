"""
tissue_instrument_recognition_endovis18.py

EndoVis 2018 VQA — instrument & tissue recognition (Classification QA, MCQ).

  - Questions: all lines in ``vqa/Classification/frame*_QA.txt``
  - ``--prompt-mode mcq``: question + comma-separated option list in the prompt
  - ``--prompt-mode ov``: open vocabulary (no options in the prompt)
  - Metrics: tissue accuracy (Q1 organ), instrument accuracy (Q2–Q4 state/location),
    tools macro AUROC (Q5 multi-label); Q5 per-sample correct = exact set match (order-free).
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
    TOOLS_QUESTION,
    collect_classification_samples,
    discover_instruments,
    options_for_question,
)
from triplet_recognition_cholect50 import (
    _canonical,
    _match_option_token,
    parse_mcq_terms,
)
from utils import load_results_for_resume, resolve_device, strip_lora_answer_tags, upsert_result

_SCRIPT_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_ROOT = _SCRIPT_ROOT / "outputs" / "tissue_instrument_recognition_endovis18"
_DEFAULT_HF_TOKEN = _SCRIPT_ROOT / ".hf_token"

ANSWER_CLASS_IDS: tuple[str, ...] = tuple(_canonical(k) for k in GLOBAL_ANSWER_KEYWORDS)


def _build_name_option_map(options: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for opt in options:
        out[_canonical(opt)] = opt
    return out


def build_recognition_prompt(
    question: str,
    options: list[str],
    *,
    prompt_mode: str = "mcq",
    multi_select: bool = False,
) -> tuple[str, dict[str, Any]]:
    """
    Build the user prompt for one VQA sample.

    - ``mcq``: include the option list in the prompt (closed-vocabulary MCQ).
    - ``ov``: open vocabulary — no options in the prompt; answers are still
      scored against the sample's option set.
    """
    mode = (prompt_mode or "mcq").strip().lower()
    option_map = _build_name_option_map(options)
    meta: dict[str, Any] = {
        "prompt_mode": mode,
        "option_map": option_map,
        "options": options,
        "multi_select": multi_select,
    }
    q = question.strip()

    if mode == "ov":
        if multi_select:
            answer_line = (
                "Answer with one or more keywords, separated by commas — "
                "no extra words or labels."
            )
        else:
            answer_line = (
                "Answer with the keyword only — no extra words, labels, or punctuation."
            )
        body = f"{q}\n\n{answer_line}"
        return body, meta

    if mode != "mcq":
        raise ValueError(f"Unknown prompt_mode {prompt_mode!r}; choose mcq or ov.")

    opts_line = ", ".join(options)
    if multi_select:
        body = (
            f"{q}\n\n"
            "Answer with one or more keywords from the options below. "
            "Reply with keywords only, separated by commas — no extra words or labels.\n\n"
            f"Options: {opts_line}"
        )
    else:
        body = (
            f"{q}\n\n"
            "Answer with exactly one keyword from the options below. "
            "Reply with the keyword only — no extra words, labels, or punctuation.\n\n"
            f"Options: {opts_line}"
        )
    return body, meta


def wrap_vlm_prompt(body: str) -> str:
    return body.strip()


def parse_keyword_response(
    text: str,
    *,
    options: list[str],
    option_map: dict[str, str],
) -> dict[str, Any]:
    raw = strip_lora_answer_tags(text)
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


def parse_tools_response(
    text: str,
    *,
    options: list[str],
    option_map: dict[str, str],
) -> dict[str, Any]:
    raw = strip_lora_answer_tags(text)
    keywords: list[str] = []
    seen: set[str] = set()

    def _add_hit(hit: str | None) -> None:
        if not hit:
            return
        cid = _canonical(hit)
        if cid in seen:
            return
        seen.add(cid)
        keywords.append(hit)

    chunks = re.split(r"[,;\n]+", raw) if raw else []
    for chunk in chunks:
        s = chunk.strip()
        s = re.sub(r"^[-*•]\s*", "", s)
        s = re.sub(r"^\d+[.)]\s*", "", s).strip()
        if not s:
            continue
        _add_hit(_match_option_token(s, option_map, options))

    if not keywords:
        hits = parse_mcq_terms(raw, options)
        for hit in hits:
            _add_hit(hit)

    joined = ", ".join(keywords) if keywords else None
    return {"keywords": keywords, "keyword": joined, "raw": raw}


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
    qtype = lc.get("question_type")

    if qtype == "tools":
        gold_list = lc.get("gold_keywords") or []
        if not gold_list and lc.get("gold_keyword"):
            gold_list = [
                k.strip()
                for k in str(lc["gold_keyword"]).split(",")
                if k.strip()
            ]
        pred_list = parsed.get("keywords") or []
        if not pred_list and parsed.get("keyword"):
            pred_list = [
                k.strip()
                for k in str(parsed["keyword"]).split(",")
                if k.strip()
            ]
        if not gold_list:
            return None
        gold_set = {_canonical(k) for k in gold_list}
        pred_set = {_canonical(k) for k in pred_list if k}
        # Order-independent: correct only if predicted set equals gold (no missing/extra).
        exact_match = gold_set == pred_set
        return {
            "question_type": "tools",
            "gold_keywords": list(gold_list),
            "pred_keywords": list(pred_list),
            "gold_keyword": lc.get("gold_keyword"),
            "pred_keyword": parsed.get("keyword"),
            "gold_set_ids": sorted(gold_set),
            "pred_set_ids": sorted(pred_set),
            "exact_set_match": exact_match,
            "correct": exact_match,
        }

    gold = str(lc.get("gold_keyword") or "")
    pred = parsed.get("keyword")
    if not gold:
        return None
    gold_id = _canonical(gold)
    pred_id = _canonical(pred) if pred else ""
    return {
        "question_type": qtype,
        "gold_keyword": gold,
        "pred_keyword": pred,
        "gold_keyword_id": gold_id,
        "pred_keyword_id": pred_id or None,
        "correct": bool(pred_id and pred_id == gold_id),
    }


def _accuracy_block(correct: int, total: int) -> dict[str, Any]:
    return {
        "correct": correct,
        "total": total,
        "value": (correct / total) if total else None,
    }


def _roc_auc_binary(y_true: list[int], y_score: list[float]) -> float | None:
    """ROC-AUC via Mann–Whitney U (handles tied scores)."""
    pos = [s for s, t in zip(y_score, y_true, strict=True) if t == 1]
    neg = [s for s, t in zip(y_score, y_true, strict=True) if t == 0]
    if not pos or not neg:
        return None
    wins = 0.0
    for ps in pos:
        for ns in neg:
            if ps > ns:
                wins += 1.0
            elif ps == ns:
                wins += 0.5
    return wins / (len(pos) * len(neg))


def mean_auroc_multilabel(
    gold_sets: list[set[str]],
    pred_sets: list[set[str]],
    classes: list[str],
) -> float | None:
    """Macro-averaged AUROC over instrument labels (Q5 multi-label)."""
    if not gold_sets:
        return None
    aucs: list[float] = []
    for cls in classes:
        ckey = _canonical(cls)
        y_true = [1 if ckey in gs else 0 for gs in gold_sets]
        y_score = [1.0 if ckey in ps else 0.0 for ps in pred_sets]
        auc = _roc_auc_binary(y_true, y_score)
        if auc is not None:
            aucs.append(auc)
    return sum(aucs) / len(aucs) if aucs else None


def aggregate_metrics(
    results: list[dict[str, Any]],
    *,
    instrument_classes: list[str],
) -> dict[str, Any]:
    organ_evs: list[dict[str, Any]] = []
    instrument_evs: list[dict[str, Any]] = []
    tools_gold_sets: list[set[str]] = []
    tools_pred_sets: list[set[str]] = []
    tools_exact_correct = 0
    tools_n = 0

    for rec in results:
        ev = rec.get("evaluation")
        if not ev:
            continue
        qtype = ev.get("question_type") or (rec.get("input") or {}).get(
            "label_context", {}
        ).get("question_type")

        if qtype == "tools":
            tools_n += 1
            if ev.get("correct"):
                tools_exact_correct += 1
            gold_ids = ev.get("gold_set_ids")
            pred_ids = ev.get("pred_set_ids")
            if gold_ids is not None:
                tools_gold_sets.append(set(gold_ids))
            else:
                tools_gold_sets.append(
                    {_canonical(k) for k in ev.get("gold_keywords") or []}
                )
            if pred_ids is not None:
                tools_pred_sets.append(set(pred_ids))
            else:
                tools_pred_sets.append(
                    {_canonical(k) for k in ev.get("pred_keywords") or []}
                )
            continue

        if qtype == "organ":
            organ_evs.append(ev)
        elif qtype in ("state", "location"):
            instrument_evs.append(ev)

    n_scored = len(organ_evs) + len(instrument_evs) + tools_n
    if n_scored == 0:
        return {"n_results": len(results), "n_scored": 0}

    tissue_correct = sum(1 for e in organ_evs if e.get("correct"))
    instrument_correct = sum(1 for e in instrument_evs if e.get("correct"))
    tools_auroc = mean_auroc_multilabel(
        tools_gold_sets,
        tools_pred_sets,
        instrument_classes,
    )

    all_single_correct = tissue_correct + instrument_correct
    all_single_n = len(organ_evs) + len(instrument_evs)

    return {
        "protocol": "endovis18_tissue_instrument_recognition_mcq",
        "n_scored": n_scored,
        "n_results": len(results),
        "tissue_accuracy": _accuracy_block(tissue_correct, len(organ_evs)),
        "instrument_accuracy": _accuracy_block(instrument_correct, len(instrument_evs)),
        "overall_accuracy": _accuracy_block(all_single_correct, all_single_n),
        "tools_auroc": {
            "value": tools_auroc,
            "n_frames": tools_n,
            "n_labels": len(instrument_classes),
        },
        "tools_exact_set_accuracy": _accuracy_block(tools_exact_correct, tools_n),
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
    if (inp.get("label_context") or {}).get("question_type") == "tools":
        return bool(parsed.get("keywords"))
    return bool(parsed.get("keyword"))


def _label_context_from_sample(sample: dict[str, Any]) -> dict[str, Any]:
    lc: dict[str, Any] = {
        "seq": sample["seq"],
        "frame_index": sample["frame_index"],
        "question": sample["question"],
        "question_template": sample["question_template"],
        "question_type": sample["question_type"],
        "gold_keyword": sample["gold_keyword"],
        "options": sample["options"],
    }
    if sample.get("instrument") is not None:
        lc["instrument"] = sample["instrument"]
    if sample.get("gold_keywords") is not None:
        lc["gold_keywords"] = sample["gold_keywords"]
    return lc


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
            "label_context": _label_context_from_sample(sample),
            "eval_protocol": args.prompt_mode,
            "prompt_mode": args.prompt_mode,
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
    p.add_argument(
        "--prompt-mode",
        choices=("mcq", "ov"),
        default="mcq",
        help="mcq: include option lists in the prompt; ov: open vocabulary (no options shown).",
    )
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
            / f"tir_{args.backend}_{model_name}_{args.prompt_mode}_{split_slug}"
            / f"endovis18_tir_{split_slug}.json"
        ).resolve()
    )

    n_frames = len({(s["seq"], s["frame_index"]) for s in samples})
    n_seq = len({s["seq"] for s in samples})
    print(
        f"EndoVis18 tissue/instrument recognition: seq={n_seq}, frames={n_frames}, "
        f"qa_pairs={len(samples)}, image_split={args.image_split}, "
        f"prompt_mode={args.prompt_mode}.",
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
            inp["label_context"] = _label_context_from_sample(sample)
            ev = _score_record(rec)
            if ev:
                rec["evaluation"] = ev
            continue

        options = sample["options"]
        multi_select = sample.get("question_type") == "tools"
        user_prompt, prompt_meta = build_recognition_prompt(
            sample["question"],
            options,
            prompt_mode=args.prompt_mode,
            multi_select=multi_select,
        )
        option_map = prompt_meta["option_map"]

        try:
            text = _generate_vlm_text(
                backend=backend,
                pil_side=pil_side,
                image_path=Path(sample["img_path"]),
                user_prompt=user_prompt,
                args=args,
            )
            if multi_select:
                parsed = parse_tools_response(
                    text,
                    options=options,
                    option_map=option_map,
                )
            else:
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

    instrument_opts = list(discover_instruments(vqa_root))
    metrics = aggregate_metrics(results, instrument_classes=instrument_opts)
    payload = {
        "task": "tissue_instrument_recognition",
        "eval_protocol": f"endovis18_classification_{args.prompt_mode}",
        "prompt_mode": args.prompt_mode,
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
            "organ": {
                "question": "What organ is being operated?",
                "options": list(options_for_question("What organ is being operated?")),
            },
            "state": {
                "question": "What is the state of {instrument}?",
                "options": list(options_for_question("What is the state of bipolar_forceps?")),
            },
            "location": {
                "question": "Where is {instrument} located?",
                "options": list(options_for_question("Where is bipolar_forceps located?")),
            },
            "tools": {
                "question": TOOLS_QUESTION,
                "options": instrument_opts,
            },
        },
        "instrument_options": instrument_opts,
        "vlm_forward_passes": vlm_calls,
        "metrics": metrics,
        "count": len(results),
        "results": results,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    m = metrics
    tissue = (m.get("tissue_accuracy") or {}).get("value")
    inst = (m.get("instrument_accuracy") or {}).get("value")
    tools_auc = (m.get("tools_auroc") or {}).get("value")
    tools_exact = (m.get("tools_exact_set_accuracy") or {}).get("value")
    print(
        f"Wrote {len(results)} entries to {out_path}\n"
        f"  Tissue accuracy (Q1)={tissue}\n"
        f"  Instrument accuracy (Q2–Q4)={inst}\n"
        f"  Tools macro AUROC (Q5)={tools_auc}\n"
        f"  Tools exact-set accuracy (Q5)={tools_exact}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
