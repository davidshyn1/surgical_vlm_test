"""
language_grounding_surgical_prompts.py

Language-only eval on CholecT50 surgical_prompts.json (triplet completion).

Dataset (surgical_prompts.json):
  1. given (phase, verb, target)       → instrument set  [pvt_to_instrument]
  2. given (phase, instrument, target) → verb set        [pit_to_verb]
  3. given (phase, instrument, verb)    → target set       [piv_to_target]

Metrics:
  - Sample-averaged multi-label F1 (≤3 comma-separated terms in the answer)
  - Macro AUROC + macro mAP per output_field (vocab + synthetic ``others`` class)

Scoring (parsed answer → per-label 0/1):
  - Token in vocab and in answer → that label = 1, else 0
  - Any OOV token in answer → ``others`` = 1 (gold for ``others`` is always 0)
  - Per-label AUROC uses fallback 0.5 when only one class appears in y_true

Backends (via ``grounding_task.sh`` + ``load_backend``):
  - **prismatic**: LLM-only ``generate_text`` (no vision forward)
  - **hf**, **qwen3-***, **cosmos-***, **internvl3.5**, **paligemma2**, **groot**: HF
    ``AutoProcessor`` text path (``_prepare_hf_text_inputs``)
  - **openai** / **gpt** / **gemini** / **claude**: cloud **text** API (no image)

Usage:
  BACKEND=qwen3-4b bash grounding_task.sh language_grounding_surgical_prompts --limit 20
  BACKEND=prismatic bash grounding_task.sh language_grounding_surgical_prompts --limit 20
  BACKEND=gpt MODEL_ID=gpt-4o-mini bash grounding_task.sh language_grounding_surgical_prompts --limit 20
  python language_grounding_surgical_prompts.py --metrics-only outputs/.../surgical_prompts.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from backend_registry import (
    BACKEND_CHOICES,
    is_api_backend,
    resolve_hf_token,
    resolve_model_id,
    resolve_output_model_name,
)
from backends import build_vlm_user_prompt, generate_language_answer, load_backend
from utils import (
    OTHERS_LABEL_KEY,
    build_pred_label_scores,
    multilabel_classification_metrics,
    normalize_label_key,
    resolve_device,
    strip_lora_answer_tags,
)

_SCRIPT_ROOT = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_ROOT.parent
_DEFAULT_JSON = _REPO_ROOT / "eval" / "prompts" / "surgical_prompts.json"
_DEFAULT_HF_TOKEN = _SCRIPT_ROOT / ".hf_token"
_DEFAULT_OUTPUT_ROOT = _SCRIPT_ROOT / "outputs" / "language_grounding_surgical_prompts"
_PROMPTS_DIR = _REPO_ROOT / "eval" / "prompts"
if str(_PROMPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_PROMPTS_DIR))
from build_surgical_prompt import build_prompt as build_surgical_prompt_text  # noqa: E402

def _split_response_tokens(value: str | list[str]) -> list[str]:
    if isinstance(value, list):
        return [normalize_label_key(t) for t in value if str(t).strip()]
    text = strip_lora_answer_tags(str(value) if value is not None else "")
    if not text:
        return []
    return [
        normalize_label_key(t)
        for t in text.split(",")
        if t.strip()
    ]


def _response_to_gold_string(value: str | list[str]) -> str:
    if isinstance(value, list):
        return ", ".join(str(t).strip() for t in value if str(t).strip())
    return str(value).strip()


def _token_set(value: str | list[str]) -> frozenset[str]:
    return frozenset(_split_response_tokens(value))


def _truncate_predictions(pred_raw: str, k: int = 3) -> str:
    tokens = _split_response_tokens(pred_raw)
    if len(tokens) <= k:
        return pred_raw
    # Reconstruct display string from normalized keys (hyphens); good enough for logging.
    return ", ".join(tokens[:k])


def build_label_vocabulary(raw: list[Any]) -> dict[str, list[str]]:
    """Union of all gold labels per output_field in the dataset file."""
    vocab: dict[str, set[str]] = defaultdict(set)
    for obj in raw:
        if not isinstance(obj, dict):
            continue
        field = str(obj.get("output_field") or "")
        if field not in ("instrument", "verb", "target"):
            continue
        for lab in _split_response_tokens(obj.get("response")):
            vocab[field].add(lab)
    return {k: sorted(v) for k, v in vocab.items()}


def score_sample(
    gold_raw: str | list[str],
    pred_raw: str,
    *,
    label_vocab: list[str],
) -> dict[str, Any]:
    gset = _token_set(gold_raw)
    pred_tokens = _split_response_tokens(pred_raw)
    vocab_keys = {normalize_label_key(c) for c in label_vocab}
    pred_in_vocab = [t for t in pred_tokens if t in vocab_keys]
    pred_oov = [t for t in pred_tokens if t not in vocab_keys]
    pset_vocab = frozenset(pred_in_vocab)
    inter = len(gset & pset_vocab)
    precision = inter / len(pset_vocab) if pset_vocab else 0.0
    recall = inter / len(gset) if gset else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    label_scores, pred_in_vocab_display, pred_oov_raw = build_pred_label_scores(
        pred_tokens, label_vocab,
    )
    return {
        "gold_response": _response_to_gold_string(gold_raw),
        "pred_response": pred_raw,
        "gold_terms": sorted(gset),
        "pred_terms": sorted(pred_in_vocab),
        "pred_terms_oov": pred_oov_raw,
        "pred_others": bool(pred_oov_raw),
        "intersection": inter,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "label_scores": label_scores,
    }


def build_human_message(prompt: str) -> str:
    body = prompt.strip()
    instr = (
        "Answer using a **comma-separated list of at most 3** short lowercase terms only "
        "(e.g. `grasper, hook`). No numbering, bullets, or long prose unless a single short phrase is required. "
    )
    return f"{body}\n\n{instr}"


def load_surgical_prompt_rows(
    raw: list[Any],
    source_label: str,
    *,
    filter_category: str | None,
    filter_subtype: str | None,
    label_options: dict[str, list[str]] | None = None,
) -> list[dict[str, Any]]:
    if not isinstance(raw, list):
        raise ValueError(f"{source_label}: expected a JSON array at root")
    rows: list[dict[str, Any]] = []
    for i, obj in enumerate(raw):
        if not isinstance(obj, dict):
            continue
        response = obj.get("response")
        if not isinstance(response, (str, list)):
            continue
        if isinstance(response, list) and not response:
            continue
        if isinstance(response, str) and not response.strip():
            continue
        try:
            prompt = build_surgical_prompt_text(obj, label_options=label_options)
        except (ValueError, KeyError):
            continue
        cat = obj.get("category")
        sub = obj.get("subtype")
        if filter_category is not None and str(cat) != filter_category:
            continue
        if filter_subtype is not None and str(sub) != filter_subtype:
            continue
        output_field = str(obj.get("output_field") or "")
        rows.append(
            {
                "index": i,
                "id": obj.get("id"),
                "category": cat,
                "subtype": sub,
                "phase": obj.get("phase"),
                "instrument": obj.get("instrument"),
                "verb": obj.get("verb"),
                "target": obj.get("target"),
                "output_field": output_field,
                "prompt": prompt.strip(),
                "gold_response": _response_to_gold_string(response),
                "gold_labels": _split_response_tokens(response),
            }
        )
    return rows


def load_rows_multi_subtypes(
    raw: list[Any],
    source_label: str,
    *,
    filter_category: str | None,
    subtypes: list[str],
    per_subtype_limit: int,
    label_options: dict[str, list[str]] | None = None,
) -> list[dict[str, Any]]:
    if per_subtype_limit < 0:
        raise ValueError("per_subtype_limit must be >= 0")
    out: list[dict[str, Any]] = []
    for st in subtypes:
        st = st.strip()
        if not st:
            continue
        part = load_surgical_prompt_rows(
            raw,
            source_label,
            filter_category=filter_category,
            filter_subtype=st,
            label_options=label_options,
        )
        out.extend(part[:per_subtype_limit])
    return out


def aggregate_f1_by_subtype(results: list[dict[str, Any]]) -> dict[str, Any]:
    by_sub: dict[str, list[float]] = defaultdict(list)
    by_field: dict[str, list[float]] = defaultdict(list)
    for rec in results:
        ev = rec.get("evaluation")
        if not isinstance(ev, dict):
            continue
        f1 = float(ev.get("f1", 0.0))
        inp = rec.get("input") or {}
        by_sub[str(inp.get("subtype") or "unknown")].append(f1)
        by_field[str(inp.get("output_field") or "unknown")].append(f1)

    def _avg(vals: list[float]) -> float:
        return sum(vals) / len(vals) if vals else 0.0

    return {
        "by_subtype": {
            k: {"count": len(v), "sample_averaged_f1": _avg(v)}
            for k, v in sorted(by_sub.items())
        },
        "by_output_field": {
            k: {"count": len(v), "sample_averaged_f1": _avg(v)}
            for k, v in sorted(by_field.items())
        },
    }


def _pool_results_for_metrics(
    results: list[dict[str, Any]],
) -> tuple[
    dict[str, list[set[str]]],
    dict[str, list[dict[str, float]]],
    dict[str, list[set[str]]],
    dict[str, list[dict[str, float]]],
    dict[str, str],
]:
    by_field_gold: dict[str, list[set[str]]] = defaultdict(list)
    by_field_scores: dict[str, list[dict[str, float]]] = defaultdict(list)
    by_sub_gold: dict[str, list[set[str]]] = defaultdict(list)
    by_sub_scores: dict[str, list[dict[str, float]]] = defaultdict(list)
    by_sub_field: dict[str, str] = {}

    for rec in results:
        ev = rec.get("evaluation")
        if not isinstance(ev, dict):
            continue
        inp = rec.get("input") or {}
        field = str(inp.get("output_field") or "")
        sub = str(inp.get("subtype") or "unknown")
        gold_terms = ev.get("gold_terms") or []
        label_scores = ev.get("label_scores")
        if not field or not isinstance(label_scores, dict):
            continue
        gset = {normalize_label_key(t) for t in gold_terms}
        by_field_gold[field].append(gset)
        by_field_scores[field].append(label_scores)
        by_sub_gold[sub].append(gset)
        by_sub_scores[sub].append(label_scores)
        by_sub_field[sub] = field

    return by_field_gold, by_field_scores, by_sub_gold, by_sub_scores, by_sub_field


def aggregate_classification_metrics(
    results: list[dict[str, Any]],
    label_vocab: dict[str, list[str]],
) -> dict[str, Any]:
    """Macro AUROC + macro mAP (vocab + ``others``) per output_field and subtype."""
    by_field_gold, by_field_scores, by_sub_gold, by_sub_scores, by_sub_field = (
        _pool_results_for_metrics(results)
    )

    by_output_field: dict[str, Any] = {}
    field_aurocs: list[float] = []
    field_maps: list[float] = []
    for field, gold_sets in sorted(by_field_gold.items()):
        classes = label_vocab.get(field, [])
        block = multilabel_classification_metrics(
            gold_sets, by_field_scores[field], classes,
        )
        by_output_field[field] = {
            "n_samples": len(gold_sets),
            "n_vocab_labels": len(classes),
            "n_eval_labels": block.get("n_labels_total"),
            "macro_auroc": block.get("macro_auroc"),
            "macro_map": block.get("macro_map"),
            "per_label": block.get("per_label"),
        }
        if block.get("macro_auroc") is not None:
            field_aurocs.append(float(block["macro_auroc"]))
        if block.get("macro_map") is not None:
            field_maps.append(float(block["macro_map"]))

    by_subtype: dict[str, Any] = {}
    for sub, gold_sets in sorted(by_sub_gold.items()):
        field = by_sub_field.get(sub, "")
        classes = label_vocab.get(field, [])
        block = multilabel_classification_metrics(
            gold_sets, by_sub_scores[sub], classes,
        )
        by_subtype[sub] = {
            "output_field": field,
            "n_samples": len(gold_sets),
            "n_vocab_labels": len(classes),
            "n_eval_labels": block.get("n_labels_total"),
            "macro_auroc": block.get("macro_auroc"),
            "macro_map": block.get("macro_map"),
            "per_label": block.get("per_label"),
        }

    return {
        "scoring": {
            "in_vocab_in_answer": 1,
            "in_vocab_not_in_answer": 0,
            "oov_in_answer": f"{OTHERS_LABEL_KEY}=1 (gold always 0)",
            "auroc_undefined_per_label_fallback": 0.5,
        },
        "overall_macro_auroc": (
            sum(field_aurocs) / len(field_aurocs) if field_aurocs else None
        ),
        "overall_macro_map": sum(field_maps) / len(field_maps) if field_maps else None,
        "by_output_field": by_output_field,
        "by_subtype": by_subtype,
        "label_vocabulary": label_vocab,
    }


def recompute_metrics_from_results(
    payload: dict[str, Any],
    *,
    label_vocab: dict[str, list[str]],
    max_pred_labels: int,
) -> dict[str, Any]:
    results = payload.get("results") or []
    sum_f1 = 0.0
    n_scored = 0
    for rec in results:
        if rec.get("error"):
            continue
        inp = rec.get("input") or {}
        out = rec.get("output") or {}
        field = str(inp.get("output_field") or "")
        vocab = label_vocab.get(field, [])
        pred_raw = out.get("text") or out.get("text_raw") or ""
        pred_truncated = _truncate_predictions(str(pred_raw), k=max_pred_labels)
        gold = inp.get("gold_response") or ""
        if not gold and inp.get("gold_labels"):
            gold = ", ".join(inp["gold_labels"])
        scores = score_sample(gold, pred_truncated, label_vocab=vocab)
        rec["evaluation"] = scores
        sum_f1 += scores["f1"]
        n_scored += 1

    payload["counts"] = {
        **(payload.get("counts") or {}),
        "rows_scored": n_scored,
        "sample_averaged_f1": (sum_f1 / n_scored) if n_scored else 0.0,
    }
    payload["breakdown_f1"] = aggregate_f1_by_subtype(results)
    payload["classification_metrics"] = aggregate_classification_metrics(
        results, label_vocab,
    )
    return payload


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Language-only CholecT50 surgical_prompts eval "
            "(multi-label F1 + macro AUROC + macro mAP; no image)."
        ),
    )
    p.add_argument("--backend", choices=BACKEND_CHOICES, default="qwen3-4b")
    p.add_argument("--dataset-json", type=Path, default=_DEFAULT_JSON)
    p.add_argument("--output-root", type=Path, default=_DEFAULT_OUTPUT_ROOT)
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--model-id", type=str, default=None)
    p.add_argument("--model-name", type=str, default=None)
    p.add_argument("--vlm-checkpoint", type=Path, default=None)
    p.add_argument("--vlm-config", type=Path, default=None)
    p.add_argument("--hf-token", type=Path, default=_DEFAULT_HF_TOKEN)
    p.add_argument("--api-key-file", type=Path, default=None)
    p.add_argument("--api-timeout-sec", type=int, default=120)
    p.add_argument("--device", type=str, default="0")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--filter-category", type=str, default=None)
    p.add_argument("--filter-subtype", type=str, default=None)
    p.add_argument("--multi-subtypes", type=str, default=None)
    p.add_argument("--per-subtype-limit", type=int, default=None)
    p.add_argument("--do-sample", action="store_true")
    p.add_argument("--temperature", type=float, default=0.5)
    p.add_argument("--max-new-tokens", type=int, default=16)
    p.add_argument("--max-pred-labels", type=int, default=3,
                   help="Max comma-separated labels kept from each answer.")
    p.add_argument(
        "--metrics-only",
        type=Path,
        default=None,
        help="Recompute F1/AUROC from an existing results JSON (skip VLM).",
    )
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def _place_backend_on_device(backend: Any, *, backend_name: str, device_str: str) -> None:
    """Move local VLMs to GPU; API backends are no-ops."""
    if is_api_backend(backend_name):
        return
    import torch

    device = resolve_device(device_str)
    if device.type == "cuda" and torch.cuda.is_available():
        backend.to(device, dtype=torch.bfloat16)
    else:
        backend.to(device)


def main() -> None:
    args = parse_args()

    json_path = args.dataset_json.resolve()
    if not json_path.is_file():
        raise FileNotFoundError(json_path)
    raw_data = json.loads(json_path.read_text(encoding="utf-8"))
    if not isinstance(raw_data, list):
        raise ValueError(f"{json_path}: expected JSON array")
    label_vocab = build_label_vocabulary(raw_data)

    if args.metrics_only is not None:
        mp = args.metrics_only.resolve()
        if not mp.is_file():
            raise FileNotFoundError(mp)
        payload = json.loads(mp.read_text(encoding="utf-8"))
        payload = recompute_metrics_from_results(
            payload,
            label_vocab=label_vocab,
            max_pred_labels=args.max_pred_labels,
        )
        with mp.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        cm = payload.get("classification_metrics") or {}
        f1 = (payload.get("counts") or {}).get("sample_averaged_f1")
        print(
            f"Updated metrics in {mp}  F1={f1}  "
            f"macro_AUROC={cm.get('overall_macro_auroc')}  "
            f"macro_mAP={cm.get('overall_macro_map')}",
            file=sys.stderr,
        )
        return

    model_id = resolve_model_id(args.backend, args.model_id)
    model_name = resolve_output_model_name(args.backend, args.model_name, model_id)

    multi_list: list[str] | None = None
    if args.multi_subtypes:
        if args.filter_subtype is not None:
            raise ValueError("Use either --filter-subtype or --multi-subtypes, not both.")
        if args.per_subtype_limit is None:
            raise ValueError("--per-subtype-limit is required when --multi-subtypes is set.")
        multi_list = [s.strip() for s in args.multi_subtypes.split(",") if s.strip()]
        rows = load_rows_multi_subtypes(
            raw_data,
            str(json_path),
            filter_category=args.filter_category,
            subtypes=multi_list,
            per_subtype_limit=max(0, args.per_subtype_limit),
            label_options=label_vocab,
        )
    else:
        rows = load_surgical_prompt_rows(
            raw_data,
            str(json_path),
            filter_category=args.filter_category,
            filter_subtype=args.filter_subtype,
            label_options=label_vocab,
        )
        if args.limit is not None:
            rows = rows[: max(0, args.limit)]

    hf_token = None if is_api_backend(args.backend) else resolve_hf_token(args.backend, args.hf_token)

    out_path = (
        args.output.resolve()
        if args.output is not None
        else (
            args.output_root
            / f"lang_{args.backend}_{model_name}"
            / "surgical_prompts.json"
        ).resolve()
    )

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
    _place_backend_on_device(backend, backend_name=args.backend, device_str=args.device)

    gen_kw: dict[str, Any] = {
        "do_sample": args.do_sample,
        "max_new_tokens": args.max_new_tokens,
        "min_length": 1,
        "request_timeout_sec": args.api_timeout_sec,
    }
    if args.do_sample:
        gen_kw["temperature"] = args.temperature

    results: list[dict[str, Any]] = []
    sum_f1 = 0.0
    n_scored = 0

    for row in rows:
        human_message = build_human_message(row["prompt"])
        prompt_text = build_vlm_user_prompt(backend, human_message)
        field = str(row.get("output_field") or "")
        vocab = label_vocab.get(field, [])

        item: dict[str, Any] = {
            "input": {
                "row_index": row["index"],
                "id": row.get("id"),
                "category": row.get("category"),
                "subtype": row.get("subtype"),
                "phase": row.get("phase"),
                "instrument": row.get("instrument"),
                "verb": row.get("verb"),
                "target": row.get("target"),
                "output_field": field,
                "prompt": row["prompt"],
                "gold_response": row["gold_response"],
                "gold_labels": row["gold_labels"],
                "human_message": human_message,
                "prompt_text": prompt_text,
                "input_mode": "text_only",
            },
            "output": None,
        }
        try:
            pred_raw = generate_language_answer(backend, prompt_text, **gen_kw)
        except Exception as e:
            item["error"] = str(e)
            results.append(item)
            print(f"[{len(results)}/{len(rows)}] ERROR: {e}", file=sys.stderr)
            continue

        pred_truncated = _truncate_predictions(pred_raw, k=args.max_pred_labels)
        scores = score_sample(row["gold_labels"], pred_truncated, label_vocab=vocab)
        item["output"] = {
            "text_raw": pred_raw,
            "text": pred_truncated,
            "truncated": pred_truncated != pred_raw,
        }
        item["evaluation"] = scores

        sum_f1 += scores["f1"]
        n_scored += 1
        print(
            f"[{n_scored}/{len(rows)}] {row.get('subtype')} "
            f"gold={scores['gold_terms']} pred={scores['pred_terms']} "
            f"F1={scores['f1']:.4f}",
            file=sys.stderr,
        )
        results.append(item)

    sample_avg_f1 = (sum_f1 / n_scored) if n_scored else 0.0
    cls_metrics = aggregate_classification_metrics(results, label_vocab)

    payload = {
        "task": "language_grounding_surgical_prompts",
        "dataset_json": str(json_path),
        "dataset_schema": "cholect50_triplet_completion_v1",
        "input_mode": "text_only",
        "image_policy": (
            "Text-only: generate_language_answer() → generate_text() on prismatic/HF/API. "
            "No PIL image or blank placeholder."
        ),
        "backend": args.backend,
        "model_id": meta.get("model_id")
        if meta.get("source") in ("local_checkpoint", "prismatic_checkpoint")
        else model_id,
        "hub_model_id_cli": model_id,
        "model_name": model_name,
        "vlm_load": meta,
        "filters": {
            "category": args.filter_category,
            "subtype": args.filter_subtype,
            "multi_subtypes": multi_list,
            "per_subtype_limit": args.per_subtype_limit,
            "limit": args.limit,
        },
        "generation": {
            "do_sample": args.do_sample,
            "temperature": args.temperature if args.do_sample else None,
            "max_new_tokens": args.max_new_tokens,
            "max_pred_labels": args.max_pred_labels,
        },
        "metrics": {
            "f1": "sample-averaged multi-label F1 (in-vocab terms only, ≤max_pred_labels)",
            "auroc": "macro AUROC over vocab + others (0/1 parsing, AUC fallback=0.5)",
            "map": "macro mAP over vocab + others (others GT always 0)",
        },
        "counts": {
            "rows_in_dataset_file": len(raw_data),
            "rows_after_filters": len(rows),
            "rows_scored": n_scored,
            "sample_averaged_f1": sample_avg_f1,
        },
        "breakdown_f1": aggregate_f1_by_subtype(results),
        "classification_metrics": cls_metrics,
        "results": results,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(
        f"\nWrote {out_path}\n"
        f"  sample_averaged_F1={sample_avg_f1:.4f}  ({n_scored} questions)\n"
        f"  overall_macro_AUROC={cls_metrics.get('overall_macro_auroc')}\n"
        f"  overall_macro_mAP={cls_metrics.get('overall_macro_map')}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
