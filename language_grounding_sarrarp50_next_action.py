"""
language_grounding_sarrarp50_next_action.py

Language-only next-action planning on SAR-RARP50 action sequences (no image/video).

Prompts live under eval/prompts/sarrarp50_next_action/ (build with
build_sarrarp50_next_action_prompts.py).

Usage:
  BACKEND=qwen3-4b bash grounding_task.sh language_grounding_sarrarp50_next_action --limit 20
  python language_grounding_sarrarp50_next_action.py --metrics-only outputs/.../sarrarp50_next_action.json
"""

from __future__ import annotations

import argparse
import json
import re
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
from utils import resolve_device, strip_lora_answer_tags

_SCRIPT_ROOT = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_ROOT.parent
_DEFAULT_JSON = (
    _REPO_ROOT
    / "eval"
    / "prompts"
    / "sarrarp50_next_action"
    / "sarrarp50_next_action_prompts.json"
)
_DEFAULT_HF_TOKEN = _SCRIPT_ROOT / ".hf_token"
_DEFAULT_OUTPUT_ROOT = _SCRIPT_ROOT / "outputs" / "language_grounding_sarrarp50_next_action"

ACTION_DISPLAY_NAMES: dict[int, str] = {
    0: "Other",
    1: "Picking-up the needle",
    2: "Positioning the needle tip",
    3: "Pushing the needle through the tissue",
    4: "Pulling the needle out of the tissue",
    5: "Tying a knot",
    6: "Cutting the suture",
    7: "Returning/dropping the needle",
}

OTHER_ACTION_ID = 0
ACTION_OPTIONS: list[str] = [
    ACTION_DISPLAY_NAMES[i] for i in sorted(ACTION_DISPLAY_NAMES) if i != OTHER_ACTION_ID
]


def _normalize_display_key(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (text or "").strip().lower())


CANONICAL_TO_DISPLAY: dict[str, str] = {f"a{i}": ACTION_DISPLAY_NAMES[i] for i in ACTION_DISPLAY_NAMES}
CANONICAL_TO_ID: dict[str, int] = {f"a{i}": i for i in ACTION_DISPLAY_NAMES}


def _normalize_label_key(term: str) -> str:
    return re.sub(r"[\s\-_]+", "-", (term or "").strip().lower())


def build_label_vocabulary() -> list[str]:
    return [_normalize_label_key(name) for name in ACTION_OPTIONS]


def build_option_map() -> dict[str, str]:
    out: dict[str, str] = {}
    for i, name in ACTION_DISPLAY_NAMES.items():
        canonical = f"a{i}"
        out[canonical] = canonical
        out[str(i)] = canonical
        out[_normalize_display_key(name)] = canonical
        out[_normalize_label_key(name)] = canonical
        letter = chr(ord("A") + i)
        out[letter.upper()] = canonical
        out[letter.lower()] = canonical
    return out


def _match_action_token(token: str, option_map: dict[str, str]) -> str | None:
    t = (token or "").strip()
    if not t:
        return None
    if len(t) == 1 and t.upper() in option_map:
        return option_map[t.upper()]
    key = _normalize_display_key(t)
    if key in option_map:
        return option_map[key]
    norm = _normalize_label_key(t)
    if norm in option_map:
        return option_map[norm]
    for canonical, disp in CANONICAL_TO_DISPLAY.items():
        if disp.lower() == t.lower():
            return canonical
        if disp.lower() in t.lower() or t.lower() in disp.lower():
            return canonical
    return None


def parse_action_response(text: str, *, option_map: dict[str, str]) -> dict[str, Any]:
    raw = strip_lora_answer_tags(text)
    action_canonical: str | None = None

    m = re.search(r"action\s*[:=]\s*([^\n.;]+)", raw, re.IGNORECASE)
    if m:
        action_canonical = _match_action_token(m.group(1), option_map)

    if action_canonical is None:
        for line in raw.splitlines():
            s = line.strip()
            s = re.sub(r"^[-*•]\s*", "", s)
            s = re.sub(r"^\d+[.)]\s*", "", s).strip()
            hit = _match_action_token(s, option_map)
            if hit:
                action_canonical = hit
                break

    if action_canonical is None:
        for part in re.split(r"[,;\n]", raw):
            hit = _match_action_token(part, option_map)
            if hit:
                action_canonical = hit
                break

    if action_canonical is None:
        action_canonical = _match_action_token(raw, option_map)

    action_id = CANONICAL_TO_ID.get(action_canonical) if action_canonical else None
    return {
        "action_canonical": action_canonical,
        "action_id": action_id,
        "action_display": CANONICAL_TO_DISPLAY.get(action_canonical or "") or None,
        "raw": raw,
    }


def score_sample(
    gold_display: str,
    gold_id: int | None,
    pred_parsed: dict[str, Any],
) -> dict[str, Any]:
    gold_canonical = f"a{gold_id}" if gold_id is not None else None
    pred_canonical = pred_parsed.get("action_canonical")
    pred_id = pred_parsed.get("action_id")
    exact_id = (
        gold_id is not None
        and pred_id is not None
        and int(gold_id) == int(pred_id)
    )
    exact_canonical = (
        gold_canonical is not None
        and pred_canonical is not None
        and gold_canonical == pred_canonical
    )
    exact_display = (
        pred_parsed.get("action_display") or ""
    ).strip().lower() == gold_display.strip().lower()
    return {
        "gold_response": gold_display,
        "gold_id": gold_id,
        "gold_canonical": gold_canonical,
        "pred_response": pred_parsed.get("raw") or "",
        "pred_display": pred_parsed.get("action_display"),
        "pred_id": pred_id,
        "pred_canonical": pred_canonical,
        "exact_match_id": exact_id,
        "exact_match_canonical": exact_canonical,
        "exact_match_display": exact_display,
        "correct": exact_id,
    }


def build_human_message(prompt: str) -> str:
    body = prompt.strip()
    instr = (
        "Answer with exactly one short action phrase from the listed options. "
        "No numbering, bullets, or explanation."
    )
    if "available action options" in body.lower():
        return body
    return f"{body}\n\n{instr}"


def load_prompt_rows(
    raw: list[Any],
    source_label: str,
    *,
    filter_video: str | None,
    filter_history_mode: str | None,
    filter_template_id: int | None,
) -> list[dict[str, Any]]:
    if not isinstance(raw, list):
        raise ValueError(f"{source_label}: expected JSON array at root")

    rows: list[dict[str, Any]] = []
    for i, obj in enumerate(raw):
        if not isinstance(obj, dict):
            continue
        response = obj.get("response")
        if not isinstance(response, str) or not response.strip():
            continue
        prompt = obj.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            continue

        video = str(obj.get("video") or "")
        if filter_video is not None and video != filter_video:
            continue
        history_mode = str(obj.get("history_mode") or "")
        if filter_history_mode is not None and history_mode != filter_history_mode:
            continue
        template_id = obj.get("template_id")
        if filter_template_id is not None and template_id != filter_template_id:
            continue

        rows.append(
            {
                "index": i,
                "id": obj.get("id"),
                "numeric_id": obj.get("numeric_id"),
                "video": video,
                "video_num": obj.get("video_num"),
                "surgery": obj.get("surgery"),
                "phase": obj.get("phase"),
                "segment_index": obj.get("segment_index"),
                "history_mode": history_mode,
                "template_id": template_id,
                "history_actions": obj.get("history_actions") or [],
                "output_field": str(obj.get("output_field") or "action"),
                "prompt": prompt.strip(),
                "gold_response": response.strip(),
                "gold_id": obj.get("response_id"),
                "gold_canonical": obj.get("response_canonical"),
            }
        )
    return rows


def aggregate_metrics(results: list[dict[str, Any]]) -> dict[str, Any]:
    def _acc(items: list[dict[str, Any]], key: str) -> float | None:
        scored = [r for r in items if isinstance(r.get("evaluation"), dict)]
        if not scored:
            return None
        hits = sum(1 for r in scored if r["evaluation"].get(key))
        return hits / len(scored)

    by_video: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_history: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_template: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for rec in results:
        inp = rec.get("input") or {}
        by_video[str(inp.get("video") or "unknown")].append(rec)
        by_history[str(inp.get("history_mode") or "unknown")].append(rec)
        by_template[str(inp.get("template_id") or "unknown")].append(rec)

    def _block(items: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "count": len(items),
            "accuracy_id": _acc(items, "correct"),
            "accuracy_display": _acc(items, "exact_match_display"),
        }

    return {
        "overall": _block(results),
        "by_video": {k: _block(v) for k, v in sorted(by_video.items())},
        "by_history_mode": {k: _block(v) for k, v in sorted(by_history.items())},
        "by_template_id": {k: _block(v) for k, v in sorted(by_template.items())},
    }


def recompute_metrics_from_results(payload: dict[str, Any], option_map: dict[str, str]) -> dict[str, Any]:
    results = payload.get("results") or []
    n_correct = 0
    n_scored = 0
    for rec in results:
        if rec.get("error"):
            continue
        inp = rec.get("input") or {}
        out = rec.get("output") or {}
        pred_raw = out.get("text") or out.get("text_raw") or ""
        pred_parsed = parse_action_response(str(pred_raw), option_map=option_map)
        scores = score_sample(
            str(inp.get("gold_response") or ""),
            inp.get("gold_id"),
            pred_parsed,
        )
        rec["evaluation"] = scores
        rec["parsed"] = pred_parsed
        if scores["correct"]:
            n_correct += 1
        n_scored += 1

    payload["counts"] = {
        **(payload.get("counts") or {}),
        "rows_scored": n_scored,
        "accuracy": (n_correct / n_scored) if n_scored else 0.0,
    }
    payload["breakdown"] = aggregate_metrics(results)
    return payload


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Language-only SAR-RARP50 next-action planning (no image).",
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
    p.add_argument("--filter-video", type=str, default=None)
    p.add_argument("--filter-history-mode", type=str, default=None)
    p.add_argument("--filter-template-id", type=int, default=None)
    p.add_argument("--do-sample", action="store_true")
    p.add_argument("--temperature", type=float, default=0.5)
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument(
        "--metrics-only",
        type=Path,
        default=None,
        help="Recompute accuracy from an existing results JSON (skip VLM).",
    )
    return p.parse_args()


def _place_backend_on_device(backend: Any, *, backend_name: str, device_str: str) -> None:
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
    option_map = build_option_map()
    label_vocab = build_label_vocabulary()

    json_path = args.dataset_json.resolve()
    if not json_path.is_file():
        raise FileNotFoundError(
            f"{json_path} not found. Run eval/prompts/build_sarrarp50_next_action_prompts.py first."
        )
    raw_data = json.loads(json_path.read_text(encoding="utf-8"))

    if args.metrics_only is not None:
        mp = args.metrics_only.resolve()
        if not mp.is_file():
            raise FileNotFoundError(mp)
        payload = json.loads(mp.read_text(encoding="utf-8"))
        payload = recompute_metrics_from_results(payload, option_map)
        with mp.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        acc = (payload.get("counts") or {}).get("accuracy")
        print(f"Updated metrics in {mp}  accuracy={acc}", file=sys.stderr)
        return

    rows = load_prompt_rows(
        raw_data,
        str(json_path),
        filter_video=args.filter_video,
        filter_history_mode=args.filter_history_mode,
        filter_template_id=args.filter_template_id,
    )
    if args.limit is not None:
        rows = rows[: max(0, args.limit)]

    model_id = resolve_model_id(args.backend, args.model_id)
    model_name = resolve_output_model_name(args.backend, args.model_name, model_id)
    hf_token = None if is_api_backend(args.backend) else resolve_hf_token(args.backend, args.hf_token)

    out_path = (
        args.output.resolve()
        if args.output is not None
        else (
            args.output_root
            / f"lang_{args.backend}_{model_name}"
            / "sarrarp50_next_action.json"
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
    n_correct = 0
    n_scored = 0

    for row in rows:
        human_message = build_human_message(row["prompt"])
        prompt_text = build_vlm_user_prompt(backend, human_message)

        item: dict[str, Any] = {
            "input": {
                "row_index": row["index"],
                "id": row.get("id"),
                "numeric_id": row.get("numeric_id"),
                "video": row.get("video"),
                "video_num": row.get("video_num"),
                "surgery": row.get("surgery"),
                "phase": row.get("phase"),
                "segment_index": row.get("segment_index"),
                "history_mode": row.get("history_mode"),
                "template_id": row.get("template_id"),
                "history_actions": row.get("history_actions"),
                "output_field": row.get("output_field"),
                "prompt": row["prompt"],
                "gold_response": row["gold_response"],
                "gold_id": row.get("gold_id"),
                "gold_canonical": row.get("gold_canonical"),
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

        pred_parsed = parse_action_response(pred_raw, option_map=option_map)
        scores = score_sample(row["gold_response"], row.get("gold_id"), pred_parsed)
        item["output"] = {"text_raw": pred_raw, "text": pred_parsed.get("raw") or pred_raw}
        item["parsed"] = pred_parsed
        item["evaluation"] = scores

        if scores["correct"]:
            n_correct += 1
        n_scored += 1
        print(
            f"[{n_scored}/{len(rows)}] {row.get('video')} seg={row.get('segment_index')} "
            f"gold={row['gold_response']!r} pred={pred_parsed.get('action_display')!r} "
            f"ok={scores['correct']}",
            file=sys.stderr,
        )
        results.append(item)

    accuracy = (n_correct / n_scored) if n_scored else 0.0
    breakdown = aggregate_metrics(results)

    payload = {
        "task": "language_grounding_sarrarp50_next_action",
        "dataset_json": str(json_path),
        "dataset_schema": "sarrarp50_next_action_planning_v1",
        "input_mode": "text_only",
        "image_policy": "No image or video input; text-only generate_language_answer().",
        "backend": args.backend,
        "model_id": meta.get("model_id")
        if meta.get("source") in ("local_checkpoint", "prismatic_checkpoint")
        else model_id,
        "hub_model_id_cli": model_id,
        "model_name": model_name,
        "vlm_load": meta,
        "filters": {
            "video": args.filter_video,
            "history_mode": args.filter_history_mode,
            "template_id": args.filter_template_id,
            "limit": args.limit,
        },
        "generation": {
            "do_sample": args.do_sample,
            "temperature": args.temperature if args.do_sample else None,
            "max_new_tokens": args.max_new_tokens,
        },
        "metrics": {
            "accuracy": "exact match on action class id (1–7; Other excluded)",
            "action_vocabulary": ACTION_OPTIONS,
            "label_vocabulary_normalized": label_vocab,
        },
        "counts": {
            "rows_in_dataset_file": len(raw_data),
            "rows_after_filters": len(rows),
            "rows_scored": n_scored,
            "rows_correct": n_correct,
            "accuracy": accuracy,
        },
        "breakdown": breakdown,
        "results": results,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(
        f"\nWrote {out_path}\n"
        f"  accuracy={accuracy:.4f}  ({n_correct}/{n_scored})",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
