"""
triplet_recognition_cholect50.py

CholecT50 triplet recognition (instrument, verb, target) — single-prompt evaluation.

  - Prompt: instrument + verb + target in one question (bench figure style)
  - --prompt-mode mcq: MCQ option lists (--mcq-option-format list|lettered)
  - --prompt-mode ov:  open vocabulary (no option list)
  - Metrics: per-component Accuracy, Triplet Accuracy, per-component mAP
  - Default: evaluate all triplet annotations (--eval-all)
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from backends import load_backend
from cholect50_data import (
    CHALLENGE_VAL_ROOT,
    _DEFAULT_MODEL_IDS,
    collect_instrument_annotations,
    discover_video_roots,
    infer_pil_side,
    sample_by_instrument,
)
from utils import (
    ACTION_OPTIONS_FIXED,
    CHOLECT_ROOT,
    TARGET_OPTIONS_FIXED,
    load_results_for_resume,
    normalize_instrument_name,
    resolve_device,
    upsert_result,
)

_SCRIPT_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_ROOT = _SCRIPT_ROOT / "outputs" / "triplet_recognition_cholect50"
_DEFAULT_HF_TOKEN = _SCRIPT_ROOT / ".hf_token"

INSTRUMENT_OPTIONS = [
    "grasper", "bipolar", "hook", "scissors", "clipper", "irrigator",
]
VERB_OPTIONS = list(ACTION_OPTIONS_FIXED)
TARGET_OPTIONS = list(TARGET_OPTIONS_FIXED)

def _canonical(s: str | None) -> str:
    t = (s or "").strip().lower()
    t = t.replace("null-verb", "null_verb").replace("null-target", "null_target")
    t = t.replace("abd-wall/cavity", "abdominal_wall_cavity")
    t = t.replace("cystic-plate", "cystic_plate").replace("cystic-duct", "cystic_duct")
    t = t.replace("cystic-artery", "cystic_artery").replace("cystic-pedicle", "cystic_pedicle")
    t = t.replace("blood-vessel", "blood_vessel").replace("specimen-bag", "specimen_bag")
    return normalize_instrument_name(t)


def cholect_display_label(name: str) -> str:
    key = _canonical(name)
    mapping = {
        "abdominal_wall_cavity": "abd-wall/cavity",
        "null_verb": "null-verb",
        "null_target": "null-target",
    }
    if key in mapping:
        return mapping[key]
    return (name or "").strip().replace("_", "-")


def _build_name_option_map(options: list[str]) -> dict[str, str]:
    """Map canonical / display spellings -> canonical option id (no A/B/C letters)."""
    out: dict[str, str] = {}
    for opt in options:
        disp = cholect_display_label(opt)
        out[_canonical(opt)] = opt
        out[_canonical(disp)] = opt
    return out


def _format_option_list_block(title: str, options: list[str]) -> tuple[str, dict[str, str]]:
    """Comma-separated option list only (no A, B, C prefixes)."""
    labels = [cholect_display_label(o) for o in options if str(o).strip()]
    line = f"{title}: {', '.join(labels)}"
    return line, _build_name_option_map(options)


def _format_lettered_block(title: str, options: list[str]) -> tuple[str, dict[str, str]]:
    """Lettered block (A. …, B. …) plus map for letter and name matching."""
    lines = [f"{title}:"]
    letter_to_option: dict[str, str] = {}
    name_map = _build_name_option_map(options)
    for i, opt in enumerate(options):
        letter = chr(ord("A") + i)
        disp = cholect_display_label(opt)
        lines.append(f"{letter}. {disp}")
        letter_to_option[letter.upper()] = opt
        letter_to_option[letter.lower()] = opt
    return "\n".join(lines), {**letter_to_option, **name_map}


def _build_output_format_instruction(*, mcq: bool) -> str:
    """Structured triplet blocks; field names 'instrument' / 'target' / 'verb' highlighted."""
    lines = [
        "Answer using the following structure only. "
        "List every instrument–verb–target triplet you see (one triplet per block). "
        "Use these field names exactly:",
        "",
        "'instrument': <instrument name>",
        "'target': <anatomical target>",
        "'verb': <surgical verb>",
        "",
        "Example (single triplet):",
        "'instrument': grasper",
        "'target': gallbladder",
        "'verb': grasp",
        "",
        "If there are multiple triplets, separate each block with a blank line. "
        "Do not use bullet numbers or extra prose outside these lines.",
    ]
    if mcq:
        lines.insert(
            1,
            "Each value must be chosen from the corresponding 'instrument', 'verb', or 'target' option list below.",
        )
    return "\n".join(lines)


def build_triplet_recognition_prompt(
    *,
    prompt_mode: str,
    mcq_option_format: str = "list",
) -> tuple[str, dict[str, Any]]:
    """
    Bench-style single triplet question.
    MCQ: instrument / verb / target option lists (--mcq-option-format).
    OV:  no option lists.
    """
    mode = (prompt_mode or "mcq").strip().lower()
    opt_fmt = (mcq_option_format or "list").strip().lower()
    meta: dict[str, Any] = {"prompt_mode": mode, "mcq_option_format": opt_fmt}

    core = (
        "What tasks are the instruments accomplishing with the targets in this surgical image? "
        "There may be more than one 'instrument'–'verb'–'target' triplet in the frame."
    )
    output_fmt = _build_output_format_instruction(mcq=(mode == "mcq"))

    if mode == "ov":
        body = f"{core}\n\n{output_fmt}"
        return body, meta

    if mode != "mcq":
        raise ValueError(f"Unknown --prompt-mode {prompt_mode!r}; choose mcq or ov.")

    if opt_fmt not in ("list", "lettered"):
        raise ValueError(f"Unknown --mcq-option-format {mcq_option_format!r}; choose list or lettered.")

    fmt_block = _format_option_list_block if opt_fmt == "list" else _format_lettered_block
    inst_block, inst_map = fmt_block("'instrument' options", INSTRUMENT_OPTIONS)
    verb_block, verb_map = fmt_block("'verb' options", VERB_OPTIONS)
    tgt_block, tgt_map = fmt_block("'target' options", TARGET_OPTIONS)
    meta["option_maps"] = {
        "instrument": inst_map,
        "verb": verb_map,
        "target": tgt_map,
    }

    body = (
        f"{core}\n\n"
        f"{output_fmt}\n\n"
        "The available 'instrument', 'verb', and 'target' options are:\n\n"
        f"{inst_block}\n\n{verb_block}\n\n{tgt_block}"
    )
    return body, meta


def wrap_vlm_prompt(body: str) -> str:
    return body.strip()


def _split_terms(text: str) -> list[str]:
    if not text or not str(text).strip():
        return []
    parts = re.split(r"[,;\n]+", str(text))
    out: list[str] = []
    for p in parts:
        t = p.strip()
        t = re.sub(r"^[-*•]\s*", "", t)
        t = re.sub(r"^\d+[.)]\s*", "", t)
        t = re.sub(r"^[A-Za-z][.)]\s*", "", t).strip()
        if t:
            out.append(t)
    return out[:3]


def _match_option_token(token: str, option_map: dict[str, str], options: list[str]) -> str | None:
    t = (token or "").strip()
    if not t:
        return None
    letter = t.upper()
    if len(letter) == 1 and letter in option_map:
        return option_map[letter]
    ck = _canonical(t)
    if ck in option_map:
        return option_map[ck]
    for o in options:
        if _canonical(o) == ck:
            return o
    raw = t.lower()
    for key, orig in option_map.items():
        if len(key) == 1:
            continue
        if key in raw or raw in key:
            return orig
    return None


def parse_mcq_terms(text: str, options: list[str]) -> list[str]:
    norm_map: dict[str, str] = {}
    for o in options:
        k = _canonical(o)
        if k:
            norm_map[k] = o

    matched: list[str] = []
    seen: set[str] = set()
    for term in _split_terms(text):
        hit = _match_option_token(term, norm_map, options)
        if hit:
            ck = _canonical(hit)
            if ck not in seen:
                seen.add(ck)
                matched.append(hit)
    if matched:
        return matched[:3]

    raw = (text or "").lower()
    for key, orig in norm_map.items():
        if len(key) == 1:
            continue
        if key in raw and key not in seen:
            seen.add(key)
            matched.append(orig)
        if len(matched) >= 3:
            break
    return matched[:3]


def _extract_labeled_field(text: str, labels: tuple[str, ...]) -> str | None:
    for lab in labels:
        m = re.search(
            rf"(?:^|\n)\s*['\"]?{re.escape(lab)}['\"]?\s*[:=]\s*([^\n,;]+)",
            text,
            re.IGNORECASE,
        )
        if m:
            val = m.group(1).strip().strip("'\"")
            return val if val else None
    return None


_FIELD_LINE_RE = re.compile(
    r"^\s*['\"]?(instrument|instruments|target|targets|action|actions|verb|verbs)['\"]?\s*[:=]\s*(.+?)\s*$",
    re.IGNORECASE,
)


def _field_key_to_role(key: str) -> str | None:
    k = (key or "").strip().lower().rstrip("s")
    if k in ("instrument",):
        return "instrument"
    if k in ("target",):
        return "target"
    if k in ("action", "verb"):
        return "verb"
    return None


def _finalize_labeled_triplet(
    fields: dict[str, str],
    *,
    prompt_mode: str,
    prompt_meta: dict[str, Any],
) -> dict[str, str | None] | None:
    if not fields:
        return None
    return _parse_one_triplet_tokens(
        fields.get("instrument"),
        fields.get("verb"),
        fields.get("target"),
        prompt_mode=prompt_mode,
        prompt_meta=prompt_meta,
    )


def _parse_labeled_triplet_blocks(
    raw: str,
    *,
    prompt_mode: str,
    prompt_meta: dict[str, Any],
) -> list[dict[str, str | None]]:
    """
    Parse blocks like:
      'instrument': grasper
      'target': gallbladder
      'verb': grasp
    """
    triplets: list[dict[str, str | None]] = []
    seen: set[tuple[str, str, str]] = set()
    current: dict[str, str] = {}

    def flush() -> None:
        nonlocal current
        if not current:
            return
        t = _finalize_labeled_triplet(current, prompt_mode=prompt_mode, prompt_meta=prompt_meta)
        current = {}
        if not t or not (t.get("instrument") and t.get("verb") and t.get("target")):
            return
        key = (_canonical(t["instrument"]), _canonical(t["verb"]), _canonical(t["target"]))
        if key in seen:
            return
        seen.add(key)
        triplets.append(t)

    for line in raw.splitlines():
        s = line.strip()
        if not s:
            flush()
            continue
        m = _FIELD_LINE_RE.match(s)
        if m:
            role = _field_key_to_role(m.group(1))
            val = m.group(2).strip().strip("'\"")
            if role and val:
                if role in current and current[role] != val:
                    flush()
                current[role] = val
            continue
        if current:
            flush()

    flush()
    return triplets


def _parse_one_triplet_tokens(
    inst_tok: str | None,
    verb_tok: str | None,
    tgt_tok: str | None,
    *,
    prompt_mode: str,
    prompt_meta: dict[str, Any],
) -> dict[str, str | None]:
    mode = (prompt_mode or "mcq").strip().lower()
    meta = prompt_meta or {}

    if mode == "mcq":
        maps = meta.get("option_maps") or {}
        imap = maps.get("instrument") or {}
        vmap = maps.get("verb") or {}
        tmap = maps.get("target") or {}
        inst = _match_option_token(inst_tok or "", imap, INSTRUMENT_OPTIONS) if inst_tok else None
        verb = _match_option_token(verb_tok or "", vmap, VERB_OPTIONS) if verb_tok else None
        tgt = _match_option_token(tgt_tok or "", tmap, TARGET_OPTIONS) if tgt_tok else None
        if not (inst and verb and tgt):
            parts = _split_terms(",".join(x for x in (inst_tok, verb_tok, tgt_tok) if x))
            if len(parts) >= 3:
                inst = inst or _match_option_token(parts[0], imap, INSTRUMENT_OPTIONS)
                verb = verb or _match_option_token(parts[1], vmap, VERB_OPTIONS)
                tgt = tgt or _match_option_token(parts[2], tmap, TARGET_OPTIONS)
        return {"instrument": inst, "verb": verb, "target": tgt}

    inst = parse_mcq_terms(inst_tok or "", INSTRUMENT_OPTIONS)
    verb = parse_mcq_terms(verb_tok or "", VERB_OPTIONS)
    tgt = parse_mcq_terms(tgt_tok or "", TARGET_OPTIONS)
    line = ",".join(x for x in (inst_tok, verb_tok, tgt_tok) if x)
    parts = _split_terms(line) if line else []
    if len(parts) >= 3:
        inst = inst or parse_mcq_terms(parts[0], INSTRUMENT_OPTIONS)
        verb = verb or parse_mcq_terms(parts[1], VERB_OPTIONS)
        tgt = tgt or parse_mcq_terms(parts[2], TARGET_OPTIONS)
    return {
        "instrument": inst[0] if inst else None,
        "verb": verb[0] if verb else None,
        "target": tgt[0] if tgt else None,
    }


def _parse_triplet_lines_from_text(
    raw: str,
    *,
    prompt_mode: str,
    prompt_meta: dict[str, Any],
) -> list[dict[str, str | None]]:
    """Parse zero or more triplets from structured or free-form response."""
    labeled = _parse_labeled_triplet_blocks(raw, prompt_mode=prompt_mode, prompt_meta=prompt_meta)
    if labeled:
        return labeled

    triplets: list[dict[str, str | None]] = []
    seen: set[tuple[str, str, str]] = set()

    def add(t: dict[str, str | None]) -> None:
        if not (t.get("instrument") and t.get("verb") and t.get("target")):
            return
        key = (_canonical(t["instrument"]), _canonical(t["verb"]), _canonical(t["target"]))
        if key in seen:
            return
        seen.add(key)
        triplets.append(t)

    for line in raw.splitlines():
        s = line.strip()
        if not s:
            continue
        s = re.sub(r"^[-*•]\s*", "", s)
        s = re.sub(r"^\d+[.)]\s*", "", s).strip()
        parts = _split_terms(s)
        if len(parts) >= 3:
            if prompt_mode == "mcq":
                maps = prompt_meta.get("option_maps") or {}
                t = {
                    "instrument": _match_option_token(parts[0], maps.get("instrument") or {}, INSTRUMENT_OPTIONS),
                    "verb": _match_option_token(parts[1], maps.get("verb") or {}, VERB_OPTIONS),
                    "target": _match_option_token(parts[2], maps.get("target") or {}, TARGET_OPTIONS),
                }
            else:
                inst = parse_mcq_terms(parts[0], INSTRUMENT_OPTIONS)
                verb = parse_mcq_terms(parts[1], VERB_OPTIONS)
                tgt = parse_mcq_terms(parts[2], TARGET_OPTIONS)
                t = {
                    "instrument": inst[0] if inst else None,
                    "verb": verb[0] if verb else None,
                    "target": tgt[0] if tgt else None,
                }
            add(t)

    if triplets:
        return triplets

    inst_tok = _extract_labeled_field(raw, ("instrument", "instruments"))
    verb_tok = _extract_labeled_field(raw, ("verb", "verbs", "action", "actions"))
    tgt_tok = _extract_labeled_field(raw, ("target", "targets"))
    if inst_tok or verb_tok or tgt_tok:
        add(_parse_one_triplet_tokens(
            inst_tok, verb_tok, tgt_tok, prompt_mode=prompt_mode, prompt_meta=prompt_meta,
        ))
    return triplets


def parse_triplet_response(
    text: str,
    *,
    prompt_mode: str,
    prompt_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Parse all triplets mentioned in the model response."""
    meta = prompt_meta or {}
    raw = text or ""
    triplets = _parse_triplet_lines_from_text(raw, prompt_mode=prompt_mode, prompt_meta=meta)

    out: dict[str, Any] = {"triplets": triplets, "instrument": None, "verb": None, "target": None}
    if triplets:
        out["instrument"] = triplets[0]["instrument"]
        out["verb"] = triplets[0]["verb"]
        out["target"] = triplets[0]["target"]
    return out


def _average_precision(y_true: list[int], y_score: list[float]) -> float:
    if not y_true or sum(y_true) == 0:
        return 0.0
    pairs = sorted(zip(y_score, y_true), key=lambda x: (-x[0], -x[1]))
    tp = fp = 0
    precisions: list[float] = []
    n_pos = sum(y_true)
    for _score, label in pairs:
        if label:
            tp += 1
            precisions.append(tp / (tp + fp))
        else:
            fp += 1
    return sum(precisions) / n_pos if n_pos else 0.0


def mean_ap_multilabel(
    gold_sets: list[set[str]],
    pred_sets: list[set[str]],
    classes: list[str],
) -> float | None:
    if not gold_sets:
        return None
    aps: list[float] = []
    for cls in classes:
        ckey = _canonical(cls)
        y_true = [1 if ckey in gs else 0 for gs in gold_sets]
        y_score = [1.0 if ckey in ps else 0.0 for ps in pred_sets]
        aps.append(_average_precision(y_true, y_score))
    return sum(aps) / len(aps) if aps else None


def _label_context_from_parsed(parsed: dict[str, Any]) -> dict[str, Any]:
    return {
        "triplet_id": parsed["triplet_id"],
        "triplet_str": parsed["triplet_str"],
        "label_instrument": parsed["instrument_name"],
        "label_verb": parsed["verb_name"],
        "label_target": parsed["target_name"],
        "label_phase_id": parsed["phase_id"],
        "label_phase_name": parsed["phase_name"],
    }


def _predicted_triplets(parsed: dict[str, Any]) -> list[dict[str, str | None]]:
    raw = parsed.get("triplets")
    if isinstance(raw, list) and raw:
        return [t for t in raw if isinstance(t, dict)]
    if parsed.get("instrument") or parsed.get("verb") or parsed.get("target"):
        return [{
            "instrument": parsed.get("instrument"),
            "verb": parsed.get("verb"),
            "target": parsed.get("target"),
        }]
    return []


def _score_record(rec: dict[str, Any]) -> dict[str, Any] | None:
    inp = rec.get("input") or {}
    lc = inp.get("label_context") or {}
    out = rec.get("output")
    if not isinstance(out, dict):
        return None
    parsed = out.get("parsed") or {}
    triplets = _predicted_triplets(parsed)

    gi = str(lc.get("label_instrument") or "")
    gv = str(lc.get("label_verb") or "")
    gt = str(lc.get("label_target") or "")
    g_inst, g_verb, g_tgt = _canonical(gi), _canonical(gv), _canonical(gt)

    pred_inst = {_canonical(t["instrument"]) for t in triplets if t.get("instrument")}
    pred_verb = {_canonical(t["verb"]) for t in triplets if t.get("verb")}
    pred_tgt = {_canonical(t["target"]) for t in triplets if t.get("target")}

    inst_correct = bool(g_inst and g_inst in pred_inst)
    verb_correct = bool(g_verb and g_verb in pred_verb)
    tgt_correct = bool(g_tgt and g_tgt in pred_tgt)
    triplet_correct = any(
        g_inst
        and g_verb
        and g_tgt
        and _canonical(t.get("instrument")) == g_inst
        and _canonical(t.get("verb")) == g_verb
        and _canonical(t.get("target")) == g_tgt
        for t in triplets
    )

    primary = triplets[0] if triplets else {}

    return {
        "gold_instrument": gi,
        "gold_verb": gv,
        "gold_target": gt,
        "pred_triplets": triplets,
        "pred_instrument": primary.get("instrument"),
        "pred_verb": primary.get("verb"),
        "pred_target": primary.get("target"),
        "n_pred_triplets": len(triplets),
        "instrument_correct": inst_correct,
        "verb_correct": verb_correct,
        "target_correct": tgt_correct,
        "triplet_correct": triplet_correct,
    }


def aggregate_metrics(results: list[dict[str, Any]]) -> dict[str, Any]:
    evs = [r["evaluation"] for r in results if r.get("evaluation")]
    if not evs:
        return {"n_results": len(results), "n_scored": 0}

    def acc_block(field: str) -> dict[str, Any]:
        ok = [e for e in evs if e.get(field) is not None]
        if not ok:
            return {"correct": 0, "total": 0, "accuracy": None}
        c = sum(1 for e in ok if e.get(field))
        return {"correct": c, "total": len(ok), "accuracy": c / len(ok)}

    inst_gold = [{_canonical(e["gold_instrument"])} for e in evs if e.get("gold_instrument")]
    verb_gold = [{_canonical(e["gold_verb"])} for e in evs if e.get("gold_verb")]
    tgt_gold = [{_canonical(e["gold_target"])} for e in evs if e.get("gold_target")]

    def _pred_component_sets(key: str) -> list[set[str]]:
        out_sets: list[set[str]] = []
        for e in evs:
            s: set[str] = set()
            for t in e.get("pred_triplets") or []:
                if isinstance(t, dict) and t.get(key):
                    s.add(_canonical(t[key]))
            if not s and e.get(f"pred_{key}"):
                s.add(_canonical(e[f"pred_{key}"]))
            out_sets.append(s)
        return out_sets

    inst_pred = _pred_component_sets("instrument")
    verb_pred = _pred_component_sets("verb")
    tgt_pred = _pred_component_sets("target")

    return {
        "protocol": "cholect50_triplet_recognition",
        "instrument_accuracy": acc_block("instrument_correct"),
        "verb_accuracy": acc_block("verb_correct"),
        "target_accuracy": acc_block("target_correct"),
        "triplet_accuracy": acc_block("triplet_correct"),
        "instrument_mAP": mean_ap_multilabel(inst_gold, inst_pred, INSTRUMENT_OPTIONS),
        "verb_mAP": mean_ap_multilabel(verb_gold, verb_pred, VERB_OPTIONS),
        "target_mAP": mean_ap_multilabel(tgt_gold, tgt_pred, TARGET_OPTIONS),
        "n_results": len(results),
        "n_scored": len(evs),
    }


def _row_key(sample: dict[str, Any]) -> tuple[str, str]:
    path_str = str(sample["img_path"])
    tool = (
        f"cholect50-triplet|{sample['vid']}|f{sample['frame_index']:06d}"
        f"|a{sample['ann_index']}|{sample['instrument_name']}"
    )
    return path_str, tool


def _group_samples_by_frame(items: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for it in items:
        grouped[str(it["img_path"])].append(it)
    return dict(grouped)


def _find_cached_frame_output(
    results: list[dict],
    image_path: str,
) -> dict[str, Any] | None:
    """Reuse an existing successful VLM output for this frame (resume)."""
    for rec in results:
        inp = rec.get("input") or {}
        if inp.get("image_path") != image_path:
            continue
        out = rec.get("output")
        if not isinstance(out, dict) or rec.get("error"):
            continue
        parsed = out.get("parsed") or {}
        if parsed.get("triplets"):
            return out
    return None


def _should_skip_resume(rec: dict, tool: str) -> bool:
    if rec.get("error"):
        return False
    inp = rec.get("input") or {}
    if inp.get("tool") != tool:
        return False
    out = rec.get("output")
    if not isinstance(out, dict):
        return False
    p = out.get("parsed") or {}
    return bool(p.get("triplets"))


def _run_vlm_on_frame(
    *,
    backend,
    pil_side: int,
    image_path: Path,
    user_prompt: str,
    prompt_meta: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Single VLM forward pass on one frame (shared across annotations on that frame)."""
    try:
        image = Image.open(image_path).convert("RGB")
        image = image.resize((pil_side, pil_side), resample=Image.Resampling.BICUBIC)

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

        triplet = parse_triplet_response(
            text,
            prompt_mode=args.prompt_mode,
            prompt_meta=prompt_meta,
        )

        return {"text": text, "parsed": triplet}
    except Exception as e:
        print(f"SKIP {image_path}: {e}", file=sys.stderr)
        return {"error": str(e)}


def _make_result_entry(
    *,
    sample: dict[str, Any],
    user_prompt: str,
    args: argparse.Namespace,
    frame_output: dict[str, Any] | None,
) -> dict[str, Any]:
    path_str, tool = _row_key(sample)
    label_context = _label_context_from_parsed(sample["parsed"])
    entry: dict[str, Any] = {
        "input": {
            "image_path": path_str,
            "tool": tool,
            "label_context": label_context,
            "eval_protocol": "cholect50_triplet_recognition",
            "prompt_mode": args.prompt_mode,
            "mcq_option_format": args.mcq_option_format if args.prompt_mode == "mcq" else None,
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
        description="CholecT50 triplet recognition (single-prompt, component + triplet metrics).",
    )
    p.add_argument("--backend", choices=("prismatic", "cosmos", "groot"), default="prismatic")
    p.add_argument("--dataset-root", type=Path, default=CHALLENGE_VAL_ROOT)
    p.add_argument("--videos-root", type=Path, default=None)
    p.add_argument("--cholect-root-fallback", type=Path, default=CHOLECT_ROOT)
    p.add_argument("--video", type=str, default=None)
    p.add_argument("--instrument", type=str, default=None)
    p.add_argument(
        "--prompt-mode",
        choices=("mcq", "ov"),
        default="mcq",
        help="mcq: option lists in prompt; ov: open vocabulary (no options).",
    )
    p.add_argument(
        "--mcq-option-format",
        choices=("list", "lettered"),
        default="list",
        help="With --prompt-mode mcq: list=comma-separated labels only; lettered=A. B. C. prefixes.",
    )
    p.add_argument(
        "--samples-per-instrument",
        type=int,
        default=10,
        help="Used only with --samples-only.",
    )
    p.add_argument(
        "--eval-all",
        action="store_true",
        default=True,
        help="Evaluate all triplet annotations (default).",
    )
    p.add_argument(
        "--samples-only",
        action="store_true",
        help="Sample up to --samples-per-instrument per instrument instead of full eval.",
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
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.samples_only:
        args.eval_all = False

    args.dataset_root = args.dataset_root.resolve()
    labels_dir = args.dataset_root / "labels"
    if not labels_dir.is_dir():
        raise FileNotFoundError(f"labels directory not found: {labels_dir}")

    video_roots = discover_video_roots(args)
    if not video_roots:
        raise FileNotFoundError(
            "프레임 이미지 루트를 찾지 못했습니다. --videos-root 또는 CHOLECT50_VIDEOS_ROOT 를 지정해 주세요."
        )

    user_prompt, prompt_meta = build_triplet_recognition_prompt(
        prompt_mode=args.prompt_mode,
        mcq_option_format=args.mcq_option_format,
    )

    model_name = re.sub(r"[^a-zA-Z0-9._-]+", "_", (args.model_name or "original").strip() or "original")
    out_root = args.output_root.resolve()
    mode_slug = args.prompt_mode
    if args.prompt_mode == "mcq":
        mode_slug = f"{mode_slug}_{args.mcq_option_format}"
    out_path = (
        args.output.resolve()
        if args.output is not None
        else (
            out_root
            / f"triplet_{args.backend}_{model_name}_{mode_slug}"
            / "cholect50_challenge_val_triplet.json"
        ).resolve()
    )

    all_items = collect_instrument_annotations(
        labels_dir=labels_dir,
        video_roots=video_roots,
        video_filter=args.video,
        instrument_filter=args.instrument,
    )
    if not all_items:
        raise RuntimeError("평가 가능한 annotation이 없습니다.")

    if args.eval_all:
        sampled = list(all_items)
        print(f"Full eval: {len(sampled)} annotations.", file=sys.stderr)
    else:
        sampled = sample_by_instrument(
            all_items,
            cap_per_instrument=max(0, args.samples_per_instrument),
            seed=int(args.seed),
        )
        print(
            f"Sampled eval: {len(sampled)} annotations "
            f"({args.samples_per_instrument} per instrument).",
            file=sys.stderr,
        )

    fmt_note = f", mcq_option_format={args.mcq_option_format}" if args.prompt_mode == "mcq" else ""
    print(f"CholecT50 triplet recognition: prompt_mode={args.prompt_mode}{fmt_note}.", file=sys.stderr)

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

    def _upsert_scored_entry(sample: dict[str, Any], frame_output: dict[str, Any] | None) -> None:
        row_key = _row_key(sample)
        _, tool = row_key
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

    by_frame = _group_samples_by_frame(sampled)
    n_frames = len(by_frame)
    print(
        f"Frame-level inference: {n_frames} unique frame(s), "
        f"{len(sampled)} annotation(s), one model load.",
        file=sys.stderr,
    )
    for path_str in sorted(by_frame.keys()):
        frame_items = by_frame[path_str]
        all_done = True
        if not args.force:
            for s in frame_items:
                _, tool = _row_key(s)
                rk = (path_str, tool)
                if rk not in key_to_idx or not _should_skip_resume(results[key_to_idx[rk]], tool):
                    all_done = False
                    break
        if all_done:
            for s in frame_items:
                _, tool = _row_key(s)
                rec = results[key_to_idx[(path_str, tool)]]
                inp = rec.setdefault("input", {})
                inp["label_context"] = _label_context_from_parsed(s["parsed"])
                ev = _score_record(rec)
                if ev:
                    rec["evaluation"] = ev
            continue

        frame_output = None if args.force else _find_cached_frame_output(results, path_str)
        if frame_output is None:
            frame_output = _run_vlm_on_frame(
                backend=backend,
                pil_side=pil_side,
                image_path=Path(path_str),
                user_prompt=user_prompt,
                prompt_meta=prompt_meta,
                args=args,
            )
            vlm_calls += 1

        for s in frame_items:
            _, tool = _row_key(s)
            if (
                not args.force
                and (path_str, tool) in key_to_idx
                and _should_skip_resume(results[key_to_idx[(path_str, tool)]], tool)
            ):
                rec = results[key_to_idx[(path_str, tool)]]
                inp = rec.setdefault("input", {})
                inp["label_context"] = _label_context_from_parsed(s["parsed"])
                if frame_output and not frame_output.get("error"):
                    rec["output"] = {
                        "text": frame_output.get("text"),
                        "parsed": frame_output.get("parsed"),
                    }
                ev = _score_record(rec)
                if ev:
                    rec["evaluation"] = ev
                continue
            _upsert_scored_entry(s, frame_output)

    print(f"VLM forward passes this run: {vlm_calls}", file=sys.stderr)

    for rec in results:
        if "evaluation" not in rec:
            ev = _score_record(rec)
            if ev:
                rec["evaluation"] = ev

    metrics = aggregate_metrics(results)
    payload = {
        "task": "triplet_recognition",
        "eval_protocol": "cholect50_triplet_recognition",
        "dataset": "cholect50-challenge-val",
        "dataset_root": str(args.dataset_root),
        "video_roots": [str(p) for p in video_roots],
        "backend": args.backend,
        "model_id": meta.get("model_id") if meta.get("source") == "local_checkpoint" else model_id,
        "hub_model_id_cli": model_id,
        "model_name": model_name,
        "vlm_load": meta,
        "prompt_mode": args.prompt_mode,
        "mcq_option_format": args.mcq_option_format if args.prompt_mode == "mcq" else None,
        "user_prompt_template": user_prompt,
        "eval_all": bool(args.eval_all),
        "samples_only": bool(args.samples_only),
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
        f"  Instrument Acc={m.get('instrument_accuracy', {}).get('accuracy')}  "
        f"mAP={m.get('instrument_mAP')}\n"
        f"  Verb Acc={m.get('verb_accuracy', {}).get('accuracy')}  "
        f"mAP={m.get('verb_mAP')}\n"
        f"  Target Acc={m.get('target_accuracy', {}).get('accuracy')}  "
        f"mAP={m.get('target_mAP')}\n"
        f"  Triplet Acc={m.get('triplet_accuracy', {}).get('accuracy')}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
