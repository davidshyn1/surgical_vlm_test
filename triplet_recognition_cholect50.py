"""
triplet_recognition_cholect50.py

CholecT50 triplet recognition (instrument, verb, target).

  --eval-protocol joint: one question, all triplets per frame (<instrument, verb, target>)
  --eval-protocol sequential_gt: instrument → verb → target; context = GT for prior steps
  --eval-protocol sequential_pred: same order; context = model answers for prior steps

  --prompt-mode mcq | ov
  - Metrics: per-component Accuracy, Triplet Accuracy, per-component mAP
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from api_backends import ApiVisionJob, api_parallel_enabled, clear_image_encode_cache
from backend_registry import (
    BACKEND_CHOICES,
    is_api_backend,
    resolve_hf_token,
    resolve_model_id,
    resolve_output_model_name,
)
from backends import build_vlm_user_prompt, load_backend
from cholect50_data import (
    CHALLENGE_VAL_ROOT,
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

EVAL_PROTOCOL_JOINT = "joint"
EVAL_PROTOCOL_SEQUENTIAL_GT = "sequential_gt"
EVAL_PROTOCOL_SEQUENTIAL_PRED = "sequential_pred"
EVAL_PROTOCOLS = (
    EVAL_PROTOCOL_JOINT,
    EVAL_PROTOCOL_SEQUENTIAL_GT,
    EVAL_PROTOCOL_SEQUENTIAL_PRED,
)

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


_ANGLE_TRIPLET_RE = re.compile(
    r"<\s*([^,<]+?)\s*,\s*([^,<]+?)\s*,\s*([^>]+?)\s*>",
    re.IGNORECASE,
)


def build_triplet_recognition_prompt(
    *,
    prompt_mode: str,
) -> tuple[str, dict[str, Any]]:
    """
    Bench-style single triplet question.
    MCQ: instrument / verb / target comma-separated option lists.
    OV:  no option lists.
    """
    mode = (prompt_mode or "mcq").strip().lower()
    meta: dict[str, Any] = {"prompt_mode": mode}

    core = (
        "What tasks are the instruments accomplishing with the targets in this surgical image? "
        "Answer with one triplet per line: <instrument, verb, target>"
    )

    if mode == "ov":
        return core, meta

    if mode != "mcq":
        raise ValueError(f"Unknown --prompt-mode {prompt_mode!r}; choose mcq or ov.")

    inst_block, inst_map = _format_option_list_block("instrument", INSTRUMENT_OPTIONS)
    verb_block, verb_map = _format_option_list_block("verb", VERB_OPTIONS)
    tgt_block, tgt_map = _format_option_list_block("target", TARGET_OPTIONS)
    meta["option_maps"] = {
        "instrument": inst_map,
        "verb": verb_map,
        "target": tgt_map,
    }

    body = f"{core}\n\n{inst_block}\n{verb_block}\n{tgt_block}"
    return body, meta


def build_mcq_option_meta() -> dict[str, Any]:
    inst_block, inst_map = _format_option_list_block("instrument", INSTRUMENT_OPTIONS)
    verb_block, verb_map = _format_option_list_block("verb", VERB_OPTIONS)
    tgt_block, tgt_map = _format_option_list_block("target", TARGET_OPTIONS)
    return {
        "prompt_mode": "mcq",
        "option_maps": {
            "instrument": inst_map,
            "verb": verb_map,
            "target": tgt_map,
        },
        "option_blocks": {
            "instrument": inst_block,
            "verb": verb_block,
            "target": tgt_block,
        },
    }


def build_prompt_meta(*, prompt_mode: str) -> dict[str, Any]:
    mode = (prompt_mode or "mcq").strip().lower()
    if mode == "ov":
        return {"prompt_mode": mode}
    if mode != "mcq":
        raise ValueError(f"Unknown --prompt-mode {prompt_mode!r}; choose mcq or ov.")
    return build_mcq_option_meta()


def build_component_prompt(
    component: str,
    *,
    prompt_mode: str,
    prompt_meta: dict[str, Any],
    context_instrument: str | None = None,
    context_verb: str | None = None,
) -> str:
    """Single-step prompt for instrument, verb, or target (sequential protocols)."""
    comp = (component or "").strip().lower()
    if comp not in ("instrument", "verb", "target"):
        raise ValueError(f"Unknown component {component!r}")

    if comp == "instrument":
        lead = "What is the instrument in this surgical image?"
    elif comp == "verb":
        inst_disp = cholect_display_label(context_instrument or "")
        lead = (
            f"In this surgical image, the instrument is {inst_disp}.\n"
            "What is the verb (surgical action)?"
        )
    else:
        inst_disp = cholect_display_label(context_instrument or "")
        verb_disp = cholect_display_label(context_verb or "")
        lead = (
            f"In this surgical image, the instrument is {inst_disp} and the verb is {verb_disp}.\n"
            "What is the target?"
        )

    answer_line = f"Answer with the {comp} name only."
    mode = (prompt_mode or "mcq").strip().lower()
    if mode == "ov":
        return f"{lead}\n{answer_line}"

    blocks = prompt_meta.get("option_blocks") or {}
    opt_line = blocks.get(comp, "")
    if opt_line:
        return f"{lead}\n{answer_line}\n\n{opt_line}"
    return f"{lead}\n{answer_line}"


def parse_component_response(
    text: str,
    component: str,
    *,
    prompt_mode: str,
    prompt_meta: dict[str, Any] | None = None,
) -> str | None:
    """Parse a single instrument, verb, or target from model text."""
    comp = (component or "").strip().lower()
    meta = prompt_meta or {}
    raw = (text or "").strip()
    if not raw:
        return None

    options = {
        "instrument": INSTRUMENT_OPTIONS,
        "verb": VERB_OPTIONS,
        "target": TARGET_OPTIONS,
    }.get(comp, [])
    if not options:
        return None

    mode = (prompt_mode or "mcq").strip().lower()
    candidates = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    if not candidates:
        candidates = [raw]

    if mode == "mcq":
        maps = (meta.get("option_maps") or {}).get(comp) or {}
        for cand in candidates:
            hit = _match_option_token(cand, maps, options)
            if hit:
                return hit
        return _match_option_token(raw, maps, options)

    for cand in candidates:
        hits = parse_mcq_terms(cand, options)
        if hits:
            return hits[0]
    hits = parse_mcq_terms(raw, options)
    return hits[0] if hits else None


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


def _parse_angle_bracket_triplets(
    raw: str,
    *,
    prompt_mode: str,
    prompt_meta: dict[str, Any],
) -> list[dict[str, str | None]]:
    """Parse <instrument, verb, target> (one or more per line / inline)."""
    triplets: list[dict[str, str | None]] = []
    seen: set[tuple[str, str, str]] = set()
    for m in _ANGLE_TRIPLET_RE.finditer(raw or ""):
        t = _parse_one_triplet_tokens(
            m.group(1).strip().strip("'\""),
            m.group(2).strip().strip("'\""),
            m.group(3).strip().strip("'\""),
            prompt_mode=prompt_mode,
            prompt_meta=prompt_meta,
        )
        if not (t.get("instrument") and t.get("verb") and t.get("target")):
            continue
        key = (_canonical(t["instrument"]), _canonical(t["verb"]), _canonical(t["target"]))
        if key in seen:
            continue
        seen.add(key)
        triplets.append(t)
    return triplets


def _parse_triplet_lines_from_text(
    raw: str,
    *,
    prompt_mode: str,
    prompt_meta: dict[str, Any],
) -> list[dict[str, str | None]]:
    """Parse zero or more triplets from <inst, verb, tgt> or legacy formats."""
    bracketed = _parse_angle_bracket_triplets(
        raw, prompt_mode=prompt_mode, prompt_meta=prompt_meta,
    )
    if bracketed:
        return bracketed

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


def _parsed_from_components(
    instrument: str | None,
    verb: str | None,
    target: str | None,
) -> dict[str, Any]:
    triplet = {
        "instrument": instrument,
        "verb": verb,
        "target": target,
    }
    if instrument and verb and target:
        triplets = [triplet]
    else:
        triplets = []
    return {
        "triplets": triplets,
        "instrument": instrument,
        "verb": verb,
        "target": target,
    }


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


def _class_tp_fp_fn(
    gold_sets: list[set[str]],
    pred_sets: list[set[str]],
    ckey: str,
) -> tuple[int, int, int]:
    tp = fp = fn = 0
    for gs, ps in zip(gold_sets, pred_sets, strict=True):
        in_g = ckey in gs
        in_p = ckey in ps
        if in_g and in_p:
            tp += 1
        elif in_p:
            fp += 1
        elif in_g:
            fn += 1
    return tp, fp, fn


def macro_precision_recall_multilabel(
    gold_sets: list[set[str]],
    pred_sets: list[set[str]],
    classes: list[str],
    *,
    min_support: int = 1,
) -> dict[str, float | None]:
    """
  Macro-averaged precision / recall over vocabulary classes (support >= min_support).

  Per class: TP = in both gold and pred sets for a sample; FP = in pred only; FN = in gold only.
    """
    precs: list[float] = []
    recs: list[float] = []
    for cls in classes:
        ckey = _canonical(cls)
        tp, fp, fn = _class_tp_fp_fn(gold_sets, pred_sets, ckey)
        support = tp + fn
        if support < min_support:
            continue
        if tp + fp > 0:
            precs.append(tp / (tp + fp))
        if tp + fn > 0:
            recs.append(tp / (tp + fn))
    return {
        "macro_precision": sum(precs) / len(precs) if precs else None,
        "macro_recall": sum(recs) / len(recs) if recs else None,
        "n_classes_scored": len(precs),
    }


def component_accuracy(gold_sets: list[set[str]], pred_sets: list[set[str]]) -> float | None:
    """Fraction of samples where the gold label (singleton set) is contained in pred set."""
    if not gold_sets:
        return None
    ok = sum(1 for gs, ps in zip(gold_sets, pred_sets, strict=True) if gs and gs <= ps)
    return ok / len(gold_sets)


def triplet_sample_pr_metrics(evs: list[dict[str, Any]]) -> dict[str, float | None]:
    """
    Per-annotation triplet metrics (one gold triplet per sample).

    - accuracy / recall: any predicted triplet equals gold (inst, verb, target)
    - precision: (# predicted triplets matching gold) / |predicted triplets|
    """
    if not evs:
        return {"accuracy": None, "precision": None, "recall": None}

    accs: list[float] = []
    precs: list[float] = []
    recs: list[float] = []

    for e in evs:
        g_inst = _canonical(e.get("gold_instrument"))
        g_verb = _canonical(e.get("gold_verb"))
        g_tgt = _canonical(e.get("gold_target"))
        if not (g_inst and g_verb and g_tgt):
            continue
        gold_t = (g_inst, g_verb, g_tgt)

        pred_triples: set[tuple[str, str, str]] = set()
        for t in e.get("pred_triplets") or []:
            if not isinstance(t, dict):
                continue
            pi = _canonical(t.get("instrument"))
            pv = _canonical(t.get("verb"))
            pt = _canonical(t.get("target"))
            if pi and pv and pt:
                pred_triples.add((pi, pv, pt))

        hit = gold_t in pred_triples
        recs.append(1.0 if hit else 0.0)
        accs.append(1.0 if hit else 0.0)
        if pred_triples:
            n_match = sum(1 for p in pred_triples if p == gold_t)
            precs.append(n_match / len(pred_triples))
        else:
            precs.append(0.0)

    if not accs:
        return {"accuracy": None, "precision": None, "recall": None}
    return {
        "accuracy": sum(accs) / len(accs),
        "precision": sum(precs) / len(precs),
        "recall": sum(recs) / len(recs),
    }


def component_pr_block(
    gold_sets: list[set[str]],
    pred_sets: list[set[str]],
    classes: list[str],
) -> dict[str, float | None]:
    pr = macro_precision_recall_multilabel(gold_sets, pred_sets, classes)
    return {
        "accuracy": component_accuracy(gold_sets, pred_sets),
        "precision": pr["macro_precision"],
        "recall": pr["macro_recall"],
    }


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

    protocols = {
        (r.get("input") or {}).get("eval_protocol")
        for r in results
        if (r.get("input") or {}).get("eval_protocol")
    }
    protocol_label = next(iter(protocols)) if len(protocols) == 1 else "cholect50_triplet_recognition"

    inst_pr = macro_precision_recall_multilabel(inst_gold, inst_pred, INSTRUMENT_OPTIONS)
    verb_pr = macro_precision_recall_multilabel(verb_gold, verb_pred, VERB_OPTIONS)
    tgt_pr = macro_precision_recall_multilabel(tgt_gold, tgt_pred, TARGET_OPTIONS)
    trip_block = triplet_sample_pr_metrics(evs)

    inst_block = {
        "accuracy": acc_block("instrument_correct")["accuracy"],
        "precision": inst_pr["macro_precision"],
        "recall": inst_pr["macro_recall"],
    }
    verb_block = {
        "accuracy": acc_block("verb_correct")["accuracy"],
        "precision": verb_pr["macro_precision"],
        "recall": verb_pr["macro_recall"],
    }
    tgt_block = {
        "accuracy": acc_block("target_correct")["accuracy"],
        "precision": tgt_pr["macro_precision"],
        "recall": tgt_pr["macro_recall"],
    }

    return {
        "protocol": protocol_label,
        "instrument_accuracy": acc_block("instrument_correct"),
        "verb_accuracy": acc_block("verb_correct"),
        "target_accuracy": acc_block("target_correct"),
        "triplet_accuracy": acc_block("triplet_correct"),
        "instrument_mAP": mean_ap_multilabel(inst_gold, inst_pred, INSTRUMENT_OPTIONS),
        "verb_mAP": mean_ap_multilabel(verb_gold, verb_pred, VERB_OPTIONS),
        "target_mAP": mean_ap_multilabel(tgt_gold, tgt_pred, TARGET_OPTIONS),
        "instrument": inst_block,
        "verb": verb_block,
        "target": tgt_block,
        "triplet": trip_block,
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
    gen_kw: dict[str, Any] = {
        "do_sample": args.do_sample,
        "min_length": 1,
        "max_new_tokens": args.max_new_tokens,
        "request_timeout_sec": args.api_timeout_sec,
        "image_cache_key": str(image_path.resolve()),
    }
    if args.do_sample:
        gen_kw["temperature"] = args.temperature
    prompt_text = build_vlm_user_prompt(
        backend, user_prompt, wrap=wrap_vlm_prompt,
    )
    return backend.generate(
        image,
        prompt_text,
        **gen_kw,
    )


def _api_gen_kw_base(args: argparse.Namespace) -> dict[str, Any]:
    gen_kw: dict[str, Any] = {
        "do_sample": args.do_sample,
        "min_length": 1,
        "max_new_tokens": args.max_new_tokens,
        "request_timeout_sec": args.api_timeout_sec,
    }
    if args.do_sample:
        gen_kw["temperature"] = args.temperature
    return gen_kw


def _run_vlm_on_frame(
    *,
    backend,
    pil_side: int,
    image_path: Path,
    user_prompt: str,
    prompt_meta: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Single VLM forward pass on one frame (joint protocol; shared across annotations)."""
    try:
        text = _generate_vlm_text(
            backend=backend,
            pil_side=pil_side,
            image_path=image_path,
            user_prompt=user_prompt,
            args=args,
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


def _run_sequential_triplet(
    *,
    backend,
    pil_side: int,
    image_path: Path,
    sample: dict[str, Any],
    prompt_meta: dict[str, Any],
    args: argparse.Namespace,
    use_gt_context: bool,
) -> dict[str, Any]:
    """Three VLM calls: instrument → verb → target (per annotation)."""
    try:
        parsed_ann = sample["parsed"]
        gt_inst = str(parsed_ann.get("instrument_name") or "")
        gt_verb = str(parsed_ann.get("verb_name") or "")
        gt_tgt = str(parsed_ann.get("target_name") or "")

        steps: list[dict[str, Any]] = []
        pred_inst: str | None = None
        pred_verb: str | None = None
        pred_tgt: str | None = None

        p_inst = build_component_prompt(
            "instrument", prompt_mode=args.prompt_mode, prompt_meta=prompt_meta,
        )
        t_inst = _generate_vlm_text(
            backend=backend, pil_side=pil_side, image_path=image_path,
            user_prompt=p_inst, args=args,
        )
        pred_inst = parse_component_response(
            t_inst, "instrument", prompt_mode=args.prompt_mode, prompt_meta=prompt_meta,
        )
        steps.append({"step": "instrument", "prompt": p_inst, "text": t_inst, "parsed": pred_inst})

        ctx_inst = gt_inst if use_gt_context else (pred_inst or "")
        p_verb = build_component_prompt(
            "verb",
            prompt_mode=args.prompt_mode,
            prompt_meta=prompt_meta,
            context_instrument=ctx_inst,
        )
        t_verb = _generate_vlm_text(
            backend=backend, pil_side=pil_side, image_path=image_path,
            user_prompt=p_verb, args=args,
        )
        pred_verb = parse_component_response(
            t_verb, "verb", prompt_mode=args.prompt_mode, prompt_meta=prompt_meta,
        )
        steps.append({"step": "verb", "prompt": p_verb, "text": t_verb, "parsed": pred_verb})

        ctx_verb = gt_verb if use_gt_context else (pred_verb or "")
        p_tgt = build_component_prompt(
            "target",
            prompt_mode=args.prompt_mode,
            prompt_meta=prompt_meta,
            context_instrument=ctx_inst,
            context_verb=ctx_verb,
        )
        t_tgt = _generate_vlm_text(
            backend=backend, pil_side=pil_side, image_path=image_path,
            user_prompt=p_tgt, args=args,
        )
        pred_tgt = parse_component_response(
            t_tgt, "target", prompt_mode=args.prompt_mode, prompt_meta=prompt_meta,
        )
        steps.append({"step": "target", "prompt": p_tgt, "text": t_tgt, "parsed": pred_tgt})

        parsed = _parsed_from_components(pred_inst, pred_verb, pred_tgt)
        combined_text = "\n---\n".join(
            f"[{s['step']}]\n{s['text']}" for s in steps
        )
        return {
            "text": combined_text,
            "parsed": parsed,
            "sequential_steps": steps,
            "context_mode": "gt" if use_gt_context else "predicted",
        }
    except Exception as e:
        print(f"SKIP {image_path} sequential: {e}", file=sys.stderr)
        return {"error": str(e)}


def _make_result_entry(
    *,
    sample: dict[str, Any],
    user_prompt: str | dict[str, str],
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
            "eval_protocol": args.eval_protocol,
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
    out: dict[str, Any] = {
        "text": frame_output.get("text"),
        "parsed": frame_output.get("parsed"),
    }
    if frame_output.get("sequential_steps") is not None:
        out["sequential_steps"] = frame_output["sequential_steps"]
        out["context_mode"] = frame_output.get("context_mode")
    entry["output"] = out
    return entry


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="CholecT50 triplet recognition (single-prompt, component + triplet metrics).",
    )
    p.add_argument("--backend", choices=BACKEND_CHOICES, default="prismatic")
    p.add_argument("--dataset-root", type=Path, default=CHALLENGE_VAL_ROOT)
    p.add_argument("--videos-root", type=Path, default=None)
    p.add_argument("--cholect-root-fallback", type=Path, default=CHOLECT_ROOT)
    p.add_argument("--video", type=str, default=None)
    p.add_argument("--instrument", type=str, default=None)
    p.add_argument(
        "--eval-protocol",
        choices=EVAL_PROTOCOLS,
        default=EVAL_PROTOCOL_JOINT,
        help=(
            "joint: one prompt, all triplets per frame; "
            "sequential_gt: instrument→verb→target with GT context; "
            "sequential_pred: same with predicted context."
        ),
    )
    p.add_argument(
        "--prompt-mode",
        choices=("mcq", "ov"),
        default="mcq",
        help="mcq: option lists in prompt; ov: open vocabulary (no options).",
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
    p.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Output folder slug (default: size alias, e.g. qwen3-vl-32b).",
    )
    p.add_argument("--vlm-checkpoint", type=Path, default=None)
    p.add_argument("--vlm-config", type=Path, default=None)
    p.add_argument("--hf-token", type=Path, default=_DEFAULT_HF_TOKEN)
    p.add_argument(
        "--api-key-file",
        type=Path,
        default=None,
        help="API key file for openai/gemini/claude backends (default: .openai_api_key, etc.).",
    )
    p.add_argument("--api-timeout-sec", type=int, default=120)
    p.add_argument(
        "--api-workers",
        type=int,
        default=1,
        help="Parallel cloud API requests (openai/gemini/claude only; default 1 = serial).",
    )
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

    prompt_meta = build_prompt_meta(prompt_mode=args.prompt_mode)
    if args.eval_protocol == EVAL_PROTOCOL_JOINT:
        user_prompt, joint_meta = build_triplet_recognition_prompt(
            prompt_mode=args.prompt_mode,
        )
        prompt_meta = joint_meta
    else:
        user_prompt = {
            "instrument": build_component_prompt(
                "instrument", prompt_mode=args.prompt_mode, prompt_meta=prompt_meta,
            ),
            "verb": "(context-dependent)",
            "target": "(context-dependent)",
        }

    model_id = resolve_model_id(args.backend, args.model_id)
    model_name = resolve_output_model_name(args.backend, model_id, args.model_name)
    out_root = args.output_root.resolve()
    mode_slug = f"{args.prompt_mode}_{args.eval_protocol}"
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

    print(
        f"CholecT50 triplet recognition: prompt_mode={args.prompt_mode}, "
        f"eval_protocol={args.eval_protocol}.",
        file=sys.stderr,
    )

    hf_token = resolve_hf_token(args.backend, args.hf_token)
    device = resolve_device(args.device)

    api_workers = max(1, int(args.api_workers)) if is_api_backend(args.backend) else 1
    if is_api_backend(args.backend):
        clear_image_encode_cache()

    backend, meta = load_backend(
        args.backend,
        model_id=model_id,
        hf_token=hf_token,
        api_key_file=args.api_key_file,
        vlm_checkpoint=args.vlm_checkpoint,
        vlm_config=args.vlm_config,
        device=device,
        api_timeout_sec=args.api_timeout_sec,
        api_workers=api_workers,
    )
    backend.to(device, dtype=torch.bfloat16)
    pil_side = getattr(backend, "image_size", None) or infer_pil_side(args)
    use_api_parallel = api_parallel_enabled(backend, api_workers)
    joint_prompt_text = ""
    if args.eval_protocol == EVAL_PROTOCOL_JOINT:
        if not isinstance(user_prompt, str):
            raise TypeError("joint eval expects user_prompt to be a string")
        joint_prompt_text = build_vlm_user_prompt(
            backend, user_prompt, wrap=wrap_vlm_prompt,
        )

    results, key_to_idx = load_results_for_resume(out_path)
    vlm_calls = 0

    def _upsert_scored_entry(
        sample: dict[str, Any],
        frame_output: dict[str, Any] | None,
        *,
        prompt_field: str | dict[str, str],
    ) -> None:
        row_key = _row_key(sample)
        entry = _make_result_entry(
            sample=sample,
            user_prompt=prompt_field,
            args=args,
            frame_output=frame_output,
        )
        ev = _score_record(entry)
        if ev:
            entry["evaluation"] = ev
        upsert_result(results, key_to_idx, row_key, entry)

    use_gt_context = args.eval_protocol == EVAL_PROTOCOL_SEQUENTIAL_GT

    if args.eval_protocol == EVAL_PROTOCOL_JOINT:
        by_frame = _group_samples_by_frame(sampled)
        n_frames = len(by_frame)
        parallel_note = f", api_workers={api_workers}" if use_api_parallel else ""
        print(
            f"Frame-level inference: {n_frames} unique frame(s), "
            f"{len(sampled)} annotation(s), one model load{parallel_note}.",
            file=sys.stderr,
        )

        pending_joint: list[tuple[str, list[dict[str, Any]]]] = []
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
                pending_joint.append((path_str, frame_items))
            else:
                for s in frame_items:
                    _upsert_scored_entry(s, frame_output, prompt_field=str(user_prompt))

        if pending_joint:
            if use_api_parallel:
                api_jobs: list[ApiVisionJob] = []
                for path_str, _frame_items in pending_joint:
                    gkw = _api_gen_kw_base(args)
                    gkw["image_cache_key"] = str(Path(path_str).resolve())
                    api_jobs.append(
                        ApiVisionJob(
                            job_id=path_str,
                            prompt=joint_prompt_text,
                            image_path=path_str,
                            pil_side=pil_side,
                            gen_kw=gkw,
                        )
                    )

                def _on_joint_api_done(res: Any) -> None:
                    if res.error:
                        print(f"SKIP {res.job_id}: {res.error}", file=sys.stderr)

                api_results = backend.generate_parallel(
                    api_jobs,
                    workers=api_workers,
                    on_result=_on_joint_api_done,
                )
                vlm_calls += len(api_results)
                out_by_path = {r.job_id: r for r in api_results}
                for path_str, frame_items in pending_joint:
                    res = out_by_path.get(path_str)
                    if res is None or res.error:
                        frame_output = {"error": res.error if res else "missing result"}
                    else:
                        triplet = parse_triplet_response(
                            res.text or "",
                            prompt_mode=args.prompt_mode,
                            prompt_meta=prompt_meta,
                        )
                        frame_output = {"text": res.text, "parsed": triplet}
                    for s in frame_items:
                        _upsert_scored_entry(s, frame_output, prompt_field=str(user_prompt))
            else:
                for path_str, frame_items in pending_joint:
                    frame_output = _run_vlm_on_frame(
                        backend=backend,
                        pil_side=pil_side,
                        image_path=Path(path_str),
                        user_prompt=str(user_prompt),
                        prompt_meta=prompt_meta,
                        args=args,
                    )
                    vlm_calls += 1
                    for s in frame_items:
                        _upsert_scored_entry(s, frame_output, prompt_field=str(user_prompt))
    else:
        parallel_note = f", api_workers={api_workers} (per-annotation)" if use_api_parallel else ""
        print(
            f"Sequential inference: {len(sampled)} annotation(s), "
            f"3 VLM calls each, context={'GT' if use_gt_context else 'predicted'}"
            f"{parallel_note}.",
            file=sys.stderr,
        )
        pending_seq: list[dict[str, Any]] = []
        for sample in sampled:
            path_str, tool = _row_key(sample)
            if (
                not args.force
                and (path_str, tool) in key_to_idx
                and _should_skip_resume(results[key_to_idx[(path_str, tool)]], tool)
            ):
                rec = results[key_to_idx[(path_str, tool)]]
                inp = rec.setdefault("input", {})
                inp["label_context"] = _label_context_from_parsed(sample["parsed"])
                ev = _score_record(rec)
                if ev:
                    rec["evaluation"] = ev
                continue
            pending_seq.append(sample)

        def _run_one_sequential(sample_row: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
            path_str, _tool = _row_key(sample_row)
            out = _run_sequential_triplet(
                backend=backend,
                pil_side=pil_side,
                image_path=Path(path_str),
                sample=sample_row,
                prompt_meta=prompt_meta,
                args=args,
                use_gt_context=use_gt_context,
            )
            return sample_row, out

        if use_api_parallel and pending_seq:
            with ThreadPoolExecutor(max_workers=api_workers) as pool:
                futures = [pool.submit(_run_one_sequential, s) for s in pending_seq]
                for fut in as_completed(futures):
                    sample_row, frame_output = fut.result()
                    if not frame_output.get("error"):
                        vlm_calls += len(frame_output.get("sequential_steps") or [])
                    step_prompts = {
                        st["step"]: st["prompt"]
                        for st in (frame_output.get("sequential_steps") or [])
                        if isinstance(st, dict) and st.get("step")
                    }
                    _upsert_scored_entry(sample_row, frame_output, prompt_field=step_prompts)
        else:
            for sample in pending_seq:
                sample_row, frame_output = _run_one_sequential(sample)
                if not frame_output.get("error"):
                    vlm_calls += len(frame_output.get("sequential_steps") or [])
                step_prompts = {
                    st["step"]: st["prompt"]
                    for st in (frame_output.get("sequential_steps") or [])
                    if isinstance(st, dict) and st.get("step")
                }
                _upsert_scored_entry(sample_row, frame_output, prompt_field=step_prompts)

    print(f"VLM forward passes this run: {vlm_calls}", file=sys.stderr)

    for rec in results:
        if "evaluation" not in rec:
            ev = _score_record(rec)
            if ev:
                rec["evaluation"] = ev

    metrics = aggregate_metrics(results)
    payload = {
        "task": "triplet_recognition",
        "eval_protocol": args.eval_protocol,
        "dataset": "cholect50-challenge-val",
        "dataset_root": str(args.dataset_root),
        "video_roots": [str(p) for p in video_roots],
        "backend": args.backend,
        "model_id": meta.get("model_id") if meta.get("source") == "local_checkpoint" else model_id,
        "hub_model_id_cli": model_id,
        "model_name": model_name,
        "vlm_load": meta,
        "prompt_mode": args.prompt_mode,
        "user_prompt_template": user_prompt,
        "sequential_context": (
            "gt" if args.eval_protocol == EVAL_PROTOCOL_SEQUENTIAL_GT
            else "predicted" if args.eval_protocol == EVAL_PROTOCOL_SEQUENTIAL_PRED
            else None
        ),
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
        f"  Triplet Acc={m.get('triplet_accuracy', {}).get('accuracy')}\n"
        f"  Instrument P/R={m.get('instrument', {}).get('precision')}/"
        f"{m.get('instrument', {}).get('recall')}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
