#!/usr/bin/env python3
"""Build LaTeX table with per-component Accuracy / Precision / Recall from triplet JSON."""

from __future__ import annotations

import json
import sys
from pathlib import Path

_SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(_SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_ROOT))

from triplet_recognition_cholect50 import (  # noqa: E402
    _score_record,
    aggregate_metrics,
)

_REPO_ROOT = _SCRIPT_ROOT.parent
_DEFAULT_JSON_ROOT = _SCRIPT_ROOT / "outputs" / "triplet_recognition_cholect50"
_DEFAULT_OUT_TEX = _REPO_ROOT / "table_components.tex"

# Display order and labels (folder slug fragment -> row name)
MODEL_ROWS: list[tuple[str, str]] = [
    ("prismatic", "Prismatic-VLM 7B"),
    ("cosmos-reason2-2b", "Cosmos-Reason 2B"),
    ("qwen3-vl-4b", "Qwen3-VL 4B"),
    ("internvl3.5-38b", "InternVL3.5 38B"),
    ("cosmos-reason2-32b", "Cosmos-Reason 32B"),
    ("qwen2.5-vl-32b", "Qwen2.5-VL-Instruct 32B"),
    ("paligemma2-28b", "Paligemma2 28B"),
    ("qwen3-vl-32b", "Qwen3-VL-Instruct 32B"),
    ("gpt-4o", "GPT-4o"),
    ("gemini-2.5-flash", "Gemini 2.5 Flash"),
    ("surgsigma_qwen3vl_full", "Qwen3-VL 4B Full-tuning (Ours)"),
]


def _pct(v: float | None) -> str:
    if v is None:
        return "--"
    return f"{100 * v:.1f}"


def _metrics_from_payload(payload: dict) -> dict:
    results = payload.get("results") or []
    for rec in results:
        if rec.get("evaluation"):
            continue
        ev = _score_record(rec)
        if ev:
            rec["evaluation"] = ev
    return aggregate_metrics(results)


def _row_metrics(m: dict) -> dict[str, dict[str, float | None]]:
    inst = m.get("instrument") or {}
    verb = m.get("verb") or {}
    tgt = m.get("target") or {}
    trip = m.get("triplet") or {}
    if not inst and m.get("instrument_accuracy"):
        inst = {
            "accuracy": (m.get("instrument_accuracy") or {}).get("accuracy"),
            "precision": m.get("instrument_mAP"),
            "recall": None,
        }
    return {
        "instrument": inst,
        "verb": verb,
        "target": tgt,
        "triplet": trip,
    }


def _find_json_for_slug(root: Path, slug: str) -> Path | None:
    """Match output folder by slug; prefer ``*_sequential_gt`` over ``*_joint`` when both exist."""
    candidates = [
        p
        for p in sorted(root.glob("*/cholect50_challenge_val_triplet.json"))
        if slug in p.parent.name
    ]
    if not candidates:
        return None
    for p in candidates:
        if "sequential_gt" in p.parent.name:
            return p
    for p in candidates:
        if "joint" in p.parent.name:
            return p
    return candidates[0]


def collect_all_metrics(root: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for slug, _label in MODEL_ROWS:
        jp = _find_json_for_slug(root, slug)
        if jp is None:
            out[slug] = {}
            continue
        with jp.open(encoding="utf-8") as f:
            payload = json.load(f)
        m = _metrics_from_payload(payload)
        out[slug] = _row_metrics(m)
    return out


def render_latex(rows: dict[str, dict], *, ours_slug: str = "surgsigma_qwen3vl_full") -> str:
    lines = [
        r"\begin{table*}[t]",
        r"    \centering",
        r"    \caption{CholecT50 triplet recognition: Accuracy, macro Precision, and macro Recall "
        r"per component (Instrument, Verb, Target) and full triplet. "
        r"Component P/R are macro-averaged over vocabulary classes with support $\geq 1$. "
        r"Triplet P/R are per-annotation (one gold triplet per sample).}",
        r"    \label{tab:cholecT50_triplet_apr}",
        r"    \small",
        r"    \resizebox{\textwidth}{!}{",
        r"    \setlength{\tabcolsep}{4pt}",
        r"    \renewcommand{\arraystretch}{1.1}",
        r"    \begin{tabular}{l|ccc|ccc|ccc|ccc}",
        r"    \toprule",
        r"    \multirow{2}{*}{Model}",
        r"    & \multicolumn{3}{c|}{Instrument}",
        r"    & \multicolumn{3}{c|}{Verb}",
        r"    & \multicolumn{3}{c|}{Target}",
        r"    & \multicolumn{3}{c}{Triplet} \\",
        r"    \cmidrule(lr){2-4} \cmidrule(lr){5-7} \cmidrule(lr){8-10} \cmidrule(lr){11-13}",
        r"    & Acc & P & R & Acc & P & R & Acc & P & R & Acc & P & R \\",
        r"    \midrule",
    ]

    for slug, label in MODEL_ROWS:
        blk = rows.get(slug) or {}
        cells: list[str] = []
        for comp in ("instrument", "verb", "target", "triplet"):
            c = blk.get(comp) or {}
            cells.extend([_pct(c.get("accuracy")), _pct(c.get("precision")), _pct(c.get("recall"))])
        row_prefix = ""
        if slug == ours_slug:
            row_prefix = r"    \rowcolor{green!8}" + "\n"
        line = " & ".join([label, *cells])
        lines.append(f"{row_prefix}    {line} \\\\")

    lines.extend([
        r"    \bottomrule",
        r"    \end{tabular}",
        r"    }",
        r"    \vspace{-0.1in}",
        r"    \end{table*}",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Build table_components.tex from triplet JSON outputs.")
    p.add_argument("--json-root", type=Path, default=_DEFAULT_JSON_ROOT)
    p.add_argument("--output", type=Path, default=_DEFAULT_OUT_TEX)
    args = p.parse_args()

    rows = collect_all_metrics(args.json_root.resolve())
    tex = render_latex(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(tex, encoding="utf-8")
    print(f"Wrote {args.output}", file=sys.stderr)

    # CSV-style log for quick inspection
    print("slug\tcomponent\tacc\tprecision\trecall", file=sys.stderr)
    for slug, _ in MODEL_ROWS:
        for comp in ("instrument", "verb", "target", "triplet"):
            c = (rows.get(slug) or {}).get(comp) or {}
            print(
                f"{slug}\t{comp}\t{c.get('accuracy')}\t{c.get('precision')}\t{c.get('recall')}",
                file=sys.stderr,
            )


if __name__ == "__main__":
    main()
