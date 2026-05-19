"""
action_recognition_sarrarp50.py

SAR-RARP50 surgical gesture / action recognition on segmentation-indexed frames.

Prerequisite:
  python scripts/extract_sarrarp50_frames.py --dataset-root ../eval/sarrarp50

Prompt (bench-style MCQ):
  "What Action related to the needle and suture is the surgeon focusing on right now?
   The available action options are ..."
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

from api_backends import ApiVisionJob, api_parallel_enabled, clear_image_encode_cache
from backend_registry import (
    BACKEND_CHOICES,
    is_api_backend,
    resolve_hf_token,
    resolve_model_id,
    resolve_output_model_name,
)
from backends import build_vlm_user_prompt, load_backend
from cholec50_data import infer_pil_side
from sarrarp50_data import (
    ACTION_CANONICAL_IDS,
    ACTION_CANONICAL_TO_ID,
    ACTION_DISPLAY_NAMES,
    CANONICAL_TO_DISPLAY,
    collect_action_samples,
    iter_samples_by_video,
    load_sample_frame_rgb,
    parse_video_dir_name,
    resolve_sarrarp50_root,
)
from utils import load_results_for_resume, resolve_device, upsert_result

_SCRIPT_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_ROOT = _SCRIPT_ROOT / "outputs" / "action_recognition_sarrarp50"
_DEFAULT_HF_TOKEN = _SCRIPT_ROOT / ".hf_token"


def _format_lettered_actions(
    action_ids: tuple[int, ...] | None = None,
) -> tuple[str, dict[str, str]]:
  lines = ["The available action options are:"]
  letter_map: dict[str, str] = {}
  name_map: dict[str, str] = {}

  ids = sorted(action_ids if action_ids is not None else ACTION_DISPLAY_NAMES.keys())
  for i, aid in enumerate(ids):
    letter = chr(ord("A") + i)
    canonical = f"a{aid}"
    disp = ACTION_DISPLAY_NAMES[aid]
    lines.append(f"{letter}. {disp}")
    letter_map[letter.upper()] = canonical
    letter_map[letter.lower()] = canonical
    key = re.sub(r"[^a-z0-9]+", "", disp.lower())
    name_map[key] = canonical
    name_map[canonical] = canonical
    name_map[str(aid)] = canonical

  return "\n".join(lines), {**letter_map, **name_map}


def build_action_recognition_prompt(*, prompt_mode: str) -> tuple[str, dict[str, Any]]:
  mode = (prompt_mode or "mcq").strip().lower()
  lead = (
    "What Action related to the needle and suture is the surgeon focusing on right now?"
  )
  if mode == "ov":
    return lead, {"prompt_mode": mode}

  if mode != "mcq":
    raise ValueError(f"Unknown --prompt-mode {prompt_mode!r}; choose mcq or ov.")

  options_block, option_map = _format_lettered_actions()
  body = f"{lead}\n{options_block}"
  return body, {"prompt_mode": mode, "option_map": option_map}


def wrap_vlm_prompt(body: str) -> str:
  return body.strip()


def _match_action_token(token: str, option_map: dict[str, str]) -> str | None:
  t = (token or "").strip()
  if not t:
    return None
  if len(t) == 1 and t.upper() in option_map:
    return option_map[t.upper()]
  key = re.sub(r"[^a-z0-9]+", "", t.lower())
  if key in option_map:
    return option_map[key]
  for canonical, disp in CANONICAL_TO_DISPLAY.items():
    if disp.lower() == t.lower():
      return canonical
    if disp.lower() in t.lower():
      return canonical
  return None


def parse_action_response(text: str, *, option_map: dict[str, str]) -> dict[str, Any]:
  raw = (text or "").strip()
  action_canonical: str | None = None

  m = re.search(
    r"action\s*[:=]\s*([^\n.;]+)",
    raw,
    re.IGNORECASE,
  )
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
    action_canonical = _match_action_token(raw, option_map)

  action_id = ACTION_CANONICAL_TO_ID.get(action_canonical) if action_canonical else None
  return {
    "action_canonical": action_canonical,
    "action_id": action_id,
    "action_display": CANONICAL_TO_DISPLAY.get(action_canonical or "", "") or None,
    "raw": raw,
  }


def _row_key(sample: dict[str, Any]) -> tuple[str, str]:
  tool = f"sarrarp50-action|{sample['vid']}|f{int(sample['frame_index']):09d}"
  return str(sample.get("img_path") or sample["video_path"]), tool


def _score_record(rec: dict[str, Any]) -> dict[str, Any] | None:
  inp = rec.get("input") or {}
  lc = inp.get("label_context") or {}
  out = rec.get("output")
  if not isinstance(out, dict):
    return None
  parsed = out.get("parsed") or {}
  gold = str(lc.get("label_action_canonical") or "")
  pred = str(parsed.get("action_canonical") or "")
  if not gold:
    return None
  return {
    "gold_action_canonical": gold,
    "gold_action_id": lc.get("label_action_id"),
    "gold_action_display": lc.get("label_action_display"),
    "pred_action_canonical": pred or None,
    "pred_action_id": parsed.get("action_id"),
    "pred_action_display": parsed.get("action_display"),
    "correct": bool(pred and pred == gold),
  }


def aggregate_metrics(results: list[dict[str, Any]]) -> dict[str, Any]:
  evs = [r["evaluation"] for r in results if r.get("evaluation")]
  if not evs:
    return {"n_results": len(results), "n_scored": 0}

  y_true = [e["gold_action_canonical"] for e in evs]
  y_pred = [e.get("pred_action_canonical") or "__none__" for e in evs]

  n = len(evs)
  correct = sum(1 for e in evs if e.get("correct"))
  accuracy = correct / n if n else None

  per_class: dict[str, dict[str, Any]] = {}
  recalls: list[float] = []
  precisions: list[float] = []
  jaccards: list[float] = []

  for cid in ACTION_CANONICAL_IDS:
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
    "protocol": "sarrarp50_action_recognition",
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
  return bool(parsed.get("action_canonical"))


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
    gen_kw: dict[str, Any] = {
      "do_sample": args.do_sample,
      "min_length": 1,
      "max_new_tokens": args.max_new_tokens,
      "request_timeout_sec": args.api_timeout_sec,
    }
    if args.do_sample:
      gen_kw["temperature"] = args.temperature

    prompt_text = build_vlm_user_prompt(
      backend, user_prompt, wrap=wrap_vlm_prompt,
    )
    text = backend.generate(image, prompt_text, **gen_kw)
    parsed = parse_action_response(text, option_map=option_map)
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
        "label_action_id": sample["action_id"],
        "label_action_canonical": sample["action_canonical"],
        "label_action_display": sample["action_display"],
        "vid": sample["vid"],
        "vid_num": sample["vid_num"],
      },
      "eval_protocol": "sarrarp50_action_recognition",
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
    description="SAR-RARP50 action recognition (segmentation-indexed frames).",
  )
  p.add_argument("--backend", choices=BACKEND_CHOICES, default="prismatic")
  p.add_argument("--dataset-root", type=Path, default=None)
  p.add_argument("--video", type=str, default=None, help="e.g. 47 or video_47")
  p.add_argument(
    "--prompt-mode",
    choices=("mcq", "ov"),
    default="mcq",
  )
  p.add_argument("--max-samples", type=int, default=None)
  p.add_argument(
    "--frame-reader",
    choices=("auto", "ffmpeg", "opencv"),
    default="auto",
    help="Only used if a PNG is missing under video_xx/frames/.",
  )
  p.add_argument("--seed", type=int, default=42)
  p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
  p.add_argument("--output", type=Path, default=None)
  p.add_argument("--model-id", type=str, default=None)
  p.add_argument("--model-name", type=str, default=None)
  p.add_argument("--vlm-checkpoint", type=Path, default=None)
  p.add_argument("--vlm-config", type=Path, default=None)
  p.add_argument("--hf-token", type=Path, default=_DEFAULT_HF_TOKEN)
  p.add_argument("--api-key-file", type=Path, default=None)
  p.add_argument("--api-timeout-sec", type=int, default=120)
  p.add_argument("--api-workers", type=int, default=1)
  p.add_argument("--device", type=str, default="0")
  p.add_argument("--do-sample", action="store_true")
  p.add_argument("--temperature", type=float, default=0.4)
  p.add_argument("--max-new-tokens", type=int, default=64)
  p.add_argument("--force", action="store_true")
  return p.parse_args()


def main() -> None:
  args = parse_args()
  dataset_root = resolve_sarrarp50_root(args.dataset_root)
  args.dataset_root = dataset_root

  video_filter: int | None = None
  if args.video:
    raw = args.video.strip()
    if raw.lower().startswith("video_"):
      raw = raw.split("_", 1)[1]
    video_filter = int(raw)

  user_prompt, prompt_meta = build_action_recognition_prompt(
    prompt_mode=args.prompt_mode,
  )
  option_map = prompt_meta.get("option_map") or {}

  samples = collect_action_samples(
    dataset_root,
    video_filter=video_filter,
    max_samples=args.max_samples,
    require_frames=True,
  )
  if not samples:
    raise RuntimeError(
      "No samples found. Run scripts/extract_sarrarp50_frames.py first "
      f"(dataset-root={dataset_root})."
    )

  n_with_img = sum(1 for s in samples if s.get("img_path"))
  if n_with_img < len(samples):
    raise RuntimeError(
      f"Missing PNGs for {len(samples) - n_with_img}/{len(samples)} samples. "
      "Re-run scripts/extract_sarrarp50_frames.py."
    )

  model_id = resolve_model_id(args.backend, args.model_id)
  model_name = resolve_output_model_name(args.backend, model_id, args.model_name)
  out_root = args.output_root.resolve()
  out_path = (
    args.output.resolve()
    if args.output is not None
    else (
      out_root
      / f"action_{args.backend}_{model_name}_{args.prompt_mode}"
      / "sarrarp50_action_seg1hz.json"
    ).resolve()
  )

  print(
    f"SAR-RARP50 action recognition: videos={len({s['vid_num'] for s in samples})}, "
    f"frames={len(samples)}, prompt_mode={args.prompt_mode}.",
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
  action_prompt_text = build_vlm_user_prompt(
    backend, user_prompt, wrap=wrap_vlm_prompt,
  )

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

  def _api_gen_kw(img_path: str) -> dict[str, Any]:
    gen_kw: dict[str, Any] = {
      "do_sample": args.do_sample,
      "min_length": 1,
      "max_new_tokens": args.max_new_tokens,
      "request_timeout_sec": args.api_timeout_sec,
    }
    if args.do_sample:
      gen_kw["temperature"] = args.temperature
    gen_kw["image_cache_key"] = img_path
    return gen_kw

  pending: list[dict[str, Any]] = []
  for _video_path, video_samples in iter_samples_by_video(samples):
    for sample in video_samples:
      row_key = _row_key(sample)
      _path_str, tool = row_key
      if (
        not args.force
        and row_key in key_to_idx
        and _should_skip_resume(results[key_to_idx[row_key]], tool)
      ):
        rec = results[key_to_idx[row_key]]
        inp = rec.setdefault("input", {})
        inp["label_context"] = {
          "label_action_id": sample["action_id"],
          "label_action_canonical": sample["action_canonical"],
          "label_action_display": sample["action_display"],
          "vid": sample["vid"],
          "vid_num": sample["vid_num"],
        }
        ev = _score_record(rec)
        if ev:
          rec["evaluation"] = ev
        continue
      pending.append(sample)

  if use_api_parallel and pending:
    print(
      f"Parallel API action inference: {len(pending)} frame(s), "
      f"api_workers={api_workers}.",
      file=sys.stderr,
    )
    api_jobs: list[ApiVisionJob] = []
    job_sample: dict[str, dict[str, Any]] = {}
    for sample in pending:
      img_path = sample.get("img_path")
      if not img_path:
        _upsert_scored(sample, {"error": "missing img_path"})
        continue
      job_id = str(Path(img_path).resolve())
      api_jobs.append(
        ApiVisionJob(
          job_id=job_id,
          prompt=action_prompt_text,
          image_path=img_path,
          pil_side=pil_side,
          gen_kw=_api_gen_kw(job_id),
        )
      )
      job_sample[job_id] = sample

    api_results = backend.generate_parallel(
      api_jobs,
      workers=api_workers,
    )
    vlm_calls += len(api_results)
    for res in api_results:
      sample = job_sample.get(str(res.job_id))
      if sample is None:
        continue
      if res.error:
        frame_output: dict[str, Any] = {"error": res.error}
      else:
        parsed = parse_action_response(res.text or "", option_map=option_map)
        frame_output = {"text": res.text, "parsed": parsed}
      _upsert_scored(sample, frame_output)
  else:
    for sample in pending:
      try:
        pil_image = load_sample_frame_rgb(
          sample,
          frame_reader=args.frame_reader,
        )
      except Exception as e:
        print(
          f"SKIP {sample['vid']} f{sample['frame_index']}: {e}",
          file=sys.stderr,
        )
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
    "task": "action_recognition",
    "eval_protocol": "sarrarp50_action_recognition",
    "dataset": "sarrarp50",
    "dataset_root": str(dataset_root),
    "frame_sampling": "segmentation_1hz",
    "action_gt_source": "action_discrete.txt",
    "backend": args.backend,
    "model_id": meta.get("model_id") if meta.get("source") == "local_checkpoint" else model_id,
    "hub_model_id_cli": model_id,
    "model_name": model_name,
    "prompt_mode": args.prompt_mode,
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

  acc = (metrics.get("accuracy") or {}).get("value")
  print(
    f"Wrote {len(results)} entries to {out_path}\n"
    f"  Accuracy={acc}\n"
    f"  Macro Recall={metrics.get('macro_recall')}  "
    f"Precision={metrics.get('macro_precision')}  "
    f"Jaccard={metrics.get('macro_jaccard')}",
    file=sys.stderr,
  )


if __name__ == "__main__":
  main()
