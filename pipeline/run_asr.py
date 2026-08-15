"""
run_asr.py — ASR-track evaluation for neuro_bench (recognition + equity layers).

TORGO is an ASR-only corpus (read words and sentences, no conversational turns),
so this runner skips the endpointing track entirely and focuses on:

    Layer 1 — Recognition   WER, CER, Entity Error Rate, Hallucination Rate,
                            Semantic Similarity, S/D/I breakdown,
                            content-word vs function-word WER
    Layer 3 — Equity        per-severity breakdown + Equity Gap (impaired - control)

It runs every selected ASR system over the manifest, one system at a time
(so a 4 GB GPU never holds two models at once), caches the raw hypotheses,
and writes a paper-ready summary.

Usage:
    python run_asr.py \
        --manifest   ../data/manifests/torgo_manifest.json \
        --output_dir ../results/torgo_asr_01 \
        --systems    whisper_large_v3 wav2vec2_960h parakeet_tdt \
        --device     cuda

Outputs (in --output_dir):
    hyps/<system>.jsonl   raw ASR hypotheses per system (reusable with --reuse_hyps)
    per_item.csv          one row per (item x system) with all recognition metrics
    summary.json          aggregated metrics: overall / by condition / by severity / equity gap
    summary.md            human-readable system x metric table
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from asr_systems import DEFAULT_SYSTEMS, SYSTEM_REGISTRY
from metrics import compute_recognition_metrics

WORKER = Path(__file__).resolve().parent / "asr_worker.py"

# Parakeet/NeMo crashes on this machine's cuDNN when using the GPU, so it runs
# CPU-only by default. Override per-run with --parakeet_device cuda once fixed.
GPU_UNSAFE_SYSTEMS = {"parakeet_tdt"}

# Reported recognition metrics. Entity Error Rate is dropped for TORGO — read
# prompts contain essentially no healthcare entities, so it is degenerate (=1.0).
#
# All rate metrics are MICRO-averaged: total errors / total units across items,
# NOT the mean of per-item rates. Micro-averaging is essential here because
# TORGO has many single-word prompts; a verbose hallucination on a 1-word
# reference yields a per-item WER of 5-7, and a macro mean of those is
# meaningless (and exceeds 1.0). semantic_similarity is a per-item score, so it
# stays a NaN-aware mean.
#   reported metric -> (numerator_count_col, denominator_count_col)
MICRO_RATES = {
    "wer":               ("wer_err",        "n_ref_words"),
    "cer":               ("cer_errors",     "cer_ref_chars"),
    "content_word_wer":  ("content_errors", "n_content_words"),
    "function_word_wer": ("function_errors","n_function_words"),
    "hallucination_rate":("halluc_count",   "hyp_word_count"),
    "sub_rate":          ("substitutions",  "n_ref_words"),
    "del_rate":          ("deletions",      "n_ref_words"),
    "ins_rate":          ("insertions",     "n_ref_words"),
}
REPORTED = ["wer", "cer", "hallucination_rate", "semantic_similarity",
            "content_word_wer", "function_word_wer",
            "sub_rate", "del_rate", "ins_rate"]


# ---------------------------------------------------------------------------
# ASR stage (with on-disk hypothesis cache)
# ---------------------------------------------------------------------------

def _load_hyps(cache: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    with open(cache, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            out[rec["id"]] = {"text": rec["text"], "words": rec.get("words", [])}
    return out


def run_or_load_hyps(
    system: str,
    manifest_path: str,
    manifest: list[dict],
    hyps_dir: Path,
    device: str,
    reuse: bool,
    limit: int | None,
) -> dict[str, dict]:
    """
    Return {item_id -> {"text","words"}} for one system.

    ASR runs in a SEPARATE process (asr_worker.py) so the GPU is fully released
    between systems and a single system can be pinned to CPU (Parakeet/NeMo).
    """
    hyps_dir.mkdir(parents=True, exist_ok=True)
    cache = hyps_dir / f"{system}.jsonl"

    if reuse and cache.exists():
        out = _load_hyps(cache)
        if all(item["id"] in out for item in manifest):
            print(f"  [{system}] reusing {len(out)} cached hypotheses")
            return out
        print(f"  [{system}] cache incomplete — re-running ASR")

    cmd = [sys.executable, str(WORKER),
           "--system", system, "--manifest", manifest_path,
           "--out", str(cache), "--device", device]
    if limit:
        cmd += ["--limit", str(limit)]

    print(f"  [{system}] -> subprocess on {device}: {' '.join(cmd[2:])}")
    proc = subprocess.run(cmd, cwd=str(WORKER.parent))
    if proc.returncode != 0 or not cache.exists():
        print(f"  [{system}] WARNING: worker exited {proc.returncode} — "
              f"recording empty hypotheses for this system")
        return {item["id"]: {"text": "", "words": []} for item in manifest}

    return _load_hyps(cache)


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

def _micro(sub: pd.DataFrame) -> dict[str, float]:
    """Micro-averaged recognition metrics over a slice of per-item rows."""
    out: dict[str, float] = {}
    for metric, (num, den) in MICRO_RATES.items():
        d = float(sub[den].sum())
        out[metric] = round(float(sub[num].sum()) / d, 4) if d > 0 else float("nan")
    ss = sub["semantic_similarity"].dropna()
    out["semantic_similarity"] = round(float(ss.mean()), 4) if len(ss) else float("nan")
    out["n_items"]   = int(len(sub))
    out["ref_words"] = int(sub["n_ref_words"].sum())
    return out


def _micro_by(df: pd.DataFrame, by: str) -> dict[str, dict[str, float]]:
    """Micro-averaged metrics within each value of column `by`."""
    return {str(val): _micro(sub) for val, sub in df.groupby(by)}


def _equity_gap(df_sys: pd.DataFrame, keys: list[str]) -> dict[str, float]:
    """Micro metric(impaired) - micro metric(control), per key. NaN if a side is empty."""
    imp  = _micro(df_sys[~df_sys["is_control"]])
    ctrl = _micro(df_sys[df_sys["is_control"]])
    gaps = {}
    for k in keys:
        a, b = imp.get(k), ctrl.get(k)
        gaps[k] = round(a - b, 4) if (a is not None and b is not None
                                      and not (np.isnan(a) or np.isnan(b))) else float("nan")
    return gaps


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_asr_eval(
    manifest_path: str,
    output_dir:    str,
    systems:       list[str],
    device:        str = "cuda",
    parakeet_device: str = "cpu",
    entity_list_path: str | None = None,
    limit:         int | None = None,
    reuse_hyps:    bool = False,
):
    out_dir  = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    hyps_dir = out_dir / "hyps"

    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)
    if limit:
        manifest = manifest[:limit]
    print(f"Loaded {len(manifest)} items from {manifest_path}")

    entity_list = None
    if entity_list_path:
        with open(entity_list_path, encoding="utf-8") as f:
            entity_list = [ln.strip() for ln in f if ln.strip()]
        print(f"Loaded {len(entity_list)} entities from {entity_list_path}")

    # --- Run every system, compute per-item metrics ---
    rows = []
    for system in systems:
        print(f"\n=== ASR system: {system} ===")
        sys_device = parakeet_device if system in GPU_UNSAFE_SYSTEMS else device
        hyps = run_or_load_hyps(
            system, manifest_path, manifest, hyps_dir, sys_device, reuse_hyps, limit,
        )

        print(f"  scoring {len(manifest)} items ...")
        for item in manifest:
            asr = hyps.get(item["id"], {"text": "", "words": []})
            rec = compute_recognition_metrics(item, asr, entity_list)
            n_ref = len(item["transcript"].split())
            rows.append({
                "item_id":    item["id"],
                "system":     system,
                "speaker_id": item.get("speaker_id", ""),
                "condition":  item.get("condition", ""),
                "severity":   item.get("severity", "unknown"),
                "is_control": item.get("is_control", False),
                "prompt_type": "word" if n_ref <= 1 else "sentence",
                "reference":  item["transcript"],
                "hypothesis": asr.get("text", ""),
                **rec,
            })

    df = pd.DataFrame(rows)
    # Numerator for micro-averaged WER: total word-edit operations per item.
    df["wer_err"] = df[["substitutions", "deletions", "insertions"]].sum(axis=1)
    csv_path = out_dir / "per_item.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nWrote per_item.csv ({len(df)} rows) -> {csv_path}")

    # --- Aggregate per system (all metrics MICRO-averaged) ---
    summary: dict = {
        "manifest":   manifest_path,
        "n_items":    len(manifest),
        "averaging":  "micro (total errors / total units); semantic_similarity is a mean",
        "systems":    {},
    }
    for system in systems:
        sdf  = df[df["system"] == system]
        sent = sdf[sdf["prompt_type"] == "sentence"]   # severity story uses sentences only

        summary["systems"][system] = {
            "label":              SYSTEM_REGISTRY[system]["label"],
            "checkpoint":         SYSTEM_REGISTRY[system]["checkpoint"],
            "overall":            _micro(sdf),
            "by_prompt_type":     _micro_by(sdf, "prompt_type"),
            "by_condition":       _micro_by(sdf, "condition"),
            # severity breakdown for impaired speakers, sentences only
            "by_severity_sentences": _micro_by(sent[~sent["is_control"]], "severity"),
            "equity_gaps":        _equity_gap(sdf, REPORTED),
            "equity_gaps_sentences": _equity_gap(sent, REPORTED),
            "n_impaired":         int((~sdf["is_control"]).sum()),
            "n_control":          int(sdf["is_control"].sum()),
        }

    summary_path = out_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(_sanitise(summary), f, indent=2)
    print(f"Wrote summary.json -> {summary_path}")

    _write_markdown(summary, out_dir / "summary.md")
    print(f"Wrote summary.md -> {out_dir / 'summary.md'}")
    return df, summary


def _write_markdown(summary: dict, path: Path) -> None:
    """Paper-ready, micro-averaged system x metric tables."""
    syslist = summary["systems"].items()
    lines = ["# neuro_bench — ASR track summary\n",
             f"- Manifest: `{summary['manifest']}`",
             f"- Items: {summary['n_items']}",
             f"- Averaging: {summary['averaging']}",
             "- Entity Error Rate omitted (degenerate on TORGO read prompts).\n",
             "## Overall (micro-averaged, all items)\n"]
    lines.append("| system | " + " | ".join(REPORTED) + " |")
    lines.append("|" + "---|" * (len(REPORTED) + 1))
    for _, s in syslist:
        cells = [s["label"]] + [_fmt(s["overall"].get(k)) for k in REPORTED]
        lines.append("| " + " | ".join(cells) + " |")

    # WER by prompt type — exposes the single-word hallucination effect
    lines.append("\n## WER by prompt type (micro)\n")
    lines.append("| system | word | sentence |")
    lines.append("|---|---|---|")
    for _, s in summary["systems"].items():
        bp = s["by_prompt_type"]
        lines.append(f"| {s['label']} | {_fmt(bp.get('word', {}).get('wer'))} "
                     f"| {_fmt(bp.get('sentence', {}).get('wer'))} |")

    # Severity — sentences only, impaired (FDA tiers)
    tiers = ("very_mild", "moderate", "moderate_severe", "severe")
    lines.append("\n## WER by severity — sentences only, impaired (micro; FDA tiers)\n")
    lines.append("| system | " + " | ".join(tiers) + " |")
    lines.append("|" + "---|" * (len(tiers) + 1))
    for _, s in summary["systems"].items():
        sb = s["by_severity_sentences"]
        row = [s["label"]] + [_fmt(sb.get(t, {}).get("wer")) for t in tiers]
        lines.append("| " + " | ".join(row) + " |")

    # Equity gap (sentences only) — the headline
    lines.append("\n## Equity gap (impaired − control), sentences only (micro)\n")
    lines.append("| system | " + " | ".join(REPORTED) + " |")
    lines.append("|" + "---|" * (len(REPORTED) + 1))
    for _, s in summary["systems"].items():
        cells = [s["label"]] + [_fmt(s["equity_gaps_sentences"].get(k)) for k in REPORTED]
        lines.append("| " + " | ".join(cells) + " |")

    lines.append("\n_Positive equity gap = impaired speakers fare worse. "
                 "semantic_similarity gap is negative because higher similarity is better._")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _fmt(v) -> str:
    if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
        return "—"
    return f"{v:.3f}" if isinstance(v, float) else str(v)


def _sanitise(obj):
    if isinstance(obj, dict):
        return {k: _sanitise(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitise(v) for v in obj]
    if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
        return None
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return obj


def main():
    p = argparse.ArgumentParser(description="neuro_bench ASR-track evaluation")
    p.add_argument("--manifest",   required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--systems",    nargs="+", default=DEFAULT_SYSTEMS,
                   choices=list(SYSTEM_REGISTRY),
                   help=f"ASR systems to run (default: {DEFAULT_SYSTEMS})")
    p.add_argument("--device",     default="cuda", choices=["cuda", "cpu"],
                   help="Device for GPU-safe systems (Whisper, wav2vec2)")
    p.add_argument("--parakeet_device", default="cpu", choices=["cuda", "cpu"],
                   help="Device for Parakeet/NeMo (default cpu; NeMo GPU crashes "
                        "on this machine's cuDNN)")
    p.add_argument("--entity_list", default=None,
                   help="Optional text file of entity tokens (one per line)")
    p.add_argument("--limit",      type=int, default=None,
                   help="Only process the first N manifest items (smoke test)")
    p.add_argument("--reuse_hyps", action="store_true",
                   help="Reuse cached hypotheses in <output_dir>/hyps if present")
    args = p.parse_args()

    run_asr_eval(
        manifest_path    = args.manifest,
        output_dir       = args.output_dir,
        systems          = args.systems,
        device           = args.device,
        parakeet_device  = args.parakeet_device,
        entity_list_path = args.entity_list,
        limit            = args.limit,
        reuse_hyps       = args.reuse_hyps,
    )


if __name__ == "__main__":
    main()
