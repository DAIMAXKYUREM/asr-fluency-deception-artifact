"""
novel_metrics.py — Interaction-aware ASR metrics for neuro_bench.

Computes the new metrics proposed for neurodegenerative speech, off the cached
hypotheses + per_item.csv (no re-transcription, no sentence-transformers reload):

  1. Fluent Misrecognition Rate (FMR) + Deception index D
       D_i   = phi_i * (1 - s_i)          phi = fluency (GPT-2 perplexity -> [0,1]),
                                          s   = semantic similarity (from per_item.csv)
       FMR   = P(phi >= 0.5  AND  s <= 0.5)   "reads fluent, means something else"
     Plus phi-on-wrong: how fluent the system's *errors* are (the harm signal).

  2. Cost-to-be-Understood (communication throughput)
       U     = sum(content_hits) / sum(speech_seconds)   content words understood / sec
       cost  = 1 / U                                       seconds spoken per understood word
       effort gap = U_control - U_impaired

  3. Error Consistency (adaptability)
       For each (speaker, reference) read >= 2 times, modal-hypothesis fraction.
       High = predictable errors a user could learn to compensate for.

Usage:
    python novel_metrics.py            # uses results/torgo_asr_01
    python novel_metrics.py --run_dir ../results/torgo_asr_01
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent

TAU_S   = 0.5   # semantic-similarity threshold below which meaning is "wrong"
TAU_PHI = 0.5   # fluency threshold above which output "reads fluent"


# ---------------------------------------------------------------------------
# Fluency via GPT-2 perplexity (deduped over unique hypotheses)
# ---------------------------------------------------------------------------

def compute_perplexities(texts: list[str]) -> dict[str, float]:
    import torch
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(device).eval()

    out: dict[str, float] = {}
    for t in texts:
        s = (t or "").strip()
        if not s:
            out[t] = float("nan")
            continue
        ids = tok(s, return_tensors="pt").input_ids.to(device)
        if ids.shape[1] < 2:                       # single token -> prepend BOS
            ids = tok(tok.bos_token + " " + s, return_tensors="pt").input_ids.to(device)
        if ids.shape[1] < 2:
            out[t] = float("nan")
            continue
        with torch.no_grad():
            nll = model(ids, labels=ids).loss
        out[t] = float(torch.exp(nll))
    return out


def _normalise(text: str) -> str:
    return re.sub(r"[^\w\s]", "", str(text).lower()).strip()


def is_degenerate(hyp, ref) -> bool:
    """
    Catastrophic ('degenerate') output, detected structurally (casing-independent,
    not perplexity-based): empty, length-collapse vs the reference, or a
    repetition loop / very low lexical diversity. Captures the TDT failure mode
    (gibberish/repetition/blank) without confounding by all-caps CTC output.
    """
    if not isinstance(hyp, str) or not hyp.strip():
        return True                                  # empty
    w = hyp.split()
    rw = str(ref).split()
    if len(rw) >= 4 and len(w) <= 1:
        return True                                  # collapse (>=4-word target -> <=1 word)
    run = mx = 1
    for a, b in zip(w, w[1:]):
        run = run + 1 if a.lower() == b.lower() else 1
        mx = max(mx, run)
    if mx >= 3:
        return True                                  # >=3 identical words in a row
    if len(w) >= 5 and len(set(x.lower() for x in w)) / len(w) < 0.5:
        return True                                  # repetition loop / low diversity
    return False


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def fmr_block(g: pd.DataFrame) -> dict:
    """
    FMR / deception over a slice. Fluency-given-wrong uses the MEDIAN phi over
    semantically-wrong outputs — robust to the heavy perplexity tail that
    degenerate outputs produce (the mean is dragged toward 0 by gibberish).
    """
    wrong = g.loc[g["s"] <= TAU_S, "phi"]
    ppl   = g["ppl"].replace([np.inf], np.nan).dropna()
    return {
        "n":                    int(len(g)),
        "median_ppl":           round(float(ppl.median()), 1) if len(ppl) else float("nan"),
        "FMR":                  round(float(g["fluent_wrong"].mean()), 4),
        "mean_D":               round(float(g["D"].mean()), 4),
        "median_phi_on_wrong":  round(float(wrong.median()), 4) if len(wrong) else float("nan"),
    }


def throughput_block(g: pd.DataFrame) -> dict:
    hits = float(g["content_hits"].sum())
    secs = float(g["duration"].dropna().sum())
    U = hits / secs if secs > 0 else float("nan")
    return {
        "content_words_understood_per_sec": round(U, 4),
        "sec_per_understood_word":          round(1.0 / U, 3) if U and U > 0 else float("nan"),
        "n": int(len(g)),
    }


def consistency_block(g: pd.DataFrame) -> dict:
    """Modal-hypothesis fraction over (speaker, reference) groups read >= 2x."""
    fracs = []
    for (_, _), grp in g.groupby(["speaker_id", "reference"]):
        if len(grp) < 2:
            continue
        norm = grp["hypothesis"].fillna("").map(_normalise)
        counts = Counter(norm)
        fracs.append(max(counts.values()) / len(grp))
    return {
        "mean_modal_fraction": round(float(np.mean(fracs)), 4) if fracs else float("nan"),
        "n_repeated_groups":   len(fracs),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", default=str(ROOT / "results" / "torgo_asr_01"))
    ap.add_argument("--manifest", default=str(ROOT / "data" / "manifests" / "torgo_manifest.json"))
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    df = pd.read_csv(run_dir / "per_item.csv")
    manifest = {i["id"]: i for i in json.load(open(args.manifest, encoding="utf-8"))}

    # --- durations (per audio file, cached) ---
    dur_cache: dict[str, float] = {}
    def duration(item_id: str) -> float:
        it = manifest.get(item_id)
        if not it:
            return float("nan")
        p = it["audio_path"]
        if p not in dur_cache:
            try:
                info = sf.info(p)
                dur_cache[p] = info.frames / info.samplerate
            except Exception:
                dur_cache[p] = float("nan")
        return dur_cache[p]
    print("reading audio durations ...", flush=True)
    df["duration"] = df["item_id"].map(duration)

    # --- fluency (GPT-2 perplexity) ---
    print("scoring fluency with GPT-2 (deduped) ...", flush=True)
    uniq = compute_perplexities(list(df["hypothesis"].fillna("").unique()))
    df["ppl"]    = df["hypothesis"].fillna("").map(uniq)
    df["logppl"] = np.log(df["ppl"].replace([np.inf, -np.inf], np.nan))

    valid = df["logppl"].dropna()
    mu, sd = float(valid.mean()), float(valid.std()) or 1.0
    df["phi"] = 1.0 / (1.0 + np.exp((df["logppl"] - mu) / sd))      # high = fluent
    df.loc[df["hypothesis"].fillna("").str.strip() == "", "phi"] = 0.0
    df["phi"] = df["phi"].fillna(0.0)

    df["s"] = df["semantic_similarity"].clip(0, 1).fillna(0.0)
    df["D"] = df["phi"] * (1 - df["s"])
    df["fluent_wrong"] = ((df["phi"] >= TAU_PHI) & (df["s"] <= TAU_S)).astype(float)
    df["content_hits"] = (df["n_content_words"] - df["content_errors"]).clip(lower=0).fillna(0)
    df["degenerate"]   = [is_degenerate(h, r) for h, r in zip(df["hypothesis"], df["reference"])]

    sysnames = list(df["system"].unique())
    lines = ["# neuro_bench — Interaction-aware (novel) metrics\n",
             f"- Run: `{run_dir.name}`  ·  Items: {len(df)//len(sysnames)} per system",
             f"- Fluent Misrecognition: phi>= {TAU_PHI} (fluent) AND semantic sim <= {TAU_S} (wrong)\n"]

    # 1. FMR — sentences only (fluency meaningful on connected speech)
    lines.append("## 1. Fluent Misrecognition (sentences; impaired)\n")
    lines.append("_Fluency-given-wrong = **median** phi over semantically-wrong outputs "
                 "(robust to the degenerate-output tail; mean is misleading)._\n")
    lines.append("| system | median PPL | FMR | mean D | median fluency-given-wrong |")
    lines.append("|---|---|---|---|---|")
    for sysn in sysnames:
        g = df[(df["system"] == sysn) & (df["prompt_type"] == "sentence") & (~df["is_control"])]
        b = fmr_block(g)
        lines.append(f"| {sysn} | {b['median_ppl']} | {b['FMR']} | {b['mean_D']} | {b['median_phi_on_wrong']} |")

    lines.append("\n## 1b. FMR control vs impaired (sentences)\n")
    lines.append("| system | FMR control | FMR impaired |")
    lines.append("|---|---|---|")
    for sysn in sysnames:
        sent = df[(df["system"] == sysn) & (df["prompt_type"] == "sentence")]
        fc = fmr_block(sent[sent["is_control"]])["FMR"]
        fi = fmr_block(sent[~sent["is_control"]])["FMR"]
        lines.append(f"| {sysn} | {fc} | {fi} |")

    # 2. Cost-to-be-Understood
    lines.append("\n## 2. Cost-to-be-Understood (all items)\n")
    lines.append("| system | U control | U impaired | sec/word impaired | sec/word control | effort gap (U) |")
    lines.append("|---|---|---|---|---|---|")
    for sysn in sysnames:
        g = df[df["system"] == sysn]
        uc = throughput_block(g[g["is_control"]])
        ui = throughput_block(g[~g["is_control"]])
        gap = round(uc["content_words_understood_per_sec"] - ui["content_words_understood_per_sec"], 4)
        lines.append(f"| {sysn} | {uc['content_words_understood_per_sec']} | "
                     f"{ui['content_words_understood_per_sec']} | {ui['sec_per_understood_word']} | "
                     f"{uc['sec_per_understood_word']} | {gap} |")

    # 3. Consistency (impaired)
    lines.append("\n## 3. Error consistency (impaired; repeated prompts)\n")
    lines.append("| system | mean modal fraction | n repeated groups |")
    lines.append("|---|---|---|")
    for sysn in sysnames:
        c = consistency_block(df[(df["system"] == sysn) & (~df["is_control"])])
        lines.append(f"| {sysn} | {c['mean_modal_fraction']} | {c['n_repeated_groups']} |")

    # 4. Degenerate-output rate
    lines.append("\n## 4. Degenerate-output rate (empty / collapse / repetition)\n")
    lines.append("| system | impaired | control |")
    lines.append("|---|---|---|")
    for sysn in sysnames:
        g = df[df["system"] == sysn]
        di = round(float(g.loc[~g["is_control"], "degenerate"].mean()), 4)
        dc = round(float(g.loc[g["is_control"], "degenerate"].mean()), 4)
        lines.append(f"| {sysn} | {di} | {dc} |")

    out_md = run_dir / "novel_metrics.md"
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    # also persist the augmented per-item table for inspection
    df.to_csv(run_dir / "per_item_novel.csv", index=False)
    print("\n".join(lines))
    print(f"\nWrote {out_md} and per_item_novel.csv")


if __name__ == "__main__":
    main()
