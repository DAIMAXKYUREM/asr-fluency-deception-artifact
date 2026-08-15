"""
novel_metrics_mdvrkcl.py — deception metrics for MDVR-KCL (Parkinson's ReadText).

MDVR is 37 long read-passage recordings (one per speaker), not per-utterance
prompts, so the TORGO novel_metrics.py assumptions (prompt_type split, repeated
prompts, item_id+manifest durations) don't apply. This computes the corpus-agnostic
deception core off results/mdvrkcl_asr_01/per_item.csv:

    phi   = fluency (GPT-2 perplexity -> corpus-relative sigmoid)
    s     = semantic similarity (from per_item.csv)
    D     = phi*(1-s);  FMR = P(phi>=0.5 & s<=0.5);  fluency-given-wrong = median phi over {s<=0.5}
    degenerate-output rate (empty / collapse / repetition)

Reported overall, control vs PD, and by UPDRS severity. (Consistency + cost are
omitted: no repeated prompts, and passages aren't utterance-level.)

Usage:
    python novel_metrics_mdvrkcl.py --run_dir ../results/mdvrkcl_asr_01
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

TAU_S, TAU_PHI = 0.5, 0.5
SEV_ORDER = ["control", "none", "mild", "moderate", "severe", "profound"]


def compute_perplexities(texts):
    import torch
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(dev).eval()
    out = {}
    for t in texts:
        s = (t or "").strip()
        if not s:
            out[t] = float("nan"); continue
        ids = tok(s, return_tensors="pt").input_ids.to(dev)
        if ids.shape[1] < 2:
            ids = tok(tok.bos_token + " " + s, return_tensors="pt").input_ids.to(dev)
        out[t] = float(torch.exp(model(ids, labels=ids).loss)) if ids.shape[1] >= 2 else float("nan")
    return out


def is_degenerate(hyp, ref) -> bool:
    if not isinstance(hyp, str) or not hyp.strip():
        return True
    w = hyp.split(); rw = str(ref).split()
    if len(rw) >= 4 and len(w) <= 1:
        return True
    run = mx = 1
    for a, b in zip(w, w[1:]):
        run = run + 1 if a.lower() == b.lower() else 1
        mx = max(mx, run)
    if mx >= 3:
        return True
    if len(w) >= 5 and len(set(x.lower() for x in w)) / len(w) < 0.5:
        return True
    return False


def block(g):
    wrong = g.loc[g["s"] <= TAU_S, "phi"]
    return {"n": int(len(g)),
            "FMR": round(float(g["fluent_wrong"].mean()), 4) if len(g) else float("nan"),
            "mean_D": round(float(g["D"].mean()), 4) if len(g) else float("nan"),
            "fluency_given_wrong": round(float(wrong.median()), 4) if len(wrong) else float("nan"),
            "degenerate": round(float(g["degenerate"].mean()), 4) if len(g) else float("nan")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    args = ap.parse_args()
    run = Path(args.run_dir)
    df = pd.read_csv(run / "per_item.csv")
    if "hypothesis" not in df.columns:
        raise SystemExit("per_item.csv has no 'hypothesis' column — re-run run_asr_mdvrkcl.py "
                         "(aggregation pass) after the reference/hypothesis fix.")
    df["hypothesis"] = df["hypothesis"].fillna(""); df["reference"] = df["reference"].fillna("")

    print("scoring fluency (GPT-2) ...", flush=True)
    ppl = compute_perplexities(list(df["hypothesis"].unique()))
    df["ppl"] = df["hypothesis"].map(ppl)
    lp = np.log(df["ppl"].replace([np.inf, -np.inf], np.nan))
    mu, sd = float(lp.mean()), (float(lp.std()) or 1.0)
    df["phi"] = (1.0 / (1.0 + np.exp((lp - mu) / sd))).fillna(0.0)
    df.loc[df["hypothesis"].str.strip() == "", "phi"] = 0.0
    df["s"] = df["semantic_similarity"].clip(0, 1).fillna(0.0)
    df["D"] = df["phi"] * (1 - df["s"])
    df["fluent_wrong"] = ((df["phi"] >= TAU_PHI) & (df["s"] <= TAU_S)).astype(float)
    df["degenerate"] = [is_degenerate(h, r) for h, r in zip(df["hypothesis"], df["reference"])]
    df.to_csv(run / "per_item_novel.csv", index=False)

    systems = list(df["system"].unique())
    L = ["# MDVR-KCL (Parkinson's) — deception metrics\n",
         f"- {df['id'].nunique()} recordings/system. phi>= {TAU_PHI} (fluent), s<= {TAU_S} (wrong).",
         "- Read-passage corpus: per-recording (coarser than TORGO's per-utterance); "
         "consistency/cost omitted (no repeated prompts).\n",
         "## FMR / deception — overall, control vs PD\n",
         "| system | scope | FMR | mean D | fluency-given-wrong | degenerate |",
         "|---|---|---|---|---|---|"]
    for sysn in systems:
        g = df[df.system == sysn]
        for scope, sub in (("overall", g), ("control", g[g.is_control]), ("PD", g[~g.is_control])):
            b = block(sub)
            L.append(f"| {sysn} | {scope} | {b['FMR']} | {b['mean_D']} | "
                     f"{b['fluency_given_wrong']} | {b['degenerate']} |")

    L.append("\n## By UPDRS severity (mean D / FMR)\n")
    tiers = [t for t in SEV_ORDER if t in set(df["severity"])]
    L.append("| system | " + " | ".join(tiers) + " |")
    L.append("|---|" + "---|" * len(tiers))
    for sysn in systems:
        g = df[df.system == sysn]
        cells = []
        for t in tiers:
            b = block(g[g.severity == t])
            cells.append(f"D={b['mean_D']} FMR={b['FMR']}" if b["n"] else "—")
        L.append(f"| {sysn} | " + " | ".join(cells) + " |")
    L.append("\n_Deception is coarser here than on TORGO (whole read passages, not utterances); "
             "read as: does the fluent-but-wrong pattern also appear in Parkinson's speech._")

    (run / "novel_metrics.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L)); print(f"\nWrote {run/'novel_metrics.md'} + per_item_novel.csv")


if __name__ == "__main__":
    main()
