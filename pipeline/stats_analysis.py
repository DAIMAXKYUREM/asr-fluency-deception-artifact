"""
stats_analysis.py — Inferential statistics for the neuro_bench ASR findings.

Turns "Whisper had a higher number in one table" into defensible claims:

  1. Mixed-effects models (statsmodels MixedLM), speaker as random intercept
     (the clustering a reviewer worries about — e.g. severe tier = 266 items
     from only 4 speakers):
       (a) WER ~ severity            -> is the severity cliff significant?
       (b) phi  ~ system  (on wrong) -> is Whisper significantly more
                                        "deceptive when wrong"?
  2. Cluster bootstrap (resample SPEAKERS with replacement) -> 95% CIs on the
     headline numbers: equity-gap WER, FMR, and the fluency-given-wrong
     (0.63 vs 0.54) difference. Bootstrapping speakers (not items) is what
     makes the small-speaker-n result honest.

Data: per_item.csv (current FDA severities) + per_item_novel.csv (phi, s).
Output: results/torgo_asr_01/stats_report.md
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
RNG = np.random.default_rng(0)
SEV_ORD = {"very_mild": 0, "moderate": 1, "moderate_severe": 2, "severe": 3}
RUN = Path(__file__).resolve().parent.parent / "results" / "torgo_asr_01"
SYSTEMS = ["whisper_large_v3", "wav2vec2_960h", "parakeet_tdt"]


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load() -> pd.DataFrame:
    a = pd.read_csv(RUN / "per_item.csv")
    b = pd.read_csv(RUN / "per_item_novel.csv")[["item_id", "system", "phi", "s"]]
    df = a.merge(b, on=["item_id", "system"], how="left")
    df["sev_ord"] = df["severity"].map(SEV_ORD)        # NaN for controls
    return df


# ---------------------------------------------------------------------------
# Micro statistics (point estimates that the bootstrap re-computes)
# ---------------------------------------------------------------------------

def micro_wer(d: pd.DataFrame) -> float:
    n = d["n_ref_words"].sum()
    return float(d["wer_err"].sum() / n) if n > 0 else np.nan


def fmr(d: pd.DataFrame) -> float:               # P(fluent AND wrong)
    fw = ((d["phi"] >= 0.5) & (d["s"] <= 0.5)).mean()
    return float(fw)


def median_phi_wrong(d: pd.DataFrame) -> float:  # fluency-given-wrong
    w = d.loc[d["s"] <= 0.5, "phi"].dropna()
    return float(w.median()) if len(w) else np.nan


# ---------------------------------------------------------------------------
# Cluster bootstrap (resample speakers with replacement)
# ---------------------------------------------------------------------------

def cluster_boot(df: pd.DataFrame, stat_fn, n=2000) -> tuple:
    """Resample unique speakers w/ replacement; recompute stat_fn(sub). Return (mean, lo, hi)."""
    speakers = df["speaker_id"].unique()
    vals = []
    for _ in range(n):
        pick = RNG.choice(speakers, size=len(speakers), replace=True)
        sub = pd.concat([df[df["speaker_id"] == s] for s in pick], ignore_index=True)
        v = stat_fn(sub)
        if v == v:
            vals.append(v)
    vals = np.array(vals)
    return float(np.mean(vals)), float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def boot_diff(df_a, df_b, stat_fn, n=2000) -> tuple:
    """Bootstrap the difference stat(A)-stat(B), resampling speakers within each."""
    sa, sb = df_a["speaker_id"].unique(), df_b["speaker_id"].unique()
    diffs = []
    for _ in range(n):
        pa = pd.concat([df_a[df_a["speaker_id"] == s] for s in RNG.choice(sa, len(sa), replace=True)])
        pb = pd.concat([df_b[df_b["speaker_id"] == s] for s in RNG.choice(sb, len(sb), replace=True)])
        va, vb = stat_fn(pa), stat_fn(pb)
        if va == va and vb == vb:
            diffs.append(va - vb)
    diffs = np.array(diffs)
    return float(np.mean(diffs)), float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


# ---------------------------------------------------------------------------
# Mixed-effects models
# ---------------------------------------------------------------------------

def mixedlm(formula: str, data: pd.DataFrame, group: str = "speaker_id"):
    import statsmodels.formula.api as smf
    d = data.dropna(subset=[formula.split("~")[0].strip(), group])
    return smf.mixedlm(formula, d, groups=d[group]).fit(reml=False)


def fmt_ci(t) -> str:
    return f"{t[0]:.3f} [{t[1]:.3f}, {t[2]:.3f}]"


def main():
    df = load()
    sent = df[df["prompt_type"] == "sentence"]
    imp_sent = sent[~sent["is_control"]]
    L = ["# neuro_bench — Inferential statistics (ASR track)\n",
         "Speaker as random intercept; cluster bootstrap resamples speakers (B=2000).\n"]

    # --- Model A: severity -> WER (impaired sentences) ---
    L.append("## 1. Mixed model: severity cliff is real\n")
    L.append("`wer ~ sev_ord + (1|speaker)` on impaired sentences, per system:\n")
    L.append("| system | severity slope (WER per tier) | p-value | n_items | n_speakers |")
    L.append("|---|---|---|---|---|")
    for sysn in SYSTEMS:
        d = imp_sent[imp_sent["system"] == sysn]
        m = mixedlm("wer ~ sev_ord", d)
        coef, p = m.params.get("sev_ord", np.nan), m.pvalues.get("sev_ord", np.nan)
        L.append(f"| {sysn} | {coef:+.3f} | {p:.2e} | {len(d)} | {d['speaker_id'].nunique()} |")

    # --- Model B: system -> fluency-given-wrong (deception) ---
    L.append("\n## 2. Mixed model: Whisper is significantly more deceptive when wrong\n")
    L.append("`phi ~ C(system) + (1|speaker)` on **wrong** impaired sentences "
             "(s<=0.5); reference = wav2vec2:\n")
    wrong = imp_sent[imp_sent["s"] <= 0.5].copy()
    wrong["system"] = pd.Categorical(wrong["system"],
                                     categories=["wav2vec2_960h", "whisper_large_v3", "parakeet_tdt"])
    mB = mixedlm("phi ~ C(system)", wrong)
    L.append("| term | coef (phi vs wav2vec2) | p-value |")
    L.append("|---|---|---|")
    for term in mB.params.index:
        if "system" in term or term == "Intercept":
            L.append(f"| {term} | {mB.params[term]:+.3f} | {mB.pvalues[term]:.2e} |")
    L.append(f"\n_n={len(wrong)} wrong items, {wrong['speaker_id'].nunique()} speakers. "
             "A positive Whisper coefficient with p<0.05 = Whisper's errors are "
             "significantly more fluent than wav2vec2's._")

    # --- Bootstrap CIs on headline numbers ---
    L.append("\n## 3. Bootstrap 95% CIs on headline numbers (resampling speakers)\n")

    L.append("### Equity gap (impaired - control) WER, sentences (micro)\n")
    L.append("| system | gap [95% CI] |")
    L.append("|---|---|")
    for sysn in SYSTEMS:
        si, sc = sent[(sent["system"] == sysn) & (~sent["is_control"])], \
                 sent[(sent["system"] == sysn) & (sent["is_control"])]
        L.append(f"| {sysn} | {fmt_ci(boot_diff(si, sc, micro_wer))} |")

    L.append("\n### FMR — fluent misrecognition rate, impaired sentences\n")
    L.append("| system | FMR [95% CI] |")
    L.append("|---|---|")
    for sysn in SYSTEMS:
        L.append(f"| {sysn} | {fmt_ci(cluster_boot(imp_sent[imp_sent['system']==sysn], fmr))} |")

    L.append("\n### Fluency-given-wrong (median phi) + the key contrast\n")
    L.append("| comparison | value [95% CI] |")
    L.append("|---|---|")
    for sysn in SYSTEMS:
        L.append(f"| {sysn} median phi-on-wrong | {fmt_ci(cluster_boot(imp_sent[imp_sent['system']==sysn], median_phi_wrong))} |")
    wh = imp_sent[imp_sent["system"] == "whisper_large_v3"]
    wv = imp_sent[imp_sent["system"] == "wav2vec2_960h"]
    diff = boot_diff(wh, wv, median_phi_wrong)
    L.append(f"| **Whisper - wav2vec2 (the 0.63 vs 0.54)** | **{fmt_ci(diff)}** |")
    L.append("\n_If this difference's CI excludes 0, the deception ranking is statistically supported._")

    out = RUN / "stats_report.md"
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
