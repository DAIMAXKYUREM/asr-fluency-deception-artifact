r"""
make_figures_sigconf.py -- CODS 2026 Task 5: regenerate the three figures
actually used in main.tex, sized for the ACM sigconf two-column layout.

    fig_severity_cliff.pdf   \label{fig:cliff}   (was fig1_severity_cliff.png)
    fig_per_error.pdf        \label{fig:perror}  (was fig4_deception.png)
    fig_cross_corpus.pdf     \label{fig:cross}   (was fig_severity_generalization.png)

Sizing: 3.33in wide (\columnwidth), 8pt tick labels, 9pt axis labels, vector
PDF, no titles (the paper's \caption carries that). Fig 3 (failure-mode bars)
and Fig 5 (AURC bars) are NOT regenerated -- cut from the ACM version as
redundant with Table~\ref{tab:overall}/\ref{tab:detector} per the task note.

Data: same sources as make_figures.py / make_figures_mdvrkcl.py --
results/torgo_asr_01/per_item.csv + per_item_novel.csv, and
results/mdvrkcl_asr_01/per_item_novel.csv.

Usage:
    python make_figures_sigconf.py [--copy_to DIR]
        --copy_to defaults to the directory containing main.tex (Downloads),
        so pdflatex finds the files by the bare names main.tex references.
"""
from __future__ import annotations

import argparse
import shutil
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- sigconf sizing: 3.33in = \columnwidth in acmart sigconf, two-column ----
COL_WIDTH_IN = 3.33
plt.rcParams.update({
    "font.family": "serif",
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.labelsize": 9,
    "legend.fontsize": 7,
    "axes.titlesize": 9,   # unused -- no titles are set
    "pdf.fonttype": 42,    # embed as real (not Type-3 bitmap) fonts
    "ps.fonttype": 42,
})

ROOT = Path(__file__).resolve().parent.parent
TORGO_RUN = ROOT / "results" / "torgo_asr_01"
MDVR_RUN = ROOT / "results" / "mdvrkcl_asr_01"
OUT_DIR = ROOT / "results" / "paper_figures_sigconf"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RNG = np.random.default_rng(0)
SYS = {"whisper_large_v3": ("Whisper", "#1f77b4"),
       "wav2vec2_960h":   ("wav2vec2", "#ff7f0e"),
       "parakeet_tdt":    ("Parakeet", "#2ca02c")}
SYSTEMS = list(SYS)
COLORS = [SYS[s][1] for s in SYSTEMS]
LABELS = [SYS[s][0] for s in SYSTEMS]
TIERS = ["very_mild", "moderate", "moderate_severe", "severe"]
TIERLAB = {"very_mild": "V.mild", "moderate": "Mod.",
           "moderate_severe": "Mod-sev.", "severe": "Severe"}


def load_torgo() -> pd.DataFrame:
    a = pd.read_csv(TORGO_RUN / "per_item.csv")
    b = pd.read_csv(TORGO_RUN / "per_item_novel.csv")[["item_id", "system", "phi", "s"]]
    return a.merge(b, on=["item_id", "system"], how="left")


def micro_wer(d: pd.DataFrame) -> float:
    n = d["n_ref_words"].sum()
    return float(d["wer_err"].sum() / n) if n else np.nan


def boot_ci(d: pd.DataFrame, fn, n=1000):
    spk = d["speaker_id"].unique()
    v = []
    for _ in range(n):
        sub = pd.concat([d[d.speaker_id == s] for s in RNG.choice(spk, len(spk), True)])
        x = fn(sub)
        if x == x:
            v.append(x)
    return (np.percentile(v, 2.5), np.percentile(v, 97.5)) if v else (np.nan, np.nan)


def med_phi_wrong(d: pd.DataFrame) -> float:
    w = d.loc[d.s <= 0.5, "phi"].dropna()
    return float(w.median()) if len(w) else np.nan


def fig_severity_cliff():
    df = load_torgo()
    sent = df[df.prompt_type == "sentence"]
    imp_sent = sent[~sent.is_control]

    fig, ax = plt.subplots(figsize=(COL_WIDTH_IN, 2.35))
    x = np.arange(len(TIERS)); w = 0.8 / len(SYSTEMS)
    for k, s in enumerate(SYSTEMS):
        d = imp_sent[imp_sent.system == s]
        vals = [micro_wer(d[d.severity == t]) for t in TIERS]
        ax.bar(x + k * w - 0.4 + w / 2, vals, w, label=SYS[s][0], color=SYS[s][1])
    ax.set_xticks(x); ax.set_xticklabels([TIERLAB[t] for t in TIERS])
    ax.set_ylabel("WER (micro)")
    ax.set_ylim(0, 0.85)
    ax.legend(ncol=3, columnspacing=0.8, handlelength=1.2, loc="upper left", frameon=False)
    fig.tight_layout(pad=0.3)
    out = OUT_DIR / "fig_severity_cliff.pdf"
    fig.savefig(out, format="pdf"); plt.close(fig)
    return out


def fig_per_error():
    df = load_torgo()
    sent = df[df.prompt_type == "sentence"]
    imp_sent = sent[~sent.is_control]

    pts, los, his = [], [], []
    for s in SYSTEMS:
        d = imp_sent[imp_sent.system == s]
        pts.append(med_phi_wrong(d))
        lo, hi = boot_ci(d, med_phi_wrong)
        los.append(lo); his.append(hi)
    yerr = [np.array(pts) - np.array(los), np.array(his) - np.array(pts)]

    fig, ax = plt.subplots(figsize=(COL_WIDTH_IN, 2.35))
    ax.bar(LABELS, pts, color=COLORS, yerr=yerr, capsize=4, error_kw={"elinewidth": 1})
    ax.set_ylabel(r"median $\varphi$ | wrong")
    ax.set_ylim(0, 0.75)
    for i, v in enumerate(pts):
        ax.annotate(f"{v:.3f}", (i, v + 0.025), ha="center", fontsize=7)
    fig.tight_layout(pad=0.3)
    out = OUT_DIR / "fig_per_error.pdf"
    fig.savefig(out, format="pdf"); plt.close(fig)
    return out


def fig_cross_corpus():
    # NOTE: per_item_novel.csv's OWN severity column is stale for TORGO (predates
    # the FDA correction in align/torgo_to_manifest.py -- see the Check-A audit in
    # this session: F03/F04/M02/M03/M05 are mislabelled there). Use per_item.csv,
    # which carries the corrected severity and matches SPEAKER_META exactly. It
    # also has NaN severity (not the literal string "control") for control rows,
    # so control must be reconstructed from is_control explicitly -- silently
    # relying on the raw severity column here previously caused control, very_mild
    # and moderate_severe to drop out of the TORGO curve entirely.
    tor = pd.read_csv(TORGO_RUN / "per_item.csv")
    tor = tor[tor["prompt_type"] == "sentence"].copy()
    tor["tier"] = np.where(tor["is_control"], "control", tor["severity"])

    # MDVR-KCL's severity column already stores the literal string "control" for
    # control rows (run_asr_mdvrkcl.py::parse_fn), so no such reconstruction is
    # needed there; wer_err is not persisted, so rebuild it the same way
    # run_asr_mdvrkcl.py does (sum of substitutions/deletions/insertions).
    mdv = pd.read_csv(MDVR_RUN / "per_item_novel.csv").copy()
    mdv["tier"] = mdv["severity"]
    mdv["wer_err"] = mdv[["substitutions", "deletions", "insertions"]].sum(axis=1)

    TORD = {"control": 0, "very_mild": 1, "moderate": 2, "moderate_severe": 3, "severe": 4}
    MORD = {"control": 0, "none": 1, "mild": 2, "moderate": 3, "severe": 4}

    def curve(d, omap):
        """Per tier: micro-WER (sum errors / sum ref words) PER SYSTEM, then averaged
        across the 3 systems -- matches the caption's 'WER averaged over systems',
        and avoids a raw pooled-row mean that would over/under-weight systems with
        different row counts per tier."""
        d = d.copy(); d["ord"] = d["tier"].map(omap)
        d = d.dropna(subset=["ord"])
        rows = []
        for o in sorted(d["ord"].unique()):
            sub = d[d["ord"] == o]
            per_sys = []
            for s in SYSTEMS:
                g = sub[sub.system == s]
                n = g["n_ref_words"].sum()
                if n:
                    per_sys.append(float(g["wer_err"].sum() / n))
            rows.append({"ord": o, "wer": float(np.mean(per_sys)) if per_sys else np.nan})
        c = pd.DataFrame(rows).sort_values("ord")
        span = c["ord"].max() - c["ord"].min()
        c["rel"] = (c["ord"] - c["ord"].min()) / (span if span else 1)
        return c

    ct, cm = curve(tor, TORD), curve(mdv, MORD)
    fig, ax = plt.subplots(figsize=(COL_WIDTH_IN, 2.35))
    ax.plot(ct["rel"], ct["wer"], "o-", lw=1.6, ms=4, label="TORGO (dysarthria)")
    ax.plot(cm["rel"], cm["wer"], "s--", lw=1.6, ms=4, label="MDVR-KCL (Parkinson's)")
    ax.set_xlabel("relative severity (control $\\to$ most severe)")
    ax.set_ylabel("WER (mean over systems)")
    ax.legend(loc="upper left", frameon=False)
    fig.tight_layout(pad=0.3)
    out = OUT_DIR / "fig_cross_corpus.pdf"
    fig.savefig(out, format="pdf"); plt.close(fig)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--copy_to", default=str(Path.home() / "Downloads"))
    args = ap.parse_args()

    saved = [fig_severity_cliff(), fig_per_error(), fig_cross_corpus()]
    print(f"Wrote {len(saved)} sigconf-sized figures to {OUT_DIR}:")
    for p in saved:
        print("  ", p.name, f"({p.stat().st_size} bytes)")

    if args.copy_to:
        dst = Path(args.copy_to)
        if dst.exists():
            for p in saved:
                shutil.copy2(p, dst / p.name)
            print(f"\nCopied to {dst} (next to main.tex).")
        else:
            print(f"\n--copy_to target {dst} does not exist -- skipped copy.")


if __name__ == "__main__":
    main()
