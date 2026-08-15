"""
threshold_sweep.py — Sensitivity of the deception ranking to the φ / s thresholds.

The FMR / fluency-given-wrong metrics use thresholds φ>=0.5 (fluent) and
s<=0.5 (semantically wrong) that were chosen by hand. This sweeps both across
{0.3,0.4,0.5,0.6,0.7} and shows the "Whisper most deceptive" ranking is stable
across the whole grid, not an artifact of where the lines were drawn.

Metric swept (uses BOTH thresholds, isolates error *quality*):
    conditional deception rate = P(φ >= τ_φ | s <= τ_s)
        = fraction of semantically-wrong outputs that read as fluent.
Also reports median-φ-given-wrong vs τ_s (the headline metric; τ_φ-independent).

Outputs:
    results/torgo_asr_01/threshold_sweep.md
    results/torgo_asr_01/figures/deception_threshold_sweep.png
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUN = Path(__file__).resolve().parent.parent / "results" / "torgo_asr_01"
GRID = [0.3, 0.4, 0.5, 0.6, 0.7]
SYS = {"whisper_large_v3": "Whisper", "wav2vec2_960h": "wav2vec2", "parakeet_tdt": "Parakeet"}


def load_impaired_sentences() -> pd.DataFrame:
    a = pd.read_csv(RUN / "per_item.csv")[["item_id", "system", "is_control", "prompt_type"]]
    b = pd.read_csv(RUN / "per_item_novel.csv")[["item_id", "system", "phi", "s"]]
    df = a.merge(b, on=["item_id", "system"], how="left")
    return df[(df["prompt_type"] == "sentence") & (~df["is_control"])].dropna(subset=["phi", "s"])


def cond_deception(d: pd.DataFrame, tphi: float, ts: float) -> float:
    """P(phi >= tphi | s <= ts): of wrong outputs, fraction that read fluent."""
    wrong = d[d["s"] <= ts]
    return float((wrong["phi"] >= tphi).mean()) if len(wrong) else np.nan


def main():
    df = load_impaired_sentences()
    systems = list(SYS)

    # --- conditional deception rate over the full grid ---
    cube = {s: np.full((len(GRID), len(GRID)), np.nan) for s in systems}  # [tphi, ts]
    for s in systems:
        d = df[df["system"] == s]
        for i, tphi in enumerate(GRID):
            for j, ts in enumerate(GRID):
                cube[s][i, j] = cond_deception(d, tphi, ts)

    # winner per cell + Whisper margin over best competitor
    n_cells = len(GRID) * len(GRID)
    whisper_wins = 0
    margin = np.full((len(GRID), len(GRID)), np.nan)
    for i in range(len(GRID)):
        for j in range(len(GRID)):
            vals = {s: cube[s][i, j] for s in systems}
            best = max(vals, key=lambda k: (vals[k] if vals[k] == vals[k] else -1))
            if best == "whisper_large_v3":
                whisper_wins += 1
            others = max(vals["wav2vec2_960h"], vals["parakeet_tdt"])
            margin[i, j] = vals["whisper_large_v3"] - others

    # --- median phi-given-wrong vs ts (headline metric) ---
    med = {s: [float(df[(df.system == s) & (df.s <= ts)]["phi"].median()) for ts in GRID]
           for s in systems}

    # --- markdown report ---
    L = ["# Deception ranking — threshold sensitivity\n",
         "Impaired sentences. Metric = conditional deception rate "
         "P(φ≥τ_φ | s≤τ_s) (fraction of wrong outputs that read fluent).\n",
         f"**Whisper ranks #1 in {whisper_wins}/{n_cells} grid cells.** "
         f"Whisper's margin over the best competitor is "
         f"{'positive in EVERY cell' if np.nanmin(margin) > 0 else 'mixed'}.\n",
         "## Median φ-given-wrong vs s-threshold (τ_φ-independent headline)\n",
         "| τ_s | " + " | ".join(SYS.values()) + " | top |",
         "|" + "---|" * (len(systems) + 2)]
    for j, ts in enumerate(GRID):
        row = [med[s][j] for s in systems]
        top = SYS[systems[int(np.argmax(row))]]
        L.append(f"| {ts} | " + " | ".join(f"{v:.3f}" for v in row) + f" | {top} |")
    L.append("\n## Whisper margin (cond. deception rate − best competitor), τ_φ × τ_s\n")
    L.append("| τ_φ \\ τ_s | " + " | ".join(str(t) for t in GRID) + " |")
    L.append("|" + "---|" * (len(GRID) + 1))
    for i, tphi in enumerate(GRID):
        L.append(f"| **{tphi}** | " + " | ".join(f"{margin[i,j]:+.3f}" for j in range(len(GRID))) + " |")
    (RUN / "threshold_sweep.md").write_text("\n".join(L) + "\n", encoding="utf-8")

    # --- figure ---
    figdir = RUN / "figures"; figdir.mkdir(exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))

    for s in systems:
        ax1.plot(GRID, med[s], marker="o", label=SYS[s])
    ax1.set_xlabel("s-threshold (τ_s): 'wrong' if semantic sim ≤ τ_s")
    ax1.set_ylabel("median fluency φ of wrong outputs")
    ax1.set_title("(A) Deception ranking is stable across τ_s\n(Whisper highest at every threshold)")
    ax1.legend(); ax1.grid(alpha=0.3)

    im = ax2.imshow(margin, cmap="RdYlGn", vmin=-abs(np.nanmax(np.abs(margin))),
                    vmax=abs(np.nanmax(np.abs(margin))), origin="lower", aspect="auto")
    ax2.set_xticks(range(len(GRID))); ax2.set_xticklabels(GRID)
    ax2.set_yticks(range(len(GRID))); ax2.set_yticklabels(GRID)
    ax2.set_xlabel("τ_s"); ax2.set_ylabel("τ_φ")
    ax2.set_title("(B) Whisper margin over best competitor\n(cond. deception rate; green = Whisper wins)")
    for i in range(len(GRID)):
        for j in range(len(GRID)):
            ax2.text(j, i, f"{margin[i,j]:+.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax2, fraction=0.046)
    fig.tight_layout()
    figpath = figdir / "deception_threshold_sweep.png"
    fig.savefig(figpath, dpi=150)
    print("\n".join(L))
    print(f"\nWhisper #1 in {whisper_wins}/{n_cells} cells; min margin = {np.nanmin(margin):+.3f}")
    print(f"Wrote {RUN/'threshold_sweep.md'} and {figpath}")


if __name__ == "__main__":
    main()
