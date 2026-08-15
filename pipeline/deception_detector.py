"""
deception_detector.py — the experiment that proves (or kills) the thesis.

An INFERENCE-TIME estimator of the deception index D = phi*(1-s): replace the
unavailable semantic term (1-s) with acoustic uncertainty (1-conf):

        D_hat = phi * (1 - conf_norm)

and ask whether D_hat flags "fluent-but-wrong" output better than either signal
alone. Compares four risk scorers against the wrong / clinical-error label:

    phi_only    = phi                         (fluency alone)
    conf_only   = 1 - conf_norm               (acoustic uncertainty alone; the standard baseline)
    D_hat       = phi * (1 - conf_norm)        (the proposed proxy)
    oracle_D    = phi * (1 - s)               (uses s at eval only; UPPER BOUND, not deployable)

Confidence is z-scored PER MODEL (scales differ across architectures) then
squashed to [0,1] like phi. Ground-truth label:
    TORGO   : wrong           = (s <= 0.5)
    PriMock : clinical error  = any_clinical == 1

Reports (per system + pooled): ROC-AUC & PR-AUC (threshold-free headline),
grouped-CV F1 (leave-one-speaker-out / grouped-by-consultation, honest tau),
and the risk-coverage curve + AURC (the selective-prediction safety payoff:
residual error after abstaining on the riskiest utterances). Also evaluates the
literal AND rule (phi>=t1 & conf_norm<=t2) as an operating point.

Usage:
    python deception_detector.py --run_dir ../results/torgo_asr_01   --corpus torgo
    python deception_detector.py --run_dir ../results/primock57_asr_01 --corpus primock
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAVE_PLT = True
except Exception:
    HAVE_PLT = False


# ---- threshold-free ranking metrics (rank-based, no sklearn dependency) -------

def roc_auc(score: np.ndarray, label: np.ndarray) -> float:
    pos, neg = score[label == 1], score[label == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score), float)
    sc = score[order]
    i = 0
    while i < len(sc):                       # average ranks over ties
        j = i
        while j + 1 < len(sc) and sc[j + 1] == sc[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    r_pos = ranks[label == 1].sum()
    n1, n0 = len(pos), len(neg)
    return float((r_pos - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def pr_auc(score: np.ndarray, label: np.ndarray) -> float:
    """Average precision (area under precision-recall, step interpolation)."""
    if label.sum() == 0:
        return float("nan")
    order = np.argsort(-score, kind="mergesort")
    lab = label[order]
    tp = np.cumsum(lab)
    fp = np.cumsum(1 - lab)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / label.sum()
    ap, prev_r = 0.0, 0.0
    for p, r in zip(precision, recall):
        ap += p * (r - prev_r); prev_r = r
    return float(ap)


def best_f1(score, label):
    """Best F1 over all thresholds + the threshold achieving it (score>=thr => flag)."""
    order = np.argsort(-score, kind="mergesort")
    lab = label[order]; sc = score[order]
    tp = np.cumsum(lab); fp = np.cumsum(1 - lab)
    P = label.sum()
    prec = tp / np.maximum(tp + fp, 1); rec = tp / max(P, 1)
    f1 = np.where((prec + rec) > 0, 2 * prec * rec / (prec + rec + 1e-12), 0.0)
    k = int(np.argmax(f1))
    return float(f1[k]), float(sc[k])


def risk_coverage(score, label, grid=None):
    """Keep lowest-risk first; residual error rate vs coverage. Returns (cov, resid, AURC)."""
    order = np.argsort(score, kind="mergesort")     # ascending risk
    lab = label[order]
    n = len(lab)
    cum = np.cumsum(lab)
    cov = np.arange(1, n + 1) / n
    resid = cum / np.arange(1, n + 1)                # error rate among the kept (lowest-risk) prefix
    aurc = float(np.trapz(resid, cov))               # lower = risk concentrated where flagged
    if grid is not None:
        resid_g = np.interp(grid, cov, resid)
        return grid, resid_g, aurc
    return cov, resid, aurc


def grouped_cv_f1(df, score_col, groups, k=5):
    """Leave-one-group-out (<=8 groups) or grouped K-fold: tune tau on train, F1 on test."""
    uniq = list(pd.unique(groups))
    if len(uniq) <= 8:
        folds = [[g] for g in uniq]
    else:
        rng = np.arange(len(uniq))
        folds = [[uniq[i] for i in rng[f::k]] for f in range(k)]
    f1s, precs, recs = [], [], []
    y = df["label"].to_numpy()
    s = df[score_col].to_numpy()
    g = groups.to_numpy()
    for test_groups in folds:
        te = np.isin(g, test_groups); tr = ~te
        if y[tr].sum() == 0 or y[te].sum() == 0:
            continue
        _, thr = best_f1(s[tr], y[tr])
        flag = s[te] >= thr
        tp = int((flag & (y[te] == 1)).sum()); fp = int((flag & (y[te] == 0)).sum())
        fn = int((~flag & (y[te] == 1)).sum())
        prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        f1s.append(f1); precs.append(prec); recs.append(rec)
    m = lambda a: float(np.mean(a)) if a else float("nan")
    return m(f1s), m(precs), m(recs)


def load_conf(conf_dir: Path, key: str) -> dict:
    """system -> {id: conf_value} from conf/*.jsonl."""
    out = {}
    for jf in conf_dir.glob("*.jsonl"):
        sysn = jf.stem
        d = {}
        for line in jf.open(encoding="utf-8"):
            try:
                r = json.loads(line); d[r["id"]] = r.get(key)
            except Exception:
                pass
        out[sysn] = d
    return out


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _rankdata(a):
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), float)
    sa = a[order]; i = 0
    while i < len(sa):
        j = i
        while j + 1 < len(sa) and sa[j + 1] == sa[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def spearman(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3:
        return float("nan")
    rx, ry = _rankdata(x[m]), _rankdata(y[m])
    rx -= rx.mean(); ry -= ry.mean()
    denom = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    return float((rx * ry).sum() / denom) if denom else float("nan")


def deception_risk_coverage(score, mass, grid):
    """Keep lowest-risk first; residual DECEPTION MASS (oracle_D) among kept."""
    order = np.argsort(score, kind="mergesort")   # ascending risk
    m = mass[order]
    resid = np.cumsum(m) / np.arange(1, len(m) + 1)
    cov = np.arange(1, len(m) + 1) / len(m)
    aurc = float(np.trapz(resid, cov))
    return grid, np.interp(grid, cov, resid), aurc


DETECTORS = ["phi_only", "conf_only", "D_hat", "oracle_D"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--corpus", required=True, choices=["torgo", "primock"])
    ap.add_argument("--per_item", default=None)
    ap.add_argument("--conf_key", default="min_conf", choices=["min_conf", "avg_conf"])
    args = ap.parse_args()

    run = Path(args.run_dir)
    if args.corpus == "torgo":
        pi = Path(args.per_item) if args.per_item else run / "per_item_novel.csv"
        uid_col, group_col = "item_id", "speaker_id"
    else:
        pi = Path(args.per_item) if args.per_item else run / "per_item_clinical_med7.csv"
        uid_col, group_col = "id", "consultation"

    df = pd.read_csv(pi)
    df["uid"] = df[uid_col].astype(str)
    df["group"] = df[group_col].astype(str)
    df["phi"] = df["phi"].astype(float)
    df["s"] = df["s"].astype(float) if "s" in df else df["semantic_similarity"].astype(float)
    df["label"] = ((df["s"] <= 0.5).astype(int) if args.corpus == "torgo"
                   else df["any_clinical"].astype(int))

    conf_map = load_conf(run / "conf", args.conf_key)
    if not conf_map:
        raise SystemExit(f"No confidence caches in {run/'conf'} — run run_confidence first.")
    df["conf"] = [conf_map.get(sys, {}).get(uid, np.nan) for sys, uid in zip(df["system"], df["uid"])]
    n0 = len(df)
    df = df[df["conf"].notna() & df["phi"].notna() & df["s"].notna()].copy()
    print(f"{len(df)}/{n0} rows joined to confidence ({args.conf_key})", flush=True)

    # per-model z-score of confidence -> conf_norm in [0,1]
    df["conf_norm"] = np.nan
    for sysn, g in df.groupby("system"):
        c = g["conf"].to_numpy(float)
        mu, sd = np.nanmean(c), (np.nanstd(c) or 1.0)
        df.loc[g.index, "conf_norm"] = sigmoid((c - mu) / sd)

    df["phi_only"] = df["phi"]
    df["conf_only"] = 1.0 - df["conf_norm"]
    df["D_hat"] = df["phi"] * (1.0 - df["conf_norm"])
    df["oracle_D"] = df["phi"] * (1.0 - df["s"])

    systems = list(df["system"].unique())
    L = [f"# Inference-time deception detector — {args.corpus.upper()}",
         f"\nJoined {len(df)} utterances; confidence = **{args.conf_key}** (z-scored per model).",
         f"Label = {'wrong (s<=0.5)' if args.corpus=='torgo' else 'clinical error (any_clinical)'}; "
         f"base rate = {df['label'].mean():.3f}.",
         "\nDetectors: `phi_only` (fluency), `conf_only` (acoustic uncertainty, the baseline), "
         "`D_hat = phi*(1-conf)` (proposed), `oracle_D = phi*(1-s)` (upper bound, uses s).",
         "\n## ROC-AUC / PR-AUC vs label (threshold-free; higher = better ranking of danger)\n"]

    # per-system + pooled AUC table
    def auc_row(scope, sub):
        y = sub["label"].to_numpy()
        cells = []
        for det in DETECTORS:
            cells.append(f"{roc_auc(sub[det].to_numpy(), y):.3f}/{pr_auc(sub[det].to_numpy(), y):.3f}")
        return f"| {scope} | {sub['label'].mean():.3f} | " + " | ".join(cells) + " |"

    L.append("| scope | base | phi_only | conf_only | **D_hat** | oracle_D |")
    L.append("|---|---|---|---|---|---|")
    for s in systems:
        L.append(auc_row(s, df[df.system == s]))
    L.append(auc_row("**pooled**", df))

    # verdict: does D_hat beat both singles on pooled ROC-AUC?
    y = df["label"].to_numpy()
    a_phi, a_conf, a_dhat, a_orc = (roc_auc(df[d].to_numpy(), y) for d in DETECTORS)
    win = a_dhat > max(a_phi, a_conf) + 1e-4
    L.append(f"\n**Verdict (pooled ROC-AUC):** D_hat={a_dhat:.3f} vs "
             f"max(phi={a_phi:.3f}, conf={a_conf:.3f})={max(a_phi,a_conf):.3f}, "
             f"oracle={a_orc:.3f} -> **D_hat {'BEATS' if win else 'does NOT beat'} both singles.**")

    # grouped-CV F1 (honest operating point)
    L.append("\n## Grouped-CV F1 (leave-one-group-out; tau tuned on train only)\n")
    L.append("| detector | F1 | precision | recall |")
    L.append("|---|---|---|---|")
    for det in DETECTORS:
        f1, pr, rc = grouped_cv_f1(df, det, df["group"])
        L.append(f"| {det} | {f1:.3f} | {pr:.3f} | {rc:.3f} |")

    # the literal AND rule as an operating point
    and_flag = (df["phi"] >= 0.5) & (df["conf_norm"] <= 0.5)
    tp = int((and_flag & (df.label == 1)).sum()); fp = int((and_flag & (df.label == 0)).sum())
    fn = int((~and_flag & (df.label == 1)).sum())
    prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    L.append(f"\n**AND rule** (phi>=0.5 & conf_norm<=0.5): precision {prec:.3f}, recall {rec:.3f}, "
             f"F1 {f1:.3f}, flags {int(and_flag.sum())}/{len(df)}.")

    # risk-coverage / AURC (safety payoff) + a fixed operating point
    L.append("\n## Risk–coverage (abstain on riskiest; residual error among kept; AURC lower=better)\n")
    L.append("| detector | AURC | residual@90% cov | residual@75% cov |")
    L.append("|---|---|---|---|")
    grid = np.linspace(0.05, 1.0, 96)
    rc_curves = {}
    for det in DETECTORS:
        cov, resid, aurc = risk_coverage(df[det].to_numpy(), y, grid)
        rc_curves[det] = (cov, resid)
        r90 = float(np.interp(0.90, cov, resid)); r75 = float(np.interp(0.75, cov, resid))
        L.append(f"| {det} | {aurc:.4f} | {r90:.3f} | {r75:.3f} |")
    L.append(f"\n_Base error rate (keep everything) = {df['label'].mean():.3f}. "
             "A good detector pushes residual error well below base at high coverage._")

    # --- thesis-aligned, non-circular evaluation --------------------------------
    # (1) is D_hat a faithful proxy for the true deception index oracle_D = phi*(1-s)?
    # (2) does acoustic uncertainty (1-conf) track semantic wrongness (1-s)?
    L.append("\n## Is D_hat a faithful proxy? (Spearman rank correlation)\n")
    L.append("| scope | rho(D_hat, oracle_D) | rho(1-conf, 1-s) | rho(conf_norm, s) |")
    L.append("|---|---|---|---|")
    def corr_row(scope, sub):
        dh, od = sub["D_hat"].to_numpy(), sub["oracle_D"].to_numpy()
        cterm, sterm = (1 - sub["conf_norm"]).to_numpy(), (1 - sub["s"]).to_numpy()
        cn, sv = sub["conf_norm"].to_numpy(), sub["s"].to_numpy()
        return (f"| {scope} | {spearman(dh, od):.3f} | {spearman(cterm, sterm):.3f} "
                f"| {spearman(cn, sv):.3f} |")
    for s in systems:
        L.append(corr_row(s, df[df.system == s]))
    L.append(corr_row("**pooled**", df))
    L.append("\n_rho(1-conf,1-s) is the linchpin: if acoustic uncertainty tracks semantic "
             "wrongness, D_hat approximates oracle_D and the proxy is principled._")

    # (3) risk-coverage on residual DECEPTION MASS (oracle_D) — the thesis-aligned safety metric.
    # D_hat should beat conf_only here: conf_only wastes abstention on low-phi garbles
    # (near-zero deception mass), D_hat targets high-phi-wrong (high mass).
    # dangerous mass to suppress: clinical = fluent x clinical-error (CCER integrand);
    # dysarthric = fluent x semantic-wrongness. This is the thesis-aligned safety target.
    if args.corpus == "primock":
        mass = (df["phi"] * df["label"]).to_numpy()   # phi * any_clinical
        mass_name = "phi*any_clinical (CCER mass)"
    else:
        mass = df["oracle_D"].to_numpy()               # phi * (1 - s)
        mass_name = "phi*(1-s)"
    L.append(f"\n## Residual deception mass = {mass_name} (abstain on riskiest; mean among kept; lower=better)\n")
    L.append("| detector | AURC(deception) | resid-mass@90% | resid-mass@75% |")
    L.append("|---|---|---|---|")
    dm_aurc = {}
    for det in DETECTORS:
        _, resid_m, aurc_m = deception_risk_coverage(df[det].to_numpy(), mass, grid)
        dm_aurc[det] = aurc_m
        m90 = float(np.interp(0.90, grid, resid_m)); m75 = float(np.interp(0.75, grid, resid_m))
        L.append(f"| {det} | {aurc_m:.4f} | {m90:.3f} | {m75:.3f} |")
    win2 = dm_aurc["D_hat"] < min(dm_aurc["phi_only"], dm_aurc["conf_only"]) - 1e-5
    L.append(f"\n**Verdict (residual-deception AURC, lower=better):** D_hat={dm_aurc['D_hat']:.4f} vs "
             f"min(phi={dm_aurc['phi_only']:.4f}, conf={dm_aurc['conf_only']:.4f})"
             f"={min(dm_aurc['phi_only'],dm_aurc['conf_only']):.4f}, oracle={dm_aurc['oracle_D']:.4f} "
             f"-> D_hat {'BEATS' if win2 else 'does NOT beat'} both singles at removing fluent-wrong mass.")
    L.append(f"\n_Base deception mass (keep everything) = {mass.mean():.4f}._")

    out_md = run / f"deception_detector_{args.corpus}.md"
    out_md.write_text("\n".join(L) + "\n", encoding="utf-8")
    df.to_csv(run / f"deception_detector_{args.corpus}_scores.csv", index=False)

    # figure: risk-coverage curves + AUC bars
    if HAVE_PLT:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
        for det, style in zip(DETECTORS, ["-", "-", "-", "--"]):
            cov, resid = rc_curves[det]
            ax1.plot(cov, resid, style, label=det, lw=2 if det == "D_hat" else 1.3)
        ax1.axhline(df["label"].mean(), color="grey", ls=":", lw=1, label="base rate")
        ax1.set_xlabel("coverage (fraction kept)"); ax1.set_ylabel("residual error among kept")
        ax1.set_title(f"Risk–coverage ({args.corpus})"); ax1.legend(fontsize=8); ax1.grid(alpha=.3)
        x = np.arange(len(systems) + 1); w = 0.2
        labels = systems + ["pooled"]
        for i, det in enumerate(DETECTORS):
            vals = [roc_auc(df[df.system == s][det].to_numpy(), df[df.system == s]["label"].to_numpy())
                    for s in systems] + [roc_auc(df[det].to_numpy(), y)]
            ax2.bar(x + (i - 1.5) * w, vals, w, label=det)
        ax2.axhline(0.5, color="grey", ls=":", lw=1)
        ax2.set_xticks(x); ax2.set_xticklabels([s.split("_")[0] for s in labels], fontsize=8)
        ax2.set_ylabel("ROC-AUC"); ax2.set_ylim(0.4, 1.0); ax2.set_title("Detector AUC by system")
        ax2.legend(fontsize=8); ax2.grid(alpha=.3, axis="y")
        fig.tight_layout()
        fig_path = run / f"deception_detector_{args.corpus}.png"
        fig.savefig(fig_path, dpi=130); plt.close(fig)
        print(f"wrote {fig_path}")

    print("\n".join(L))
    print(f"\nWrote {out_md} + scores csv")


if __name__ == "__main__":
    main()
