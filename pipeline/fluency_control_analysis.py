"""
fluency_control_analysis.py -- CODS 2026 Section 5.3 control analysis.

Answers the reviewer objection: is Whisper's higher fluency-given-wrong just a
restatement of its stronger language-model prior (same prior that GPT-2, the
phi-scorer, also has)?

Data: results/torgo_asr_01/per_item.csv (recognition metrics + CORRECT FDA
severity/is_control) merged with per_item_novel.csv (phi, s), on [item_id,
system]. Subset throughout: impaired speakers (is_control == False), sentence
prompts (prompt_type == "sentence") -- matching Table tab:deception / Table
tab:sweep in main.tex.

Task 1 -- reproduce the published FMR-companion numbers (median phi | s<=0.5):
    Whisper 0.632, wav2vec2 0.536, Parakeet 0.522
  HARD STOP if these do not reproduce to <0.001.

Task 2 -- the control:
    phi_correct = median phi | s > 0.5
    phi_wrong   = median phi | s <= 0.5   (== the published fluency-given-wrong)
    delta_phi   = phi_wrong - phi_correct
  Mixed-effects model with interaction, wav2vec2 as reference level:
    phi ~ C(system, Treatment('wav2vec2_960h')) * wrong + (1|speaker)
  Report the Whisper x wrong interaction coefficient and p-value.
  Cluster-bootstrap (resample speakers w/ replacement, B=2000, same procedure
  as the existing Whisper-minus-wav2vec2 fluency-given-wrong CI in
  stats_analysis.py) for the Whisper-minus-wav2vec2 difference in delta_phi;
  report whether the 95% interval excludes zero.

Outputs:
    results/torgo_asr_01/fluency_control_results.csv   (all computed values)
    results/torgo_asr_01/fluency_control_report.md      (human-readable)

Usage:
    python fluency_control_analysis.py
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

RNG = np.random.default_rng(0)
RUN = Path(__file__).resolve().parent.parent / "results" / "torgo_asr_01"
SYSTEMS = ["whisper_large_v3", "wav2vec2_960h", "parakeet_tdt"]
LABEL = {"whisper_large_v3": "Whisper", "wav2vec2_960h": "wav2vec2", "parakeet_tdt": "Parakeet"}
EXPECTED_FGW = {"whisper_large_v3": 0.632, "wav2vec2_960h": 0.536, "parakeet_tdt": 0.522}
B = 2000


def load() -> pd.DataFrame:
    a = pd.read_csv(RUN / "per_item.csv")
    b = pd.read_csv(RUN / "per_item_novel.csv")[["item_id", "system", "phi", "s"]]
    df = a.merge(b, on=["item_id", "system"], how="left")
    return df


def median_phi(g: pd.DataFrame, wrong: bool) -> float:
    sub = g.loc[g["s"] <= 0.5] if wrong else g.loc[g["s"] > 0.5]
    return float(sub["phi"].median()) if len(sub) else float("nan")


def delta_phi(g: pd.DataFrame) -> float:
    return median_phi(g, True) - median_phi(g, False)


def cluster_boot_delta_diff(df: pd.DataFrame, sys_a: str, sys_b: str, n=B):
    """Resample impaired speakers w/ replacement; recompute delta_phi(sys_a)-delta_phi(sys_b)
    on the SAME resampled speaker composition for both systems (paired resample),
    matching the resample-speakers-not-items logic already used for the
    Whisper-minus-wav2vec2 fluency-given-wrong CI (Section 5.2, 0.096 [0.057, 0.122])."""
    speakers = df["speaker_id"].unique()
    vals = []
    for _ in range(n):
        pick = RNG.choice(speakers, size=len(speakers), replace=True)
        sub = pd.concat([df[df["speaker_id"] == s] for s in pick], ignore_index=True)
        da = delta_phi(sub[sub.system == sys_a])
        db = delta_phi(sub[sub.system == sys_b])
        if da == da and db == db:
            vals.append(da - db)
    vals = np.array(vals)
    return float(np.mean(vals)), float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def main():
    df = load()
    sent = df[df["prompt_type"] == "sentence"]
    imp_sent = sent[~sent["is_control"]].copy()
    imp_sent["wrong"] = imp_sent["s"] <= 0.5

    report = []
    def log(s=""): report.append(s)

    # ================= Task 1: reproduce published numbers =========================
    log("# Fluency control analysis -- CODS 2026 Section 5.3\n")
    log("## Task 1 -- reproduction check\n")
    log("Subset: impaired speakers, sentence prompts. Metric: median phi | s<=0.5.\n")
    log("| system | computed | published | match (<0.001) |")
    log("|---|---|---|---|")
    all_match = True
    for sysn in SYSTEMS:
        g = imp_sent[imp_sent.system == sysn]
        computed = median_phi(g, True)
        expected = EXPECTED_FGW[sysn]
        ok = abs(computed - expected) < 0.001
        all_match = all_match and ok
        log(f"| {sysn} | {computed:.4f} | {expected:.3f} | {'MATCH' if ok else 'MISMATCH'} |")
    log("")
    if not all_match:
        log("HARD STOP: published numbers did not reproduce on this subset. Not proceeding.")
        (RUN / "fluency_control_report.md").write_text("\n".join(report), encoding="utf-8")
        raise SystemExit("Reproduction FAILED -- see fluency_control_report.md")
    log("Reproduction PASSED -- all three values match to <0.001. Proceeding to Task 2.\n")

    # ================= Task 2: the control ==========================================
    log("## Task 2 -- the control\n")
    log("| system | phi_correct (s>0.5) | phi_wrong (s<=0.5) | delta_phi |")
    log("|---|---|---|---|")
    rows = []
    for sysn in SYSTEMS:
        g = imp_sent[imp_sent.system == sysn]
        pc, pw = median_phi(g, False), median_phi(g, True)
        d = pw - pc
        rows.append({"system": sysn, "phi_correct": pc, "phi_wrong": pw, "delta_phi": d})
        log(f"| {sysn} | {pc:.4f} | {pw:.4f} | {d:+.4f} |")
    res = pd.DataFrame(rows).set_index("system")
    log("")

    largest_delta_system = res["delta_phi"].idxmax()
    branch = "A" if largest_delta_system == "whisper_large_v3" else "B"
    log(f"delta_phi is largest for: **{LABEL[largest_delta_system]}** "
        f"({res.loc[largest_delta_system,'delta_phi']:+.4f}).")
    log(f"BRANCH: {'A (Whisper largest -- existing prose is correct)' if branch=='A' else 'B (Whisper NOT largest -- prose must be rewritten)'}\n")

    # ---- mixed-effects model with interaction --------------------------------------
    log("### Mixed-effects model: phi ~ C(system, Treatment('wav2vec2_960h')) * wrong + (1|speaker)\n")
    import statsmodels.formula.api as smf
    d = imp_sent.dropna(subset=["phi", "speaker_id"]).copy()
    d["wrong"] = d["wrong"].astype(int)
    m = smf.mixedlm(
        "phi ~ C(system, Treatment('wav2vec2_960h')) * wrong",
        d, groups=d["speaker_id"],
    ).fit(reml=False)
    log("Full coefficient table:\n")
    log("| term | coef | p-value |")
    log("|---|---|---|")
    whisper_interaction_coef, whisper_interaction_p = None, None
    for term in m.params.index:
        coef, p = m.params[term], m.pvalues[term]
        log(f"| {term} | {coef:+.4f} | {p:.3e} |")
        if "whisper_large_v3" in term and "wrong" in term:
            whisper_interaction_coef, whisper_interaction_p = coef, p
    log("")
    if whisper_interaction_coef is None:
        log("WARNING: could not locate the Whisper x wrong interaction term by name match; "
            "inspect the coefficient table above manually.")
    else:
        log(f"**Whisper x wrong interaction: {whisper_interaction_coef:+.4f} "
            f"(p = {whisper_interaction_p:.3e})**\n")

    # ---- cluster bootstrap on delta_phi(Whisper) - delta_phi(wav2vec2) ------------
    log(f"### Cluster bootstrap (B={B}, resample speakers w/ replacement, "
        "same procedure as the Section 5.2 CI) on delta_phi(Whisper) - delta_phi(wav2vec2)\n")
    mean_diff, lo, hi = cluster_boot_delta_diff(imp_sent, "whisper_large_v3", "wav2vec2_960h")
    excl_zero = (lo > 0) or (hi < 0)
    log(f"Whisper-minus-wav2vec2 delta_phi difference: {mean_diff:+.4f} "
        f"[{lo:+.4f}, {hi:+.4f}] (95% CI).")
    log(f"**Excludes zero: {excl_zero}.**\n")

    # ================= save CSV ======================================================
    res["expected_fluency_given_wrong"] = res.index.map(EXPECTED_FGW)
    res["mixedlm_whisper_x_wrong_coef"] = whisper_interaction_coef
    res["mixedlm_whisper_x_wrong_p"] = whisper_interaction_p
    res["boot_diff_whisper_minus_wav2vec2_delta_phi_mean"] = mean_diff
    res["boot_diff_ci_lo"] = lo
    res["boot_diff_ci_hi"] = hi
    res["boot_diff_excludes_zero"] = excl_zero
    res["branch"] = branch
    res.to_csv(RUN / "fluency_control_results.csv")

    log(f"## Summary\n")
    log(f"BRANCH {branch}: {'Whisper has the largest delta_phi -- existing Section 5.3 prose is correct as written.' if branch=='A' else 'Whisper does NOT have the largest delta_phi -- Section 5.3 prose must be rewritten per Task 3 branch-B instructions.'}")

    (RUN / "fluency_control_report.md").write_text("\n".join(report), encoding="utf-8")
    print("\n".join(report))
    print(f"\n=== Wrote {RUN/'fluency_control_report.md'} and {RUN/'fluency_control_results.csv'} ===")


if __name__ == "__main__":
    main()
