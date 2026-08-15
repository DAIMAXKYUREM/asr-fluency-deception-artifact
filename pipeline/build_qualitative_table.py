"""
build_qualitative_table.py — filter-and-sort over ALREADY-COMPUTED TORGO results
to select qualitative fluent-misrecognition examples for the paper. No text is
generated, paraphrased, or reconstructed: reference/hypothesis strings are read
verbatim from results/torgo_asr_01/per_item_novel.csv.

Source-of-truth note: that file's `severity` column is STALE (predates the FDA
correction in align/torgo_to_manifest.py). Severity is re-derived here from
speaker_id via the same SPEAKER_META used in the paper. is_control is correct
in the source file and is left as-is.

Usage:
    python build_qualitative_table.py --run_dir ../results/torgo_asr_01 \
        --out_dir ../results/qualitative
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

# FDA severity (Rudzicz et al. 2012) — same mapping as align/torgo_to_manifest.py::SPEAKER_META
FDA_SEVERITY = {
    "F01": "severe", "M01": "severe", "M02": "severe", "M04": "severe",
    "M05": "moderate_severe",
    "F03": "moderate",
    "F04": "very_mild", "M03": "very_mild",
}
TIER_ORDER = ["very_mild", "moderate", "moderate_severe", "severe"]

LATEX_SPECIAL = {
    "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_",
    "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}",
    "\\": r"\textbackslash{}",
}


def escape_latex(s: str) -> str:
    if not isinstance(s, str):
        return ""
    # backslash first so we don't double-escape the ones we just inserted
    out = s.replace("\\", "\x00BACKSLASH\x00")
    for ch, esc in LATEX_SPECIAL.items():
        if ch == "\\":
            continue
        out = out.replace(ch, esc)
    out = out.replace("\x00BACKSLASH\x00", r"\textbackslash{}")
    return out


def truncate(s: str, limit: int = 90) -> tuple[str, bool]:
    if not isinstance(s, str):
        return "", False
    if len(s) <= limit:
        return s, False
    return s[:limit].rstrip() + r"\ldots", True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", default="../results/torgo_asr_01")
    ap.add_argument("--scores_file", default=None,
                     help="deception_detector_torgo_scores.csv, for conf join (optional)")
    ap.add_argument("--out_dir", default="../results/qualitative")
    args = ap.parse_args()

    run = Path(args.run_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report: list[str] = []

    def log(line: str = ""):
        report.append(line)

    src = run / "per_item_novel.csv"
    log(f"# Qualitative-table build report\n")
    log(f"Source file: {src}")
    df = pd.read_csv(src)
    log(f"Rows loaded: {len(df)}")
    log(f"Columns: {list(df.columns)}\n")

    required = ["item_id", "speaker_id", "system", "prompt_type", "is_control",
                "reference", "hypothesis", "phi", "s"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        log(f"HARD STOP: missing required columns: {missing}")
        (out_dir / "report.txt").write_text("\n".join(report), encoding="utf-8")
        raise SystemExit(f"Missing required columns: {missing} — see {out_dir/'report.txt'}")
    log("Required-column check: all present (item_id, speaker_id, system, prompt_type, "
        "is_control, reference, hypothesis, phi, s).\n")

    # --- re-derive severity from speaker_id (source column is stale) --------------
    df["severity_stale"] = df["severity"]
    df["severity"] = df["speaker_id"].map(FDA_SEVERITY)  # NaN for control speakers, as intended
    mismatch = df[(~df["is_control"]) & (df["severity_stale"] != df["severity"])]
    n_mismatch_speakers = mismatch["speaker_id"].nunique()
    log("## Severity column correction")
    log(f"Speakers with a stale-vs-FDA severity mismatch in the source file: "
        f"{n_mismatch_speakers} of 8 impaired speakers.")
    for spk in sorted(set(mismatch["speaker_id"])):
        log(f"  {spk}: stale='{mismatch[mismatch.speaker_id==spk]['severity_stale'].iloc[0]}' "
            f"-> corrected='{FDA_SEVERITY[spk]}'")
    log("Corrected severity (FDA, Rudzicz et al. 2012) is used for all filtering below.\n")

    # --- optional conf join -------------------------------------------------------
    if args.scores_file:
        sc = pd.read_csv(args.scores_file, usecols=["item_id", "system", "conf", "conf_norm"])
        before = len(df)
        df = df.merge(sc, on=["item_id", "system"], how="left")
        log(f"Joined conf/conf_norm from {args.scores_file} "
            f"({df['conf'].notna().sum()}/{before} rows matched).\n")
    else:
        df["conf"] = np.nan

    # =========================== STEP 3: sanity check ==============================
    log("## Step 3 — sanity check against published numbers")
    log("FMR = P(phi>=0.5 AND s<=0.5), impaired sentences, per system:")
    expected_fmr = {"whisper_large_v3": 0.184, "wav2vec2_960h": 0.262, "parakeet_tdt": 0.144}
    expected_fgw = {"whisper_large_v3": 0.632, "wav2vec2_960h": 0.536, "parakeet_tdt": 0.522}

    imp_sent = df[(~df.is_control) & (df.prompt_type == "sentence")]
    all_ok = True
    for sysn in ["whisper_large_v3", "wav2vec2_960h", "parakeet_tdt"]:
        g = imp_sent[imp_sent.system == sysn]
        fmr = float(((g.phi >= 0.5) & (g.s <= 0.5)).mean())
        wrong = g.loc[g.s <= 0.5, "phi"]
        fgw = float(wrong.median()) if len(wrong) else float("nan")
        ok_fmr = abs(fmr - expected_fmr[sysn]) < 0.001
        ok_fgw = abs(fgw - expected_fgw[sysn]) < 0.001
        all_ok = all_ok and ok_fmr and ok_fgw
        log(f"  {sysn:18s} FMR computed={fmr:.4f} expected={expected_fmr[sysn]:.3f} "
            f"{'MATCH' if ok_fmr else 'MISMATCH'}  |  "
            f"fluency-given-wrong computed={fgw:.4f} expected={expected_fgw[sysn]:.3f} "
            f"{'MATCH' if ok_fgw else 'MISMATCH'}")
    log("")
    if not all_ok:
        log("HARD STOP: sanity check failed — filter/column mapping does not reproduce the "
            "published numbers. Not proceeding to selection.")
        (out_dir / "report.txt").write_text("\n".join(report), encoding="utf-8")
        raise SystemExit(f"Sanity check FAILED — see {out_dir/'report.txt'}. "
                         "Not selecting examples on an unverified filter.")
    log("Sanity check PASSED — all 6 values match the paper to <0.001. Proceeding.\n")

    # =========================== STEP 4: select candidates ==========================
    log("## Step 4 — candidate selection (Whisper, impaired, sentences)")
    wh = df[(df.system == "whisper_large_v3") & (~df.is_control) & (df.prompt_type == "sentence")].copy()
    log(f"Whisper impaired-sentence rows: {len(wh)}")
    fmr_pos = wh[(wh.phi >= 0.5) & (wh.s <= 0.5)].copy()
    log(f"FMR-positive rows (phi>=0.5 & s<=0.5): {len(fmr_pos)}")
    if len(fmr_pos) == 0:
        log("HARD STOP: empty FMR-positive set — cannot select examples.")
        (out_dir / "report.txt").write_text("\n".join(report), encoding="utf-8")
        raise SystemExit(f"Empty FMR-positive set — see {out_dir/'report.txt'}")

    fmr_pos["D_manual"] = fmr_pos["phi"] * (1 - fmr_pos["s"])
    fmr_pos = fmr_pos.sort_values("D_manual", ascending=False)

    tiers_present = [t for t in TIER_ORDER if t in set(fmr_pos.severity)]
    log(f"Tiers present in the FMR-positive set: {tiers_present}")
    for t in tiers_present:
        log(f"  {t}: {len(fmr_pos[fmr_pos.severity == t])} candidate rows, "
            f"speakers={sorted(fmr_pos[fmr_pos.severity==t].speaker_id.unique())}")

    # stratified selection: round-robin across tiers (by D desc within tier),
    # capped at 2 examples per speaker, until 20 are picked or candidates exhaust.
    selected, speaker_count = [], {}
    per_tier_iter = {t: iter(fmr_pos[fmr_pos.severity == t].itertuples()) for t in tiers_present}
    exhausted = set()
    while len(selected) < 20 and len(exhausted) < len(tiers_present):
        for t in tiers_present:
            if t in exhausted or len(selected) >= 20:
                continue
            picked_this_tier = False
            for row in per_tier_iter[t]:
                if speaker_count.get(row.speaker_id, 0) >= 2:
                    continue
                selected.append(row)
                speaker_count[row.speaker_id] = speaker_count.get(row.speaker_id, 0) + 1
                picked_this_tier = True
                break
            if not picked_this_tier:
                exhausted.add(t)

    log(f"\nSelected {len(selected)} candidates after tier-stratification "
        f"(cap 2/speaker, round-robin across {len(tiers_present)} tiers).")
    if len(selected) < 20:
        log(f"NOTE: fewer than 20 available after stratification+cap "
            f"({len(selected)} selected) — not loosening the filter to fill the table.")
    sel_ids = {r.item_id for r in selected}
    log(f"Speaker distribution in final selection: "
        f"{ {r.speaker_id: speaker_count[r.speaker_id] for r in selected} }\n")

    # --- pull the other two systems' hyps on the SAME item_id (no re-generation) ---
    other = df[(df.item_id.isin(sel_ids)) & (df.system.isin(["wav2vec2_960h", "parakeet_tdt"]))]
    hyp_map = other.pivot_table(index="item_id", columns="system", values="hypothesis", aggfunc="first")

    rows_out = []
    for r in selected:
        rows_out.append({
            "utt_id": r.item_id,
            "speaker": r.speaker_id,
            "tier": r.severity,
            "reference": r.reference,
            "whisper_hyp": r.hypothesis,
            "wav2vec2_hyp": hyp_map.loc[r.item_id, "wav2vec2_960h"] if r.item_id in hyp_map.index else "",
            "parakeet_hyp": hyp_map.loc[r.item_id, "parakeet_tdt"] if r.item_id in hyp_map.index else "",
            "phi": round(float(r.phi), 4),
            "sem_sim": round(float(r.s), 4),
            "D": round(float(r.D_manual), 4),
            "conf": round(float(r.conf), 4) if pd.notna(r.conf) else "",
        })
    cand_df = pd.DataFrame(rows_out)

    # =========================== STEP 5: outputs =====================================
    fmr_csv = out_dir / "fmr_candidates.csv"
    cand_df.to_csv(fmr_csv, index=False)
    log(f"## Step 5 — outputs")
    log(f"Wrote {fmr_csv}  ({len(cand_df)} rows)")

    top8 = cand_df.head(8).reset_index(drop=True)

    # LaTeX table: 4 text columns (ref + 3 hyps) is too cramped for LNCS single-column
    # at readable font size -> two-row-per-example layout (reference full-width row,
    # the three system hypotheses as a sub-row beneath), per the plan's fallback option.
    tex = []
    tex.append(r"\begin{table}[t]")
    tex.append(r"\caption{Fluent misrecognition examples (Whisper, impaired sentences, "
               r"top 8 by $D=\phi\cdot(1-s)$). Hypotheses over 90 characters are truncated "
               r"with \ldots; casing/punctuation preserved as transcribed.}")
    tex.append(r"\label{tab:qualitative}")
    tex.append(r"\footnotesize")
    tex.append(r"\begin{tabular}{@{}p{0.14\textwidth}p{0.10\textwidth}p{0.72\textwidth}@{}}")
    tex.append(r"\toprule")
    tex.append(r"\textbf{Speaker (tier)} & \textbf{$\phi$ / $s$ / $D$} & \textbf{Reference \& hypotheses} \\")
    tex.append(r"\midrule")
    trunc_flagged = False
    for i, row in top8.iterrows():
        ref_esc, _ = truncate(escape_latex(row["reference"]), 90)
        wh_esc, t1 = truncate(escape_latex(row["whisper_hyp"]), 90)
        wv_esc, t2 = truncate(escape_latex(row["wav2vec2_hyp"]), 90)
        pk_esc, t3 = truncate(escape_latex(row["parakeet_hyp"]), 90)
        trunc_flagged = trunc_flagged or t1 or t2 or t3
        meta = f"{row['speaker']} ({row['tier']})"
        stats = f"{row['phi']:.2f} / {row['sem_sim']:.2f} / {row['D']:.2f}"
        text = (rf"\textit{{Ref:}} {ref_esc}\newline "
                rf"\textbf{{Whisper:}} {wh_esc}\newline "
                rf"\textbf{{wav2vec2:}} {wv_esc}\newline "
                rf"\textbf{{Parakeet:}} {pk_esc}")
        tex.append(rf"{meta} & {stats} & {text} \\")
        if i < len(top8) - 1:
            tex.append(r"\addlinespace")
    tex.append(r"\bottomrule")
    tex.append(r"\end{tabular}")
    tex.append(r"\end{table}")
    tex_path = out_dir / "tab_qualitative.tex"
    tex_path.write_text("\n".join(tex) + "\n", encoding="utf-8")
    log(f"Wrote {tex_path}  (top 8; two-row-per-example layout chosen: 4 side-by-side "
        f"p{{}} text columns at footnotesize did not fit \\textwidth for a single-column "
        f"LNCS page at readable width, so reference+hypotheses are stacked as one cell "
        f"per example instead.)")
    log(f"Truncation occurred in the LaTeX table: {trunc_flagged}")

    log("\n## Top 8 selected examples\n")
    for _, row in top8.iterrows():
        log(f"[{row['speaker']} / {row['tier']}]  phi={row['phi']:.3f}  s={row['sem_sim']:.3f}  D={row['D']:.3f}")
        log(f"  ref:      {row['reference']}")
        log(f"  whisper:  {row['whisper_hyp']}")
        log(f"  wav2vec2: {row['wav2vec2_hyp']}")
        log(f"  parakeet: {row['parakeet_hyp']}")
        log("")

    report_path = out_dir / "report.txt"
    report_path.write_text("\n".join(report), encoding="utf-8")
    print("\n".join(report))
    print(f"\n=== Wrote {report_path}, {fmr_csv}, {tex_path} ===")


if __name__ == "__main__":
    main()
