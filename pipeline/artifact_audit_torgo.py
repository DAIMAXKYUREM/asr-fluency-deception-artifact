"""
artifact_audit_torgo.py — CHECK B: stock-phrase contamination of the Whisper
FMR-positive set on TORGO impaired sentences. Read-only: does not modify any
existing results file. Writes results/qualitative/artifact_audit.md.

Artifact detection is DATA-DERIVED, not a hardcoded guess list:
a hypothesis string is flagged as a stock-phrase artifact iff, among the
system's impaired-set outputs, it:
  (a) recurs >= MIN_COUNT times, AND
  (b) is paired with a highly diverse set of DIFFERENT references
      (n_distinct_refs / count >= DIVERSITY_THRESH) -- the tell that it is a
      canned output unrelated to the acoustic input, not a genuinely correct
      repeated transcription of a repeated prompt, AND
  (c) has low mean semantic similarity to those references (mean_s < S_THRESH)
      -- confirming it is usually NOT a correct answer.
This rule is applied independently per system (whisper/wav2vec2/parakeet) so
no phrase list is assumed in advance; the printed tables make the rule and
its inputs fully auditable.

Severity is irrelevant to this check (uses is_control + prompt_type only), so
the per_item_novel.csv severity staleness noted in Check A does not apply here.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

RUN = Path(__file__).resolve().parent.parent / "results" / "torgo_asr_01"
OUT_DIR = Path(__file__).resolve().parent.parent / "results" / "qualitative"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MIN_COUNT = 3          # must recur at least this often to be a candidate
DIVERSITY_THRESH = 0.8  # >=80% of its occurrences pair with a distinct reference
S_THRESH = 0.30         # mean semantic similarity to those references must be low

EXPECTED_FGW = {"whisper_large_v3": 0.632, "wav2vec2_960h": 0.536, "parakeet_tdt": 0.522}


def norm(s):
    """Near-exact match key: lowercase, strip punctuation/whitespace."""
    if not isinstance(s, str):
        return ""
    return re.sub(r"[^\w\s]", "", s.lower()).strip()


def fmr_block(g: pd.DataFrame) -> dict:
    wrong = g.loc[g["s"] <= 0.5, "phi"]
    fmr = float(((g.phi >= 0.5) & (g.s <= 0.5)).mean()) if len(g) else float("nan")
    fgw = float(wrong.median()) if len(wrong) else float("nan")
    return {"n": int(len(g)), "n_wrong": int(len(wrong)), "FMR": fmr, "fluency_given_wrong": fgw}


def artifact_table(g_impaired: pd.DataFrame) -> pd.DataFrame:
    """Per distinct hypothesis string (normalised): count, distinct refs, mean s, artifact flag."""
    g = g_impaired.copy()
    g["hyp_norm"] = g["hypothesis"].map(norm)
    rows = []
    for h, sub in g.groupby("hyp_norm"):
        if h == "":
            continue
        n = len(sub)
        n_distinct_refs = sub["reference"].nunique()
        mean_s = float(sub["s"].mean())
        diversity = n_distinct_refs / n
        is_artifact = (n >= MIN_COUNT) and (diversity >= DIVERSITY_THRESH) and (mean_s < S_THRESH)
        rows.append({"hyp_norm": h, "example": sub["hypothesis"].iloc[0], "count": n,
                     "n_distinct_refs": n_distinct_refs, "diversity": round(diversity, 3),
                     "mean_s": round(mean_s, 3), "is_artifact": is_artifact})
    return pd.DataFrame(rows).sort_values("count", ascending=False)


def main():
    a = pd.read_csv(RUN / "per_item.csv")
    b = pd.read_csv(RUN / "per_item_novel.csv")[["item_id", "system", "phi", "s"]]
    df = a.merge(b, on=["item_id", "system"], how="left")
    df["hypothesis"] = df["hypothesis"].fillna("")

    sent = df[df.prompt_type == "sentence"]
    imp_sent = sent[~sent.is_control]

    L = ["# Artifact audit — Whisper stock-phrase contamination of the FMR set (TORGO)\n",
         "Read-only check; no existing results file was modified. "
         "Artifact classification is derived from the data (frequency + cross-reference "
         "diversity + low semantic similarity), not a hardcoded phrase list — see rule "
         f"in the script header (MIN_COUNT={MIN_COUNT}, DIVERSITY_THRESH={DIVERSITY_THRESH}, "
         f"S_THRESH={S_THRESH}).\n"]

    # ---------- Step 1: Whisper FMR-positive impaired-sentence rows, duplicates -----
    wh_sent = imp_sent[imp_sent.system == "whisper_large_v3"]
    fmr_pos = wh_sent[(wh_sent.phi >= 0.5) & (wh_sent.s <= 0.5)].copy()
    L.append(f"## Step 1 — Whisper FMR-positive impaired-sentence rows\n")
    L.append(f"n = {len(fmr_pos)} (expected 127 from the prior qualitative-table run).\n")
    fmr_pos["hyp_norm"] = fmr_pos["hypothesis"].map(norm)
    dupe_counts = fmr_pos["hyp_norm"].value_counts()
    dupes = dupe_counts[dupe_counts > 1]
    L.append("Hypothesis strings occurring more than once within the 127 (exact/near-exact, "
             "case+punctuation-insensitive):\n")
    if len(dupes) == 0:
        L.append("_None — every FMR-positive Whisper hypothesis in this set is unique._\n")
    else:
        L.append("| hypothesis (as first seen) | count |")
        L.append("|---|---|")
        for h, c in dupes.items():
            example = fmr_pos[fmr_pos.hyp_norm == h]["hypothesis"].iloc[0]
            L.append(f"| {example} | {c} |")
        L.append("")

    # ---------- Step 2: top 40 hypothesis strings, whole Whisper impaired set -------
    L.append("## Step 2 — 40 most frequent hypothesis strings, whole Whisper impaired set\n")
    L.append(f"(All prompt types, is_control=False; n={len(wh_sent) if False else len(df[(df.system=='whisper_large_v3') & (~df.is_control)])} rows.)\n")
    wh_imp_all = df[(df.system == "whisper_large_v3") & (~df.is_control)].copy()
    wh_imp_all["hyp_norm"] = wh_imp_all["hypothesis"].map(norm)
    top40 = wh_imp_all["hyp_norm"].value_counts().head(40)
    L.append("| rank | hypothesis (as first seen) | count | n_distinct_refs | mean s |")
    L.append("|---|---|---|---|---|")
    for i, (h, c) in enumerate(top40.items(), 1):
        sub = wh_imp_all[wh_imp_all.hyp_norm == h]
        example = sub["hypothesis"].iloc[0] if h else "(empty)"
        L.append(f"| {i} | {example} | {c} | {sub['reference'].nunique()} | {sub['s'].mean():.3f} |")
    L.append("")

    # ---------- Step 3: data-derived artifact classification, per system -----------
    # IMPORTANT: the artifact list is derived from the WHOLE impaired set (all prompt
    # types, matching Step 2's population) -- NOT the sentence-only FMR population --
    # then APPLIED as an exclusion filter to the sentence-only FMR-relevant rows. This
    # matters: stock phrases like "Thank you." recur heavily on isolated WORD prompts
    # (very short/ambiguous audio) but are almost absent on SENTENCE prompts, so
    # deriving+applying within sentence-only scope alone would trivially find nothing.
    L.append("## Step 3 — data-derived artifact classification and FMR recomputation\n")
    L.append("Artifact list is derived from the WHOLE impaired set (word+sentence, "
             "Step 2's population) per system, then applied as an exclusion filter to "
             "that system's sentence-only FMR-relevant population.\n")
    results_summary = []
    for sysn in ["whisper_large_v3", "wav2vec2_960h", "parakeet_tdt"]:
        g_whole = df[(df.system == sysn) & (~df.is_control)]   # whole impaired set, all prompt types
        atab = artifact_table(g_whole)
        artifacts = atab[atab.is_artifact]
        g_all = imp_sent[imp_sent.system == sysn]               # sentence-only, for FMR recompute
        L.append(f"### {sysn}\n")
        L.append(f"Distinct hypothesis strings in the WHOLE impaired set (n={len(g_whole)}, "
                 f"word+sentence): {len(atab)}. Flagged as artifact: {len(artifacts)}.\n")
        if len(artifacts):
            L.append("| hypothesis | count | n_distinct_refs | diversity | mean s |")
            L.append("|---|---|---|---|---|")
            for _, r in artifacts.sort_values("count", ascending=False).iterrows():
                L.append(f"| {r['example']} | {r['count']} | {r['n_distinct_refs']} | "
                         f"{r['diversity']} | {r['mean_s']} |")
            L.append("")
        else:
            L.append("_No hypothesis strings met the artifact rule for this system._\n")

        artifact_norms = set(artifacts["hyp_norm"])
        g_all2 = g_all.copy()
        g_all2["hyp_norm"] = g_all2["hypothesis"].map(norm)

        fmr_pos_sys = g_all2[(g_all2.phi >= 0.5) & (g_all2.s <= 0.5)]
        n_fmr_pos = len(fmr_pos_sys)
        n_fmr_pos_artifact = int(fmr_pos_sys["hyp_norm"].isin(artifact_norms).sum())
        frac_artifact = n_fmr_pos_artifact / n_fmr_pos if n_fmr_pos else float("nan")

        before = fmr_block(g_all2)
        after_df = g_all2[~g_all2["hyp_norm"].isin(artifact_norms)]
        after = fmr_block(after_df)

        L.append(f"FMR-positive rows for {sysn}: {n_fmr_pos}. Of these, "
                 f"{n_fmr_pos_artifact} ({frac_artifact:.1%}) have an artifact-flagged hypothesis.\n")
        L.append("| | n (impaired sentences) | FMR | median fluency-given-wrong |")
        L.append("|---|---|---|---|")
        L.append(f"| before exclusion | {before['n']} | {before['FMR']:.4f} | {before['fluency_given_wrong']:.4f} |")
        L.append(f"| after excluding artifact-hypothesis rows | {after['n']} | {after['FMR']:.4f} | "
                 f"{after['fluency_given_wrong']:.4f} |")
        L.append(f"\n_Published fluency-given-wrong for {sysn}: {EXPECTED_FGW[sysn]:.3f}._\n")

        results_summary.append({
            "system": sysn, "n_artifact_strings": len(artifacts),
            "n_fmr_pos": n_fmr_pos, "n_fmr_pos_artifact": n_fmr_pos_artifact,
            "frac_artifact": frac_artifact,
            "fmr_before": before["FMR"], "fmr_after": after["FMR"],
            "fgw_before": before["fluency_given_wrong"], "fgw_after": after["fluency_given_wrong"],
        })

    # ---------- Step 4: does excluding artifacts change the ranking? ---------------
    L.append("## Step 4 — does excluding artifacts change the published ranking?\n")
    rs = pd.DataFrame(results_summary).set_index("system")
    L.append("| system | fluency-given-wrong BEFORE | fluency-given-wrong AFTER | rank BEFORE | rank AFTER |")
    L.append("|---|---|---|---|---|")
    order_before = rs["fgw_before"].rank(ascending=False).astype(int)
    order_after = rs["fgw_after"].rank(ascending=False).astype(int)
    for sysn in ["whisper_large_v3", "wav2vec2_960h", "parakeet_tdt"]:
        L.append(f"| {sysn} | {rs.loc[sysn,'fgw_before']:.4f} | {rs.loc[sysn,'fgw_after']:.4f} | "
                 f"#{order_before[sysn]} | #{order_after[sysn]} |")
    rank_changed = not order_before.equals(order_after)
    L.append(f"\n**Ranking changed: {rank_changed}.**\n")
    L.append(f"Whisper artifact share of its own FMR-positive set: "
             f"{rs.loc['whisper_large_v3','frac_artifact']:.1%} "
             f"({rs.loc['whisper_large_v3','n_fmr_pos_artifact']}/{rs.loc['whisper_large_v3','n_fmr_pos']} rows).")

    report_path = OUT_DIR / "artifact_audit.md"
    report_path.write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))
    print(f"\n=== Wrote {report_path} (no existing results file modified) ===")


if __name__ == "__main__":
    main()
