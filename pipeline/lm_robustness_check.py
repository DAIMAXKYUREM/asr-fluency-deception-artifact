"""
lm_robustness_check.py -- CODS 2026 Task 4: LM-choice robustness of the
fluency-given-wrong ranking.

Re-scores the SAME hypothesis strings (impaired speakers, sentence prompts,
TORGO) under a small causal LM from a DIFFERENT family than GPT-2 --
EleutherAI/gpt-neo-125m -- using the identical recipe already used for phi:
teacher-forced NLL -> perplexity -> z-score log-PPL across the pooled subset
-> sigmoid. No audio, text-only, CPU/GPU in minutes.

Recomputes median phi (call it phi_altlm) over s<=0.5 on the same subset and
checks whether the system ranking (Whisper > wav2vec2 > Parakeet) is
unchanged relative to the published GPT-2-based fluency-given-wrong
(0.632 / 0.536 / 0.522).

Outputs:
    results/torgo_asr_01/lm_robustness_results.csv
    results/torgo_asr_01/lm_robustness_report.md

Usage:
    python lm_robustness_check.py
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

RUN = Path(__file__).resolve().parent.parent / "results" / "torgo_asr_01"
SYSTEMS = ["whisper_large_v3", "wav2vec2_960h", "parakeet_tdt"]
LABEL = {"whisper_large_v3": "Whisper", "wav2vec2_960h": "wav2vec2", "parakeet_tdt": "Parakeet"}
EXPECTED_FGW = {"whisper_large_v3": 0.632, "wav2vec2_960h": 0.536, "parakeet_tdt": 0.522}
ALT_MODEL = "EleutherAI/gpt-neo-125m"


def compute_perplexities(texts, model_name: str) -> dict[str, float]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(dev).eval()
    out = {}
    with torch.no_grad():
        for t in texts:
            s = (t or "").strip()
            if not s:
                out[t] = float("nan"); continue
            ids = tok(s, return_tensors="pt").input_ids.to(dev)
            if ids.shape[1] < 2:
                bos = tok.bos_token or tok.eos_token or ""
                ids = tok(bos + " " + s, return_tensors="pt").input_ids.to(dev)
            if ids.shape[1] < 2:
                out[t] = float("nan"); continue
            out[t] = float(torch.exp(model(ids, labels=ids).loss))
    del model
    return out


def main():
    a = pd.read_csv(RUN / "per_item.csv")
    b = pd.read_csv(RUN / "per_item_novel.csv")[["item_id", "system", "phi", "s"]]
    df = a.merge(b, on=["item_id", "system"], how="left")
    df["hypothesis"] = df["hypothesis"].fillna("")

    sent = df[df.prompt_type == "sentence"]
    imp_sent = sent[~sent.is_control].copy()

    L = [f"# LM-choice robustness check -- {ALT_MODEL}\n",
         "Same subset as the published fluency-given-wrong (impaired speakers, "
         "sentence prompts, TORGO). Same recipe: teacher-forced NLL -> perplexity "
         "-> z-score log-PPL across this pooled subset -> sigmoid.\n"]

    print(f"loading {ALT_MODEL} ...", flush=True)
    uniq_texts = list(imp_sent["hypothesis"].unique())
    print(f"scoring {len(uniq_texts)} unique hypothesis strings ...", flush=True)
    ppl_map = compute_perplexities(uniq_texts, ALT_MODEL)
    imp_sent["ppl_alt"] = imp_sent["hypothesis"].map(ppl_map)
    lp = np.log(imp_sent["ppl_alt"].replace([np.inf, -np.inf], np.nan))
    mu, sd = float(lp.mean()), (float(lp.std()) or 1.0)
    imp_sent["phi_altlm"] = (1.0 / (1.0 + np.exp((lp - mu) / sd))).fillna(0.0)
    imp_sent.loc[imp_sent["hypothesis"].str.strip() == "", "phi_altlm"] = 0.0

    L.append("## Median phi_altlm over s<=0.5 (the fluency-given-wrong analog), per system\n")
    L.append("| system | phi_altlm (GPT-Neo-125m) | published fluency-given-wrong (GPT-2) | rank (alt) | rank (published) |")
    L.append("|---|---|---|---|---|")
    rows = []
    for sysn in SYSTEMS:
        g = imp_sent[imp_sent.system == sysn]
        wrong = g.loc[g.s <= 0.5, "phi_altlm"]
        fgw_alt = float(wrong.median()) if len(wrong) else float("nan")
        rows.append({"system": sysn, "fluency_given_wrong_altlm": fgw_alt,
                     "fluency_given_wrong_gpt2": EXPECTED_FGW[sysn]})
    res = pd.DataFrame(rows).set_index("system")
    rank_alt = res["fluency_given_wrong_altlm"].rank(ascending=False).astype(int)
    rank_pub = res["fluency_given_wrong_gpt2"].rank(ascending=False).astype(int)
    for sysn in SYSTEMS:
        L.append(f"| {sysn} | {res.loc[sysn,'fluency_given_wrong_altlm']:.4f} | "
                 f"{res.loc[sysn,'fluency_given_wrong_gpt2']:.3f} | #{rank_alt[sysn]} | #{rank_pub[sysn]} |")

    ranking_unchanged = rank_alt.equals(rank_pub)
    L.append(f"\n**Ranking unchanged: {ranking_unchanged}.**\n")
    res["rank_altlm"] = rank_alt
    res["rank_gpt2"] = rank_pub
    res["ranking_unchanged"] = ranking_unchanged
    res.to_csv(RUN / "lm_robustness_results.csv")

    L.append(f"Footnote sentence (drop into Section 5.2):\n")
    vals = " / ".join(f"{res.loc[s,'fluency_given_wrong_altlm']:.3f}" for s in SYSTEMS)
    L.append(f'"Re-scoring the same outputs under GPT-Neo-125M '
             f'(EleutherAI), a causal LM from a different family than GPT-2, gives '
             f'fluency-given-wrong of {vals} for Whisper / wav2vec2 / Parakeet respectively, '
             f'{"the same" if ranking_unchanged else "a DIFFERENT"} ranking as the GPT-2-scored '
             f'result reported above."')

    (RUN / "lm_robustness_report.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))
    print(f"\n=== Wrote {RUN/'lm_robustness_report.md'} and {RUN/'lm_robustness_results.csv'} ===")


if __name__ == "__main__":
    main()
