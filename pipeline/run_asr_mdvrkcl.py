"""
run_asr_mdvrkcl.py — ASR-track evaluation on MDVR-KCL ReadText (Parkinson's arm).

MDVR-KCL ships no transcripts. Each ReadText recording is a free read of TWO
passages — "The North Wind and the Sun" (NWS) + a BNC technical passage on
atmospheric scattering — in VARYING order, plus a spoken citation header and
filler. So:
  * Reference = canonical NWS + technical passage (the technical text is
    RECONSTRUCTED from clean control reads — flagged as a limitation).
  * Order varies, so we score against both orderings (NWS+TECH and TECH+NWS)
    and keep the lower-WER one.
  * Header/filler are not in the reference -> uniform insertions across systems,
    so cross-system / PD-vs-control comparisons are valid; absolute WER is
    mildly inflated.
  * Long recordings are chunked for wav2vec2 / Parakeet (Whisper handles long
    audio natively).

Severity = UPDRS speech score from the filename (ID<n>_pd_<H&Y>_<UPDRS-II>_<UPDRS-III>).

Outputs: results/mdvrkcl_asr_01/{per_item.csv, summary.md, summary.json}
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from asr_systems import DEFAULT_SYSTEMS, SYSTEM_REGISTRY, build_asr_system, load_audio
from metrics import compute_recognition_metrics, _WER_TRANSFORM
import jiwer

ROOT = Path(__file__).resolve().parent.parent
READDIR = ROOT / "data" / "MDVR-KCL" / "26-29_09_2017_KCL" / "ReadText"
OUT = ROOT / "results" / "mdvrkcl_asr_01"
SR = 16_000
CHUNK = 20  # seconds, for non-Whisper long-audio chunking

NWS = ("The North Wind and the Sun were disputing which was the stronger, when a "
       "traveller came along wrapped in a warm cloak. They agreed that the one who "
       "first succeeded in making the traveller take his cloak off should be considered "
       "stronger than the other. Then the North Wind blew as hard as he could, but the "
       "more he blew the more closely did the traveller fold his cloak around him; and "
       "at last the North Wind gave up the attempt. Then the Sun shone out warmly, and "
       "immediately the traveller took off his cloak. And so the North Wind was obliged "
       "to confess that the Sun was the stronger of the two.")

# RECONSTRUCTED from clean control reads (ID00/ID01 HC) — not an authoritative transcript.
TECH = ("This is because there is less scattering of blue light as the atmospheric path "
        "length and consequently the degree of scattering of the incoming radiation is "
        "reduced. For the same reason the sun appears to be whiter and less orange "
        "coloured as the observer's altitude increases. This is because a greater "
        "proportion of the sunlight comes directly to the observer's eye. Figure 5.7 is a "
        "schematic representation of the path of electromagnetic energy in the visible "
        "spectrum as it travels from the sun to the earth and back again towards a sensor "
        "mounted on an orbiting satellite. The paths of waves representing energy prone to "
        "scattering, that is the shorter wavelengths, as it travels from sun to earth are "
        "shown to the sensor. It appears that all the energy has been reflected from point "
        "P on the ground, whereas in fact it has not because some has been scattered "
        "within the atmosphere and has never reached the ground at all.")

REF_AB = NWS + " " + TECH
REF_BA = TECH + " " + NWS
UPDRS_TIER = {0: "none", 1: "mild", 2: "moderate", 3: "severe", 4: "profound"}
_FN = re.compile(r"ID(\d+)_(hc|pd)_(\d+)_(\d+)_(\d+)", re.I)


def parse_fn(stem: str):
    m = _FN.match(stem)
    if not m:
        return None
    idn, grp, hy, u2, u3 = m.groups()
    ctrl = grp.lower() == "hc"
    usp = max(int(u2), int(u3))
    return {"speaker_id": f"ID{idn}", "is_control": ctrl,
            "condition": "control" if ctrl else "parkinsons",
            "hoehn_yahr": int(hy), "updrs_speech": usp,
            "severity": "control" if ctrl else UPDRS_TIER.get(usp, "unknown")}


def _wer(ref, hyp):
    return jiwer.process_words(ref, hyp, reference_transform=_WER_TRANSFORM,
                               hypothesis_transform=_WER_TRANSFORM).wer


def transcribe_long(backend, audio, is_whisper):
    if is_whisper:
        return backend.transcribe(audio)
    parts, w = [], CHUNK * SR
    for st in range(0, len(audio), w):
        parts.append(backend.transcribe(audio[st:st + w]).get("text", ""))
    return {"text": " ".join(p for p in parts if p).strip(), "words": []}


def run_system(system, recs, hyps_dir, device):
    """Transcribe every ReadText recording for one system; resumable via jsonl cache."""
    cache = hyps_dir / f"{system}.jsonl"
    done = {}
    if cache.exists():
        for line in cache.open(encoding="utf-8"):
            try:
                r = json.loads(line); done[r["id"]] = r["text"]
            except Exception:
                pass
    if all(r.stem in done for r in recs):
        print(f"  [{system}] all {len(done)} cached", flush=True)
        return
    is_wh = system == "whisper_large_v3"
    print(f"  [{system}] {SYSTEM_REGISTRY[system]['label']} on {device}", flush=True)
    backend = build_asr_system(system, device=device)
    try:
        with cache.open("a", encoding="utf-8") as f:
            for r in recs:
                if r.stem in done:
                    continue
                audio = load_audio(str(r))
                try:
                    hyp = transcribe_long(backend, audio, is_wh)
                except Exception as exc:
                    hyp = {"text": "", "words": []}
                    print(f"    err {r.stem}: {exc}", flush=True)
                done[r.stem] = hyp["text"]
                f.write(json.dumps({"id": r.stem, "text": hyp["text"]}) + "\n"); f.flush()
                print(f"    {r.stem:22} ok", flush=True)
    finally:
        backend.unload()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--systems", nargs="+", default=DEFAULT_SYSTEMS, choices=list(SYSTEM_REGISTRY))
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    hyps_dir = OUT / "hyps"; hyps_dir.mkdir(exist_ok=True)
    recs = sorted(READDIR.rglob("*.wav"))
    meta = {r.stem: parse_fn(r.stem) for r in recs}
    recs = [r for r in recs if meta[r.stem]]
    print(f"{len(recs)} ReadText recordings", flush=True)

    # transcription: ONE SYSTEM PER PROCESS (avoids the CTranslate2<->torch cuDNN crash)
    for system in args.systems:
        dev = "cpu" if system == "parakeet_tdt" else args.device
        run_system(system, recs, hyps_dir, dev)

    # scoring/aggregation: every system whose cache is complete (loads no ASR model)
    rows = []
    for system in DEFAULT_SYSTEMS:
        cache = hyps_dir / f"{system}.jsonl"
        if not cache.exists():
            continue
        hyps = {json.loads(l)["id"]: json.loads(l)["text"] for l in cache.open(encoding="utf-8")}
        if not all(r.stem in hyps for r in recs):
            print(f"  [{system}] cache incomplete ({len(hyps)}/{len(recs)}) — skip scoring", flush=True)
            continue
        print(f"  [{system}] scoring ...", flush=True)
        for r in recs:
            text = hyps[r.stem]
            ref = REF_AB if _wer(REF_AB, text) <= _wer(REF_BA, text) else REF_BA
            rec = compute_recognition_metrics({"transcript": ref}, {"text": text, "words": []})
            rows.append({"id": r.stem, "system": system, **meta[r.stem],
                         "ref_words": len(ref.split()), "reference": ref, "hypothesis": text, **rec})
    if not rows:
        print("no system has a complete cache yet — re-run to continue.", flush=True)
        return

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "per_item.csv", index=False)

    keys = ["wer", "cer", "hallucination_rate", "semantic_similarity",
            "content_word_wer", "function_word_wer", "sub_rate", "del_rate", "ins_rate"]
    def agg(d):
        return {k: round(float(d[k].dropna().mean()), 4) for k in keys}

    present = [s for s in DEFAULT_SYSTEMS if s in set(df.system)]
    summary = {"corpus": "MDVR-KCL ReadText", "n_recordings": len(recs),
               "reference": "canonical NWS + reconstructed technical passage; best-of-2 orderings",
               "systems": {}}
    L = ["# MDVR-KCL (Parkinson's) — ASR track\n",
         f"- {len(recs)} ReadText recordings (read passages); reference reconstructed (see header).",
         "- Macro-mean over recordings (each is one long passage read).\n",
         "## Overall + control vs PD\n",
         "| system | scope | WER | CER | Halluc | SemSim |", "|---|---|---|---|---|---|"]
    for s in present:
        sd = df[df.system == s]
        ov, hc, pd_ = agg(sd), agg(sd[sd.is_control]), agg(sd[~sd.is_control])
        summary["systems"][s] = {"label": SYSTEM_REGISTRY[s]["label"], "overall": ov,
                                 "control": hc, "parkinsons": pd_,
                                 "by_severity": {sev: agg(g) for sev, g in sd.groupby("severity")}}
        for scope, a in (("overall", ov), ("control", hc), ("PD", pd_)):
            L.append(f"| {SYSTEM_REGISTRY[s]['label']} | {scope} | {a['wer']:.3f} | {a['cer']:.3f} "
                     f"| {a['hallucination_rate']:.3f} | {a['semantic_similarity']:.3f} |")
    L.append("\n## PD WER by UPDRS speech severity\n")
    L.append("| system | none | mild | moderate |"); L.append("|---|---|---|---|")
    for s in present:
        bs = summary["systems"][s]["by_severity"]
        L.append(f"| {SYSTEM_REGISTRY[s]['label']} | " +
                 " | ".join(f"{bs.get(t,{}).get('wer','—')}" for t in ("none", "mild", "moderate")) + " |")
    L.append("\n_Reference for the technical passage is reconstructed from clean control reads; "
             "header/filler are uniform insertions. Cross-system and PD-vs-control contrasts are "
             "valid; absolute WER is mildly inflated._")
    (OUT / "summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    with open(OUT / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=lambda o: None if isinstance(o, float) and np.isnan(o) else o)
    print("\n".join(L))
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
