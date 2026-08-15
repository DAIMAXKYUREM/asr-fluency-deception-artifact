"""
run_confidence.py — Step 0/1 of the inference-time deception detector.

Re-runs inference (same audio, inference only) to capture ONE confidence scalar
per utterance, per architecture. The text cache has no scores, so this cannot be
skipped. RESUMABLE (per-system jsonl cache), so it survives interruption.

Confidence is exposed completely differently by the three models:
  whisper_large_v3  faster-whisper: avg_conf = mean exp(avg_logprob) over segments;
                    min_conf = min word.probability  (weakest-link signal)
  wav2vec2_960h     CTC: softmax over logits, keep NON-BLANK frames only
                    (blank==pad_token_id==0); avg/min of the per-frame max-prob
  parakeet_tdt      onnx-asr .with_timestamps() -> per-token logprobs;
                    avg_conf = exp(mean logprob), min_conf = exp(min logprob)
                    (empty output -> no tokens -> conf recorded as 0.0, degenerate)

Every scalar is on a DIFFERENT scale per model — normalization happens later,
per-model, in deception_detector.py. Here we just record the raw numbers.

Usage:
    python run_confidence.py --manifest ../data/manifests/torgo_manifest.json \
        --output_dir ../results/torgo_asr_01 --device cuda
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from tqdm import tqdm

from asr_systems import (SYSTEM_REGISTRY, DEFAULT_SYSTEMS, build_asr_system,
                         load_audio, SAMPLE_RATE)

SR = SAMPLE_RATE
GPU_UNSAFE = {"parakeet_tdt"}


def _slice(audio, item):
    s, e = item.get("start"), item.get("end")
    if s is None or e is None:
        return audio
    return audio[int(float(s) * SR): int(float(e) * SR)]


# --- per-architecture confidence extractors ------------------------------------

def conf_whisper(backend, audio) -> dict:
    segs, _ = backend.model.transcribe(
        audio, language="en", beam_size=5, word_timestamps=True,
        condition_on_previous_text=False, no_speech_threshold=0.6,
        log_prob_threshold=-1.0, hallucination_silence_threshold=2.0)
    segs = list(segs)
    seg_conf = [np.exp(s.avg_logprob) for s in segs if s.avg_logprob is not None]
    wp = [w.probability for s in segs for w in (s.words or []) if w.probability is not None]
    avg = float(np.mean(seg_conf)) if seg_conf else float("nan")
    mn = float(min(wp)) if wp else avg
    return {"avg_conf": avg, "min_conf": mn, "n_units": len(wp)}


def conf_wav2vec2(backend, audio) -> dict:
    import torch
    import torch.nn.functional as F
    proc, model, dev = backend.processor, backend.model, backend.device
    blank = model.config.pad_token_id or 0
    iv = proc(audio, sampling_rate=SR, return_tensors="pt").input_values.to(dev)
    with torch.no_grad():
        logits = model(iv).logits
    probs = F.softmax(logits, dim=-1)
    maxp, ids = probs.max(dim=-1)
    ids, maxp = ids.squeeze(0), maxp.squeeze(0)
    mask = ids != blank
    if bool(mask.any()):
        emit = maxp[mask]
        avg, mn = float(emit.mean()), float(emit.min())
        n = int(mask.sum())
    else:
        avg = mn = float(maxp.mean()); n = 0
    return {"avg_conf": avg, "min_conf": mn, "n_units": n}


def conf_parakeet(backend, audio) -> dict:
    wav = np.ascontiguousarray(audio, dtype=np.float32)
    res = backend._ts.recognize(wav, sample_rate=SR)
    lp = np.asarray(res.logprobs, dtype=float) if res.logprobs is not None else None
    if lp is not None and len(lp):
        return {"avg_conf": float(np.exp(lp.mean())),
                "min_conf": float(np.exp(lp.min())), "n_units": int(len(lp))}
    return {"avg_conf": 0.0, "min_conf": 0.0, "n_units": 0}   # empty/degenerate output


EXTRACTOR = {"whisper": conf_whisper, "wav2vec2": conf_wav2vec2, "parakeet": conf_parakeet}


def run_system(system, manifest, conf_dir, device):
    cache = conf_dir / f"{system}.jsonl"
    done = set()
    if cache.exists():
        for line in cache.open(encoding="utf-8"):
            try:
                done.add(json.loads(line)["id"])
            except Exception:
                pass
    if all(u["id"] in done for u in manifest):
        print(f"  [{system}] all {len(done)} cached", flush=True)
        return
    kind = SYSTEM_REGISTRY[system]["kind"]
    extract = EXTRACTOR[kind]
    print(f"  [{system}] {SYSTEM_REGISTRY[system]['label']} on {device}", flush=True)
    backend = build_asr_system(system, device=device)
    if kind == "parakeet":
        backend._ts = backend.model.with_timestamps()   # timestamped adapter -> logprobs

    by_wav = defaultdict(list)
    for u in manifest:
        by_wav[u["audio_path"]].append(u)
    try:
        with cache.open("a", encoding="utf-8") as f:
            for wav, utts in tqdm(by_wav.items(), desc=f"conf:{system}"):
                if all(u["id"] in done for u in utts):
                    continue
                audio = load_audio(wav)
                for u in utts:
                    if u["id"] in done:
                        continue
                    try:
                        c = extract(backend, _slice(audio, u))
                    except Exception as exc:
                        c = {"avg_conf": float("nan"), "min_conf": float("nan"), "n_units": 0}
                        print(f"    err {u['id']}: {exc}", flush=True)
                    f.write(json.dumps({"id": u["id"], **c}) + "\n"); f.flush()
                    done.add(u["id"])
    finally:
        backend.unload()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--systems", nargs="+", default=DEFAULT_SYSTEMS, choices=list(SYSTEM_REGISTRY))
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--parakeet_device", default="cpu", choices=["cuda", "cpu"])
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    out = Path(args.output_dir); conf_dir = out / "conf"; conf_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.load(open(args.manifest, encoding="utf-8"))
    if args.limit:
        manifest = manifest[: args.limit]
    print(f"{len(manifest)} utterances -> {conf_dir}", flush=True)
    for system in args.systems:
        dev = args.parakeet_device if system in GPU_UNSAFE else args.device
        run_system(system, manifest, conf_dir, dev)
    print("\nconfidence capture complete. Next: deception_detector.py")


if __name__ == "__main__":
    main()
