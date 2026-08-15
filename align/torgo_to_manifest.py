"""
torgo_to_manifest.py

Build a neuro_bench manifest directly from the TORGO corpus — no MFA needed.

For the ASR track each item needs only (audio, transcript), so we iterate over
the prompt files and pair each with its head-microphone WAV.  We:
  1. Skip non-speech motor tasks (prompts in square brackets) and unintelligible
     'xxx' prompts; strip inline instructions like "tear [as in ...]".
  2. Attach phone-level timing (true_eos_sec, intra-utterance pauses) from the
     TIMIT-format .PHN files WHEN PRESENT — they exist for only ~30% of TORGO
     and are missing entirely for several speakers, so PHN is optional and used
     only by the (separate) endpointing track, never by the ASR track.
  3. Use the head-microphone WAV (better quality than the array mic).

TORGO speaker labels — FDA severity (Rudzicz et al. 2012; Mengistu & Rudzicz 2011):
    Severe:          F01, M01, M02, M04
    Moderate-severe: M05
    Moderate:        F03
    Very mild:       F04, M03   (approach control-level intelligibility)
    Controls:        FC01, FC02, FC03, MC01, MC02, MC03, MC04

Usage:
    python torgo_to_manifest.py \
        --torgo_dir  data/TORGO \
        --output     data/manifests/torgo_manifest.json \
        [--mic       headMic]     # headMic (default) or arrayMic
        [--min_pause 0.15]        # minimum intra-utterance silence to annotate (sec)
"""

import argparse
import json
import re
from pathlib import Path

SAMPLE_RATE = 16_000

SILENCE_LABELS = {"sil", "sp", "h#", "pau", "epi"}

# Corpus-level severity metadata — Frenchay Dysarthria Assessment (FDA)
# administered at the University of Toronto, per Rudzicz, Namasivayam & Wolff
# (Language Resources and Evaluation, 2012); intelligibility % from Mengistu &
# Rudzicz (2011), 14 naive listeners, closed-set word recognition.
# Tiers: very_mild < moderate < moderate_severe < severe.
SPEAKER_META = {
    # Dysarthric — FDA severity (intelligibility %)
    "F01": {"condition": "dysarthria", "severity": "severe",          "is_control": False},  # ~69-82%
    "F03": {"condition": "dysarthria", "severity": "moderate",        "is_control": False},  # 90.0%
    "F04": {"condition": "dysarthria", "severity": "very_mild",       "is_control": False},  # ~control-level
    "M01": {"condition": "dysarthria", "severity": "severe",          "is_control": False},  # ~69-82%
    "M02": {"condition": "dysarthria", "severity": "severe",          "is_control": False},  # ~69-82%
    "M03": {"condition": "dysarthria", "severity": "very_mild",       "is_control": False},  # ~control-level
    "M04": {"condition": "dysarthria", "severity": "severe",          "is_control": False},  # ~69-82%
    "M05": {"condition": "dysarthria", "severity": "moderate_severe", "is_control": False},  # 87.88%
    # Controls
    "FC01": {"condition": "control", "severity": "N/A", "is_control": True},
    "FC02": {"condition": "control", "severity": "N/A", "is_control": True},
    "FC03": {"condition": "control", "severity": "N/A", "is_control": True},
    "MC01": {"condition": "control", "severity": "N/A", "is_control": True},
    "MC02": {"condition": "control", "severity": "N/A", "is_control": True},
    "MC03": {"condition": "control", "severity": "N/A", "is_control": True},
    "MC04": {"condition": "control", "severity": "N/A", "is_control": True},
}

_BRACKET_RE   = re.compile(r"^\s*\[")          # prompts starting with '[' are motor tasks
_INLINE_BRKT  = re.compile(r"\[[^\]]*\]")      # inline instructions, e.g. "tear [as in ...]"
_XXX_RE       = re.compile(r"\bx{2,}\b", re.I)  # 'xxx' = unintelligible marker


def clean_transcript(raw: str) -> str | None:
    """
    Turn a raw TORGO prompt into a clean spoken-reference string, or None to skip.

    - Inline instructions in square brackets are NOT spoken -> stripped
      ("tear [as in tear up that paper]" -> "tear").
    - 'xxx' marks an unintelligible/unreliable reference -> skip the item.
    - Anything left without alphabetic content -> skip.
    """
    text = _INLINE_BRKT.sub(" ", raw)
    text = re.sub(r"\s+", " ", text).strip()
    if not text or _XXX_RE.search(text):
        return None
    if not re.search(r"[a-zA-Z]", text):
        return None
    return text


def parse_phn(phn_path: Path, min_pause_dur: float = 0.15):
    """
    Parse a TIMIT-format .PHN file.

    Returns:
        transcript       — empty string (TORGO PHNs have no word labels, only phones)
        true_eos_sec     — end time of the last non-silence phone (seconds)
        intra_pauses     — [[start, end], ...] silence spans strictly inside the utterance
    Returns (None, None, []) if the file has only silence.
    """
    intervals = []  # (start_sec, end_sec, label)
    with open(phn_path, encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 3:
                continue
            start = int(parts[0]) / SAMPLE_RATE
            end   = int(parts[1]) / SAMPLE_RATE
            label = parts[2].lower()
            intervals.append((round(start, 4), round(end, 4), label))

    speech_intervals = [(s, e, l) for s, e, l in intervals if l not in SILENCE_LABELS]
    if not speech_intervals:
        return None, None, []

    utt_start   = speech_intervals[0][0]
    true_eos    = speech_intervals[-1][1]

    # Intra-pauses: silence intervals strictly inside [utt_start, true_eos]
    intra_pauses = [
        [round(s, 4), round(e, 4)]
        for s, e, l in intervals
        if l in SILENCE_LABELS
        and s > utt_start
        and e < true_eos
        and (e - s) >= min_pause_dur
    ]

    return "", true_eos, intra_pauses


def build_torgo_manifest(
    torgo_dir:    str,
    output_path:  str,
    mic:          str   = "headMic",
    min_pause_dur: float = 0.15,
):
    torgo_root = Path(torgo_dir).resolve()
    out_path   = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    manifest = []
    skipped  = 0

    speaker_dirs = sorted(
        d for d in torgo_root.iterdir()
        if d.is_dir() and d.name in SPEAKER_META
    )

    if not speaker_dirs:
        raise FileNotFoundError(
            f"No recognised TORGO speaker directories found under {torgo_root}. "
            f"Expected: {list(SPEAKER_META.keys())}"
        )

    for speaker_dir in speaker_dirs:
        speaker_id = speaker_dir.name
        meta       = SPEAKER_META[speaker_id]

        for session_dir in sorted(speaker_dir.glob("Session*")):
            phn_dir    = session_dir / "phn_arrayMic"   # PHN always from arrayMic
            wav_dir    = session_dir / f"wav_{mic}"
            prompt_dir = session_dir / "prompts"

            # The ASR track needs only (audio + transcript); PHN alignment is
            # optional (present for ~30% of TORGO, and missing entirely for
            # several speakers). Iterate over prompts so coverage spans every
            # speaker and severity tier.
            if not wav_dir.exists() or not prompt_dir.exists():
                continue

            for prompt_file in sorted(prompt_dir.glob("*.txt")):
                utt_id = prompt_file.stem   # e.g. "0030"
                wav_path = wav_dir / f"{utt_id}.wav"
                if not wav_path.exists():
                    wav_path = wav_dir / f"{utt_id}.WAV"
                if not wav_path.exists():
                    skipped += 1
                    continue

                raw_prompt = prompt_file.read_text(encoding="utf-8",
                                                   errors="replace").strip()

                # Skip motor-task prompts (whole prompt in square brackets)
                if not raw_prompt or _BRACKET_RE.match(raw_prompt):
                    skipped += 1
                    continue

                # Clean inline instructions / drop unintelligible prompts
                transcript = clean_transcript(raw_prompt)
                if transcript is None:
                    skipped += 1
                    continue

                # Phone alignment is optional — used only by the endpointing
                # track. Carry it when present, leave null otherwise.
                phn_file = phn_dir / f"{utt_id}.PHN"
                if phn_file.exists():
                    _, true_eos, intra_pauses = parse_phn(phn_file, min_pause_dur)
                else:
                    true_eos, intra_pauses = None, []

                item_id = f"{speaker_id}_{session_dir.name}_{utt_id}"
                manifest.append({
                    "id":           item_id,
                    "audio_path":   str(wav_path),
                    "transcript":   transcript,
                    "true_eos_sec": true_eos,
                    "intra_pauses": intra_pauses,
                    "condition":    meta["condition"],
                    "severity":     meta["severity"],
                    "speaker_id":   speaker_id,
                    "is_control":   meta["is_control"],
                })

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    dysarthric = sum(1 for it in manifest if not it["is_control"])
    controls   = sum(1 for it in manifest if it["is_control"])
    print(f"Wrote {len(manifest)} items to {out_path}")
    print(f"  dysarthric: {dysarthric}  controls: {controls}  skipped: {skipped}")

    # Severity breakdown
    from collections import Counter
    sev = Counter(it["severity"] for it in manifest if not it["is_control"])
    print(f"  severity breakdown (dysarthric): {dict(sev)}")

    return manifest


def main():
    parser = argparse.ArgumentParser(
        description="Build neuro_bench manifest directly from TORGO corpus."
    )
    parser.add_argument("--torgo_dir",   required=True,
                        help="Root TORGO directory (contains F01/, F03/, FC01/, ... subdirs)")
    parser.add_argument("--output",      required=True,
                        help="Path to write the manifest JSON")
    parser.add_argument("--mic",         default="headMic",
                        choices=["headMic", "arrayMic"],
                        help="Microphone to use for audio (default: headMic)")
    parser.add_argument("--min_pause",   type=float, default=0.15,
                        help="Minimum intra-utterance silence to annotate in seconds "
                             "(default: 0.15)")
    args = parser.parse_args()

    build_torgo_manifest(
        torgo_dir    = args.torgo_dir,
        output_path  = args.output,
        mic          = args.mic,
        min_pause_dur = args.min_pause,
    )


if __name__ == "__main__":
    main()
