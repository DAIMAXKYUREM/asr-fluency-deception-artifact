"""
asr_systems.py — Multi-system ASR backends for the neuro_bench ASR track.

Three open-weight systems (no commercial APIs, per the project constraint):

    whisper_large_v3   OpenAI Whisper large-v3 via faster-whisper   (primary)
    wav2vec2_960h      facebook/wav2vec2-large-960h-lv60-self       (academic baseline)
    parakeet_tdt       nvidia/parakeet-tdt-0.6b-v2 via NeMo         (non-Whisper open model)

Every backend exposes the same interface:

    backend = build_asr_system(name, device="cuda")
    out     = backend.transcribe(waveform_float32_16k)   # -> {"text": str, "words": [...]}
    backend.unload()                                     # free GPU memory before the next system

`words` is a list of {"word","start","end"} when the backend provides timing
(Whisper does; the CTC/TDT models return [] — Layer-1 metrics do not need it).

Systems are loaded one at a time so a 4 GB GPU never holds two models at once.
"""

from __future__ import annotations

import gc
from pathlib import Path
from typing import Protocol

import numpy as np
import soundfile as sf
import soxr

SAMPLE_RATE = 16_000

# Locally-converted weights live here (see download_models.py). Used to work
# around the torch<2.6 ban on loading .bin checkpoints: we convert the cached
# wav2vec2 .bin to safetensors once and load that.
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

# Canonical system ids -> human-readable label + underlying checkpoint
SYSTEM_REGISTRY: dict[str, dict[str, str]] = {
    "whisper_large_v3": {
        "label":      "Whisper large-v3",
        "checkpoint": "large-v3",
        "kind":       "whisper",
    },
    "wav2vec2_960h": {
        "label":      "wav2vec2-large-960h-lv60-self",
        "checkpoint": "facebook/wav2vec2-large-960h-lv60-self",
        "kind":       "wav2vec2",
    },
    "parakeet_tdt": {
        "label":      "Parakeet-TDT-0.6b-v2 (ONNX)",
        "checkpoint": "istupakov/parakeet-tdt-0.6b-v2-onnx",
        "kind":       "parakeet",
    },
}

DEFAULT_SYSTEMS = ["whisper_large_v3", "wav2vec2_960h", "parakeet_tdt"]


# ---------------------------------------------------------------------------
# Shared audio loader
# ---------------------------------------------------------------------------

def load_audio(path: str) -> np.ndarray:
    """Load any WAV to 16 kHz mono float32."""
    waveform, sr = sf.read(path, dtype="float32", always_2d=False)
    if waveform.ndim == 2:
        waveform = waveform.mean(axis=1)
    if sr != SAMPLE_RATE:
        waveform = soxr.resample(waveform, sr, SAMPLE_RATE)
    return np.ascontiguousarray(waveform, dtype=np.float32)


class ASRBackend(Protocol):
    label: str
    def transcribe(self, audio: np.ndarray) -> dict: ...
    def unload(self) -> None: ...


# ---------------------------------------------------------------------------
# 1. Whisper large-v3  (faster-whisper)
# ---------------------------------------------------------------------------

class WhisperBackend:
    def __init__(self, checkpoint: str = "large-v3", device: str = "cuda",
                 compute_type: str | None = None, beam_size: int = 5):
        from faster_whisper import WhisperModel
        if compute_type is None:
            # int8_float16 keeps large-v3 inside a 4 GB GPU; int8 on CPU.
            compute_type = "int8_float16" if device == "cuda" else "int8"
        self.label        = SYSTEM_REGISTRY["whisper_large_v3"]["label"]
        self.beam_size    = beam_size
        self.device       = device
        self.model        = WhisperModel(checkpoint, device=device, compute_type=compute_type)

    def transcribe(self, audio: np.ndarray) -> dict:
        segments, _ = self.model.transcribe(
            audio,
            language="en",
            beam_size=self.beam_size,
            word_timestamps=True,
            condition_on_previous_text=False,     # no hallucination carry-over
            no_speech_threshold=0.6,
            log_prob_threshold=-1.0,
            hallucination_silence_threshold=2.0,
        )
        words, parts = [], []
        for seg in segments:
            for w in (seg.words or []):
                words.append({"word": w.word.strip(),
                              "start": round(w.start, 4),
                              "end":   round(w.end, 4)})
            parts.append(seg.text.strip())
        return {"text": " ".join(parts).strip(), "words": words}

    def unload(self) -> None:
        del self.model
        _free_gpu()


# ---------------------------------------------------------------------------
# 2. wav2vec2-large-960h-lv60-self  (HuggingFace transformers, CTC)
# ---------------------------------------------------------------------------

class Wav2Vec2Backend:
    def __init__(self, checkpoint: str = "facebook/wav2vec2-large-960h-lv60-self",
                 device: str = "cuda"):
        import torch
        from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
        self.label  = SYSTEM_REGISTRY["wav2vec2_960h"]["label"]
        self.torch  = torch
        self.device = device if (device == "cuda" and torch.cuda.is_available()) else "cpu"

        # Prefer the locally-converted safetensors copy if present (avoids the
        # torch<2.6 .bin load ban); fall back to the hub id otherwise.
        local = MODELS_DIR / "wav2vec2_960h"
        src   = str(local) if (local / "model.safetensors").exists() else checkpoint

        self.processor = Wav2Vec2Processor.from_pretrained(src)
        self.model     = Wav2Vec2ForCTC.from_pretrained(src).to(self.device).eval()

    def transcribe(self, audio: np.ndarray) -> dict:
        torch = self.torch
        inputs = self.processor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt")
        input_values = inputs.input_values.to(self.device)
        with torch.no_grad():
            logits = self.model(input_values).logits
        pred_ids = torch.argmax(logits, dim=-1)
        text = self.processor.batch_decode(pred_ids)[0].strip()
        return {"text": text, "words": []}   # CTC greedy decode gives no reliable word timing

    def unload(self) -> None:
        del self.model
        del self.processor
        _free_gpu()


# ---------------------------------------------------------------------------
# 3. Parakeet-TDT-0.6b-v2  (NVIDIA, run via ONNX Runtime / onnx-asr)
# ---------------------------------------------------------------------------

class ParakeetBackend:
    """
    NVIDIA Parakeet-TDT-0.6b-v2 through ONNX Runtime (onnx-asr).

    NeMo segfaults on this Windows machine (cuDNN symbol load), so instead of
    the NeMo runtime we use the ONNX export and run it through onnxruntime.
    Weights live locally in models/parakeet_tdt_onnx (real files, not the HF
    symlinked cache, which fails on Windows) — see download_models.py.
    """

    def __init__(self, checkpoint: str = "nemo-parakeet-tdt-0.6b-v2",
                 device: str = "cpu", quantization: str | None = None):
        import onnx_asr
        import onnxruntime as ort
        self.label = SYSTEM_REGISTRY["parakeet_tdt"]["label"]

        local = MODELS_DIR / "parakeet_tdt_onnx"
        if not (local / "encoder-model.onnx").exists():
            raise FileNotFoundError(
                f"ONNX Parakeet weights not found at {local}. "
                "Run: python download_models.py --only parakeet"
            )

        # Use CUDA EP only if this onnxruntime build actually has it.
        avail = ort.get_available_providers()
        wanted = ["CUDAExecutionProvider", "CPUExecutionProvider"] if device == "cuda" \
            else ["CPUExecutionProvider"]
        providers = [p for p in wanted if p in avail] or ["CPUExecutionProvider"]

        self.model = onnx_asr.load_model(
            "nemo-parakeet-tdt-0.6b-v2", path=str(local),
            quantization=quantization, providers=providers,
        )

    def transcribe(self, audio: np.ndarray) -> dict:
        wav = np.ascontiguousarray(audio, dtype=np.float32)
        txt = self.model.recognize(wav, sample_rate=SAMPLE_RATE)
        if isinstance(txt, list):
            txt = txt[0] if txt else ""
        return {"text": (txt or "").strip(), "words": []}

    def unload(self) -> None:
        del self.model
        _free_gpu()


# ---------------------------------------------------------------------------
# Factory + helpers
# ---------------------------------------------------------------------------

def build_asr_system(name: str, device: str = "cuda") -> ASRBackend:
    if name not in SYSTEM_REGISTRY:
        raise ValueError(f"Unknown ASR system '{name}'. "
                         f"Choose from: {list(SYSTEM_REGISTRY)}")
    kind = SYSTEM_REGISTRY[name]["kind"]
    ckpt = SYSTEM_REGISTRY[name]["checkpoint"]
    if kind == "whisper":
        return WhisperBackend(ckpt, device=device)
    if kind == "wav2vec2":
        return Wav2Vec2Backend(ckpt, device=device)
    if kind == "parakeet":
        return ParakeetBackend(ckpt, device=device)
    raise ValueError(f"Unhandled backend kind '{kind}'")


def _free_gpu() -> None:
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
