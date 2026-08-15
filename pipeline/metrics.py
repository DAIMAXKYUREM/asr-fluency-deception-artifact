"""
metrics.py — Three-layer metric suite for neuro_bench.

Layer 1 — Recognition
    WER, CER, Entity Error Rate, Hallucination Rate, Semantic Similarity

Layer 2 — Turn-taking / Endpointing  (names align with Full-Duplex-Bench)
    Premature Endpoint Rate (≡ FDB Takeover Rate inverse)
    False-Endpoint-on-Pause Rate
    Endpoint Margin (signed seconds: negative = cut off early)
    False Trigger Rate (spurious commits per minute)

Layer 3 — Equity / Robustness
    Severity-stratified breakdown for every metric above
    Equity Gap: metric(impaired) − metric(matched control)

All deterministic metrics are computed analytically here.
Soft metrics (intent_recovery, reply_appropriateness) are scored by an
LLM judge in run.py using Ollama.
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any

import numpy as np
import jiwer
from sentence_transformers import SentenceTransformer


# ---------------------------------------------------------------------------
# Sentence encoder (lazy-loaded)
# ---------------------------------------------------------------------------

_encoder: SentenceTransformer | None = None


def _get_encoder() -> SentenceTransformer:
    global _encoder
    if _encoder is None:
        _encoder = SentenceTransformer("all-MiniLM-L6-v2")
    return _encoder


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_WER_TRANSFORM = jiwer.Compose([
    jiwer.ToLowerCase(),
    jiwer.RemovePunctuation(),
    jiwer.Strip(),
    jiwer.ReduceToListOfListOfWords(),
])


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower().strip())


# ---------------------------------------------------------------------------
# Layer 1 — Recognition metrics
# ---------------------------------------------------------------------------

def wer(reference: str, hypothesis: str) -> float:
    if not reference.strip():
        return float("nan")
    measures = jiwer.compute_measures(
        reference, hypothesis,
        truth_transform=_WER_TRANSFORM,
        hypothesis_transform=_WER_TRANSFORM,
    )
    return round(measures["wer"], 4)


def cer(reference: str, hypothesis: str) -> float:
    if not reference.strip():
        return float("nan")
    ref = _normalise(reference)
    hyp = _normalise(hypothesis)
    measures = jiwer.cer(ref, hyp)
    return round(measures, 4)


def entity_error_rate(
    reference: str,
    hypothesis: str,
    entity_list: list[str] | None = None,
) -> float:
    """
    Fraction of target entities in the reference that are wrong in hypothesis.

    entity_list: explicit list of tokens to treat as entities.
    If None, falls back to a built-in healthcare-domain entity regex
    (medication names, times, numeric dosages).
    """
    ref_tokens = set(_normalise(reference).split())
    hyp_tokens = set(_normalise(hypothesis).split())

    if entity_list is not None:
        entities = [e.lower() for e in entity_list]
    else:
        # Fallback: alphanumeric tokens that look like medical entities
        # (numbers, dosage patterns like "10mg", common drug suffixes)
        entities = [
            t for t in ref_tokens
            if re.search(r"\d", t) or t.endswith(("mg", "ml", "mcg", "mg/ml"))
        ]

    if not entities:
        return float("nan")

    errors = sum(1 for e in entities if e not in hyp_tokens)
    return round(errors / len(entities), 4)


def hallucination_rate(
    reference: str,
    hypothesis: str,
    asr_words: list[dict] | None = None,
) -> float:
    """
    Fraction of hypothesis words not present in the reference.

    On quiet or heavily disordered audio, Whisper sometimes invents words
    not in the reference transcript — this captures that failure mode.

    asr_words: word-level ASR output (used to detect insertions on silence
    segments, but not required for the basic rate).
    """
    ref_tokens = set(_normalise(reference).split())
    hyp_tokens = _normalise(hypothesis).split()
    if not hyp_tokens:
        return 0.0
    hallucinated = [w for w in hyp_tokens if w not in ref_tokens]
    return round(len(hallucinated) / len(hyp_tokens), 4)


def semantic_similarity(reference: str, hypothesis: str) -> float:
    """Cosine similarity of sentence embeddings."""
    if not reference.strip() or not hypothesis.strip():
        return float("nan")
    enc = _get_encoder()
    embs = enc.encode([reference, hypothesis], convert_to_numpy=True)
    cos  = float(np.dot(embs[0], embs[1]) /
                 (np.linalg.norm(embs[0]) * np.linalg.norm(embs[1]) + 1e-9))
    return round(cos, 4)


# --- Error-type breakdown (substitutions / deletions / insertions) ----------

def wer_breakdown(reference: str, hypothesis: str) -> dict[str, float]:
    """
    WER decomposed into substitution / deletion / insertion counts and rates,
    all normalised by the number of reference words (guide §3.4).

    Rates sum to the WER: sub_rate + del_rate + ins_rate == wer.
    Returns NaN-filled dict if the reference is empty.
    """
    nan = float("nan")
    if not reference.strip():
        return {"wer": nan, "n_ref_words": 0,
                "substitutions": nan, "deletions": nan, "insertions": nan, "hits": nan,
                "sub_rate": nan, "del_rate": nan, "ins_rate": nan}
    out = jiwer.process_words(
        reference, hypothesis,
        reference_transform=_WER_TRANSFORM,
        hypothesis_transform=_WER_TRANSFORM,
    )
    n_ref = out.substitutions + out.deletions + out.hits
    denom = n_ref if n_ref > 0 else 1
    return {
        "wer":           round(out.wer, 4),
        "n_ref_words":   n_ref,
        "substitutions": out.substitutions,
        "deletions":     out.deletions,
        "insertions":    out.insertions,
        "hits":          out.hits,
        "sub_rate":      round(out.substitutions / denom, 4),
        "del_rate":      round(out.deletions     / denom, 4),
        "ins_rate":      round(out.insertions    / denom, 4),
    }


# --- Content-word vs function-word WER (guide §3.4) -------------------------

# Closed-class English function words: articles, pronouns, prepositions,
# conjunctions, auxiliaries, particles, common determiners/quantifiers.
FUNCTION_WORDS: frozenset[str] = frozenset("""
a an the this that these those some any each every all both few many several no
i me my mine myself you your yours yourself he him his himself she her hers herself
it its itself we us our ours ourselves they them their theirs themselves who whom
whose which what whoever whatever
am is are was were be been being do does did doing have has had having
will would shall should can could may might must ought need dare
and or but nor so yet for because although though while whereas if unless until
since as than that whether
of in on at by to from with about against between into through during before after
above below up down out off over under again further then once here there
when where why how
not no nor only just also too very
i'm you're he's she's it's we're they're i've you've we've they've
don't doesn't didn't won't wouldn't can't couldn't shouldn't isn't aren't wasn't
weren't haven't hasn't hadn't
""".split())


def _classify_tokens(tokens: list[str]) -> list[bool]:
    """True = function word, False = content word, per token."""
    return [t in FUNCTION_WORDS for t in tokens]


def content_function_wer(reference: str, hypothesis: str) -> dict[str, float]:
    """
    WER computed separately over content words and function words in the
    reference, using the word-level alignment.

    Definition: for each reference-word class C,
        WER_C = (substitutions + deletions on class-C reference words) / |class-C ref words|
    Insertions have no reference word, so they are not attributed to a class
    (reported globally by wer_breakdown). This is the window choice documented
    in the framework section.
    """
    nan = float("nan")
    empty = {"content_word_wer": nan, "function_word_wer": nan,
             "n_content_words": 0, "n_function_words": 0,
             "content_errors": 0, "function_errors": 0}
    if not reference.strip():
        return dict(empty)

    out = jiwer.process_words(
        reference, hypothesis,
        reference_transform=_WER_TRANSFORM,
        hypothesis_transform=_WER_TRANSFORM,
    )
    if not out.references or not out.references[0]:
        return dict(empty)

    ref_tokens = out.references[0]
    is_func    = _classify_tokens(ref_tokens)
    n_func     = sum(is_func)
    n_content  = len(ref_tokens) - n_func

    err_func = err_content = 0
    for chunk in out.alignments[0]:
        if chunk.type in ("substitute", "delete"):
            for idx in range(chunk.ref_start_idx, chunk.ref_end_idx):
                if is_func[idx]:
                    err_func += 1
                else:
                    err_content += 1

    return {
        "content_word_wer":  round(err_content / n_content, 4) if n_content else nan,
        "function_word_wer": round(err_func    / n_func,    4) if n_func    else nan,
        "n_content_words":   n_content,
        "n_function_words":  n_func,
        "content_errors":    err_content,
        "function_errors":   err_func,
    }


# --- Character-level counts (for micro-averaged CER) ------------------------

def cer_counts(reference: str, hypothesis: str) -> dict[str, int]:
    """Char-level edit counts: (errors, ref_chars) for micro-averaged CER."""
    if not reference.strip():
        return {"cer_errors": 0, "cer_ref_chars": 0}
    ref = _normalise(reference)
    hyp = _normalise(hypothesis)
    out = jiwer.process_characters(ref, hyp)
    return {
        "cer_errors":    out.substitutions + out.deletions + out.insertions,
        "cer_ref_chars": out.substitutions + out.deletions + out.hits,
    }


def compute_recognition_metrics(
    item: dict,
    asr_result: dict,
    entity_list: list[str] | None = None,
) -> dict[str, float]:
    ref = item["transcript"]
    hyp = asr_result.get("text", "")
    metrics = {
        "wer":                  wer(ref, hyp),
        "cer":                  cer(ref, hyp),
        "entity_error_rate":    entity_error_rate(ref, hyp, entity_list),
        "hallucination_rate":   hallucination_rate(ref, hyp, asr_result.get("words")),
        "semantic_similarity":  semantic_similarity(ref, hyp),
    }
    # Error-type breakdown (S/D/I) + content/function split
    bd = wer_breakdown(ref, hyp)
    metrics.update({
        "sub_rate":          bd["sub_rate"],
        "del_rate":          bd["del_rate"],
        "ins_rate":          bd["ins_rate"],
        "substitutions":     bd["substitutions"],
        "deletions":         bd["deletions"],
        "insertions":        bd["insertions"],
        "n_ref_words":       bd["n_ref_words"],
    })
    cf = content_function_wer(ref, hyp)
    metrics.update({
        "content_word_wer":  cf["content_word_wer"],
        "function_word_wer": cf["function_word_wer"],
        "n_content_words":   cf["n_content_words"],
        "n_function_words":  cf["n_function_words"],
        "content_errors":    cf["content_errors"],
        "function_errors":   cf["function_errors"],
    })
    # Char-level counts (micro-averaged CER) and hallucination counts
    metrics.update(cer_counts(ref, hyp))
    hyp_tokens = _normalise(hyp).split()
    ref_set    = set(_normalise(ref).split())
    metrics["hyp_word_count"] = len(hyp_tokens)
    metrics["halluc_count"]   = sum(1 for w in hyp_tokens if w not in ref_set)
    return metrics


# ---------------------------------------------------------------------------
# Layer 2 — Endpointing / Turn-taking metrics
# ---------------------------------------------------------------------------

def endpoint_metrics_for_item(
    item: dict,
    endpoint_time: float | None,
) -> dict[str, float]:
    """
    Compute per-item endpointing metrics for a single detector result.

    item must have: true_eos_sec, intra_pauses
    endpoint_time: seconds at which the detector committed, or None.

    Returns:
        premature_endpoint  — 1 if endpoint_time < true_eos_sec else 0 (NaN if None)
        endpoint_margin     — endpoint_time - true_eos_sec  (negative = cut off early)
        fired_on_pause      — 1 if endpoint_time falls inside an intra-pause else 0
    """
    true_eos      = item["true_eos_sec"]
    intra_pauses  = item.get("intra_pauses", [])

    if endpoint_time is None:
        # Detector never committed — conservative but not a truncation error
        return {
            "premature_endpoint": float("nan"),
            "endpoint_margin":    float("nan"),
            "fired_on_pause":     0.0,
        }

    premature = 1.0 if endpoint_time < true_eos else 0.0
    margin    = round(endpoint_time - true_eos, 4)

    fired_on_pause = 0.0
    for (ps, pe) in intra_pauses:
        if ps <= endpoint_time <= pe:
            fired_on_pause = 1.0
            break

    return {
        "premature_endpoint": premature,
        "endpoint_margin":    margin,
        "fired_on_pause":     fired_on_pause,
    }


def aggregate_endpoint_metrics(
    items: list[dict],
    endpoint_times: dict[str, float | None],
) -> dict[str, float]:
    """
    Aggregate per-item endpoint metrics into rates and means.

    endpoint_times: {item_id -> endpoint_time_sec or None}

    Returns:
        premature_endpoint_rate     — fraction of turns cut off early
        false_endpoint_on_pause_rate — fraction of intra-pauses that triggered commit
        endpoint_margin_mean         — mean signed margin (sec)
        false_trigger_rate           — spurious commits per minute on silence/control items
    """
    premature_flags   = []
    margins           = []
    pause_fired_count = 0
    total_pauses      = 0

    for item in items:
        iid = item["id"]
        et  = endpoint_times.get(iid)
        res = endpoint_metrics_for_item(item, et)

        if not np.isnan(res["premature_endpoint"]):
            premature_flags.append(res["premature_endpoint"])

        if not np.isnan(res["endpoint_margin"]):
            margins.append(res["endpoint_margin"])

        # False-endpoint-on-pause: count events across all intra-pauses
        for (ps, pe) in item.get("intra_pauses", []):
            total_pauses += 1
            if et is not None and ps <= et <= pe:
                pause_fired_count += 1

    # False trigger rate: spurious commits per minute on control items
    # (items where is_control=True and endpoint_time < true_eos)
    control_items   = [it for it in items if it.get("is_control", False)]
    spurious_count  = 0
    control_dur_min = 0.0
    for it in control_items:
        et = endpoint_times.get(it["id"])
        if et is not None and et < it["true_eos_sec"]:
            spurious_count += 1
        control_dur_min += it["true_eos_sec"] / 60.0

    false_trigger_rate = (
        spurious_count / control_dur_min if control_dur_min > 0 else float("nan")
    )

    return {
        "premature_endpoint_rate":      _nanmean(premature_flags),
        "false_endpoint_on_pause_rate": (
            pause_fired_count / total_pauses if total_pauses > 0 else float("nan")
        ),
        "endpoint_margin_mean":         _nanmean(margins),
        "false_trigger_rate_per_min":   round(false_trigger_rate, 4),
    }


def _nanmean(values: list[float]) -> float:
    clean = [v for v in values if not np.isnan(v)]
    return round(float(np.mean(clean)), 4) if clean else float("nan")


# ---------------------------------------------------------------------------
# Layer 3 — Equity / severity analysis
# ---------------------------------------------------------------------------

SEVERITY_ORDER = ["very_mild", "moderate", "moderate_severe", "severe", "unknown"]


def stratify_by_severity(
    items: list[dict],
    per_item_results: list[dict],
) -> dict[str, list[dict]]:
    """Group (item, result) pairs by severity bin."""
    bins: dict[str, list[dict]] = defaultdict(list)
    for item, result in zip(items, per_item_results):
        sev = item.get("severity", "unknown").lower()
        if sev not in SEVERITY_ORDER:
            sev = "unknown"
        bins[sev].append(result)
    return dict(bins)


def equity_gap(
    impaired_items: list[dict],
    control_items:  list[dict],
    impaired_results: list[dict],
    control_results:  list[dict],
    metric_keys: list[str],
) -> dict[str, float]:
    """
    Compute equity gap = metric(impaired) - metric(control) for each key.
    Positive gap = impaired speakers fare worse.
    """
    def mean_for(results: list[dict], key: str) -> float:
        vals = [r[key] for r in results if key in r and not np.isnan(r[key])]
        return float(np.mean(vals)) if vals else float("nan")

    gaps = {}
    for key in metric_keys:
        imp_val  = mean_for(impaired_results, key)
        ctrl_val = mean_for(control_results,  key)
        gaps[f"equity_gap_{key}"] = (
            round(imp_val - ctrl_val, 4)
            if not (np.isnan(imp_val) or np.isnan(ctrl_val))
            else float("nan")
        )
    return gaps


def build_equity_table(
    items: list[dict],
    per_item_results: list[dict],
    metric_keys: list[str],
) -> dict[str, Any]:
    """
    Full equity analysis:
      - aggregate metrics for impaired vs control
      - equity gap per metric
      - severity-stratified breakdowns for impaired speakers
    """
    impaired_pairs = [
        (it, res)
        for it, res in zip(items, per_item_results)
        if not it.get("is_control", False)
    ]
    control_pairs = [
        (it, res)
        for it, res in zip(items, per_item_results)
        if it.get("is_control", False)
    ]

    imp_items,  imp_results  = (
        ([p[0] for p in impaired_pairs], [p[1] for p in impaired_pairs])
        if impaired_pairs else ([], [])
    )
    ctrl_items, ctrl_results = (
        ([p[0] for p in control_pairs], [p[1] for p in control_pairs])
        if control_pairs else ([], [])
    )

    gaps = equity_gap(imp_items, ctrl_items, imp_results, ctrl_results, metric_keys)

    # Severity breakdown for impaired speakers
    sev_bins: dict[str, list[dict]] = defaultdict(list)
    for it, res in zip(imp_items, imp_results):
        sev = it.get("severity", "unknown")
        sev_bins[sev].append(res)

    severity_breakdown = {}
    for sev in SEVERITY_ORDER:
        bin_results = sev_bins.get(sev, [])
        if not bin_results:
            continue
        severity_breakdown[sev] = {
            k: _nanmean([r[k] for r in bin_results if k in r])
            for k in metric_keys
        }

    return {
        "equity_gaps":          gaps,
        "severity_breakdown":   severity_breakdown,
        "n_impaired":           len(imp_items),
        "n_control":            len(ctrl_items),
    }
