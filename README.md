# Reproducibility artifact — "Fluent but Wrong: Deceptive ASR Errors Grow with Model Quality and Speaker Impairment" (CODS 2026)

This directory reproduces every number, table, and figure in the paper from
the included **per-utterance CSVs**, without needing to re-run ASR inference,
download model checkpoints, or obtain the TORGO / MDVR-KCL audio corpora --
except where noted under "Scope" below.

Author/institution identifiers, emails, and absolute local paths have been
stripped from every file in this directory (see "Anonymisation" below).

## Directory layout

```
pipeline/     all analysis / regeneration scripts (flat, mirrors the original repo's pipeline/)
align/        torgo_to_manifest.py -- documents how FDA severity tiers are assigned (SPEAKER_META)
results/
  torgo_asr_01/
    per_item.csv                       one row per (TORGO utterance x system); recognition metrics + CORRECT FDA severity
    per_item_novel.csv                 same rows, + phi / s / D / degenerate (its OWN `severity` column is stale -- see note below)
    conf/*.jsonl                       per-utterance model confidence (Whisper/wav2vec2/Parakeet), for the detector table
    deception_detector_torgo_scores.csv   phi/conf/D_hat/oracle_D per utterance -- backs Table 8 (detector)
    fluency_control_results.csv        backs Section 5.3 / Table "tab:control"
    lm_robustness_results.csv          backs the GPT-Neo-125M footnote in Section 5.2
    stats_report.md                    mixed-effects models + bootstrap CIs (severity cliff, Whisper deception)
    threshold_sweep.md                 backs Table "tab:sweep"
  mdvrkcl_asr_01/
    per_item.csv, per_item_novel.csv   one row per (MDVR-KCL recording x system)
  qualitative/
    fmr_candidates.csv                 the 20 stratified FMR-positive candidates Table "tab:examples" was hand-picked from
  paper_figures_sigconf/
    fig_severity_cliff.pdf, fig_per_error.pdf, fig_cross_corpus.pdf   the exact PDFs used in the paper
requirements.txt
```

## Paper element -> script -> command

All commands assume the working directory is this artifact's root (the
directory containing `pipeline/`, `align/`, `results/`). Every script computes
its own paths relative to its own file location, so it does not matter
whether you `cd pipeline` first or invoke `python pipeline/script.py` from
here -- both resolve to the same `results/` directory.

| Paper element | Script | Command | Needs beyond the bundled CSVs? |
|---|---|---|---|
| Table "tab:overall", Table "tab:equity" (recognition, equity gap) | `run_asr.py` | *(already computed -- see Scope)* | raw TORGO audio + downloaded ASR checkpoints |
| Fig. 1 `fig_severity_cliff.pdf` | `make_figures_sigconf.py` | `python pipeline/make_figures_sigconf.py --copy_to ""` | none |
| Table "tab:deception" (FMR / D / fluency-given-wrong), Table "tab:cost" | `novel_metrics.py` | `python pipeline/novel_metrics.py --run_dir results/torgo_asr_01` | Table "tab:cost" throughput numbers need the raw TORGO audio (for per-utterance duration); everything else in this table does not |
| Fig. 2 `fig_per_error.pdf` | `make_figures_sigconf.py` | *(same command as Fig. 1 -- both are written in one run)* | none |
| Table "tab:sweep" (threshold sensitivity) | `threshold_sweep.py` | `python pipeline/threshold_sweep.py` | none |
| Section 5.1 stock-phrase artifact audit (the "1.6 percent" paragraph) | `artifact_audit_torgo.py` | `python pipeline/artifact_audit_torgo.py` | none |
| Table "tab:examples" (qualitative fluent-misrecognition examples) | `build_qualitative_table.py` | `python pipeline/build_qualitative_table.py --run_dir results/torgo_asr_01 --scores_file results/torgo_asr_01/deception_detector_torgo_scores.csv --out_dir results/qualitative` | none |
| Section 5.2 mixed-effects models + bootstrap CIs (severity cliff significance, Whisper-deception significance) | `stats_analysis.py` | `python pipeline/stats_analysis.py` | none |
| Section 5.2 footnote (GPT-Neo-125M robustness) | `lm_robustness_check.py` | `python pipeline/lm_robustness_check.py` | one-time download of GPT-Neo-125M (~500MB) from Hugging Face |
| Section 5.3 / Table "tab:control" (the language-prior control) | `fluency_control_analysis.py` | `python pipeline/fluency_control_analysis.py` | none |
| Fig. 3 `fig_cross_corpus.pdf` | `make_figures_sigconf.py` | *(same command as Fig. 1)* | none |
| Table "tab:detector" (AURC comparison) | `deception_detector.py` | `python pipeline/deception_detector.py --run_dir results/torgo_asr_01 --corpus torgo` | none |

Cut from the ACM version as redundant with tables (not regenerated here, per
the paper's own figure list): the failure-mode bar chart and the AURC bar
chart.

## Scope of what is and is not reproducible from this artifact alone

**Fully reproducible from the bundled CSVs, no external data or models
needed:** every deception metric (phi, s, D, FMR, fluency-given-wrong), the
threshold sweep, the stock-phrase artifact audit, the language-prior control
(Section 5.3), the mixed-effects models and bootstrap CIs, the LM-choice
robustness check, the detector AURC comparison, and all three regenerated
figures.

**Requires data not included here** (per the paper's own Reproducibility
statement -- these are third-party corpora obtained under their own
data-use agreements, and are not ours to redistribute):
- Re-running `run_asr.py` / `run_asr_mdvrkcl.py` from scratch requires the
  raw TORGO (`cs.toronto.edu/~complingweb/data/TORGO`) and MDVR-KCL
  (Zenodo DOI 10.5281/zenodo.2867215) audio, plus the three ASR checkpoints
  (Whisper large-v3, wav2vec2-large-960h-lv60-self, Parakeet-TDT-0.6b-v2,
  all public). `results/*/per_item.csv` already contain their output.
- `novel_metrics.py`'s Cost-to-be-Understood section (Table "tab:cost")
  reads per-utterance audio duration and needs the raw TORGO audio; it
  degrades gracefully (NaN, not a crash) without it, and every other number
  `novel_metrics.py` produces does not depend on audio at all.
- `align/torgo_to_manifest.py` is included to document how FDA severity
  tiers are assigned to each speaker (the source of truth the rest of the
  pipeline relies on); it is not standalone-runnable without the raw
  corpus directory tree.

**A data-provenance note carried over from the paper's own analysis:**
`per_item_novel.csv`'s own `severity` column predates a correction and is
stale for 5 of 8 impaired TORGO speakers. Every script in this artifact that
needs severity reads it from `per_item.csv` instead (or, for
`make_figures_sigconf.py`, reconstructs it explicitly from `is_control`),
which matches `align/torgo_to_manifest.py::SPEAKER_META` exactly. This is
noted directly in the scripts' own comments (see `make_figures_sigconf.py`
and `fluency_control_analysis.py`).

## Windows console note

Every script in `pipeline/` has been run end-to-end from a fresh copy of
this exact directory to confirm it reproduces the paper's numbers (e.g.
`fluency_control_analysis.py` reproduces +0.093 [0.056, 0.125], and
`artifact_audit_torgo.py` reproduces "1.6 percent", matching the paper
text exactly). On a default Windows console, several scripts' final
`print()` (not their file output, which completes first) can raise
`UnicodeEncodeError` on the `phi`/`varphi` character. This is a console
encoding issue, not a correctness issue -- the `.md`/`.csv` outputs are
written before the crash. Run with `PYTHONIOENCODING=utf-8` set (or on
Linux/macOS, where this does not occur) to see the printed summary as well.

## Anonymisation

The following were checked (via full-text search across every file in this
directory) and are absent: author names, email addresses, institution
strings, and absolute local filesystem paths (e.g. `C:\Users\...`). All
scripts locate their data via paths relative to their own file location
(`Path(__file__).resolve().parent...`), computed at runtime, never
hardcoded. `main.tex` is not included in this directory; its ACM `anonymous`
class option already suppresses the author block.

## Requirements

See `requirements.txt`. All analysis scripts run on CPU; only the original
(not-included-here) ASR inference benefits from a GPU.
