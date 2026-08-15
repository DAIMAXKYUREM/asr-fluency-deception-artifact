# MDVR-KCL (Parkinson's) — deception metrics

- 37 recordings/system. phi>= 0.5 (fluent), s<= 0.5 (wrong).
- Read-passage corpus: per-recording (coarser than TORGO's per-utterance); consistency/cost omitted (no repeated prompts).

## FMR / deception — overall, control vs PD

| system | scope | FMR | mean D | fluency-given-wrong | degenerate |
|---|---|---|---|---|---|
| whisper_large_v3 | overall | 0.0541 | 0.0845 | 0.3712 | 0.0811 |
| whisper_large_v3 | control | 0.0476 | 0.0659 | 0.6222 | 0.0476 |
| whisper_large_v3 | PD | 0.0625 | 0.109 | 0.2878 | 0.125 |
| wav2vec2_960h | overall | 0.0 | 0.1054 | 0.2643 | 0.0 |
| wav2vec2_960h | control | 0.0 | 0.0857 | nan | 0.0 |
| wav2vec2_960h | PD | 0.0 | 0.1312 | 0.2643 | 0.0 |
| parakeet_tdt | overall | 0.027 | 0.0829 | 0.2877 | 0.0541 |
| parakeet_tdt | control | 0.0 | 0.0662 | 0.4708 | 0.0476 |
| parakeet_tdt | PD | 0.0625 | 0.1047 | 0.1763 | 0.0625 |

## By UPDRS severity (mean D / FMR)

| system | control | none | mild | moderate | severe |
|---|---|---|---|---|---|
| whisper_large_v3 | D=0.0659 FMR=0.0476 | D=0.0382 FMR=0.0 | D=0.1262 FMR=0.1429 | D=0.2165 FMR=0.0 | D=0.0198 FMR=0.0 |
| wav2vec2_960h | D=0.0857 FMR=0.0 | D=0.0664 FMR=0.0 | D=0.1539 FMR=0.0 | D=0.2229 FMR=0.0 | D=0.0203 FMR=0.0 |
| parakeet_tdt | D=0.0662 FMR=0.0 | D=0.0601 FMR=0.0 | D=0.1423 FMR=0.1429 | D=0.1238 FMR=0.0 | D=0.0068 FMR=0.0 |

_Deception is coarser here than on TORGO (whole read passages, not utterances); read as: does the fluent-but-wrong pattern also appear in Parkinson's speech._
