# neuro_bench — Inferential statistics (ASR track)

Speaker as random intercept; cluster bootstrap resamples speakers (B=2000).

## 1. Mixed model: severity cliff is real

`wer ~ sev_ord + (1|speaker)` on impaired sentences, per system:

| system | severity slope (WER per tier) | p-value | n_items | n_speakers |
|---|---|---|---|---|
| whisper_large_v3 | +0.190 | 1.84e-12 | 692 | 8 |
| wav2vec2_960h | +0.220 | 6.26e-46 | 692 | 8 |
| parakeet_tdt | +0.190 | 3.75e-23 | 692 | 8 |

## 2. Mixed model: Whisper is significantly more deceptive when wrong

`phi ~ C(system) + (1|speaker)` on **wrong** impaired sentences (s<=0.5); reference = wav2vec2:

| term | coef (phi vs wav2vec2) | p-value |
|---|---|---|
| Intercept | +0.528 | 2.05e-162 |
| C(system)[T.whisper_large_v3] | +0.073 | 9.43e-07 |
| C(system)[T.parakeet_tdt] | -0.039 | 5.48e-03 |

_n=601 wrong items, 8 speakers. A positive Whisper coefficient with p<0.05 = Whisper's errors are significantly more fluent than wav2vec2's._

## 3. Bootstrap 95% CIs on headline numbers (resampling speakers)

### Equity gap (impaired - control) WER, sentences (micro)

| system | gap [95% CI] |
|---|---|
| whisper_large_v3 | 0.268 [0.127, 0.446] |
| wav2vec2_960h | 0.387 [0.206, 0.586] |
| parakeet_tdt | 0.296 [0.135, 0.477] |

### FMR — fluent misrecognition rate, impaired sentences

| system | FMR [95% CI] |
|---|---|
| whisper_large_v3 | 0.189 [0.071, 0.335] |
| wav2vec2_960h | 0.264 [0.135, 0.402] |
| parakeet_tdt | 0.148 [0.057, 0.253] |

### Fluency-given-wrong (median phi) + the key contrast

| comparison | value [95% CI] |
|---|---|
| whisper_large_v3 median phi-on-wrong | 0.632 [0.614, 0.648] |
| wav2vec2_960h median phi-on-wrong | 0.536 [0.523, 0.555] |
| parakeet_tdt median phi-on-wrong | 0.524 [0.499, 0.559] |
| **Whisper - wav2vec2 (the 0.63 vs 0.54)** | **0.096 [0.057, 0.122]** |

_If this difference's CI excludes 0, the deception ranking is statistically supported._
