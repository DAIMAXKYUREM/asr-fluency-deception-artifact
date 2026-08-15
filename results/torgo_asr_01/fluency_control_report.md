# Fluency control analysis -- CODS 2026 Section 5.3

## Task 1 -- reproduction check

Subset: impaired speakers, sentence prompts. Metric: median phi | s<=0.5.

| system | computed | published | match (<0.001) |
|---|---|---|---|
| whisper_large_v3 | 0.6324 | 0.632 | MATCH |
| wav2vec2_960h | 0.5357 | 0.536 | MATCH |
| parakeet_tdt | 0.5223 | 0.522 | MATCH |

Reproduction PASSED -- all three values match to <0.001. Proceeding to Task 2.

## Task 2 -- the control

| system | phi_correct (s>0.5) | phi_wrong (s<=0.5) | delta_phi |
|---|---|---|---|
| whisper_large_v3 | 0.6484 | 0.6324 | -0.0160 |
| wav2vec2_960h | 0.6479 | 0.5357 | -0.1121 |
| parakeet_tdt | 0.6399 | 0.5223 | -0.1176 |

delta_phi is largest for: **Whisper** (-0.0160).
BRANCH: A (Whisper largest -- existing prose is correct)

### Mixed-effects model: phi ~ C(system, Treatment('wav2vec2_960h')) * wrong + (1|speaker)

Full coefficient table:

| term | coef | p-value |
|---|---|---|
| Intercept | +0.6410 | 0.000e+00 |
| C(system, Treatment('wav2vec2_960h'))[T.parakeet_tdt] | -0.0187 | 7.157e-03 |
| C(system, Treatment('wav2vec2_960h'))[T.whisper_large_v3] | +0.0017 | 8.087e-01 |
| wrong | -0.1181 | 3.179e-41 |
| C(system, Treatment('wav2vec2_960h'))[T.parakeet_tdt]:wrong | -0.0164 | 1.815e-01 |
| C(system, Treatment('wav2vec2_960h'))[T.whisper_large_v3]:wrong | +0.0753 | 3.474e-09 |
| Group Var | +0.0264 | 7.611e-02 |

**Whisper x wrong interaction: +0.0753 (p = 3.474e-09)**

### Cluster bootstrap (B=2000, resample speakers w/ replacement, same procedure as the Section 5.2 CI) on delta_phi(Whisper) - delta_phi(wav2vec2)

Whisper-minus-wav2vec2 delta_phi difference: +0.0931 [+0.0560, +0.1250] (95% CI).
**Excludes zero: True.**

## Summary

BRANCH A: Whisper has the largest delta_phi -- existing Section 5.3 prose is correct as written.