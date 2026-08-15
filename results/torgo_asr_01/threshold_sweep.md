# Deception ranking — threshold sensitivity

Impaired sentences. Metric = conditional deception rate P(φ≥τ_φ | s≤τ_s) (fraction of wrong outputs that read fluent).

**Whisper ranks #1 in 15/25 grid cells.** Whisper's margin over the best competitor is mixed.

## Median φ-given-wrong vs s-threshold (τ_φ-independent headline)

| τ_s | Whisper | wav2vec2 | Parakeet | top |
|---|---|---|---|---|
| 0.3 | 0.640 | 0.510 | 0.510 | Whisper |
| 0.4 | 0.632 | 0.519 | 0.510 | Whisper |
| 0.5 | 0.632 | 0.536 | 0.522 | Whisper |
| 0.6 | 0.630 | 0.553 | 0.530 | Whisper |
| 0.7 | 0.627 | 0.560 | 0.536 | Whisper |

## Whisper margin (cond. deception rate − best competitor), τ_φ × τ_s

| τ_φ \ τ_s | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 |
|---|---|---|---|---|---|
| **0.3** | -0.027 | -0.014 | -0.013 | -0.012 | -0.005 |
| **0.4** | -0.023 | -0.003 | -0.012 | -0.018 | -0.012 |
| **0.5** | +0.313 | +0.301 | +0.236 | +0.190 | +0.153 |
| **0.6** | +0.390 | +0.394 | +0.388 | +0.368 | +0.319 |
| **0.7** | +0.252 | +0.209 | +0.186 | +0.176 | +0.171 |
