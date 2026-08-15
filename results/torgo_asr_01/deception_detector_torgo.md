# Inference-time deception detector — TORGO

Joined 23891 utterances; confidence = **min_conf** (z-scored per model).
Label = wrong (s<=0.5); base rate = 0.317.

Detectors: `phi_only` (fluency), `conf_only` (acoustic uncertainty, the baseline), `D_hat = phi*(1-conf)` (proposed), `oracle_D = phi*(1-s)` (upper bound, uses s).

## ROC-AUC / PR-AUC vs label (threshold-free; higher = better ranking of danger)

| scope | base | phi_only | conf_only | **D_hat** | oracle_D |
|---|---|---|---|---|---|
| whisper_large_v3 | 0.272 | 0.561/0.336 | 0.777/0.565 | 0.751/0.561 | 0.989/0.981 |
| wav2vec2_960h | 0.371 | 0.320/0.273 | 0.755/0.657 | 0.526/0.394 | 0.954/0.932 |
| parakeet_tdt | 0.310 | 0.414/0.304 | 0.788/0.702 | 0.530/0.433 | 0.858/0.851 |
| **pooled** | 0.317 | 0.412/0.299 | 0.771/0.644 | 0.581/0.449 | 0.928/0.911 |

**Verdict (pooled ROC-AUC):** D_hat=0.581 vs max(phi=0.412, conf=0.771)=0.771, oracle=0.928 -> **D_hat does NOT beat both singles.**

## Grouped-CV F1 (leave-one-group-out; tau tuned on train only)

| detector | F1 | precision | recall |
|---|---|---|---|
| phi_only | 0.479 | 0.321 | 1.000 |
| conf_only | 0.579 | 0.494 | 0.712 |
| D_hat | 0.479 | 0.321 | 1.000 |
| oracle_D | 0.840 | 0.907 | 0.783 |

**AND rule** (phi>=0.5 & conf_norm<=0.5): precision 0.414, recall 0.379, F1 0.395, flags 6943/23891.

## Risk–coverage (abstain on riskiest; residual error among kept; AURC lower=better)

| detector | AURC | residual@90% cov | residual@75% cov |
|---|---|---|---|
| phi_only | 0.4140 | 0.311 | 0.327 |
| conf_only | 0.1573 | 0.265 | 0.213 |
| D_hat | 0.3251 | 0.291 | 0.272 |
| oracle_D | 0.0839 | 0.242 | 0.113 |

_Base error rate (keep everything) = 0.317. A good detector pushes residual error well below base at high coverage._

## Is D_hat a faithful proxy? (Spearman rank correlation)

| scope | rho(D_hat, oracle_D) | rho(1-conf, 1-s) | rho(conf_norm, s) |
|---|---|---|---|
| whisper_large_v3 | 0.635 | 0.454 | 0.454 |
| wav2vec2_960h | 0.470 | 0.506 | 0.506 |
| parakeet_tdt | 0.644 | 0.615 | 0.615 |
| **pooled** | 0.581 | 0.529 | 0.529 |

_rho(1-conf,1-s) is the linchpin: if acoustic uncertainty tracks semantic wrongness, D_hat approximates oracle_D and the proxy is principled._

## Residual deception mass = phi*(1-s) (abstain on riskiest; mean among kept; lower=better)

| detector | AURC(deception) | resid-mass@90% | resid-mass@75% |
|---|---|---|---|
| phi_only | 0.0773 | 0.114 | 0.103 |
| conf_only | 0.0666 | 0.114 | 0.094 |
| D_hat | 0.0549 | 0.105 | 0.081 |
| oracle_D | 0.0223 | 0.080 | 0.031 |

**Verdict (residual-deception AURC, lower=better):** D_hat=0.0549 vs min(phi=0.0773, conf=0.0666)=0.0666, oracle=0.0223 -> D_hat BEATS both singles at removing fluent-wrong mass.

_Base deception mass (keep everything) = 0.1292._
