# HCVL Phase 1 — Confirmatory Synthetic Experiment Report

Date: 2026-08-09  
Preregistration SHA-256: `038291d3db9fec2be0abdbe9e191f03172f469dc3e9d02829149a28e914ead9e`  
Decision: **SUPPORT**

## Question
After matching the current observable state, does a low-dimensional history representation improve prediction of future intervention value under a common OOD probe?

## Frozen design
- 30 seeds per generator.
- 800 training episodes/seed.
- 250 AB/BA candidate test pairs/seed.
- State-matching threshold: `|x_AB - x_BA| < 0.75`.
- Training probe range: `[-1, 1]`.
- OOD probe actions: `-1.5, +1.5`.
- Baselines: state-only ridge; history-shuffled HCVL.
- HCVL history: 3 short lags + 2 exponentially decayed action summaries.

## Aggregate results

| Generator | State RMSE | HCVL RMSE | HCVL/State | Shuffle/HCVL | Match retention | State regret | HCVL regret |
|---|---:|---:|---:|---:|---:|---:|---:|
| Null-0 Markov | 0.050465 | 0.051093 | 1.0124 | 1.0012 | 1.000 | 0.000000 | 0.000000 |
| Positive-1 Delay | 0.731291 | 0.052607 | 0.0720 | 14.0399 | 1.000 | 0.121562 | 0.000050 |
| Positive-2 Order | 0.893463 | 0.051141 | 0.0575 | 17.2515 | 1.000 | 0.067690 | 0.000003 |

Positive-generator mean decision-regret ratio: `0.000244`.

## Gate results
- PASS — `null0_no_spurious_advantage`
- PASS — `positive1_rmse`
- PASS — `positive1_shuffle`
- PASS — `positive2_rmse`
- PASS — `positive2_shuffle`
- PASS — `state_matching`
- PASS — `decision_regret`

## Interpretation
Phase 1 passes all preregistered gates in the synthetic benchmark. The null generator does not show a spurious HCVL advantage, while both positive controls show large OOD value-prediction gains that disappear under history shuffling. Online action-selection regret is also lower for HCVL in the positive generators.

This is a **method/calibration result**, not evidence that real-world systems are history-dependent. The positive generators were deliberately constructed to contain history-dependent intervention value, so the scientific content of Phase 1 is that the pipeline can recover known history value while remaining calibrated on a Markov null. A stronger next phase should reduce alignment between the hand-built history representation and the generator, add capacity-matched nonlinear baselines, and test misspecified/partial-history cases.
