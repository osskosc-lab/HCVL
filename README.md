# HCVL — History-Conditioned Value Learning

Phase 1 tests a narrow falsifiable claim: **after matching the current observable state, does low-dimensional history improve prediction of future intervention value under a common OOD probe?**

## Confirmatory design
- Generators: Null-0 Markov, Positive-1 delayed state, Positive-2 order-sensitive memory.
- State matching: AB/BA terminal observable distance `< 0.75`.
- History representation: three short lags + two exponential action summaries; never the full raw past.
- Target: discounted future intervention **value**, not direct next-outcome prediction.
- OOD probes: `-1.5` or `+1.5`, outside training probe range `[-1, 1]`.
- Controls: state-only ridge and history-shuffled HCVL.
- Decision: every frozen gate in `preregistration/phase1.yaml` must pass.

## Reproduce
```bash
python -m pip install numpy pytest
python scripts/verify_phase1.py
pytest -q
python run_phase1.py
```

`phase1.freeze.json` contains the SHA-256 of the preregistration. Editing the frozen preregistration invalidates the confirmatory run.
