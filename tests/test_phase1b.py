import numpy as np
import run_phase1b as p

def test_unseen_family_is_not_markov_equivalent():
    a = np.array([0,0,0,0,1,0,-.6,0,0,0,0,0.], dtype=float)
    b = np.array([0,0,0,0,-.6,0,1,0,0,0,0,0.], dtype=float)
    xa, ma, aa = p._rollout("positive3_unseen", 0.0, a)
    xb, mb, ab = p._rollout("positive3_unseen", 0.0, b)
    va = p._value("positive3_unseen", xa, ma, aa, a, 1.5)
    vb = p._value("positive3_unseen", xb, mb, ab, b, 1.5)
    assert not np.isclose(va, vb)

def test_misspecified_history_has_different_basis():
    xs = np.arange(13, dtype=float)
    u = np.linspace(-1, 1, 12)
    assert p._history(xs, u).shape == (5,)
    assert p._history_misspecified(xs, u).shape == (3,)

def test_nonlinear_state_baseline_has_more_capacity_than_linear():
    assert p._state_nonlinear(0.2, 1.5).shape[0] > p._state(0.2, 1.5).shape[0]

def test_stricter_matching_threshold_is_frozen():
    assert p.CFG["design"]["state_match_epsilon"] == 0.35
