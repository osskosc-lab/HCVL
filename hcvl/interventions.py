from __future__ import annotations
import numpy as np

def random_training_history(rng, length=12):
    return rng.choice(np.array([-1.0, 0.0, 1.0]), size=length, p=[0.30,0.40,0.30])

def matched_ab_ba_histories(rng, length=12, A=1.0, B=-0.6):
    if length < 4:
        raise ValueError("history length must be >=4")
    prefix = rng.choice(np.array([-0.4,0.0,0.4]), size=length-4, p=[0.25,0.50,0.25])
    ab = np.concatenate([prefix, np.array([A,0.0,B,0.0])])
    ba = np.concatenate([prefix, np.array([B,0.0,A,0.0])])
    return ab, ba

def sample_train_probe(rng):
    return float(rng.uniform(-1.0, 1.0))

def sample_ood_probe(rng):
    return float(rng.choice(np.array([-1.5, 1.5])))

DECISION_ACTIONS = np.array([-1.5, 0.0, 1.5])
