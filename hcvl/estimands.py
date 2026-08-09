from __future__ import annotations
import numpy as np

def history_features(xs, actions, decays=(0.35,0.82)):
    """Low-dimensional history, never the full raw past."""
    xs = np.asarray(xs, dtype=float)
    actions = np.asarray(actions, dtype=float)
    lagged = [xs[-2], xs[-3], xs[-4]]
    summaries = []
    for rho in decays:
        s = 0.0
        for u in actions:
            s = rho*s + float(u)
        summaries.append(s)
    return np.asarray([*lagged, *summaries], dtype=float)

def state_features(x, probe):
    x, q = float(x), float(probe)
    return np.asarray([x, q, x*q, 1.0])

def hcvl_features(xs, actions, probe):
    q = float(probe)
    h = history_features(xs, actions)
    x = float(np.asarray(xs)[-1])
    return np.asarray([x, q, *h, x*q, *(q*h), 1.0], dtype=float)

def ridge_fit(X, y, lam=0.1):
    X=np.asarray(X,float); y=np.asarray(y,float)
    I=np.eye(X.shape[1])
    return np.linalg.solve(X.T@X + lam*I, X.T@y)

def predict(X, w):
    return np.asarray(X,float) @ np.asarray(w,float)
