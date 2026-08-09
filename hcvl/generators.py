from __future__ import annotations
import numpy as np

DEFAULTS = dict(a=0.65, b=0.15, c=0.28, eta=1.0, tau=3, memory_decays=(0.35, 0.82))

def rollout_history(kind: str, x0: float, actions, *, a=.65, b=.15, c=.28, tau=3, memory_decays=(.35,.82)):
    """Generate the pre-probe observable trajectory and latent order memory."""
    xs = [float(x0)]
    memory = np.zeros(len(memory_decays), dtype=float)
    for u in np.asarray(actions, dtype=float):
        x = xs[-1]
        delayed = xs[-1-tau] if len(xs) > tau else xs[0]
        if kind == "null0_markov":
            nx = a*x + b*u
        elif kind == "positive1_delay":
            nx = a*x + b*u + c*delayed
        elif kind == "positive2_order":
            hidden = memory[1] - memory[0]
            nx = a*x + b*u + c*hidden
        else:
            raise ValueError(f"unknown generator: {kind}")
        xs.append(float(nx))
        memory = np.asarray(memory_decays)*memory + u
    return np.asarray(xs), memory

def intervention_value(kind: str, xs, memory, probe: float, *, horizon=6, gamma=.90,
                       a=.65, b=.15, c=.28, eta=1.0, tau=3, memory_decays=(.35,.82)):
    """Discounted future value under a one-step probe then zero actions."""
    states = list(np.asarray(xs, dtype=float))
    mem = np.asarray(memory, dtype=float).copy()
    total = 0.0
    for k in range(horizon):
        u = float(probe) if k == 0 else 0.0
        x = states[-1]
        delayed = states[-1-tau] if len(states) > tau else states[0]
        if kind == "null0_markov":
            nx = a*x + b*u
        elif kind == "positive1_delay":
            nx = a*x + b*u + c*delayed + eta*u*delayed
        elif kind == "positive2_order":
            hidden = mem[1] - mem[0]
            nx = a*x + b*u + c*hidden + eta*u*hidden
        else:
            raise ValueError(f"unknown generator: {kind}")
        total += (gamma**k) * nx
        states.append(float(nx))
        mem = np.asarray(memory_decays)*mem + u
    return float(total)
