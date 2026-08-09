from __future__ import annotations
import csv, hashlib, json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
CFG = json.loads((ROOT / "preregistration/phase1b.yaml").read_text())
KINDS = CFG["design"]["generators"]
SEEDS = CFG["design"]["seeds"]

def verify_freeze():
    fr = json.loads((ROOT / "preregistration/phase1b.freeze.json").read_text())
    actual = hashlib.sha256((ROOT / fr["file"]).read_bytes()).hexdigest()
    if actual != fr["sha256"]:
        raise RuntimeError(f"Phase 1B preregistration hash mismatch: {actual} != {fr['sha256']}")

def _rollout(kind, x0, actions, a=.65, b=.15, c=.28, tau=3):
    xs = [float(x0)]
    mem = np.zeros(2, dtype=float)
    alt = np.zeros(2, dtype=float)
    acts = []
    for u in np.asarray(actions, dtype=float):
        x = xs[-1]
        delayed = xs[-1-tau] if len(xs) > tau else xs[0]
        hidden = mem[1] - mem[0]
        unseen = np.tanh(0.9*alt[0] - 0.7*alt[1] + 0.35*(acts[-4] if len(acts) >= 4 else 0.0))
        if kind == "null0_markov":
            nx = a*x + b*u
        elif kind == "positive1_delay":
            nx = a*x + b*u + c*delayed
        elif kind == "positive2_order":
            nx = a*x + b*u + c*hidden
        elif kind == "positive3_unseen":
            nx = a*x + b*u + 0.12*unseen
        else:
            raise ValueError(kind)
        xs.append(float(nx))
        acts.append(float(u))
        mem = np.asarray((0.35, 0.82))*mem + u
        alt = np.asarray((0.17, 0.63))*alt + u
    return np.asarray(xs), mem, alt

def _value(kind, xs, mem, alt, actions, probe, horizon=6, gamma=.9, a=.65, b=.15, c=.28, tau=3):
    states = list(np.asarray(xs, dtype=float))
    memory = np.asarray(mem, dtype=float).copy()
    unseen_mem = np.asarray(alt, dtype=float).copy()
    acts = list(np.asarray(actions, dtype=float))
    total = 0.0
    for k in range(horizon):
        u = float(probe) if k == 0 else 0.0
        x = states[-1]
        delayed = states[-1-tau] if len(states) > tau else states[0]
        hidden = memory[1] - memory[0]
        unseen = np.tanh(0.9*unseen_mem[0] - 0.7*unseen_mem[1] + 0.35*(acts[-4] if len(acts) >= 4 else 0.0))
        if kind == "null0_markov":
            nx = a*x + b*u
        elif kind == "positive1_delay":
            nx = a*x + b*u + c*delayed + u*delayed
        elif kind == "positive2_order":
            nx = a*x + b*u + c*hidden + u*hidden
        elif kind == "positive3_unseen":
            nx = a*x + b*u + 0.12*unseen + 0.85*u*unseen
        else:
            raise ValueError(kind)
        total += (gamma**k)*nx
        states.append(float(nx))
        acts.append(float(u))
        memory = np.asarray((0.35, 0.82))*memory + u
        unseen_mem = np.asarray((0.17, 0.63))*unseen_mem + u
    return float(total)

def _random_history(rng, n=12):
    return rng.choice(np.asarray([-1.0, 0.0, 1.0]), size=n, p=[.30, .40, .30])

def _matched_histories(rng, n=12):
    sd = CFG["design"]["pair_action_jitter_sd"]
    A = 1.0 + rng.normal(0, sd)
    B = -0.6 + rng.normal(0, sd)
    prefix = rng.choice(np.asarray([-.4, 0.0, .4]), size=n-4, p=[.25, .50, .25])
    return np.r_[prefix, A, 0.0, B, 0.0], np.r_[prefix, B, 0.0, A, 0.0]

def _state(x, q):
    return np.asarray([x, q, x*q, 1.0], dtype=float)

def _state_nonlinear(x, q):
    return np.asarray([x, q, x*q, x*x, q*q, x*x*q, x*q*q, x**3, q**3, 1.0], dtype=float)

def _history(xs, actions):
    xs = np.asarray(xs, dtype=float)
    actions = np.asarray(actions, dtype=float)
    sums = []
    for rho in (0.35, 0.82):
        s = 0.0
        for u in actions:
            s = rho*s + float(u)
        sums.append(s)
    return np.asarray([xs[-2], xs[-3], xs[-4], *sums], dtype=float)

def _history_misspecified(xs, actions):
    xs = np.asarray(xs, dtype=float)
    s = 0.0
    for u in np.asarray(actions, dtype=float):
        s = 0.55*s + float(u)
    return np.asarray([xs[-2], xs[-5], s], dtype=float)

def _hcvl(xs, actions, q):
    h = _history(xs, actions)
    x = float(xs[-1])
    return np.asarray([x, q, *h, x*q, *(q*h), 1.0], dtype=float)

def _hcvl_misspecified(xs, actions, q):
    h = _history_misspecified(xs, actions)
    x = float(xs[-1])
    return np.asarray([x, q, *h, x*q, *(q*h), 1.0], dtype=float)

def _ridge(X, y, lam=.1):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    return np.linalg.solve(X.T@X + lam*np.eye(X.shape[1]), X.T@y)

def _rmse(y, pred):
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    return float(np.sqrt(np.mean((y-pred)**2)))

def one_seed(kind, seed):
    rng = np.random.default_rng(seed)
    n_train = CFG["design"]["train_episodes_per_seed"]
    hlen = CFG["design"]["history_length"]
    noise = CFG["design"]["value_noise_sd"]
    lam = CFG["models"]["ridge_lambda"]

    Xs, Xn, Xh, Xm, y = [], [], [], [], []
    for _ in range(n_train):
        x0 = rng.normal(0, .5)
        actions = _random_history(rng, hlen)
        xs, mem, alt = _rollout(kind, x0, actions)
        q = rng.uniform(-1.0, 1.0)
        target = _value(kind, xs, mem, alt, actions, q) + rng.normal(0, noise)
        Xs.append(_state(xs[-1], q))
        Xn.append(_state_nonlinear(xs[-1], q))
        Xh.append(_hcvl(xs, actions, q))
        Xm.append(_hcvl_misspecified(xs, actions, q))
        y.append(target)

    Xh = np.asarray(Xh)
    ws = _ridge(Xs, y, lam)
    wn = _ridge(Xn, y, lam)
    wh = _ridge(Xh, y, lam)
    wm = _ridge(Xm, y, lam)

    perm = rng.permutation(n_train)
    Xsh = Xh.copy()
    Xsh[:, 2:7] = Xh[perm, 2:7]
    Xsh[:, 8:13] = Xh[perm, 8:13]
    wsh = _ridge(Xsh, y, lam)

    rng = np.random.default_rng(seed + 10000)
    yt, ps, pn, ph, pm, psh = [], [], [], [], [], []
    retained = 0
    n_pairs = CFG["design"]["test_pairs_per_seed"]
    eps = CFG["design"]["state_match_epsilon"]
    for _ in range(n_pairs):
        x0 = rng.normal(0, .5)
        ab, ba = _matched_histories(rng, hlen)
        xa, ma, aa = _rollout(kind, x0, ab)
        xb, mb, bb = _rollout(kind, x0, ba)
        if abs(float(xa[-1]) - float(xb[-1])) >= eps:
            continue
        retained += 1
        for actions, xs, mem, alt in ((ab, xa, ma, aa), (ba, xb, mb, bb)):
            q = float(rng.choice(np.asarray([-1.5, 1.5])))
            target = _value(kind, xs, mem, alt, actions, q) + rng.normal(0, noise)
            yt.append(target)
            ps.append(_state(xs[-1], q) @ ws)
            pn.append(_state_nonlinear(xs[-1], q) @ wn)
            ph.append(_hcvl(xs, actions, q) @ wh)
            pm.append(_hcvl_misspecified(xs, actions, q) @ wm)
            psh.append(_hcvl(xs, actions, q) @ wsh)

    r_state = _rmse(yt, ps)
    r_nonlin = _rmse(yt, pn)
    r_h = _rmse(yt, ph)
    r_m = _rmse(yt, pm)
    r_sh = _rmse(yt, psh)
    return {
        "generator": kind,
        "seed": seed,
        "rmse_state": r_state,
        "rmse_nonlinear": r_nonlin,
        "rmse_hcvl": r_h,
        "rmse_misspecified": r_m,
        "rmse_shuffle": r_sh,
        "hcvl_to_state_ratio": r_h/r_state,
        "hcvl_to_nonlinear_ratio": r_h/r_nonlin,
        "misspecified_to_nonlinear_ratio": r_m/r_nonlin,
        "shuffle_ratio": r_sh/r_h,
        "misspec_degradation": r_m/r_h,
        "match_retention": retained/n_pairs,
    }

def aggregate(rows):
    out = {}
    metric_keys = [k for k in rows[0] if k not in ("generator", "seed")]
    for kind in KINDS:
        rr = [r for r in rows if r["generator"] == kind]
        out[kind] = {k: float(np.mean([r[k] for r in rr])) for k in metric_keys}

    gates = CFG["falsification_gates"]
    known = ("positive1_delay", "positive2_order")
    all_positive = ("positive1_delay", "positive2_order", "positive3_unseen")
    checks = {
        "null_no_spurious_history_gain":
            out["null0_markov"]["hcvl_to_state_ratio"] >= gates["null_no_spurious_history_gain"]["hcvl_to_state_ratio_min"],
        "known_positive_vs_nonlinear":
            np.mean([out[k]["hcvl_to_nonlinear_ratio"] for k in known]) <= gates["known_positive_vs_nonlinear"]["mean_hcvl_to_nonlinear_ratio_max"],
        "misspec_robustness":
            max(out[k]["misspecified_to_nonlinear_ratio"] for k in all_positive) <= gates["misspec_robustness"]["max_misspecified_to_nonlinear_ratio"],
        "history_specificity":
            min(out[k]["shuffle_ratio"] for k in all_positive) >= gates["history_specificity"]["min_shuffle_ratio"],
        "feature_alignment_stress":
            min(out[k]["misspec_degradation"] for k in all_positive) >= gates["feature_alignment_stress"]["min_misspec_degradation"],
        "strict_state_matching":
            min(out[k]["match_retention"] for k in KINDS) >= gates["strict_state_matching"]["min_retention"],
        "unseen_family_generalization":
            out["positive3_unseen"]["hcvl_to_nonlinear_ratio"] <= gates["unseen_family_generalization"]["hcvl_to_nonlinear_ratio_max"]
            and out["positive3_unseen"]["shuffle_ratio"] >= gates["unseen_family_generalization"]["shuffle_ratio_min"],
    }
    checks = {k: bool(v) for k, v in checks.items()}
    out["gates"] = checks
    out["failed_gates"] = [k for k, v in checks.items() if not v]
    out["decision"] = "SUPPORT" if all(checks.values()) else "FALSIFIED_OR_INCONCLUSIVE"
    return out

def main():
    verify_freeze()
    rows = [one_seed(kind, seed) for kind in KINDS for seed in range(SEEDS)]
    summary = aggregate(rows)
    outdir = ROOT / "results"
    outdir.mkdir(exist_ok=True)
    with (outdir / "phase1b_seed_results.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    (outdir / "phase1b_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    if summary["decision"] != "SUPPORT":
        raise SystemExit(2)

if __name__ == "__main__":
    main()
