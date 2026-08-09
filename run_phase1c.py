from __future__ import annotations
import csv, hashlib, json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
CFG = json.loads((ROOT / "preregistration/phase1c.yaml").read_text())
D = CFG["design"]
KINDS = D["generators"]


def ridge(X, y, lam=0.1):
    X = np.asarray(X, float); y = np.asarray(y, float)
    I = np.eye(X.shape[1]); I[-1, -1] = 0.0
    return np.linalg.solve(X.T @ X + lam * I, X.T @ y)


def rmse(y, p):
    y = np.asarray(y, float); p = np.asarray(p, float)
    return float(np.sqrt(np.mean((y-p)**2)))


def rollout(kind, x0, actions):
    x = [float(x0)]
    mf = 0.0; ms = 0.0
    for a in np.asarray(actions, float):
        if kind == "null_markov":
            nx = 0.68*x[-1] + 0.16*a
        elif kind == "adversarial_delay":
            delayed = x[-4] if len(x) >= 4 else x[0]
            nx = 0.58*x[-1] + 0.12*a + 0.26*np.tanh(1.4*delayed) + 0.08*np.sin(delayed)
        elif kind == "adversarial_switching":
            mf = 0.28*mf + a; ms = 0.86*ms + a
            s = np.tanh(ms-mf)
            nx = 0.60*x[-1] + 0.12*a + 0.20*s + 0.05*np.sin(2*s)
        else:
            raise ValueError(kind)
        x.append(float(nx))
    return np.asarray(x)


def intervention_value(kind, xs, actions, q, mechanism):
    states = list(np.asarray(xs, float))
    mf = 0.0; ms = 0.0
    if kind == "adversarial_switching":
        for a in actions:
            mf = 0.28*mf + a; ms = 0.86*ms + a
    total = 0.0
    for k in range(D["value_horizon"]):
        if k == 0:
            a = float(q)
        elif mechanism == "two_pulse" and k == 2:
            a = -0.45*float(q)
        else:
            a = 0.0
        if kind == "null_markov":
            nx = 0.68*states[-1] + 0.16*a
        elif kind == "adversarial_delay":
            delayed = states[-4] if len(states) >= 4 else states[0]
            nx = (0.58*states[-1] + 0.12*a + 0.26*np.tanh(1.4*delayed)
                  + 0.08*np.sin(delayed) + 0.75*a*np.tanh(delayed))
        else:
            mf = 0.28*mf + a; ms = 0.86*ms + a
            s = np.tanh(ms-mf)
            nx = 0.60*states[-1] + 0.12*a + 0.20*s + 0.05*np.sin(2*s) + 0.72*a*s
        total += (D["discount"]**k) * nx
        states.append(float(nx))
    return float(total)


def state_features(x, q):
    x=float(x); q=float(q)
    base = [x,q,x*q,x*x,q*q,x*x*q,x*q*q,np.sin(x),np.cos(x),np.sin(q),np.cos(q),
            np.tanh(x+q),np.tanh(x-q),x**3,q**3,(x*q)**2,1.0]
    return np.asarray(base,float)


def hist_head(x, q, z):
    z=np.asarray(z,float); x=float(x); q=float(q)
    return np.r_[x,q,z,x*q,q*z,1.0]


def train_psr(seqs, actions, future_proxy, latent_dim=6):
    P = np.concatenate([seqs[:,:-1], actions], axis=1)
    mu = P.mean(0)
    Pc = P-mu; Fc = future_proxy-future_proxy.mean(0)
    C = Pc.T @ Fc / len(P)
    U, _, _ = np.linalg.svd(C, full_matrices=False)
    B = U[:, :min(latent_dim,U.shape[1])]
    return mu, B, (P-mu) @ B


def train_rnn_encoder(seqs, actions, hidden=8, epochs=60, lr=0.03, seed=0):
    rng=np.random.default_rng(seed)
    N,L=actions.shape
    X=np.stack([seqs[:,:-1],actions],axis=2)
    Y=seqs[:,1:]
    Wxh=rng.normal(0,0.25,(2,hidden)); Whh=rng.normal(0,0.15,(hidden,hidden))
    Wy=rng.normal(0,0.2,(hidden,1)); bh=np.zeros(hidden); by=0.0
    for _ in range(epochs):
        h=np.zeros((N,hidden)); hs=[h]; preds=[]
        for t in range(L):
            h=np.tanh(X[:,t,:]@Wxh + h@Whh + bh)
            hs.append(h); preds.append((h@Wy).ravel()+by)
        P=np.stack(preds,axis=1)
        err=(P-Y)/(N*L)
        dWxh=np.zeros_like(Wxh); dWhh=np.zeros_like(Whh); dWy=np.zeros_like(Wy)
        dbh=np.zeros_like(bh); dby=0.0; dh_next=np.zeros((N,hidden))
        for t in reversed(range(L)):
            h=hs[t+1]; hp=hs[t]; e=2*err[:,t]
            dWy += h.T@e[:,None]; dby += e.sum()
            dh=e[:,None]@Wy.T + dh_next
            da=dh*(1-h*h)
            dWxh += X[:,t,:].T@da; dWhh += hp.T@da; dbh += da.sum(0)
            dh_next=da@Whh.T
        norm=np.sqrt(sum((g*g).sum() for g in (dWxh,dWhh,dWy))+(dbh*dbh).sum()+dby*dby)
        sc=min(1.0,5.0/(norm+1e-12))
        Wxh-=lr*sc*dWxh; Whh-=lr*sc*dWhh; Wy-=lr*sc*dWy; bh-=lr*sc*dbh; by-=lr*sc*dby
    return Wxh,Whh,bh


def rnn_encode(seqs, actions, pars):
    Wxh,Whh,bh=pars; N,L=actions.shape
    h=np.zeros((N,Whh.shape[0])); X=np.stack([seqs[:,:-1],actions],axis=2)
    for t in range(L): h=np.tanh(X[:,t,:]@Wxh+h@Whh+bh)
    return h


def build_train(kind, seed):
    rng=np.random.default_rng(seed); L=D["history_length"]
    seqs=[]; acts=[]; qs=[]; ys=[]; fut=[]
    for _ in range(D["train_episodes_per_seed"]):
        u=rng.choice(np.array([-1.,0.,1.]),L,p=[.3,.4,.3])
        xs=rollout(kind,rng.normal(0,.45),u); q=float(rng.uniform(*D["train_probe_range"]))
        y=intervention_value(kind,xs,u,q,"single_pulse")+rng.normal(0,D["noise_sd"])
        f=rollout(kind,xs[0],np.r_[u,np.zeros(4)])[-4:]
        seqs.append(xs); acts.append(u); qs.append(q); ys.append(y); fut.append(f)
    return tuple(map(np.asarray,(seqs,acts,qs,ys,fut)))


def matched_pair(kind, rng):
    L=D["history_length"]; prefix=rng.choice(np.array([-.4,0.,.4]),L-4,p=[.25,.5,.25])
    ab=np.r_[prefix,1.,0.,-.6,0.]; ba=np.r_[prefix,-.6,0.,1.,0.]; x0=rng.normal(0,.45)
    return ab,ba,rollout(kind,x0,ab),rollout(kind,x0,ba)


def build_test(kind, seed):
    rng=np.random.default_rng(seed+10000); rows=[]; retained=0
    for _ in range(D["test_pairs_per_seed"]):
        ab,ba,xa,xb=matched_pair(kind,rng)
        if abs(xa[-1]-xb[-1]) >= D["state_match_epsilon"]: continue
        retained += 1
        for u,xs in ((ab,xa),(ba,xb)):
            q=float(rng.choice(np.asarray(D["test_probe_actions"])))
            y=intervention_value(kind,xs,u,q,"two_pulse")+rng.normal(0,D["noise_sd"])
            rows.append((xs,u,q,y))
    return rows, retained/D["test_pairs_per_seed"]


def one_seed(kind, seed):
    seqs,acts,qs,ys,fut=build_train(kind,seed)
    ws=ridge(np.asarray([state_features(x[-1],q) for x,q in zip(seqs,qs)]),ys)
    mu,B,Zp=train_psr(seqs,acts,fut,latent_dim=6)
    wp=ridge(np.asarray([hist_head(x[-1],q,z) for x,q,z in zip(seqs,qs,Zp)]),ys)
    rpars=train_rnn_encoder(seqs,acts,hidden=8,epochs=60,seed=seed+500)
    Zr=rnn_encode(seqs,acts,rpars)
    wr=ridge(np.asarray([hist_head(x[-1],q,z) for x,q,z in zip(seqs,qs,Zr)]),ys)
    rng=np.random.default_rng(seed+777); perm=rng.permutation(len(ys))
    wps=ridge(np.asarray([hist_head(x[-1],q,z) for x,q,z in zip(seqs,qs,Zp[perm])]),ys)
    wrs=ridge(np.asarray([hist_head(x[-1],q,z) for x,q,z in zip(seqs,qs,Zr[perm])]),ys)
    rows,ret=build_test(kind,seed)
    yt=[]; pred_s=[]; pred_p=[]; pred_r=[]; pred_ps=[]; pred_rs=[]
    for xs,u,q,y in rows:
        pvec=np.r_[xs[:-1],u]; zp=(pvec-mu)@B; zr=rnn_encode(xs[None,:],u[None,:],rpars)[0]
        yt.append(y); pred_s.append(state_features(xs[-1],q)@ws)
        pred_p.append(hist_head(xs[-1],q,zp)@wp); pred_r.append(hist_head(xs[-1],q,zr)@wr)
        pred_ps.append(hist_head(xs[-1],q,zp)@wps); pred_rs.append(hist_head(xs[-1],q,zr)@wrs)
    rs,rp,rr,rps,rrs=[rmse(yt,p) for p in (pred_s,pred_p,pred_r,pred_ps,pred_rs)]
    best=min(rp,rr); best_shuffle=min(rps,rrs)
    return {"generator":kind,"seed":seed,"rmse_state":rs,"rmse_psr":rp,"rmse_rnn":rr,
            "rmse_psr_shuffle":rps,"rmse_rnn_shuffle":rrs,"best_to_state":best/rs,
            "psr_to_state":rp/rs,"rnn_to_state":rr/rs,"shuffle_to_best":best_shuffle/best,
            "match_retention":ret}


def aggregate(rows):
    out={}
    for kind in KINDS:
        rr=[r for r in rows if r["generator"]==kind]
        keys=[k for k in rr[0] if k not in ("generator","seed")]
        out[kind]={k:float(np.mean([r[k] for r in rr])) for k in keys}
    G=CFG["gates"]; pos=[out["adversarial_delay"],out["adversarial_switching"]]
    checks={
      "null_learned_no_spurious_gain": min(out["null_markov"]["psr_to_state"],out["null_markov"]["rnn_to_state"]) >= G["null_learned_no_spurious_gain_min"],
      "positive_best_to_state": max(p["best_to_state"] for p in pos) <= G["positive_best_to_state_max"],
      "positive_both_to_state": max(max(p["psr_to_state"],p["rnn_to_state"]) for p in pos) <= G["positive_both_to_state_max"],
      "positive_shuffle_specificity": min(p["shuffle_to_best"] for p in pos) >= G["positive_shuffle_ratio_min"],
      "strict_state_matching": min(out[k]["match_retention"] for k in KINDS) >= G["strict_match_retention_min"],
      "heldout_mechanism": float(np.mean([p["best_to_state"] for p in pos])) <= G["heldout_mechanism_positive_max"]
    }
    checks={k:bool(v) for k,v in checks.items()}; out["gates"]=checks
    out["failed_gates"]=[k for k,v in checks.items() if not v]
    out["decision"]="SUPPORT" if all(checks.values()) else "FALSIFIED_OR_INCONCLUSIVE"
    return out


def verify_freeze():
    fr=json.loads((ROOT/"preregistration/phase1c.freeze.json").read_text())
    actual=hashlib.sha256((ROOT/fr["file"]).read_bytes()).hexdigest()
    if actual != fr["sha256"]: raise RuntimeError(f"Phase 1C preregistration hash mismatch: {actual}")


def main():
    verify_freeze(); rows=[one_seed(k,s) for k in KINDS for s in range(D["seeds"])]
    summary=aggregate(rows); out=ROOT/"results"; out.mkdir(exist_ok=True)
    with (out/"phase1c_seed_results.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    (out/"phase1c_summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps(summary,indent=2))
    if summary["decision"] != "SUPPORT": raise SystemExit(2)

if __name__ == "__main__": main()
