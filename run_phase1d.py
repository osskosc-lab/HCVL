from __future__ import annotations
import csv, hashlib, json
from pathlib import Path
import numpy as np
from run_phase1c import rollout, intervention_value

ROOT = Path(__file__).resolve().parent
CFG = json.loads((ROOT / "preregistration/phase1d.yaml").read_text())
D = CFG["design"]
P = CFG["paired_architecture"]
KINDS = D["generators"]


def ridge(X, y, lam=None):
    X=np.asarray(X,float); y=np.asarray(y,float)
    lam=P["ridge_lambda"] if lam is None else lam
    I=np.eye(X.shape[1]); I[-1,-1]=0.0
    return np.linalg.solve(X.T@X + lam*I, X.T@y)


def rmse(y,p):
    y=np.asarray(y,float); p=np.asarray(p,float)
    return float(np.sqrt(np.mean((y-p)**2)))


def histories(seqs,acts):
    return np.stack([np.asarray(seqs)[:,:-1],np.asarray(acts)],axis=2)


def flatten_history(H):
    H=np.asarray(H,float)
    return H.reshape(H.shape[0],-1)


def fit_shared_encoder(H,future_proxy):
    X=flatten_history(H); F=np.asarray(future_proxy,float)
    mu=X.mean(0); Xc=X-mu; Fc=F-F.mean(0)
    C=Xc.T@Fc/max(len(X),1)
    U,_,_=np.linalg.svd(C,full_matrices=False)
    B=U[:,:min(int(P["latent_dim"]),U.shape[1])]
    return mu,B


def encode(H,mu,B):
    return (flatten_history(H)-mu)@B


def control_histories(H,mode,rng,mean_flat=None):
    H=np.asarray(H,float)
    if mode=="raw_history":
        return H.copy()
    if mode=="order_preserving":
        return H[rng.permutation(len(H))].copy()
    if mode=="order_destroying":
        out=H.copy()
        for i in range(len(out)):
            out[i]=out[i,rng.permutation(out.shape[1]),:]
        return out
    if mode=="constant_history":
        if mean_flat is None: mean_flat=flatten_history(H).mean(0)
        return np.repeat(np.asarray(mean_flat,float)[None,:],len(H),axis=0).reshape(H.shape)
    raise ValueError(mode)


def state_base(x,q):
    x=float(x); q=float(q)
    return np.asarray([x,q,x*q,x*x,q*q,x*x*q,x*q*q,np.sin(x),np.cos(x),
        np.sin(q),np.cos(q),np.tanh(x+q),np.tanh(x-q),x**3,q**3,(x*q)**2],float)


def paired_head(x,q,z):
    z=np.asarray(z,float); x=float(x); q=float(q)
    return np.r_[state_base(x,q),z,x*z,q*z,z*z,1.0]


def build_train(kind,seed):
    rng=np.random.default_rng(seed); L=D["history_length"]
    seqs=[]; acts=[]; qs=[]; ys=[]; fut=[]
    for _ in range(D["train_episodes_per_seed"]):
        u=rng.choice(np.array([-1.,0.,1.]),L,p=[.3,.4,.3])
        x0=rng.normal(0,.45); xs=rollout(kind,x0,u)
        q=float(rng.uniform(*D["train_probe_range"]))
        y=intervention_value(kind,xs,u,q,D["train_mechanism"])+rng.normal(0,D["noise_sd"])
        f=rollout(kind,x0,np.r_[u,np.zeros(4)])[-4:]
        seqs.append(xs); acts.append(u); qs.append(q); ys.append(y); fut.append(f)
    return tuple(map(np.asarray,(seqs,acts,qs,ys,fut)))


def matched_pair(kind,rng):
    L=D["history_length"]; prefix=rng.choice(np.array([-.4,0.,.4]),L-4,p=[.25,.5,.25])
    ab=np.r_[prefix,1.,0.,-.6,0.]; ba=np.r_[prefix,-.6,0.,1.,0.]; x0=rng.normal(0,.45)
    return ab,ba,rollout(kind,x0,ab),rollout(kind,x0,ba)


def build_test(kind,seed):
    rng=np.random.default_rng(seed+10000); rows=[]; retained=0
    for _ in range(D["test_pairs_per_seed"]):
        ab,ba,xa,xb=matched_pair(kind,rng)
        if abs(xa[-1]-xb[-1]) >= D["state_match_epsilon"]: continue
        retained += 1
        for u,xs in ((ab,xa),(ba,xb)):
            q=float(rng.choice(np.asarray(D["test_probe_actions"])))
            y=intervention_value(kind,xs,u,q,D["test_mechanism"])+rng.normal(0,D["noise_sd"])
            rows.append((xs,u,q,y))
    return rows,retained/D["test_pairs_per_seed"]


def fit_heads(seqs,acts,qs,ys,fut,seed):
    H=histories(seqs,acts); mu,B=fit_shared_encoder(H,fut)
    heads={}; modes=list(P["controls"].keys())
    for j,mode in enumerate(modes):
        rng=np.random.default_rng(seed+2000+101*j)
        Hm=control_histories(H,mode,rng,mean_flat=mu)
        Z=encode(Hm,mu,B)
        X=np.asarray([paired_head(x[-1],q,z) for x,q,z in zip(seqs,qs,Z)])
        heads[mode]=ridge(X,ys)
    return mu,B,heads


def one_seed(kind,seed):
    seqs,acts,qs,ys,fut=build_train(kind,seed)
    mu,B,heads=fit_heads(seqs,acts,qs,ys,fut,seed)
    rows,ret=build_test(kind,seed)
    tseqs=np.asarray([r[0] for r in rows]); tacts=np.asarray([r[1] for r in rows])
    tqs=np.asarray([r[2] for r in rows]); yt=np.asarray([r[3] for r in rows])
    H=histories(tseqs,tacts); preds={}; rmses={}
    modes=list(P["controls"].keys())
    for j,mode in enumerate(modes):
        rng=np.random.default_rng(seed+9000+101*j)
        Hm=control_histories(H,mode,rng,mean_flat=mu)
        Z=encode(Hm,mu,B)
        X=np.asarray([paired_head(x[-1],q,z) for x,q,z in zip(tseqs,tqs,Z)])
        preds[mode]=X@heads[mode]; rmses[mode]=rmse(yt,preds[mode])
    raw=rmses["raw_history"]; const=rmses["constant_history"]
    return {
        "generator":kind,"seed":seed,
        "rmse_raw":raw,"rmse_order_preserving":rmses["order_preserving"],
        "rmse_order_destroying":rmses["order_destroying"],"rmse_constant":const,
        "raw_to_constant":raw/const,
        "order_preserving_to_raw":rmses["order_preserving"]/raw,
        "order_destroying_to_raw":rmses["order_destroying"]/raw,
        "match_retention":ret,"feature_dim":len(paired_head(0.,0.,np.zeros(B.shape[1])))
    }


def aggregate(rows):
    out={}
    for kind in KINDS:
        rr=[r for r in rows if r["generator"]==kind]
        keys=[k for k in rr[0] if k not in ("generator","seed")]
        out[kind]={k:float(np.mean([r[k] for r in rr])) for k in keys}
    G=CFG["gates"]; null=out["null_markov"]
    pos=[out["adversarial_delay"],out["adversarial_switching"]]
    dims={int(r["feature_dim"]) for r in rows}
    checks={
      "null_calibration_lower": null["raw_to_constant"] >= G["null_raw_to_constant_min"],
      "null_calibration_upper": null["raw_to_constant"] <= G["null_raw_to_constant_max"],
      "positive_raw_information_gain": max(p["raw_to_constant"] for p in pos) <= G["positive_raw_to_constant_max"],
      "positive_order_destroying_specificity": min(p["order_destroying_to_raw"] for p in pos) >= G["positive_order_destroying_to_raw_min"],
      "positive_order_preserving_specificity": min(p["order_preserving_to_raw"] for p in pos) >= G["positive_order_preserving_to_raw_min"],
      "strict_state_matching": min(out[k]["match_retention"] for k in KINDS) >= G["strict_match_retention_min"],
      "heldout_mechanism": float(np.mean([p["raw_to_constant"] for p in pos])) <= G["heldout_mechanism_positive_mean_max"],
      "paired_feature_dim_equal": len(dims)==1 and G["paired_feature_dim_equal"]
    }
    checks={k:bool(v) for k,v in checks.items()}; out["gates"]=checks
    out["failed_gates"]=[k for k,v in checks.items() if not v]
    out["feature_dims_observed"]=sorted(dims)
    out["decision"]="SUPPORT" if all(checks.values()) else "FALSIFIED_OR_INCONCLUSIVE"
    return out


def verify_freeze():
    fr=json.loads((ROOT/"preregistration/phase1d.freeze.json").read_text())
    actual=hashlib.sha256((ROOT/fr["file"]).read_bytes()).hexdigest()
    if actual!=fr["sha256"]: raise RuntimeError(f"Phase 1D preregistration hash mismatch: {actual}")


def main():
    verify_freeze()
    rows=[one_seed(k,s) for k in KINDS for s in range(D["seeds"])]
    summary=aggregate(rows); out=ROOT/"results"; out.mkdir(exist_ok=True)
    with (out/"phase1d_seed_results.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    (out/"phase1d_summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps(summary,indent=2))
    if summary["decision"]!="SUPPORT": raise SystemExit(2)

if __name__=="__main__": main()
