from __future__ import annotations
import csv, hashlib, json
from pathlib import Path
import numpy as np

from hcvl.generators import rollout_history, intervention_value
from hcvl.interventions import random_training_history, matched_ab_ba_histories, sample_train_probe, sample_ood_probe, DECISION_ACTIONS
from hcvl.estimands import state_features, hcvl_features, ridge_fit
from hcvl.state_matching import matched
from hcvl.metrics import rmse, safe_ratio

ROOT=Path(__file__).resolve().parent
PREREG=json.loads((ROOT/"preregistration/phase1.yaml").read_text())
KINDS=PREREG["design"]["generators"]
SEEDS=PREREG["design"]["seeds"]

def build_train(kind, seed):
    rng=np.random.default_rng(seed)
    n=PREREG["design"]["train_episodes_per_seed"]
    h=PREREG["design"]["history_length"]
    noise=PREREG["design"]["value_noise_sd"]
    Xs=[]; Xh=[]; y=[]
    for _ in range(n):
        x0=rng.normal(0,.5)
        actions=random_training_history(rng,h)
        xs,mem=rollout_history(kind,x0,actions)
        q=sample_train_probe(rng)
        val=intervention_value(kind,xs,mem,q)+rng.normal(0,noise)
        Xs.append(state_features(xs[-1],q))
        Xh.append(hcvl_features(xs,actions,q))
        y.append(val)
    return np.asarray(Xs),np.asarray(Xh),np.asarray(y)

def build_test_records(kind, seed):
    rng=np.random.default_rng(seed+10000)
    n=PREREG["design"]["test_pairs_per_seed"]
    h=PREREG["design"]["history_length"]
    eps=PREREG["design"]["state_match_epsilon"]
    records=[]; retained=0
    for pair in range(n):
        x0=rng.normal(0,.5)
        ab,ba=matched_ab_ba_histories(rng,h)
        xab,mab=rollout_history(kind,x0,ab)
        xba,mba=rollout_history(kind,x0,ba)
        if not matched(xab[-1],xba[-1],eps):
            continue
        retained+=1
        records += [(pair,"AB",ab,xab,mab),(pair,"BA",ba,xba,mba)]
    return records, retained/n

def one_seed(kind, seed):
    Xs,Xh,y=build_train(kind,seed)
    lam=PREREG["models"]["ridge_lambda"]
    ws=ridge_fit(Xs,y,lam)
    wh=ridge_fit(Xh,y,lam)

    rng=np.random.default_rng(seed+777)
    perm=rng.permutation(len(Xh))
    Xsh=Xh.copy()
    Xsh[:,2:7]=Xh[perm,2:7]
    Xsh[:,8:13]=Xh[perm,8:13]
    wsh=ridge_fit(Xsh,y,lam)

    records, retention=build_test_records(kind,seed)
    yt=[]; ps=[]; ph=[]; psh=[]; regret_s=[]; regret_h=[]
    noise=PREREG["design"]["value_noise_sd"]
    for _,_,actions,xs,mem in records:
        q=sample_ood_probe(rng)
        truth=intervention_value(kind,xs,mem,q)+rng.normal(0,noise)
        yt.append(truth)
        ps.append(state_features(xs[-1],q)@ws)
        ph.append(hcvl_features(xs,actions,q)@wh)
        psh.append(hcvl_features(xs,actions,q)@wsh)

        true=np.asarray([intervention_value(kind,xs,mem,a) for a in DECISION_ACTIONS])
        pred_s=np.asarray([state_features(xs[-1],a)@ws for a in DECISION_ACTIONS])
        pred_h=np.asarray([hcvl_features(xs,actions,a)@wh for a in DECISION_ACTIONS])
        regret_s.append(true.max()-true[pred_s.argmax()])
        regret_h.append(true.max()-true[pred_h.argmax()])

    r_s=rmse(yt,ps); r_h=rmse(yt,ph); r_sh=rmse(yt,psh)
    return {
      "generator":kind,"seed":seed,
      "rmse_state":r_s,"rmse_hcvl":r_h,"rmse_shuffle":r_sh,
      "rmse_ratio":safe_ratio(r_h,r_s),"shuffle_ratio":safe_ratio(r_sh,r_h),
      "match_retention":retention,
      "regret_state":float(np.mean(regret_s)),"regret_hcvl":float(np.mean(regret_h)),
      "regret_ratio":safe_ratio(np.mean(regret_h)+1e-9,np.mean(regret_s)+1e-9)
    }

def aggregate(rows):
    out={}
    for kind in KINDS:
        rr=[r for r in rows if r["generator"]==kind]
        keys=[k for k in rr[0] if k not in ("generator","seed")]
        out[kind]={k:float(np.mean([r[k] for r in rr])) for k in keys}
    pos_regret=np.mean([out[k]["regret_ratio"] for k in ("positive1_delay","positive2_order")])
    out["positive_mean_regret_ratio"]=float(pos_regret)
    gates=PREREG["falsification_gates"]
    checks={
      "null0_no_spurious_advantage": bool(out["null0_markov"]["rmse_ratio"] >= gates["null0_no_spurious_advantage"]["rmse_ratio_min"]),
      "positive1_rmse": bool(out["positive1_delay"]["rmse_ratio"] <= gates["positive1_history_value"]["rmse_ratio_max"]),
      "positive1_shuffle": bool(out["positive1_delay"]["shuffle_ratio"] >= gates["positive1_history_value"]["shuffle_ratio_min"]),
      "positive2_rmse": bool(out["positive2_order"]["rmse_ratio"] <= gates["positive2_history_value"]["rmse_ratio_max"]),
      "positive2_shuffle": bool(out["positive2_order"]["shuffle_ratio"] >= gates["positive2_history_value"]["shuffle_ratio_min"]),
      "state_matching": bool(min(out[k]["match_retention"] for k in KINDS) >= gates["state_matching"]["retention_min"]),
      "decision_regret": bool(pos_regret <= gates["decision_regret"]["positive_mean_regret_ratio_max"]),
    }
    out["gates"]=checks
    out["decision"]="SUPPORT" if all(checks.values()) else "FALSIFIED_OR_INCONCLUSIVE"
    return out

def verify_freeze():
    fr=json.loads((ROOT/"preregistration/phase1.freeze.json").read_text())
    actual=hashlib.sha256((ROOT/fr["file"]).read_bytes()).hexdigest()
    if actual != fr["sha256"]:
        raise RuntimeError(f"Preregistration hash mismatch: {actual} != {fr['sha256']}")

def main():
    verify_freeze()
    rows=[one_seed(kind,seed) for kind in KINDS for seed in range(SEEDS)]
    summary=aggregate(rows)
    outdir=ROOT/"results"; outdir.mkdir(exist_ok=True)
    with (outdir/"phase1_seed_results.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    (outdir/"phase1_summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps(summary,indent=2))
    if summary["decision"] != "SUPPORT":
        raise SystemExit(2)

if __name__=="__main__":
    main()
