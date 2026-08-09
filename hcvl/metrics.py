from __future__ import annotations
import numpy as np

def rmse(y, pred):
    y=np.asarray(y,float); pred=np.asarray(pred,float)
    return float(np.sqrt(np.mean((pred-y)**2)))

def safe_ratio(num, den, eps=1e-12):
    return float(num/max(float(den),eps))

def mean_regret(true_values, chosen_indices):
    values=np.asarray(true_values,float)
    idx=np.asarray(chosen_indices,int)
    oracle=values.max(axis=1)
    chosen=values[np.arange(len(values)),idx]
    return float(np.mean(oracle-chosen))
