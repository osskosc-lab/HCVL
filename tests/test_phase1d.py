import numpy as np
from run_phase1d import histories, control_histories, fit_shared_encoder, encode, paired_head


def toy_history(n=8,L=12):
    s=np.arange(n*(L+1),dtype=float).reshape(n,L+1)/100.0
    a=np.tile(np.arange(L,dtype=float),(n,1))
    return histories(s,a)


def test_order_preserving_keeps_internal_sequence():
    H=toy_history(); rng=np.random.default_rng(1)
    Hp=control_histories(H,"order_preserving",rng)
    originals={tuple(row.ravel()) for row in H}
    assert all(tuple(row.ravel()) in originals for row in Hp)


def test_order_destroying_preserves_pair_multiset():
    H=toy_history(n=4); rng=np.random.default_rng(2)
    Hd=control_histories(H,"order_destroying",rng)
    for a,b in zip(H,Hd):
        sa=sorted(map(tuple,a.tolist())); sb=sorted(map(tuple,b.tolist()))
        assert sa==sb


def test_constant_history_maps_to_zero_centered_latent():
    H=toy_history(); fut=np.arange(32,dtype=float).reshape(8,4)
    mu,B=fit_shared_encoder(H,fut)
    Hc=control_histories(H,"constant_history",np.random.default_rng(3),mean_flat=mu)
    Z=encode(Hc,mu,B)
    assert np.allclose(Z,0.0)


def test_paired_head_dimension_is_input_mode_independent():
    z=np.zeros(4)
    assert paired_head(0.2,1.8,z).shape==paired_head(-0.3,-1.8,z).shape
    assert len(paired_head(0.0,0.0,z))==33
