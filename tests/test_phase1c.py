import numpy as np
from run_phase1c import rollout, intervention_value, train_psr, train_rnn_encoder, rnn_encode

def test_null_future_ignores_history_given_same_state():
    xs=np.array([0.,0.1,0.2,0.3]); u=np.zeros(3)
    a=intervention_value("null_markov",xs,u,1.8,"two_pulse")
    b=intervention_value("null_markov",xs,np.ones(3),1.8,"two_pulse")
    assert np.isclose(a,b)

def test_delay_is_history_sensitive():
    u1=np.array([1.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.])
    u2=np.array([0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.])
    x1=rollout("adversarial_delay",0.,u1); x2=rollout("adversarial_delay",0.,u2)
    assert not np.isclose(intervention_value("adversarial_delay",x1,u1,1.8,"two_pulse"),intervention_value("adversarial_delay",x2,u2,1.8,"two_pulse"))

def test_psr_is_learned_from_data():
    rng=np.random.default_rng(3); acts=rng.choice([-1.,0.,1.],size=(20,12)); seqs=np.array([rollout("adversarial_delay",0.1,a) for a in acts]); fut=np.array([rollout("adversarial_delay",0.1,np.r_[a,np.zeros(4)])[-4:] for a in acts])
    mu,B,Z=train_psr(seqs,acts,fut,latent_dim=4)
    assert B.shape[1]==4 and Z.shape==(20,4) and np.isfinite(Z).all()

def test_rnn_encoder_learns_and_encodes():
    rng=np.random.default_rng(4); acts=rng.choice([-1.,0.,1.],size=(24,12)); seqs=np.array([rollout("adversarial_switching",0.2,a) for a in acts]); pars=train_rnn_encoder(seqs,acts,hidden=5,epochs=4,seed=4); Z=rnn_encode(seqs,acts,pars)
    assert Z.shape==(24,5) and np.std(Z)>0
