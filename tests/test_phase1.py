import numpy as np
from hcvl.generators import rollout_history, intervention_value
from hcvl.state_matching import matched
from hcvl.estimands import history_features

def test_null_is_markov_under_same_current_state_and_probe():
    xs=np.array([0.0,0.2,-0.1,0.3])
    v1=intervention_value("null0_markov",xs,np.array([9.0,-9.0]),1.5)
    v2=intervention_value("null0_markov",xs,np.array([-7.0,7.0]),1.5)
    assert np.isclose(v1,v2)

def test_order_generator_has_order_memory():
    ab=np.array([0,0,0,0,1,0,-.6,0.])
    ba=np.array([0,0,0,0,-.6,0,1,0.])
    xa,ma=rollout_history("positive2_order",0.0,ab)
    xb,mb=rollout_history("positive2_order",0.0,ba)
    assert not np.isclose(ma[1]-ma[0], mb[1]-mb[0])

def test_matching_threshold():
    assert matched(0.1,0.7,0.75)
    assert not matched(0.1,1.0,0.75)

def test_history_is_low_dimensional():
    xs=np.arange(13,dtype=float)
    us=np.zeros(12)
    assert history_features(xs,us).shape==(5,)
