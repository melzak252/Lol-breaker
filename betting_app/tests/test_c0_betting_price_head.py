"""Historical price-aware probabilities: symmetry, support and artifact safety."""
import copy
import numpy as np
import pandas as pd
import pytest
from scipy.special import expit, logit
from scipy.optimize import minimize
from src.models.c0_betting_price_head import C0PriceHead, fit_price_head, price_features
from src.models.ev_approach_models import select_c0_market_head


def population():
    rng = np.random.default_rng(20261006)
    m, z, e = rng.normal(size=(3, 400))
    frame = pd.DataFrame({'row_id':np.arange(400).astype(str), 'q':expit(m),
        'p_C0':expit(z), 'odds_a':1.01+np.exp(-e/2), 'odds_b':1.01+np.exp(e/2)})
    frame['y'] = rng.binomial(1, expit(.35*z+.7*m+.3*np.log(frame.odds_b/frame.odds_a)))
    frame.attrs['c0_bindings'] = {'scope':'historical_development_only','protocol_sha256':'synthetic'}
    return frame


@pytest.mark.parametrize('recipe',['M-price','H-price'])
def test_price_head_independent_optimum_and_complement(recipe):
    frame = population()
    head = fit_price_head(recipe,frame,regularization=.1)
    x, y = price_features(frame,recipe), frame.y.to_numpy(float)
    center = np.array([1.,1.,0.] + ([0.] if recipe=='H-price' else []))
    def objective(theta):
        eta=x@theta
        return np.mean(np.logaddexp(0,eta)-y*eta)+.05*np.sum((theta-center)**2)
    bounds=[(0,None),(0,None),(None,None)]+([(0,None)] if recipe=='H-price' else [])
    oracle=minimize(objective,center,method='SLSQP',bounds=bounds,options={'ftol':1e-12,'maxiter':2000})
    assert oracle.success
    assert objective(np.array(head.coefficients)) == pytest.approx(oracle.fun,abs=1e-8)
    swapped=frame.assign(q=1-frame.q,p_C0=1-frame.p_C0,odds_a=frame.odds_b,odds_b=frame.odds_a,y=1-frame.y)
    np.testing.assert_allclose(head.predict(frame)+head.predict(swapped),1,atol=1e-14)
    np.testing.assert_array_equal(head.predict(frame), C0PriceHead.from_dict(head.to_dict()).predict(frame))
    changed=frame.copy()
    changed['odds_a']=frame.odds_a*1.25
    assert np.max(abs(head.predict(frame)-head.predict(changed))) > 1e-5


@pytest.mark.parametrize('column,value',[('odds_a',True),('odds_b',1.),('odds_a',np.inf),('q',-.1),('p_C0',True)])
def test_invalid_price_or_probability_is_not_silently_usable(column,value):
    frame=population()
    frame[column]=frame[column].astype(object)
    frame.loc[0,column]=value
    with pytest.raises(ValueError):
        fit_price_head('H-price',frame,regularization=.1)


def test_market_control_does_not_require_native_and_degenerate_support_fails():
    frame=population().drop(columns='p_C0')
    assert np.isfinite(fit_price_head('M-price',frame,regularization=.1).predict(frame)).all()
    with pytest.raises(ValueError,match='INSUFFICIENT_FIT_SUPPORT'):
        fit_price_head('M-price',frame.assign(q=.5),regularization=.1)
    with pytest.raises(ValueError,match='INSUFFICIENT_FIT_SUPPORT'):
        fit_price_head('M-price',frame.assign(y=0),regularization=.1)


def test_cal_selection_does_not_refit_and_artifact_rejects_changed_contract():
    frame=population()
    heads=[fit_price_head('H-price',frame.iloc[:250],regularization=value) for value in (.01,.1,1.)]
    selected,_=select_c0_market_head(heads,frame.iloc[250:])
    assert any(selected is head for head in heads)
    for key,value in [('native_identity','EXP081'),('mode','no_history'),('epsilon',.01),('coefficients',[0.])]:
        broken=copy.deepcopy(selected.to_dict())
        broken[key]=value
        with pytest.raises(ValueError):
            C0PriceHead.from_dict(broken)
