"""Consumer-visible historical assessment, not tests of CLI wiring."""
import numpy as np
import pandas as pd
import pytest
from src.models.ev_approach_models import C0MarketHead, C0_RECIPES
from src.models.c0_betting_price_head import fit_price_head
from src.analysis.c0_betting_development_evaluation import evaluate_development


def heads():
    result={recipe:C0MarketHead(recipe, (.5,.5) if recipe=='Fixed50' else
        ((0.,1.) if recipe=='H' else (1.,1.) if recipe=='M-shape' else (1.,)),
        0.,{'success':True},{}) for recipe in C0_RECIPES}
    training=[]
    for q in (.1,.2,.8,.9):
        for index in range(10):
            for z in (-1.,1.):
                for e in (-1.,1.):
                    training.append({'q':q,'p_C0':1/(1+np.exp(-z)),
                        'odds_a':1.01+np.exp(-e/2),'odds_b':1.01+np.exp(e/2),
                        'y':int(index<int(q*10))})
    training=pd.DataFrame(training)
    result['M-price']=fit_price_head('M-price',training,regularization=.1)
    result['H-price']=fit_price_head('H-price',training,regularization=.1)
    return result


def panel():
    return pd.DataFrame({'row_id':['a','b','c','d'],'month':['2026-07','2026-07','2026-08','2026-09'],
        'p_C0':[.7]*4,'q':[.2]*4,'y':[0,1,0,1], 'odds_a':[5.]*4,'odds_b':[1.3]*4,
        'native_admitted':[True]*4,'financial_eligible':[True]*4,
        'quotes':[[{'bookmaker':'fixture','quote_id':'synthetic','odds_a':5.,'odds_b':1.3}] for _ in range(4)]})


def test_fixed_price_selection_retains_cash_and_decomposes_probability_error():
    result=evaluate_development(panel(),heads(),draws=200,seed=20261006)
    selected=result['fixed_S_selection']['H-price']
    assert selected['count']==4
    assert selected['overstatement']['point']==pytest.approx(.88*5*(.2-.5))
    assert selected['model_market_component']['point']==pytest.approx(0.)
    assert selected['market_residual']['point']==pytest.approx(.88*5*(.2-.5))
    assert selected['fixed_realized_return']==pytest.approx(.88*5*.5-1)
    assert result['metrics']['H-price']['selected_count']==0
    assert result['metrics']['H-price']['ROI']['point'] is None
    assert result['decision']=='NOT_ADOPTED'
    assert result['primary_contrasts']['M-price']['delta_LL']==pytest.approx(0.)


def test_sparse_selected_months_never_discard_undefined_draws():
    frame=panel()
    frame.loc[frame.month!='2026-07','p_C0']=.2
    result=evaluate_development(frame,heads(),draws=500,seed=20261006)
    metric=result['metrics']['S']['overstatement']
    assert metric['interval'] is None
    assert metric['undefined_draws']>0
    assert metric['equal_month_interval'] is None
    assert result['fixed_S_selection']['H-price']['overstatement']['interval'] is None


def test_price_absence_retains_native_denominator_and_never_creates_a_bet():
    frame=panel()
    frame.loc[3,['q','odds_a','odds_b']]=np.nan
    frame.at[3,'quotes']=[]
    result=evaluate_development(frame,heads(),draws=100)
    assert result['metrics']['S']['coverage_count']==4
    assert result['metrics']['H-price']['coverage_count']==3
    assert result['metrics']['H-price']['common_count']==3
    assert result['metrics']['S']['financial_eligible_count']==4
    assert result['metrics']['S']['profit_per_opportunity']['point']==pytest.approx((.88*5-3)/4)


def test_undefined_draw_count_is_actual_not_a_sparse_support_placeholder():
    frame=panel()
    frame.loc[frame.month!='2026-07','p_C0']=.2
    result=evaluate_development(frame,heads(),draws=500,seed=20261006)
    samples=np.random.default_rng(20261006).integers(0,3,(500,3))
    actual_undefined=int(np.all(samples!=0,axis=1).sum())
    assert result['metrics']['S']['overstatement']['undefined_draws']==actual_undefined
    assert result['metrics']['S']['overstatement']['interval'] is None
