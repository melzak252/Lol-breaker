"""Probability-score and descriptive calibration diagnostics for the thesis."""
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit
from sklearn.metrics import roc_auc_score
from src.analysis.metrics import binary_log_loss_vector


def _metrics(y, p):
    """Ten equal-width bins; endpoints included; proper scores use raw probabilities."""
    n=len(y); bins=np.minimum((p*10).astype(int),9); mean=float(y.mean())
    table=[]
    for b in range(10):
        mask=bins==b
        if mask.any():
            table.append({'bin':b,'lower':b/10,'upper':(b+1)/10,'n':int(mask.sum()),
                          'mean_probability':float(p[mask].mean()),'observed_rate':float(y[mask].mean())})
    reliability=sum(r['n']/n*(r['mean_probability']-r['observed_rate'])**2 for r in table)
    resolution=sum(r['n']/n*(r['observed_rate']-mean)**2 for r in table)
    uncertainty=mean*(1-mean); brier=float(np.mean((p-y)**2))
    slope=intercept=float('nan')
    # A calibration fit is descriptive, not a recalibration of predictions.
    if len(np.unique(y))==2 and np.std(p)>1e-12:
        x=logit(np.clip(p,1e-15,1-1e-15))
        def objective(theta):
            z=theta[0]+theta[1]*x
            return np.mean(np.logaddexp(0,z)-y*z)
        def gradient(theta):
            e=expit(theta[0]+theta[1]*x)-y
            return np.array([e.mean(),np.mean(e*x)])
        fitted=minimize(objective,[0.,1.],jac=gradient,method='BFGS')
        # Diverging separation fits do not provide a finite identifiable slope.
        if fitted.success and np.max(np.abs(fitted.x))<30:
            intercept,slope=map(float,fitted.x)
    return dict(n=n,logloss=float(binary_log_loss_vector(y,p).mean()),brier=brier,
                auc=float(roc_auc_score(y,p)) if len(np.unique(y))==2 else float('nan'),
                accuracy=float(np.mean((p>=.5)==y)),
                ece=sum(r['n']/n*abs(r['mean_probability']-r['observed_rate']) for r in table),
                mce=max(abs(r['mean_probability']-r['observed_rate']) for r in table),
                calibration_slope=slope,calibration_intercept=intercept,
                reliability=reliability,resolution=resolution,uncertainty=uncertainty,
                binning_residual=brier-(reliability-resolution+uncertainty)),table
