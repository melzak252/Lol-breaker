"""Restore full thesis model evidence without changing frozen model selection."""
from pathlib import Path
import argparse,json,hashlib,sys
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from scipy.special import expit,logit
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.impute import SimpleImputer
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.analysis.thesis_models import positive_slope_calibration
from src.analysis.thesis_metrics import _metrics
from src.analysis.thesis_evaluation import block_difference
from src.analysis.metrics import binary_log_loss_vector

METHODS=['identity','positive_slope','platt','isotonic','symmetric_platt','symmetric_isotonic']
LABELS={'identity':'Symetryczny surowy','positive_slope':'Dodatni logit bez wyrazu wolnego',
        'platt':'Platt (bez ograniczenia symetrii)','isotonic':'Izotoniczna (bez ograniczenia symetrii)',
        'symmetric_platt':'Platt + symetryzacja odwzorowania','symmetric_isotonic':'Izotoniczna + symetryzacja odwzorowania'}
FAMILY={'elo':'Elo','gl':'Glicko-2','ts':'TrueSkill','os':'OpenSkill','pl':'Plackett--Luce','tm':'Thurstone--Mosteller'}

def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def symmetric_map(mapping,p):
    p=np.asarray(p,float)
    return .5*(mapping(p)+1-mapping(1-p))

def calibration_replay(data,min_samples=200):
    d=data.copy();d['date']=pd.to_datetime(d.date);d['result_day']=pd.to_datetime(d.result_day)
    if d.golgg_match_id.duplicated().any():raise ValueError('Duplicate calibration match ID')
    rows=[];receipts=[]
    for month,test in d.groupby(d.date.dt.to_period('M'),sort=True):
        origin=month.start_time
        prior=d[(d.date<origin)&(d.result_day<origin)&(d.date>=pd.Timestamp('2024-01-01'))]
        available=len(prior)>=min_samples and prior.y_true.nunique()==2
        maps={m:lambda p:np.asarray(p,float) for m in METHODS}
        receipt={'origin':str(origin.date()),'cal_rows':len(prior),'cal_ids':prior.golgg_match_id.tolist(),'available':bool(available)}
        if available:
            p=np.clip(prior.p_raw.to_numpy(float),1e-6,1-1e-6);y=prior.y_true.to_numpy(int)
            slope=positive_slope_calibration(p,y)
            platt=LogisticRegression(C=1.,solver='lbfgs',max_iter=1000).fit(logit(p).reshape(-1,1),y)
            iso=IsotonicRegression(out_of_bounds='clip',y_min=1e-6,y_max=1-1e-6).fit(p,y)
            pg=lambda q:platt.predict_proba(logit(np.clip(q,1e-6,1-1e-6)).reshape(-1,1))[:,1]
            ig=lambda q:iso.predict(np.asarray(q,float))
            maps.update(positive_slope=lambda q:expit(slope*logit(np.clip(q,1e-6,1-1e-6))),
                        platt=pg,isotonic=ig,symmetric_platt=lambda q:symmetric_map(pg,q),
                        symmetric_isotonic=lambda q:symmetric_map(ig,q))
            receipt.update(slope=slope,platt_coef=float(platt.coef_[0,0]),platt_intercept=float(platt.intercept_[0]),
                           iso_x=iso.X_thresholds_.tolist(),iso_y=iso.y_thresholds_.tolist())
        receipts.append(receipt)
        for method,mapping in maps.items():
            out=test[['golgg_match_id','date','result_day','y_true']].copy();p=test.p_raw.to_numpy(float)
            out['method']=method;out['probability']=mapping(p)
            out['symmetry_error']=abs(mapping(p)+mapping(1-p)-1);out['origin']=str(origin.date());rows.append(out)
    result=pd.concat(rows,ignore_index=True)
    source=d.set_index('golgg_match_id')
    for method,group in result.groupby('method'):
        if group.golgg_match_id.duplicated().any() or set(group.golgg_match_id)!=set(source.index):
            raise ValueError(f'Calibration coverage mismatch: {method}')
        aligned=source.loc[group.golgg_match_id]
        for column in ('date','result_day','y_true'):
            if not np.array_equal(group[column].to_numpy(),aligned[column].to_numpy()):
                raise ValueError(f'Calibration identity metadata mismatch: {method}/{column}')
    return result,receipts

def fmt(x,n=6):
    return r'$\infty$' if not np.isfinite(float(x)) else f'{float(x):.{n}f}'.replace('.',r'{,}')

def table(caption,label,headers,rows):
    return '\n'.join([r'\begin{table}[htbp]',r'\centering\small',r'\caption{'+caption+'}',r'\label{'+label+'}',
        r'\begin{tabular}{'+'l'+'r'*(len(headers)-1)+'}',r'\toprule',' & '.join(headers)+r' \\',r'\midrule',
        *[' & '.join(map(str,r))+r' \\' for r in rows],r'\bottomrule',r'\end{tabular}',r'\end{table}',''])

def main():
    ap=argparse.ArgumentParser()
    for x in ('evaluation','models','output','features','thesis'):ap.add_argument('--'+x,type=Path,required=True)
    a=ap.parse_args();evaluation=a.evaluation;output=a.output;output.mkdir(parents=True,exist_ok=False)
    figures=a.thesis/'figures/thesis_full_20260921';figures.mkdir(parents=True,exist_ok=True)
    generated=a.thesis/'generated';generated.mkdir(exist_ok=True)
    modelroot=a.models
    inputs=[a.features,evaluation/'paired_predictions.parquet',evaluation/'metrics.csv',evaluation/'comparisons.csv',evaluation/'controlled_comparisons.csv',Path(__file__),
            *[modelroot/f/'predictions.parquet' for f in ('mlp','logistic')]]
    protocol={'recorded_at':datetime.now(timezone.utc).isoformat(),'sources':{str(p):sha(p) for p in inputs},
      'correlation_vif':'2023 dates+released before2024; median imputer fit on dates+released before2023;12 rawrating signals only; no featureselection',
      'calibration':'MLP and LR saved symmetric raw OOS2024+; monthlystrict earlierdate+release;min200; sixfixedmethods; original positive slope remains primary; no promotion based on diagnosticresults',
      'methods':METHODS,'symmetry_policy':'Standard Platt/isotonic labeled diagnostic violations; symmetric variants use0.5*(g(p)+1-g(1-p)).',
      'bootstrap':'5000pairedmonthlyresamples; candidate minus originalpositive-slope; unadjusted exploratoryintervals'}
    (output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    m=pd.read_csv(evaluation/'metrics.csv');c=pd.concat([pd.read_csv(evaluation/'comparisons.csv'),pd.read_csv(evaluation/'controlled_comparisons.csv')]).drop_duplicates(['cohort','candidate','reference','metric','blocking'])
    panel=pd.read_parquet(evaluation/'paired_predictions.parquet');marketids=set(panel.loc[panel.market_open.notna()&panel.market_close.notna(),'golgg_match_id'].astype(str))
    def metric(model,cohort='sports'):return m[(m.model==model)&(m.cohort==cohort)].iloc[0]
    def ci(candidate,reference='p_logistic'):
        matching=c[(c.cohort=='sports')&(c.candidate==candidate)&(c.reference==reference)&(c.metric=='logloss')&(c.blocking=='month')]
        if len(matching):return matching.iloc[0]
        delta=binary_log_loss_vector(panel.y_true,panel[candidate])-binary_log_loss_vector(panel.y_true,panel[reference])
        return pd.Series(block_difference(delta,pd.to_datetime(panel.date).dt.strftime('%Y-%m')))
    def save(name):
        plt.tight_layout();plt.savefig(figures/(name+'.png'),dpi=180,bbox_inches='tight');plt.savefig(figures/(name+'.pdf'),bbox_inches='tight');plt.close()
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    rows=[]
    for f,label in FAMILY.items():
        for scope in ('player','team'):
            r=metric('p_'+scope+'_'+f);q=metric('p_'+scope+'_'+f,'market_common')
            rows.append([('Player ' if scope=='player' else 'Team ')+label,fmt(r.logloss),fmt(r.brier),fmt(q.auc,4),fmt(q.logloss),fmt(100*q.ece,2)])
    (generated/'full_models_ratings.tex').write_text(table('Ratingi z tym samym schematem projekcji i kalibracji. Sport: 11 550; rynek: 3229. ECE w pp.', 'tab:ranking_common_sample_results', ['Rating','LL sport','Brier sport','AUC rynek','LL rynek','ECE rynek'],rows))
    x=np.arange(6);plt.figure(figsize=(9,4.5))
    for shift,scope,label in [(-.18,'player','Zawodnicy'),(.18,'team','Organizacja')]:
        plt.bar(x+shift,[metric('p_'+scope+'_'+f).auc for f in FAMILY],width=.36,label=label)
    plt.xticks(x,list(FAMILY.values()),rotation=18);plt.ylabel('AUC');plt.ylim(.5,.85);plt.legend();plt.title('Ratingi po projekcji i kalibracji; 11 550 serii');save('full_models_rating_auc')
    labels=['Player '+FAMILY[f] for f in FAMILY]+['OPEN','CLOSE'];vals=[metric('p_player_'+f,'market_common').auc for f in FAMILY]+[metric('market_'+h,'market_common').auc for h in ('open','close')]
    plt.figure(figsize=(9,4.5));plt.barh(labels,vals);plt.xlim(.5,.85);plt.xlabel('AUC');plt.title('Ratingi po projekcji i kalibracji oraz rynek; 3229 serii');save('full_models_market_auc')
    rel=pd.read_csv(evaluation/'reliability.csv');plt.figure(figsize=(7,5));plt.plot([0,1],[0,1],'--',color='gray')
    for model,label in [('p_player_gl','Player Glicko-2'),('p_player_elo','Player Elo'),('market_open','OPEN'),('market_close','CLOSE')]:
        z=rel[(rel.cohort=='market_common')&(rel.model==model)];plt.plot(z.mean_probability,z.observed_rate,'o-',label=label)
    plt.xlabel('Średnie prawdopodobieństwo');plt.ylabel('Częstość zwycięstwa');plt.legend();plt.title('Kalibracja; wspólne 3229 serii, 10 przedziałów');save('full_models_rating_calibration')
    features=pd.read_parquet(a.features);dates=pd.to_datetime(features.date);release=pd.to_datetime(features.result_day)
    cols=[scope+'_'+f for scope in ('player','team') for f in FAMILY]
    train=features[(dates<'2023-01-01')&(release<'2023-01-01')];dev=features[(dates>='2023-01-01')&(dates<'2024-01-01')&(release<'2024-01-01')]
    values=SimpleImputer(strategy='median',keep_empty_features=True).fit(train[cols]).transform(dev[cols]);corr=pd.DataFrame(values,columns=cols).corr();corr.to_csv(output/'correlations_2023.csv')
    z=(values-values.mean(axis=0))/values.std(axis=0);vifs=[]
    for i,col in enumerate(cols):
        design=np.c_[np.ones(len(z)),np.delete(z,i,axis=1)];residual=z[:,i]-design@np.linalg.lstsq(design,z[:,i],rcond=None)[0]
        unexplained=float(np.sum(residual**2)/np.sum(z[:,i]**2));vifs.append(dict(feature=col,vif=np.inf if unexplained<1e-12 else 1/unexplained))
    vif=pd.DataFrame(vifs).sort_values('vif',ascending=False);vif.to_csv(output/'vif_2023.csv',index=False)
    (generated/'full_models_vif.tex').write_text(table('VIF 12 sygnałów ratingowych na 4836 meczach rozwojowych 2023; imputacja wyłącznie z wcześniejszego TRAIN.', 'tab:rating_signal_vif',['Sygnał','VIF'],[[r.feature.replace('_',r'\_'),fmt(r.vif,2)] for r in vif.itertuples()]))
    plt.figure(figsize=(9,8));plt.imshow(corr,vmin=0,vmax=1,cmap='viridis');plt.colorbar(label='Korelacja Pearsona');plt.xticks(range(12),cols,rotation=75,ha='right');plt.yticks(range(12),cols);plt.title('Współzależność sygnałów; tylko 2023, N = 4836');save('full_models_correlation')
    groups={
      'window':([('W3','p_lr_w3'),('W5','p_lr_w5'),('W10','p_lr_w10'),('W20','p_logistic'),('W50','p_lr_w50')],'tab:metamodel_window_ablation'),
      'families':([('LR ElasticNet','p_logistic'),('ExtraTrees','p_extra_trees'),('LightGBM','p_lightgbm'),('HGB','p_hist_gradient_boosting'),('MLP','p_mlp')],'tab:w20_binomial_model_family_comparison'),
      'masking':([('LR pełny','p_logistic'),('Glicko + kontekst','p_lr_glicko_context'),(r'Mask baza 10\%','p_supp_mask_base10'),(r'Mask pGL 10\%','p_supp_mask_playergl10'),(r'Mask pGL 30\%','p_supp_mask_playergl30'),('Bez pGL','p_lr_no_player_gl'),('Bez Glicko','p_lr_no_glicko')],'tab:logistic_masking_ablation'),
      'team':([('Player (46 cech)','p_logistic'),('Player+Team (58 cech)','p_lr_player_team'),('Player+Team (80 cech)','p_supp_full_player_team')],'tab:player_team_ablation')}
    for kind,(items,label) in groups.items():
        rows=[];deltas=[]
        for title,model in items:
            r=metric(model);rows.append([title,fmt(r.auc,4),fmt(r.logloss),fmt(r.brier),fmt(100*r.ece,2)])
            if model!='p_logistic':d=ci(model);deltas.append((title,d.delta,d.ci_lower,d.ci_upper))
        (generated/f'full_models_{kind}.tex').write_text(table('Porównanie na tej samej pełnej próbie 11 550 serii; ECE w pp. Parametry i kalibracja ustalone przed oceną.',label,['Wariant','AUC','LogLoss','Brier','ECE'],rows))
        if kind in ('window','families','masking'):
            plt.figure(figsize=(9,4.5));y=np.arange(len(deltas))
            plt.errorbar([d[1] for d in deltas],y,xerr=[[d[1]-d[2] for d in deltas],[d[3]-d[1] for d in deltas]],fmt='o',capsize=4)
            plt.axvline(0,color='gray',ls='--');plt.yticks(y,[d[0].replace('\\%','%') for d in deltas]);plt.xlabel('Δ LogLoss: wariant − LR W20 (mniej jest lepiej)');plt.title('Miesięczny bootstrap: 5000 replik, 33 bloki');save(f'full_models_{kind}_bootstrap')
    allpred=[];allfits={};metrics=[];contrasts=[];symrows=[]
    for family in ('mlp','logistic'):
        source=pd.read_parquet(modelroot/family/'predictions.parquet');pred,receipts=calibration_replay(source);pred['family']=family;allpred.append(pred);allfits[family]=receipts
        source=source.set_index('golgg_match_id');current=pred[pred.method=='positive_slope'].set_index('golgg_match_id')
        if not np.allclose(current.probability,source.loc[current.index].p_calibrated,atol=1e-12,rtol=0):raise ValueError('Original primary calibration reproduction failed')
        rawdiff=abs(source.p_original+source.p_swapped-1);symrows.append(dict(family=family,mean=float(rawdiff.mean()),median=float(rawdiff.median()),maximum=float(rawdiff.max())))
        for cohort,frame in [('sports',pred),('market_common',pred[pred.golgg_match_id.astype(str).isin(marketids)])]:
            base=frame[frame.method=='positive_slope'].set_index('golgg_match_id')
            for method,group in frame.groupby('method',sort=False):
                score,_=_metrics(group.y_true.to_numpy(),group.probability.to_numpy());metrics.append(dict(family=family,cohort=cohort,method=method,max_symmetry_error=float(group.symmetry_error.max()),**score))
                loss=binary_log_loss_vector(group.y_true,group.probability);ref=base.loc[group.golgg_match_id]
                delta=loss-binary_log_loss_vector(ref.y_true,ref.probability);contrasts.append(dict(family=family,cohort=cohort,method=method,**block_difference(delta,group.date.dt.strftime('%Y-%m'))))
    pd.concat(allpred).to_parquet(output/'calibration_predictions.parquet',index=False);pd.DataFrame(metrics).to_csv(output/'calibration_metrics.csv',index=False);pd.DataFrame(contrasts).to_csv(output/'calibration_contrasts.csv',index=False)
    (output/'calibration_fit_receipts.json').write_text(json.dumps(allfits,separators=(',',':')));pd.DataFrame(symrows).to_csv(output/'raw_symmetry.csv',index=False)
    cm=pd.DataFrame(metrics);rows=[]
    for method in METHODS:
        r=cm[(cm.family=='logistic')&(cm.cohort=='market_common')&(cm.method==method)].iloc[0]
        rows.append([LABELS[method],fmt(r.auc,4),fmt(r.logloss),fmt(r.brier),fmt(100*r.ece,2)])
    (generated/'full_models_calibration.tex').write_text(table('Diagnostyka kalibracji ElasticNet; te same 3229 serii rynkowych. ECE w pp; brak ponownego wyboru metody.', 'tab:symmetry_calibration_comparison',['Wariant','AUC','LogLoss','Brier','ECE'],rows))
    for metricname,ylabel in [('logloss','LogLoss'),('ece','ECE [pp]')]:
        sub=cm[(cm.family=='logistic')&(cm.cohort=='market_common')].set_index('method').loc[METHODS]
        plt.figure(figsize=(10,5));plt.barh([LABELS[x] for x in METHODS],sub[metricname]*(100 if metricname=='ece' else 1));plt.xlabel(ylabel);plt.title('ElasticNet: diagnostyka metod kalibracji; 3229 tych samych serii');save('full_models_calibration_'+metricname)
    (output/'COMPLETE.json').write_text(json.dumps({'status':'COMPLETE','dev_rows':len(dev),'train_rows':len(train),'methods':METHODS,'fits_per_family':len(allfits['mlp']),'unchanged_primary':True,'source_hashes_unchanged':all(sha(p)==v for p,v in protocol['sources'].items())},indent=2)+'\n')

if __name__=='__main__':main()
