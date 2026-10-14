"""Regenerate full thesis descriptive evidence from pinned corrected inputs.

No model selection or fitting uses 2024+ descriptive outcomes. Historical GD15
relationships are restricted to released 2023 development labels. Tournament
names are reported literally, never promoted into unverified official tiers.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from scipy.stats import pearsonr,spearmanr
from sklearn.metrics import log_loss,brier_score_loss,roc_auc_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.analysis.thesis_figures import style

BOOKS={'betclic':'Betclic','betfan':'Betfan','efortuna':'eFortuna','lv_bet':'LV BET','sts':'STS','superbet':'Superbet','fuksiarz':'Fuksiarz'}

def orient_market(panel, odds, identities):
    """Align every quote pair to canonical target IDs, without examining outcomes."""
    p=panel.copy();o=odds.copy();i=identities.copy()
    for d in [p,o]:d['golgg_match_id']=d.golgg_match_id.astype(str)
    i['golgg_match_id']=i.match_id.astype(str)
    i=i.rename(columns={'team1_id':'source_team1_id','team2_id':'source_team2_id'})
    d=p.merge(o,on='golgg_match_id',how='inner',validate='one_to_one').merge(i[['golgg_match_id','source_team1_id','source_team2_id']],on='golgg_match_id',validate='one_to_one')
    same=(d.team1_id.astype(str)==d.source_team1_id.astype(str))&(d.team2_id.astype(str)==d.source_team2_id.astype(str))
    swapped=(d.team1_id.astype(str)==d.source_team2_id.astype(str))&(d.team2_id.astype(str)==d.source_team1_id.astype(str))
    if not (same|swapped).all():raise ValueError('Inconsistent market identity')
    pairs=[('avg_open_home','avg_open_away'),('avg_odds_home','avg_odds_away')]+[(f'odds1_{b}_{t}',f'odds2_{b}_{t}') for b in BOOKS for t in ['open','close']]
    for a,b in pairs:
        if a in d and b in d:d.loc[swapped,[a,b]]=d.loc[swapped,[b,a]].to_numpy()
    return d

def gd15_development(features):
    d=features[(features.date>='2023-01-01')&(features.date<'2024-01-01')&(features.result_day<'2024-01-01')].copy()
    d['delta']=d.t1_rolling_gd15_w20-d.t2_rolling_gd15_w20
    return d[np.isfinite(d.delta)].copy()

def quote_metrics(a,b,y):
    a,b,y=map(np.asarray,(a,b,y));valid=np.isfinite(a)&np.isfinite(b)&(a>1)&(b>1)
    a,b,y=a[valid],b[valid],y[valid]
    if not len(a):return {'n':0,'margin':np.nan,'auc':np.nan,'logloss':np.nan,'brier':np.nan,'ece':np.nan}
    p=b/(a+b);bins=np.minimum((p*10).astype(int),9)
    return {'n':len(y),'margin':float(np.mean(1/a+1/b-1)),'auc':float(roc_auc_score(y,p)) if len(np.unique(y))==2 else np.nan,'logloss':float(log_loss(y,p,labels=[0,1])),'brier':float(brier_score_loss(y,p)),'ece':float(sum(abs((p[bins==k]-y[bins==k]).sum())/len(y) for k in range(10)))}

def tex(s):
    return str(s).replace('\\',r'\textbackslash{}').replace('&',r'\&').replace('%',r'\%').replace('_',r'\_').replace('#',r'\#')
def num(x,places=0):
    if pd.isna(x):return '--'
    if places==0:return f'{int(x):,}'.replace(',','~')
    return f'{x:.{places}f}'.replace('.',r'{,}')
def table(path,caption,label,headers,rows,align=None):
    align=align or 'l'+'r'*(len(headers)-1)
    body=['\\begin{table}[H]','\\centering','\\small',f'\\caption{{{caption}}}',f'\\label{{{label}}}',r'\resizebox{\textwidth}{!}{%',f'\\begin{{tabular}}{{{align}}}',r'\toprule',' & '.join(headers)+r' \\',r'\midrule']
    body+=[' & '.join(row)+r' \\' for row in rows]
    body += [r'\bottomrule',r'\end{tabular}',r'}',r'\end{table}']
    path.write_text('\n'.join(body)+'\n')
def save(fig,folder,name):
    fig.tight_layout();fig.savefig(folder/(name+'.png'),dpi=180,bbox_inches='tight');fig.savefig(folder/(name+'.pdf'),bbox_inches='tight');plt.close(fig)
def sha(p):
    with Path(p).open('rb') as h:return hashlib.file_digest(h,'sha256').hexdigest()

def plot_gd15_development(bins, figures):
    """Present unchanged development-bin statistics in the original visual style."""
    fig,ax=plt.subplots(figsize=(10,6))
    ax.set_axisbelow(True)
    ax.grid(color='#e2e8f0',linestyle='--',linewidth=.8)
    ax.axhline(50,color='#717784',linestyle='--',linewidth=1.2)
    ax.axvline(0,color='#717784',linestyle=':',linewidth=1.3)
    ax.plot(bins.delta,100*bins.win_rate,color='#505660',linewidth=2.2,
            label='Odsetek zwycięstw w przedziałach GD@15',zorder=2)
    points=ax.scatter(bins.delta,100*bins.win_rate,s=40+bins.n*.35,
                      c=100*bins.win_rate,cmap='YlGnBu',vmin=20,vmax=80,
                      edgecolors='white',linewidth=1.2,alpha=.95,zorder=3)
    for row in bins.itertuples():
        ax.annotate(f'n = {row.n}',(row.delta,100*row.win_rate),
                    xytext=(0,15),textcoords='offset points',ha='center',fontsize=9,color='#505660')
    ax.set(xlabel='Różnica średniego historycznego GD@15 (W20): Team 1 − Team 2',
           ylabel='Rzeczywisty odsetek zwycięstw Team 1 [%]',ylim=(0,100))
    ax.set_title('Historyczne GD@15 a odsetek zwycięstw Team 1',fontweight='bold',pad=14)
    ax.legend(loc='upper left',frameon=False,fontsize=9)
    fig.colorbar(points,ax=ax,pad=.025,label='Odsetek zwycięstw w przedziale [%]')
    save(fig,figures,'full_eda_gd15_development')

def plot_target_formats(counts, figures, out):
    """Compare format shares within each cohort, keeping counts for provenance."""
    percentages=counts.div(counts.sum(axis=0),axis=1)*100
    percentages.to_csv(out/'full_eda_target_formats_percent.csv')
    fig,ax=plt.subplots(figsize=(8,4))
    percentages.plot.bar(ax=ax,color=['#2376A8','#C87529','#598447'])
    ax.set_xticklabels(['Bo'+str(i) for i in counts.index],rotation=0)
    ax.set(xlabel='Format',ylabel='Udział serii w kohorcie [%]',
           title='Udział formatów serii w trzech kohortach',ylim=(0,75))
    for bars in ax.containers:
        ax.bar_label(bars,labels=[f'{v:.1f}%'.replace('.',',') for v in bars.datavalues],padding=3,fontsize=8)
    ax.legend(frameon=False)
    save(fig,figures,'full_eda_target_formats')

def build(raw,feature_path,panel_path,odds_path,identity_path,out,figures,generated):
    out,figures,generated=map(Path,(out,figures,generated))
    for folder in (out,figures,generated):folder.mkdir(parents=True,exist_ok=True)
    style();f=pd.read_parquet(feature_path);panel=pd.read_parquet(panel_path);odds=pd.read_csv(odds_path);identities=pd.DataFrame(json.loads(Path(identity_path).read_text()))
    for d in (f,panel,odds):d['golgg_match_id']=d.golgg_match_id.astype(str)
    source_path=out/'full_eda_source_series.csv';map_path=out/'full_eda_source_maps_by_year.csv'
    if source_path.exists() or map_path.exists():raise FileExistsError('Choose an empty descriptive output directory to preserve evidence')
    from scripts.export_prospective_features import iter_matches
    source=[];map_years=Counter();fallback=0
    for r in iter_matches(Path(raw)):
        source.append({'golgg_match_id':str(r['match_id']),'date':r['date'],'best_of':int(r.get('best_of',r.get('BoN'))),'games':len(r['games']),'draw':bool(r.get('draw',False)),'tournament_name':r.get('tournament_name',r.get('tournament',''))})
        for game in r['games']:
            day=game.get('date') or r['date'];fallback+=int(not game.get('date'));map_years[str(day)[:4]]+=1
    source=pd.DataFrame(source);source.to_csv(source_path,index=False)
    maps=pd.DataFrame(sorted(map_years.items()),columns=['year','maps']);maps.to_csv(map_path,index=False)
    assert len(source)==41915 and maps.maps.sum()==69370 and len(f)==40636 and len(panel)==11550
    common=panel[panel.market_open.notna()&panel.market_close.notna()].copy();assert len(common)==3229
    yearly=source.groupby(source.date.str[:4]).agg(series=('golgg_match_id','size'),maps_assigned_to_series_start=('games','sum')).reset_index(names='year')
    yearly=yearly.merge(maps,on='year');yearly['feature_targets']=yearly.year.map(f.date.str[:4].value_counts()).fillna(0).astype(int);yearly['matched_odds']=yearly.year.map(odds.golgg_date.str[:4].value_counts()).fillna(0).astype(int);yearly.to_csv(out/'full_eda_yearly.csv',index=False)
    for col,name,title,ylabel in [('series','full_eda_series_by_year','Serie w źródle GOL.GG (N = 41 915)','Liczba serii'),('maps','full_eda_maps_by_year','Mapy w źródle GOL.GG (N = 69 370)','Liczba map')]:
        fig,ax=plt.subplots(figsize=(8,4));ax.bar(yearly.year,yearly[col],color='#2376A8');ax.set(title=title,ylabel=ylabel);ax.tick_params(axis='x',rotation=45);ax.grid(axis='y',alpha=.2);save(fig,figures,name)
    byformat=pd.crosstab(source.date.str[:4],source.best_of);byformat.to_csv(out/'full_eda_source_formats_by_year.csv');fig,ax=plt.subplots(figsize=(8,4.5));byformat.rename(columns=lambda n:'Bo'+str(n)).plot.bar(stacked=True,ax=ax,colormap='tab20');ax.set(title='Formaty wszystkich serii źródłowych',ylabel='Liczba serii',xlabel='Rok');ax.legend(title='Format',frameon=False,ncol=3);save(fig,figures,'full_eda_formats_by_year')
    balance=[]
    for label,d in [('Kohorta cech',f),('Ocena 2024–2026',panel),('Wspólna próba rynkowa',common)]:
        y=d['y_true'];balance.append({'cohort':label,'n':len(d),'wins1':int(y.sum()),'win_rate1':float(y.mean())})
    balance=pd.DataFrame(balance);balance.to_csv(out/'full_eda_class_balance.csv',index=False);fig,ax=plt.subplots(figsize=(8,4));x=np.arange(len(balance));ax.bar(x-.18,100*balance.win_rate1,.36,label='Strona 1',color='#2376A8');ax.bar(x+.18,100*(1-balance.win_rate1),.36,label='Strona 2',color='#C87529');ax.axhline(50,color='gray',ls='--');ax.set_xticks(x,[f'{r.cohort}\nN = {r.n:,}'.replace(',',' ') for r in balance.itertuples()]);ax.set(ylabel='Udział zwycięstw [%]',ylim=(0,65),title='Balans celu w kanonicznej orientacji identyfikatorów');ax.legend(frameon=False);save(fig,figures,'full_eda_class_balance')
    fm=pd.crosstab(f.best_of,np.repeat('Kohorta cech',len(f)))
    fm['Ocena 2024–2026']=panel.BoN.value_counts()
    fm['Wspólny rynek']=common.BoN.value_counts()
    fm.to_csv(out/'full_eda_target_formats.csv')
    plot_target_formats(fm,figures,out)
    # Literal labels: no unverified official tier or edition inference.
    tc=panel.tournament_name.fillna('(brak nazwy)').value_counts().rename('sports').to_frame();tc['market']=common.tournament_name.fillna('(brak nazwy)').value_counts();tc['market']=tc.market.fillna(0).astype(int);tc['coverage_pct']=100*tc.market/tc.sports;tc.to_csv(out/'full_eda_tournament_coverage.csv',index_label='tournament_name')
    top=tc.head(10).sort_values('sports');fig,ax=plt.subplots(figsize=(9,5));y=np.arange(len(top));ax.barh(y-.18,top.sports,.36,label='Kohorta sportowa',color='#2376A8');ax.barh(y+.18,top.market,.36,label='Wspólny rynek',color='#C87529');ax.set_yticks(y,top.index);ax.set(xlabel='Liczba serii',title='Dziesięć najliczniejszych nazw turniejów w ocenie 2024–2026');ax.legend(frameon=False);save(fig,figures,'full_eda_tournament_coverage')
    table(generated/'full_eda_tournaments.tex','Pokrycie nazw turniejów w kohorcie oceny 2024--2026; dziesięć największych grup. Nazwa źródłowa nie jest zweryfikowanym identyfikatorem edycji.','tab:eda_tier_coverage_common_sample',['Nazwa w GOL.GG','Sport','Rynek','Pokrycie [\\%]'],[[tex(n),num(r.sports),num(r.market),num(r.coverage_pct,1)] for n,r in tc.head(10).iterrows()])
    market=orient_market(common[['golgg_match_id','team1_id','team2_id','y_true','BoN']],odds,identities)
    market_rows=[]
    for period,a,b in [('OPEN','avg_open_home','avg_open_away'),('CLOSE','avg_odds_home','avg_odds_away')]:
        for bon,d in [('Wszystkie',market),*[(f'Bo{n}',market[market.BoN==n]) for n in [1,3,5]]]:market_rows.append({'format':bon,'period':period,**quote_metrics(d[a],d[b],d.y_true)})
    mm=pd.DataFrame(market_rows);mm.to_csv(out/'full_eda_market_by_format.csv',index=False)
    table(generated/'full_eda_market_overall.tex','Benchmark bukmacherski na tej samej próbie 3~229 meczów z lat 2024--2026.','tab:eda_market_benchmark',['Benchmark','N','AUC','LogLoss','Brier','ECE'],[[r.period,num(r.n),num(r.auc,4),num(r.logloss,4),num(r.brier,4),num(r.ece,4)] for r in mm[mm['format']=='Wszystkie'].itertuples()])
    table(generated/'full_eda_market_formats.tex','Jakość rynku według formatu, wspólna próba OPEN/CLOSE. ECE: 10 przedziałów o równej szerokości.','tab:eda_market_quality_by_bon',['Format','Benchmark','N','AUC','LogLoss','ECE'],[[r.format,r.period,num(r.n),num(r.auc,4),num(r.logloss,4),num(r.ece,4)] for r in mm[mm['format']!='Wszystkie'].sort_values(['format','period']).itertuples()],align='llrrrr')
    overall=mm[mm['format']=='Wszystkie'];fig,axes=plt.subplots(1,2,figsize=(8,3.6))
    for ax,col,title in [(axes[0],'auc','AUC — więcej oznacza lepiej'),(axes[1],'logloss','LogLoss — mniej oznacza lepiej')]:
        ax.bar(overall.period,overall[col],color=['#2376A8','#C87529']);ax.set_title(title);ax.set_ylim(0,1)
        for i,value in enumerate(overall[col]):ax.text(i,value+.015,f'{value:.4f}',ha='center')
    fig.suptitle('Rynek na wspólnej próbie N = 3 229');save(fig,figures,'full_eda_market_overall')
    chart=mm[mm['format']!='Wszystkie'].pivot(index='format',columns='period',values='logloss');fig,ax=plt.subplots(figsize=(7.5,4));chart[['OPEN','CLOSE']].plot.bar(ax=ax,color=['#2376A8','#C87529']);ax.set(title='LogLoss rynku według formatu',ylabel='LogLoss',xlabel='Format');ax.tick_params(axis='x',rotation=0);ax.set_ylim(0,.85);ax.legend(title='Benchmark',frameon=False,ncol=2,loc='upper center');save(fig,figures,'full_eda_market_by_format')
    books=[]
    for key,label in BOOKS.items():books.append({'bookmaker':label,**quote_metrics(market[f'odds1_{key}_open'],market[f'odds2_{key}_open'],market.y_true)})
    books.append({'bookmaker':'Średni kurs źródłowy',**quote_metrics(market.avg_open_home,market.avg_open_away,market.y_true)})
    bs=pd.DataFrame(books);bs['coverage_pct']=100*bs.n/len(market);bs.to_csv(out/'full_eda_bookmakers.csv',index=False)
    table(generated/'full_eda_bookmakers.tex','Pokrycie, średnia nadwyżka prawdopodobieństw (marża) i jakość kursów OPEN. Mianownik pokrycia: 3~229 meczów; jakość każdego operatora na jego dostępnych wierszach.','tab:bookmaker_margin_summary',['Bukmacher','N','Pokrycie [\\%]','Marża [\\%]','AUC','LogLoss'],[[tex(r.bookmaker),num(r.n),num(r.coverage_pct,1),num(100*r.margin,2),num(r.auc,4),num(r.logloss,4)] for r in bs.itertuples()])
    b=bs[bs.n>0];fig,axes=plt.subplots(1,2,figsize=(10,4.5));axes[0].barh(b.bookmaker,b.coverage_pct,color='#2376A8');axes[0].set(xlabel='Pokrycie wspólnej próby [%]',xlim=(0,105));axes[1].barh(b.bookmaker,100*b.margin,color='#C87529');axes[1].set_xlabel('Średnia marża OPEN [%]');fig.suptitle('Pokrycie i marża kursów; populacje operatorów różnią się');save(fig,figures,'full_eda_bookmaker_coverage_margins')
    ranges=[]
    for label,col in [('OPEN -- strona 1','avg_open_home'),('OPEN -- strona 2','avg_open_away'),('CLOSE -- strona 1','avg_odds_home'),('CLOSE -- strona 2','avg_odds_away')]:
        x=market[col];ranges.append({'variable':label,'n':len(x),'min':float(x.min()),'median':float(x.median()),'max':float(x.max())})
    rr=pd.DataFrame(ranges);rr.to_csv(out/'full_eda_odds_ranges.csv',index=False);table(generated/'full_eda_odds_ranges.tex','Zakres kursów średnich w finalnej wspólnej próbie 3~229 meczów, po orientacji do kanonicznych stron i odrzuceniu kursów nie większych od 1.','tab:odds_value_ranges',['Zmienna','N','Minimum','Mediana','Maksimum'],[[r.variable,num(r.n),num(r.min,2),num(r.median,2),num(r.max,2)] for r in rr.itertuples()])
    gd=gd15_development(f);gd[['golgg_match_id','date','result_day','delta','y_true']].to_csv(out/'full_eda_gd15_development_rows.csv',index=False)
    gd['bin']=pd.cut(gd.delta,[-np.inf,-2000,-1000,-500,0,500,1000,2000,np.inf],include_lowest=True)
    gb=gd.groupby('bin',observed=True).agg(n=('y_true','size'),delta=('delta','mean'),win_rate=('y_true','mean')).reset_index();gb.to_csv(out/'full_eda_gd15_bins.csv',index=False)
    gs={'development_rows_total':int(((f.date>='2023-01-01')&(f.date<'2024-01-01')&(f.result_day<'2024-01-01')).sum()),'usable_rows':len(gd),'positive_rows':int((gd.delta>0).sum()),'negative_rows':int((gd.delta<0).sum()),'zero_rows':int((gd.delta==0).sum()),'positive_win_rate':float(gd.loc[gd.delta>0,'y_true'].mean()),'negative_win_rate':float(gd.loc[gd.delta<0,'y_true'].mean()),'pearson':float(pearsonr(gd.delta,gd.y_true).statistic),'spearman':float(spearmanr(gd.delta,gd.y_true).statistic),'rank_auc':float(roc_auc_score(gd.y_true,gd.delta)),'policy':'2023 development only, result_day<2024; no odds restriction; no fitted classifier; zero differences retained; fixed gold-difference bin boundaries; W20 joint-complete historical statistics.'};(out/'full_eda_gd15_summary.json').write_text(json.dumps(gs,indent=2)+'\n')
    plot_gd15_development(gb,figures)
    window=[]
    for w in [3,5,10,20,50]:window.append({'w':w,'usable':int((f[f't1_w{w}_complete'].eq(1)&f[f't2_w{w}_complete'].eq(1)).sum()),'full':int((f[f't1_w{w}_full'].eq(1)&f[f't2_w{w}_full'].eq(1)).sum())})
    windows=pd.DataFrame(window);windows.to_csv(out/'full_eda_window_availability.csv',index=False);fig,ax=plt.subplots(figsize=(8,4));x=np.arange(5);ax.bar(x-.18,100*windows.usable/len(f),.36,label='Bez braków w obserwowanej historii',color='#2376A8');ax.bar(x+.18,100*windows.full/len(f),.36,label='Pełne W map bez braków',color='#C87529');ax.set_xticks(x,['W'+str(w) for w in windows.w]);ax.set(ylabel='Odsetek 40 636 celów [%]',ylim=(0,100),title='Dostępność historii obu drużyn');ax.legend(frameon=False);save(fig,figures,'full_eda_window_availability')
    inputs=[raw,feature_path,panel_path,odds_path,identity_path,Path(__file__),ROOT/'src/models/competition_tiers.py'];receipt={'status':'COMPLETE','source_sha256':{str(Path(p).resolve()):sha(p) for p in inputs},'source_series':len(source),'source_maps':int(maps.maps.sum()),'maps_without_own_date':fallback,'map_year_policy':'Explicit game date if present, otherwise series date; fallback count reported.','feature_targets':len(f),'evaluation_targets':len(panel),'market_common':len(common),'tournament_names_sports':int(panel.tournament_name.nunique()),'tournament_names_market':int(common.tournament_name.nunique()),'tier_policy':'No verified official historical tier labels; report exact source tournament names instead. competition_tiers.py is a rule taxonomy, not independently validated truth.','gd15':gs,'output_sha256':{str(p.resolve()):sha(p) for folder in (out,figures,generated) for p in folder.glob('full_eda*') if p.is_file()}}
    (out/'full_eda_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps({k:v for k,v in receipt.items() if k not in ['source_sha256','output_sha256']},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for arg in ['raw','features','panel','odds','identities','output','figures','generated']:p.add_argument('--'+arg,required=True)
    a=p.parse_args();build(a.raw,a.features,a.panel,a.odds,a.identities,a.output,a.figures,a.generated)
