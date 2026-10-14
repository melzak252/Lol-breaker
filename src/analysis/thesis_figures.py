"""Polish thesis figures with explicit source, target, and market cohorts.

EDA never infers the source map count from series scores (initial advantages can
make that incorrect). It reads the pinned historical bank receipt instead.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COLORS=['#2376A8','#C87529','#598447','#8E5A99','#666666','#AA4455','#559999','#997744']
LABELS={'p_logistic':'Regresja logistyczna','p_extra_trees':'Extra Trees','p_hist_gradient_boosting':'Histogram GB','p_hist_gb':'Histogram GB','p_player_gl':'Glicko-2 graczy','market_open':'Rynek OPEN','market_close':'Rynek CLOSE','p_mlp':'Sieć MLP (wybór 2023)','p_lightgbm':'LightGBM'}

def style():
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':12,'axes.labelsize':10,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white','axes.facecolor':'white','savefig.dpi':180})

def save(fig,output,name):
    fig.savefig(output/f'{name}.png',bbox_inches='tight')
    fig.savefig(output/f'{name}.pdf',bbox_inches='tight')
    plt.close(fig)

def build_eda(feature_path,identity_path,market_path,bank_receipt_path,output_dir):
    style();output=Path(output_dir);output.mkdir(parents=True,exist_ok=True)
    f=pd.read_parquet(feature_path);identities=pd.DataFrame(json.loads(Path(identity_path).read_text()));odds=pd.read_csv(market_path)
    receipt=json.loads(Path(bank_receipt_path).read_text())
    if len(identities)!=receipt['history_series'] or len(f)!=receipt['rows']:raise ValueError('Source cohort counts disagree with pinned receipt')
    counts=pd.DataFrame({'cohort':['source_series','source_maps','feature_targets','identity_matched_market'],'n':[len(identities),receipt['history_maps'],len(f),len(odds)],'description':['Wszystkie serie w źródle; także remisy i serie poza kohortą celu','Wszystkie mapy źródłowe; liczba z audytu banku','Serie spełniające kontrakt celu i jawnego składu','Serie po niezależnym dopasowaniu tożsamości; przed filtrem jakości kursów']})
    counts.to_csv(output/'eda_cohort_counts.csv',index=False)
    years=sorted(set(identities.date.str[:4])|set(f.date.str[:4])|set(odds.golgg_date.str[:4]))
    yearly=pd.DataFrame({'year':years})
    for label,days in [('source_series',identities.date),('feature_targets',f.date),('matched_market',odds.golgg_date)]:
        yearly[label]=yearly.year.map(days.str[:4].value_counts()).fillna(0).astype(int)
    yearly.to_csv(output/'eda_series_by_year.csv',index=False)
    formats=f.best_of.value_counts().sort_index().rename_axis('best_of').reset_index(name='n');formats.to_csv(output/'eda_target_formats.csv',index=False)
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),gridspec_kw={'width_ratios':[2.0,1]})
    x=np.arange(len(years))
    for label,title,color in [('source_series','Serie źródłowe',COLORS[0]),('feature_targets','Kohorta cech',COLORS[1]),('matched_market','Dopasowany rynek',COLORS[2])]:axes[0].plot(x,yearly[label],marker='o',markersize=4,label=title,color=color)
    axes[0].set_xticks(x,years,rotation=45);axes[0].set_ylabel('Liczba serii');axes[0].set_title('Pokrycie w kolejnych latach');axes[0].legend(frameon=False,fontsize=9);axes[0].grid(axis='y',alpha=.2)
    axes[1].bar(['Bo'+str(v) for v in formats.best_of],formats.n,color=COLORS[0],width=.65)
    for i,n in enumerate(formats.n):axes[1].text(i,n+max(formats.n)*.015,f'{n:,}'.replace(',',' '),ha='center',fontsize=9)
    axes[1].set_ylim(0,max(formats.n)*1.14);axes[1].set_title('Formaty w kohorcie cech');axes[1].set_ylabel('Liczba serii');axes[1].grid(axis='y',alpha=.2)
    fig.suptitle(f'Źródło: {len(identities):,} serii / {receipt["history_maps"]:,} map; kohorta cech: {len(f):,} serii'.replace(',',' '),fontsize=13,y=1.02)
    fig.text(.01,-.055,'Rynek: dopasowanie nazw i dat bez użycia wyniku do wyboru strony. Rok 2026 jest niepełny; zakres dat rynku i sportu różni się.',fontsize=9)
    fig.tight_layout();save(fig,output,'eda_coverage_and_formats')
    window=[]
    for w in [3,5,10,20,50]:
        window.append({'window':w,'usable':int((f[f't1_w{w}_complete'].eq(1)&f[f't2_w{w}_complete'].eq(1)).sum()),'full':int((f[f't1_w{w}_full'].eq(1)&f[f't2_w{w}_full'].eq(1)).sum()),'all_targets':len(f)})
    windows=pd.DataFrame(window);windows.to_csv(output/'eda_window_availability.csv',index=False)
    fig,ax=plt.subplots(figsize=(8.5,4.5));x=np.arange(len(windows));width=.36
    for offset,col,title,color in [(-width/2,'usable','Dostępne obserwacje bez braków',COLORS[0]),(width/2,'full','Pełne okno W map',COLORS[1])]:ax.bar(x+offset,100*windows[col]/len(f),width,label=title,color=color)
    ax.set_xticks(x,['W'+str(w) for w in windows.window]);ax.set_ylim(0,100);ax.set_ylabel('Odsetek kohorty cech [%]');ax.set_title(f'Dostępność historii obu drużyn; N = {len(f):,}'.replace(',',' '));ax.legend(frameon=False,loc='upper right');ax.grid(axis='y',alpha=.2)
    fig.text(.01,-.02,'Dostępne okno może zawierać mniej niż W map. Brak pomiaru pozostaje brakiem; nie jest zastępowany zerem.',fontsize=9);fig.tight_layout();save(fig,output,'eda_window_availability')
    inputs=[feature_path,identity_path,market_path,bank_receipt_path]
    provenance={'source_sha256':{str(Path(p).resolve()):hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in inputs},'source_map_count_provenance':'Pinned bank completion receipt; not inferred from final scores','scope':'Source series dates from identities, format distribution from40636targetmetadata, market10720identitymatches before valid-price/evaluation-time filtering.'}
    (output/'eda_provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    return counts

def build_evaluation_figures(evaluation_dir,output_dir,primary_model="p_mlp"):
    style();evaluation=Path(evaluation_dir);output=Path(output_dir);output.mkdir(parents=True,exist_ok=True)
    metrics=pd.read_csv(evaluation/'metrics.csv');reliability=pd.read_csv(evaluation/'reliability.csv')
    main=['p_logistic','p_extra_trees','p_hist_gradient_boosting','p_hist_gb','p_lightgbm','p_mlp']
    for cohort,title in [('sports','Pełna kohorta sportowa'),('market_common','Wspólna kohorta model–rynek')]:
        wanted=main+['p_player_gl']+(['market_open','market_close'] if cohort=='market_common' else [])
        rows=metrics[(metrics.cohort==cohort)&metrics.model.isin(wanted)].copy()
        if rows.empty:continue
        rows=rows.sort_values('logloss');title=title+f' (N = {int(rows.iloc[0]["n"]):,})'.replace(',',' ');fig,ax=plt.subplots(figsize=(9,5.5));ax.barh([LABELS.get(m,m) for m in rows.model],rows.logloss,color=[COLORS[i%len(COLORS)] for i in range(len(rows))])
        ax.invert_yaxis();ax.set_xlabel('LogLoss — mniej oznacza lepiej');ax.set_title(title);ax.grid(axis='x',alpha=.2)
        for i,value in enumerate(rows.logloss):ax.text(value+.002,i,f'{value:.4f}',va='center',fontsize=10)
        ax.set_xlim(0,float(rows.logloss.max())*1.15);fig.tight_layout();save(fig,output,'logloss_'+cohort)
        fig,ax=plt.subplots(figsize=(6.5,5.5));ax.plot([0,1],[0,1],ls='--',color='#777777',lw=1,label='Idealna kalibracja')
        reliability_models=list(dict.fromkeys([primary_model,'p_logistic','p_player_gl']+(['market_open'] if cohort=='market_common' else [])))
        for i,model in enumerate(reliability_models):
            if model not in rows.model.values:continue
            bins=reliability[(reliability.cohort==cohort)&(reliability.model==model)].sort_values('mean_probability')
            ax.plot(bins.mean_probability,bins.observed_rate,marker='o',markersize=4,label=LABELS.get(model,model),color=COLORS[i%len(COLORS)],lw=1.5)
        ax.set(xlim=(0,1),ylim=(0,1),xlabel='Średnia prognozowana szansa',ylabel='Obserwowana częstość zwycięstw',title=title+' — kalibracja');ax.legend(frameon=False,fontsize=8,loc='upper left');ax.grid(alpha=.2)
        fig.text(.01,-.02,'10 przedziałów o równej szerokości. Punkty są opisowe; małe grupy mają dużą niepewność.',fontsize=8);fig.tight_layout();save(fig,output,'reliability_'+cohort)

    (output/'evaluation_figure_provenance.json').write_text(json.dumps({'evaluation_sources':{str((evaluation/name).resolve()):hashlib.sha256((evaluation/name).read_bytes()).hexdigest() for name in ['metrics.csv','reliability.csv']},'primary_model':primary_model,'scope':'Loss plots show five principal families and Glicko, plus both market references on common cohort. Reliability plots show prespecified primary, LR, Glicko, and OPEN where available.'},indent=2)+'\n')

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--features');parser.add_argument('--identities');parser.add_argument('--market');parser.add_argument('--bank-receipt');parser.add_argument('--evaluation-dir');parser.add_argument('--primary-model',default='p_mlp');parser.add_argument('--output-dir',required=True)
    args=parser.parse_args()
    if args.features:build_eda(args.features,args.identities,args.market,args.bank_receipt,args.output_dir)
    if args.evaluation_dir:build_evaluation_figures(args.evaluation_dir,args.output_dir,args.primary_model)
