"""Render TeX tables exclusively from completed, corrected thesis evidence."""
import argparse
import json
from pathlib import Path
import pandas as pd

NAMES={'logistic':'LR ElasticNet','extra_trees':'ExtraTrees','hist_gradient_boosting':'HGB',
       'lightgbm':'LightGBM','mlp':'MLP', 'player_gl':'Player Glicko-2',
       'market_open':'Market OPEN','market_close':'Market CLOSE'}

def fmt(x,n=6):return f'{float(x):.{n}f}'.replace('.',r'{,}')
def name(x):
    x=x.removeprefix('p_')
    if x.endswith('_raw'):return NAMES.get(x[:-4],x[:-4])+' (sym.)'
    return NAMES.get(x,x.replace('_',r'\_'))

def table(caption,headers,rows,width=None):
    spec=width or 'l'+'r'*(len(headers)-1)
    return '\n'.join([r'\begin{table}[htbp]',r'\centering\small',r'\caption{'+caption+'}',
        r'\begin{tabular}{'+spec+'}',r'\toprule',' & '.join(headers)+r' \\',r'\midrule',
        *[' & '.join(map(str,r))+r' \\' for r in rows],r'\bottomrule',r'\end{tabular}',r'\end{table}',''])


def main():
    a=argparse.ArgumentParser(description=__doc__)
    a.add_argument('--evaluation',type=Path,required=True);a.add_argument('--selection',type=Path,required=True)
    a.add_argument('--thesis',type=Path,required=True)
    p=a.parse_args();out=p.thesis/'generated';out.mkdir(parents=True,exist_ok=True)
    m=pd.read_csv(p.evaluation/'metrics.csv');c=pd.read_csv(p.evaluation/'comparisons.csv');controlled=pd.read_csv(p.evaluation/'controlled_comparisons.csv')
    s=json.loads(p.selection.read_text());primary='p_'+s['primary_family']
    def metric(cohort,model):return m[(m.cohort==cohort)&(m.model==model)].iloc[0]
    def ci(candidate,reference,cohort='sports'):
        source=pd.concat([c,controlled]);d=source[(source.candidate==candidate)&(source.reference==reference)&(source.cohort==cohort)&(source.metric=='logloss')&(source.blocking=='month')].iloc[0]
        return d
    rows=[]
    for family in ('elo','gl','ts','os','pl','tm'):
        for scope in ('player','team'):
            label=scope+'_'+family;x=metric('sports','p_'+label);y=metric('market_common','p_'+label)
            rows.append([label.replace('_',r'\_'),fmt(x.logloss),fmt(x.brier),fmt(y.logloss),fmt(y.brier)])
    (out/'rating_results.tex').write_text(table('Skalibrowane ratingi po projekcji formatu serii. Pełna próba: 11\\,550; wspólna rynkowa: 3229.',
        ['Rating','LL sport','Brier sport','LL rynek','Brier rynek'],rows))
    rows=[]
    for family,params in s['selected'].items():
        r=next(r for r in s['candidates'] if r['family']==family and r['params']==params)
        rows.append([NAMES[family],r['candidate']+1,fmt(r['logloss']),fmt(r['brier'])])
    text=table('Wybór konfiguracji wyłącznie na 4836 meczach rozwojowych z 2023 roku; TRAIN: 24\\,250 wcześniejszych celów.',
               ['Rodzina','Wariant','LogLoss','Brier'],rows)
    text+=f"Rodziną wskazaną przez historyczną regułę wyboru przed oceną 2024+ jest {NAMES[s['primary_family']]}. "
    text+='Wybrana MLP ma jedną warstwę ukrytą z 8 neuronami i regularyzację $\\alpha=0{,}01$. '
    text+='Wybrana LR używa $C=0{,}1$ oraz $l_1\\_ratio=0{,}5$. Pełne konfiguracje wszystkich rodzin zapisano w artefakcie wyboru.\n'
    (out/'selection_results.tex').write_text(text)
    for cohort,title in [('sports','pełnej próbie sportowej (11\\,550 meczów)'),('market_common','wspólnej próbie rynkowej (3229 meczów)')]:
        models=['p_'+f for f in s['selected']]+['p_player_gl']+(['market_open','market_close'] if cohort=='market_common' else [])
        rows=[]
        for model in models:
            r=metric(cohort,model);rows.append([name(model),fmt(r.logloss),fmt(r.brier),fmt(r.auc,4),fmt(100*r.ece,2)])
        (out/f'metrics_{cohort}.tex').write_text(table('Wyniki na '+title+'. ECE podano w punktach procentowych (10 przedziałów).',
            ['Model','LogLoss','Brier','AUC','ECE [pp]'],rows))
    rows=[]
    for cohort in ['sports','market_common']:
        for reference in ['p_player_gl']+(['market_open','market_close'] if cohort=='market_common' else []):
            r=ci(primary,reference,cohort)
            rows.append(['sport' if cohort=='sports' else 'rynek',name(reference),fmt(r.delta),f'[{fmt(r.ci_lower)}; {fmt(r.ci_upper)}]',int(r.blocks)])
    (out/'primary_comparisons.tex').write_text(table('Różnica LogLoss: wybrana MLP minus odniesienie. Ujemna wartość sprzyja MLP; sparowany bootstrap miesięczny, 5000 replik.',
        ['Próba','Odniesienie',r'$\Delta$ LL','95\\% przedział','Bloki'],rows))
    aliases={'p_lr_w3':'W3','p_lr_w5':'W5','p_lr_w10':'W10','p_lr_w50':'W50','p_lr_player_only':'Bez kontekstu',
      'p_lr_glicko_context':'Glicko + kontekst','p_lr_no_player_gl':'Bez surowego p Glicko','p_lr_no_glicko':'Bez rodziny Glicko',
      'p_lr_player_team':'Dodatkowe 12 cech team','p_supp_no_binomial':'Bez cech dwumianowych',
      'p_supp_full_player_team':'Player+Team (80 cech)','p_supp_mask_base10':'Maskowanie bazy 10\\%',
      'p_supp_mask_playergl10':'Maskowanie p Glicko 10\\%','p_supp_mask_playergl30':'Maskowanie p Glicko 30\\%'}
    rows=[]
    for model,label in aliases.items():
        r=metric('sports',model);d=ci(model,'p_logistic')
        rows.append([label,fmt(r.logloss),fmt(d.delta),f'[{fmt(d.ci_lower)}; {fmt(d.ci_upper)}]'])
    (out/'ablation_results.tex').write_text(table('Kontrolowane ablacje LR na pełnej próbie. Odniesienie: LR W20-Binomial z tymi samymi parametrami i kalibracją.',
        ['Wariant','LogLoss',r'$\Delta$ LL','95\\% przedział'],rows,width='p{5cm}rrr'))
    rows=[]
    for model,label in [('p_lr_player_only','Bez kontekstu'),('p_supp_no_binomial','Bez cech dwumianowych')]:
        d=ci(model,'p_logistic');r=metric('sports',model)
        rows.append([label,fmt(r.logloss),fmt(d.delta),f'[{fmt(d.ci_lower)}; {fmt(d.ci_upper)}]'])
    (out/'elasticnet_ablations.tex').write_text(table('Ablacje ElasticNet wobec pełnego wejścia W20-Binomial; te same 11 550 celów.',
        ['Wariant ElasticNet','LogLoss',r'$\Delta$ LL','95\\% przedział'],rows))
    rows=[]
    for model in ['p_supp_primary_no_context','p_supp_primary_no_binomial']:
        d=ci(model,primary);r=metric('sports',model)
        rows.append(['Bez kontekstu' if model.endswith('context') else 'Bez cech dwumianowych',fmt(r.logloss),fmt(d.delta),f'[{fmt(d.ci_lower)}; {fmt(d.ci_upper)}]'])
    (out/'primary_ablations.tex').write_text(table('Ablacje wybranej MLP względem jej pełnego wejścia, na tych samych 11\\,550 celach.',
        ['Wariant MLP','LogLoss',r'$\Delta$ LL','95\\% przedział'],rows))
    rows=[]
    for family in s['selected']:
        raw=metric('sports','p_'+family+'_raw');cal=metric('sports','p_'+family);d=ci('p_'+family,'p_'+family+'_raw')
        rows.append([NAMES[family],fmt(raw.logloss),fmt(cal.logloss),fmt(100*raw.ece,2),fmt(100*cal.ece,2),fmt(d.delta)])
    (out/'calibration_results.tex').write_text(table('Symetryczne prognozy surowe i kalibracja na wcześniejszych wynikach. ECE w punktach procentowych; pełna próba sportowa.',
        ['Rodzina','LL sym.','LL kal.','ECE sym.','ECE kal.',r'$\Delta$ LL'],rows))
    rows=[]
    for model in ['p_logistic',primary,'p_player_gl','market_open','market_close']:
        r=metric('market_common',model)
        rows.append([name(model),fmt(100*r.ece_5,2),fmt(100*r.ece,2),fmt(100*r.ece_15,2)])
    (out/'ece_sensitivity.tex').write_text(table('Wrażliwość ECE na liczbę przedziałów; 3229 wspólnych meczów. Wszystkie wartości w punktach procentowych.',
        ['Model','5 przedziałów','10 przedziałów','15 przedziałów'],rows))
    evidence={'primary':primary,'sports':metric('sports',primary).to_dict(),'market':metric('market_common',primary).to_dict(),
              'primary_vs_glicko':ci(primary,'p_player_gl').to_dict(),
              'primary_vs_open':ci(primary,'market_open','market_common').to_dict(),
              'primary_vs_close':ci(primary,'market_close','market_common').to_dict(),
              'primary_no_context':ci('p_supp_primary_no_context',primary).to_dict()}
    (out/'evidence.json').write_text(json.dumps(evidence,indent=2,default=str)+'\n')
    print(json.dumps(evidence,indent=2,default=str))

if __name__=='__main__':main()
