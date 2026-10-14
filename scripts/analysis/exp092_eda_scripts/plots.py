from pathlib import Path
import pandas as pd,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(__file__).parent
h=pd.read_csv(P/'probe_paired_uncertainty.csv');h=h[h.probe!='ratings'].iloc[::-1];labels={'ratings_winrate':'Win rate','ratings_economy':'Ekonomia','ratings_combat':'Walka / obrażenia','ratings_objectives':'Obiekty','ratings_vision_duration':'Wizja i czas gry','ratings_w20':'Całe W20','ratings_uncertainty_experience':'Niepewność i doświadczenie'}
fig,ax=plt.subplots(figsize=(10,5));ax.errorbar(h.delta,range(len(h)),xerr=np.array([h.delta-h.ci_low,h.ci_high-h.delta]),fmt='o',capsize=4);ax.set_yticks(range(len(h)),[labels[x] for x in h.probe]);ax.axvline(0,color='gray',linestyle='--');ax.set(xlabel='Zmiana LogLoss względem samych ratingów; ujemna = poprawa',title='Wartość grup cech: 5400 meczów z 2025–2026\n95% CI, bootstrap miesięcy; archiwalne cechy wymagają audytu czasowego');ax.grid(axis='x',alpha=.2);fig.tight_layout();fig.savefig(P/'feature_value.png',dpi=160)
c=pd.read_csv(P/'rating_correlations.csv',index_col=0);fig,ax=plt.subplots(figsize=(10,8));im=ax.imshow(c,vmin=0,vmax=1,cmap='viridis');ax.set_xticks(range(len(c)),c.columns,rotation=60,ha='right');ax.set_yticks(range(len(c)),c.index);ax.set_title('Korelacja rangowa sygnałów ratingowych');fig.colorbar(im,ax=ax);fig.tight_layout();fig.savefig(P/'rating_redundancy.png',dpi=140)
