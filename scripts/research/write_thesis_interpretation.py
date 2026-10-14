"""Generate result-dependent thesis prose from verified scores; no model selection."""
import argparse
import json
from pathlib import Path
import pandas as pd
from render_thesis_tables import fmt,name


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--evaluation',type=Path,required=True);p.add_argument('--thesis',type=Path,required=True)
    a=p.parse_args();out=a.thesis/'generated'
    e=json.loads((out/'evidence.json').read_text());m=pd.read_csv(a.evaluation/'metrics.csv')
    c=pd.read_csv(a.evaluation/'comparisons.csv');control=pd.read_csv(a.evaluation/'controlled_comparisons.csv')
    def verdict(r,benefit_negative=True):
        if r['ci_lower']<=0<=r['ci_upper']:
            return 'Przedział obejmuje zero, więc wynik nie potwierdza stabilnej przewagi ani równoważności.'
        improves=r['ci_upper']<0 if benefit_negative else r['ci_lower']>0
        return ('Przedział wskazuje kierunek korzystny dla pełnego modelu w tym historycznym porównaniu.' if improves else
                'Przedział wskazuje kierunek niekorzystny dla pełnego modelu w tym historycznym porównaniu.')
    def comparison(reference,cohort='sports'):
        rows=c[(c.cohort==cohort)&(c.candidate=='p_logistic')&(c.reference==reference)&(c.metric=='logloss')&(c.blocking=='month')]
        if len(rows)!=1:raise ValueError('Expected unique ElasticNet comparison')
        return rows.iloc[0].to_dict()
    e=dict(e,primary='p_logistic',sports=m[(m.cohort=='sports')&(m.model=='p_logistic')].iloc[0].to_dict(),
           market=m[(m.cohort=='market_common')&(m.model=='p_logistic')].iloc[0].to_dict(),
           primary_vs_glicko=comparison('p_player_gl'),primary_vs_open=comparison('market_open','market_common'),
           primary_vs_close=comparison('market_close','market_common'),
           primary_no_context=control[(control.cohort=='sports')&(control.candidate=='p_lr_player_only')&(control.reference=='p_logistic')&(control.metric=='logloss')&(control.blocking=='month')].iloc[0].to_dict())
    sport=e['sports'];market=e['market']
    headline=f"Regresja logistyczna ElasticNet osiągnęła na pełnej próbie LogLoss ${fmt(sport['logloss'])}$ i Brier ${fmt(sport['brier'])}$. "
    headline+=f"Na wspólnej próbie rynkowej wartości wyniosły odpowiednio ${fmt(market['logloss'])}$ i ${fmt(market['brier'])}$. "
    headline+='Konfigurację ElasticNet ustalono na 2023 roku. Rodzinę tę eksponuje się ze względu na późniejsze wyniki; wybór między rodzinami na 2023 roku wskazał MLP.\n'
    for key,label in [('primary_vs_glicko','skalibrowanego Player Glicko-2 na pełnej próbie'),('primary_vs_open','OPEN na wspólnej próbie'),('primary_vs_close','CLOSE na wspólnej próbie')]:
        r=e[key]
        headline+=f"\nWzględem {label} różnica LogLoss (ElasticNet minus odniesienie) wynosi ${fmt(r['delta'])}$, z przedziałem $[{fmt(r['ci_lower'])}; {fmt(r['ci_upper'])}]$. "+verdict(r)+'\n'
    (out/'headline_interpretation.tex').write_text(headline)
    primary=e['primary']; rows=[]
    for ref in ['market_open','market_close']:
        r=c[(c.cohort=='market_common')&(c.candidate==primary)&(c.reference==ref)&(c.metric=='logloss')&(c.blocking=='month')].iloc[0]
        t=c[(c.cohort=='market_common')&(c.candidate==primary)&(c.reference==ref)&(c.metric=='logloss')&(c.blocking=='tournament_year_proxy')].iloc[0]
        rows.append(f"Dla {name(ref)} estymand z jednakową wagą miesięcy wynosi ${fmt(r.equal_block_delta)}$ "
                    f"(przedział $[{fmt(r.equal_block_ci_lower)}; {fmt(r.equal_block_ci_upper)}]$). "
                    f"Przy blokowaniu turniej--rok, obejmującym {int(t.blocks)} bloków, przedział ważonej różnicy wynosi "
                    f"$[{fmt(t.ci_lower)}; {fmt(t.ci_upper)}]$. ")
    open_equal=c[(c.cohort=='market_common')&(c.candidate==primary)&(c.reference=='market_open')&(c.metric=='logloss')&(c.blocking=='month')].iloc[0]
    close_inconclusive=e['primary_vs_close']['ci_lower']<=0<=e['primary_vs_close']['ci_upper']
    equal_inconclusive=open_equal.equal_block_ci_lower<=0<=open_equal.equal_block_ci_upper
    robustness_pl=('Przy jednakowej wadze miesięcy porównanie z OPEN jest nierozstrzygające. '
                   'Wniosek o przewadze zależy zatem od przyjętego estymandu i nie jest odporny na tę zmianę ważenia. '
                   if equal_inconclusive else '')
    close_pl=('Porównanie z CLOSE pozostaje nierozstrzygające: przedział różnicy obejmuje zero. '
              if close_inconclusive else 'W porównaniu z CLOSE: '+verdict(e['primary_vs_close'])+' ')
    if equal_inconclusive:
        rows.append(robustness_pl+'Średnia równoważąca miesiące odpowiada innemu pytaniu niż średnia ważona liczbą meczów; nie zastępuje wyniku głównego, lecz ogranicza jego uogólnienie.')
    (out/'stability_results.tex').write_text('\n\n'.join(rows)+'\n')
    h1=control[(control.cohort=='sports')&(control.metric=='logloss')&control.candidate.str.startswith('p_team_')]
    better=int((h1.ci_lower>0).sum());worse=int((h1.ci_upper<0).sum())
    h2=e['primary_vs_glicko'];h3=e['primary_no_context']
    text=f"\\paragraph{{H1: rating zawodniczy a organizacja.}} Dla {better} z sześciu rodzin miesięczny przedział różnicy "
    text+=f"(organizacja minus zawodnicy) leży powyżej zera; dla {worse} leży poniżej zera. "
    text+='Hipotezę ocenia się zatem osobno dla rodzin, bez zastępowania tych porównań jedną wartością AUC z innej próby.\n'
    text+='\\paragraph{H2: metamodel a pojedynczy rating.} '+verdict(h2)+' Odniesieniem jest skalibrowany Player Glicko-2, nie surowy rating bez korekty formatu.\n'
    text+=f"\\paragraph{{H3: wkład kontekstu.}} Dla ElasticNet różnica LogLoss (bez kontekstu minus pełne wejście) wynosi ${fmt(h3['delta'])}$ "
    text+=f"z przedziałem $[{fmt(h3['ci_lower'])}; {fmt(h3['ci_upper'])}]$. "+verdict(h3,False)+'\n'
    text+='\\paragraph{H4: konkurencyjność wobec rynku.} '+verdict(e['primary_vs_open'])+' Wniosek dotyczy OPEN na wspólnej próbie; porównanie z CLOSE raportowane jest osobno. '
    text+='Brak dowodu zgodności chwil dostępności informacji ogranicza interpretację nawet przy przedziale nieobejmującym zera.\n'
    (out/'hypotheses_results.tex').write_text(text)
    conclusion=f"Omawiany model ElasticNet uzyskał LogLoss ${fmt(sport['logloss'])}$ na 11\\,550 późniejszych meczach "
    conclusion+=f"oraz ${fmt(market['logloss'])}$ na 3229 wspólnych spotkaniach z kursem OPEN i CLOSE. "
    conclusion+='W porównaniu z OPEN: '+verdict(e['primary_vs_open'])+' W porównaniu z Player Glicko-2: '+verdict(h2)+'\n\n'
    conclusion+=close_pl+robustness_pl+'\n\n'
    conclusion+='ElasticNet ma najniższy LogLoss spośród pięciu ocenianych rodzin na pełnej próbie. Nie dowodzi to jednak przewagi nad oboma kursami ani nie zmienia historycznego wyboru MLP w 2023 roku. '
    conclusion+='Odpowiedź badawcza wynika z poprawionych tabel, kontrolowanych ablacji i jawnego zakresu niepewności. '
    conclusion+='Wykazano wykonalność pełnego procesu prognozowania i jego niezależnej kontroli; trwała przewaga predykcyjna wymaga dalszej oceny prospektywnej.\n'
    (out/'conclusion_results.tex').write_text(conclusion)
    pl=r'''Praca dotyczy predykcji wyników profesjonalnych meczów \textit{League of Legends} na podstawie historycznych danych sportowych oraz porównania jakości prognoz z benchmarkiem bukmacherskim. Wykorzystano dane GOL.GG oraz kursy z serwisu OddsPortal. Celem pracy jest sprawdzenie, czy systemy rankingowe zawodników i drużyn stanowią skuteczną bazę predykcyjną oraz czy metamodel rankingowo-kontekstowy poprawia probabilistyczną ocenę wyniku serii.

Porównano rodziny Elo, Glicko-2, TrueSkill, OpenSkill, Plackett--Luce oraz Thurstone--Mosteller. Ich predykcje, miary niepewności, historyczny kontekst ostatnich 20 map i sygnały skorygowane dwumianowo względem formatu serii wykorzystano jako 46 cech metamodelu W20-Binomial. Porównanie objęło regresję logistyczną ElasticNet, ExtraTrees, HistGradientBoosting, LightGBM i MLP. Piętnaście zadeklarowanych konfiguracji oceniono na danych rozwojowych z 2023 roku, po czym zamrożono wybór przed miesięczną oceną expanding-window od 2024 roku.

Trening obejmuje całą kwalifikującą się historię sportową niezależnie od dostępności kursów. Cechy korzystają wyłącznie z serii udostępnionych przed dniem prognozy. Symetryzację kolejności drużyn połączono z kalibracją dodatnim współczynnikiem logitu bez wyrazu wolnego, dopasowywaną na wcześniejszych predykcjach poza próbą treningową. Zastosowano LogLoss, Brier Score, AUC, ECE oraz sparowane miesięczne przedziały bootstrapowe. Uzupełnieniem są analizy cech, ablacji, kalibracji i segmentów danych.

'''
    pl+=f"ElasticNet uzyskała LogLoss ${fmt(sport['logloss'])}$ na 11\\,550 meczach i ${fmt(market['logloss'])}$ na 3229 wspólnych meczach rynkowych. "
    pl+='Na pełnej próbie jest to najniższa strata spośród pięciu rodzin. ElasticNet stanowi główny przedmiot prezentacji wyników; selekcję rodziny na 2023 roku wygrała MLP, więc akcent ten ma charakter retrospektywny. '
    pl+='W porównaniu z OPEN: '+verdict(e['primary_vs_open'])+' Porównanie pozostaje historycznym scenariuszem znanego składu; nie certyfikuje identycznych godzin dostępności informacji ani opłacalności zakładów. '
    pl+=close_pl+robustness_pl
    pl+='Wkład inżynierski obejmuje odtwarzalny potok, kontrolę chronologii i tożsamości oraz audyt zgodności wyników.\n\n\\textbf{Słowa kluczowe:} esport, systemy rankingowe, predykcja probabilistyczna, metamodel, walidacja chronologiczna, kalibracja.\n'
    (out/'abstract_pl.tex').write_text(pl)
    r=e['primary_vs_open'];qualified=('The monthly interval includes zero, providing no clear evidence of an advantage over opening odds.' if r['ci_lower']<=0<=r['ci_upper'] else
       ('The monthly interval favors the model over opening odds on this historical cohort.' if r['ci_upper']<0 else 'The monthly interval favors opening odds over the model on this historical cohort.'))
    en=r'''This thesis studies prediction of professional \textit{League of Legends} match outcomes from historical sports data and compares forecast quality with bookmaker benchmarks. GOL.GG sports records and archival OddsPortal prices are used to examine whether player- and team-level rating systems provide a useful predictive foundation and whether a ranking-context metamodel improves probabilistic series outcome prediction.

The study compares Elo, Glicko-2, TrueSkill, OpenSkill, Plackett--Luce, and Thurstone--Mosteller. Rating predictions, uncertainty measures, context from the last 20 maps, and binomial series-format transformations form the 46-feature W20-Binomial input. Five estimator families are considered: ElasticNet logistic regression, ExtraTrees, HistGradientBoosting, LightGBM, and MLP. Fifteen declared configurations are evaluated on 2023 development data, and selection is frozen before monthly expanding-window evaluation from 2024.

Training uses all eligible sports history regardless of odds availability. Features consume only series released before the prediction day. Team-order symmetrization is followed by positive zero-intercept logit calibration fitted on earlier out-of-sample forecasts. Evaluation includes log loss, Brier score, AUC, ECE, paired monthly bootstrap intervals, and supplementary feature, ablation, calibration, and subgroup analyses.

'''
    en+=f"ElasticNet achieved log loss {sport['logloss']:.6f} on 11,550 matches and {market['logloss']:.6f} on the common 3,229-match market cohort. "+qualified+' '
    en+='It achieved the lowest full-cohort log loss among the five families and is the focus of the results presentation. The 2023 family-selection rule chose MLP; emphasizing ElasticNet is therefore a retrospective interpretation. '
    if close_inconclusive:
        en+='The comparison with closing odds remains inconclusive. '
    if equal_inconclusive:
        en+='The opening-odds comparison is also inconclusive when months receive equal weight, limiting robustness across estimands. '
    en+='The evaluation remains a historical known-roster scenario, without certification of equal information availability times or betting profitability. The engineering contribution includes a reproducible pipeline, chronology and identity checks, and independent result verification.\n\n\\textbf{Keywords:} esports, rating systems, probabilistic prediction, metamodel, chronological evaluation, calibration.\n'
    (out/'abstract_en.tex').write_text(en)

if __name__=='__main__':main()
