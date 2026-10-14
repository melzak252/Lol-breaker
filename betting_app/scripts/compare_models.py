#!/usr/bin/env python3
"""
Porównanie modeli predykcyjnych na meczach z wynikami.
Oblicza LogLoss i AUC dla każdego modelu.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
from sklearn.metrics import log_loss, roc_auc_score

from betting_app.core.db import query_df


def compare_models():
    """Porównaj wszystkie modele na meczach z wynikami."""
    
    # Pobierz predykcje dla meczów z wynikami
    df = query_df("""
        SELECT 
            cp.model_name,
            cp.model_version,
            cp.prob_a,
            cm.winner_side,
            cm.id as match_id
        FROM canonical_predictions cp
        JOIN canonical_matches cm ON cp.canonical_match_id = cm.id
        WHERE cm.winner_side IS NOT NULL
          AND cp.prediction_status = 'active'
          AND cp.prob_a > 0 AND cp.prob_a < 1
        ORDER BY cp.model_name, cm.id
    """)
    
    if df.empty:
        print("Brak danych do porównania")
        return
    
    # Konwertuj winner_side na binarne (1 = team_a wygrał)
    df['y_true'] = (df['winner_side'] == 'team_a').astype(int)
    
    print("=" * 80)
    print("PORÓWNANIE MODELI PREDYKCYJNYCH")
    print("=" * 80)
    print()
    
    results = []
    
    for model_name in df['model_name'].unique():
        model_df = df[df['model_name'] == model_name].copy()
        
        if len(model_df) < 5:
            print(f"{model_name}: za mało predykcji ({len(model_df)})")
            continue
        
        # Oblicz metryki
        y_true = model_df['y_true'].values
        y_prob = model_df['prob_a'].values
        
        try:
            ll = log_loss(y_true, y_prob)
            auc = roc_auc_score(y_true, y_prob)
            
            # Dodatkowe statystyki
            accuracy = ((y_prob > 0.5) == y_true).mean()
            avg_prob = y_prob.mean()
            
            results.append({
                'model': model_name,
                'n': len(model_df),
                'logloss': ll,
                'auc': auc,
                'accuracy': accuracy,
                'avg_prob': avg_prob,
            })
            
        except Exception as e:
            print(f"{model_name}: błąd obliczeń - {e}")
    
    # Sortuj po LogLoss (niższy = lepszy)
    results.sort(key=lambda x: x['logloss'])
    
    # Wyświetl tabelę
    print(f"{'Model':<45} {'N':>5} {'LogLoss':>10} {'AUC':>8} {'Acc':>8} {'Avg P':>8}")
    print("-" * 80)
    
    for r in results:
        print(f"{r['model']:<45} {r['n']:>5} {r['logloss']:>10.4f} {r['auc']:>8.4f} "
              f"{r['accuracy']:>8.1%} {r['avg_prob']:>8.4f}")
    
    print()
    print("=" * 80)
    print("INTERPRETACJA:")
    print("-" * 80)
    print("LogLoss: niższy = lepsza kalibracja (random baseline = 0.6931)")
    print("AUC: wyższy = lepsza dyskryminacja (random = 0.5, perfect = 1.0)")
    print("Acc: accuracy przy threshold 0.5")
    print("Avg P: średnie prawdopodobieństwo dla team_a")
    print()
    
    if len(results) >= 2:
        best = results[0]
        print(f"🏆 Najlepszy model: {best['model']}")
        print(f"   LogLoss={best['logloss']:.4f}, AUC={best['auc']:.4f}")
        
        if len(results) >= 2:
            second = results[1]
            improvement = second['logloss'] - best['logloss']
            print(f"   Przewaga nad {second['model']}: {improvement:.4f} LogLoss")


if __name__ == "__main__":
    compare_models()
