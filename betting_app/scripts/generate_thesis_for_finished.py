#!/usr/bin/env python3
"""
Generuje predykcje thesis model dla WSZYSTKICH meczów z wynikami i odds.
Następnie porównuje z pozostałymi modelami.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import json
import numpy as np
from datetime import UTC, datetime
from sklearn.metrics import log_loss, roc_auc_score

from betting_app.core.db import query_df, transaction
from betting_app.services.thesis_inference_service import (
    THESIS_MODEL_NAME,
    THESIS_MODEL_VERSION,
    _load_model,
    _register_thesis_model,
    build_thesis_features_for_match,
    _swap_feature_vector,
    _symmetrize,
    _logit,
    EPSILON,
)


def generate_thesis_for_finished():
    """Generuj predykcje thesis model dla wszystkich meczów z wynikami."""
    
    print("=" * 80)
    print("GENEROWANIE PREDYKCJI THESIS MODEL DLA MECZÓW Z WYNIKAMI")
    print("=" * 80)
    print()
    
    # Pobierz wszystkie mecze z wynikami i odds
    matches = query_df("""
        SELECT DISTINCT cm.id, cm.team_a_name, cm.team_b_name, cm.winner_side
        FROM canonical_matches cm
        JOIN odds_snapshots os ON os.canonical_match_id = cm.id
        WHERE cm.winner_side IS NOT NULL
        ORDER BY cm.id
    """)
    
    print(f"Znaleziono {len(matches)} meczów z wynikami i odds")
    print()
    
    # Załaduj model
    pipeline, calibrator = _load_model()
    model_artifact_id = _register_thesis_model()
    
    # Generuj predykcje
    results = []
    
    with transaction() as conn:
        # Usuń stare predykcje thesis model dla tych meczów
        match_ids = tuple(matches['id'].tolist())
        if match_ids:
            placeholders = ','.join(['?'] * len(match_ids))
            conn.execute(f"""
                DELETE FROM canonical_predictions
                WHERE model_name = ? AND canonical_match_id IN ({placeholders})
            """, (THESIS_MODEL_NAME,) + match_ids)
        
        for _, match in matches.iterrows():
            match_id = int(match['id'])
            team_a = str(match['team_a_name'])
            team_b = str(match['team_b_name'])
            
            # Build features
            feature_vec, diagnostics = build_thesis_features_for_match(
                team_a, team_b,
                ratings_version="latest-full",
                w20_version="w20-latest",
            )
            
            if feature_vec is None:
                print(f"  ⚠ {team_a} vs {team_b}: brak features")
                continue
            
            # Original prediction
            original_prob = float(np.clip(pipeline.predict_proba(feature_vec)[0, 1], EPSILON, 1.0 - EPSILON))
            
            # Swapped prediction
            swapped_vec = _swap_feature_vector(feature_vec)
            swapped_prob = float(np.clip(pipeline.predict_proba(swapped_vec)[0, 1], EPSILON, 1.0 - EPSILON))
            
            # Order symmetry
            sym_prob = _symmetrize(original_prob, swapped_prob)
            
            # Platt calibration
            calibrated_prob = float(np.clip(
                calibrator.predict_proba(_logit(np.array([sym_prob])))[0, 1],
                EPSILON, 1.0 - EPSILON,
            ))
            
            # Store prediction
            predicted_at = datetime.now(UTC).replace(microsecond=0).isoformat()
            conn.execute(
                """
                INSERT INTO canonical_predictions(
                    canonical_match_id, model_artifact_id, model_name, model_version, predicted_at,
                    prob_a, prob_b, features_version, ratings_version, data_cutoff_at, 
                    prediction_status, diagnostics_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'active', ?)
                """,
                (
                    match_id,
                    model_artifact_id,
                    THESIS_MODEL_NAME,
                    THESIS_MODEL_VERSION,
                    predicted_at,
                    calibrated_prob,
                    1.0 - calibrated_prob,
                    "thesis-exp039",
                    "latest-full",
                    None,
                    json.dumps(diagnostics, ensure_ascii=False, sort_keys=True),
                ),
            )
            
            results.append({
                'match_id': match_id,
                'team_a': team_a,
                'team_b': team_b,
                'prob_a': calibrated_prob,
                'winner_side': match['winner_side'],
            })
    
    print(f"✓ Wygenerowano {len(results)} predykcji thesis model")
    print()
    
    return results


def compare_models():
    """Porównaj wszystkie modele na meczach z wynikami."""
    
    print("=" * 80)
    print("PORÓWNANIE MODELI PREDYKCYJNYCH")
    print("=" * 80)
    print()
    
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
    df['y_true'] = (df['winner_side'] == 'a').astype(int)
    
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
    generate_thesis_for_finished()
    compare_models()
