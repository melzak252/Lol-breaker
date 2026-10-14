#!/usr/bin/env python3
"""Compare all models on finished matches with results."""
import sys
import math
import psycopg2

def main():
    conn = psycopg2.connect(
        host="timescaledb",
        port=5432,
        user="betting",
        password="betting_local_password",
        dbname="betting"
    )
    cur = conn.cursor()

    cur.execute("""
    SELECT cp.model_name, cp.model_version, cp.match_id, cp.prob_a, cm.winner_side
    FROM canonical_predictions cp
    JOIN canonical_matches cm ON cp.match_id = cm.match_id
    WHERE cm.winner_side IS NOT NULL
    ORDER BY cp.model_name, cp.match_id
    """)

    rows = cur.fetchall()
    print(f"Total prediction rows with results: {len(rows)}", flush=True)

    from collections import defaultdict
    models = defaultdict(list)
    match_ids_per_model = defaultdict(set)
    
    for model_name, model_version, match_id, prob_a, winner_side in rows:
        key = f"{model_name} ({model_version})"
        actual = 1 if winner_side == 'team_a' else 0
        p = max(min(prob_a, 1 - 1e-15), 1e-15)
        ll = -(actual * math.log(p) + (1 - actual) * math.log(1 - p))
        correct = 1 if (actual == 1 and prob_a > 0.5) or (actual == 0 and prob_a < 0.5) else 0
        models[key].append((ll, correct, prob_a))
        match_ids_per_model[key].add(match_id)

    print(f"\n{'Model':<50} {'N':>4} {'LogLoss':>8} {'Acc%':>6} {'AvgP':>6}", flush=True)
    print("-" * 80, flush=True)
    for key in sorted(models.keys()):
        data = models[key]
        n = len(data)
        avg_ll = sum(d[0] for d in data) / n
        acc = sum(d[1] for d in data) / n * 100
        avg_p = sum(d[2] for d in data) / n
        print(f"{key:<50} {n:>4} {avg_ll:>8.4f} {acc:>6.1f} {avg_p:>6.4f}", flush=True)

    # Common matches analysis
    all_match_sets = list(match_ids_per_model.values())
    if all_match_sets:
        common = set.intersection(*all_match_sets)
    else:
        common = set()
    
    print(f"\nCommon matches across ALL models: {len(common)}", flush=True)

    if common:
        print(f"\n--- On {len(common)} common matches ---", flush=True)
        print(f"{'Model':<50} {'N':>4} {'LogLoss':>8} {'Acc%':>6} {'AvgP':>6}", flush=True)
        print("-" * 80, flush=True)
        for key in sorted(models.keys()):
            model_name_only = key.split(" (")[0]
            cur.execute("""
            SELECT cp.prob_a, cm.winner_side
            FROM canonical_predictions cp
            JOIN canonical_matches cm ON cp.match_id = cm.match_id
            WHERE cm.winner_side IS NOT NULL 
            AND cp.match_id = ANY(%s)
            AND cp.model_name = %s
            """, (list(common), model_name_only))
            
            data = []
            for prob_a, winner_side in cur.fetchall():
                actual = 1 if winner_side == 'team_a' else 0
                p = max(min(prob_a, 1 - 1e-15), 1e-15)
                ll = -(actual * math.log(p) + (1 - actual) * math.log(1 - p))
                correct = 1 if (actual == 1 and prob_a > 0.5) or (actual == 0 and prob_a < 0.5) else 0
                data.append((ll, correct, prob_a))
            
            if data:
                n = len(data)
                avg_ll = sum(d[0] for d in data) / n
                acc = sum(d[1] for d in data) / n * 100
                avg_p = sum(d[2] for d in data) / n
                print(f"{key:<50} {n:>4} {avg_ll:>8.4f} {acc:>6.1f} {avg_p:>6.4f}", flush=True)

    conn.close()
    print("\nDone.", flush=True)

if __name__ == "__main__":
    main()
