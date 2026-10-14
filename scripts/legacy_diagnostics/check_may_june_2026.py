import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

def main():
    df = pd.read_csv('data/golgg_y_predicts.csv')
    
    # Filtrujemy maj i czerwiec 2026
    mask = df['date'].str.startswith('2026-05') | df['date'].str.startswith('2026-06')
    df_filtered = df[mask]
    
    systems = ['player_elo', 'player_gl', 'player_ts']
    
    print(f"Performance of Base Rating Systems for May-June 2026 (N={len(df_filtered)}):")
    print("=" * 60)
    
    for s in systems:
        valid = df_filtered.dropna(subset=[s, 'y_true'])
        if len(valid) > 0:
            probs = valid[s].clip(0.001, 0.999)
            ll = log_loss(valid['y_true'], probs)
            try:
                auc = roc_auc_score(valid['y_true'], probs)
            except ValueError:
                auc = 0.5
            print(f"{s:10} | LogLoss: {ll:.4f} | AUC: {auc:.4f}")

if __name__ == "__main__":
    main()
