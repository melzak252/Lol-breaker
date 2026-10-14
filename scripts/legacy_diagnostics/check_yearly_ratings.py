import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

def main():
    df = pd.read_csv('data/golgg_y_predicts.csv')
    df['year'] = df['date'].str[:4]
    
    systems = ['player_elo', 'player_gl', 'player_ts']
    
    print("Yearly Performance of Base Rating Systems:")
    print("=" * 60)
    
    for year, group in df.groupby('year'):
        print(f"--- Year: {year} (N={len(group)}) ---")
        for s in systems:
            valid = group.dropna(subset=[s, 'y_true'])
            if len(valid) > 0:
                # Clip probabilities to avoid log_loss issues with 0 or 1
                probs = valid[s].clip(0.001, 0.999)
                ll = log_loss(valid['y_true'], probs)
                try:
                    auc = roc_auc_score(valid['y_true'], probs)
                except ValueError:
                    auc = 0.5
                print(f"{s:10} | LogLoss: {ll:.4f} | AUC: {auc:.4f}")
        print()

if __name__ == "__main__":
    main()
