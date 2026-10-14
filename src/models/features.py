import pandas as pd
from typing import List, Tuple

def prepare_metamodel_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Selects and engineers features for the metamodel.
    
    Args:
        df: Input DataFrame containing rating and rank information.
        
    Returns:
        Tuple containing the modified DataFrame and the list of feature names.
    """
    # 1. Base probabilities (BoN adjusted)
    features = [
        "bon_player_elo", "bon_player_gl", "bon_player_ts", "bon_player_pl", "bon_player_tm",
        "bon_team_elo", "bon_team_gl", "bon_team_ts", "bon_team_pl", "bon_team_tm",
        "BoN"
    ]
    
    # Optional features that might not be in all datasets
    optional_features = ["is_top_league", "BoN"]
    for feat in optional_features:
        if feat in df.columns and feat not in features:
            features.append(feat)
    
    # 2. Role-based differences (Rating and Rank)
    roles = ["top", "jungle", "mid", "adc", "support"]
    systems = ["elo", "gl", "ts", "pl", "tm"]
    
    for role in roles:
        for sys in systems:
            # Rating difference (T1 - T2)
            col_1 = f"{role}_{sys}_1"
            col_2 = f"{role}_{sys}_2"
            if col_1 in df.columns and col_2 in df.columns:
                df[f"{role}_{sys}_diff"] = df[col_1] - df[col_2]
                features.append(f"{role}_{sys}_diff")
            
            # Rank difference (T2 - T1) -> Positive means T1 has a better (lower) rank
            rank_1 = f"{role}_rank_{sys}_1"
            rank_2 = f"{role}_rank_{sys}_2"
            if rank_1 in df.columns and rank_2 in df.columns:
                df[f"{role}_rank_{sys}_diff"] = df[rank_2] - df[rank_1]
                features.append(f"{role}_rank_{sys}_diff")
            
    return df, features
