
import json
from betting_app.services.thesis_inference_service import build_thesis_features_for_match, ALL_FEATURES
from betting_app.core.db import transaction

def test_feature_building():
    # Manual test case: T1 vs Gen.G
    team_a = "T1"
    team_b = "Gen.G"
    id_a = 1848
    id_b = 845
    
    print(f"Testing with match: {team_a} vs {team_b}")
    print(f"IDs: {id_a} vs {id_b}")

    vec, diag = build_thesis_features_for_match(
        team_a,
        team_b,
        team_a_golgg_id=id_a,
        team_b_golgg_id=id_b,
        best_of=3
    )

    if vec is None:
        print("Failed to build features:")
        print(json.dumps(diag, indent=2))
        return

    print(f"Feature vector shape: {vec.shape}")
    print(f"Diagnostics: {json.dumps(diag, indent=2)}")

    # Print some key features
    feat_dict = dict(zip(ALL_FEATURES, vec[0]))
    print("\nKey Features:")
    for f in ["player_elo", "player_gl", "player_ts", "player_os", "player_pl", "player_tm"]:
        print(f"{f}: {feat_dict[f]:.4f}")
    
    print("\nUncertainty Features:")
    for f in ["player_elo_min1", "player_gl_max1", "player_gl_rd_avg1", "player_ts_sigma_avg1"]:
        print(f"{f}: {feat_dict[f]:.4f}")

if __name__ == "__main__":
    test_feature_building()
