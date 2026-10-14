from betting_app.services.thesis_inference_service import predict_upcoming_with_thesis_model
import json

def test_full_prediction():
    print("Running predict_upcoming_with_thesis_model()...")
    results = predict_upcoming_with_thesis_model(limit=5)
    
    print(f"Generated {len(results)} predictions.")
    
    for res in results:
        print(f"\nMatch: {res['match']}")
        print(f"Prob A: {res['prob_a']:.4f}, Prob B: {res['prob_b']:.4f}")
        diag = res['diagnostics']
        print(f"GOL.GG Resolved from IDs: {diag.get('golgg_resolved', {}).get('from_ids')}")
        print(f"Rosters found: A={diag.get('team_a_golgg_name') is not None}, B={diag.get('team_b_golgg_name') is not None}")
        print(f"Missing data: {diag.get('missing')}")
        if 'side_consistency' in diag:
            print(f"Side consistency: {diag['side_consistency']}")

if __name__ == "__main__":
    test_full_prediction()
