from betting_app.services.thesis_inference_service import predict_upcoming_with_thesis_model
import pandas as pd

print("Generating predictions for upcoming matches...")
results = predict_upcoming_with_thesis_model(include_past=False)

if results:
    df = pd.DataFrame(results)
    # Flatten diagnostics for display
    df['side_swap_fixed'] = df['diagnostics'].apply(lambda d: d.get('side_swap_fixed', False))
    df['mapping_source'] = df['diagnostics'].apply(lambda d: d.get('golgg_resolved', {}).get('from_ids', False))
    
    print(f"\nGenerated {len(df)} predictions:")
    cols = ['match', 'prob_a', 'prob_b', 'side_swap_fixed', 'mapping_source']
    print(df[cols].to_string(index=False))
else:
    print("No upcoming matches found or all skipped due to missing data.")
