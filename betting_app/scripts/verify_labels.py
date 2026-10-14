#!/usr/bin/env python3
"""Verify that transformer and baseline labels match after fix."""
import json
import pandas as pd

# Load transformer data
with open('data/transformer_team_sequences_v1.json', 'r') as f:
    transformer_data = json.load(f)

# Load baseline data
baseline_df = pd.read_csv('data/golgg_y_predicts.csv')

# Create lookup from baseline using correct column name
baseline_lookup = {}
for _, row in baseline_df.iterrows():
    match_id = int(row['golgg_match_id'])
    baseline_lookup[match_id] = int(row['y_true'])

# Compare labels
mismatches = 0
total = 0
for sample in transformer_data:
    match_id = sample['match_id']
    if match_id in baseline_lookup:
        total += 1
        transformer_label = sample['y']
        baseline_label = baseline_lookup[match_id]
        if transformer_label != baseline_label:
            mismatches += 1

print(f"Total matches compared: {total}")
print(f"Mismatches: {mismatches}")
print(f"Mismatch rate: {mismatches/total*100:.2f}%")
