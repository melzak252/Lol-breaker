"""Repair truncated golgg_matches.json - find last complete match by tracking brace depth."""
import json, sys

JSON_PATH = "data/golgg_matches.json"

# Read file in chunks and find positions where brace depth returns to 1 (array level)
# Structure: [ {match1}, {match2}, ... ]
# Depth 1 = inside array, depth 0 = before/after array

batch_size = 10 * 1024 * 1024  # 10MB chunks
total_matches = 0
truncate_pos = 0
brace_depth = 0
in_string = False
escaped = False
last_complete_pos = 0  # position after last complete match's closing brace
found_complete = False

print("Scanning JSON file for last complete match...")
with open(JSON_PATH, "rb") as f:
    while True:
        chunk = f.read(batch_size)
        if not chunk:
            break
        
        # Track if we're inside a string to avoid counting braces in strings
        for i, byte in enumerate(chunk):
            char = chr(byte)
            
            if escaped:
                escaped = False
                continue
            
            if char == '\\' and in_string:
                escaped = True
                continue
            
            if char == '"':
                in_string = not in_string
                continue
            
            if in_string:
                continue
            
            if char == '{':
                brace_depth += 1
            elif char == '}':
                brace_depth -= 1
                if brace_depth == 1:
                    # End of a match object
                    pos = f.tell() - len(chunk) + i + 1
                    last_complete_pos = pos
                    found_complete = True
                    total_matches += 1
                    if total_matches % 10000 == 0:
                        print(f"  ... {total_matches} matches found (at {pos//1024//1024}MB)...", end='\r')
                        sys.stdout.flush()

# Now truncate
if not found_complete:
    print("No complete matches found!")
    sys.exit(1)

print(f"\nTotal complete matches found: {total_matches}")
print(f"Truncating at byte: {last_complete_pos}")

repair_path = JSON_PATH + ".repaired"
with open(JSON_PATH, "rb") as src:
    data = src.read(last_complete_pos)

with open(repair_path, "wb") as dst:
    dst.write(data)
    dst.write(b"\n]\n")

import os
orig_size = os.path.getsize(JSON_PATH)
new_size = os.path.getsize(repair_path)
print(f"Original size: {orig_size:,} bytes ({orig_size/1024/1024:.1f}MB)")
print(f"Repaired size:  {new_size:,} bytes ({new_size/1024/1024:.1f}MB)")
print(f"Removed:        {(orig_size - new_size):,} bytes ({(orig_size - new_size)/1024/1024:.1f}MB)")

# Verify
print("\nVerifying repaired JSON...")
try:
    with open(repair_path, "r", encoding="utf-8") as f:
        parsed = json.load(f)
    print(f"✅ Valid JSON! {len(parsed)} matches.")
    
    # Date range
    dates = set()
    for m in parsed:
        d = m.get("date")
        if d:
            dates.add(d)
    if dates:
        print(f"Date range: {min(dates)} to {max(dates)}")
except Exception as e:
    print(f"❌ Still invalid: {e}")
