"""Fast repair of truncated golgg_matches.json.
Reads file backwards in chunks from error position to find last complete match."""
import json
import os
import sys

JSON_PATH = "data/golgg_matches.json"
ERROR_POS = 2217047703  # from JSONDecodeError on server

print(f"File: {JSON_PATH}")
print(f"Size: {os.path.getsize(JSON_PATH):,} bytes")
print(f"Error pos: {ERROR_POS:,}")

# Read the file up to error position to find last complete match
# Use raw_decode for correct brace counting (handles strings)
with open(JSON_PATH, "rb") as f:
    data = f.read(ERROR_POS)

print(f"Loaded {len(data):,} bytes into memory")

# Parse with raw_decode to find the last complete object
decoder = json.JSONDecoder()
pos = 0
last_valid_end = 0
count = 0

# Skip initial whitespace and opening bracket
text = data.decode("utf-8", errors="replace")
pos = 0
while pos < len(text) and text[pos] in " \t\n\r":
    pos += 1
if pos < len(text) and text[pos] == "[":
    pos += 1
    while pos < len(text) and text[pos] in " \t\n\r":
        pos += 1

while pos < len(text):
    try:
        obj, end = decoder.raw_decode(text, pos)
        pos = end
        last_valid_end = end
        count += 1
        # Skip whitespace and commas
        while pos < len(text) and text[pos] in " \t\n\r,":
            pos += 1
    except (json.JSONDecodeError, ValueError) as e:
        print(f"Parse stopped at char {pos}: {e}")
        break

print(f"Parsed {count} complete objects")
print(f"Last valid object ends at char: {last_valid_end}")

# Truncate and repair
repair_path = JSON_PATH + ".repaired"
with open(JSON_PATH, "rb") as src:
    out = src.read(last_valid_end)

with open(repair_path, "wb") as dst:
    dst.write(out)
    dst.write(b"\n]\n")

new_size = os.path.getsize(repair_path)
print(f"\nRepaired file: {new_size:,} bytes ({new_size/1024/1024:.1f}MB)")
print(f"Removed: {(os.path.getsize(JSON_PATH)-new_size)/1024/1024:.1f}MB of corrupted data")

# Verify
print("\nVerifying...")
try:
    with open(repair_path, "r", encoding="utf-8") as f:
        parsed = json.load(f)
    print(f"✅ Valid! {len(parsed)} matches.")
    
    dates = {m.get("date") for m in parsed if m.get("date")}
    if dates:
        print(f"Date range: {min(dates)} to {max(dates)}")
    else:
        print("No dates found in matches")
except Exception as e:
    print(f"❌ Invalid: {e}")
