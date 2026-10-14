"""Repair truncated golgg_matches.json by finding the last valid match object."""
import json

JSON_PATH = "data/golgg_matches.json"

error_pos = 2217047703  # from JSONDecodeError
print(f"Error reported at position: {error_pos}")

# Read backwards from error position to find last complete match
with open(JSON_PATH, "rb") as f:
    # Read the last ~100KB around the error
    read_start = max(0, error_pos - 100000)
    f.seek(read_start)
    chunk = f.read(200000)  # read generous chunk

chunk_str = chunk.decode("utf-8", errors="replace")

# Find position of error within chunk
chunk_error_offset = error_pos - read_start
print(f"Error at offset {chunk_error_offset} within chunk (len={len(chunk_str)})")

# We need the last complete '}\n' before the error that is a match-end (not nested)
# Strategy: find all '}\n' positions before the error, check which one is a valid JSON separator
# A match ends with: `}` followed by `,\n` or `\n]`

# Find the last complete match by scanning backwards
# Look for "\n    }" pattern (match closing) followed by "," or "\n]"
before_error = chunk_str[:chunk_error_offset]

# Find all occurrences of "}\n" that could be match boundaries
# A match looks like: ...\n    }\n        ]... or ...\n    }\n    {
import re

# Pattern for end of a match (closing brace at indentation level 4 spaces)
# Match objects end with: \n    }  (4-space indent)
# Followed by: ,\n    {  (next match) or \n]  (end of array)
match_end_pattern = re.compile(r'\n    \}\n')

matches = list(match_end_pattern.finditer(before_error))
if not matches:
    print("No match endings found!")
    exit(1)

last_match_end = matches[-1]
truncate_pos = read_start + last_match_end.end()  # position after '}\n'
print(f"Last complete match ends at char {truncate_pos}")
print(f"Context around truncation: {repr(chunk_str[last_match_end.start():last_match_end.start()+100])}")

# Check what follows - should be ,\n    { or \n]
after_truncate = read_start + min(last_match_end.end() + 10, len(chunk_str))
if truncate_pos < error_pos:
    f = open(JSON_PATH, "rb")
    f.seek(truncate_pos)
    next_chars = f.read(50).decode("utf-8", errors="replace")
    f.close()
    print(f"Next chars after truncation: {repr(next_chars)}")

# Repair: truncate file at truncate_pos and add closing bracket
repair_path = JSON_PATH + ".repaired"
with open(JSON_PATH, "rb") as src:
    data = src.read(truncate_pos)

with open(repair_path, "wb") as dst:
    dst.write(data)
    dst.write(b"\n]\n")

print(f"Repaired file written to {repair_path}")
print(f"Original: {error_pos} bytes corrupted; Repaired: {truncate_pos} bytes + closing")

# Verify the repaired file
import json
print("\nVerifying repaired JSON...")
try:
    with open(repair_path, "r", encoding="utf-8") as f:
        parsed = json.load(f)
    print(f"✅ Valid JSON! {len(parsed)} matches loaded successfully.")
    
    # Check dates
    from datetime import datetime
    dates = set()
    for m in parsed:
        d = m.get("date")
        if d:
            dates.add(d)
    if dates:
        print(f"Date range: {min(dates)} to {max(dates)}")
    
except Exception as e:
    print(f"❌ Still invalid: {e}")
