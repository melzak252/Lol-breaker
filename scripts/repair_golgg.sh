#!/bin/bash
# Fast repair of truncated golgg_matches.json
# Strategy: find the last complete match by counting braces

JSON_PATH="/app/data/golgg_matches.json"
ERROR_POS=2217047703

# Read only up to error position and find last complete match
python3 -c "
import json, os

JSON_PATH = '$JSON_PATH'
error_pos = $ERROR_POS

print(f'File size: {os.path.getsize(JSON_PATH):,} bytes')

with open(JSON_PATH, 'rb') as f:
    data = f.read(error_pos)

print(f'Read {len(data):,} bytes')

# Fast byte-level scan for brace depth
# We track positions where depth returns to 1 (end of match)
depth = 0
in_str = False
esc = False
last_match_end = 0

# Pre-decode as text for faster char processing
# But for 2.1GB this is still a lot. Let's only scan the last 50MB
# where the last matches probably are

# Actually let's just find the last '}' where depth becomes 1
# by scanning backwards from error_pos

# Count total braces in prefix
opens = data.count(b'{')
closes = data.count(b'}')
# But this doesn't account for braces in strings...

print(f'Total braces: {opens} open, {closes} close')
print(f'Net depth at error: {opens - closes}')
" 2>&1