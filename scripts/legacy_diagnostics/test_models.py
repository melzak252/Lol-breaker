import torch
import os
import sys

models_dir = "/app/models"
files = [
    "fusion_v2_core.pt",
    "fusion_v2_sym_aug.pt",
    "fusion_v2_arch_sym.pt",
    "transformer_checkpoint.pt"
]

print(f"Python version: {sys.version}")
print(f"Torch version: {torch.__version__}")

for f in files:
    path = os.path.join(models_dir, f)
    if os.path.exists(path):
        print(f"Checking {f}...")
        try:
            sd = torch.load(path, map_location='cpu')
            print(f"  Successfully loaded {f}, keys: {len(sd.keys())}")
        except Exception as e:
            print(f"  Error loading {f}: {e}")
    else:
        print(f"File {f} NOT FOUND at {path}")
