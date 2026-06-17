import torch
from pathlib import Path
from pprint import pprint

ckpt = torch.load(Path("surrogate_model.pt"), map_location="cpu")
info = ckpt["info"]

print("=== info keys ===")
print(list(info.keys()))

print("\n=== info full structure ===")
pprint(info, width=120, sort_dicts=False)

print("\n=== model_state_dict shapes ===")
for key, value in ckpt["model_state_dict"].items():
    print(f"{key:30s} {tuple(value.shape)}")
