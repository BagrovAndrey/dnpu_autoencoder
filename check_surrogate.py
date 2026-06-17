import torch
from pathlib import Path

print("Torch version:", torch.__version__)

try:
    import brainspy
    print("brainspy imported successfully")
except ImportError as e:
    print("brainspy import failed:", e)

path = Path("surrogate_model.pt")

if not path.exists():
    raise FileNotFoundError(f"Could not find {path.resolve()}")

print("\nLoading:", path.resolve())
ckpt = torch.load(path, map_location="cpu")

print("\nCheckpoint type:", type(ckpt))

if isinstance(ckpt, dict):
    print("\nTop-level keys:")
    for key in ckpt.keys():
        print("  ", repr(key))

    print("\nDetailed summary:")
    for key, value in ckpt.items():
        if isinstance(value, dict):
            print(f"{key}: dict with {len(value)} keys")
            print("   first keys:", list(value.keys())[:20])
        elif isinstance(value, list):
            print(f"{key}: list with length {len(value)}")
            print("   first items:", value[:5])
        elif isinstance(value, tuple):
            print(f"{key}: tuple with length {len(value)}")
            print("   first items:", value[:5])
        elif torch.is_tensor(value):
            print(f"{key}: tensor, shape={tuple(value.shape)}, dtype={value.dtype}")
        else:
            print(f"{key}: {type(value)} -> {value}")
else:
    print("\nNon-dict checkpoint:")
    print(ckpt)
