# inspect_ckpt.py
import torch
from pathlib import Path
ckpt_path = Path("models/sequence/FD001/lstm_best.pth")
print("loading:", ckpt_path)
state = torch.load(ckpt_path, map_location="cpu")
# get raw state dict if wrapped
sd = state.get("model_state", state) if isinstance(state, dict) else state
print("Type of sd:", type(sd))
print("Keys in state_dict:", list(sd.keys())[:60])
# print shapes for top-level tensors
for k, v in list(sd.items()):
    if hasattr(v, "shape"):
        print(k, tuple(v.shape))
