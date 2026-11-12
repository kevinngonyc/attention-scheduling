import numpy as np
import torch
from pathlib import Path

d_k, d_v = 64, 64

def load(dir, name, dtype, cols=None):
    a = np.fromfile(f"{dir}/{name}", dtype=dtype)
    return torch.from_numpy(a if cols is None else a.reshape(-1, cols))

for folder in sorted(Path("data").iterdir()):
    if not folder.is_dir():
        continue

    cu = load(folder, "cu_seqlens.bin", np.int32)
    q = load(folder, "q.bin", np.float16, d_k).float()
    k = load(folder, "k.bin", np.float16, d_k).float()
    v = load(folder, "v.bin", np.float16, d_v).float()
    o = load(folder, "o.bin", np.float16, d_v).float()

    ref = torch.empty(q.shape[0], d_v)
    for i in range(len(cu) - 1):
        start, end = cu[i].item(), cu[i + 1].item()
        scores = q[start:end] @ k[start:end].T / d_k ** 0.5
        ref[start:end] = torch.softmax(scores, dim=-1) @ v[start:end]

    max_abs_diff = torch.abs(ref - o).max()
    print(f"{folder.name}: error = {max_abs_diff:.2e}")