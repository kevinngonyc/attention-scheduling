import random
from pathlib import Path

OUT = Path("inputs")
random.seed(0)

def random_range(batch, lo, hi):
    return [random.randint(lo, hi) for _ in range(batch)]

def bimodal(batch, short, long, long_frac):
    n_long = max(1, round(batch * long_frac))
    lens = [long] * n_long + [short] * (batch - n_long)
    random.shuffle(lens)
    return lens

def write(name, workloads):
    OUT.mkdir(exist_ok=True)
    with open(OUT / f"{name}.txt", "w") as f:
        for w in workloads:
            f.write(" ".join(map(str, w)) + "\n")
    print(f"Wrote {name}.txt")

write("edge", [[8], [64, 128], [65, 129], [1, 100, 2, 200, 3, 300]])
write("skew", [bimodal(64, 128, 128 * r, 0.05) for r in [1, 2, 4, 8, 16, 32]])
write("frac", [bimodal(64, 128, 4096, f) for f in [0.02, 0.05, 0.1, 0.25, 0.5]])
write("batch", [random_range(b, 64, 2048) for b in [8, 16, 32, 64, 128, 256]])