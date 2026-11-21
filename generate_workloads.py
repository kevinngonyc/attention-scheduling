import random
from pathlib import Path

OUT = Path("inputs")
random.seed(0)

# Token budget shared by every workload in the skew and frac sweeps, so those
# sweeps vary only how the tokens are distributed, not how many there are.
TOTAL_TOKENS = 64 * 512

def random_range(batch, lo, hi):
    return [random.randint(lo, hi) for _ in range(batch)]

def bimodal(batch, ratio, long_frac, total=TOTAL_TOKENS):
    """`batch` sequences summing to exactly `total` tokens: round(batch * long_frac)
    (at least 1) long ones, each `ratio` times as long as the short ones."""
    n_long = max(1, round(batch * long_frac))
    n_short = batch - n_long
    short = total // (n_short + ratio * n_long)
    lens = [ratio * short] * n_long + [short] * n_short
    # Spread the leftover tokens from the integer division one per sequence.
    for i in range(total - sum(lens)):
        lens[i % batch] += 1
    random.shuffle(lens)
    assert sum(lens) == total
    return lens

def write(name, workloads):
    OUT.mkdir(exist_ok=True)
    with open(OUT / f"{name}.txt", "w") as f:
        for w in workloads:
            f.write(" ".join(map(str, w)) + "\n")
    print(f"Wrote {name}.txt")

write("edge", [[8], [64, 128], [65, 129], [1, 100, 2, 200, 3, 300]])
write("skew", [bimodal(64, r, 0.05) for r in [1, 2, 4, 8, 16, 32]])
write("frac", [bimodal(64, 32, f) for f in [0.02, 0.05, 0.1, 0.25, 0.5]])
write("batch", [random_range(b, 64, 2048) for b in [8, 16, 32, 64, 128, 256]])

SHORTS = sorted(random.randint(16, 128) for _ in range(512))
LONG_TILES = [4, 8, 16, 32, 64, 128]
write("ascending", [SHORTS + [64 * k] for k in LONG_TILES])
write("descending", [[64 * k] + SHORTS[::-1] for k in LONG_TILES])
