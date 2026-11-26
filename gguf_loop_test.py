#!/usr/bin/env python3
"""Blind weight-space loop test on a GGUF checkpoint (numpy only).
Compares RESIDUAL-SIDE subspaces only (the one basis all layers share):
  read side  (right singular vectors) of q, k, v, gate, rel, router
  write side (left singular vectors)  of attn_output
then finds the contiguous block of layers that is most mutually similar
beyond what layer distance predicts, with a permutation p-value.
The loop core from the file's metadata is NOT used by the search; it is
only printed afterwards to grade the result."""
import sys, json, time, argparse
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("gguf"); ap.add_argument("--gguf-py", required=True)
ap.add_argument("--k", type=int, default=64); ap.add_argument("--perms", type=int, default=1000)
ap.add_argument("--out", required=True)
ap.add_argument("--only", default=None, help="comma list of types to (re)compute")
ap.add_argument("--combine", action="store_true")
a = ap.parse_args()
sys.path.insert(0, a.gguf_py)
from gguf import GGUFReader
from gguf.quants import dequantize

r = GGUFReader(a.gguf)
T = {t.name: t for t in r.tensors}
def kv(name):
    f = r.fields.get(name); return None if f is None else f.contents()
arch = kv("general.architecture"); L = int(kv(f"{arch}.block_count"))
truth = (kv(f"{arch}.loop.start"), kv(f"{arch}.loop.end"), kv(f"{arch}.loop.count"))

def W(name):
    t = T[name]
    x = dequantize(t.data, t.tensor_type).astype(np.float32)
    return x.reshape(int(t.shape[1]), int(t.shape[0]))      # (out, in)

def basis(w, side, k):
    # residual side: 'in' -> right singular vectors, 'out' -> left
    m = w.T @ w if side == "in" else w @ w.T
    ev, vec = np.linalg.eigh(m.astype(np.float64))
    return vec[:, ::-1][:, :k].astype(np.float32)

TYPES = {"attn_q": "in", "attn_k": "in", "attn_v": "in", "attn_gate": "in",
         "attn_rel": "in", "ffn_gate_inp": "in", "attn_output": "out"}

def ov(A, B): return float(((A.T @ B) ** 2).sum() / A.shape[1])

def detrend(M):
    n = len(M); R = M.copy()
    for d in range(1, n):
        i = np.arange(n - d); mu = np.median(M[i, i + d])
        R[i, i + d] -= mu; R[i + d, i] -= mu
    np.fill_diagonal(R, 0); return R

def wscore(R, s, e):
    n = len(R); ins = np.zeros(n, bool); ins[s:e] = True; m = e - s
    off = ~np.eye(m, dtype=bool)
    win = R[np.ix_(ins, ins)][off].mean(); wout = R[np.ix_(ins, ~ins)].mean()
    return (win - wout) * np.sqrt(m * (m - 1) / 2), win, wout

def best(R, lo=3, hi=16):
    b = (-1e9, 0, 0)
    for m in range(lo, min(hi, len(R) - 1) + 1):
        for s in range(len(R) - m + 1):
            sc = wscore(R, s, s + m)[0]
            if sc > b[0]: b = (sc, s, s + m)
    return b

rng = np.random.default_rng(0)
import os
out = {"truth_loop_core": [int(truth[0]), int(truth[1]) - 1], "passes": int(truth[2]), "types": {}}
Ms = []
only = set(a.only.split(",")) if a.only else set(TYPES)
for ty, side in TYPES.items():
    t0 = time.time()
    cache = f"{a.out}/type_{ty}.json"
    if os.path.exists(cache) and (a.combine or ty not in only):
        res = json.load(open(cache)); out["types"][ty] = res; Ms.append(np.array(res["M"])); continue
    if a.combine or ty not in only: continue
    names = [f"blk.{l}.{ty}.weight" for l in range(L)]
    if not all(n in T for n in names): print("skip", ty); continue
    k = min(a.k, int(T[names[0]].shape[1]) // 2) if side == "in" and ty == "ffn_gate_inp" else a.k
    Bs = [basis(W(n), side, k) for n in names]
    M = np.array([[ov(Bs[i], Bs[j]) for j in range(L)] for i in range(L)])
    Ms.append(M)
    R = detrend(M); sc, s, e = best(R)
    # p for the searched block (search repeated under each shuffle)
    hits = 0; core_hits = 0
    cs = wscore(R, truth[0], truth[1])[0]
    for _ in range(a.perms):
        p = rng.permutation(L); Rp = detrend(M[np.ix_(p, p)])
        hits += best(Rp)[0] >= sc
        core_hits += wscore(Rp, truth[0], truth[1])[0] >= cs
    res = {"best_block": [int(s), int(e - 1)], "score": float(sc), "core_window_score_": 0, "p_search": (hits + 1) / (a.perms + 1),
           "core_window_score": float(cs), "p_core_window": (core_hits + 1) / (a.perms + 1),
           "six_blocks": [float(wscore(R, b, b + 6)[0]) for b in (0, 6, 12, 18)],
           "random_baseline": k / 2304, "M": M.tolist()}
    out["types"][ty] = res; json.dump(res, open(cache, "w"))
    print(f"{ty:13s} best {s:2d}-{e-1:2d}  p={res['p_search']:.3f} | core 6-17 score {cs:+.2f} p={res['p_core_window']:.3f} "
          f"| 6-blocks {[round(x,2) for x in res['six_blocks']]}  [{time.time()-t0:.0f}s]", flush=True)

if not a.combine: sys.exit(0)
for ty, res in out["types"].items():
    print(f"{ty:13s} best {res['best_block'][0]:2d}-{res['best_block'][1]:2d}  p={res['p_search']:.3f} | core 6-17 p={res['p_core_window']:.3f} | 6-blocks {[round(x,2) for x in res['six_blocks']]}")
Mavg = np.mean(Ms, 0); R = detrend(Mavg); sc, s, e = best(R)
out["combined"] = {"best_block": [int(s), int(e - 1)], "score": float(sc)}
print(f"\nCOMBINED best block {s}-{e-1}   (truth: loop core {truth[0]}-{truth[1]-1}, {truth[2]} passes)")
json.dump(out, open(a.out + "/report.json", "w"))
try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
    for axi, m, ttl in ((ax[0], Mavg, "residual-subspace overlap (avg of 7 matrix types)"),
                        (ax[1], R, "after removing layer-distance trend")):
        im = axi.imshow(m, cmap="viridis"); axi.set_title(ttl, fontsize=9)
        axi.set_xlabel("layer"); axi.set_ylabel("layer"); fig.colorbar(im, ax=axi, fraction=.046)
        axi.add_patch(plt.Rectangle((truth[0]-.5, truth[0]-.5), truth[1]-truth[0], truth[1]-truth[0], fill=False, ec="white", ls="--", lw=1.5))
    ax[1].add_patch(plt.Rectangle((s-.5, s-.5), e-s, e-s, fill=False, ec="red", lw=2))
    fig.suptitle(f"Inkling-2-Mini: blind best block {s}-{e-1} (red) vs true loop core {truth[0]}-{truth[1]-1} (white dashed)", fontsize=10)
    fig.tight_layout(); fig.savefig(a.out + "/inkling2_mini_weightspace.png", dpi=120)
except ImportError: print("no matplotlib; skipped plot")
