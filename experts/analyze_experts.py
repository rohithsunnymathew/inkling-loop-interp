#!/usr/bin/env python3
"""Expert-routing specialisation per layer, from llama-imatrix counts.
For each layer: how differently do two domains use the 128 experts,
beyond the noise seen between two halves of the same domain?
  between = mean JSD over the 4 cross-domain half pairs
  within  = mean JSD of (A vs B) inside each domain
  specialisation = between - within   (> 0 means real domain preference)
Note: core layers (6-17) run 3 times, so their counts pool 3 passes."""
import sys, glob, os, json
import numpy as np
D = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, sys.argv[1])
from gguf import GGUFReader
L, CORE = 24, (6, 18)

def load(path):
    r = GGUFReader(path); c = {}
    for t in r.tensors:
        if t.name.endswith(".counts") and "ffn_gate_up_exps" in t.name:
            l = int(t.name.split(".")[1]); c[l] = np.asarray(t.data, dtype=np.float64).ravel()
    return np.stack([c[l] for l in range(L)])          # (L, 128)

C = {os.path.basename(p)[:-5]: load(p) for p in sorted(glob.glob(f"{D}/imatrix/*.gguf"))}
doms = sorted({k.rsplit("_", 1)[0] for k in C})
print("loaded", list(C), "| selections per layer:", {k: int(v[0].sum()) for k, v in C.items()})

def jsd(p, q):
    p = (p + .5) / (p + .5).sum(); q = (q + .5) / (q + .5).sum(); m = (p + q) / 2
    kl = lambda a, b: (a * np.log(a / b)).sum()
    return .5 * kl(p, m) + .5 * kl(q, m)

core = np.zeros(L, bool); core[CORE[0]:CORE[1]] = True
res = {}
for i, a in enumerate(doms):
    for b in doms[i + 1:]:
        spec = []
        for l in range(L):
            bt = np.mean([jsd(C[f"{a}_{x}"][l], C[f"{b}_{y}"][l]) for x in "AB" for y in "AB"])
            wi = np.mean([jsd(C[f"{d}_A"][l], C[f"{d}_B"][l]) for d in (a, b)])
            spec.append(bt - wi)
        spec = np.array(spec); res[f"{a}-{b}"] = spec
        print(f"{a:>8s} vs {b:<8s} specialisation  prelude {spec[:6].mean():.3f}  core {spec[core].mean():.3f}  coda {spec[18:].mean():.3f}")

# experts that prefer security, replicated in both halves vs both controls
ctrl = [d for d in doms if d != "security"]
def share(k, l): v = C[k][l]; return (v + .5) / (v + .5).sum()
top = {}
for l in range(L):
    lr = []
    for e in range(C["security_A"].shape[1]):
        r = [np.log(share(f"security_{h}", l)[e] / np.mean([share(f"{c}_{g}", l)[e] for c in ctrl for g in "AB"])) for h in "AB"]
        lr.append(min(r))                                  # must hold in BOTH halves
    lr = np.array(lr); o = np.argsort(-lr)[:5]
    top[l] = [(int(e), round(float(lr[e]), 2)) for e in o]
print("\nsecurity-preferring experts (log share ratio vs controls, worst of the two halves):")
for l in range(L): print(f"  layer {l:2d}{' core' if core[l] else '     '}: {top[l][:3]}")
json.dump({"spec": {k: v.tolist() for k, v in res.items()}, "top_security_experts": top}, open(f"{D}/expert_report.json", "w"))

import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(8, 3.6)); ax.axvspan(CORE[0] - .5, CORE[1] - .5, color="0.9", label="loop core")
for k, v in res.items(): ax.plot(range(L), v, marker="o", ms=3, lw=1.6, label=k)
ax.axhline(0, color="0.5", lw=.8, ls=":"); ax.set_xlabel("layer"); ax.set_ylabel("routing difference\nbeyond split-half noise (JSD)")
ax.set_title("Inkling-2-Mini: how differently each pair of domains uses the 128 experts", fontsize=10); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(f"{D}/inkling2_mini_expert_routing.png", dpi=130)
