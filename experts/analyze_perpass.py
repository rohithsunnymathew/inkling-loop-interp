#!/usr/bin/env python3
"""Per-pass expert routing for Inkling-2-Mini (core layers 6-17 run 3 times).
Visit order: prelude 0-5 | pass1 6-17 | pass2 slots 24-35 | pass3 slots 36-47 | coda 18-23"""
import sys, glob, os, json
import numpy as np
D = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, sys.argv[1])
from gguf import GGUFReader
ORDER = list(range(0, 18)) + list(range(24, 48)) + list(range(18, 24))
def label(s):
    if s < 6: return f"L{s}"
    if s < 18: return f"L{s}·p1"
    if s < 24: return f"L{s}"
    p = 2 if s < 36 else 3; return f"L{6 + (s - 24) % 12}·p{p}"
def load(path):
    c = {}
    for t in GGUFReader(path).tensors:
        if t.name.endswith(".counts") and "ffn_gate_up_exps" in t.name:
            c[int(t.name.split(".")[1])] = np.asarray(t.data, dtype=np.float64).ravel()
    return c
C = {os.path.basename(p)[:-5]: load(p) for p in sorted(glob.glob(f"{D}/imatrix_perpass/*.gguf"))}
k0 = next(iter(C)); slots = sorted(C[k0])
print("slots found:", len(slots), "| selections per slot (first file):", sorted({int(C[k0][s].sum()) for s in slots}))
assert len(slots) == 48, "expected 48 layer visits; patch may not be active"
doms = sorted({k.rsplit("_", 1)[0] for k in C})
def jsd(p, q):
    p = (p + .5) / (p + .5).sum(); q = (q + .5) / (q + .5).sum(); m = (p + q) / 2
    return .5 * (p * np.log(p / m)).sum() + .5 * (q * np.log(q / m)).sum()
spec = {}
for i, a in enumerate(doms):
    for b in doms[i + 1:]:
        spec[f"{a}-{b}"] = np.array([
            np.mean([jsd(C[f"{a}_{x}"][s], C[f"{b}_{y}"][s]) for x in "AB" for y in "AB"])
            - np.mean([jsd(C[f"{d}_A"][s], C[f"{d}_B"][s]) for d in (a, b)]) for s in ORDER])
print("\nmean topic specialisation (JSD beyond split-half noise):")
segs = {"prelude": range(0, 6), "core p1": range(6, 18), "core p2": range(18, 30), "core p3": range(30, 42), "coda": range(42, 48)}
for k, v in spec.items(): print(f"  {k:17s} " + "  ".join(f"{n} {v[list(r)].mean():.3f}" for n, r in segs.items()))
print("\nsecond half of core (L12-17) by pass:")
for p, base in ((1, 12), (2, 30), (3, 42)):
    idx = [ORDER.index(s) for s in ([12,13,14,15,16,17] if p == 1 else range(base, base + 6))]
    print(f"  pass {p}: " + "  ".join(f"{k} {v[idx].mean():.3f}" for k, v in spec.items()))
# how much does routing change between passes of the same layer (same text, pooled halves)?
print("\nrouting drift between passes, same text (JSD, pooled over files):")
for ip in range(6, 18):
    s1, s2, s3 = ip, 24 + ip - 6, 36 + ip - 6
    tot = {s: sum(C[k][s] for k in C) for s in (s1, s2, s3)}
    print(f"  L{ip:2d}: p1→p2 {jsd(tot[s1], tot[s2]):.3f}   p2→p3 {jsd(tot[s2], tot[s3]):.3f}   p1→p3 {jsd(tot[s1], tot[s3]):.3f}")
json.dump({"order": [label(s) for s in ORDER], "spec": {k: v.tolist() for k, v in spec.items()}}, open(f"{D}/perpass_report.json", "w"))
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(12, 3.8))
for n, r, c in (("pass 1", (6, 18), "0.93"), ("pass 2", (18, 30), "0.85"), ("pass 3", (30, 42), "0.93")):
    ax.axvspan(r[0] - .5, r[1] - .5, color=c); ax.text((r[0] + r[1]) / 2 - .5, 0.99, f"core {n}", ha="center", va="top", transform=ax.get_xaxis_transform(), fontsize=8)
for k, v in spec.items(): ax.plot(range(48), v, marker="o", ms=2.5, lw=1.5, label=k)
ax.axhline(0, color="0.5", lw=.8, ls=":"); ax.set_xticks(range(0, 48, 2)); ax.set_xticklabels([label(s) for s in ORDER][::2], rotation=60, fontsize=7)
ax.set_ylabel("topic routing difference\nbeyond split-half noise"); ax.legend(fontsize=8, loc="upper left")
ax.set_title("Inkling-2-Mini: expert routing by topic at each of the 48 layer visits", fontsize=10)
fig.tight_layout(); fig.savefig(f"{D}/inkling2_mini_perpass_routing.png", dpi=130)
