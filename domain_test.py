#!/usr/bin/env python3
"""Domain-direction test in weight space (Inkling-2-Mini GGUF).
Question: do cybersecurity token directions load more onto the loop core
(layers 6-17) than control-domain directions (cooking, sport) do?

READ view : how much of each domain's (normed, centred) input-embedding
            directions falls inside a layer's top-k read subspace
            (q, k, v, router). Strictly valid for early layers; deeper
            layers see a transformed residual, so it's a proxy there.
WRITE view: how much of a layer's attn_output write subspace points at the
            domain's unembedding rows (does the layer write "towards"
            security tokens?).
Each score is divided by the same score for a random vocab baseline, then
security minus control is compared core vs non-core, with a label-
shuffling permutation test."""
import sys, os, json, argparse, time
import numpy as np
ap = argparse.ArgumentParser()
ap.add_argument("gguf"); ap.add_argument("--gguf-py", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--bases", default=None); ap.add_argument("--analyze", action="store_true")
ap.add_argument("--k", type=int, default=64); ap.add_argument("--perms", type=int, default=2000)
a = ap.parse_args(); sys.path.insert(0, a.gguf_py)
from gguf import GGUFReader
from gguf.quants import dequantize
r = GGUFReader(a.gguf); T = {t.name: t for t in r.tensors}; L = 24; CORE = (6, 18)
SIDE = {"attn_q": "in", "attn_k": "in", "attn_v": "in", "ffn_gate_inp": "in", "attn_output": "out"}

def W(name):
    t = T[name]; x = dequantize(t.data, t.tensor_type).astype(np.float32)
    return x.reshape(int(t.shape[1]), int(t.shape[0]))
def rows(name, ids):
    t = T[name]; d = np.asarray(t.data)
    x = dequantize(d[ids], t.tensor_type).astype(np.float32)
    return x.reshape(len(ids), int(t.shape[0]))

if a.bases:
    for ty in a.bases.split(","):
        side = SIDE[ty]; k = 32 if ty == "ffn_gate_inp" else a.k; B = []
        for l in range(L):
            w = W(f"blk.{l}.{ty}.weight"); m = w.T @ w if side == "in" else w @ w.T
            ev, v = np.linalg.eigh(m.astype(np.float64)); B.append(v[:, ::-1][:, :k].astype(np.float32))
        np.save(f"{a.out}/bases_{ty}.npy", np.stack(B)); print("saved", ty, flush=True)
    sys.exit(0)

# ---------------- analysis
SEC = """exploit malware vulnerability vulnerabilities attacker attackers ransomware phishing payload firewall
encryption breach backdoor privilege credential credentials password passwords authentication trojan botnet
intrusion exfiltration mitigation threat adversary rootkit keylogger spoofing injection overflow cipher
hacker hackers cyber cybersecurity antivirus spyware forensic forensics patch patched zero-day worm
decrypt encrypted ddos penetration sandbox brute attack attacks attacked security secure hack hacked hacking
virus viruses vulnerable exploits exploited malicious breaches threats proxy privileges certificate ssl tls vpn
cve xss csrf sql exploitation compromise compromised unauthorized auth spam fraud scam scams
tls encrypt firewall phish leak leaked""".split()
COOK = """recipe garlic onion onions simmer bake baking flour butter oven sauce pasta salt pepper roast boil
dough chopped tablespoon teaspoon cheese soup grill spices cinnamon vanilla sugar broth skillet marinade
knead whisk stir fry fried carrots potatoes tomato tomatoes cream lemon chicken beef bread cooking cook
cooked recipes ingredients ingredient baked delicious flavor flavour bowl kitchen meal dinner lunch breakfast
dessert cake cookies chocolate honey rice noodles vegetables vegetable meat pork salad eggs egg milk""".split()
SPORT = """football goal goals striker referee tournament cricket wicket rugby stadium coach league championship
penalty midfielder touchdown innings batting tennis racket sprint marathon defender goalkeeper playoff
umpire playoffs soccer basketball baseball hockey quarterback scored scoring athletes athlete olympic
medal team match teams players player win won wins game games season coaches league leagues ball
fans fan stadium score scores champion champions final finals victory defeat matches tournament cup trophy""".split()
toks = r.fields["tokenizer.ggml.tokens"].contents()
idx = {t: i for i, t in enumerate(toks)}
def ids(words):
    got = {}
    for w in dict.fromkeys(words):
        for v in ("Ġ" + w, w, "Ġ" + w.capitalize(), w.capitalize(), "Ġ" + w.upper(), w.upper()):
            if v in idx and len(v.lstrip("Ġ")) > 2: got[idx[v]] = v
    return list(got.values()), list(got.keys())
rng = np.random.default_rng(0)
cand = [i for i, t in enumerate(toks) if t.startswith("Ġ") and t[1:].isalpha() and t[1:].islower() and len(t) > 4]
base_ids = sorted(rng.choice(cand, 1500, replace=False).tolist())
dom = {"security": ids(SEC), "cooking": ids(COOK), "sport": ids(SPORT)}
for d, (w, i) in dom.items(): print(f"{d}: {len(i)} single-token words")

def prep(tname, normname, all_ids):
    X = rows(tname, all_ids)
    if normname in T: X = X * dequantize(T[normname].data, T[normname].tensor_type).astype(np.float32)[None]
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    return X
allid = sorted(set(base_ids) | {i for _, I in dom.values() for i in I})
pos = {i: n for n, i in enumerate(allid)}
E = prep("token_embd.weight", "token_embd_norm.weight", allid)
U = prep("output.weight", "output_norm.weight", allid)
mu_E = E[[pos[i] for i in base_ids]].mean(0); mu_U = U[[pos[i] for i in base_ids]].mean(0)
Ec = E - mu_E; Uc = U - mu_U
Ec /= np.linalg.norm(Ec, axis=1, keepdims=True); Uc /= np.linalg.norm(Uc, axis=1, keepdims=True)

def frac(X, B):   # per-token fraction of energy inside each layer's subspace: (L, n)
    return np.stack([((X @ B[l]) ** 2).sum(1) for l in range(L)])

out = {"words": {d: w for d, (w, _) in dom.items()}, "views": {}}
core = np.zeros(L, bool); core[CORE[0]:CORE[1]] = True
for ty, side in SIDE.items():
    B = np.load(f"{a.out}/bases_{ty}.npy"); X = Ec if side == "in" else Uc
    F = frac(X, B); fb = F[:, [pos[i] for i in base_ids]].mean(1)
    view = {"layer_ratio": {}}
    for d, (_, I) in dom.items(): view["layer_ratio"][d] = (F[:, [pos[i] for i in I]].mean(1) / fb).tolist()
    for ctrl in ("cooking", "sport"):
        si = [pos[i] for i in dom["security"][1]]; ci = [pos[i] for i in dom[ctrl][1]]
        def stat(s_idx, c_idx):
            diff = F[:, s_idx].mean(1) / fb - F[:, c_idx].mean(1) / fb
            return diff[core].mean() - diff[~core].mean()
        obs = stat(si, ci); pool = np.array(si + ci); hits = 0
        for _ in range(a.perms):
            p = rng.permutation(pool); hits += stat(list(p[:len(si)]), list(p[len(si):])) >= obs
        view[f"vs_{ctrl}"] = {"core_minus_rest": float(obs), "p": (hits + 1) / (a.perms + 1)}
    out["views"][ty] = view
    lr = view["layer_ratio"]
    print(f"{ty:13s} ({'read ' if side=='in' else 'write'}) "
          f"sec/cook core-rest {view['vs_cooking']['core_minus_rest']:+.3f} p={view['vs_cooking']['p']:.3f} | "
          f"sec/sport {view['vs_sport']['core_minus_rest']:+.3f} p={view['vs_sport']['p']:.3f}", flush=True)
json.dump(out, open(f"{a.out}/domain_report.json", "w"))
try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 5, figsize=(17, 3.4), sharex=True)
    cols = {"security": "#d62728", "cooking": "#2ca02c", "sport": "#1f77b4"}
    for ax, (ty, v) in zip(axs, out["views"].items()):
        ax.axvspan(CORE[0] - .5, CORE[1] - .5, color="0.9")
        for d, y in v["layer_ratio"].items(): ax.plot(range(L), y, color=cols[d], label=d, lw=1.6)
        ax.axhline(1, color="0.5", lw=.8, ls=":"); ax.set_title(f"{ty} ({'read' if SIDE[ty]=='in' else 'write'})", fontsize=9)
        ax.set_xlabel("layer")
    axs[0].set_ylabel("energy in layer subspace\n÷ random-word baseline"); axs[0].legend(fontsize=8)
    fig.suptitle("Inkling-2-Mini: domain token directions vs layer weights (grey = loop core 6-17)", fontsize=10)
    fig.tight_layout(); fig.savefig(f"{a.out}/inkling2_mini_domain.png", dpi=120)
except ImportError: pass
