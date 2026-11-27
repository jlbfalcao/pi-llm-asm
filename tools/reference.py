#!/usr/bin/env python3
"""NumPy (float32) reference of the kernel forward pass: reads the .piml and generates greedily.
Usage: python3 tools/reference.py build/model.piml "ROMEO:" 40
"""
import struct, sys
import numpy as np

path, prompt, n_new = sys.argv[1], sys.argv[2], int(sys.argv[3])
raw = open(path, "rb").read()
assert raw[:4] == b"PIML"
_, D, H, L, NH, NKV, V, S = struct.unpack("<8I", raw[4:36])
itos = raw[64:64 + V].decode("ascii")
stoi = {c: i for i, c in enumerate(itos)}
w = np.frombuffer(raw, dtype="<f4", offset=320)
off = 0
def take(*shape):
    global off
    n = int(np.prod(shape)); a = w[off:off + n].reshape(shape); off += n; return a
hd = D // NH; KV = NKV * hd
emb = take(V, D); rms_att = take(L, D); wq = take(L, D, D); wk = take(L, KV, D)
wv = take(L, KV, D); wo = take(L, D, D); rms_ffn = take(L, D)
w1 = take(L, H, D); w2 = take(L, D, H); w3 = take(L, H, D); rms_f = take(D)
assert off == len(w)

f = np.float32
freq = np.exp(-(np.arange(0, hd, 2, dtype=f) / f(hd)) * f(np.log(10000.0))).astype(f)
ang = np.outer(np.arange(S, dtype=f), freq).astype(f)
cos, sin = np.cos(ang).astype(f), np.sin(ang).astype(f)
kc = np.zeros((L, S, KV), f); vc = np.zeros((L, S, KV), f)

def rms(x, g): return (x * (f(1) / np.sqrt(np.mean(x * x) + f(1e-5)))).astype(f) * g
def rope(v, p):
    v = v.reshape(-1, hd // 2, 2).copy(); a, b = v[..., 0].copy(), v[..., 1].copy()
    v[..., 0] = a * cos[p] - b * sin[p]; v[..., 1] = a * sin[p] + b * cos[p]
    return v.reshape(-1)

def forward(tok, p):
    x = emb[tok].copy()
    for l in range(L):
        h = rms(x, rms_att[l])
        q = rope(wq[l] @ h, p); kc[l, p] = rope(wk[l] @ h, p); vc[l, p] = wv[l] @ h
        out = np.zeros(D, f)
        for hh in range(NH):
            kv = hh // (NH // NKV)
            qh = q[hh * hd:(hh + 1) * hd]
            K = kc[l, :p + 1, kv * hd:(kv + 1) * hd]; Vv = vc[l, :p + 1, kv * hd:(kv + 1) * hd]
            s = (K @ qh) / np.sqrt(f(hd)); s = np.exp(s - s.max()); s /= s.sum()
            out[hh * hd:(hh + 1) * hd] = s @ Vv
        x = x + wo[l] @ out
        h = rms(x, rms_ffn[l]); a = w1[l] @ h
        x = x + w2[l] @ ((a / (f(1) + np.exp(-a))) * (w3[l] @ h))
    return emb @ rms(x, rms_f)

toks = [stoi[c] for c in prompt if c in stoi]
for i, t in enumerate(toks): logits = forward(t, i)
out, p = "", len(toks)
for _ in range(n_new):
    t = int(np.argmax(logits)); out += itos[t]
    if p >= S: break
    logits = forward(t, p); p += 1
sys.stdout.write(prompt + out)
