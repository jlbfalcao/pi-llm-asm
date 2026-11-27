#!/usr/bin/env python3
"""Trains a tiny character-level LLM (Llama/Qwen3 style: RMSNorm + RoPE + GQA + SwiGLU)
on TinyShakespeare and exports it to the PIML format read by the bare-metal
AArch64 kernel.

Usage: python3 tools/train.py --steps 2000 --out build/model.piml
"""
import argparse, math, struct, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

p = argparse.ArgumentParser()
p.add_argument("--data", default="data/input.txt")
p.add_argument("--out", default="build/model.piml")
p.add_argument("--ckpt", default="build/model.pt")
p.add_argument("--dim", type=int, default=192)
p.add_argument("--layers", type=int, default=6)
p.add_argument("--heads", type=int, default=6)
p.add_argument("--kv_heads", type=int, default=2)
p.add_argument("--hidden", type=int, default=512)
p.add_argument("--seq", type=int, default=256)
p.add_argument("--batch", type=int, default=32)
p.add_argument("--steps", type=int, default=2000)
p.add_argument("--lr", type=float, default=1.5e-3)
p.add_argument("--threads", type=int, default=2)
p.add_argument("--export_only", action="store_true")
a = p.parse_args()
torch.manual_seed(1337)
torch.set_num_threads(a.threads)

text = open(a.data, encoding="utf-8").read()
chars = sorted(set(text))
assert all(ord(c) < 128 for c in chars), "vocab must be ASCII"
stoi = {c: i for i, c in enumerate(chars)}
data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
n = int(0.9 * len(data))
train_d, val_d = data[:n], data[n:]
V = len(chars)


def rope_tables(hd, seq, theta=10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, hd, 2).float() / hd))
    ang = torch.outer(torch.arange(seq).float(), freqs)
    return ang.cos(), ang.sin()


def apply_rope(x, cos, sin):
    # x: (B,T,H,hd) ; interleaved pairs (2i, 2i+1) as in llama2.c
    x0, x1 = x[..., 0::2], x[..., 1::2]
    c, s = cos[None, :, None, :], sin[None, :, None, :]
    out = torch.stack([x0 * c - x1 * s, x0 * s + x1 * c], dim=-1)
    return out.flatten(-2)


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-5):
        super().__init__(); self.eps = eps; self.w = nn.Parameter(torch.ones(d))
    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.w


class Block(nn.Module):
    def __init__(s):
        super().__init__()
        hd = a.dim // a.heads
        s.hd = hd
        s.n1, s.n2 = RMSNorm(a.dim), RMSNorm(a.dim)
        s.wq = nn.Linear(a.dim, a.heads * hd, bias=False)
        s.wk = nn.Linear(a.dim, a.kv_heads * hd, bias=False)
        s.wv = nn.Linear(a.dim, a.kv_heads * hd, bias=False)
        s.wo = nn.Linear(a.heads * hd, a.dim, bias=False)
        s.w1 = nn.Linear(a.dim, a.hidden, bias=False)
        s.w2 = nn.Linear(a.hidden, a.dim, bias=False)
        s.w3 = nn.Linear(a.dim, a.hidden, bias=False)

    def forward(s, x, cos, sin):
        B, T, _ = x.shape
        h = s.n1(x)
        q = apply_rope(s.wq(h).view(B, T, a.heads, s.hd), cos, sin)
        k = apply_rope(s.wk(h).view(B, T, a.kv_heads, s.hd), cos, sin)
        v = s.wv(h).view(B, T, a.kv_heads, s.hd)
        rep = a.heads // a.kv_heads
        k = k.repeat_interleave(rep, dim=2); v = v.repeat_interleave(rep, dim=2)
        o = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2),
                                           v.transpose(1, 2), is_causal=True)
        x = x + s.wo(o.transpose(1, 2).reshape(B, T, -1))
        h = s.n2(x)
        return x + s.w2(F.silu(s.w1(h)) * s.w3(h))


class LM(nn.Module):
    def __init__(s):
        super().__init__()
        s.emb = nn.Embedding(V, a.dim)
        s.blocks = nn.ModuleList([Block() for _ in range(a.layers)])
        s.norm = RMSNorm(a.dim)
        cos, sin = rope_tables(a.dim // a.heads, a.seq)
        s.register_buffer("cos", cos, persistent=False)
        s.register_buffer("sin", sin, persistent=False)
        for m in s.modules():
            if isinstance(m, (nn.Linear, nn.Embedding)):
                nn.init.normal_(m.weight, std=0.02)
        for b in s.blocks:  # residual scaling (GPT-2 style)
            nn.init.normal_(b.wo.weight, std=0.02 / math.sqrt(2 * a.layers))
            nn.init.normal_(b.w2.weight, std=0.02 / math.sqrt(2 * a.layers))

    def forward(s, idx):
        T = idx.shape[1]
        x = s.emb(idx)
        for b in s.blocks:
            x = b(x, s.cos[:T], s.sin[:T])
        return s.norm(x) @ s.emb.weight.T  # tied embeddings


model = LM()
nparams = sum(p.numel() for p in model.parameters())
print(f"vocab={V} params={nparams/1e6:.2f}M")


def batch(d):
    ix = torch.randint(len(d) - a.seq - 1, (a.batch,))
    x = torch.stack([d[i:i + a.seq] for i in ix])
    y = torch.stack([d[i + 1:i + a.seq + 1] for i in ix])
    return x, y


@torch.no_grad()
def val_loss(iters=20):
    model.eval(); l = 0.0
    for _ in range(iters):
        x, y = batch(val_d); l += F.cross_entropy(model(x).view(-1, V), y.view(-1)).item()
    model.train(); return l / iters


if not a.export_only:
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.95), weight_decay=0.1)
    warm = 100
    t0 = time.time()
    for step in range(a.steps + 1):
        lr = a.lr * min(1.0, (step + 1) / warm) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * step / a.steps)))
        for g in opt.param_groups: g["lr"] = lr
        x, y = batch(train_d)
        loss = F.cross_entropy(model(x).view(-1, V), y.view(-1))
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if step % 100 == 0:
            vl = val_loss() if step % 500 == 0 else float("nan")
            print(f"step {step:5d} loss {loss.item():.4f} val {vl:.4f} lr {lr:.2e} {time.time()-t0:.0f}s", flush=True)
            torch.save(model.state_dict(), a.ckpt)
    torch.save(model.state_dict(), a.ckpt)
    print(f"final val_loss {val_loss(50):.4f}")
else:
    model.load_state_dict(torch.load(a.ckpt))

# ---------------- PIML v1 export ----------------
# header: 'PIML' u32 version, dim, hidden, layers, heads, kv_heads, vocab, seq  (9 x u32 = 36 B)
#         + 28 B padding  -> 64 B ; then 256 B with the vocab character table
# fp32 weights (little endian), 64 B aligned:
#   emb[V,D] | rms_att[L,D] | wq[L,D,D] | wk[L,KV,D] | wv[L,KV,D] | wo[L,D,D]
#   rms_ffn[L,D] | w1[L,H,D] | w2[L,D,H] | w3[L,H,D] | rms_final[D]
def f32(t): return t.detach().float().contiguous().numpy().astype("<f4").tobytes()
B = model.blocks
blob = bytearray()
blob += b"PIML" + struct.pack("<8I", 1, a.dim, a.hidden, a.layers, a.heads, a.kv_heads, V, a.seq)
blob += b"\0" * (64 - len(blob))
blob += "".join(chars).encode("ascii").ljust(256, b"\0")
blob += f32(model.emb.weight)
for name in ["n1", "wq", "wk", "wv", "wo", "n2", "w1", "w2", "w3"]:
    for b in B:
        m = getattr(b, name)
        blob += f32(m.w if name in ("n1", "n2") else m.weight)
blob += f32(model.norm.w)
open(a.out, "wb").write(blob)
print(f"exported {a.out}: {len(blob)} bytes")
