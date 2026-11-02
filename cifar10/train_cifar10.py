"""
CIFAR-10: train DSM score models and track the gradient / solenoidal split of the
score error during training (Figure 3).

Every `eval_every` epochs the EMA weights are frozen and Algorithm 1 is run:
a freshly initialised critic phi is trained to maximise
    J = E[ 2 grad_x phi . r  -  |grad_x phi|^2 ],    r = s_theta - grad log p(x_t | x_0),
whose value estimates ||Pi_G e||^2 (gradient part); the full error is E|r|^2.
Results are written to <out_dir>/ch<c>/seed<s>/hist.json, read by plot_cifar10.py.

Usage:  python train_cifar10.py --device cuda:0
"""
import math, json, argparse
from pathlib import Path
from collections import defaultdict
from contextlib import contextmanager, nullcontext

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as T
from tqdm import tqdm

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════════════════════
CFG = dict(
    data_dir       = "./data",
    n_res          = 2,
    time_dim       = 256,
    dropout        = 0.1,
    beta_min       = 0.1,
    beta_max       = 20.0,
    T_sde          = 1.0,
    t0             = 1e-3,
    # ── training ──────────────────────────────────────────────────────────────
    epochs         = 350,
    batch_size     = 128,
    lr             = 2e-4,
    lr_min         = 1e-5,
    warmup_steps   = 3000,
    clip_grad      = 1.0,
    ema_decay      = 0.9999,
    weight_decay   = 0.01,
    min_snr_gamma  = 5.0,
    # ── diagnostic (Algorithm 1) ──────────────────────────────────────────────
    eval_every     = 10,
    n_eval_batches = 50,      # 50 x 128 = 6,400 noised training samples, fixed for the whole run
    n_critic_eval  = 4000,    # critic optimisation steps per evaluation
    critic_lr      = 1e-3,
    # ── runs ──────────────────────────────────────────────────────────────────
    channels       = [96],
    seeds          = [45, 47, 49, 52],
    num_workers    = 4,
    out_dir        = "runs",
    seed           = 42,      # seeds the fixed evaluation set
)

_pa = argparse.ArgumentParser()
_pa.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
_args = _pa.parse_args()
device = torch.device(_args.device)
print(f"Device: {device}")

C, H, W = 3, 32, 32
D = C * H * W

_amp_ok = device.type == "cuda"
_amp_dt = (torch.bfloat16 if (_amp_ok and torch.cuda.is_bf16_supported())
           else torch.float16)

def amp():
    return (torch.amp.autocast("cuda", _amp_dt) if _amp_ok else nullcontext())

def make_scaler():
    return torch.amp.GradScaler(enabled=(_amp_ok and _amp_dt == torch.float16))

# ══════════════════════════════════════════════════════════════════════════════
#  VP-SDE
# ══════════════════════════════════════════════════════════════════════════════
def _int_beta(t):
    return CFG["beta_min"] * t + 0.5 * (CFG["beta_max"] - CFG["beta_min"]) * t ** 2

def alpha(t):   return torch.exp(-0.5 * _int_beta(t))
def sigma(t):   return (1.0 - torch.exp(-_int_beta(t)) + 1e-8).sqrt()

def randt(n):
    t0 = CFG["t0"]
    return torch.rand(n, device=device) * (CFG["T_sde"] - t0) + t0

def q_sample(x0, t):
    a, s = alpha(t)[:, None], sigma(t)[:, None]
    eps  = torch.randn_like(x0)
    return a * x0 + s * eps, eps

def min_snr_weight(t, gamma=5.0):
    """Min-SNR-gamma loss weighting (Hang et al. 2023): min(SNR, gamma) / SNR."""
    a = alpha(t)
    s = sigma(t).clamp(min=1e-3)
    snr = (a / s) ** 2
    return torch.clamp(snr, max=gamma) / snr

# ══════════════════════════════════════════════════════════════════════════════
#  DATA
# ══════════════════════════════════════════════════════════════════════════════
class DequantizeOnly(nn.Module):
    """Add uniform noise in [0, 1/255]."""
    def forward(self, x):
        return (x + torch.rand_like(x) / 255.0).clamp(0, 1)

def make_loaders():
    train_tf = T.Compose([
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        DequantizeOnly(),
        T.Lambda(lambda x: (2.0 * x - 1.0).reshape(-1)),
    ])
    eval_tf = T.Compose([
        T.ToTensor(),
        T.Lambda(lambda x: (2.0 * x - 1.0).reshape(-1)),
    ])
    train_ds = torchvision.datasets.CIFAR10(
        CFG["data_dir"], train=True, download=True, transform=train_tf)
    eval_ds = torchvision.datasets.CIFAR10(
        CFG["data_dir"], train=True, download=False, transform=eval_tf)
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=CFG["batch_size"], shuffle=True, drop_last=True,
        pin_memory=(device.type == "cuda"), num_workers=CFG["num_workers"])
    # no shuffle: the diagnostic always uses the same training images
    eval_loader = torch.utils.data.DataLoader(
        eval_ds, batch_size=128, shuffle=False, drop_last=False,
        pin_memory=(device.type == "cuda"), num_workers=CFG["num_workers"])
    return train_loader, eval_loader

def make_fixed_eval_tuples(loader, n_batches):
    """Noised samples (x_t, t, eps) drawn once and reused at every evaluation."""
    fixed = []
    it = iter(loader)
    for _ in range(n_batches):
        try:
            x0, _ = next(it)
        except StopIteration:
            break
        t = randt(len(x0)).detach().cpu()
        eps = torch.randn_like(x0).detach().cpu()
        xt = alpha(t)[:, None] * x0 + sigma(t)[:, None] * eps
        fixed.append({"t": t.clone(), "eps": eps.clone(), "xt": xt.clone()})
    return fixed

# ══════════════════════════════════════════════════════════════════════════════
#  SCORE NETWORK  (predicts the noise eps)
# ══════════════════════════════════════════════════════════════════════════════
class SinEmb(nn.Module):
    def __init__(self, dim):
        super().__init__(); self.dim = dim
    def forward(self, t):
        half = self.dim // 2
        freq = torch.exp(-math.log(10000) *
                         torch.arange(half, device=t.device) / (half - 1))
        e = t[:, None] * freq[None]
        return torch.cat([e.sin(), e.cos()], dim=-1)

class ResBlock(nn.Module):
    def __init__(self, ic, oc, td, dropout=0.0):
        super().__init__()
        gn_ic = min(32, ic); gn_oc = min(32, oc)
        self.b1 = nn.Sequential(nn.GroupNorm(gn_ic, ic), nn.SiLU(),
                                nn.Conv2d(ic, oc, 3, padding=1))
        self.tp = nn.Sequential(nn.SiLU(), nn.Linear(td, oc))
        self.b2 = nn.Sequential(nn.GroupNorm(gn_oc, oc), nn.SiLU(),
                                nn.Dropout(dropout),
                                nn.Conv2d(oc, oc, 3, padding=1))
        self.sk = nn.Conv2d(ic, oc, 1) if ic != oc else nn.Identity()
    def forward(self, x, te):
        h = self.b1(x) + self.tp(te)[:, :, None, None]
        return self.b2(h) + self.sk(x)

class SelfAttn(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.norm = nn.GroupNorm(min(32, ch), ch)
        self.qkv  = nn.Conv2d(ch, ch * 3, 1, bias=False)
        self.proj = nn.Conv2d(ch, ch, 1)
    def forward(self, x):
        B, Ck, Hk, Wk = x.shape
        q, k, v = self.qkv(self.norm(x)).reshape(B, 3, Ck, Hk * Wk).unbind(1)
        a = (q.transpose(-2, -1) @ k * Ck ** -0.5).softmax(-1)
        return x + self.proj((v @ a.transpose(-2, -1)).reshape(B, Ck, Hk, Wk))

class ScoreNet(nn.Module):
    def __init__(self, base_ch=128, n_res=2, dropout=0.0):
        super().__init__()
        ch = [base_ch, base_ch * 2, base_ch * 2]
        td = CFG["time_dim"] * 4
        self.temb = nn.Sequential(SinEmb(CFG["time_dim"]),
            nn.Linear(CFG["time_dim"], td), nn.SiLU(), nn.Linear(td, td))
        self.head = nn.Conv2d(C, ch[0], 3, padding=1)
        self.enc1 = nn.ModuleList([ResBlock(ch[0], ch[0], td, dropout) for _ in range(n_res)])
        self.dwn1 = nn.Conv2d(ch[0], ch[0], 3, stride=2, padding=1)
        self.enc2 = nn.ModuleList([ResBlock(ch[0] if i==0 else ch[1], ch[1], td, dropout) for i in range(n_res)])
        self.att2 = SelfAttn(ch[1])
        self.dwn2 = nn.Conv2d(ch[1], ch[1], 3, stride=2, padding=1)
        self.enc3 = nn.ModuleList([ResBlock(ch[1] if i==0 else ch[2], ch[2], td, dropout) for i in range(n_res)])
        self.att3 = SelfAttn(ch[2])
        self.dwn3 = nn.Conv2d(ch[2], ch[2], 3, stride=2, padding=1)
        self.mid = nn.ModuleList([ResBlock(ch[2], ch[2], td, dropout), SelfAttn(ch[2]),
                                  ResBlock(ch[2], ch[2], td, dropout)])
        self.up3 = nn.ConvTranspose2d(ch[2], ch[2], 4, stride=2, padding=1)
        self.dec3 = nn.ModuleList([ResBlock(ch[2]*2 if i==0 else ch[2], ch[2], td, dropout) for i in range(n_res)])
        self.dat3 = SelfAttn(ch[2])
        self.up2 = nn.ConvTranspose2d(ch[2], ch[1], 4, stride=2, padding=1)
        self.dec2 = nn.ModuleList([ResBlock(ch[1]*2 if i==0 else ch[1], ch[1], td, dropout) for i in range(n_res)])
        self.dat2 = SelfAttn(ch[1])
        self.up1 = nn.ConvTranspose2d(ch[1], ch[0], 4, stride=2, padding=1)
        self.dec1 = nn.ModuleList([ResBlock(ch[0]*2 if i==0 else ch[0], ch[0], td, dropout) for i in range(n_res)])
        self.out = nn.Sequential(nn.GroupNorm(min(32, ch[0]), ch[0]), nn.SiLU(),
                                 nn.Conv2d(ch[0], C, 3, padding=1))

    def forward(self, x, t):
        te = self.temb(t)
        h = self.head(x.view(-1, C, H, W))
        for b in self.enc1: h = b(h, te)
        s1 = h; h = self.dwn1(h)
        for b in self.enc2: h = b(h, te)
        h = self.att2(h); s2 = h; h = self.dwn2(h)
        for b in self.enc3: h = b(h, te)
        h = self.att3(h); s3 = h; h = self.dwn3(h)
        h = self.mid[0](h, te); h = self.mid[1](h); h = self.mid[2](h, te)
        h = self.up3(h); h = torch.cat([h, s3], 1)
        for b in self.dec3: h = b(h, te)
        h = self.dat3(h)
        h = self.up2(h); h = torch.cat([h, s2], 1)
        for b in self.dec2: h = b(h, te)
        h = self.dat2(h)
        h = self.up1(h); h = torch.cat([h, s1], 1)
        for b in self.dec1: h = b(h, te)
        return self.out(h).view(-1, D)

# ══════════════════════════════════════════════════════════════════════════════
#  CRITIC  (scalar potential phi(x, t); its x-gradient is a gradient field)
# ══════════════════════════════════════════════════════════════════════════════
class CriticResBlock(nn.Module):
    def __init__(self, ic, oc, td):
        super().__init__()
        self.conv1 = nn.Conv2d(ic, oc, 3, padding=1)
        self.conv2 = nn.Conv2d(oc, oc, 3, padding=1)
        self.temb  = nn.Sequential(nn.SiLU(), nn.Linear(td, oc))
        self.skip  = nn.Conv2d(ic, oc, 1) if ic != oc else nn.Identity()
        self.norm1 = nn.GroupNorm(min(32, ic), ic)
        self.norm2 = nn.GroupNorm(min(32, oc), oc)
    def forward(self, x, te):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.temb(te)[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)

class CriticUNet(nn.Module):
    def __init__(self, ch_base=64):
        super().__init__()
        td = CFG["time_dim"] * 2
        self.temb = nn.Sequential(
            SinEmb(CFG["time_dim"]),
            nn.Linear(CFG["time_dim"], td), nn.SiLU(), nn.Linear(td, td))
        self.head = nn.Conv2d(C, ch_base, 3, padding=1)
        self.enc1 = CriticResBlock(ch_base, ch_base, td)
        self.down1 = nn.Conv2d(ch_base, ch_base, 3, stride=2, padding=1)
        self.enc2 = CriticResBlock(ch_base, ch_base*2, td)
        self.down2 = nn.Conv2d(ch_base*2, ch_base*2, 3, stride=2, padding=1)
        self.mid   = CriticResBlock(ch_base*2, ch_base*2, td)
        self.up2   = nn.ConvTranspose2d(ch_base*2, ch_base*2, 4, stride=2, padding=1)
        self.dec2  = CriticResBlock(ch_base*4, ch_base, td)
        self.up1   = nn.ConvTranspose2d(ch_base, ch_base, 4, stride=2, padding=1)
        self.dec1  = CriticResBlock(ch_base*2, ch_base, td)
        self.out = nn.Sequential(
            nn.GroupNorm(min(32, ch_base), ch_base), nn.SiLU(),
            nn.Conv2d(ch_base, 1, 3, padding=1))
        nn.init.zeros_(self.out[-1].weight)
        nn.init.zeros_(self.out[-1].bias)

    def forward(self, x, t):
        te = self.temb(t)
        h = self.head(x.view(-1, C, H, W))
        h1 = self.enc1(h, te)
        h = self.down1(h1)
        h2 = self.enc2(h, te)
        h = self.down2(h2)
        h = self.mid(h, te)
        h = self.up2(h)
        h = self.dec2(torch.cat([h, h2], 1), te)
        h = self.up1(h)
        h = self.dec1(torch.cat([h, h1], 1), te)
        return self.out(h).sum(dim=(1, 2, 3))

    def grad_x(self, x, t, create_graph=False):
        with torch.set_grad_enabled(True):
            xin = x.detach().float().requires_grad_(True)
            phi = self(xin, t).sum()
            g = torch.autograd.grad(phi, xin, create_graph=create_graph)[0]
        return g

def critic_ch_for_score(base_ch):
    if base_ch <= 32:
        return 64
    elif base_ch <= 64:
        return 96
    return 128

# ══════════════════════════════════════════════════════════════════════════════
#  EMA
# ══════════════════════════════════════════════════════════════════════════════
class EMA:
    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = {k: v.clone().float() for k, v in model.state_dict().items()}
    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            self.shadow[k].mul_(self.decay).add_(v.float(), alpha=1 - self.decay)
    @contextmanager
    def scope(self, model):
        backup = {k: v.clone() for k, v in model.state_dict().items()}
        dev = next(model.parameters()).device
        model.load_state_dict({k: v.to(dev) for k, v in self.shadow.items()})
        try:    yield
        finally: model.load_state_dict(backup)

# ══════════════════════════════════════════════════════════════════════════════
#  DIAGNOSTIC (Algorithm 1)
# ══════════════════════════════════════════════════════════════════════════════
def get_lr(global_step, total_steps):
    w = CFG["warmup_steps"]
    hi, lo = CFG["lr"], CFG["lr_min"]
    if global_step < w: return hi * global_step / max(1, w)
    p = (global_step - w) / max(1, total_steps - w)
    return lo + 0.5 * (hi - lo) * (1 + math.cos(math.pi * p))

def score_residual(eps_pred, eps_true, t):
    """r = s_theta - grad log p(x_t | x_0), for a network that predicts eps."""
    return (eps_true - eps_pred) / sigma(t)[:, None].clamp(min=1e-3)

def forcing_J(u, r):
    """J = E[2 u.r - |u|^2], with squared norms averaged over the D pixel values."""
    return (2.0 * (u * r).mean(-1) - (u ** 2).mean(-1)).mean()

def eval_error_decomposition(score_net, fixed_eval, n_critic_steps, critic_ch,
                             critic_lr=1e-3):
    """Train a fresh critic on the fixed noised samples, then evaluate
    full = E|r|^2 and grad = J(critic) on the same samples."""
    score_net.eval()
    ec = CriticUNet(ch_base=critic_ch).to(device)
    oc = optim.Adam(ec.parameters(), lr=critic_lr)
    sched = optim.lr_scheduler.CosineAnnealingLR(oc, T_max=n_critic_steps, eta_min=1e-5)

    # critic training pass
    it = iter(fixed_eval)
    for step in range(n_critic_steps):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(fixed_eval)
            batch = next(it)
        t, eps, xt = (batch[k].to(device) for k in ("t", "eps", "xt"))
        with torch.no_grad():
            eps_pred = score_net(xt, t)
        r = score_residual(eps_pred, eps, t).float()
        u = ec.grad_x(xt, t, create_graph=True)
        Jv = forcing_J(u, r.detach())
        oc.zero_grad(set_to_none=True)
        (-Jv).backward()
        nn.utils.clip_grad_norm_(ec.parameters(), 1.0)
        oc.step()
        sched.step()

    # evaluation pass
    ec.eval()
    full_vals, grad_vals = [], []
    for batch in fixed_eval:
        t, eps, xt = (batch[k].to(device) for k in ("t", "eps", "xt"))
        with torch.no_grad():
            eps_pred = score_net(xt, t)
        e = score_residual(eps_pred, eps, t).float()
        full_vals.append((e ** 2).mean(-1).mean().item())
        with torch.no_grad():
            u = ec.grad_x(xt, t)
        grad_vals.append(forcing_J(u, e).item())

    full = float(np.mean(full_vals))
    grad = max(0.0, float(np.mean(grad_vals)))
    sol = max(0.0, full - grad)
    return {"full": full, "grad": grad, "sol": sol,
            "grad_pct": 100 * grad / max(full, 1e-10),
            "sol_pct": 100 * sol / max(full, 1e-10)}

# ══════════════════════════════════════════════════════════════════════════════
#  TRAINING
# ══════════════════════════════════════════════════════════════════════════════
def save_hist(hist, path):
    with open(path, "w") as f:
        json.dump({k: [float(x) if isinstance(x, (float, np.floating)) else
                       int(x) if isinstance(x, (int, np.integer)) else x
                       for x in vv] for k, vv in hist.items()}, f, indent=2)

def train_one(base_ch, out, train_loader, fixed_eval):
    out.mkdir(exist_ok=True, parents=True)
    score = ScoreNet(base_ch, CFG["n_res"], CFG["dropout"]).to(device)
    n_p = sum(p.numel() for p in score.parameters() if p.requires_grad)
    print(f"\n  base_ch={base_ch}  ScoreNet={n_p/1e6:.1f}M  -> {out}")

    ema = EMA(score, CFG["ema_decay"])
    opt = optim.AdamW(score.parameters(), lr=0.0, weight_decay=CFG["weight_decay"])
    scaler = make_scaler()
    critic_ch = critic_ch_for_score(base_ch)

    total_steps = CFG["epochs"] * len(train_loader)
    global_step = 0
    hist = defaultdict(list)

    for epoch in tqdm(range(1, CFG["epochs"] + 1), desc=f"ch={base_ch}"):
        score.train()
        ep_loss = 0.0
        for x0, _ in train_loader:
            x0 = x0.to(device, non_blocking=True)
            for pg in opt.param_groups: pg["lr"] = get_lr(global_step, total_steps)
            global_step += 1

            t = randt(len(x0))
            xt, eps = q_sample(x0, t)
            opt.zero_grad(set_to_none=True)
            with amp():
                eps_pred = score(xt, t)
                w_snr = min_snr_weight(t, CFG["min_snr_gamma"]).detach()
                loss = (w_snr * ((eps_pred - eps) ** 2).mean(-1)).mean()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(score.parameters(), CFG["clip_grad"])
            scaler.step(opt); scaler.update()
            ema.update(score)
            ep_loss += loss.item()

        hist["epoch"].append(epoch)
        hist["loss_dsm"].append(ep_loss / len(train_loader))

        if epoch % CFG["eval_every"] == 0 or epoch == CFG["epochs"]:
            with ema.scope(score):
                dec = eval_error_decomposition(score, fixed_eval, CFG["n_critic_eval"],
                                               critic_ch, CFG["critic_lr"])
            hist["eval_epoch"].append(epoch)
            for k in ("full", "grad", "sol"):
                hist[f"dsm_{k}_l2"].append(dec[k])
            for k in ("grad_pct", "sol_pct"):
                hist[f"dsm_{k}"].append(dec[k])
            print(f"  [{base_ch}] ep {epoch}: full={dec['full']:.3f}  "
                  f"grad={dec['grad']:.3f} ({dec['grad_pct']:.1f}%)  sol={dec['sol']:.3f}")
            save_hist(hist, out / "hist.json")

    del score, ema, opt
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    torch.manual_seed(CFG["seed"]); torch.cuda.manual_seed_all(CFG["seed"])
    train_loader, eval_loader = make_loaders()
    fixed_eval = make_fixed_eval_tuples(eval_loader, CFG["n_eval_batches"])
    print(f"Stored {len(fixed_eval)} fixed eval batches")

    for ch in CFG["channels"]:
        for seed in CFG["seeds"]:
            torch.manual_seed(seed); torch.cuda.manual_seed_all(seed); np.random.seed(seed)
            train_one(ch, Path(CFG["out_dir"]) / f"ch{ch}" / f"seed{seed}",
                      train_loader, fixed_eval)


if __name__ == "__main__":
    main()
