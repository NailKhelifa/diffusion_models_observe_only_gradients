"""
Fashion-MNIST: gradient / solenoidal decomposition of the score error (Algorithm 1)
on every checkpoint written by train_fmnist.py.

Per checkpoint:
  1. full error  E|r|^2  with  r = s_theta(x_t, t) - grad log p(x_t | x_0)
     (summed over the 784 pixels);
  2. gradient part  ||Pi_G e||^2  =  sup_phi E[2 grad phi . r - |grad phi|^2],
     estimated by training a fresh critic potential phi;
  3. FID at that epoch, read from the final checkpoint's history.

Results are cached in <cache-dir>/per_checkpoint_score_dsm.json (safe to interrupt).
Writes, per capacity, error_decomposition_<cap>_score_dsm and kl_bounds_<cap>_score_dsm,
and the Spearman table <cache-dir>/spearman_table_score_dsm.json (plotted by plot_spearman.py).

Usage:
    python diagnostic_fmnist.py --device cuda:0 --results-root ./results
    python diagnostic_fmnist.py --plots-only          # redraw from the cache alone
"""
import argparse
import json
import math
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as T
from scipy.stats import spearmanr
from tqdm.auto import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["cmr10", "Computer Modern Roman", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "mathtext.rm": "serif",
    "axes.formatter.use_mathtext": True,
    "font.size": 18,
    "axes.labelsize": 18,
    "axes.titlesize": 18,
    "legend.fontsize": 18,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "figure.dpi": 200,
    "axes.linewidth": 0.8,
    "lines.linewidth": 2.0,
    "lines.markersize": 6,
    "legend.frameon": False,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.5,
    "xtick.top": True,
    "ytick.right": True,
    "xtick.direction": "in",
    "ytick.direction": "in",
})
COL_FULL = "#E63946"
COL_GRAD = "#2A9D8F"
COL_SOL  = "#6C5B7B"
COL_GAP  = "#8D99AE"


# ══════════════════════════════════════════════════════════════════════════════
# %% CLI + CFG
# ══════════════════════════════════════════════════════════════════════════════
parser = argparse.ArgumentParser()
parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
parser.add_argument("--results-root", default="./results")
parser.add_argument("--cache-dir", default="./diagnostic_cache")
parser.add_argument("--figures-dir", default="./figures_reproduced")
parser.add_argument("--data-dir", default="./data")
parser.add_argument("--plots-only", action="store_true",
                    help="Skip the diagnostic and redraw the figures from the cache.")
args = parser.parse_args()

device = torch.device(args.device)
print(f"device: {device}")

MODEL_KEY = "score_dsm"     # checkpoint key, also used in cache / figure names
CFG = dict(
    results_root = args.results_root,
    cache_dir    = args.cache_dir,
    figures_dir  = args.figures_dir,
    data_dir     = args.data_dir,

    # capacity -> (base_ch, n_res); must match the training sweep
    capacity_arch = dict(tiny=(32, 1), small=(48, 2), full=(64, 2)),

    # SDE (must match training): alpha = exp(-t/2), sigma = sqrt(1 - exp(-t)), log-uniform t
    T_sde = 4.0, t0 = 0.05,
    time_dim = 128,

    # diagnostic protocol
    n_eval_pool          = 6_400,
    n_eval_batches_full  = 50,
    eval_batch           = 128,
    critic_steps_per_ckpt= 1500,
    critic_batch         = 256,
    critic_base_ch       = 24,
    critic_lr            = 1e-3,
    critic_lr_min        = 1e-5,
    critic_clip          = 1.0,
    n_critic_restarts    = 1,

    # KL-bound prefactor K = 0.5 * (T - t0) * E_s[sigma_s^2], s log-uniform on [t0, T].
    # The published figures used a 200-draw Monte Carlo estimate of this expectation,
    # K = 0.8632 (the exact value is 0.8644); fixed here so the figures are reproduced exactly.
    K = 0.8632,
)
Path(CFG["cache_dir"]).mkdir(parents=True, exist_ok=True)
Path(CFG["figures_dir"]).mkdir(parents=True, exist_ok=True)

C, H, W = 1, 28, 28
D = C * H * W

_amp_ok = device.type == "cuda"
_amp_dt = (torch.bfloat16 if (_amp_ok and torch.cuda.is_bf16_supported())
           else torch.float16)
def amp():
    return torch.amp.autocast("cuda", _amp_dt) if _amp_ok else nullcontext()


# ══════════════════════════════════════════════════════════════════════════════
# %% SDE  (matches the training script)
# ══════════════════════════════════════════════════════════════════════════════
def alpha(t):
    return torch.exp(-0.5 * t)

def sigma(t):
    return torch.sqrt(1.0 - torch.exp(-t) + 1e-8)

def randt(n):
    """Log-uniform diffusion times in [t0, T], as in training."""
    return CFG["t0"] * torch.exp(torch.rand(n, device=device) * math.log(CFG["T_sde"] / CFG["t0"]))

def q_sample(x0, t):
    a, s = alpha(t)[:, None], sigma(t)[:, None]
    eps  = torch.randn_like(x0)
    return a * x0 + s * eps, eps

def s_target(xt, x0, t):
    """DSM target: the conditional score grad log p(x_t | x_0)."""
    a = alpha(t)[:, None]
    s = sigma(t)[:, None].clamp(min=1e-3)
    return -(xt - a * x0) / (s ** 2)


# ══════════════════════════════════════════════════════════════════════════════
# %% Score network  (same architecture as train_fmnist.py; outputs the score)
# ══════════════════════════════════════════════════════════════════════════════
class SinusoidalEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        half = dim // 2
        freqs = torch.exp(-math.log(10000) *
                          torch.arange(half).float() / (half - 1))
        self.register_buffer("freqs", freqs)
    def forward(self, t):
        x = t[:, None] * self.freqs[None, :]
        return torch.cat([x.sin(), x.cos()], dim=-1)


class TimeEmb(nn.Module):
    def __init__(self, sin_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            SinusoidalEmb(sin_dim),
            nn.Linear(sin_dim, out_dim), nn.SiLU(),
            nn.Linear(out_dim, out_dim),
        )
    def forward(self, t):
        return self.net(t)


class ResConvBlock(nn.Module):
    """FiLM-conditioned residual block."""
    def __init__(self, in_ch, out_ch, td):
        super().__init__()
        g1, g2 = min(8, in_ch), min(8, out_ch)
        self.norm1  = nn.GroupNorm(g1, in_ch)
        self.conv1  = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2  = nn.GroupNorm(g2, out_ch)
        self.conv2  = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.t_proj = nn.Linear(td, out_ch * 2)
        nn.init.zeros_(self.conv2.weight); nn.init.zeros_(self.conv2.bias)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, te):
        h = F.silu(self.norm1(x))
        h = self.conv1(h)
        sc = self.t_proj(te)[:, :, None, None]
        s, sh = sc.chunk(2, 1)
        h = F.silu(self.norm2(h) * (1 + s) + sh)
        h = self.conv2(h)
        return h + self.skip(x)


class SelfAttention2d(nn.Module):
    def __init__(self, ch):
        super().__init__()
        nh = 8
        while ch % nh != 0:
            nh -= 1
        self.norm = nn.GroupNorm(min(8, ch), ch)
        self.attn = nn.MultiheadAttention(ch, nh, batch_first=True)
        self.proj = nn.Conv2d(ch, ch, 1)
        nn.init.zeros_(self.proj.weight); nn.init.zeros_(self.proj.bias)

    def forward(self, x):
        B, Cc, Hh, Ww = x.shape
        h = self.norm(x).view(B, Cc, Hh * Ww).permute(0, 2, 1)
        h, _ = self.attn(h, h, h)
        return x + self.proj(h.permute(0, 2, 1).view(B, Cc, Hh, Ww))


class ScoreNet(nn.Module):
    def __init__(self, base_ch=64, n_res=2):
        super().__init__()
        td = CFG["time_dim"]
        Cs = [base_ch, base_ch * 2, base_ch * 4]

        self.time_emb = TimeEmb(td, td * 4)
        self.enc_in   = nn.Conv2d(1, Cs[0], 3, padding=1)

        def stack(ic, oc, n):
            blocks = [ResConvBlock(ic, oc, td * 4)]
            for _ in range(n - 1):
                blocks.append(ResConvBlock(oc, oc, td * 4))
            return nn.ModuleList(blocks)

        self.enc0 = stack(Cs[0], Cs[0], n_res)
        self.dn0  = nn.Conv2d(Cs[0], Cs[0], 3, stride=2, padding=1)
        self.enc1 = stack(Cs[0], Cs[1], n_res)
        self.dn1  = nn.Conv2d(Cs[1], Cs[1], 3, stride=2, padding=1)

        self.mid0 = ResConvBlock(Cs[1], Cs[2], td * 4)
        self.mida = SelfAttention2d(Cs[2])
        self.mid1 = ResConvBlock(Cs[2], Cs[2], td * 4)

        self.up1  = nn.Conv2d(Cs[2], Cs[1], 3, padding=1)
        self.dec1 = stack(Cs[1] * 2, Cs[1], n_res)
        self.up0  = nn.Conv2d(Cs[1], Cs[0], 3, padding=1)
        self.dec0 = stack(Cs[0] * 2, Cs[0], n_res)

        self.onorm = nn.GroupNorm(min(8, Cs[0]), Cs[0])
        self.oconv = nn.Conv2d(Cs[0], 1, 3, padding=1)
        nn.init.zeros_(self.oconv.weight); nn.init.zeros_(self.oconv.bias)

    def forward(self, x, t):
        B = x.shape[0]
        te = self.time_emb(t)
        h = self.enc_in(x.view(B, 1, 28, 28))
        for b in self.enc0: h = b(h, te)
        s0 = h
        h = self.dn0(h)
        for b in self.enc1: h = b(h, te)
        s1 = h
        h = self.dn1(h)
        h = self.mid0(h, te); h = self.mida(h); h = self.mid1(h, te)
        h = F.interpolate(self.up1(h), scale_factor=2, mode="nearest")
        h = torch.cat([h, s1], 1)
        for b in self.dec1: h = b(h, te)
        h = F.interpolate(self.up0(h), scale_factor=2, mode="nearest")
        h = torch.cat([h, s0], 1)
        for b in self.dec0: h = b(h, te)
        return self.oconv(F.silu(self.onorm(h))).view(B, D)


def valid_gn_groups(ch, max_groups=32):
    """Largest num_groups <= max_groups that divides ch."""
    for g in range(min(max_groups, ch), 0, -1):
        if ch % g == 0:
            return g
    return 1

# ══════════════════════════════════════════════════════════════════════════════
# %% Critic potential phi(x, t)  (scalar; its x-gradient is a gradient field)
# ══════════════════════════════════════════════════════════════════════════════
class _CritResBlock(nn.Module):
    def __init__(self, ic, oc, td, dropout=0.0):
        super().__init__()
        gn_ic = valid_gn_groups(ic, 32)
        gn_oc = valid_gn_groups(oc, 32)

        self.b1 = nn.Sequential(
            nn.GroupNorm(gn_ic, ic), nn.SiLU(),
            nn.Conv2d(ic, oc, 3, padding=1),
        )
        self.tp = nn.Sequential(nn.SiLU(), nn.Linear(td, oc))
        self.b2 = nn.Sequential(
            nn.GroupNorm(gn_oc, oc), nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv2d(oc, oc, 3, padding=1),
        )
        self.sk = nn.Conv2d(ic, oc, 1) if ic != oc else nn.Identity()

    def forward(self, x, te):
        h = self.b1(x) + self.tp(te)[:, :, None, None]
        return self.b2(h) + self.sk(x)


class _CritSelfAttn(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.norm = nn.GroupNorm(valid_gn_groups(ch, 32), ch)
        self.qkv  = nn.Conv2d(ch, ch * 3, 1, bias=False)
        self.proj = nn.Conv2d(ch, ch, 1)

    def forward(self, x):
        B, Cc, Hh, Ww = x.shape
        q, k, v = self.qkv(self.norm(x)).reshape(B, 3, Cc, Hh * Ww).unbind(1)
        a = (q.transpose(-2, -1) @ k * Cc**-0.5).softmax(-1)
        return x + self.proj((v @ a.transpose(-2, -1)).reshape(B, Cc, Hh, Ww))

class CriticPotential(nn.Module):
    def __init__(self, base_ch=24, n_res=2, dropout=0.0):
        super().__init__()
        ch = [base_ch, base_ch * 2]
        td = CFG["time_dim"] * 4
        self.temb = nn.Sequential(SinusoidalEmb(CFG["time_dim"]),
            nn.Linear(CFG["time_dim"], td), nn.SiLU(), nn.Linear(td, td))
        self.head = nn.Conv2d(1, ch[0], 3, padding=1)
        self.enc1 = nn.ModuleList([_CritResBlock(ch[0], ch[0], td, dropout)
                                    for _ in range(n_res)])
        self.dwn1 = nn.Conv2d(ch[0], ch[0], 3, stride=2, padding=1)
        self.enc2 = nn.ModuleList([
            _CritResBlock(ch[0] if i == 0 else ch[1], ch[1], td, dropout)
            for i in range(n_res)
        ])
        self.att2 = _CritSelfAttn(ch[1])
        self.mid  = _CritResBlock(ch[1], ch[1], td, dropout)
        self.tail = nn.Sequential(
            nn.GroupNorm(valid_gn_groups(ch[1], 32), ch[1]),
            nn.SiLU(),
            nn.Conv2d(ch[1], ch[1], 3, padding=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(ch[1], 1),
        )

    def forward(self, x, t):
        te = self.temb(t)
        h = self.head(x.view(-1, 1, H, W))
        for b in self.enc1: h = b(h, te)
        h = self.dwn1(h)
        for b in self.enc2: h = b(h, te)
        h = self.att2(h)
        h = self.mid(h, te)
        return self.tail(h).squeeze(-1)


def grad_phi(critic, x, t):
    x = x.detach().requires_grad_(True)
    phi = critic(x, t)
    return torch.autograd.grad(phi.sum(), x, create_graph=True)[0]


# ══════════════════════════════════════════════════════════════════════════════
# %% Checkpoint loading
# ══════════════════════════════════════════════════════════════════════════════
def load_score(path, base_ch, n_res):
    score = ScoreNet(base_ch, n_res).to(device)
    ck = torch.load(path, map_location=device, weights_only=False)
    score.load_state_dict(ck[MODEL_KEY])
    return score


def fid_history_from_final_ckpt(final_path):
    """{epoch: FID} from the history stored in the run's last checkpoint."""
    ck = torch.load(final_path, map_location="cpu", weights_only=False)
    hist = ck.get("history", {})
    eval_ep = hist.get("eval_epoch", [])
    fids = hist.get("fid_dsm", [])
    if not eval_ep or not fids or len(eval_ep) != len(fids):
        return {}
    return {int(e): float(f) for e, f in zip(eval_ep, fids)}


# ══════════════════════════════════════════════════════════════════════════════
# %% Diagnostic (Algorithm 1)
# ══════════════════════════════════════════════════════════════════════════════
def score_from_model(score, xt, t):
    with amp():
        return score(xt, t)


@torch.no_grad()
def compute_full_error(score, eval_pool, n_batches, batch_size):
    """E_{s, x_0, eps} |r|^2,  r = s_pred - s_target."""
    score.eval()
    total, count = 0.0, 0
    pool_size = eval_pool.shape[0]
    for _ in range(n_batches):
        idx = torch.randint(0, pool_size, (batch_size,), device=device)
        x0  = eval_pool[idx]
        t   = randt(batch_size)
        xt, _ = q_sample(x0, t)
        s_pred = score_from_model(score, xt, t)
        s_tgt  = s_target(xt, x0, t)
        r2 = ((s_pred - s_tgt) ** 2).sum(dim=-1)
        total += r2.sum().item()
        count += batch_size
    return total / count


def train_critic(score, eval_pool, n_steps, base_ch, batch_size, restarts, desc=""):
    """Estimate ||Pi_G e||^2 = sup_phi E[2 grad phi . r - |grad phi|^2].

    Each J is computed on a fresh batch before the update on that batch; the
    estimate is the mean J over the last max(50, n_steps // 20) steps."""
    pool_size = eval_pool.shape[0]
    score.eval()
    for p in score.parameters():
        p.requires_grad_(False)

    estimates = []
    for r_idx in range(restarts):
        critic = CriticPotential(base_ch=base_ch, n_res=2, dropout=0.0).to(device)
        opt = optim.Adam(critic.parameters(), lr=CFG["critic_lr"])

        def lr_at(step):
            p = step / max(1, n_steps)
            return CFG["critic_lr_min"] + 0.5 * (CFG["critic_lr"] - CFG["critic_lr_min"]) * \
                   (1 + math.cos(math.pi * p))

        window = max(50, n_steps // 20)
        recent = []
        pbar = tqdm(range(n_steps), desc=f"critic {desc} r{r_idx+1}/{restarts}",
                    leave=False)
        for step in pbar:
            for pg in opt.param_groups:
                pg["lr"] = lr_at(step)
            idx = torch.randint(0, pool_size, (batch_size,), device=device)
            x0  = eval_pool[idx]
            t   = randt(batch_size)
            xt, _ = q_sample(x0, t)
            with torch.no_grad():
                s_pred = score_from_model(score, xt, t)
                s_tgt  = s_target(xt, x0, t)
                r = s_pred - s_tgt
            g = grad_phi(critic, xt, t)
            J = 2.0 * (g * r).sum(dim=-1).mean() - (g**2).sum(dim=-1).mean()
            opt.zero_grad(set_to_none=True)
            (-J).backward()
            nn.utils.clip_grad_norm_(critic.parameters(), CFG["critic_clip"])
            opt.step()
            recent.append(J.item())
            if len(recent) > window:
                recent.pop(0)
            if step % 200 == 0 or step == n_steps - 1:
                pbar.set_postfix(J_avg=f"{np.mean(recent):.3f}")
        estimates.append(float(np.mean(recent)))
        del critic, opt
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return float(np.mean(estimates)), float(np.std(estimates)), estimates


# ══════════════════════════════════════════════════════════════════════════════
# %% Runs and evaluation pool
# ══════════════════════════════════════════════════════════════════════════════
def discover_runs(results_root):
    """Find <results_root>/<capacity>_s<seed>/ckpt_ep*.pt."""
    runs = []
    root = Path(results_root)
    if not root.is_dir():
        raise FileNotFoundError(f"Not a directory: {root.resolve()}")
    for sub in sorted(root.iterdir()):
        if not sub.is_dir() or "_s" not in sub.name:
            continue
        cap, seed_part = sub.name.rsplit("_s", 1)
        try:
            seed = int(seed_part)
        except ValueError:
            continue
        ckpts = []
        for c in sub.glob("ckpt_ep*.pt"):
            try:
                ckpts.append((int(c.stem[len("ckpt_ep"):]), c))
            except ValueError:
                continue
        if ckpts:
            ckpts.sort()
            runs.append(dict(capacity=cap, seed=seed, dir=sub, ckpts=ckpts))
    return runs


def build_eval_pool():
    """6,400 training images (fixed with seed 0), dequantized as in training."""
    eval_tf = T.Compose([
        T.ToTensor(),
        T.Lambda(lambda x: 2.0 * (x + torch.rand_like(x) / 255.0).clamp(0, 1) - 1.0),
        T.Lambda(lambda x: x.view(-1)),
    ])
    ds = torchvision.datasets.FashionMNIST(
        CFG["data_dir"], train=True, download=True, transform=eval_tf)
    torch.manual_seed(0); np.random.seed(0)
    idx = torch.randperm(len(ds))[:CFG["n_eval_pool"]]
    pool = torch.stack([ds[int(i)][0] for i in idx], dim=0).to(device)
    print(f"eval pool: {tuple(pool.shape)}")
    return pool


def run_per_checkpoint(runs, eval_pool):
    cache_path = Path(CFG["cache_dir"]) / f"per_checkpoint_{MODEL_KEY}.json"
    rows = json.loads(cache_path.read_text()) if cache_path.exists() else []
    if rows:
        print(f"Loaded {len(rows)} cached rows from {cache_path}")
    done_keys = {(r["capacity"], r["seed"], r["epoch"]) for r in rows}

    def save_rows():
        with open(cache_path, "w") as f:
            json.dump(rows, f, indent=2)

    work, fid_maps = [], {}
    for r in runs:
        cap, seed = r["capacity"], r["seed"]
        if cap not in CFG["capacity_arch"]:
            continue
        fid_maps[(cap, seed)] = fid_history_from_final_ckpt(r["ckpts"][-1][1])
        bc, nr = CFG["capacity_arch"][cap]
        for ep, path in r["ckpts"]:
            if (cap, seed, ep) not in done_keys:
                work.append((cap, seed, ep, path, bc, nr))

    print(f"{len(work)} checkpoints to evaluate ({len(done_keys)} already cached).")

    for cap, seed, ep, path, bc, nr in tqdm(work, desc="checkpoints"):
        score = load_score(path, bc, nr)
        full_e2 = compute_full_error(
            score, eval_pool,
            n_batches=CFG["n_eval_batches_full"],
            batch_size=CFG["eval_batch"],
        )
        grad_e2_mean, grad_e2_std, ests = train_critic(
            score, eval_pool,
            n_steps=CFG["critic_steps_per_ckpt"],
            base_ch=CFG["critic_base_ch"],
            batch_size=CFG["critic_batch"],
            restarts=CFG["n_critic_restarts"],
            desc=f"{cap}.s{seed}.ep{ep}",
        )
        sole_e2 = max(full_e2 - grad_e2_mean, 0.0)
        fid = fid_maps.get((cap, seed), {}).get(int(ep))

        rows.append(dict(
            capacity=cap, seed=int(seed), epoch=int(ep),
            full_err2=float(full_e2),
            grad_err2=float(grad_e2_mean),
            grad_err2_std=float(grad_e2_std),
            grad_estimates=[float(x) for x in ests],
            sole_err2=float(sole_e2),
            grad_ratio=float(grad_e2_mean / max(full_e2, 1e-12)),
            fid=(None if fid is None else float(fid)),
            model_key=MODEL_KEY,
        ))
        save_rows()
        del score
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"\nDone. {len(rows)} rows in {cache_path}.")
    return rows


# ══════════════════════════════════════════════════════════════════════════════
# %% Aggregation
# ══════════════════════════════════════════════════════════════════════════════
def aggregate(rows, caps):
    """Per capacity: (seeds x epochs) arrays of full / grad / solenoidal error and FID."""
    by_cs = defaultdict(list)
    for r in rows:
        by_cs[(r["capacity"], r["seed"])].append(r)
    for k in by_cs:
        by_cs[k].sort(key=lambda r: r["epoch"])

    by_cap = {}
    for cap in caps:
        seeds_here = sorted({s for c, s in by_cs if c == cap})
        if not seeds_here:
            continue
        eps_per_seed = [[r["epoch"] for r in by_cs[(cap, s)]] for s in seeds_here]
        common_eps = sorted(set.intersection(*[set(e) for e in eps_per_seed]))
        if not common_eps:
            continue

        def stack(field):
            return np.array([
                [next(r[field] for r in by_cs[(cap, s)] if r["epoch"] == e)
                 for e in common_eps]
                for s in seeds_here
            ])

        arr_full = stack("full_err2")
        arr_grad = stack("grad_err2")
        arr_sole = np.maximum(arr_full - arr_grad, 0.0)

        fids_seed = []
        for s in seeds_here:
            row_for = {r["epoch"]: r for r in by_cs[(cap, s)]}
            fids_seed.append([row_for[e].get("fid") for e in common_eps])
        if all(v is not None for row in fids_seed for v in row):
            arr_fid = np.array(fids_seed, dtype=float)
        else:
            arr_fid = None

        by_cap[cap] = dict(
            seeds=seeds_here, epochs=np.array(common_eps),
            full=arr_full, grad=arr_grad, sole=arr_sole, fid=arr_fid,
        )
        print(f"{cap:>6}: seeds={seeds_here}  epochs={len(common_eps)}  "
              f"fid={'yes' if arr_fid is not None else 'no'}")
    return by_cap


# ══════════════════════════════════════════════════════════════════════════════
# %% Plotting
# ══════════════════════════════════════════════════════════════════════════════
def _save_fig(fig, name):
    out_dir = Path(CFG["figures_dir"])
    for ext, kw in (("pdf", {}), ("png", {"dpi": 200})):
        path = out_dir / f"{name}_{MODEL_KEY}.{ext}"
        fig.savefig(path, bbox_inches="tight", **kw)
        print(f"  saved -> {path}")


def plot_error_decomposition(by_cap, cap):
    d = by_cap[cap]
    eps = d["epochs"]
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    for arr, color, label in [
        (d["full"], COL_FULL, r"Total $\|\mathbf{e}\|^2$"),
        (d["grad"], COL_GRAD, r"Gradient $\|\Pi_{\mathcal{G}}\mathbf{e}\|^2$"),
        (d["sole"], COL_SOL,  r"Solenoidal $\|\Pi_{\mathcal{G}^\perp}\mathbf{e}\|^2$"),
    ]:
        med = np.median(arr, axis=0)
        q25 = np.quantile(arr, 0.25, axis=0)
        q75 = np.quantile(arr, 0.75, axis=0)
        ax.plot(eps, med, color=color, label=label)
        ax.fill_between(eps, q25, q75, color=color, alpha=0.18)
    ax.set_yscale("log")
    ax.set_xlabel("Training epoch")
    ax.set_ylabel(r"$\mathbb{E}_{s,\mathbf{x}_s}[\,\|\cdot\|^2\,]$")
    ax.grid(True, which="both")
    ax.legend(loc="best")
    plt.tight_layout()
    _save_fig(fig, f"error_decomposition_{cap}")
    plt.close(fig)


def plot_kl_bounds(by_cap, cap, K):
    d = by_cap[cap]
    eps = d["epochs"]
    girs = K * d["full"]
    obs  = K * d["grad"]
    girs_med = np.median(girs, 0)
    girs_lo, girs_hi = np.quantile(girs, 0.25, 0), np.quantile(girs, 0.75, 0)
    obs_med  = np.median(obs, 0)
    obs_lo,  obs_hi  = np.quantile(obs, 0.25, 0),  np.quantile(obs, 0.75, 0)

    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    ax.fill_between(eps, obs_med, girs_med, color=COL_GAP, alpha=0.30,
                    label="Inobservability gap")
    ax.plot(eps, girs_med, color=COL_FULL, label="Girsanov bound")
    ax.fill_between(eps, girs_lo, girs_hi, color=COL_FULL, alpha=0.15)
    ax.plot(eps, obs_med, color=COL_GRAD, label="Ours")
    ax.fill_between(eps, obs_lo, obs_hi, color=COL_GRAD, alpha=0.15)
    ax.set_yscale("log")
    ax.set_xlabel("Training epoch")
    ax.set_ylabel("KL upper-bound estimate")
    ax.grid(True, which="both")
    ax.legend(loc="best")
    plt.tight_layout()
    _save_fig(fig, f"kl_bounds_{cap}")
    plt.close(fig)


def spearman_table(by_cap, caps):
    """Spearman rho(FID, full) and rho(FID, grad) per seed and pooled, per capacity."""
    out = []
    for cap in caps:
        if cap not in by_cap:
            continue
        d = by_cap[cap]
        if d["fid"] is None:
            out.append(dict(capacity=cap, reason="no FID values in cache"))
            continue
        per_seed_full, per_seed_grad = [], []
        for i in range(len(d["seeds"])):
            f = d["fid"][i]; e = d["full"][i]; g = d["grad"][i]
            if len(f) >= 3:
                per_seed_full.append(spearmanr(f, e).statistic)
                per_seed_grad.append(spearmanr(f, g).statistic)
        f_all = d["fid"].ravel(); e_all = d["full"].ravel(); g_all = d["grad"].ravel()
        rho_full_pool = spearmanr(f_all, e_all).statistic if len(f_all) >= 3 else float("nan")
        rho_grad_pool = spearmanr(f_all, g_all).statistic if len(f_all) >= 3 else float("nan")
        out.append(dict(
            capacity=cap,
            seeds=list(map(int, d["seeds"])),
            n_per_seed=int(d["fid"].shape[1]),
            rho_full_per_seed=[float(x) for x in per_seed_full],
            rho_grad_per_seed=[float(x) for x in per_seed_grad],
            rho_full_pool=float(rho_full_pool),
            rho_grad_pool=float(rho_grad_pool),
        ))
    return out


# ══════════════════════════════════════════════════════════════════════════════
# %% Main
# ══════════════════════════════════════════════════════════════════════════════
def main():
    cap_order = ["tiny", "small", "full"]
    cache_path = Path(CFG["cache_dir"]) / f"per_checkpoint_{MODEL_KEY}.json"

    K = CFG["K"]
    print(f"K = {K:.4f}")

    if args.plots_only:
        if not cache_path.exists():
            raise FileNotFoundError(f"--plots-only requires {cache_path}")
        rows = json.loads(cache_path.read_text())
        print(f"Loaded {len(rows)} rows from {cache_path}")
    else:
        runs = discover_runs(CFG["results_root"])
        if not runs:
            raise RuntimeError(f"No runs found under {CFG['results_root']}")
        for r in runs:
            print(f"  {r['capacity']:>6} seed={r['seed']}  {len(r['ckpts'])} ckpts")
        rows = run_per_checkpoint(runs, build_eval_pool())

    caps = sorted({r["capacity"] for r in rows},
                  key=lambda c: cap_order.index(c) if c in cap_order else 99)
    by_cap = aggregate(rows, caps)

    print("\nGenerating plots...")
    for cap in caps:
        if cap not in by_cap:
            continue
        plot_error_decomposition(by_cap, cap)
        plot_kl_bounds(by_cap, cap, K)

    tbl = spearman_table(by_cap, caps)
    with open(Path(CFG["cache_dir"]) / f"spearman_table_{MODEL_KEY}.json", "w") as f:
        json.dump(tbl, f, indent=2)
    print(f"\nFigures written under {Path(CFG['figures_dir']).resolve()}")


if __name__ == "__main__":
    main()
