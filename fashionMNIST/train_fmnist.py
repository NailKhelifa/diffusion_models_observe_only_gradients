"""
Fashion-MNIST: train DSM score models at three capacities x three seeds.

Every `eval_every` epochs the script computes a feature-space FID of the EMA model
(128-d penultimate features of a small CNN classifier trained once at start-up,
5,000 samples from a 200-step predictor-corrector sampler) and saves a checkpoint
    <out_dir>/<capacity>_s<seed>/ckpt_ep<epoch>.pt
containing the raw (non-EMA) score-network weights and the FID history.
diagnostic_fmnist.py then runs the critic diagnostic on every checkpoint.

Usage:  python train_fmnist.py --device cuda:0
"""
import math, os, argparse
# CUBLAS workspace config must be set before any CUDA op (deterministic GEMMs)
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as T

# ─────────────────────────────────────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────────────────────────────────────
CFG = dict(
    seeds             = [0, 1, 2],
    data_dir          = "./data",
    T                 = 4.0,
    t0                = 0.05,
    epochs            = 400,
    batch_size        = 512,
    lr_score          = 5e-4,
    lr_warmup         = 2000,
    clip_grad         = 1.0,
    ema_decay         = 0.9999,
    time_dim          = 128,
    sweep             = [("tiny",  32, 1),      # (name, base channels, res-blocks per level)
                         ("small", 48, 2),
                         ("full",  64, 2)],
    n_gen             = 5000,                   # samples per FID evaluation
    n_sampler_steps   = 200,                    # predictor-corrector steps
    eval_every        = 20,                     # FID + checkpoint every N epochs
    clf_epochs        = 20,
    clf_lr            = 1e-3,
    num_workers       = 4,
    out_dir           = "results",
)

# ─────────────────────────────────────────────────────────────────────────────
#  DEVICE + REPRODUCIBILITY
# ─────────────────────────────────────────────────────────────────────────────
_parser = argparse.ArgumentParser()
_parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
_args = _parser.parse_args()
device = torch.device(_args.device)
print(f"device: {device}")

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False
torch.cuda.manual_seed_all(0)             # re-seeded per run below
try:
    torch.use_deterministic_algorithms(True)
except Exception as e:
    print(f"[warn] use_deterministic_algorithms(True) failed: {e}")

Path(CFG["out_dir"]).mkdir(exist_ok=True)
D = 28 * 28

def worker_init_fn(worker_id):
    np.random.seed(torch.initial_seed() % 2**31)


# ─────────────────────────────────────────────────────────────────────────────
#  DATA
# ─────────────────────────────────────────────────────────────────────────────
class Dequantize(nn.Module):
    def forward(self, x):
        return 2.0 * (x + torch.rand_like(x)/255.0).clamp(0,1) - 1.0

train_tf = T.Compose([T.ToTensor(), Dequantize(), T.Lambda(lambda x: x.view(-1))])
ref_tf   = T.Compose([T.ToTensor(), T.Lambda(lambda x: (2.0*x-1.0).view(-1))])
clf_tf   = T.Compose([T.ToTensor(), T.Normalize((0.5,), (0.5,))])

train_ds = torchvision.datasets.FashionMNIST(CFG["data_dir"], True,  download=True,  transform=train_tf)
ref_ds   = torchvision.datasets.FashionMNIST(CFG["data_dir"], True,  download=False, transform=ref_tf)
clf_ds   = torchvision.datasets.FashionMNIST(CFG["data_dir"], True,  download=False, transform=clf_tf)
clf_val  = torchvision.datasets.FashionMNIST(CFG["data_dir"], False, download=True,  transform=clf_tf)

n_batches = len(train_ds) // CFG["batch_size"]


# ─────────────────────────────────────────────────────────────────────────────
#  VP-SDE:  alpha(t) = e^{-t/2},  sigma(t) = sqrt(1 - e^{-t})
# ─────────────────────────────────────────────────────────────────────────────
def alpha(t): return torch.exp(-0.5 * t)
def sigma(t): return torch.sqrt(1.0 - torch.exp(-t) + 1e-8)

def sample_noised(x0, t):
    a, s = alpha(t)[:,None], sigma(t)[:,None]
    eps  = torch.randn_like(x0)
    return a*x0 + s*eps, eps

def dsm_target(xt, x0, t):
    a, s = alpha(t)[:,None], sigma(t)[:,None]
    return -(xt - a*x0) / s**2

def sample_t(n):
    """Log-uniform diffusion times in [t0, T]."""
    return CFG["t0"] * torch.exp(
        torch.rand(n, device=device) * math.log(CFG["T"] / CFG["t0"]))


# ─────────────────────────────────────────────────────────────────────────────
#  CLASSIFIER FOR FEATURE-FID  (trained once, frozen)
# ─────────────────────────────────────────────────────────────────────────────
class FMNISTClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1,32,3,padding=1), nn.SiLU(),
            nn.Conv2d(32,32,3,padding=1), nn.SiLU(), nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1), nn.SiLU(),
            nn.Conv2d(64,64,3,padding=1), nn.SiLU(), nn.MaxPool2d(2),
            nn.Conv2d(64,128,3,padding=1), nn.SiLU(), nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(128, 10)

    def forward(self, x):
        return self.head(self.features(x).view(x.shape[0], -1))

    @torch.no_grad()
    def embed(self, x):
        """L2-normalised 128-dim feature embedding."""
        return F.normalize(self.features(x).view(x.shape[0], -1), dim=-1)


def train_classifier():
    clf = FMNISTClassifier().to(device)
    opt = optim.AdamW(clf.parameters(), lr=CFG["clf_lr"], weight_decay=1e-4)
    dl  = torch.utils.data.DataLoader(clf_ds, 256, shuffle=True,
                                       num_workers=CFG["num_workers"],
                                       worker_init_fn=worker_init_fn)
    vdl = torch.utils.data.DataLoader(clf_val, 512,
                                       num_workers=CFG["num_workers"],
                                       worker_init_fn=worker_init_fn)
    print(f"\nTraining classifier ({CFG['clf_epochs']} epochs)...")
    acc = 0.0
    for ep in range(1, CFG["clf_epochs"]+1):
        clf.train()
        for x, y in dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(); F.cross_entropy(clf(x), y).backward(); opt.step()
        if ep % 5 == 0 or ep == CFG["clf_epochs"]:
            clf.eval(); c = t = 0
            with torch.no_grad():
                for x, y in vdl:
                    x, y = x.to(device), y.to(device)
                    c += (clf(x).argmax(1)==y).sum().item(); t += len(y)
            acc = c/t
            print(f"  ep {ep:2d}  val_acc={acc:.3f}")
    clf.eval()
    for p in clf.parameters(): p.requires_grad_(False)
    print(f"Classifier frozen. acc={acc:.3f}\n")
    return clf

classifier = train_classifier()

# Real-image features for FID (computed once, reused for all runs)
_ridx      = torch.randperm(len(ref_ds))[:CFG["n_gen"]]
real_flat  = torch.stack([ref_ds[i][0] for i in _ridx])   # (N, 784)

def extract_features(flat_batch):
    imgs = flat_batch.view(-1,1,28,28).to(device)
    imgs = (imgs+1)/2; imgs = (imgs-0.5)/0.5
    out  = []
    with torch.no_grad():
        for i in range(0, len(imgs), 256):
            out.append(classifier.embed(imgs[i:i+256]).cpu())
    return torch.cat(out)

real_feats = extract_features(real_flat)   # (N, 128), L2-normalised


def _sqrtm_psd(M):
    """Matrix square root of a symmetric PSD matrix via eigendecomposition."""
    L, Q = torch.linalg.eigh(M)
    return Q @ torch.diag(L.clamp_min(0).sqrt()) @ Q.T

def feature_fid(gen_feats):
    """FID in the 128-dim classifier feature space:
       ||mu_g - mu_r||^2 + Tr(Cg + Cr - 2 sqrtm(Cr^{1/2} Cg Cr^{1/2}))."""
    g  = gen_feats.float(); r = real_feats[:len(g)].float()
    mu_g, mu_r = g.mean(0), r.mean(0)
    Cg, Cr = torch.cov(g.T), torch.cov(r.T)
    eps_jit = 1e-6 * torch.eye(Cg.shape[0], device=Cg.device)
    Cg, Cr  = Cg + eps_jit, Cr + eps_jit
    Cr_sqrt = _sqrtm_psd(Cr)
    mid     = Cr_sqrt @ Cg @ Cr_sqrt
    mid     = (mid + mid.T) * 0.5
    mid_sqrt = _sqrtm_psd(mid)
    fid = (mu_g-mu_r).pow(2).sum() + torch.trace(Cg + Cr - 2*mid_sqrt)
    return float(fid.clamp_min(0).item())


# ─────────────────────────────────────────────────────────────────────────────
#  U-NET SCORE MODEL  (outputs the score directly)
# ─────────────────────────────────────────────────────────────────────────────
class SinusoidalEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        half = dim//2
        freqs = torch.exp(-math.log(10000)*torch.arange(half).float()/(half-1))
        self.register_buffer("freqs", freqs)
    def forward(self, t):
        x = t[:,None]*self.freqs[None,:]
        return torch.cat([x.sin(), x.cos()], dim=-1)

class TimeEmb(nn.Module):
    def __init__(self, sin_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            SinusoidalEmb(sin_dim),
            nn.Linear(sin_dim, out_dim), nn.SiLU(),
            nn.Linear(out_dim, out_dim))
    def forward(self, t): return self.net(t)

class ResConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, td):
        super().__init__()
        g1, g2 = min(8,in_ch), min(8,out_ch)
        self.norm1  = nn.GroupNorm(g1, in_ch)
        self.conv1  = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2  = nn.GroupNorm(g2, out_ch)
        self.conv2  = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.t_proj = nn.Linear(td, out_ch*2)
        nn.init.zeros_(self.conv2.weight); nn.init.zeros_(self.conv2.bias)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
    def forward(self, x, te):
        h = F.silu(self.norm1(x)); h = self.conv1(h)
        sc = self.t_proj(te)[:,:,None,None]; s, sh = sc.chunk(2,1)
        h = F.silu(self.norm2(h)*(1+s)+sh); h = self.conv2(h)
        return h + self.skip(x)

class SelfAttention2d(nn.Module):
    def __init__(self, ch):
        super().__init__()
        nh = 8
        while ch % nh != 0: nh -= 1
        self.norm = nn.GroupNorm(min(8,ch), ch)
        self.attn = nn.MultiheadAttention(ch, nh, batch_first=True)
        self.proj = nn.Conv2d(ch, ch, 1)
        nn.init.zeros_(self.proj.weight); nn.init.zeros_(self.proj.bias)
    def forward(self, x):
        B,C,H,W = x.shape
        h = self.norm(x).view(B,C,H*W).permute(0,2,1)
        h,_ = self.attn(h,h,h)
        return x + self.proj(h.permute(0,2,1).view(B,C,H,W))

class ScoreNet(nn.Module):
    """U-Net: base_ch sets channel width, n_res sets depth per level.
    Input/output: flat (B,784).  Internally operates on (B,1,28,28)."""
    def __init__(self, base_ch=64, n_res=2):
        super().__init__()
        td = CFG["time_dim"]; C = [base_ch, base_ch*2, base_ch*4]
        self.time_emb = TimeEmb(td, td*4)
        self.enc_in   = nn.Conv2d(1, C[0], 3, padding=1)

        def stack(ic, oc, n):
            b = [ResConvBlock(ic, oc, td*4)]
            for _ in range(n-1): b.append(ResConvBlock(oc, oc, td*4))
            return nn.ModuleList(b)

        self.enc0 = stack(C[0], C[0], n_res);  self.dn0 = nn.Conv2d(C[0],C[0],3,stride=2,padding=1)
        self.enc1 = stack(C[0], C[1], n_res);  self.dn1 = nn.Conv2d(C[1],C[1],3,stride=2,padding=1)
        self.mid0 = ResConvBlock(C[1],C[2],td*4); self.mida = SelfAttention2d(C[2]); self.mid1 = ResConvBlock(C[2],C[2],td*4)
        self.up1  = nn.Conv2d(C[2],C[1],3,padding=1); self.dec1 = stack(C[1]*2,C[1],n_res)
        self.up0  = nn.Conv2d(C[1],C[0],3,padding=1); self.dec0 = stack(C[0]*2,C[0],n_res)
        self.onorm = nn.GroupNorm(min(8,C[0]),C[0])
        self.oconv = nn.Conv2d(C[0],1,3,padding=1)
        nn.init.zeros_(self.oconv.weight); nn.init.zeros_(self.oconv.bias)

    def forward(self, x, t):
        B = x.shape[0]; te = self.time_emb(t)
        h = self.enc_in(x.view(B,1,28,28))
        for b in self.enc0: h = b(h,te)
        s0=h; h=self.dn0(h)
        for b in self.enc1: h = b(h,te)
        s1=h; h=self.dn1(h)
        h=self.mid0(h,te); h=self.mida(h); h=self.mid1(h,te)
        h=F.interpolate(self.up1(h),scale_factor=2,mode="nearest"); h=torch.cat([h,s1],1)
        for b in self.dec1: h=b(h,te)
        h=F.interpolate(self.up0(h),scale_factor=2,mode="nearest"); h=torch.cat([h,s0],1)
        for b in self.dec0: h=b(h,te)
        return self.oconv(F.silu(self.onorm(h))).view(B,D)


# ─────────────────────────────────────────────────────────────────────────────
#  EMA, LOSS, SAMPLER, LR SCHEDULE
# ─────────────────────────────────────────────────────────────────────────────
class EMA:
    def __init__(self, m, decay):
        self.decay  = decay
        self.shadow = {k:v.clone().float() for k,v in m.state_dict().items()}
    @torch.no_grad()
    def update(self, m):
        for k,v in m.state_dict().items():
            self.shadow[k].mul_(self.decay).add_(v.float(), alpha=1-self.decay)
    def apply(self, m):
        self._bk = {k:v.clone() for k,v in m.state_dict().items()}
        m.load_state_dict({k:v.to(next(m.parameters()).device) for k,v in self.shadow.items()})
    def restore(self, m):
        m.load_state_dict(self._bk)

def loss_dsm_fn(score_net, x0, t):
    """sigma^2-weighted denoising score matching."""
    xt, _ = sample_noised(x0, t)
    diff  = score_net(xt,t) - dsm_target(xt,x0,t)
    return (sigma(t).pow(2)[:,None]*diff**2).sum(-1).mean()

@torch.no_grad()
def pc_sample(score_net, n, n_steps=None, snr=0.16):
    """Predictor-corrector sampler (reverse Euler-Maruyama + one Langevin step)."""
    if n_steps is None: n_steps = CFG["n_sampler_steps"]
    ts = torch.linspace(CFG["T"], CFG["t0"], n_steps+1, device=device)
    y  = torch.randn(n, D, device=device)
    for i in range(n_steps):
        tc,tn = ts[i],ts[i+1]; dτ=(tc-tn).item()
        s = score_net(y, tc.expand(n))
        y = y + (0.5*y+s)*dτ + math.sqrt(dτ)*torch.randn(y.shape, device=device)
        sc = score_net(y, tn.expand(n)); nm = sc.norm(dim=-1).mean().item()
        step = (snr*math.sqrt(D)/nm)**2*2 if nm > 1e-6 else 1e-4
        y = y + step*sc + math.sqrt(2*step)*torch.randn(y.shape, device=device)
    s   = score_net(y, ts[-1].expand(n))
    at0 = alpha(ts[-1].unsqueeze(0)).squeeze()
    st0 = sigma(ts[-1].unsqueeze(0)).squeeze()
    return ((y - st0*(-st0*s)) / at0.clamp(min=1e-8)).clamp(-1,1).cpu()

def make_scheduler(opt, warmup_steps, total_steps):
    def fn(step):
        if step < warmup_steps: return step/max(1,warmup_steps)
        p = (step-warmup_steps)/max(1,total_steps-warmup_steps)
        return 0.5*(1+math.cos(math.pi*p))
    return optim.lr_scheduler.LambdaLR(opt, fn)


# ─────────────────────────────────────────────────────────────────────────────
#  SINGLE TRAINING RUN
# ─────────────────────────────────────────────────────────────────────────────
def train_one_run(seed, label, base_ch, n_res):
    run_dir = Path(CFG["out_dir"]) / f"{label}_s{seed}"
    run_dir.mkdir(exist_ok=True)
    print(f"\nRUN: {run_dir.name}  base_ch={base_ch}  n_res={n_res}  seed={seed}")

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    # seeded generator: the shuffle order is reproducible and differs across seeds
    _dl_gen = torch.Generator(); _dl_gen.manual_seed(seed)
    train_loader = torch.utils.data.DataLoader(
        train_ds, CFG["batch_size"], shuffle=True,
        num_workers=CFG["num_workers"], pin_memory=True, drop_last=True,
        worker_init_fn=worker_init_fn, generator=_dl_gen)

    score = ScoreNet(base_ch, n_res).to(device)
    print(f"Score: {sum(p.numel() for p in score.parameters()):,} params")
    ema   = EMA(score, CFG["ema_decay"])
    opt   = optim.AdamW(score.parameters(), lr=CFG["lr_score"], weight_decay=1e-4)
    sched = make_scheduler(opt, CFG["lr_warmup"], CFG["epochs"] * n_batches)

    hist = defaultdict(list)
    for epoch in range(1, CFG["epochs"]+1):
        score.train()
        ep_loss = 0.0
        for x0, _ in train_loader:
            x0 = x0.to(device, non_blocking=True)
            t  = sample_t(x0.shape[0])
            opt.zero_grad()
            loss = loss_dsm_fn(score, x0, t)
            loss.backward()
            nn.utils.clip_grad_norm_(score.parameters(), CFG["clip_grad"])
            opt.step(); ema.update(score); sched.step()
            ep_loss += loss.item()
        hist["loss_dsm"].append(ep_loss / n_batches)

        if epoch % CFG["eval_every"] == 0 or epoch == CFG["epochs"]:
            score.eval()
            ema.apply(score)
            fid = feature_fid(extract_features(pc_sample(score, CFG["n_gen"])))
            ema.restore(score)
            hist["eval_epoch"].append(epoch)
            hist["fid_dsm"].append(fid)
            print(f"  [{run_dir.name}] ep {epoch:3d}  loss={ep_loss/n_batches:.4f}  FID={fid:.4f}")
            # raw (non-EMA) weights, as in the paper's runs; the FID above uses the EMA weights
            torch.save(dict(epoch=epoch, history=dict(hist), score_dsm=score.state_dict()),
                       run_dir/f"ckpt_ep{epoch:03d}.pt")


if __name__ == "__main__":
    for lbl, bc, nr in CFG["sweep"]:
        for seed in CFG["seeds"]:
            train_one_run(seed, lbl, bc, nr)
