"""
Low-dimensional experiment with analytically known Hodge decompositions
======================================================================
Companion experiment for "Diffusion Models Observe Only Gradients" (reviewer request).

Target: 8-mode anisotropic Gaussian mixture in R^2 (closed-form noised marginals and scores),
VP-linear forward SDE. Synthetic score errors with an analytically known decomposition

    e_s = a * grad(psi_s) + b * J grad(log p*_s),      J = [[0,-1],[1,0]],
    psi_s = log p~_s - log p*_s   (p~ = mixture with perturbed weights / means),

for which  Pi_G e_s = a grad(psi_s)  and  Pi_{G^perp} e_s = b J grad(log p*_s)  exactly
(div(p* J grad log p*) = div(J grad p*) = 0 for any density), and the two parts are orthogonal
in L^2(p*_s). Hence both bounds are exact time integrals:

    Girsanov = 1/2 int g^2 (a^2 ||grad psi_s||^2 + b^2 Fisher(p*_s)) ds,
    ours     = 1/2 int g^2  a^2 ||grad psi_s||^2 ds.

The endpoint law hat p_{t0} is obtained with a high-accuracy 2-D Fokker-Planck solver
(Scharfetter-Gummel finite volumes, variable-step BDF2), and KL / TV / W2 at t0 are computed
against a reference solve with the exact score on the same grid and time nodes. A companion
solve on a (2/3 N)^2 grid gives a discretisation error bar on every KL.

Output: KL_vs_bounds.pdf (+ .png if dvipng is available) and results.json.

Usage
-----
    python lowdim_hodge_experiment.py            # full configuration
    python lowdim_hodge_experiment.py --quick    # fast smoke test (~1 min)
"""

import math, time, json, argparse, shutil, io
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.optimize import brentq
from scipy.stats import spearmanr

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

try:
    import ot  # POT, exact discrete optimal transport for W2
    HAVE_POT = True
except ImportError:
    HAVE_POT = False

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════════════════════
CFG = dict(
    out_dir        = "lowdim_results",
    seed           = 0,
    # ── target ────────────────────────────────────────────────────────────────
    gmm_K          = 8,            # modes on a ring
    gmm_radius     = 2.0,
    gmm_std_rad    = 0.18,         # radial std of each (anisotropic) component
    gmm_std_tan    = 0.40,         # tangential std
    sigma_min      = 0.05,         # t0 chosen s.t. std(t0) = sigma_min
    # ── numerics ──────────────────────────────────────────────────────────────
    grid_N         = 224,
    grid_L         = 4.5,
    fp_steps       = 200,          # reverse Fokker-Planck time steps
    fp_error_est   = True,         # companion solve on a (2/3 N)^2 grid -> error bar on KL
    quad_panels    = 10,           # composite Gauss-Legendre in log(s) for the bound integrals
    quad_order     = 5,
    w2_coarsen     = 4,            # W2 by exact discrete OT on a (N/coarsen)^2 grid
    # ── synthetic error ensemble ──────────────────────────────────────────────
    n_fields       = 48,
    n_pert         = 4,            # number of random "wrong mixtures" p~ defining psi
    a_range        = (0.03, 1.0),  # gradient amplitude (log-uniform), plus exact zeros
    b_range        = (0.03, 1.5),  # solenoidal amplitude (log-uniform), plus exact zeros
)

QUICK = dict(grid_N=120, fp_steps=120, quad_panels=6, quad_order=4, n_fields=12, n_pert=2,
             out_dir="lowdim_results_quick")

# ══════════════════════════════════════════════════════════════════════════════
#  PLOTTING STYLE  (LaTeX / Computer Modern; same sizes as the CIFAR-10 script)
# ══════════════════════════════════════════════════════════════════════════════
_TEX_RC = {
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "text.latex.preamble": r"\usepackage{amsmath}\usepackage{amssymb}",
}


def _latex_works():
    """usetex needs latex (+ cm-super, + dvipng for PNG); test once instead of failing at savefig."""
    if shutil.which("latex") is None: return False
    try:
        with plt.rc_context(_TEX_RC):
            fig = plt.figure(figsize=(1, 1))
            fig.text(0.5, 0.5, r"Test $\frac{1}{2}\int g^2\|\Pi_{\mathcal{G}}e_s\|^2$")
            fig.savefig(io.BytesIO(), format="pdf")
        return True
    except Exception:
        return False
    finally:
        plt.close("all")


plt.rcParams.update({
    "font.size": 18, "axes.labelsize": 18, "axes.titlesize": 18,
    "legend.fontsize": 18, "xtick.labelsize": 18, "ytick.labelsize": 18,
    "figure.dpi": 200, "axes.linewidth": 0.8, "lines.linewidth": 2.0,
    "lines.markersize": 6, "legend.frameon": False,
    "grid.alpha": 0.3, "grid.linewidth": 0.5,
})
USE_TEX = _latex_works()
if USE_TEX:   # true LaTeX rendering (Computer Modern)
    plt.rcParams.update(_TEX_RC)
else:         # no working LaTeX: Computer Modern via matplotlib's bundled fonts
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["cmr10"],
        "mathtext.fontset": "cm",
        "axes.formatter.use_mathtext": True,
        "axes.unicode_minus": False,
    })

COLORS = {"gray": "#8D99AE", "gold": "#E9C46A"}
COL_GRAD = "#E76F51"   # = \definecolor{COL_GRAD} in the paper
COL_SOL  = "#0D695E"   # = \definecolor{COL_SOL}
CMAP_SHARE = LinearSegmentedColormap.from_list("grad_to_sol", [COL_GRAD, COLORS["gold"], COL_SOL])


def savefig(fig, out_dir, name):
    fig.savefig(Path(out_dir) / f"{name}.pdf", bbox_inches="tight")
    if not USE_TEX or shutil.which("dvipng"):   # usetex PNGs need dvipng
        fig.savefig(Path(out_dir) / f"{name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _spear(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    return float(spearmanr(a[m], b[m])[0]) if m.sum() >= 3 else float("nan")


# ══════════════════════════════════════════════════════════════════════════════
#  VP-LINEAR SCHEDULE
# ══════════════════════════════════════════════════════════════════════════════
class VPLinear:
    """dx = -1/2 beta(t) x dt + g(t) dB with g^2 = beta, x_t = alpha x0 + std eps."""
    name = "vp_linear"
    def __init__(self, bmin=0.1, bmax=20.0, T=1.0, sigma_min=0.05):
        self.bmin, self.bmax, self.T = bmin, bmax, T
        self.t0 = brentq(lambda t: math.sqrt(self.std2(t)) - sigma_min, 1e-10, T)
    def B(self, t): return self.bmin * t + 0.5 * (self.bmax - self.bmin) * t ** 2
    def beta(self, t): return self.bmin + (self.bmax - self.bmin) * t
    def g2(self, t): return self.beta(t)
    def alpha(self, t): return np.exp(-0.5 * self.B(t))
    def std2(self, t): return 1.0 - np.exp(-self.B(t))


def quad_nodes(sched, n_panels, order):
    """Composite Gauss-Legendre on geometric panels of [t0, T]: int f ds ~ sum w_k f(s_k)."""
    edges = np.geomspace(sched.t0, sched.T, n_panels + 1)
    xg, wg = np.polynomial.legendre.leggauss(order)
    s = [0.5 * (b - a) * xg + 0.5 * (a + b) for a, b in zip(edges[:-1], edges[1:])]
    w = [0.5 * (b - a) * wg for a, b in zip(edges[:-1], edges[1:])]
    return np.concatenate(s), np.concatenate(w)


# ══════════════════════════════════════════════════════════════════════════════
#  GAUSSIAN MIXTURE (closed-form noised marginals and scores)
# ══════════════════════════════════════════════════════════════════════════════
class GMM:
    def __init__(self, w, m, S):
        self.w = np.asarray(w, float) / np.sum(w)
        self.m = np.asarray(m, float); self.S = np.asarray(S, float)
        Ci = np.linalg.inv(self.S)
        self._a, self._b, self._c = Ci[:, 0, 0], Ci[:, 0, 1], Ci[:, 1, 1]
        self._lconst = np.log(self.w) - 0.5 * np.log(np.linalg.det(self.S)) - math.log(2 * math.pi)

    def at(self, sched, s):
        a, v = float(sched.alpha(s)), float(sched.std2(s))
        return GMM(self.w, a * self.m, a * a * self.S + v * np.eye(2)[None])

    def _parts(self, x):
        dx = x[:, :1] - self.m[None, :, 0]; dy = x[:, 1:2] - self.m[None, :, 1]
        cx = self._a * dx + self._b * dy; cy = self._b * dx + self._c * dy
        return self._lconst - 0.5 * (dx * cx + dy * cy), cx, cy

    def pdf(self, x):
        logc, _, _ = self._parts(x)
        mx = logc.max(1, keepdims=True)
        return np.exp(mx[:, 0]) * np.exp(logc - mx).sum(1)

    def score(self, x):
        logc, cx, cy = self._parts(x)
        r = np.exp(logc - logc.max(1, keepdims=True)); r /= r.sum(1, keepdims=True)
        return -np.stack([(r * cx).sum(1), (r * cy).sum(1)], -1)


def make_target(cfg):
    K = cfg["gmm_K"]; ang = 2 * np.pi * np.arange(K) / K
    m = cfg["gmm_radius"] * np.stack([np.cos(ang), np.sin(ang)], 1)
    S = []
    for a in ang:
        R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
        S.append(R @ np.diag([cfg["gmm_std_rad"] ** 2, cfg["gmm_std_tan"] ** 2]) @ R.T)
    return GMM(np.linspace(1.0, 2.0, K), m, np.array(S))   # unequal weights


def perturbed_mixtures(gmm, n, rng, sw=0.6, sm=0.15):
    """'Wrong' mixtures p~ (log-weights and means perturbed). grad psi = grad log p~ - grad log p*
    is a pure-gradient error; with a=1, b=0 the learned score is exactly the score of p~_s."""
    return [GMM(gmm.w * np.exp(sw * rng.standard_normal(len(gmm.w))),
                gmm.m + sm * rng.standard_normal(gmm.m.shape), gmm.S) for _ in range(n)]


def rot90(v):
    """J v with J = [[0,-1],[1,0]]; J grad log p is weighted-divergence-free for ANY density p."""
    return np.stack([-v[..., 1], v[..., 0]], -1)


# ══════════════════════════════════════════════════════════════════════════════
#  FINITE-VOLUME GRID
# ══════════════════════════════════════════════════════════════════════════════
class Grid:
    def __init__(self, N, L):
        self.N, self.L, self.h = N, L, 2 * L / N
        c = -L + (np.arange(N) + 0.5) * self.h
        f = -L + np.arange(1, N) * self.h
        X, Y = np.meshgrid(c, c, indexing="ij");  self.cells = np.stack([X.ravel(), Y.ravel()], -1)
        X, Y = np.meshgrid(f, c, indexing="ij");  self.xfaces = np.stack([X.ravel(), Y.ravel()], -1)
        X, Y = np.meshgrid(c, f, indexing="ij");  self.yfaces = np.stack([X.ravel(), Y.ravel()], -1)
        n = N * N; self.n = n
        idx = np.arange(n).reshape(N, N)
        self.Lx, self.Rx = idx[:-1, :].ravel(), idx[1:, :].ravel()
        self.Ly, self.Ry = idx[:, :-1].ravel(), idx[:, 1:].ravel()
        rows = np.concatenate([self.Lx, self.Rx, self.Ly, self.Ry, np.arange(n)])
        cols = np.concatenate([self.Rx, self.Lx, self.Ry, self.Ly, np.arange(n)])
        Mp = sp.csr_matrix((np.arange(1, len(rows) + 1, dtype=float), (rows, cols)), shape=(n, n))
        self._perm = Mp.data.astype(np.int64) - 1
        self._indices, self._indptr = Mp.indices.copy(), Mp.indptr.copy()

    def build(self, LR_x, RL_x, LR_y, RL_y, diag):
        v = np.concatenate([LR_x, RL_x, LR_y, RL_y, diag])
        return sp.csr_matrix((v[self._perm], self._indices, self._indptr), shape=(self.n, self.n))

    def bsum(self, ix, vx, iy, vy):
        return np.bincount(ix, vx, self.n) + np.bincount(iy, vy, self.n)

    def density(self, gmm):
        p = gmm.pdf(self.cells); return p / (p.sum() * self.h ** 2)


# ══════════════════════════════════════════════════════════════════════════════
#  REVERSE FOKKER-PLANCK SOLVER  (Scharfetter-Gummel FV, variable-step BDF2)
# ══════════════════════════════════════════════════════════════════════════════
def _bern(x):
    x = np.clip(x, -700.0, 700.0); out = np.empty_like(x)
    sm = np.abs(x) < 1e-6
    out[sm] = 1.0 - 0.5 * x[sm]; out[~sm] = x[~sm] / np.expm1(x[~sm])
    return out


def fp_operator(grid, sched, s, sx, sy):
    """Generator in tau = T - s of the reverse SDE with drift -f + g^2 * score (zero-flux boundary)."""
    h = grid.h; beta, g2 = float(sched.beta(s)), float(sched.g2(s)); D = 0.5 * g2
    bx = 0.5 * beta * grid.xfaces[:, 0] + g2 * sx
    by = 0.5 * beta * grid.yfaces[:, 1] + g2 * sy
    c = D / h ** 2
    aLx, aRx = c * _bern(-bx * h / D), c * _bern(bx * h / D)
    aLy, aRy = c * _bern(-by * h / D), c * _bern(by * h / D)
    diag = -(grid.bsum(grid.Lx, aLx, grid.Ly, aLy) + grid.bsum(grid.Rx, aRx, grid.Ry, aRy))
    return grid.build(aRx, aLx, aRy, aLy, diag)


def fp_time_nodes(sched, n_steps, mix=0.5):
    """s from T down to t0; blend of uniform-in-s and uniform-in-log-SNR spacing."""
    t0, T = sched.t0, sched.T
    lam = lambda t: math.log(float(sched.alpha(t)) ** 2 / float(sched.std2(t)))
    u = np.linspace(0, 1, n_steps + 1)
    lT, l0 = lam(T), lam(t0)
    s_log = np.array([T] + [brentq(lambda t: lam(t) - (lT + ui * (l0 - lT)), t0 * (1 - 1e-9), T)
                            for ui in u[1:-1]] + [t0])
    s = (1 - mix) * (T - u * (T - t0)) + mix * s_log
    s[0], s[-1] = T, t0
    return s


def solve_reverse_fp(grid, sched, score_fn, q0, n_steps):
    """score_fn(s, pts)->(M,2). q0: density at s=T (cells). Returns density at s=t0."""
    s_nodes = fp_time_nodes(sched, n_steps); tau = sched.T - s_nodes
    I = sp.identity(grid.n, format="csr")
    q_prev, q = None, q0.copy()
    for k in range(n_steps):
        sk = s_nodes[k + 1]
        Lk = fp_operator(grid, sched, sk, score_fn(sk, grid.xfaces)[:, 0], score_fn(sk, grid.yfaces)[:, 1])
        dt = tau[k + 1] - tau[k]
        if k == 0:
            A, rhs, guess = I - dt * Lk, q, q
        else:
            w = dt / (tau[k] - tau[k - 1])
            A = (1 + 2 * w) / (1 + w) * I - dt * Lk
            rhs = (1 + w) * q - (w * w / (1 + w)) * q_prev
            guess = q + w * (q - q_prev)
        x, info = spla.bicgstab(A, rhs, x0=guess, rtol=1e-11, atol=0.0,
                                M=sp.diags(1.0 / A.diagonal()), maxiter=3000)
        if info != 0:
            x = spla.spsolve(A.tocsc(), rhs)
        q_prev, q = q, x
    q = np.maximum(q, 0.0)
    return q / (q.sum() * grid.h ** 2)


# ══════════════════════════════════════════════════════════════════════════════
#  DISCREPANCIES
# ══════════════════════════════════════════════════════════════════════════════
def kl_tv(p, q, h):
    p = p / (p.sum() * h * h); q = q / (q.sum() * h * h)
    m = p > 0
    kl = h * h * np.sum(p[m] * (np.log(p[m]) - np.log(np.maximum(q[m], 1e-300))))
    return float(max(kl, 0.0)), float(0.5 * h * h * np.abs(p - q).sum())


def _sliced_w2(X, a, b, n_dir=256, seed=0):
    tot = 0.0
    for t in np.random.default_rng(seed).uniform(0, np.pi, n_dir):
        z = X @ np.array([np.cos(t), np.sin(t)]); o = np.argsort(z); z = z[o]
        Fa, Fb = np.cumsum(a[o]), np.cumsum(b[o])
        u = np.unique(np.concatenate([Fa, Fb]).clip(0, 1)); du = np.diff(np.concatenate([[0], u]))
        qa = z[np.minimum(np.searchsorted(Fa, u - 1e-15), len(z) - 1)]
        qb = z[np.minimum(np.searchsorted(Fb, u - 1e-15), len(z) - 1)]
        tot += np.sum(du * (qa - qb) ** 2)
    return math.sqrt(tot / n_dir)


def w2_grid(grid, p, q, k):
    """W2 between two grid densities: exact discrete OT on a k-times coarsened grid
    (sliced W2 if POT is unavailable)."""
    N = grid.N; Nc = N // k; n = Nc * k
    P = (p.reshape(N, N)[:n, :n]).reshape(Nc, k, Nc, k).sum((1, 3)).ravel()
    Q = (q.reshape(N, N)[:n, :n]).reshape(Nc, k, Nc, k).sum((1, 3)).ravel()
    c = -grid.L + (np.arange(Nc) + 0.5) * k * grid.h
    X, Y = np.meshgrid(c, c, indexing="ij"); Z = np.stack([X.ravel(), Y.ravel()], -1)
    m = (P > 1e-10 * P.max()) | (Q > 1e-10 * Q.max())
    a, b, Z = P[m] / P[m].sum(), Q[m] / Q[m].sum(), Z[m]
    if HAVE_POT:
        M = ot.dist(Z, Z, metric="sqeuclidean")
        return float(math.sqrt(max(ot.emd2(a, b, M, numItermax=int(1e7)), 0.0)))
    return _sliced_w2(Z, a, b)


class FPContext:
    """Reference endpoint (exact score, same grid & time nodes) on the working grid and, if
    requested, on a companion (2/3 N)^2 grid used to attach a discretisation error bar to KL."""
    def __init__(self, sched, gmm, cfg):
        self.sched, self.cfg = sched, cfg
        Ns = [cfg["grid_N"]] + ([int(round(2 * cfg["grid_N"] / 3))] if cfg["fp_error_est"] else [])
        self.levels = []
        for N in Ns:
            g = Grid(N, cfg["grid_L"]); t_ = time.time()
            q0, p_an = g.density(gmm.at(sched, sched.T)), g.density(gmm.at(sched, sched.t0))
            p_ref = solve_reverse_fp(g, sched, lambda s, x: gmm.at(sched, s).score(x), q0, cfg["fp_steps"])
            kl_f, tv_f = kl_tv(p_an, p_ref, g.h)
            self.levels.append(dict(grid=g, q0=q0, p_an=p_an, p_ref=p_ref))
            print(f"  reference solve N={N} ({time.time() - t_:.1f}s) | solver floor vs analytic p*_t0: "
                  f"KL={kl_f:.2e} TV={tv_f:.2e}")
            if len(self.levels) == 1: self.floor = dict(kl=kl_f, tv=tv_f)
        self.grid = self.levels[0]["grid"]

    def evaluate(self, score_fn):
        L0 = self.levels[0]; g = L0["grid"]
        q = solve_reverse_fp(g, self.sched, score_fn, L0["q0"], self.cfg["fp_steps"])
        kl, tv = kl_tv(L0["p_ref"], q, g.h)
        met = dict(kl=kl, tv=tv, w2=w2_grid(g, L0["p_ref"], q, self.cfg["w2_coarsen"]),
                   kl_vs_analytic=kl_tv(L0["p_an"], q, g.h)[0], kl_err=0.0)
        if len(self.levels) > 1:
            L1 = self.levels[1]
            qc = solve_reverse_fp(L1["grid"], self.sched, score_fn, L1["q0"], self.cfg["fp_steps"])
            met["kl_coarse"] = kl_tv(L1["p_ref"], qc, L1["grid"].h)[0]
            met["kl_err"] = abs(met["kl"] - met["kl_coarse"])
        return met


# ══════════════════════════════════════════════════════════════════════════════
#  EXPERIMENT
# ══════════════════════════════════════════════════════════════════════════════
def run(cfg, out):
    rng = np.random.default_rng(cfg["seed"] + 3)
    sched = VPLinear(sigma_min=cfg["sigma_min"])
    gmm = make_target(cfg)
    perts = perturbed_mixtures(gmm, cfg["n_pert"], rng)
    print(f"VP-linear: t0={sched.t0:.4g}, T={sched.T}; grid {cfg['grid_N']}^2 on [-{cfg['grid_L']},{cfg['grid_L']}]^2")
    ctx = FPContext(sched, gmm, cfg); grid = ctx.grid; p_an = ctx.levels[0]["p_an"]

    # time-integrated analytic energies: ||grad psi_j||^2_{p*_s} and Fisher(p*_s) (cell quadrature)
    sq, wq = quad_nodes(sched, cfg["quad_panels"], cfg["quad_order"])
    g2w = wq * np.array([float(sched.g2(s)) for s in sq])
    Fq, Gq = np.zeros(len(sq)), np.zeros((len(perts), len(sq)))
    for i, s in enumerate(sq):
        P = gmm.at(sched, s); p = grid.density(P); sc = P.score(grid.cells)
        Fq[i] = np.sum(p * (sc ** 2).sum(1)) * grid.h ** 2
        for j, pt in enumerate(perts):
            Gq[j, i] = np.sum(p * ((pt.at(sched, s).score(grid.cells) - sc) ** 2).sum(1)) * grid.h ** 2
    IF = 0.5 * np.sum(g2w * Fq); IG = 0.5 * (Gq * g2w[None]).sum(1)

    # amplitudes: exact zeros for pure-solenoidal / pure-gradient cases, one (a=1, b=0) case whose
    # endpoint is known analytically (hat p_t0 = p~_t0 up to KL(p*_T || p~_T), by data processing)
    n = cfg["n_fields"]; n0 = max(2, n // 8)
    a = np.exp(rng.uniform(*np.log(cfg["a_range"]), n)); b = np.exp(rng.uniform(*np.log(cfg["b_range"]), n))
    a[:n0] = 0.0; b[n0:2 * n0] = 0.0; a[2 * n0] = 1.0; b[2 * n0] = 0.0
    jj = np.arange(n) % len(perts)
    rows = []
    for i in range(n):
        A, Bv, pt = float(a[i]), float(b[i]), perts[jj[i]]
        def score_fn(s, x, A=A, Bv=Bv, pt=pt):
            sc = gmm.at(sched, s).score(x)
            return sc + A * (pt.at(sched, s).score(x) - sc) + Bv * rot90(sc)
        t_ = time.time()
        met = ctx.evaluate(score_fn)
        row = dict(a=A, b=Bv, pert=int(jj[i]), bound_ours=A * A * IG[jj[i]],
                   bound_girsanov=A * A * IG[jj[i]] + Bv * Bv * IF, **met)
        row["sol_share"] = Bv * Bv * IF / max(row["bound_girsanov"], 1e-300)
        if A == 1.0 and Bv == 0.0:
            row["kl_closed_form_pstar_vs_ptilde_t0"] = kl_tv(p_an, grid.density(pt.at(sched, sched.t0)), grid.h)[0]
            row["kl_T_mismatch"] = kl_tv(grid.density(gmm.at(sched, sched.T)),
                                         grid.density(pt.at(sched, sched.T)), grid.h)[0]
        rows.append(row)
        print(f"  [{i + 1:3d}/{n}] a={A:.3f} b={Bv:.3f}  KL={met['kl']:.2e}(±{met['kl_err']:.1e}) "
              f"TV={met['tv']:.2e} W2={met['w2']:.2e} | ours={row['bound_ours']:.2e} "
              f"Girsanov={row['bound_girsanov']:.2e} ({time.time() - t_:.1f}s)")

    # ── summary ───────────────────────────────────────────────────────────────
    kl = np.array([r["kl"] for r in rows]); err = np.array([r["kl_err"] for r in rows])
    bo = np.array([r["bound_ours"] for r in rows]); bg = np.array([r["bound_girsanov"] for r in rows])
    st = dict(n=n, solver_floor=ctx.floor,
              valid_ours=float(np.mean(kl <= bo * (1 + 1e-6) + 1e-12)),
              valid_ours_within_numerical_error=float(np.mean(kl - err <= bo + 1e-12)),
              valid_girsanov=float(np.mean(kl <= bg * (1 + 1e-6) + 1e-12)),
              median_kl_over_ours=float(np.median(kl[bo > 0] / bo[bo > 0])),
              median_kl_over_girsanov=float(np.median(kl / bg)),
              pure_solenoidal_max_kl=float(kl[a == 0].max()),
              pure_solenoidal_max_girsanov=float(bg[a == 0].max()))
    for d in ("kl", "tv", "w2"):
        v = [r[d] for r in rows]
        st[f"rho_{d}_full"], st[f"rho_{d}_grad"] = _spear(v, bg), _spear(v, bo)
    print(f"\n  KL <= ours: {100 * st['valid_ours']:.0f}% (within numerical error: "
          f"{100 * st['valid_ours_within_numerical_error']:.0f}%)   KL <= Girsanov: {100 * st['valid_girsanov']:.0f}%")
    print(f"  median KL/ours = {st['median_kl_over_ours']:.3f}   median KL/Girsanov = {st['median_kl_over_girsanov']:.3f}")
    print(f"  Spearman (full | grad):  KL {st['rho_kl_full']:.2f} | {st['rho_kl_grad']:.2f}   "
          f"TV {st['rho_tv_full']:.2f} | {st['rho_tv_grad']:.2f}   W2 {st['rho_w2_full']:.2f} | {st['rho_w2_grad']:.2f}")
    print(f"  pure solenoidal errors: max KL = {st['pure_solenoidal_max_kl']:.2e} "
          f"(Girsanov bound up to {st['pure_solenoidal_max_girsanov']:.2f})")
    plot_scatter(rows, out)
    return dict(config=cfg, rows=rows, stats=st)


def plot_scatter(rows, out):
    """(a) true KL vs Girsanov bound, (b) true KL vs our bound; colour = solenoidal share."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    kl = np.array([r["kl"] for r in rows]); share = np.array([r["sol_share"] for r in rows])
    err = np.array([r["kl_err"] for r in rows])
    allv = np.concatenate([kl[kl > 0], [r["bound_girsanov"] for r in rows],
                           [r["bound_ours"] for r in rows if r["bound_ours"] > 0]])
    lo, hi = 10 ** np.floor(np.log10(allv.min())), 10 ** np.ceil(np.log10(allv.max()))
    for ax, key, lab in [(axes[0], "bound_girsanov", r"Girsanov bound $\frac{1}{2}\int g^2\|e_s\|^2$"),
                         (axes[1], "bound_ours", r"Our bound $\frac{1}{2}\int g^2\|\Pi_{\mathcal{G}}e_s\|^2$")]:
        x = np.array([r[key] for r in rows]); ok = (x > 0) & (kl > 0)
        ax.errorbar(x[ok], kl[ok], yerr=[np.minimum(err[ok], 0.95 * kl[ok]), err[ok]], fmt="none",
                    ecolor=COLORS["gray"], elinewidth=1.0, capsize=0, zorder=2)
        ax.scatter(x[ok], kl[ok], c=share[ok], cmap=CMAP_SHARE, vmin=0, vmax=1, s=55,
                   edgecolor="black", linewidth=0.4, zorder=3)
        ax.plot([lo, hi], [lo, hi], color=COLORS["gray"], ls="--", lw=1.5, zorder=1)
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_xlabel(lab); ax.grid(True, which="major")
        ax.text(0.04, 0.95, rf"Spearman $\rho={_spear(kl, x):.2f}$", transform=ax.transAxes, va="top", fontsize=15)
        ax.text(0.96, 0.05, r"bound valid below $y=x$", transform=ax.transAxes, ha="right", fontsize=12,
                color=COLORS["gray"])
    axes[0].set_ylabel(r"True $\mathrm{KL}(p^\star_{t_0}\,\|\,\hat p_{t_0})$")
    plt.tight_layout(rect=[0, 0, 0.9, 1])
    cax = fig.add_axes([0.915, 0.2, 0.015, 0.7])
    cb = fig.colorbar(plt.cm.ScalarMappable(cmap=CMAP_SHARE, norm=plt.Normalize(0, 1)), cax=cax)
    cb.set_label("Solenoidal share", fontsize=16); cb.ax.tick_params(labelsize=14)
    savefig(fig, out, "KL_vs_bounds")


def _json(o):
    if isinstance(o, dict): return {str(k): _json(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [_json(v) for v in o]
    if isinstance(o, np.ndarray): return o.tolist()
    if isinstance(o, np.floating): return float(o)
    if isinstance(o, np.integer): return int(o)
    return o


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--quick", action="store_true")
    pa.add_argument("--out_dir", default=None)
    pa.add_argument("--replot", action="store_true", help="redraw the figure from results.json only")
    args = pa.parse_args()
    cfg = dict(CFG)
    if args.quick: cfg.update(QUICK)
    if args.out_dir: cfg["out_dir"] = args.out_dir
    out = Path(cfg["out_dir"]); out.mkdir(parents=True, exist_ok=True)
    print(f"out_dir={out}  LaTeX={'yes' if USE_TEX else 'no (Computer Modern fallback)'}  "
          f"POT={'yes' if HAVE_POT else 'no (sliced W2 fallback)'}")
    if args.replot:
        plot_scatter(json.loads((out / "results.json").read_text())["rows"], out)
    else:
        t_ = time.time()
        res = run(cfg, out)
        (out / "results.json").write_text(json.dumps(_json(res), indent=1))
        print(f"\nDone in {time.time() - t_:.0f}s → {out}/")


if __name__ == "__main__":
    main()