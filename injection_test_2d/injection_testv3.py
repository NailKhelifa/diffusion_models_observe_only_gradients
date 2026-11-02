"""
Analytic invisibility demo: headline figure for the paper.

We build a 2D 8-Gaussian mixture with closed-form noised density at every
noise level.  We construct an invisible error  e_inv = J @ grad log p*_s
(J is 90-degree rotation), which is exactly weighted-divergence-free for
p*_s and therefore in G_s^perp.  We compare:

    (a) reverse sampling with the true score,
    (b) reverse sampling with true score + invisible error (scaled to huge energy),
    (c) reverse sampling with true score + observable gradient error,
        rescaled to the SAME time-integrated L2(p*) energy as (b).

Run:  python injection_testv3.py
Output: Fig1_reproduced.pdf / Fig1_reproduced.png
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from matplotlib.colors import to_rgba
# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["cmr10", "Computer Modern Roman", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "text.usetex": False,
    "axes.formatter.use_mathtext": True,
    "font.size": 24,
    "axes.labelsize": 24,
    "axes.titlesize": 24,
    "legend.fontsize": 24,
    "xtick.labelsize": 24,
    "ytick.labelsize": 24,
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
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "-",
})
COLORS = {
    "blue": "#1A487C", "red": "#E63946", "green": "#0D695E",
    "orange": "#E76F51", "purple": "#6C5B7B", "gray": "#8D99AE",
    "dark": "#000000", "gold": "#E9C46A",
}

SEED = 0
np.random.seed(SEED)

# ---------------------------------------------------------------------------
# Target distribution: two-moons Gaussian mixture
# ---------------------------------------------------------------------------
K_PER_MOON = 32
K = 2 * K_PER_MOON
SIGMA_MIX = 0.08

theta = np.linspace(0.05, np.pi - 0.05, K_PER_MOON)

moon_1 = np.stack([
    2.4 * np.cos(theta),
    1.2 * np.sin(theta),
], axis=1)

moon_2 = np.stack([
    2.4 * np.cos(theta) + 1.2,
    -1.2 * np.sin(theta) - 0.45,
], axis=1)

MUS = np.concatenate([moon_1, moon_2], axis=0)
# ---------------------------------------------------------------------------
# VP-SDE schedule.  dX = -0.5 beta(t) X dt + sqrt(beta(t)) dW, t in [0,1].
# Marginals: X_t = alpha_t X_0 + sqrt(1 - alpha_t^2) epsilon.
# ---------------------------------------------------------------------------
BETA_MIN, BETA_MAX = 0.1, 20.0
T_MIN, T_MAX = 1e-3, 1.0


def beta(t):
    return BETA_MIN + (BETA_MAX - BETA_MIN) * t


def _int_beta(t):
    return BETA_MIN * t + 0.5 * (BETA_MAX - BETA_MIN) * t ** 2


def marginal_params(t):
    a = np.exp(-0.5 * _int_beta(t))
    return a, 1.0 - a ** 2


# ---------------------------------------------------------------------------
# Closed-form density and score of the noised mixture at time t
# ---------------------------------------------------------------------------
def _log_comps(x, alpha_t, sig2_t):
    """Log of un-normalised Gaussian component evaluations: (N, K)."""
    var_t = alpha_t ** 2 * SIGMA_MIX ** 2 + sig2_t
    centers = alpha_t * MUS                       # (K, 2)
    dx = x[:, None, :] - centers[None, :, :]      # (N, K, 2)
    return -0.5 * np.sum(dx ** 2, axis=-1) / var_t - np.log(2 * np.pi * var_t), var_t, dx


def log_pstar(x, alpha_t, sig2_t):
    lc, _, _ = _log_comps(x, alpha_t, sig2_t)
    return np.log(1.0 / K) + np.logaddexp.reduce(lc, axis=1)


def score_pstar(x, alpha_t, sig2_t):
    lc, var_t, dx = _log_comps(x, alpha_t, sig2_t)
    logw = lc - np.logaddexp.reduce(lc, axis=1, keepdims=True)
    w = np.exp(logw)                              # (N, K), sums to 1 along K
    return -np.sum(w[:, :, None] * dx, axis=1) / var_t     # (N, 2)


# ---------------------------------------------------------------------------
# Error fields
# ---------------------------------------------------------------------------
J_ROT = np.array([[0.0, -1.0], [1.0, 0.0]])       # 90-deg CCW rotation


def stream_potential_and_grad(x, t):
    """
    Localized scalar stream potential phi_t.

    The field is constructed so that

        p_t^* e_inv = J grad(phi_t),

    hence

        div(p_t^* e_inv) = div(J grad(phi_t)) = 0.

    This is an exactly invisible weighted-divergence-free perturbation,
    but it is no longer just a rotated score.
    """
    rho = 2.0 + 0.75 * t

    # Two localized vortices, one near each moon.
    c1 = np.array([-0.9, 0.75])
    c2 = np.array([1.5, -1.0])

    z1 = x - c1[None, :]
    z2 = x - c2[None, :]

    r1 = np.sum(z1 ** 2, axis=1, keepdims=True)
    r2 = np.sum(z2 ** 2, axis=1, keepdims=True)

    phi1 = np.exp(-0.5 * r1 / rho ** 2)
    phi2 = -0.85 * np.exp(-0.5 * r2 / rho ** 2)

    phi = phi1 + phi2

    grad_phi1 = -(z1 / rho ** 2) * phi1
    grad_phi2 = -(z2 / rho ** 2) * phi2
    grad_phi = grad_phi1 + grad_phi2

    return phi, grad_phi


def invisible_error(x, t, scale):
    """
    Localized stream-function invisible error.

    We define

        e_inv = scale * J grad(phi_t) / p_t^*.

    Therefore,

        p_t^* e_inv = scale * J grad(phi_t),

    and since J is skew-symmetric,

        div(p_t^* e_inv) = div(J grad(phi_t)) = 0.

    To avoid irrelevant tail explosions where p_t^* is tiny, the denominator
    is softly clipped. This does not affect samples in the high-density region
    but makes the reverse sampler numerically stable.
    """
    a, s2 = marginal_params(t)
    log_p = log_pstar(x, a, s2)

    # Soft floor only for numerical stability in extreme tails.
    p = np.exp(np.maximum(log_p, -14.0))

    _, grad_phi = stream_potential_and_grad(x, t)

    rot_grad_phi = grad_phi @ J_ROT.T
    return scale * rot_grad_phi / p[:, None]


def observable_error(x, t, scale):
    """e_obs = grad chi  where chi(x) = scale * cos(k x1) cos(k x2).
    A bounded, spatially-periodic gradient field -- the canonical example
    of high-frequency noise in a learned score.  Pure gradient, so fully
    visible to Fokker-Planck.  Tuned so its L^2 energy can be matched to
    the invisible perturbation while producing a visually recognizable
    but clearly distorted ring of modes."""
    k = 2.0
    e = np.empty_like(x)
    e[:, 0] = -scale * k * np.sin(k * x[:, 0]) * np.cos(k * x[:, 1])
    e[:, 1] = -scale * k * np.cos(k * x[:, 0]) * np.sin(k * x[:, 1])
    return e


# ---------------------------------------------------------------------------
# Monte-Carlo estimate of time-integrated L2(p*_t) energy of an error field
# ---------------------------------------------------------------------------
def l2_path_energy(err_fn, scale, n_t=25, n_x=3000):
    ts = np.linspace(T_MIN, T_MAX, n_t)
    total = 0.0
    for t in ts:
        a, s2 = marginal_params(t)
        idx = np.random.randint(0, K, size=n_x)
        x0 = MUS[idx] + SIGMA_MIX * np.random.randn(n_x, 2)
        x = a * x0 + np.sqrt(s2) * np.random.randn(n_x, 2)
        e = err_fn(x, t, scale)
        total += np.mean(np.sum(e ** 2, axis=-1))
    return total * (T_MAX - T_MIN) / n_t


# ---------------------------------------------------------------------------
# Reverse-time Euler-Maruyama sampler
# ---------------------------------------------------------------------------
def reverse_sample(err_fn, err_scale, n_samples=5000, n_steps=500, seed=1):
    rng = np.random.default_rng(seed)
    ts = np.linspace(T_MAX, T_MIN, n_steps + 1)
    dt = -(T_MAX - T_MIN) / n_steps               # negative (backward)
    x = rng.standard_normal((n_samples, 2))
    for i in range(n_steps):
        t = ts[i]
        a, s2 = marginal_params(t)
        s = score_pstar(x, a, s2)
        if err_fn is not None:
            s = s + err_fn(x, t, err_scale)
        drift = -0.5 * beta(t) * x - beta(t) * s
        x = x + drift * dt + np.sqrt(beta(t) * abs(dt)) * rng.standard_normal(x.shape)
        # mild clipping to avoid numerical blow-up in extreme cases
        x = np.clip(x, -10.0, 10.0)
    return x


# ---------------------------------------------------------------------------
# Match L2 energies
# ---------------------------------------------------------------------------
print("Computing L2 path energies ...")
SCALE_INV = 0.03
E_INV = l2_path_energy(invisible_error, SCALE_INV)
# calibrate observable to match
E_OBS_UNIT = l2_path_energy(observable_error, 1.0)
SCALE_OBS = np.sqrt(E_INV / E_OBS_UNIT)
E_OBS = l2_path_energy(observable_error, SCALE_OBS)
print(f"  invisible  scale = {SCALE_INV:.3f}, path energy = {E_INV:.3f}")
print(f"  observable scale = {SCALE_OBS:.3f}, path energy = {E_OBS:.3f}")


# ---------------------------------------------------------------------------
# Sample all three variants
# ---------------------------------------------------------------------------
print("Sampling ...")
N_SAMPLES = 8000
x_true = reverse_sample(None,               0.0,       N_SAMPLES, seed=11)
x_inv  = reverse_sample(invisible_error,    SCALE_INV, N_SAMPLES, seed=22)
x_obs  = reverse_sample(observable_error,   SCALE_OBS, N_SAMPLES, seed=33)


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
def kde_panel(ax, samples, title, energy_label, highlight,
              field_fn=None, field_scale=None):
    """KDE contour of samples, with optional streamplot overlay showing the
    error field that was applied during sampling."""
    lim = 4.5
    gx = np.linspace(-lim, lim, 220)
    gy = np.linspace(-lim, lim, 220)
    GX, GY = np.meshgrid(gx, gy)
    grid = np.stack([GX.ravel(), GY.ravel()], axis=1)

    mask = np.all(np.abs(samples) < 9.0, axis=1)
    kde = gaussian_kde(samples[mask].T, bw_method=0.12)
    Z = kde(grid.T).reshape(GX.shape)

    ax.contourf(GX, GY, Z, levels=18, cmap="Blues")

    # Optional: overlay the error vector field as streamlines, evaluated at
    # a representative mid-schedule time, to visually show the field's
    # magnitude and structure is nontrivial.
    if field_fn is not None and field_scale is not None:
        t_mid = 0.15
        n_q = 26
        qx = np.linspace(-lim, lim, n_q)
        qy = np.linspace(-lim, lim, n_q)
        QX, QY = np.meshgrid(qx, qy)
        qpts = np.stack([QX.ravel(), QY.ravel()], axis=1)
        F = field_fn(qpts, t_mid, field_scale).reshape(n_q, n_q, 2)
        U, V = F[..., 0], F[..., 1]

        # Slight transparency and darker green for the invisible field panel
        stream_color = highlight
        stream_alpha = 0.55
        if highlight == COLORS["green"]:
            stream_color = COLORS["green"]   # darker green
            stream_alpha = 0.45
        elif highlight == COLORS["orange"]:
            stream_alpha = 0.45

        ax.streamplot(
            qx, qy, U, V,
            color=to_rgba(stream_color, stream_alpha),
            density=1.15,
            linewidth=0.85,
            arrowsize=0.9,
            maxlength=3.0,
        )

    # Draw the target support as soft red curves instead of crosses
    moon1 = MUS[:K_PER_MOON]
    moon2 = MUS[K_PER_MOON:]

    # optional faint white underlay to improve visibility on top of blue KDE
    for moon in [moon1, moon2]:
        ax.plot(
            moon[:, 0], moon[:, 1],
            color=(1.0, 1.0, 1.0, 0.30),
            linewidth=5.0,
            solid_capstyle="round",
            zorder=4,
        )

    # main red semi-transparent lines
    for moon in [moon1, moon2]:
        ax.plot(
            moon[:, 0], moon[:, 1],
            color=to_rgba(COLORS["red"], 0.68),
            linewidth=3.0,
            solid_capstyle="round",
            zorder=5,
    )

    ax.set_title(title, fontsize=24, color=highlight, pad=10)
    ax.set_xlim([-lim, lim])
    ax.set_ylim([-lim, lim])
    ax.set_xticks([-4, -2, 0, 2, 4])
    ax.set_yticks([-4, -2, 0, 2, 4])
    ax.set_aspect("equal")

    ax.text(0.5, -0.17, energy_label, transform=ax.transAxes,
            ha="center", va="top", fontsize=24)


fig, axes = plt.subplots(1, 3, figsize=(16.5, 6.2))

kde_panel(axes[0], x_true,
          r"No Error",
          r"$\int_{0}^{T}$" + " " + r"$\mathbb{{E}}[\|\mathbf{{e}}_s\|^2_2]\,\mathrm{{d}}s = 0$",
          COLORS["dark"])

kde_panel(axes[1], x_inv,
          r"Invisible Solenoidal Error",
          r"$\int_{0}^{T}$" + " " + rf"$\mathbb{{E}}[\|\mathbf{{e}}_s\|^2_2]\,\mathrm{{d}}s \approx {E_INV:.1f}$",
          COLORS["green"],
          field_fn=invisible_error, field_scale=SCALE_INV)

kde_panel(axes[2], x_obs,
          r"Observable Gradient Error",
          r"$\int_{0}^{T}$" + " " + rf"$\mathbb{{E}}[\|\mathbf{{e}}_s\|^2_2]\,\mathrm{{d}}s \approx {E_OBS:.1f}$",
          COLORS["orange"],
          field_fn=observable_error, field_scale=SCALE_OBS)

plt.tight_layout()
plt.savefig("Fig1_reproduced.pdf", bbox_inches="tight")
plt.savefig("Fig1_reproduced.png", bbox_inches="tight", dpi=200)
print("Saved Fig1_reproduced.pdf / .png")
