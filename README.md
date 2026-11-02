# Diffusion Models Observe Only Gradients: A Geometric Perspective on Score Matching Errors

Code for the NeurIPS 2026 paper by Naïl B. Khelifa, Richard E. Turner and Ramji Venkataramanan (University of Cambridge).

## TL;DR

Score-based diffusion models are trained to minimise the $L^2$ score-matching error. The standard (Girsanov) analysis bounds $\mathrm{KL}(p^\star \|\ \hat p)$ by the time-integrated $L^2$ score error. But the marginals of the reverse process see the score error $\mathbf{e}_s = \mathbf s_\theta(\cdot,s) - \nabla\log p^\star_s$ only through the weighted divergence $\nabla\cdot(p^\star_s \mathbf{e}_s)$. Split $\mathbf{e}_s$ with a weighted Helmholtz–Hodge decomposition in $L^2(p^\star_s)$:

$$\mathbf{e}_s = \underbrace{\Pi_{\mathcal G_s}\mathbf{e}_s}_{\text{gradient (observable)}} + \underbrace{\Pi_{\mathcal G_s^\perp}\mathbf{e}_s}_{\text{solenoidal (invisible)}} .$$

The solenoidal part does not change the generated distribution at all. The paper shows:

1. **Impossibility.** No increasing function of the full $L^2$ score error can uniformly lower-bound any divergence between $\hat p$ and $p^\star$.
2. **A tighter KL bound.** $\mathrm{KL}(p^\star_{t_0}\|\hat p_{t_0}) \le \tfrac12\int_{t_0}^T \sigma_s^2 \|\Pi_{\mathcal G_s}\mathbf{e}_s\|^2_{L^2(p^\star_s)} \mathrm ds$.
3. **A computable diagnostic** for the gradient component, which needs only forward-noised data and the frozen score network (next section).

Empirically, on Fashion-MNIST and CIFAR-10 the gradient component of the error drops by two to three orders of magnitude during training, while the remaining error is dominated by the solenoidal component.

## How the gradient component is estimated

**The identity.** For each diffusion time $s$, the gradient component solves a variational problem over scalar potentials $\varphi$:

$$\|\Pi_{\mathcal G_s}\mathbf{e}_s\|^2_{L^2(p^\star_s)} = \sup_{\varphi} \mathbb{E}_{\mathbf x_s\sim p^\star_s}\big[2 \nabla\varphi\cdot\mathbf{e}_s - \|\nabla\varphi\|^2 \big].$$

The supremum is reached at $\nabla\varphi = \Pi_{\mathcal G_s}\mathbf{e}_s$. Only gradient fields $\nabla\varphi$ enter, so the solenoidal part of $\mathbf{e}_s$ contributes nothing.

**Replacing the unknown error with the denoising residual.** $\mathbf{e}_s$ involves the unknown score $\nabla\log p^\star_s$. Instead we use the denoising-score-matching residual

$$\mathbf{r} = \mathbf s_\theta(\mathbf x_s, s) - \nabla_{\mathbf x_s}\log p(\mathbf x_s\mid\mathbf x_0) = \mathbf s_\theta(\mathbf x_s, s) + \boldsymbol\varepsilon/\sigma_s, \qquad \mathbf x_s = \alpha_s\mathbf x_0 + \sigma_s\boldsymbol\varepsilon .$$

Because $\mathbb{E}[\mathbf{r}\mid\mathbf x_s] = \mathbf{e}_s(\mathbf x_s)$ and $\nabla\varphi$ depends on $\mathbf x_s$ only, $\mathbb{E}[\nabla\varphi\cdot\mathbf{r}] = \mathbb{E}[\nabla\varphi\cdot\mathbf{e}_s]$. So the objective can be computed from noised data alone. For a network that predicts noise, $\mathbf s_\theta = -\boldsymbol\varepsilon_\theta/\sigma_s$, so $\mathbf{r} = (\boldsymbol\varepsilon - \boldsymbol\varepsilon_\theta)/\sigma_s$.

**The procedure (Algorithm 1).**

1. Freeze the score network.
2. Draw $\mathbf x_0$ from the training images, a diffusion time $s$, and noise $\boldsymbol\varepsilon$. Form $\mathbf x_s$ and the residual $\mathbf{r}$.
3. Train a freshly initialised, time-conditioned scalar network $\varphi_\psi(\mathbf x, s)$ (the *critic*) to maximise $J(\psi) = \mathbb{E}\big[2\nabla_{\mathbf x}\varphi_\psi\cdot\mathbf{r} - \|\nabla_{\mathbf x}\varphi_\psi\|^2\big]$. $\nabla_{\mathbf x}\varphi_\psi$ is computed with autograd, and training the critic backpropagates through it (second derivatives).
4. Report:
   - **gradient part** $\hat{\mathcal{E}}_{\mathrm{grad}}$: the value of $J$ for the trained critic. A critic of finite capacity can only reach a value at or below the supremum, so this is a lower estimate.
   - **total** $\hat{\mathcal{E}}_{\mathrm{full}} = \mathbb{E}\|\mathbf{r}\|^2$.
   - **solenoidal part** $\max(\hat{\mathcal{E}}_{\mathrm{full}} - \hat{\mathcal{E}}_{\mathrm{grad}}, 0)$, which is therefore an upper estimate.

All expectations also average over diffusion time.

**About the total.** $\mathbb{E}[\|\mathbf{r}\|^2]$ equals the score error plus the constant $\mathbb{E}[\|\nabla\log p(\mathbf x_s\mid\mathbf{x}_0) - \nabla\log p_s(\mathbf{x}_s)\|^2]$, which does not depend on the model. Here $\mathbf{x}_0$ is drawn from the training images, so the reference distribution $p_s$ is the noised training set. For that distribution the constant is small: on Fashion-MNIST it is about 1.8, against a total of about 300 (both summed over pixels, same time distribution). The gradient estimate is not affected by this constant.

**Settings used for the figures.**

| | Fashion-MNIST (`diagnostic_fmnist.py`) | CIFAR-10 (`train_cifar10.py`) |
|---|---|---|
| When it runs | After training, on each saved checkpoint (every 20 epochs; raw weights, not EMA) | During training, every 10 epochs, on the EMA weights |
| Score network output | Score | Noise $\boldsymbol\varepsilon$ (converted as above) |
| Data | Fixed pool of 6,400 training images (chosen with seed 0) | First 6,400 training images. Times $s$ and noise $\boldsymbol\varepsilon$ are drawn once and then kept fixed |
| Time distribution | Log-uniform on $[0.05, 4]$ | Uniform on $[10^{-3}, 1]$ |
| Critic | Two-level conv net with self-attention, base width 24, global pooling to a scalar | Two-level U-Net with skip connections, base width 128; $\varphi$ is the sum over pixels of a one-channel output |
| Critic training | 1,500 Adam steps, batch 256, a fresh draw of $(\mathbf x_0, s, \boldsymbol\varepsilon)$ at every step | 4,000 Adam steps, batch 128, cycling through the 50 fixed batches |
| Learning rate | $10^{-3}$, cosine decay to $10^{-5}$; gradient clipping 1.0 | Same |
| Reported $\hat{\mathcal{E}}_{\mathrm{grad}}$ | Mean of $J$ over the last 75 steps. Each value is computed on a new batch before the update on that batch | $J$ of the trained critic, evaluated on the same 50 fixed batches |
| Reported $\hat{\mathcal{E}}_{\mathrm{full}}$ | $\mathbb{E}\|\mathbf{r}\|^2$ **summed** over the 784 pixels, 50 × 128 fresh draws | $\mathbb{E}\|\mathbf{r}\|^2$ **averaged** over the 3,072 pixel values, same 50 fixed batches |
| Critic restarts | 1 | 1 |

The two datasets normalise the error differently (sum vs. average over pixels), so their y-axes cannot be compared directly.

**From errors to the KL-bound curves.** Both curves are the time-averaged error multiplied by the same constant

$$K = \tfrac12 (T-t_0)\,\overline{\sigma^2},$$

where $\overline{\sigma^2}$ is the mean of $\sigma_s^2$ over diffusion time:
- Fashion-MNIST: over the log-uniform time distribution. The published figures used a 200-draw Monte Carlo estimate, $K = 0.8632$; the exact value is $0.8644$. The script fixes $K = 0.8632$ so that it reproduces the figures exactly.
- CIFAR-10: over a 256-point uniform grid.

"Girsanov bound" $= K\,\hat{\mathcal{E}}_{\mathrm{full}}$ and "Ours" $= K\,\hat{\mathcal{E}}_{\mathrm{grad}}$. The ratio between the curves, and therefore the gap, does not depend on $K$. Read the absolute values as a rescaled estimate, not as exact evaluations of the integrals.

**Sample quality (FID)** is only used for the Fashion-MNIST correlations (Figure 2). It is computed in the 128-d feature space of a small CNN classifier trained at the start of `train_fmnist.py`, on 5,000 samples from a 200-step predictor–corrector sampler of the EMA model. The CIFAR-10 figure does not use FID, and `train_cifar10.py` does not compute it.

## Repository layout

```
injection_test_2d/   2D demo: invisible vs observable errors of equal L2 energy (Figure 1)
fashionMNIST/        Train DSM models at 3 sizes x 3 seeds, run the diagnostic on every checkpoint (Figure 2, appendix)
cifar10/             Train DSM models (4 seeds) with the diagnostic run during training, plot it (Figure 3)
low_dim_hodge_experiment/   2D mixture with an exactly known Hodge decomposition: true KL vs both bounds
```

Published figures are in each folder's `figures/` (`Fig1.*` for the 2D demo). The scripts write to `figures_reproduced/` (`Fig1_reproduced.*`), so a rerun never overwrites the published files.

### `injection_test_2d/`

| File | What it does |
|---|---|
| `injection_testv3.py` | Builds a two-moons Gaussian mixture (64 components) whose noised densities and scores are known exactly under a VP-SDE. It then runs reverse Euler–Maruyama sampling three ways: (a) true score; (b) true score + an exactly weighted-divergence-free error $\mathbf{e}_{\text{inv}}$, which satisfies $p^\star_s\mathbf{e}_{\text{inv}} = J\nabla\phi_s$ with $J$ the 90° rotation; (c) true score + a pure-gradient error rescaled to the **same** time-integrated $L^2(p^\star)$ energy as (b). NumPy only, about a minute on a CPU. |

The paper's Figure 1 uses a mixture of 8 Gaussians arranged on a ring. Reproducing it only requires changing the target distribution in the script (the component means `MUS`). The two-moons target used here (`Fig1.*`) illustrates the same result on a dataset whose shape is not Gaussian.

### `fashionMNIST/`

VP-SDE with $\alpha(t)=e^{-t/2}$, $\sigma(t)=\sqrt{1-e^{-t}}$, $T=4$, $t_0=0.05$. The score network is a U-Net with self-attention that outputs the score directly.

| File | What it does |
|---|---|
| `train_fmnist.py` | **Training.** Three model sizes: `tiny` (width 32, 1 res-block), `small` (48, 2), `full` (64, 2). Each is trained with seeds 0, 1, 2 for 400 epochs (σ²-weighted DSM loss, AdamW lr 5e-4, batch 512, EMA 0.9999). Every 20 epochs it computes FID and saves `results/<size>_s<seed>/ckpt_ep*.pt`. |
| `diagnostic_fmnist.py` | **Diagnostic.** Runs the procedure above on every checkpoint and caches the results in `diagnostic_cache/`. Draws `error_decomposition_<size>_score_dsm` and `kl_bounds_<size>_score_dsm`, and writes the Spearman table. With `--plots-only` it redraws from the cache alone. |
| `plot_spearman.py` | Spearman ρ(FID, ·), mean ± std over seeds, from the cached table → `spearman_bars_score_dsm_v2` (Figure 2). |
| `diagnostic_cache/` | Logged diagnostic results behind every Fashion-MNIST figure. |

### `cifar10/`

VP-SDE with linear $\beta$ from 0.1 to 20, $T=1$, $t_0=10^{-3}$. The U-Net (width 96, 2 res-blocks per level, 14.7M parameters) predicts the noise $\boldsymbol\varepsilon$.

| File | What it does |
|---|---|
| `train_cifar10.py` | **Training + diagnostic.** Trains seeds 45, 47, 49, 52 for 350 epochs. Settings: DSM loss with Min-SNR-5 weighting, AdamW lr 2e-4 (3,000 warm-up steps, cosine decay to 1e-5), batch 128, EMA 0.9999, dropout 0.1. Every 10 epochs it runs the diagnostic and logs the results to `runs/ch96/seed<s>/hist.json`. |
| `plot_cifar10.py` | Reads the `hist.json` files and plots median + IQR over seeds of the error split and the two bound curves → `combined_metrics` (**Figure 3**). |
| `logs_v24/ch96/seed{45,47,49,52}/hist.json` | Logged histories of the four runs behind Figure 3. They also contain the FID of each evaluation (`fid_dsm`) from the original runs. |

### `low_dim_hodge_experiment/`

A low-dimensional check in which everything is known exactly: the Hodge decomposition of the score error, both bounds, and the true KL of the generated distribution. No network is trained and no critic is needed.

**Setup.**
- **Target:** a mixture of 8 anisotropic Gaussians with unequal weights, on a ring of radius 2 in $\mathbb{R}^2$. Its noised marginals $p^\star_s$ and scores are available in closed form.
- **Forward process:** VP-SDE with linear $\beta$ from 0.1 to 20, $T=1$. $t_0$ is chosen so that the noise standard deviation at $t_0$ is 0.05.
- **Score errors with a known decomposition:** $\mathbf{e}_s = a \nabla\psi_s + b J\nabla\log p^\star_s$.
  - $\psi_s = \log\tilde p_s - \log p^\star_s$, where $\tilde p$ is a mixture with perturbed weights and means.
  - $J$ is the 90° rotation. Since $\nabla\cdot(p^\star_s J\nabla\log p^\star_s) = \nabla\cdot(J\nabla p^\star_s) = 0$, this part is exactly solenoidal.
  - Hence $\Pi_{\mathcal{G}_s}\mathbf{e}_s = a\nabla\psi_s$ and $\Pi_{\mathcal{G}_s^\perp}\mathbf{e}_s = bJ\nabla\log p^\star_s$.
- **Both bounds are exact time integrals,** computed by Gauss–Legendre quadrature with $g^2 = \beta(s)$:
  - Girsanov: $\tfrac12\int_{t_0}^T \beta(s)\big(a^2\|\nabla\psi_s\|^2_{L^2(p^\star_s)} + b^2\mathbb{E}_{p^\star_s}\|\nabla\log p^\star_s\|^2\big)\mathrm ds$;
  - ours: $\tfrac12\int_{t_0}^T \beta(s) a^2\|\nabla\psi_s\|^2_{L^2(p^\star_s)}\mathrm ds$.
- **True KL:** the generated law $\hat p_{t_0}$ is computed by solving the reverse Fokker–Planck equation with the perturbed score. The solver uses Scharfetter–Gummel finite volumes and variable-step BDF2, with 200 time steps on a $224^2$ grid over $[-4.5, 4.5]^2$.
  - $\mathrm{KL}(p^\star_{t_0}\|\hat p_{t_0})$, TV and $W_2$ are measured against a reference solve with the exact score, on the same grid and time steps.
  - A second solve on a $149^2$ grid gives a discretisation error bar for each KL.
- **Ensemble:** 48 error fields using 4 perturbed mixtures. The amplitudes are log-uniform: $a \in [0.03, 1]$, $b \in [0.03, 1.5]$.
  - It includes 6 purely solenoidal fields ($a=0$) and 7 purely gradient fields ($b=0$).
  - One gradient field has $a=1, b=0$. Its endpoint is known in closed form ($\hat p_{t_0} = \tilde p_{t_0}$), which checks the solver.

**Results** (`lowdim_results/results.json`):
- Spearman correlation between the true KL and each bound: 0.15 for Girsanov vs **0.99** for ours. With TV: 0.15 vs 1.00; with $W_2$: 0.09 vs 0.95.
- Median KL / bound: 0.023 for Girsanov vs 0.96 for ours. Our bound is nearly tight here.
- KL lies below the Girsanov bound for all 48 fields. It lies below our bound for 42 of 48. The other 6 exceed it by less than their discretisation error bar.
- Purely solenoidal errors: the KL stays below $5 \times 10^{-5}$, close to the solver floor of $2 \times 10^{-5}$, while the Girsanov bound reaches 14.7.
- Closed-form check: for $a=1, b=0$ the solver gives $\mathrm{KL} = 0.4407$, against $\mathrm{KL}(p^\star_{t_0}\|\tilde p_{t_0}) = 0.4427$ (our bound: 0.4427).

| File | What it does |
|---|---|
| `low_dim_experiment.py` | Runs the whole experiment on CPU and writes `KL_vs_bounds.{pdf,png}` (true KL against each bound, coloured by the solenoidal share of $\|\mathbf{e}\|^2$) and `results.json`. `--quick` runs a smaller configuration in about 30 s; `--replot` redraws the figure from `results.json`. |
| `lowdim_results/` | The figure and `results.json` (every field's amplitudes, bounds, KL, TV, $W_2$ and error bar, plus the summary above). |

Needs only `numpy`, `scipy` and `matplotlib`. Optional extras:
- `pot`: computes $W_2$ by exact discrete optimal transport; without it, the script falls back to sliced $W_2$.
- A LaTeX installation with `dvipng`: used for the figure's text; otherwise matplotlib's Computer Modern fonts are used.

## Reproducing the figures

Dependencies: Python ≥ 3.10, `torch`, `torchvision`, `numpy`, `scipy`, `matplotlib`, `tqdm`. The image experiments were run on a single NVIDIA RTX 6000 (96 GB). Datasets are downloaded to `./data` on first use.

### Redraw from the included logs (seconds, CPU)

```bash
cd fashionMNIST && python diagnostic_fmnist.py --plots-only && python plot_spearman.py && cd ..
cd cifar10 && python plot_cifar10.py && cd ..
cd injection_test_2d && python injection_testv3.py && cd ..
cd low_dim_hodge_experiment && python low_dim_experiment.py --replot && cd ..
```

`--replot` rewrites `lowdim_results/KL_vs_bounds.*` in place, with an identical image.

### Rerun the experiments

```bash
# Figure 2 and the Fashion-MNIST appendix figures
cd fashionMNIST
python train_fmnist.py --device cuda:0                         # -> results/{tiny,small,full}_s{0,1,2}/ckpt_ep*.pt
python diagnostic_fmnist.py --device cuda:0 \
    --results-root ./results --cache-dir ./diagnostic_cache_new
python plot_spearman.py --cache-dir ./diagnostic_cache_new

# Figure 3
cd ../cifar10
python train_cifar10.py --device cuda:0                        # -> runs/ch96/seed{45,47,49,52}/hist.json
python plot_cifar10.py --runs_root runs

# Low-dimensional experiment (CPU)
cd ../low_dim_hodge_experiment
python low_dim_experiment.py --out_dir lowdim_results_new      # full configuration
python low_dim_experiment.py --quick                           # smaller configuration -> lowdim_results_quick/
```

Pass `--out_dir` for a full rerun: the default output folder is `lowdim_results/`, which would overwrite the shipped results.

Use a new `--cache-dir` for the Fashion-MNIST diagnostic. The script skips any checkpoint already present in the cache, so the shipped `diagnostic_cache/` would make it reuse the logged results.

### What "reproduce" means here

- **From the included logs:** checked against the files shipped here. Figure 1 (two-moons version), Figure 3 and all six Fashion-MNIST error-split and KL-bound plots come out pixel-identical. The Spearman chart has the same content, with an image size that differs by a few pixels. The low-dimensional figure redrawn with `--replot` is pixel-identical.
- **Retraining:** results match within the seed-to-seed spread shown by the bands, not bit for bit.
  - Mixed precision and non-deterministic GPU kernels are used (CIFAR-10 training, both diagnostics).
  - The original runs also trained a second, exploratory model in the same process, which consumed random numbers; these scripts train only the model analysed in the paper.

## Citation

```bibtex
@inproceedings{khelifa2026observe,
  title     = {Diffusion Models Observe Only Gradients: A Geometric Perspective on Score Matching Errors},
  author    = {Khelifa, Na{\"i}l B. and Turner, Richard E. and Venkataramanan, Ramji},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```

## Contact

Naïl Khelifa — nbk24@cam.ac.uk
