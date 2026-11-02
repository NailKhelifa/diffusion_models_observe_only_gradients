# Diffusion Models Observe Only Gradients: A Geometric Perspective on Score Matching Errors

Code for the NeurIPS 2026 paper by Naïl B. Khelifa, Richard E. Turner and Ramji Venkataramanan (University of Cambridge).

## TL;DR

Score-based diffusion models are trained to minimise the $`L^2`$ score-matching error. The standard (Girsanov) analysis bounds $`\mathrm{KL}(p^\star \Vert \hat p)`$ by the time-integrated $`L^2`$ score error. But the marginals of the reverse process see the score error $`𝐞_s = 𝐬_\theta(\cdot,s) - \nabla\log p^\star_s`$ only through the weighted divergence $`\nabla\cdot(p^\star_s 𝐞_s)`$. Split $`𝐞_s`$ with a weighted Helmholtz–Hodge decomposition in $`L^2(p^\star_s)`$:

```math
𝐞_s = \underbrace{\Pi_{𝒢_s}𝐞_s}_{\text{gradient (observable)}} + \underbrace{\Pi_{𝒢_s^\perp}𝐞_s}_{\text{solenoidal (invisible)}} .
```

The solenoidal part does not change the generated distribution at all. The paper shows:

1. **Impossibility.** No increasing function of the full $`L^2`$ score error can uniformly lower-bound any divergence between $`\hat p`$ and $`p^\star`$.
2. **A tighter KL bound.** $`\mathrm{KL}(p^\star_{t_0}\Vert \hat p_{t_0}) \le \tfrac12\int_{t_0}^T \sigma_s^2 \Vert \Pi_{𝒢_s}𝐞_s\Vert^2_{L^2(p^\star_s)} \mathrm{d}s`$.
3. **A computable diagnostic** for the gradient component, which needs only forward-noised data and the frozen score network (next section).

Empirically, on Fashion-MNIST and CIFAR-10 the gradient component of the error drops by two to three orders of magnitude during training, while the remaining error is dominated by the solenoidal component.

## How the gradient component is estimated

For each diffusion time $`s`$, the gradient component solves a variational problem over scalar potentials $`\varphi`$:

```math
\Vert \Pi_{𝒢_s}𝐞_s\Vert^2_{L^2(p^\star_s)} = \sup_{\varphi} 𝔼_{𝐱_s\sim p^\star_s}\big[2 \nabla\varphi\cdot𝐞_s - \Vert \nabla\varphi\Vert^2 \big].
```

The supremum is reached at $`\nabla\varphi = \Pi_{𝒢_s}𝐞_s`$. Only gradient fields $`\nabla\varphi`$ enter, so the solenoidal part of $`𝐞_s`$ contributes nothing. Since $`𝐞_s`$ involves the unknown score $`\nabla\log p^\star_s`$. Instead we use the denoising-score-matching residual

```math
𝐫 = 𝐬_\theta(𝐱_s, s) - \nabla_{𝐱_s}\log p(𝐱_s\mid𝐱_0) = 𝐬_\theta(𝐱_s, s) + 𝛆/\sigma_s, \qquad 𝐱_s = \alpha_s𝐱_0 + \sigma_s𝛆 .
```

Because $`𝔼[𝐫\mid𝐱_s] = 𝐞_s(𝐱_s)`$ and $`\nabla\varphi`$ depends on $`𝐱_s`$ only, $`𝔼[\nabla\varphi\cdot𝐫] = 𝔼[\nabla\varphi\cdot𝐞_s]`$. So the objective can be computed from noised data alone. For a network that predicts noise, $`𝐬_\theta = -𝛆_\theta/\sigma_s`$, so $`𝐫 = (𝛆 - 𝛆_\theta)/\sigma_s`$.

**The procedure (Algorithm 1).**

1. Freeze the score network.
2. Draw $`𝐱_0`$ from the training images, a diffusion time $`s`$, and noise $`𝛆`$. Form $`𝐱_s`$ and the residual $`𝐫`$.
3. Train a freshly initialised, time-conditioned scalar network $`\varphi_\psi(𝐱, s)`$ (the *critic*) to maximise $`J(\psi) = 𝔼\big[2\nabla_{𝐱}\varphi_\psi\cdot𝐫 - \Vert \nabla_{𝐱}\varphi_\psi\Vert^2\big]`$. $`\nabla_{𝐱}\varphi_\psi`$ is computed with autograd, and training the critic backpropagates through it (second derivatives).
4. Report:
   - **gradient part** $`\hat{ℰ}_{\mathrm{grad}}`$: the value of $`J`$ for the trained critic. A critic of finite capacity can only reach a value at or below the supremum, so this is a lower estimate.
   - **total** $`\hat{ℰ}_{\mathrm{full}} = 𝔼\Vert 𝐫\Vert^2`$.
   - **solenoidal part** $`\max(\hat{ℰ}_{\mathrm{full}} - \hat{ℰ}_{\mathrm{grad}}, 0)`$, which is therefore an upper estimate.

**Settings used for the figures.**

| | Fashion-MNIST (`diagnostic_fmnist.py`) | CIFAR-10 (`train_cifar10.py`) |
|---|---|---|
| When it runs | After training, on each saved checkpoint (every 20 epochs; raw weights, not EMA) | During training, every 10 epochs, on the EMA weights |
| Score network output | Score | Noise $`𝛆`$ (converted as above) |
| Data | Fixed pool of 6,400 training images (chosen with seed 0) | First 6,400 training images. Times $`s`$ and noise $`𝛆`$ are drawn once and then kept fixed |
| Time distribution | Log-uniform on $`[0.05, 4]`$ | Uniform on $`[10^{-3}, 1]`$ |
| Critic | Two-level conv net with self-attention, base width 24, global pooling to a scalar | Two-level U-Net with skip connections, base width 128; $`\varphi`$ is the sum over pixels of a one-channel output |
| Critic training | 1,500 Adam steps, batch 256, a fresh draw of $`(𝐱_0, s, 𝛆)`$ at every step | 4,000 Adam steps, batch 128, cycling through the 50 fixed batches |
| Learning rate | $`10^{-3}`$, cosine decay to $`10^{-5}`$; gradient clipping 1.0 | Same |
| Reported $`\hat{ℰ}_{\mathrm{grad}}`$ | Mean of $`J`$ over the last 75 steps. Each value is computed on a new batch before the update on that batch | $`J`$ of the trained critic, evaluated on the same 50 fixed batches |
| Reported $`\hat{ℰ}_{\mathrm{full}}`$ | $`𝔼\Vert 𝐫\Vert^2`$ summed over the 784 pixels, 50 × 128 fresh draws | $`𝔼\Vert 𝐫\Vert^2`$ averaged over the 3,072 pixel values, same 50 fixed batches |
| Critic restarts | 1 | 1 |

**Sample quality (FID)** is only used for the Fashion-MNIST correlations (Figure 2). It is computed in the 128-d feature space of a small CNN classifier trained at the start of `train_fmnist.py`, on 5,000 samples from a 200-step predictor–corrector sampler of the EMA model. 

## Repository structure

```
injection_test_2d/   2D demo: invisible vs observable errors of equal L2 energy (Figure 1)
fashionMNIST/        Train DSM models at 3 sizes x 3 seeds, run the diagnostic on every checkpoint (Figure 2, appendix)
cifar10/             Train DSM models (4 seeds) with the diagnostic run during training, plot it (Figure 3)
low_dim_hodge_experiment/   2D mixture with an exactly known Hodge decomposition: true KL vs both bounds
```

Published figures are in each folder's `figures/` (`Fig1.*` for the 2D demo). The scripts write to `figures_reproduced/` (`Fig1_reproduced.*`), so a rerun never overwrites the published files.

### `injection_test_2d/`

| File | Role |
|---|---|
| `injection_testv3.py` | Builds a two-moons Gaussian mixture (64 components) whose noised densities and scores are known exactly under a VP-SDE. It then runs reverse Euler–Maruyama sampling three ways: (a) true score; (b) true score + an exactly weighted-divergence-free error; (c) true score + a pure-gradient error rescaled to the same time-integrated $`L^2(p^\star)`$ energy as (b). |

The paper's Figure 1 uses a mixture of 8 Gaussians arranged on a ring. Reproducing it only requires changing the target distribution in the script (the component means `MUS`). The two-moons target used here (`Fig1.*`) illustrates the same result on a dataset whose shape is not Gaussian.

### `fashionMNIST/`

VP-SDE with $`\alpha(t)=e^{-t/2}`$, $`\sigma(t)=\sqrt{1-e^{-t}}`$, $`T=4`$, $`t_0=0.05`$. The score network is a U-Net with self-attention that outputs the score directly.

| File | Role |
|---|---|
| `train_fmnist.py` | **Training.** Three model sizes: `tiny` (width 32, 1 res-block), `small` (48, 2), `full` (64, 2). Each is trained with three seeds for 400 epochs. Checkpoints are saved every 20 epochs to computes the FID. |
| `diagnostic_fmnist.py` | **Diagnostic.** Runs the procedure above on every checkpoint and caches the results in `diagnostic_cache/`. Draws `error_decomposition_<size>_score_dsm` and `kl_bounds_<size>_score_dsm`, and writes the Spearman table. With `--plots-only` it redraws from the cache alone. |
| `plot_spearman.py` | Spearman ρ(FID, ·), mean ± std over seeds. |
| `diagnostic_cache/` | Logged diagnostic results behind every Fashion-MNIST figure. |

### `cifar10/`

VP-SDE with linear $`\beta`$ from 0.1 to 20, $`T=1`$, $`t_0=10^{-3}`$. The U-Net (width 96, 2 res-blocks per level, 14.7M parameters) predicts the noise $`𝛆`$.

| File | Role |
|---|---|
| `train_cifar10.py` | **Training + diagnostic.** Trains four seeds for 350 epochs via DSM loss + Min-SNR-5 weighting. Every 10 epochs it runs the diagnostic and logs the results to `runs/ch96/seed<s>/hist.json`. |
| `plot_cifar10.py` | Reads the `hist.json` files and plots median + IQR over seeds of the error split and the two bound curves → `combined_metrics` (**Figure 3**). |


### `low_dim_hodge_experiment/`

A low-dimensional close-form check for the Hodge decomposition of the score error, both bounds, and the true KL of the generated distribution. No network is trained and no critic is needed.

**Setup.**
- **Target:** a mixture of 8 anisotropic Gaussians with unequal weights, on a ring of radius 2 in $`ℝ^2`$.
- **Forward process:** VP-SDE with linear $`\beta`$ from 0.1 to 20, $`T=1`$. 
- **Score errors with a known decomposition:** We define $`𝐞_s = a \nabla\psi_s + b J\nabla\log p^\star_s`$, where $`\psi_s = \log\tilde p_s - \log p^\star_s`$, where $`\tilde p`$ is the gradient part, built as a mixture with perturbed weights and means and $`J`$ is the 90° rotation, exactly solenoidal (proof of Theorem 3.1.). Hence $`\Pi_{𝒢_s}𝐞_s = a\nabla\psi_s`$ and $`\Pi_{𝒢_s^\perp}𝐞_s = bJ\nabla\log p^\star_s`$.
- **Computing bounds:** both bounds are exact time integral computed by discretizing time on a grid (Gauss-Legendre quadrature). 
- **True KL:** the generated law $`\hat p_{t_0}`$ is computed by solving the reverse Fokker–Planck equation with the perturbed score. $`\mathrm{KL}(p^\star_{t_0}\Vert \hat p_{t_0})`$, TV and $`W_2`$ are measured against a reference solve with the exact score, on the same grid and time steps.

**Results** (`lowdim_results/results.json`): Spearman correlation between the true KL and each bound: 0.15 for Girsanov vs **0.99** for ours. With TV: 0.15 vs 1.00; with $`W_2`$: 0.09 vs 0.95.

| File | Role |
|---|---|
| `low_dim_experiment.py` | Runs the whole experiment on CPU and writes `KL_vs_bounds.{pdf,png}` (true KL against each bound, coloured by the solenoidal share of $`\Vert 𝐞\Vert^2`$) and `results.json`. `--quick` runs a smaller configuration in about 30 s; `--replot` redraws the figure from `results.json`. |
| `lowdim_results/` | The figure and metric logs in `results.json`. |

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
