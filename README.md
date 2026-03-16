# From Spikes to Manifold

A compact computational neuroscience study on latent geometry recovery from mixed-selectivity neural population activity. The pipeline compares a nonlinear autoencoder against a PCA baseline, quantifies reconstruction and decoding quality, and exports a paper-style visual package with paired MP4/GIF animations.

## At A Glance

- Focus: latent neural manifold recovery, representation learning, and robustness analysis
- Models: NumPy autoencoder and PCA baseline
- Outputs: 6 scientific PNG figures, 2 MP4 videos, 2 GIF previews, saved artifacts, and evaluation summaries
- Visual style: `DejaVu Serif`, white background, English-only labels, clean figure layouts

## Research Question

Can a compact latent representation recover the geometry of a neural population that mixes orientation and contrast selectivity while remaining robust to noise and partial neuron dropout?

## Results

Metrics below come from the default run under [`outputs/default_run`](/home/aimldl/neural_manifold_study/outputs/default_run).

| Metric | Autoencoder | PCA |
| --- | ---: | ---: |
| Reconstruction MSE | 0.0284 | 0.2802 |
| Orientation similarity | 0.9984 | 0.9991 |
| Orientation MAE (deg) | 2.69 | 1.84 |
| Contrast `R^2` | 0.9729 | 0.9695 |
| Trustworthiness | 0.9991 | 0.9971 |
| Pairwise-distance correlation | 0.9254 | 0.9401 |

## Key Findings

- The autoencoder delivers a clear reconstruction advantage, reducing held-out reconstruction error by roughly an order of magnitude relative to PCA.
- Both models recover orientation and contrast structure very strongly, which makes the comparison more interesting than a trivial nonlinear win.
- PCA remains highly competitive on orientation error and global distance preservation, while the autoencoder is stronger on reconstruction quality and trustworthiness.
- The full figure set tells a coherent story: what the population encodes, how the manifold is organized, where the two models differ, and how decoded state evolves over time.

## Visual Overview

### Main Figures

![Recovered latent manifold](outputs/default_run/figures/figure_03_latent_manifold.png)

![Autoencoder vs PCA manifold comparison](outputs/default_run/figures/figure_05_ae_vs_pca_manifold.png)

![Reconstruction residual analysis](outputs/default_run/figures/figure_06_reconstruction_residuals.png)

### Animated Views

<p align="center">
  <img src="outputs/default_run/animations/manifold_trajectory.gif" alt="Trajectory through recovered manifold" width="48%" />
  <img src="outputs/default_run/animations/latent_traversal.gif" alt="Latent traversal with decoded population profiles" width="48%" />
</p>

## Study Design

The study is organized as a reproducible analysis pipeline rather than a notebook-only exploration. Population responses are mapped into a low-dimensional latent space, decoded back into sensory variables, stress-tested under neuron dropout, and compared against a linear baseline using the same evaluation suite.

Core evaluation axes:

- reconstruction fidelity on held-out responses
- orientation recovery quality
- contrast decoding quality
- local manifold trustworthiness
- global distance preservation
- robustness to partial loss of observed neurons

## Figure Guide

1. [`figure_01_tuning_panel.png`](/home/aimldl/neural_manifold_study/outputs/default_run/figures/figure_01_tuning_panel.png)  
   Mixed-selectivity tuning structure across example neurons.
2. [`figure_02_population_heatmap.png`](/home/aimldl/neural_manifold_study/outputs/default_run/figures/figure_02_population_heatmap.png)  
   Held-out population activity over time with orientation and contrast traces.
3. [`figure_03_latent_manifold.png`](/home/aimldl/neural_manifold_study/outputs/default_run/figures/figure_03_latent_manifold.png)  
   Recovered latent geometry of population responses.
4. [`figure_04_robustness_metrics.png`](/home/aimldl/neural_manifold_study/outputs/default_run/figures/figure_04_robustness_metrics.png)  
   Training curve, held-out metrics, and neuron-dropout robustness.
5. [`figure_05_ae_vs_pca_manifold.png`](/home/aimldl/neural_manifold_study/outputs/default_run/figures/figure_05_ae_vs_pca_manifold.png)  
   Side-by-side comparison of nonlinear and linear latent spaces.
6. [`figure_06_reconstruction_residuals.png`](/home/aimldl/neural_manifold_study/outputs/default_run/figures/figure_06_reconstruction_residuals.png)  
   Reconstruction quality and residual structure on a held-out sequence.
7. [`manifold_trajectory.mp4`](/home/aimldl/neural_manifold_study/outputs/default_run/animations/manifold_trajectory.mp4) / [`manifold_trajectory.gif`](/home/aimldl/neural_manifold_study/outputs/default_run/animations/manifold_trajectory.gif)  
   Time-resolved evolution of a held-out trajectory through the recovered manifold.
8. [`latent_traversal.mp4`](/home/aimldl/neural_manifold_study/outputs/default_run/animations/latent_traversal.mp4) / [`latent_traversal.gif`](/home/aimldl/neural_manifold_study/outputs/default_run/animations/latent_traversal.gif)  
   Controlled traversal through latent space with decoded population profiles.

## Repository Layout

```text
configs/                  experiment configuration files
scripts/run_end_to_end.py command-line entrypoint
neural_manifold/          package with data, models, metrics, and plotting code
outputs/                  figures, animations, metrics, and saved artifacts
```

## Quick Start

Install dependencies:

```bash
cd /home/aimldl/neural_manifold_study
python3 -m pip install -r requirements.txt
```

Run the main configuration:

```bash
cd /home/aimldl/neural_manifold_study
python3 scripts/run_end_to_end.py --config configs/default.yaml
```

Run the faster smoke configuration:

```bash
cd /home/aimldl/neural_manifold_study
python3 scripts/run_end_to_end.py --config configs/smoke.yaml --output outputs/smoke_run
```

## Output Package

Each run writes a deterministic output directory with:

- `artifacts/dataset.npz`
- `artifacts/autoencoder_model.npz`
- `artifacts/pca_model.npz`
- `artifacts/latent_representations.npz`
- `artifacts/latent_traversal.npz`
- `metrics/summary_metrics.csv`
- `metrics/summary_metrics.json`
- `figures/figure_01_tuning_panel.png`
- `figures/figure_02_population_heatmap.png`
- `figures/figure_03_latent_manifold.png`
- `figures/figure_04_robustness_metrics.png`
- `figures/figure_05_ae_vs_pca_manifold.png`
- `figures/figure_06_reconstruction_residuals.png`
- `animations/manifold_trajectory.mp4`
- `animations/manifold_trajectory.gif`
- `animations/latent_traversal.mp4`
- `animations/latent_traversal.gif`

## Recommended Reading Order

Start with the results table, then inspect the latent manifold and AE-vs-PCA comparison, and finish with the two animations. That ordering gives the cleanest narrative from representation quality to model tradeoff to dynamical interpretation.
