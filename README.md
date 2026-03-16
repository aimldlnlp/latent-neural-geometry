<h1 align="center">From Spikes to Manifold</h1>

<p align="center">
  Tracing how neural population activity folds into low-dimensional geometry
</p>

<p align="center">
  Autoencoder vs PCA • Latent state recovery • Scientific visualization • End-to-end pipeline
</p>

<p align="center">
  <img src="docs/readme_assets/manifold_trajectory.gif" alt="Trajectory through recovered manifold" width="78%" />
</p>

<p align="center">
  <img src="docs/readme_assets/recovered_latent_manifold.png" alt="Recovered latent manifold" width="84%" />
</p>

## Opening Frame

This project follows a simple idea with a strong visual payoff: take mixed-selectivity neural population activity, recover its hidden state space, and make that state space readable. The repository compares a NumPy autoencoder against a PCA baseline, quantifies what each model preserves, and packages the result as a small research artifact with clean figures, short animations, and reproducible outputs.

## Signal

| Metric | Autoencoder | PCA |
| --- | ---: | ---: |
| Reconstruction MSE | 0.0284 | 0.2802 |
| Orientation similarity | 0.9984 | 0.9991 |
| Orientation MAE (deg) | 2.69 | 1.84 |
| Contrast `R^2` | 0.9729 | 0.9695 |
| Trustworthiness | 0.9991 | 0.9971 |

The central contrast is sharp and interpretable: the autoencoder is dramatically better at reconstruction, while PCA remains highly competitive on orientation recovery and global geometry.

## Frames

<p align="center">
  <img src="docs/readme_assets/ae_vs_pca_manifold.png" alt="Autoencoder versus PCA manifold comparison" width="49%" />
  <img src="docs/readme_assets/reconstruction_residuals.png" alt="Reconstruction residual analysis" width="49%" />
</p>

<p align="center">
  <img src="docs/readme_assets/manifold_trajectory.gif" alt="Trajectory through recovered manifold" width="49%" />
  <img src="docs/readme_assets/latent_traversal.gif" alt="Latent traversal with decoded population profiles" width="49%" />
</p>

## Why It Lands

- It combines computational neuroscience, representation learning, and quantitative evaluation in one focused repo.
- It uses a strong linear baseline instead of relying on a single-model success story.
- It supports the visual story with explicit metrics, saved artifacts, and end-to-end reproducibility.

## Quick Start

```bash
cd /home/aimldl/neural_manifold_study
python3 -m pip install -r requirements.txt
python3 scripts/run_end_to_end.py --config configs/default.yaml
```

Faster smoke run:

```bash
cd /home/aimldl/neural_manifold_study
python3 scripts/run_end_to_end.py --config configs/smoke.yaml --output outputs/smoke_run
```

## What You Get

- Scientific figures in `PNG`
- Preview animations in `GIF`
- Full-resolution animations in `MP4`
- Saved latent and model artifacts in `NPZ`
- Metric summaries in `CSV` and `JSON`

## Repository Layout

```text
configs/                  experiment configuration files
scripts/run_end_to_end.py command-line entrypoint
neural_manifold/          package with data, models, metrics, and plotting code
docs/readme_assets/       tracked showcase assets for GitHub preview
outputs/                  local figures, animations, metrics, and saved artifacts
```

<details>
<summary>More Details</summary>

### Core Question

Can a compact latent representation recover the geometry of a neural population that mixes orientation and contrast selectivity while remaining robust to noise and partial neuron dropout?

### Evaluation

- reconstruction MSE on held-out responses
- orientation similarity and orientation error in degrees
- contrast decoding `R^2`
- manifold trustworthiness
- pairwise-distance correlation
- robustness under neuron dropout

### Selected Assets

- [`recovered_latent_manifold.png`](docs/readme_assets/recovered_latent_manifold.png)
- [`ae_vs_pca_manifold.png`](docs/readme_assets/ae_vs_pca_manifold.png)
- [`reconstruction_residuals.png`](docs/readme_assets/reconstruction_residuals.png)
- [`manifold_trajectory.gif`](docs/readme_assets/manifold_trajectory.gif)
- [`latent_traversal.gif`](docs/readme_assets/latent_traversal.gif)

### Output Package

Each run writes a deterministic output directory under `outputs/<run_name>/` with:

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

</details>
