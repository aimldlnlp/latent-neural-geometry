<h1 align="center">From Spikes to Manifold</h1>

<p align="center">
  <strong>Recovering low-dimensional neural geometry from mixed-selectivity population activity</strong>
</p>

<p align="center">
  Autoencoder vs PCA • Latent state recovery • Scientific visualization • End-to-end research pipeline
</p>

<p align="center">
  <img src="docs/readme_assets/manifold_trajectory.gif" alt="Trajectory through recovered manifold" width="82%" />
</p>

<p align="center">
  <img src="docs/readme_assets/recovered_latent_manifold.png" alt="Recovered latent manifold" width="82%" />
</p>

<table align="center">
  <tr>
    <td align="center" width="33%">
      <strong>10x lower</strong><br/>
      reconstruction error than PCA
    </td>
    <td align="center" width="33%">
      <strong>0.973</strong><br/>
      contrast decoding <code>R^2</code>
    </td>
    <td align="center" width="33%">
      <strong>0.999</strong><br/>
      manifold trustworthiness
    </td>
  </tr>
</table>

## What This Project Is

A compact computational neuroscience showcase on latent manifold recovery from neural population responses. The project compares a NumPy autoencoder against a PCA baseline, quantifies what each model preserves, and packages the results as clean paper-style figures, animations, and reproducible outputs.

## Why It Stands Out

- Combines computational neuroscience, representation learning, and quantitative evaluation in one repo.
- Uses a strong linear baseline instead of presenting a single-model success story.
- Produces outputs that are both visually strong and metrically defensible.

## Result Snapshot

| Metric | Autoencoder | PCA |
| --- | ---: | ---: |
| Reconstruction MSE | 0.0284 | 0.2802 |
| Orientation similarity | 0.9984 | 0.9991 |
| Orientation MAE (deg) | 2.69 | 1.84 |
| Contrast `R^2` | 0.9729 | 0.9695 |
| Trustworthiness | 0.9991 | 0.9971 |

The core story is clear: the autoencoder wins decisively on reconstruction while PCA remains highly competitive on orientation recovery and global geometry.

## Visual Showcase

<p align="center">
  <img src="docs/readme_assets/ae_vs_pca_manifold.png" alt="Autoencoder versus PCA manifold comparison" width="49%" />
  <img src="docs/readme_assets/reconstruction_residuals.png" alt="Reconstruction residual analysis" width="49%" />
</p>

<p align="center">
  <img src="docs/readme_assets/manifold_trajectory.gif" alt="Trajectory through recovered manifold" width="49%" />
  <img src="docs/readme_assets/latent_traversal.gif" alt="Latent traversal with decoded population profiles" width="49%" />
</p>

## What The Visuals Show

- `Recovered latent manifold`: the population collapses into a structured low-dimensional state space.
- `AE vs PCA comparison`: nonlinear and linear embeddings preserve different aspects of the same geometry.
- `Residual analysis`: the reconstruction gap is visually obvious, not just numerically obvious.
- `Trajectory and traversal animations`: the latent state evolves coherently and remains decodable over time.

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

## Repo Layout

```text
configs/                  experiment configuration files
scripts/run_end_to_end.py command-line entrypoint
neural_manifold/          package with data, models, metrics, and plotting code
docs/readme_assets/       tracked showcase assets for GitHub preview
outputs/                  local figures, animations, metrics, and saved artifacts
```

## Deliverables

- Scientific figures in `PNG`
- Preview animations in `GIF`
- Full-resolution animations in `MP4`
- Saved latent and model artifacts in `NPZ`
- Metric summaries in `CSV` and `JSON`

## Selected Assets

- [`recovered_latent_manifold.png`](docs/readme_assets/recovered_latent_manifold.png)
- [`ae_vs_pca_manifold.png`](docs/readme_assets/ae_vs_pca_manifold.png)
- [`reconstruction_residuals.png`](docs/readme_assets/reconstruction_residuals.png)
- [`manifold_trajectory.gif`](docs/readme_assets/manifold_trajectory.gif)
- [`latent_traversal.gif`](docs/readme_assets/latent_traversal.gif)

<details>
<summary>Research Details</summary>

### Core Question

Can a compact latent representation recover the geometry of a neural population that mixes orientation and contrast selectivity while remaining robust to noise and partial neuron dropout?

### Evaluation

- reconstruction MSE on held-out responses
- orientation similarity and orientation error in degrees
- contrast decoding `R^2`
- manifold trustworthiness
- pairwise-distance correlation
- robustness under neuron dropout

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
