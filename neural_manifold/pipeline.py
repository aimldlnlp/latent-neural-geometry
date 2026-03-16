from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np

from .config import load_config
from .data import RecordingDataset, build_dataset
from .metrics import evaluate_dropout_curve, evaluate_latent_representation, make_polynomial_features
from .models import NumpyAutoencoder, PCABaseline
from .utils import ensure_dir, save_flat_metrics, save_json, save_yaml, standardize_train_test
from .visualization import (
    apply_paper_style,
    save_latent_manifold,
    save_latent_traversal_animation,
    save_manifold_animation,
    save_model_comparison_figure,
    save_population_heatmap,
    save_reconstruction_residual_figure,
    save_robustness_figure,
    save_tuning_panel,
)


def _prepare_output_dirs(base_dir: str | Path) -> Dict[str, Path]:
    root = ensure_dir(base_dir)
    return {
        "root": root,
        "artifacts": ensure_dir(root / "artifacts"),
        "metrics": ensure_dir(root / "metrics"),
        "figures": ensure_dir(root / "figures"),
        "animations": ensure_dir(root / "animations"),
        "configs": ensure_dir(root / "configs"),
    }


def _save_dataset_artifact(dataset: RecordingDataset, path: Path) -> None:
    np.savez_compressed(
        path,
        train_responses=dataset.train_responses,
        test_responses=dataset.test_responses,
        train_theta=dataset.train_theta,
        test_theta=dataset.test_theta,
        train_contrast=dataset.train_contrast,
        test_contrast=dataset.test_contrast,
        train_sequence_ids=dataset.train_sequence_ids,
        test_sequence_ids=dataset.test_sequence_ids,
        train_time_indices=dataset.train_time_indices,
        test_time_indices=dataset.test_time_indices,
        test_sequence_tensor=dataset.test_sequence_tensor,
        test_theta_tensor=dataset.test_theta_tensor,
        test_contrast_tensor=dataset.test_contrast_tensor,
        time_axis=dataset.time_axis,
        **{f"neuron_{key}": value for key, value in dataset.neuron_params.items()},
    )


def _build_latent_traversal(
    latent_cloud: np.ndarray,
    theta_cloud: np.ndarray,
    contrast_cloud: np.ndarray,
    num_frames: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    phase = np.linspace(0.0, 2.0 * np.pi, num_frames, endpoint=False)
    theta_target = np.mod(phase + 0.22 * np.sin(3.0 * phase), 2.0 * np.pi)
    contrast_target = np.clip(0.56 + 0.28 * np.sin(phase + 0.55) + 0.08 * np.sin(2.0 * phase + 1.2), 0.08, 0.98)

    target_features = np.stack([np.cos(theta_target), np.sin(theta_target), 1.25 * contrast_target], axis=1)
    cloud_features = np.stack([np.cos(theta_cloud), np.sin(theta_cloud), 1.25 * contrast_cloud], axis=1)

    traversal_latent = np.zeros((num_frames, latent_cloud.shape[1]), dtype=np.float32)
    for index, target in enumerate(target_features):
        distances = np.linalg.norm(cloud_features - target[None, :], axis=1)
        nearest = np.argsort(distances)[:16]
        weights = np.exp(-(distances[nearest] ** 2) / 0.18)
        weights = weights / np.maximum(weights.sum(), 1e-8)
        traversal_latent[index] = np.sum(latent_cloud[nearest] * weights[:, None], axis=0)

    return traversal_latent, theta_target.astype(np.float32), contrast_target.astype(np.float32)


def run_pipeline(config_path: str | Path, output_dir: str | Path | None = None) -> Dict[str, Any]:
    config = load_config(config_path)
    project_root = Path(config_path).resolve().parents[1]
    if output_dir is None:
        base_output = project_root / "outputs" / config["output"]["run_name"]
    else:
        base_output = Path(output_dir)
    output_dirs = _prepare_output_dirs(base_output)
    save_yaml(output_dirs["configs"] / "resolved_config.yaml", config)

    apply_paper_style()
    dataset = build_dataset(config)
    _save_dataset_artifact(dataset, output_dirs["artifacts"] / "dataset.npz")

    train_x, test_x, mean, std = standardize_train_test(dataset.train_responses, dataset.test_responses)
    np.savez_compressed(output_dirs["artifacts"] / "normalization_stats.npz", mean=mean.astype(np.float32), std=std.astype(np.float32))

    model_cfg = config["model"]
    eval_cfg = config["eval"]
    autoencoder = NumpyAutoencoder(
        input_dim=train_x.shape[1],
        hidden_dim=int(model_cfg["hidden_dim"]),
        latent_dim=int(model_cfg["latent_dim"]),
        seed=int(config["seed"]),
    )
    history = autoencoder.fit(
        train_x,
        epochs=int(model_cfg["epochs"]),
        batch_size=int(model_cfg["batch_size"]),
        learning_rate=float(model_cfg["learning_rate"]),
        weight_decay=float(model_cfg["weight_decay"]),
        validation_fraction=float(model_cfg["validation_fraction"]),
        early_stopping_patience=int(model_cfg["early_stopping_patience"]),
        input_noise_std=float(model_cfg["input_noise_std"]),
    )

    pca = PCABaseline(latent_dim=int(model_cfg["latent_dim"])).fit(train_x)

    train_latent_ae = autoencoder.encode(train_x)
    test_latent_ae = autoencoder.encode(test_x)
    train_recon_ae = autoencoder.reconstruct(train_x)
    test_recon_ae = autoencoder.reconstruct(test_x)

    train_latent_pca = pca.transform(train_x).astype(np.float32)
    test_latent_pca = pca.transform(test_x).astype(np.float32)
    train_recon_pca = pca.inverse_transform(train_latent_pca).astype(np.float32)
    test_recon_pca = pca.inverse_transform(test_latent_pca).astype(np.float32)

    metrics_ae, preds_ae = evaluate_latent_representation(
        train_latent_ae,
        test_latent_ae,
        dataset.train_theta,
        dataset.test_theta,
        dataset.train_contrast,
        dataset.test_contrast,
        test_x,
        test_recon_ae,
        ridge_penalty=float(eval_cfg["ridge_penalty"]),
        trustworthiness_k=int(eval_cfg["trustworthiness_k"]),
        trustworthiness_subset=int(eval_cfg["trustworthiness_subset"]),
        seed=int(config["seed"]),
    )
    metrics_pca, preds_pca = evaluate_latent_representation(
        train_latent_pca,
        test_latent_pca,
        dataset.train_theta,
        dataset.test_theta,
        dataset.train_contrast,
        dataset.test_contrast,
        test_x,
        test_recon_pca,
        ridge_penalty=float(eval_cfg["ridge_penalty"]),
        trustworthiness_k=int(eval_cfg["trustworthiness_k"]),
        trustworthiness_subset=int(eval_cfg["trustworthiness_subset"]),
        seed=int(config["seed"]) + 1,
    )

    dropout_fractions = [float(value) for value in eval_cfg["dropout_fractions"]]
    dropout_curve_ae = evaluate_dropout_curve(
        train_latent_ae,
        dataset.train_theta,
        dataset.train_contrast,
        test_x,
        dataset.test_theta,
        dataset.test_contrast,
        encoder_fn=lambda batch: autoencoder.encode(batch),
        dropout_fractions=dropout_fractions,
        ridge_penalty=float(eval_cfg["ridge_penalty"]),
        seed=int(config["seed"]),
    )
    dropout_curve_pca = evaluate_dropout_curve(
        train_latent_pca,
        dataset.train_theta,
        dataset.train_contrast,
        test_x,
        dataset.test_theta,
        dataset.test_contrast,
        encoder_fn=lambda batch: pca.transform(batch).astype(np.float32),
        dropout_fractions=dropout_fractions,
        ridge_penalty=float(eval_cfg["ridge_penalty"]),
        seed=int(config["seed"]) + 3,
    )

    summary_metrics = {"autoencoder": metrics_ae, "pca": metrics_pca}
    save_flat_metrics(
        output_dirs["metrics"] / "summary_metrics.csv",
        output_dirs["metrics"] / "summary_metrics.json",
        summary_metrics,
    )

    np.savez_compressed(
        output_dirs["artifacts"] / "autoencoder_model.npz",
        **{name: value.astype(np.float32) for name, value in autoencoder.params.items()},
        train_loss=np.asarray(history.train_loss, dtype=np.float32),
        val_loss=np.asarray(history.val_loss, dtype=np.float32),
    )
    np.savez_compressed(
        output_dirs["artifacts"] / "pca_model.npz",
        mean=pca.mean_.astype(np.float32),
        components=pca.components_.astype(np.float32),
        explained_variance_ratio=pca.explained_variance_ratio_.astype(np.float32),
    )
    np.savez_compressed(
        output_dirs["artifacts"] / "latent_representations.npz",
        train_latent_ae=train_latent_ae,
        test_latent_ae=test_latent_ae,
        train_latent_pca=train_latent_pca,
        test_latent_pca=test_latent_pca,
        theta_pred_ae=preds_ae["theta_pred"],
        contrast_pred_ae=preds_ae["contrast_pred"],
        theta_pred_pca=preds_pca["theta_pred"],
        contrast_pred_pca=preds_pca["contrast_pred"],
    )
    np.savez_compressed(
        output_dirs["metrics"] / "dropout_curves.npz",
        ae_dropout_fraction=dropout_curve_ae["dropout_fraction"],
        ae_orientation_circular_corr=dropout_curve_ae["orientation_circular_corr"],
        ae_orientation_mae_deg=dropout_curve_ae["orientation_mae_deg"],
        ae_contrast_r2=dropout_curve_ae["contrast_r2"],
        pca_dropout_fraction=dropout_curve_pca["dropout_fraction"],
        pca_orientation_circular_corr=dropout_curve_pca["orientation_circular_corr"],
        pca_orientation_mae_deg=dropout_curve_pca["orientation_mae_deg"],
        pca_contrast_r2=dropout_curve_pca["contrast_r2"],
    )

    dpi = int(config["visuals"]["dpi"])
    save_tuning_panel(output_dirs["figures"] / "figure_01_tuning_panel.png", dataset.neuron_params, dpi=dpi)
    sequence_index = int(config["visuals"]["animation_sequence_index"])
    save_population_heatmap(
        output_dirs["figures"] / "figure_02_population_heatmap.png",
        dataset.test_sequence_tensor[sequence_index],
        dataset.test_theta_tensor[sequence_index],
        dataset.test_contrast_tensor[sequence_index],
        dataset.time_axis,
        dataset.neuron_params,
        dpi=dpi,
    )
    save_latent_manifold(
        output_dirs["figures"] / "figure_03_latent_manifold.png",
        test_latent_ae,
        dataset.test_theta,
        dataset.test_contrast,
        dataset.test_sequence_ids,
        dataset.test_time_indices,
        dpi=dpi,
    )
    save_robustness_figure(
        output_dirs["figures"] / "figure_04_robustness_metrics.png",
        history={
            "epoch": np.arange(1, len(history.train_loss) + 1),
            "train_loss": np.asarray(history.train_loss),
            "val_loss": np.asarray(history.val_loss),
        },
        clean_metrics=summary_metrics,
        dropout_curves={"autoencoder": dropout_curve_ae, "pca": dropout_curve_pca},
        dpi=dpi,
    )
    save_model_comparison_figure(
        output_dirs["figures"] / "figure_05_ae_vs_pca_manifold.png",
        latent_ae=test_latent_ae,
        latent_pca=test_latent_pca,
        theta=dataset.test_theta,
        contrast=dataset.test_contrast,
        sequence_ids=dataset.test_sequence_ids,
        time_indices=dataset.test_time_indices,
        dpi=dpi,
    )

    seq_len = dataset.test_sequence_tensor.shape[1]
    seq_slice = slice(sequence_index * seq_len, (sequence_index + 1) * seq_len)
    save_reconstruction_residual_figure(
        output_dirs["figures"] / "figure_06_reconstruction_residuals.png",
        observed=test_x[seq_slice],
        reconstructed_ae=test_recon_ae[seq_slice],
        reconstructed_pca=test_recon_pca[seq_slice],
        neuron_params=dataset.neuron_params,
        time_axis=dataset.time_axis,
        dpi=dpi,
    )
    save_manifold_animation(
        output_dirs["animations"] / "manifold_trajectory.mp4",
        output_dirs["animations"] / "manifold_trajectory.gif",
        sequence_responses=dataset.test_sequence_tensor[sequence_index],
        sequence_theta=dataset.test_theta_tensor[sequence_index],
        sequence_contrast=dataset.test_contrast_tensor[sequence_index],
        sequence_latent=test_latent_ae[seq_slice],
        theta_pred=preds_ae["theta_pred"][seq_slice],
        contrast_pred=preds_ae["contrast_pred"][seq_slice],
        time_axis=dataset.time_axis,
        fps=int(config["visuals"]["animation_fps"]),
    )

    traversal_latent, traversal_theta_target, traversal_contrast_target = _build_latent_traversal(
        latent_cloud=test_latent_ae,
        theta_cloud=dataset.test_theta,
        contrast_cloud=dataset.test_contrast,
        num_frames=seq_len,
    )
    traversal_features = make_polynomial_features(traversal_latent)
    orientation_weights = preds_ae["orientation_decoder_weights"].astype(np.float32)
    orientation_bias = preds_ae["orientation_decoder_bias"].astype(np.float32)
    contrast_weights = preds_ae["contrast_decoder_weights"].astype(np.float32)
    contrast_bias = preds_ae["contrast_decoder_bias"].astype(np.float32)
    traversal_orientation_components = traversal_features @ orientation_weights + orientation_bias
    traversal_theta_pred = np.mod(np.arctan2(traversal_orientation_components[:, 0], traversal_orientation_components[:, 1]), 2.0 * np.pi)
    traversal_contrast_pred = np.clip((traversal_features @ contrast_weights + contrast_bias).reshape(-1), 0.0, 1.0)
    traversal_response = autoencoder.decode(traversal_latent)
    np.savez_compressed(
        output_dirs["artifacts"] / "latent_traversal.npz",
        traversal_latent=traversal_latent.astype(np.float32),
        traversal_theta_target=traversal_theta_target.astype(np.float32),
        traversal_theta_pred=traversal_theta_pred.astype(np.float32),
        traversal_contrast_target=traversal_contrast_target.astype(np.float32),
        traversal_contrast_pred=traversal_contrast_pred.astype(np.float32),
        traversal_response=traversal_response.astype(np.float32),
    )
    save_latent_traversal_animation(
        output_dirs["animations"] / "latent_traversal.mp4",
        output_dirs["animations"] / "latent_traversal.gif",
        latent_cloud=test_latent_ae,
        theta_cloud=dataset.test_theta,
        contrast_cloud=dataset.test_contrast,
        traversal_latent=traversal_latent,
        traversal_response=traversal_response,
        traversal_theta_target=traversal_theta_target,
        traversal_theta_pred=traversal_theta_pred,
        traversal_contrast_target=traversal_contrast_target,
        traversal_contrast_pred=traversal_contrast_pred,
        neuron_params=dataset.neuron_params,
        fps=int(config["visuals"]["animation_fps"]),
    )

    summary = {
        "output_dir": str(output_dirs["root"]),
        "figures": sorted(str(path.name) for path in output_dirs["figures"].glob("*.png")),
        "animations": sorted(str(path.name) for path in output_dirs["animations"].iterdir()),
        "metrics": summary_metrics,
    }
    save_json(output_dirs["root"] / "run_summary.json", summary)
    return summary
