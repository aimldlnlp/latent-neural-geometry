from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter


def apply_paper_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.grid": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
        }
    )


def save_tuning_panel(path: str | Path, neuron_params: Dict[str, np.ndarray], dpi: int) -> None:
    angles = np.linspace(0.0, 2.0 * np.pi, 181)
    contrast_levels = [0.2, 0.5, 0.85]
    contrast_labels = ["Low contrast", "Medium contrast", "High contrast"]
    line_colors = ["#355c7d", "#6c5b7b", "#c06c84"]

    preferred = neuron_params["preferred_angle"]
    sort_idx = np.argsort(preferred)
    chosen = sort_idx[np.linspace(0, len(sort_idx) - 1, 6, dtype=int)]

    fig, axes = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
    for ax, neuron_idx in zip(axes.flat, chosen):
        pref_angle = float(neuron_params["preferred_angle"][neuron_idx])
        delta = angles - pref_angle
        base_tuning = np.exp(neuron_params["kappa"][neuron_idx] * np.cos(delta) - neuron_params["kappa"][neuron_idx])
        harmonic = 0.5 * (1.0 + np.cos(2.0 * delta))
        for contrast, label, color in zip(contrast_levels, contrast_labels, line_colors):
            contrast_drive = 0.35 + contrast * neuron_params["contrast_gain"][neuron_idx]
            contrast_offset = contrast - neuron_params["contrast_preference"][neuron_idx]
            response = np.log1p(
                np.exp(
                    neuron_params["baseline"][neuron_idx]
                    + neuron_params["amplitude"][neuron_idx] * base_tuning * contrast_drive
                    + neuron_params["contrast_only_gain"][neuron_idx] * (contrast - 0.5)
                    + 0.6 * neuron_params["contrast_only_gain"][neuron_idx] * (contrast_offset ** 2)
                    + neuron_params["cross_term"][neuron_idx] * contrast_offset * np.sin(delta)
                    + neuron_params["second_harmonic"][neuron_idx] * harmonic * (0.25 + contrast)
                )
            )
            ax.plot(np.degrees(angles), response, color=color, lw=2.0, label=label)
        ax.set_title(f"Neuron {neuron_idx + 1}")
        ax.set_xlabel("Orientation (deg)")
        ax.set_ylabel("Expected response")
        ax.set_xlim(0, 360)
        ax.set_xticks([0, 90, 180, 270, 360])
    axes[0, 0].legend(frameon=False, loc="upper right")
    fig.suptitle("Mixed-Selectivity Tuning Curves", fontsize=14)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_population_heatmap(
    path: str | Path,
    sequence_responses: np.ndarray,
    sequence_theta: np.ndarray,
    sequence_contrast: np.ndarray,
    time_axis: np.ndarray,
    neuron_params: Dict[str, np.ndarray],
    dpi: int,
) -> None:
    order = np.argsort(neuron_params["preferred_angle"])
    sorted_responses = sequence_responses[:, order].T
    theta_deg = np.degrees(sequence_theta)

    fig = plt.figure(figsize=(12, 7), constrained_layout=True)
    grid = fig.add_gridspec(3, 1, height_ratios=[3.2, 1.0, 1.0])
    ax0 = fig.add_subplot(grid[0])
    ax1 = fig.add_subplot(grid[1], sharex=ax0)
    ax2 = fig.add_subplot(grid[2], sharex=ax0)

    im = ax0.imshow(
        sorted_responses,
        aspect="auto",
        cmap="magma",
        origin="lower",
        extent=[time_axis[0], time_axis[-1], 1, sorted_responses.shape[0]],
    )
    ax0.set_title("Population Response Heatmap for a Held-Out Sequence")
    ax0.set_ylabel("Neuron index (sorted by preferred angle)")
    cbar = fig.colorbar(im, ax=ax0, fraction=0.02, pad=0.01)
    cbar.set_label("Response amplitude")

    ax1.plot(time_axis, theta_deg, color="#355c7d", lw=2.0)
    ax1.set_ylabel("Orientation (deg)")
    ax1.set_ylim(0, 360)

    ax2.plot(time_axis, sequence_contrast, color="#c06c84", lw=2.0)
    ax2.set_ylabel("Contrast")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylim(0, 1.05)

    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_latent_manifold(
    path: str | Path,
    latent: np.ndarray,
    theta: np.ndarray,
    contrast: np.ndarray,
    sequence_ids: np.ndarray,
    time_indices: np.ndarray,
    dpi: int,
) -> None:
    fig = plt.figure(figsize=(14, 4.8), constrained_layout=True)
    ax0 = fig.add_subplot(1, 3, 1)
    ax1 = fig.add_subplot(1, 3, 2)
    ax2 = fig.add_subplot(1, 3, 3)

    scatter0 = ax0.scatter(latent[:, 0], latent[:, 1], c=theta, cmap="twilight", s=10, alpha=0.8, linewidths=0.0)
    ax0.set_title("Latent plane colored by orientation")
    ax0.set_xlabel("Latent dimension 1")
    ax0.set_ylabel("Latent dimension 2")
    cbar0 = fig.colorbar(scatter0, ax=ax0, fraction=0.046, pad=0.02)
    cbar0.set_label("Orientation (rad)")

    scatter1 = ax1.scatter(latent[:, 0], latent[:, 2], c=contrast, cmap="viridis", s=10, alpha=0.8, linewidths=0.0)
    ax1.set_title("Latent plane colored by contrast")
    ax1.set_xlabel("Latent dimension 1")
    ax1.set_ylabel("Latent dimension 3")
    cbar1 = fig.colorbar(scatter1, ax=ax1, fraction=0.046, pad=0.02)
    cbar1.set_label("Contrast")

    unique_sequences = np.unique(sequence_ids)
    chosen = unique_sequences[:4]
    colors = ["#355c7d", "#6c5b7b", "#c06c84", "#f67280"]
    for color, seq in zip(colors, chosen):
        mask = sequence_ids == seq
        ordered = np.argsort(time_indices[mask])
        ax2.plot(latent[mask][ordered, 0], latent[mask][ordered, 1], lw=2.0, color=color, label=f"Sequence {seq + 1}")
    ax2.set_title("Held-out trajectories in latent space")
    ax2.set_xlabel("Latent dimension 1")
    ax2.set_ylabel("Latent dimension 2")
    ax2.legend(frameon=False, loc="best")

    fig.suptitle("Recovered Latent Geometry of Population Responses", fontsize=14)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_robustness_figure(
    path: str | Path,
    history: Dict[str, np.ndarray],
    clean_metrics: Dict[str, Dict[str, float]],
    dropout_curves: Dict[str, Dict[str, np.ndarray]],
    dpi: int,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)

    axes[0, 0].plot(history["epoch"], history["train_loss"], color="#355c7d", lw=2.0, label="Train")
    axes[0, 0].plot(history["epoch"], history["val_loss"], color="#c06c84", lw=2.0, label="Validation")
    axes[0, 0].set_title("Autoencoder training curve")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].legend(frameon=False)

    labels = ["Reconstruction MSE", "Orientation similarity", "Contrast $R^2$", "Trustworthiness"]
    ae_values = [
        clean_metrics["autoencoder"]["reconstruction_mse"],
        clean_metrics["autoencoder"]["orientation_circular_corr"],
        clean_metrics["autoencoder"]["contrast_r2"],
        clean_metrics["autoencoder"]["trustworthiness"],
    ]
    pca_values = [
        clean_metrics["pca"]["reconstruction_mse"],
        clean_metrics["pca"]["orientation_circular_corr"],
        clean_metrics["pca"]["contrast_r2"],
        clean_metrics["pca"]["trustworthiness"],
    ]
    x = np.arange(len(labels))
    width = 0.36
    axes[0, 1].bar(x - width / 2, ae_values, width, color="#355c7d", label="Autoencoder")
    axes[0, 1].bar(x + width / 2, pca_values, width, color="#c06c84", label="PCA")
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(labels, rotation=20)
    axes[0, 1].set_title("Held-out metric summary")
    axes[0, 1].legend(frameon=False)

    axes[1, 0].plot(
        dropout_curves["autoencoder"]["dropout_fraction"],
        dropout_curves["autoencoder"]["orientation_circular_corr"],
        marker="o",
        color="#355c7d",
        lw=2.0,
        label="Autoencoder",
    )
    axes[1, 0].plot(
        dropout_curves["pca"]["dropout_fraction"],
        dropout_curves["pca"]["orientation_circular_corr"],
        marker="o",
        color="#c06c84",
        lw=2.0,
        label="PCA",
    )
    axes[1, 0].set_title("Orientation decoding under neuron dropout")
    axes[1, 0].set_xlabel("Dropped neuron fraction")
    axes[1, 0].set_ylabel("Circular correlation")
    axes[1, 0].set_ylim(0.0, 1.02)
    axes[1, 0].legend(frameon=False)

    axes[1, 1].plot(
        dropout_curves["autoencoder"]["dropout_fraction"],
        dropout_curves["autoencoder"]["contrast_r2"],
        marker="o",
        color="#355c7d",
        lw=2.0,
        label="Autoencoder",
    )
    axes[1, 1].plot(
        dropout_curves["pca"]["dropout_fraction"],
        dropout_curves["pca"]["contrast_r2"],
        marker="o",
        color="#c06c84",
        lw=2.0,
        label="PCA",
    )
    axes[1, 1].set_title("Contrast decoding under neuron dropout")
    axes[1, 1].set_xlabel("Dropped neuron fraction")
    axes[1, 1].set_ylabel("$R^2$")
    axes[1, 1].set_ylim(0.0, 1.02)
    axes[1, 1].legend(frameon=False)

    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_model_comparison_figure(
    path: str | Path,
    latent_ae: np.ndarray,
    latent_pca: np.ndarray,
    theta: np.ndarray,
    contrast: np.ndarray,
    sequence_ids: np.ndarray,
    time_indices: np.ndarray,
    dpi: int,
) -> None:
    fig = plt.figure(figsize=(13, 9), constrained_layout=True)
    grid = fig.add_gridspec(2, 2)
    axes = np.asarray([[fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1])], [fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[1, 1])]])

    trajectory_sequences = np.unique(sequence_ids)[:3]
    trajectory_colors = ["#111111", "#4a4a4a", "#7a7a7a"]

    scatter_ae_theta = axes[0, 0].scatter(latent_ae[:, 0], latent_ae[:, 1], c=theta, cmap="twilight", s=10, alpha=0.78, linewidths=0.0)
    axes[0, 0].set_title("Autoencoder latent space colored by orientation")
    axes[0, 0].set_xlabel("Latent dimension 1")
    axes[0, 0].set_ylabel("Latent dimension 2")

    scatter_pca_theta = axes[0, 1].scatter(latent_pca[:, 0], latent_pca[:, 1], c=theta, cmap="twilight", s=10, alpha=0.78, linewidths=0.0)
    axes[0, 1].set_title("PCA latent space colored by orientation")
    axes[0, 1].set_xlabel("Latent dimension 1")
    axes[0, 1].set_ylabel("Latent dimension 2")

    scatter_ae_contrast = axes[1, 0].scatter(latent_ae[:, 0], latent_ae[:, 2], c=contrast, cmap="viridis", s=10, alpha=0.78, linewidths=0.0)
    axes[1, 0].set_title("Autoencoder latent space colored by contrast")
    axes[1, 0].set_xlabel("Latent dimension 1")
    axes[1, 0].set_ylabel("Latent dimension 3")

    scatter_pca_contrast = axes[1, 1].scatter(latent_pca[:, 0], latent_pca[:, 2], c=contrast, cmap="viridis", s=10, alpha=0.78, linewidths=0.0)
    axes[1, 1].set_title("PCA latent space colored by contrast")
    axes[1, 1].set_xlabel("Latent dimension 1")
    axes[1, 1].set_ylabel("Latent dimension 3")

    for seq_color, seq in zip(trajectory_colors, trajectory_sequences):
        mask = sequence_ids == seq
        ordered = np.argsort(time_indices[mask])
        axes[0, 0].plot(latent_ae[mask][ordered, 0], latent_ae[mask][ordered, 1], color=seq_color, lw=1.4, alpha=0.9)
        axes[0, 1].plot(latent_pca[mask][ordered, 0], latent_pca[mask][ordered, 1], color=seq_color, lw=1.4, alpha=0.9)
        axes[1, 0].plot(latent_ae[mask][ordered, 0], latent_ae[mask][ordered, 2], color=seq_color, lw=1.4, alpha=0.9)
        axes[1, 1].plot(latent_pca[mask][ordered, 0], latent_pca[mask][ordered, 2], color=seq_color, lw=1.4, alpha=0.9)

    cbar_theta = fig.colorbar(scatter_ae_theta, ax=[axes[0, 0], axes[0, 1]], fraction=0.026, pad=0.02)
    cbar_theta.set_label("Orientation (rad)")
    cbar_contrast = fig.colorbar(scatter_ae_contrast, ax=[axes[1, 0], axes[1, 1]], fraction=0.026, pad=0.02)
    cbar_contrast.set_label("Contrast")

    fig.suptitle("Autoencoder and PCA Recover Distinct Views of the Neural Manifold", fontsize=14)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_reconstruction_residual_figure(
    path: str | Path,
    observed: np.ndarray,
    reconstructed_ae: np.ndarray,
    reconstructed_pca: np.ndarray,
    neuron_params: Dict[str, np.ndarray],
    time_axis: np.ndarray,
    dpi: int,
) -> None:
    order = np.argsort(neuron_params["preferred_angle"])
    observed_sorted = observed[:, order].T
    ae_sorted = reconstructed_ae[:, order].T
    pca_sorted = reconstructed_pca[:, order].T
    ae_residual = np.abs(observed_sorted - ae_sorted)
    pca_residual = np.abs(observed_sorted - pca_sorted)

    response_limit = np.max(np.abs(np.concatenate([observed_sorted, ae_sorted, pca_sorted], axis=1)))
    residual_limit = np.max(np.concatenate([ae_residual, pca_residual], axis=1))
    extent = [time_axis[0], time_axis[-1], 1, observed_sorted.shape[0]]

    fig, axes = plt.subplots(1, 5, figsize=(18, 5.4), constrained_layout=True)
    panels = [
        (observed_sorted, "Observed population response", "coolwarm", -response_limit, response_limit),
        (ae_sorted, "Autoencoder reconstruction", "coolwarm", -response_limit, response_limit),
        (pca_sorted, "PCA reconstruction", "coolwarm", -response_limit, response_limit),
        (ae_residual, "Absolute residual: Autoencoder", "magma", 0.0, residual_limit),
        (pca_residual, "Absolute residual: PCA", "magma", 0.0, residual_limit),
    ]

    image_handles = []
    for ax, (panel_data, title, cmap, vmin, vmax) in zip(axes, panels):
        image = ax.imshow(panel_data, aspect="auto", origin="lower", cmap=cmap, extent=extent, vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Neuron index")
        image_handles.append(image)

    cbar_response = fig.colorbar(image_handles[0], ax=axes[:3], fraction=0.022, pad=0.015)
    cbar_response.set_label("Standardized response")
    cbar_residual = fig.colorbar(image_handles[3], ax=axes[3:], fraction=0.022, pad=0.015)
    cbar_residual.set_label("Absolute residual")

    fig.suptitle("Reconstruction Quality and Residual Structure on a Held-Out Sequence", fontsize=14)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_manifold_animation(
    mp4_path: str | Path,
    gif_path: str | Path,
    sequence_responses: np.ndarray,
    sequence_theta: np.ndarray,
    sequence_contrast: np.ndarray,
    sequence_latent: np.ndarray,
    theta_pred: np.ndarray,
    contrast_pred: np.ndarray,
    time_axis: np.ndarray,
    fps: int,
) -> None:
    fig = plt.figure(figsize=(12, 7), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, height_ratios=[2.2, 1.2])
    ax0 = fig.add_subplot(grid[:, 0])
    ax1 = fig.add_subplot(grid[0, 1])
    ax2 = fig.add_subplot(grid[1, 1])
    ax2b = ax2.twinx()

    ax0.scatter(sequence_latent[:, 0], sequence_latent[:, 1], c=sequence_theta, cmap="twilight", s=18, alpha=0.18, linewidths=0.0)
    trajectory_line, = ax0.plot([], [], color="#111111", lw=2.2)
    point_marker = ax0.scatter([], [], s=70, color="#c06c84", edgecolors="black", linewidths=0.7)
    ax0.set_title("Trajectory through recovered manifold")
    ax0.set_xlabel("Latent dimension 1")
    ax0.set_ylabel("Latent dimension 2")

    heatmap = ax1.imshow(sequence_responses.T, aspect="auto", cmap="magma", origin="lower")
    current_bar = ax1.axvline(0, color="white", lw=2.0)
    ax1.set_title("Population activity over time")
    ax1.set_xlabel("Time bin")
    ax1.set_ylabel("Neuron index")
    fig.colorbar(heatmap, ax=ax1, fraction=0.046, pad=0.03).set_label("Response amplitude")

    theta_deg = np.degrees(sequence_theta)
    theta_pred_deg = np.degrees(theta_pred)
    orientation_true_line, = ax2.plot(time_axis, theta_deg, color="#355c7d", lw=2.0, label="True orientation")
    orientation_pred_line, = ax2.plot(time_axis, theta_pred_deg, color="#f67280", lw=1.8, alpha=0.9, label="Decoded orientation")
    contrast_true_line, = ax2b.plot(time_axis, sequence_contrast, color="#6c5b7b", lw=2.0, label="True contrast")
    contrast_pred_line, = ax2b.plot(time_axis, contrast_pred, color="#f8b195", lw=1.8, alpha=0.9, label="Decoded contrast")
    ax2.set_title("Decoded sensory variables")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Orientation (deg)")
    ax2.set_ylim(0, 360)
    ax2b.set_ylabel("Contrast")
    ax2b.set_ylim(0.0, 1.05)
    handles = [orientation_true_line, orientation_pred_line, contrast_true_line, contrast_pred_line]
    labels = [handle.get_label() for handle in handles]
    ax2.legend(handles, labels, frameon=False, ncol=2, loc="upper right")
    theta_cursor = ax2.axvline(time_axis[0], color="#111111", lw=1.5, alpha=0.8)

    def update(frame: int):
        current_latent = sequence_latent[: frame + 1]
        trajectory_line.set_data(current_latent[:, 0], current_latent[:, 1])
        point_marker.set_offsets(sequence_latent[frame][None, :2])
        current_bar.set_xdata([frame, frame])
        theta_cursor.set_xdata([time_axis[frame], time_axis[frame]])
        ax0.set_title(
            "Trajectory through recovered manifold\n"
            f"Time = {time_axis[frame]:.2f} s | Orientation = {theta_deg[frame]:.1f} deg | Contrast = {sequence_contrast[frame]:.2f}"
        )
        return trajectory_line, point_marker, current_bar, theta_cursor

    animation = FuncAnimation(fig, update, frames=len(time_axis), interval=1000 / max(fps, 1), blit=False)
    mp4_writer = FFMpegWriter(fps=fps, bitrate=2400)
    animation.save(mp4_path, writer=mp4_writer)
    gif_writer = PillowWriter(fps=max(1, min(fps, 12)))
    animation.save(gif_path, writer=gif_writer)
    plt.close(fig)


def save_latent_traversal_animation(
    mp4_path: str | Path,
    gif_path: str | Path,
    latent_cloud: np.ndarray,
    theta_cloud: np.ndarray,
    contrast_cloud: np.ndarray,
    traversal_latent: np.ndarray,
    traversal_response: np.ndarray,
    traversal_theta_target: np.ndarray,
    traversal_theta_pred: np.ndarray,
    traversal_contrast_target: np.ndarray,
    traversal_contrast_pred: np.ndarray,
    neuron_params: Dict[str, np.ndarray],
    fps: int,
) -> None:
    order = np.argsort(neuron_params["preferred_angle"])
    preferred_deg = np.degrees(np.mod(neuron_params["preferred_angle"][order], 2.0 * np.pi))
    traversal_sorted = traversal_response[:, order]

    fig = plt.figure(figsize=(12.5, 7.2), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, height_ratios=[1.25, 1.0])
    ax0 = fig.add_subplot(grid[:, 0])
    ax1 = fig.add_subplot(grid[0, 1])
    ax2 = fig.add_subplot(grid[1, 1])
    ax2b = ax2.twinx()

    background = ax0.scatter(latent_cloud[:, 0], latent_cloud[:, 1], c=theta_cloud, cmap="twilight", s=11, alpha=0.17, linewidths=0.0)
    ax0.set_title("Latent traversal through the recovered manifold")
    ax0.set_xlabel("Latent dimension 1")
    ax0.set_ylabel("Latent dimension 2")
    fig.colorbar(background, ax=ax0, fraction=0.046, pad=0.02).set_label("Orientation (rad)")
    traversal_line, = ax0.plot([], [], color="#111111", lw=2.0)
    traversal_point = ax0.scatter([], [], s=88, color="#f67280", edgecolors="black", linewidths=0.7, zorder=5)

    current_profile, = ax1.plot(preferred_deg, traversal_sorted[0], color="#355c7d", lw=2.2)
    ax1.set_title("Decoded population profile")
    ax1.set_xlabel("Preferred orientation (deg)")
    ax1.set_ylabel("Standardized response")
    ax1.set_xlim(0, 360)
    ax1.set_xticks([0, 90, 180, 270, 360])
    response_limit = np.max(np.abs(traversal_sorted)) * 1.08
    ax1.set_ylim(-response_limit, response_limit)

    frame_index = np.arange(traversal_latent.shape[0])
    target_theta_deg = np.degrees(traversal_theta_target)
    pred_theta_deg = np.degrees(traversal_theta_pred)
    theta_true_line, = ax2.plot(frame_index, target_theta_deg, color="#355c7d", lw=2.0, label="Target orientation")
    theta_pred_line, = ax2.plot(frame_index, pred_theta_deg, color="#f67280", lw=1.8, label="Decoded orientation")
    contrast_true_line, = ax2b.plot(frame_index, traversal_contrast_target, color="#6c5b7b", lw=2.0, label="Target contrast")
    contrast_pred_line, = ax2b.plot(frame_index, traversal_contrast_pred, color="#f8b195", lw=1.8, label="Decoded contrast")
    ax2.set_title("Traversal targets and decoded variables")
    ax2.set_xlabel("Traversal frame")
    ax2.set_ylabel("Orientation (deg)")
    ax2.set_ylim(0, 360)
    ax2b.set_ylabel("Contrast")
    ax2b.set_ylim(0.0, 1.05)
    cursor = ax2.axvline(0, color="#111111", lw=1.5, alpha=0.8)
    handles = [theta_true_line, theta_pred_line, contrast_true_line, contrast_pred_line]
    ax2.legend(handles, [handle.get_label() for handle in handles], frameon=False, ncol=2, loc="upper right")

    def update(frame: int):
        current_path = traversal_latent[: frame + 1]
        traversal_line.set_data(current_path[:, 0], current_path[:, 1])
        traversal_point.set_offsets(traversal_latent[frame][None, :2])
        current_profile.set_ydata(traversal_sorted[frame])
        cursor.set_xdata([frame, frame])
        ax0.set_title(
            "Latent traversal through the recovered manifold\n"
            f"Frame {frame + 1} | Target orientation = {target_theta_deg[frame]:.1f} deg | Target contrast = {traversal_contrast_target[frame]:.2f}"
        )
        return traversal_line, traversal_point, current_profile, cursor

    animation = FuncAnimation(fig, update, frames=traversal_latent.shape[0], interval=1000 / max(fps, 1), blit=False)
    mp4_writer = FFMpegWriter(fps=fps, bitrate=2600)
    animation.save(mp4_path, writer=mp4_writer)
    gif_writer = PillowWriter(fps=max(1, min(fps, 12)))
    animation.save(gif_path, writer=gif_writer)
    plt.close(fig)
