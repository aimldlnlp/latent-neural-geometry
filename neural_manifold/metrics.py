from __future__ import annotations

from typing import Dict

import numpy as np
from scipy.spatial.distance import cdist, pdist


def fit_ridge_regression(x: np.ndarray, y: np.ndarray, ridge_penalty: float) -> Dict[str, np.ndarray]:
    x_aug = np.concatenate([x, np.ones((x.shape[0], 1), dtype=x.dtype)], axis=1)
    reg = np.eye(x_aug.shape[1], dtype=x.dtype) * ridge_penalty
    reg[-1, -1] = 0.0
    weights = np.linalg.solve(x_aug.T @ x_aug + reg, x_aug.T @ y)
    return {"weights": weights[:-1], "bias": weights[-1]}


def predict_ridge(x: np.ndarray, model: Dict[str, np.ndarray]) -> np.ndarray:
    return x @ model["weights"] + model["bias"]


def make_polynomial_features(x: np.ndarray) -> np.ndarray:
    interactions = []
    for idx in range(x.shape[1]):
        interactions.append(x[:, idx : idx + 1] ** 2)
    for first in range(x.shape[1]):
        for second in range(first + 1, x.shape[1]):
            interactions.append(x[:, first : first + 1] * x[:, second : second + 1])
    return np.concatenate([x] + interactions, axis=1)


def circular_mean(theta: np.ndarray) -> float:
    return float(np.arctan2(np.mean(np.sin(theta)), np.mean(np.cos(theta))))


def circular_correlation(theta_true: np.ndarray, theta_pred: np.ndarray) -> float:
    return float(np.mean(np.cos(theta_pred - theta_true)))


def circular_mae_degrees(theta_true: np.ndarray, theta_pred: np.ndarray) -> float:
    delta = np.angle(np.exp(1j * (theta_pred - theta_true)))
    return float(np.mean(np.abs(delta)) * 180.0 / np.pi)


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    numerator = np.sum((y_true - y_pred) ** 2)
    denominator = np.sum((y_true - y_true.mean()) ** 2)
    return float(1.0 - numerator / np.maximum(denominator, 1e-12))


def trustworthiness(high_dim: np.ndarray, low_dim: np.ndarray, n_neighbors: int) -> float:
    n_samples = high_dim.shape[0]
    if n_samples <= 3 * n_neighbors + 1:
        raise ValueError("Need more samples to compute trustworthiness.")

    high_dist = cdist(high_dim, high_dim, metric="euclidean")
    low_dist = cdist(low_dim, low_dim, metric="euclidean")
    np.fill_diagonal(high_dist, np.inf)
    np.fill_diagonal(low_dist, np.inf)

    high_order = np.argsort(high_dist, axis=1)
    low_neighbors = np.argsort(low_dist, axis=1)[:, :n_neighbors]

    high_rank = np.empty_like(high_order)
    row_ids = np.arange(n_samples)[:, None]
    high_rank[row_ids, high_order] = np.arange(1, n_samples + 1)

    penalty = 0.0
    for i in range(n_samples):
        ranks = high_rank[i, low_neighbors[i]]
        penalty += np.sum(np.maximum(ranks - n_neighbors, 0))

    normalizer = n_samples * n_neighbors * (2 * n_samples - 3 * n_neighbors - 1)
    return float(1.0 - 2.0 * penalty / np.maximum(normalizer, 1e-12))


def pairwise_distance_correlation(high_dim: np.ndarray, low_dim: np.ndarray) -> float:
    high = pdist(high_dim, metric="euclidean")
    low = pdist(low_dim, metric="euclidean")
    corr = np.corrcoef(high, low)[0, 1]
    return float(corr)


def evaluate_latent_representation(
    train_latent: np.ndarray,
    test_latent: np.ndarray,
    train_theta: np.ndarray,
    test_theta: np.ndarray,
    train_contrast: np.ndarray,
    test_contrast: np.ndarray,
    test_input: np.ndarray,
    test_reconstruction: np.ndarray,
    ridge_penalty: float,
    trustworthiness_k: int,
    trustworthiness_subset: int,
    seed: int,
) -> tuple[Dict[str, float], Dict[str, np.ndarray]]:
    train_features = make_polynomial_features(train_latent)
    test_features = make_polynomial_features(test_latent)
    orientation_targets = np.stack([np.sin(train_theta), np.cos(train_theta)], axis=1)
    orientation_decoder = fit_ridge_regression(train_features, orientation_targets, ridge_penalty=ridge_penalty)
    contrast_decoder = fit_ridge_regression(train_features, train_contrast[:, None], ridge_penalty=ridge_penalty)

    orientation_pred_components = predict_ridge(test_features, orientation_decoder)
    theta_pred = np.mod(np.arctan2(orientation_pred_components[:, 0], orientation_pred_components[:, 1]), 2.0 * np.pi)
    contrast_pred = predict_ridge(test_features, contrast_decoder).reshape(-1)

    reconstruction_mse = float(np.mean((test_reconstruction - test_input) ** 2))

    rng = np.random.default_rng(seed)
    subset_size = min(int(trustworthiness_subset), test_input.shape[0])
    subset_idx = np.sort(rng.choice(test_input.shape[0], size=subset_size, replace=False))
    trust = trustworthiness(test_input[subset_idx], test_latent[subset_idx], n_neighbors=int(trustworthiness_k))
    distance_corr = pairwise_distance_correlation(test_input[subset_idx], test_latent[subset_idx])

    metrics = {
        "reconstruction_mse": reconstruction_mse,
        "orientation_circular_corr": circular_correlation(test_theta, theta_pred),
        "orientation_mae_deg": circular_mae_degrees(test_theta, theta_pred),
        "contrast_r2": r2_score(test_contrast, contrast_pred),
        "trustworthiness": trust,
        "pairwise_distance_corr": distance_corr,
    }
    predictions = {
        "theta_pred": theta_pred.astype(np.float32),
        "contrast_pred": contrast_pred.astype(np.float32),
        "orientation_decoder_weights": orientation_decoder["weights"].astype(np.float32),
        "orientation_decoder_bias": orientation_decoder["bias"].astype(np.float32),
        "contrast_decoder_weights": contrast_decoder["weights"].astype(np.float32),
        "contrast_decoder_bias": contrast_decoder["bias"].astype(np.float32),
    }
    return metrics, predictions


def evaluate_dropout_curve(
    train_latent: np.ndarray,
    train_theta: np.ndarray,
    train_contrast: np.ndarray,
    test_input: np.ndarray,
    test_theta: np.ndarray,
    test_contrast: np.ndarray,
    encoder_fn,
    dropout_fractions: list[float],
    ridge_penalty: float,
    seed: int,
) -> Dict[str, np.ndarray]:
    train_features = make_polynomial_features(train_latent)
    orientation_targets = np.stack([np.sin(train_theta), np.cos(train_theta)], axis=1)
    orientation_decoder = fit_ridge_regression(train_features, orientation_targets, ridge_penalty=ridge_penalty)
    contrast_decoder = fit_ridge_regression(train_features, train_contrast[:, None], ridge_penalty=ridge_penalty)

    rng = np.random.default_rng(seed)
    orientation_corr = []
    orientation_mae = []
    contrast_r2_values = []

    for frac in dropout_fractions:
        masked = test_input.copy()
        n_drop = int(round(masked.shape[1] * frac))
        if n_drop > 0:
            drop_idx = rng.choice(masked.shape[1], size=n_drop, replace=False)
            masked[:, drop_idx] = 0.0
        latent = encoder_fn(masked)
        latent_features = make_polynomial_features(latent)
        theta_components = predict_ridge(latent_features, orientation_decoder)
        theta_pred = np.mod(np.arctan2(theta_components[:, 0], theta_components[:, 1]), 2.0 * np.pi)
        contrast_pred = predict_ridge(latent_features, contrast_decoder).reshape(-1)
        orientation_corr.append(circular_correlation(test_theta, theta_pred))
        orientation_mae.append(circular_mae_degrees(test_theta, theta_pred))
        contrast_r2_values.append(r2_score(test_contrast, contrast_pred))

    return {
        "dropout_fraction": np.asarray(dropout_fractions, dtype=np.float32),
        "orientation_circular_corr": np.asarray(orientation_corr, dtype=np.float32),
        "orientation_mae_deg": np.asarray(orientation_mae, dtype=np.float32),
        "contrast_r2": np.asarray(contrast_r2_values, dtype=np.float32),
    }
