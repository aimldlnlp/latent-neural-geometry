from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
from scipy.ndimage import gaussian_filter1d


@dataclass
class RecordingDataset:
    train_responses: np.ndarray
    test_responses: np.ndarray
    train_theta: np.ndarray
    test_theta: np.ndarray
    train_contrast: np.ndarray
    test_contrast: np.ndarray
    train_sequence_ids: np.ndarray
    test_sequence_ids: np.ndarray
    train_time_indices: np.ndarray
    test_time_indices: np.ndarray
    test_sequence_tensor: np.ndarray
    test_theta_tensor: np.ndarray
    test_contrast_tensor: np.ndarray
    time_axis: np.ndarray
    neuron_params: Dict[str, np.ndarray]


def _softplus(x: np.ndarray) -> np.ndarray:
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)


def _wrap_angle(theta: np.ndarray) -> np.ndarray:
    return np.mod(theta, 2.0 * np.pi)


def _sample_sequences(rng: np.random.Generator, n_sequences: int, sequence_length: int, dt: float, config: Dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    time_axis = np.arange(sequence_length, dtype=np.float64) * dt
    theta = np.zeros((n_sequences, sequence_length), dtype=np.float64)
    contrast = np.zeros((n_sequences, sequence_length), dtype=np.float64)
    for idx in range(n_sequences):
        base_theta = rng.uniform(0.0, 2.0 * np.pi)
        velocity = rng.uniform(*config["angle_velocity_range"])
        freq_a = rng.uniform(0.08, 0.18)
        freq_b = rng.uniform(0.20, 0.36)
        warp_a = rng.uniform(0.2, 0.5)
        warp_b = rng.uniform(0.08, 0.22)
        phase_a = rng.uniform(0.0, 2.0 * np.pi)
        phase_b = rng.uniform(0.0, 2.0 * np.pi)
        trajectory = (
            base_theta
            + velocity * time_axis
            + warp_a * np.sin(2.0 * np.pi * freq_a * time_axis + phase_a)
            + warp_b * np.sin(2.0 * np.pi * freq_b * time_axis + phase_b)
        )
        theta[idx] = _wrap_angle(trajectory)

        base_contrast = rng.uniform(*config["contrast_base_range"])
        contrast_amp = rng.uniform(*config["contrast_amplitude_range"])
        contrast_freq = rng.uniform(0.06, 0.18)
        contrast_phase = rng.uniform(0.0, 2.0 * np.pi)
        secondary_amp = rng.uniform(0.04, 0.12)
        raw_contrast = (
            base_contrast
            + contrast_amp * np.sin(2.0 * np.pi * contrast_freq * time_axis + contrast_phase)
            + secondary_amp * np.cos(2.0 * np.pi * (contrast_freq * 0.5) * time_axis + phase_a)
        )
        contrast[idx] = np.clip(raw_contrast, 0.05, 1.0)
    return theta, contrast, time_axis


def _make_neuron_bank(rng: np.random.Generator, n_neurons: int) -> Dict[str, np.ndarray]:
    return {
        "preferred_angle": np.linspace(0.0, 2.0 * np.pi, n_neurons, endpoint=False) + rng.normal(0.0, 0.08, size=n_neurons),
        "kappa": rng.uniform(1.6, 4.4, size=n_neurons),
        "amplitude": rng.uniform(3.0, 8.5, size=n_neurons),
        "baseline": rng.uniform(-0.8, 0.6, size=n_neurons),
        "contrast_gain": rng.uniform(0.5, 1.7, size=n_neurons),
        "contrast_only_gain": rng.uniform(0.4, 1.8, size=n_neurons),
        "cross_term": rng.uniform(-1.3, 1.3, size=n_neurons),
        "second_harmonic": rng.uniform(0.2, 1.6, size=n_neurons),
        "contrast_preference": rng.uniform(0.15, 0.85, size=n_neurons),
        "temporal_gain": rng.uniform(-0.7, 0.7, size=n_neurons),
    }


def _compute_expected_responses(theta: np.ndarray, contrast: np.ndarray, neuron_params: Dict[str, np.ndarray], dt: float) -> np.ndarray:
    theta_expanded = theta[..., None]
    contrast_expanded = contrast[..., None]
    pref = neuron_params["preferred_angle"][None, None, :]
    delta = theta_expanded - pref
    base_tuning = np.exp(neuron_params["kappa"][None, None, :] * np.cos(delta) - neuron_params["kappa"][None, None, :])
    harmonic = 0.5 * (1.0 + np.cos(2.0 * delta))
    contrast_drive = 0.35 + contrast_expanded * neuron_params["contrast_gain"][None, None, :]
    contrast_offset = contrast_expanded - neuron_params["contrast_preference"][None, None, :]
    dcontrast = np.gradient(contrast, dt, axis=1)[..., None]

    drive = (
        neuron_params["baseline"][None, None, :]
        + neuron_params["amplitude"][None, None, :] * base_tuning * contrast_drive
        + neuron_params["contrast_only_gain"][None, None, :] * (contrast_expanded - 0.5)
        + 0.6 * neuron_params["contrast_only_gain"][None, None, :] * (contrast_offset ** 2)
        + neuron_params["cross_term"][None, None, :] * contrast_offset * np.sin(delta)
        + neuron_params["second_harmonic"][None, None, :] * harmonic * (0.25 + contrast_expanded)
        + neuron_params["temporal_gain"][None, None, :] * dcontrast
    )
    return _softplus(drive)


def _sample_population(
    rng: np.random.Generator,
    expected: np.ndarray,
    noise_std: float,
    spike_scale: float,
    smoothing_sigma: float,
) -> np.ndarray:
    spikes = rng.poisson(np.clip(expected * spike_scale, 1e-4, None))
    transformed = np.sqrt(spikes + 0.25)
    noisy = transformed + rng.normal(0.0, noise_std, size=transformed.shape)
    if smoothing_sigma > 0.0:
        noisy = gaussian_filter1d(noisy, sigma=smoothing_sigma, axis=1, mode="nearest")
    return noisy.astype(np.float32)


def build_dataset(config: Dict[str, Any]) -> RecordingDataset:
    rng = np.random.default_rng(config["seed"])
    data_cfg = config["data"]
    n_train = int(data_cfg["n_train_sequences"])
    n_test = int(data_cfg["n_test_sequences"])
    sequence_length = int(data_cfg["sequence_length"])
    n_neurons = int(data_cfg["n_neurons"])
    dt = float(data_cfg["dt"])

    neuron_params = _make_neuron_bank(rng, n_neurons)
    theta_train, contrast_train, time_axis = _sample_sequences(rng, n_train, sequence_length, dt, data_cfg)
    theta_test, contrast_test, _ = _sample_sequences(rng, n_test, sequence_length, dt, data_cfg)

    expected_train = _compute_expected_responses(theta_train, contrast_train, neuron_params, dt)
    expected_test = _compute_expected_responses(theta_test, contrast_test, neuron_params, dt)

    responses_train = _sample_population(
        rng,
        expected_train,
        noise_std=float(data_cfg["noise_std"]),
        spike_scale=float(data_cfg["spike_scale"]),
        smoothing_sigma=float(data_cfg["smoothing_sigma"]),
    )
    responses_test = _sample_population(
        rng,
        expected_test,
        noise_std=float(data_cfg["noise_std"]),
        spike_scale=float(data_cfg["spike_scale"]),
        smoothing_sigma=float(data_cfg["smoothing_sigma"]),
    )

    train_sequence_ids = np.repeat(np.arange(n_train), sequence_length)
    test_sequence_ids = np.repeat(np.arange(n_test), sequence_length)
    train_time_indices = np.tile(np.arange(sequence_length), n_train)
    test_time_indices = np.tile(np.arange(sequence_length), n_test)

    return RecordingDataset(
        train_responses=responses_train.reshape(-1, n_neurons),
        test_responses=responses_test.reshape(-1, n_neurons),
        train_theta=theta_train.reshape(-1),
        test_theta=theta_test.reshape(-1),
        train_contrast=contrast_train.reshape(-1),
        test_contrast=contrast_test.reshape(-1),
        train_sequence_ids=train_sequence_ids.astype(np.int32),
        test_sequence_ids=test_sequence_ids.astype(np.int32),
        train_time_indices=train_time_indices.astype(np.int32),
        test_time_indices=test_time_indices.astype(np.int32),
        test_sequence_tensor=responses_test,
        test_theta_tensor=theta_test,
        test_contrast_tensor=contrast_test,
        time_axis=time_axis.astype(np.float32),
        neuron_params={key: value.astype(np.float32) for key, value in neuron_params.items()},
    )
