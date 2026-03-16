from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np


@dataclass
class TrainingHistory:
    train_loss: list[float]
    val_loss: list[float]


class PCABaseline:
    def __init__(self, latent_dim: int):
        self.latent_dim = int(latent_dim)
        self.mean_: np.ndarray | None = None
        self.components_: np.ndarray | None = None
        self.explained_variance_ratio_: np.ndarray | None = None

    def fit(self, x: np.ndarray) -> "PCABaseline":
        self.mean_ = x.mean(axis=0, keepdims=True)
        centered = x - self.mean_
        _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
        self.components_ = vt[: self.latent_dim].T
        variance = (singular_values ** 2) / max(x.shape[0] - 1, 1)
        self.explained_variance_ratio_ = variance[: self.latent_dim] / np.maximum(variance.sum(), 1e-8)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean_) @ self.components_

    def inverse_transform(self, z: np.ndarray) -> np.ndarray:
        return z @ self.components_.T + self.mean_


class NumpyAutoencoder:
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int, seed: int):
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.latent_dim = int(latent_dim)
        self.rng = np.random.default_rng(seed)
        self.params = self._init_params()

    def _init_params(self) -> Dict[str, np.ndarray]:
        def scaled_normal(fan_in: int, fan_out: int) -> np.ndarray:
            scale = np.sqrt(2.0 / (fan_in + fan_out))
            return self.rng.normal(0.0, scale, size=(fan_in, fan_out)).astype(np.float64)

        return {
            "w1": scaled_normal(self.input_dim, self.hidden_dim),
            "b1": np.zeros(self.hidden_dim, dtype=np.float64),
            "w2": scaled_normal(self.hidden_dim, self.latent_dim),
            "b2": np.zeros(self.latent_dim, dtype=np.float64),
            "w3": scaled_normal(self.latent_dim, self.hidden_dim),
            "b3": np.zeros(self.hidden_dim, dtype=np.float64),
            "w4": scaled_normal(self.hidden_dim, self.input_dim),
            "b4": np.zeros(self.input_dim, dtype=np.float64),
        }

    def _forward(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
        h1_pre = x @ self.params["w1"] + self.params["b1"]
        h1 = np.tanh(h1_pre)
        z = h1 @ self.params["w2"] + self.params["b2"]
        h2_pre = z @ self.params["w3"] + self.params["b3"]
        h2 = np.tanh(h2_pre)
        recon = h2 @ self.params["w4"] + self.params["b4"]
        cache = {"x": x, "h1": h1, "z": z, "h2": h2}
        return recon, z, cache

    def encode(self, x: np.ndarray) -> np.ndarray:
        _, z, _ = self._forward(x.astype(np.float64))
        return z.astype(np.float32)

    def reconstruct(self, x: np.ndarray) -> np.ndarray:
        recon, _, _ = self._forward(x.astype(np.float64))
        return recon.astype(np.float32)

    def decode(self, z: np.ndarray) -> np.ndarray:
        z = z.astype(np.float64)
        h2_pre = z @ self.params["w3"] + self.params["b3"]
        h2 = np.tanh(h2_pre)
        recon = h2 @ self.params["w4"] + self.params["b4"]
        return recon.astype(np.float32)

    def fit(
        self,
        x: np.ndarray,
        epochs: int,
        batch_size: int,
        learning_rate: float,
        weight_decay: float,
        validation_fraction: float,
        early_stopping_patience: int,
        input_noise_std: float,
    ) -> TrainingHistory:
        x = x.astype(np.float64)
        n_samples = x.shape[0]
        n_val = max(1, int(n_samples * validation_fraction))
        perm = self.rng.permutation(n_samples)
        val_idx = perm[:n_val]
        train_idx = perm[n_val:]
        train_x = x[train_idx]
        val_x = x[val_idx]

        moments_m = {name: np.zeros_like(value) for name, value in self.params.items()}
        moments_v = {name: np.zeros_like(value) for name, value in self.params.items()}
        beta1 = 0.9
        beta2 = 0.999
        epsilon = 1e-8
        step = 0
        history = TrainingHistory(train_loss=[], val_loss=[])
        best_val = np.inf
        best_params = {name: value.copy() for name, value in self.params.items()}
        patience = 0

        for epoch in range(int(epochs)):
            batch_perm = self.rng.permutation(train_x.shape[0])
            for start in range(0, train_x.shape[0], int(batch_size)):
                batch = train_x[batch_perm[start : start + int(batch_size)]]
                noisy_batch = batch + self.rng.normal(0.0, input_noise_std, size=batch.shape)
                recon, _, cache = self._forward(noisy_batch)
                grad = 2.0 * (recon - batch) / max(batch.shape[0], 1)

                grad_w4 = cache["h2"].T @ grad + 2.0 * weight_decay * self.params["w4"]
                grad_b4 = grad.sum(axis=0)
                grad_h2 = grad @ self.params["w4"].T
                grad_h2_pre = grad_h2 * (1.0 - cache["h2"] ** 2)

                grad_w3 = cache["z"].T @ grad_h2_pre + 2.0 * weight_decay * self.params["w3"]
                grad_b3 = grad_h2_pre.sum(axis=0)
                grad_z = grad_h2_pre @ self.params["w3"].T

                grad_w2 = cache["h1"].T @ grad_z + 2.0 * weight_decay * self.params["w2"]
                grad_b2 = grad_z.sum(axis=0)
                grad_h1 = grad_z @ self.params["w2"].T
                grad_h1_pre = grad_h1 * (1.0 - cache["h1"] ** 2)

                grad_w1 = cache["x"].T @ grad_h1_pre + 2.0 * weight_decay * self.params["w1"]
                grad_b1 = grad_h1_pre.sum(axis=0)

                grads = {
                    "w1": grad_w1,
                    "b1": grad_b1,
                    "w2": grad_w2,
                    "b2": grad_b2,
                    "w3": grad_w3,
                    "b3": grad_b3,
                    "w4": grad_w4,
                    "b4": grad_b4,
                }

                step += 1
                for name, grad_value in grads.items():
                    moments_m[name] = beta1 * moments_m[name] + (1.0 - beta1) * grad_value
                    moments_v[name] = beta2 * moments_v[name] + (1.0 - beta2) * (grad_value ** 2)
                    m_hat = moments_m[name] / (1.0 - beta1 ** step)
                    v_hat = moments_v[name] / (1.0 - beta2 ** step)
                    self.params[name] -= learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)

            train_loss = self.loss(train_x, weight_decay)
            val_loss = self.loss(val_x, weight_decay)
            history.train_loss.append(train_loss)
            history.val_loss.append(val_loss)

            if val_loss < best_val - 1e-6:
                best_val = val_loss
                best_params = {name: value.copy() for name, value in self.params.items()}
                patience = 0
            else:
                patience += 1
                if patience >= int(early_stopping_patience):
                    break

        self.params = best_params
        return history

    def loss(self, x: np.ndarray, weight_decay: float) -> float:
        recon, _, _ = self._forward(x.astype(np.float64))
        mse = float(np.mean((recon - x) ** 2))
        penalty = weight_decay * sum(float(np.sum(self.params[name] ** 2)) for name in ("w1", "w2", "w3", "w4"))
        return mse + penalty
