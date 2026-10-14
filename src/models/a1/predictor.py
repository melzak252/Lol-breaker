"""A1 inference adapter with side-symmetric slope calibration."""

from __future__ import annotations

import torch
import numpy as np

from src.models.a1.architecture import A1Model


class A1Predictor:
    """Inference predictor for the trained A1 neural model."""

    def __init__(self, model: A1Model, calibration_params: dict | None = None):

        self.model = model
        self.model.eval()
        params = {"platt_a": 1.0, "platt_b": 0.0}
        if calibration_params is not None:
            params.update(calibration_params)
        try:
            a = float(params["platt_a"])
            b = float(params["platt_b"])
        except (TypeError, ValueError) as exc:
            raise ValueError("Calibration requires numeric platt_a and platt_b") from exc
        if not np.isfinite(a) or a <= 0.0:
            raise ValueError("Calibration platt_a must be finite and positive")
        if not np.isfinite(b) or b != 0.0:
            raise ValueError("Side-symmetric A1 calibration requires finite platt_b=0")
        self.calibration_params = {"platt_a": a, "platt_b": 0.0}

    @torch.no_grad()
    def predict_batch(
        self,
        context: np.ndarray,
        player_ratings: np.ndarray,
        player_history: np.ndarray,
        player_mask: np.ndarray,
        player_champions: np.ndarray,
        meta_champions: np.ndarray,
        best_of: np.ndarray,
    ) -> np.ndarray:
        """Predict win probabilities for a batch of matches."""
        parameter = next(self.model.parameters())
        device = parameter.device
        floating_dtype = parameter.dtype
        ctx_t = torch.as_tensor(context, dtype=floating_dtype, device=device)
        pr_t = torch.as_tensor(player_ratings, dtype=floating_dtype, device=device)
        ph_t = torch.as_tensor(player_history, dtype=floating_dtype, device=device)
        pm_t = torch.as_tensor(player_mask, dtype=torch.bool, device=device)
        pc_t = torch.as_tensor(player_champions, dtype=torch.long, device=device)
        mc_t = torch.as_tensor(meta_champions, dtype=floating_dtype, device=device)
        bo_t = torch.as_tensor(best_of, device=device)

        raw_probs = self.model(
            context=ctx_t,
            player_ratings=pr_t,
            player_history=ph_t,
            player_mask=pm_t,
            player_champions=pc_t,
            meta_champions=mc_t,
            best_of=bo_t,
        ).float().cpu().numpy()

        # Apply calibration if parameters exist
        a = self.calibration_params.get("platt_a", 1.0)
        b = self.calibration_params.get("platt_b", 0.0)
        if a != 1.0 or b != 0.0:
            logit_raw = np.log(np.clip(raw_probs, 1e-6, 1.0 - 1e-6) / np.clip(1.0 - raw_probs, 1e-6, 1.0 - 1e-6))
            logit_cal = a * logit_raw + b
            cal_probs = 1.0 / (1.0 + np.exp(-logit_cal))
            return np.clip(cal_probs, 1e-6, 1.0 - 1e-6)

        return np.clip(raw_probs, 1e-6, 1.0 - 1e-6)
