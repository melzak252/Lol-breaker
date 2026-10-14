"""Odds-free, no-intercept linear students with optional auxiliary supervision."""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit


def fit_auxiliary_student(
    x: np.ndarray,
    y: np.ndarray,
    teacher: np.ndarray,
    weight: float,
    c: float = 0.1,
) -> tuple[np.ndarray, dict]:
    """Fit outcome BCE on every row plus separately normalized teacher BCE.

    The objective is mean_all BCE(y, sigmoid(x @ coef)) + weight *
    mean_covered BCE(teacher, sigmoid(x @ coef)) + ||coef||² / (2*c*n_all).
    Only NaN marks absent teacher supervision; observed teacher labels must be
    finite probabilities. Outcomes must be binary and may contain one class.
    Positive auxiliary weight requires at least one covered row. At weight zero,
    this is the existing all-row ``train_student`` objective and optimizer.

    Inputs must already be scaled without centering. No intercept or inference
    teacher is fitted: negating the input negates the logit exactly. Stopping
    uses only the training objective, never evaluation outcomes.
    """
    for name, value in (("x", x), ("y", y), ("teacher", teacher)):
        if np.iscomplexobj(value):
            raise ValueError(f"{name} must be real-valued")
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    teacher = np.asarray(teacher, dtype=float)
    if x.ndim != 2 or not x.shape[0] or not x.shape[1]:
        raise ValueError("x must be a nonempty two-dimensional feature matrix")
    n_all = len(x)
    if y.shape != (n_all,) or teacher.shape != (n_all,):
        raise ValueError("y and teacher must be one-dimensional with one target per row")
    if not np.isfinite(x).all():
        raise ValueError("x must contain only finite features")
    if not np.isfinite(y).all() or np.any((y != 0) & (y != 1)):
        raise ValueError("y must contain only binary outcome labels")
    covered = ~np.isnan(teacher)
    q = teacher[covered]
    if not np.isfinite(q).all() or np.any((q < 0) | (q > 1)):
        raise ValueError("observed teacher targets must be finite probabilities in [0, 1]")
    for name, value in (("weight", weight), ("c", c)):
        if np.ndim(value) != 0 or np.iscomplexobj(value):
            raise ValueError(f"{name} must be a real scalar")
    weight, c = float(weight), float(c)
    if not np.isfinite(weight) or weight < 0:
        raise ValueError("weight must be finite and nonnegative")
    if not np.isfinite(c) or c <= 0:
        raise ValueError("c must be finite and positive")
    n_covered = int(covered.sum())
    if weight > 0 and n_covered == 0:
        raise ValueError("positive auxiliary weight requires observed teacher targets")

    def objective(coef):
        z = x @ coef
        loss = np.mean(np.logaddexp(0, z) - y * z) + np.dot(coef, coef) / (
            2 * c * n_all
        )
        p = expit(z)
        residual = p - y
        if weight > 0:
            covered_z = z[covered]
            loss += weight * np.mean(np.logaddexp(0, covered_z) - q * covered_z)
            # Keep the final /n_all shared with the outcome gradient, but make
            # the auxiliary contribution a mean over covered rows, not all rows.
            residual[covered] += (weight * n_all / n_covered) * (p[covered] - q)
        gradient = x.T @ residual / n_all + coef / (c * n_all)
        return loss, gradient

    fit = minimize(
        objective,
        np.zeros(x.shape[1]),
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": 3000, "ftol": 1e-12, "gtol": 1e-8, "maxls": 50},
    )
    if not fit.success or not (
        np.isfinite(fit.x).all() and np.isfinite(fit.fun) and np.isfinite(fit.jac).all()
    ):
        raise ValueError(
            f"student optimizer failed: status={fit.status}, iterations={fit.nit}, "
            f"objective={fit.fun}, message={fit.message}"
        )
    z = x @ fit.x
    covered_z = z[covered]
    return fit.x, {
        "iterations": int(fit.nit),
        "objective": float(fit.fun),
        "converged": bool(fit.success),
        "message": str(fit.message),
        "gradient_max_abs": float(np.max(abs(fit.jac))),
        "status": int(fit.status),
        "function_evaluations": int(fit.nfev),
        "gradient_evaluations": int(fit.njev),
        "n_all": n_all,
        "n_covered": n_covered,
        "teacher_coverage_fraction": n_covered / n_all,
        "weight": weight,
        "c": c,
        "outcome_loss": float(np.mean(np.logaddexp(0, z) - y * z)),
        "auxiliary_loss": (
            float(np.mean(np.logaddexp(0, covered_z) - q * covered_z))
            if n_covered else None
        ),
        "l2_penalty": float(np.dot(fit.x, fit.x) / (2 * c * n_all)),
    }
