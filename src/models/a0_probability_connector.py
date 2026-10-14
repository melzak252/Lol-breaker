"""Learn one bounded specialist weight without changing native A0 calibration."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from src.analysis.a0_sports_correction import evaluate_correction
from src.analysis.research_benchmark import losses, safe, validate_frame
from src.models.a0_correction_experiment import (
    CODE_FILES, _json, _prediction_column, _quarter_end, _set_c0_release_bounds,
    _validate_c0_protocol, _verified, _verify_c0_prepared_sources,
    directories, select_penalty, sha256, training_indices, write_json,
)
from src.models.a0_uncertainty import _resolve_research_root


class ProbabilityConnector:
    """Convex SERIES-probability connector, fitted with a proper scoring rule."""

    SCHEMA = "a0_single_weight_probability_connector_v1"

    def __init__(self, penalty: float):
        if not np.isfinite(penalty) or penalty < 0:
            raise ValueError("penalty must be finite and nonnegative")
        self.penalty = float(penalty)
        self.alpha = 0.
        self.fit_diagnostics = {"status": "IDENTITY_UNFITTED", "n": 0}

    @staticmethod
    def _inputs(p_a0, p_specialist, y=None):
        a, b = (np.asarray(values, dtype=float) for values in (p_a0, p_specialist))
        if a.ndim != 1 or b.shape != a.shape:
            raise ValueError("Probability inputs must be aligned vectors")
        if any(not np.isfinite(values).all() or np.any((values <= 0) | (values >= 1)) for values in (a, b)):
            raise ValueError("Probabilities must be finite strict interior values")
        target = None if y is None else np.asarray(y, dtype=float)
        if target is not None and (target.shape != a.shape or not np.isin(target, [0, 1]).all()):
            raise ValueError("Outcomes must be an aligned binary vector")
        return a, b, target

    def fit(self, p_a0, p_specialist, y):
        a, b, target = self._inputs(p_a0, p_specialist, y)
        if not len(a) or target is None:
            raise ValueError("Fitting requires nonempty labeled rows")
        difference = b - a

        def gradient(alpha):
            p = (1 - alpha) * a + alpha * b
            return float(np.mean(difference * (p - target) / (p * (1 - p))) + self.penalty * alpha)

        lower, upper = gradient(0.), gradient(1.)
        if lower >= 0:
            self.alpha = 0.
            calls = 2
        elif upper <= 0:
            self.alpha = 1.
            calls = 2
        else:
            self.alpha, solution = brentq(gradient, 0., 1., xtol=1e-14, rtol=1e-14, full_output=True)
            if not solution.converged:
                raise RuntimeError("Scalar convex connector did not converge")
            calls = int(solution.function_calls) + 2
        p = self.predict_proba(a, b)
        derivative = gradient(self.alpha)
        projected = 0. if (self.alpha == 0. and derivative >= 0.) or (self.alpha == 1. and derivative <= 0.) else derivative
        self.fit_diagnostics = {
            "status": "CONVERGED", "n": len(a), "alpha": self.alpha,
            "objective": float(losses(target, p).mean() + .5 * self.penalty * self.alpha ** 2),
            "gradient": derivative, "projected_gradient": projected, "gradient_evaluations": calls,
        }
        return self

    def predict_proba(self, p_a0, p_specialist):
        a, b, _ = self._inputs(p_a0, p_specialist)
        if self.alpha == 0.:
            return a.copy()
        if self.alpha == 1.:
            return b.copy()
        p = (1 - self.alpha) * a + self.alpha * b
        if np.any((p <= 0.) | (p >= 1.)):
            raise ValueError("Floating-point mixture left the strict probability interior")
        return p

    def to_dict(self):
        return {"schema": self.SCHEMA, "penalty": self.penalty, "alpha": self.alpha,
                "fit_diagnostics": self.fit_diagnostics}

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict) or value.get("schema") != cls.SCHEMA:
            raise ValueError("Incompatible scalar connector artifact")
        model = cls(value["penalty"])
        if not np.isfinite(value["alpha"]) or not 0 <= value["alpha"] <= 1:
            raise ValueError("Invalid probability mixture weight")
        model.alpha = float(value["alpha"])
        model.fit_diagnostics = dict(value["fit_diagnostics"])
        return model


class C0ProbabilityConnector(ProbabilityConnector):
    """Identity-aware convex connector anchored to a named native C0 recipe."""

    SCHEMA = "c0_raw_specialist_probability_connector_v1"

    def __init__(self, penalty: float, anchor_identity: str = "native12__full__full"):
        if anchor_identity != "native12__full__full":
            raise ValueError("C0 connector must use the pinned native12__full__full anchor")
        super().__init__(penalty)
        self.anchor_identity = anchor_identity

    def to_dict(self):
        return {**super().to_dict(), "schema": self.SCHEMA,
                "anchor_identity": self.anchor_identity}

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict) or value.get("schema") != cls.SCHEMA:
            raise ValueError("Incompatible C0 connector artifact")
        model = cls(value["penalty"], value.get("anchor_identity"))
        if not np.isfinite(value["alpha"]) or not 0 <= value["alpha"] <= 1:
            raise ValueError("Invalid probability mixture weight")
        model.alpha = float(value["alpha"])
        model.fit_diagnostics = dict(value["fit_diagnostics"])
        return model


class C0SlopeCalibration:
    """Identity-centred nonnegative scalar slope calibration of C0 logits."""

    SCHEMA = "c0_nonnegative_logit_slope_v1"

    def __init__(self, penalty: float):
        if not np.isfinite(penalty) or penalty < 0:
            raise ValueError("penalty must be finite and nonnegative")
        self.penalty = float(penalty)
        self.slope = 1.
        self.fit_diagnostics = {"status": "IDENTITY_UNFITTED", "n": 0}

    @staticmethod
    def _inputs(p, y=None):
        p = np.asarray(p, dtype=float)
        if p.ndim != 1 or not np.isfinite(p).all() or np.any((p <= 0) | (p >= 1)):
            raise ValueError("C0 probabilities must be finite strict interior values")
        target = None if y is None else np.asarray(y, dtype=float)
        if target is not None and (target.shape != p.shape or not np.isin(target, [0, 1]).all()):
            raise ValueError("Outcomes must be an aligned binary vector")
        return p, target

    def predict_proba(self, p_c0):
        p, _ = self._inputs(p_c0)
        logits = np.log(p) - np.log1p(-p)
        z = self.slope * logits
        out = np.exp(-np.logaddexp(0., -z))
        if np.any((out <= 0) | (out >= 1)):
            raise ValueError("Slope forecast left strict probability interior")
        return out

    def fit(self, p_c0, y):
        p, target = self._inputs(p_c0, y)
        if not len(p):
            raise ValueError("Fitting requires nonempty labeled rows")
        logits = np.log(p) - np.log1p(-p)

        def gradient(slope):
            q = np.exp(-np.logaddexp(0., -slope * logits))
            return float(np.mean(logits * (q - target)) + self.penalty * (slope - 1.))

        lo, hi = gradient(0.), gradient(1.)
        if lo >= 0:
            self.slope = 0.
        elif hi <= 0:
            upper = 2.
            while gradient(upper) < 0:
                upper *= 2
            self.slope = float(brentq(gradient, 1., upper, xtol=1e-14, rtol=1e-14))
        else:
            self.slope = float(brentq(gradient, 0., 1., xtol=1e-14, rtol=1e-14))
        q = self.predict_proba(p)
        self.fit_diagnostics = {"status": "CONVERGED", "n": len(p), "slope": self.slope,
                                "objective": float(losses(target, q).mean()
                                                   + .5 * self.penalty * (self.slope - 1.) ** 2),
                                "gradient": gradient(self.slope)}
        return self

    def to_dict(self):
        return {"schema": self.SCHEMA, "penalty": self.penalty, "slope": self.slope,
                "fit_diagnostics": self.fit_diagnostics}

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict) or value.get("schema") != cls.SCHEMA:
            raise ValueError("Incompatible C0 calibration artifact")
        model = cls(value["penalty"])
        if not np.isfinite(value["slope"]) or value["slope"] < 0:
            raise ValueError("Invalid nonnegative logit slope")
        model.slope = float(value["slope"])
        model.fit_diagnostics = dict(value["fit_diagnostics"])
        return model

def train_c0(protocol_path: Path, research_root: str | Path | None = None) -> dict:
    """Train the frozen C0 connector/control grid through the existing stage runner."""
    from src.models.a0_correction_experiment import _prediction_column

    protocol = _json(protocol_path)
    _validate_c0_protocol(protocol)
    paths = directories(protocol)
    if protocol["market_diagnostic"]["enabled"]:
        _verified(protocol["market_diagnostic"]["protocol"], protocol["market_diagnostic"]["sha256"])
    prepared_path = paths["input"] / "prepared.json"
    prepared = _json(prepared_path)
    _verified(protocol_path, prepared["protocol_sha256"])
    _verified(prepared["joined_path"], prepared["joined_sha256"])
    origins = protocol["chronology"]["cv_origins"]
    penalties = protocol["model"]["penalties"]
    arms = protocol["model"]["arms"]
    if (len(origins) != 9 or len(set(origins)) != 9 or len(protocol["chronology"]["forecast_origins"]) != 3
            or penalties != [0.0001, 0.001, 0.01, 0.1]
            or arms != ["c0_connector", "c0_calibration"]):
        raise ValueError("Frozen Plan09 9-origin x 4-penalty x 2-arm grid changed")
    planned_fits = len(origins) * len(penalties) * len(arms)
    if planned_fits != 72:
        raise ValueError("Plan09 fit budget must be exactly 72 heads")
    if paths["model"].exists() or paths["prediction"].exists():
        raise FileExistsError("Plan09 model or prediction output already exists")
    frame = pd.read_parquet(prepared["joined_path"])
    root = _resolve_research_root(research_root)
    input_verification = _verify_c0_prepared_sources(prepared, protocol, root)
    code_paths = tuple(dict.fromkeys((*CODE_FILES, "src/models/a0_probability_connector.py")))
    execution = {"status": "SEALED_BEFORE_C0_HEAD_FITS",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "protocol_sha256": sha256(protocol_path), "prepared_sha256": sha256(prepared_path),
                "input_verification": input_verification,
                "code_sha256": {path: sha256(path) for path in code_paths},
                "packages": {name: importlib.metadata.version(name)
                             for name in ("numpy", "pandas", "scipy", "scikit-learn")},
                "native_anchor_identity": protocol["model"]["anchor_identity"],
                "base_fits": 0, "planned_head_fits": planned_fits}
    write_json(paths["input"] / "execution.json", execution)
    with zipfile.ZipFile(paths["input"] / "source_snapshot.zip", "x", zipfile.ZIP_DEFLATED) as snapshot:
        snapshot.write(protocol_path, arcname=str(protocol_path))
        for path in code_paths:
            snapshot.write(path, arcname=path)
    paths["model"].mkdir(parents=True)
    paths["prediction"].mkdir(parents=True)
    cv = pd.DataFrame(index=frame.index)
    cv["cv_origin"] = pd.Series(index=frame.index, dtype="string")
    grid = penalties
    fit_records, models = [], {}
    for origin in origins:
        fit = training_indices(frame, origin)
        if len(fit) < protocol["model"]["minimum_fit_rows"]:
            raise ValueError(f"Insufficient closed fit support at {origin}")
        target = frame.date.ge(origin) & frame.date.lt(_quarter_end(origin))
        training = frame.iloc[fit]
        cv.loc[target, "cv_origin"] = origin
        membership = paths["model"] / origin / "training_membership.json"
        write_json(membership, {"origin": origin, "ids": training.golgg_match_id.tolist(),
                                "n": len(fit), "max_label_release_day": training.head_label_release_day.max()})
        for arm in ("candidate", "control"):
            for penalty in grid:
                model = (C0ProbabilityConnector(penalty, protocol["model"]["anchor_identity"])
                         if arm == "candidate" else C0SlopeCalibration(penalty))
                if arm == "candidate":
                    model.fit(training.p_c0.to_numpy(float), training.p_specialist.to_numpy(float), training.y.to_numpy(int))
                    values = model.predict_proba(frame.loc[target, "p_c0"], frame.loc[target, "p_specialist"])
                else:
                    model.fit(training.p_c0.to_numpy(float), training.y.to_numpy(int))
                    values = model.predict_proba(frame.loc[target, "p_c0"])
                family = "candidate" if arm == "candidate" else "control"
                cv.loc[target, _prediction_column(family, penalty)] = values
                model_path = paths["model"] / origin / f"{family}_{penalty:g}.json"
                write_json(model_path, {"origin": origin, "family": family, "penalty": penalty,
                    "anchor_identity": protocol["model"]["anchor_identity"],
                    "valid_target_interval": [origin, _quarter_end(origin)], "model": model.to_dict(),
                    "training_membership": str(membership), "membership_sha256": sha256(membership)})
                models[(origin, family, penalty)] = model_path
                fit_records.append({"origin": origin, "family": family, "penalty": penalty,
                                    "path": str(model_path), "sha256": sha256(model_path), "n_train": len(fit)})
    result = frame.copy()
    result["head_status"] = "C0_WARMUP"
    result["head_origin"] = "C0_WARMUP"
    selections = []
    for origin in protocol["chronology"]["forecast_origins"]:
        target = frame.date.ge(origin) & frame.date.lt(_quarter_end(origin))
        for family, column in (("candidate", "p_c0_connector"), ("control", "p_c0_calibration")):
            penalty, receipt = select_penalty(frame, cv, origin, family, grid)
            model_path = models[(origin, family, penalty)]
            result.loc[target, column] = cv.loc[target, _prediction_column(family, penalty)]
            selection = {"origin": origin, "family": family, "penalty": penalty,
                         "selection": receipt, "model_path": str(model_path), "sha256": sha256(model_path),
                         "anchor_identity": protocol["model"]["anchor_identity"],
                         "valid_target_interval": [origin, _quarter_end(origin)]}
            write_json(paths["model"] / origin / f"selected_{family}.json", selection)
            selections.append(selection)
        result.loc[target, "head_status"] = "PAST_ONLY_FITTED_C0_HEAD"
        result.loc[target, "head_origin"] = origin
    _set_c0_release_bounds(result)
    generated_at = datetime.now(timezone.utc).isoformat()
    result["generated_at"] = generated_at
    cv.insert(0, "golgg_match_id", frame.golgg_match_id)
    result.to_parquet(paths["prediction"] / "predictions.parquet", index=False)
    cv.to_parquet(paths["prediction"] / "validation_grid.parquet", index=False)
    receipt = {"status": "COMPLETE_C0_HEAD_FITS", "created_at": generated_at,
               "protocol_sha256": sha256(protocol_path), "prepared_sha256": sha256(prepared_path),
               "fit_count": len(fit_records), "fits": fit_records, "selections": selections,
               "rows": len(result), "warmup_rows": int(result.head_status.eq("C0_WARMUP").sum()),
               "execution_sha256": sha256(paths["input"] / "execution.json"),
               "prediction_path": str(paths["prediction"] / "predictions.parquet"),
               "prediction_sha256": sha256(paths["prediction"] / "predictions.parquet"),
               "validation_grid_sha256": sha256(paths["prediction"] / "validation_grid.parquet"),
               "base_fits": 0, "head_fits": len(fit_records)}
    write_json(paths["report"] / "training.json", receipt)
    return receipt


def run(protocol_path: Path, research_root: str | Path | None = None) -> dict:
    if _json(protocol_path).get("schema") == "c0_raw_specialist_connector_v1":
        return train_c0(protocol_path, research_root)
    protocol = _json(protocol_path)
    paths = directories(protocol)
    _verified(protocol["source_prepared"], protocol["source_prepared_sha256"])
    _verified(protocol["source_predictions"], protocol["source_predictions_sha256"])
    prepared = _json(protocol["source_prepared"])
    _verified(prepared["joined_path"], prepared["joined_sha256"])
    root = _resolve_research_root(research_root)
    for name, digest in prepared["input_hashes"].items():
        _verified(root / name.removeprefix("research:") if name.startswith("research:") else name, digest)
    frame = pd.read_parquet(protocol["source_predictions"])
    validate_frame(frame, ["p_a0", "p_specialist", "p_calibration"])
    if paths["model"].exists() or paths["prediction"].exists():
        raise FileExistsError("Connector run already exists")
    paths["model"].mkdir(parents=True)
    paths["prediction"].mkdir(parents=True)
    code = {name: sha256(name) for name in (*CODE_FILES, "src/models/a0_probability_connector.py")}
    execution = {"status": "SEALED_BEFORE_SCALAR_FITS", "created_at": datetime.now(timezone.utc).isoformat(),
                 "protocol_sha256": sha256(protocol_path), "code_sha256": code,
                 "source_prepared_sha256": protocol["source_prepared_sha256"],
                 "source_predictions_sha256": protocol["source_predictions_sha256"]}
    write_json(paths["input"] / "execution.json", execution)
    with zipfile.ZipFile(paths["input"] / "source_snapshot.zip", "x", zipfile.ZIP_DEFLATED) as snapshot:
        for name in code:
            snapshot.write(name, arcname=name)
    cv = pd.DataFrame(index=frame.index)
    cv["cv_origin"] = pd.Series(index=frame.index, dtype="string")
    grid = protocol["model"]["regularization_grid"]
    fits, models = [], {}
    for origin in protocol["chronology"]["cv_origins"]:
        indices = training_indices(frame, origin)
        if len(indices) < protocol["chronology"]["minimum_fit_rows"]:
            raise ValueError("Insufficient closed head training support")
        train_frame = frame.iloc[indices]
        target = frame.date.ge(origin) & frame.date.lt(_quarter_end(origin))
        cv.loc[target, "cv_origin"] = origin
        membership = paths["model"] / origin / "training_membership.json"
        write_json(membership, {"ids": train_frame.golgg_match_id.tolist(), "origin": origin,
                               "n": len(indices), "last_release": train_frame.head_label_release_day.max()})
        for penalty in grid:
            model = ProbabilityConnector(penalty).fit(train_frame.p_a0, train_frame.p_specialist, train_frame.y)
            cv.loc[target, _prediction_column("candidate", penalty)] = model.predict_proba(frame.loc[target, "p_a0"], frame.loc[target, "p_specialist"])
            model_path = paths["model"] / origin / f"candidate_{penalty:g}.json"
            write_json(model_path, {"origin": origin, "family": "candidate", "penalty": penalty,
                                   "training_membership": str(membership), "membership_sha256": sha256(membership),
                                   "model": model.to_dict(), "fit_diagnostics": model.fit_diagnostics})
            fits.append({"origin": origin, "penalty": penalty, "path": str(model_path), "sha256": sha256(model_path),
                         "n_train": len(indices), "n_validation": int(target.sum())})
            models[(origin, penalty)] = model_path
    result = frame.copy()
    result["p_correction"] = result.p_a0
    selections = []
    for origin in protocol["chronology"]["forecast_origins"]:
        target = frame.date.ge(origin) & frame.date.lt(_quarter_end(origin))
        penalty, selected = select_penalty(frame, cv, origin, "candidate", grid)
        model_path = models[(origin, penalty)]
        result.loc[target, "p_correction"] = cv.loc[target, _prediction_column("candidate", penalty)]
        artifact = {"origin": origin, "family": "candidate", "penalty": penalty, "selection": selected,
                    "model_path": str(model_path), "sha256": sha256(model_path),
                    "specialist_artifact": prepared["specialist_origins"][origin]["member"],
                    "native_recipe": prepared["native_oof_provenance"]["years"][origin[:4]],
                    "valid_target_interval": [origin, _quarter_end(origin)], "research_only": True}
        write_json(paths["model"] / origin / "selected_candidate.json", artifact)
        selections.append(artifact)
    result["generated_at"] = datetime.now(timezone.utc).isoformat()
    result["head_status"] = np.where(result.date.lt(protocol["chronology"]["warmup_end"]),
                                    "WARMUP_IDENTITY_A0", "PAST_ONLY_FIXED_CALIBRATION_CONNECTOR")
    validate_frame(result, ["p_a0", "p_specialist", "p_calibration", "p_correction"])
    cv.insert(0, "golgg_match_id", frame.golgg_match_id)
    cv.to_parquet(paths["prediction"] / "validation_grid.parquet", index=False)
    prediction_path = paths["prediction"] / "predictions.parquet"
    result.to_parquet(prediction_path, index=False)
    training = {"status": "COMPLETE_SCALAR_FITS", "protocol_sha256": sha256(protocol_path),
                "execution_sha256": sha256(paths["input"] / "execution.json"), "fits": fits, "selections": selections,
                "rows": len(result), "warmup_rows": int(result.date.lt(protocol["chronology"]["warmup_end"]).sum()),
                "prediction_path": str(prediction_path), "prediction_sha256": sha256(prediction_path),
                "new_base_fits": 0, "correction_fits": len(fits), "controls_reused_unchanged": True}
    write_json(paths["report"] / "training.json", training)
    assessment = evaluate_correction(result, protocol)
    assessment.update(protocol_sha256=sha256(protocol_path), prediction_sha256=sha256(prediction_path))
    write_json(paths["report"] / "results.json", safe(assessment))
    return assessment


def predict(artifact_path: Path, inputs_path: Path) -> dict:
    artifact = _json(artifact_path)
    _verified(artifact["model_path"], artifact["sha256"])
    saved = _json(artifact["model_path"])
    model_state = saved["model"]
    if model_state.get("schema") in (C0ProbabilityConnector.SCHEMA, C0SlopeCalibration.SCHEMA):
        expected_family = "candidate" if model_state["schema"] == C0ProbabilityConnector.SCHEMA else "control"
        if (saved.get("origin") != artifact.get("origin")
                or saved.get("valid_target_interval") != artifact.get("valid_target_interval")
                or saved.get("family") != expected_family
                or artifact.get("family") != expected_family
                or saved.get("anchor_identity") != artifact.get("anchor_identity")):
            raise ValueError("Selected wrapper does not match the saved head origin/interval/family/anchor")
    if model_state.get("schema") == C0ProbabilityConnector.SCHEMA:
        model = C0ProbabilityConnector.from_dict(model_state)
        inputs = _json(inputs_path)
        start, end = artifact["valid_target_interval"]
        days = pd.to_datetime(inputs["date"], errors="raise").strftime("%Y-%m-%d")
        if (not ((days >= start) & (days < end)).all()
                or not np.isin(inputs["best_of"], [1, 3, 5]).all()
                or len(days) != len(inputs["best_of"])
                or len(days) != len(inputs["p_c0"]) or len(days) != len(inputs["p_specialist"])):
            raise ValueError("Input is outside the saved C0 target/series-format contract")
        if artifact.get("anchor_identity") != model.anchor_identity or inputs.get("anchor_identity") != model.anchor_identity:
            raise ValueError("C0 anchor identity does not match the saved artifact")
        p = model.predict_proba(inputs["p_c0"], inputs["p_specialist"])
        return {"origin": artifact["origin"], "anchor_identity": model.anchor_identity,
                "valid_target_interval": [start, end], "alpha": model.alpha,
                "p": p.tolist(), "research_only": True}
    if model_state.get("schema") == C0SlopeCalibration.SCHEMA:
        model = C0SlopeCalibration.from_dict(model_state)
        inputs = _json(inputs_path)
        start, end = artifact["valid_target_interval"]
        days = pd.to_datetime(inputs["date"], errors="raise").strftime("%Y-%m-%d")
        if (not ((days >= start) & (days < end)).all()
                or not np.isin(inputs["best_of"], [1, 3, 5]).all()
                or len(days) != len(inputs["best_of"])
                or len(days) != len(inputs["p_c0"])):
            raise ValueError("Input is outside the saved C0 target/series-format contract")
        if artifact.get("anchor_identity") != "native12__full__full" or inputs.get("anchor_identity") != artifact["anchor_identity"]:
            raise ValueError("C0 calibration anchor identity mismatch")
        return {"origin": artifact["origin"], "anchor_identity": artifact["anchor_identity"],
                "valid_target_interval": [start, end], "slope": model.slope,
                "p": model.predict_proba(inputs["p_c0"]).tolist(), "research_only": True}
    model = ProbabilityConnector.from_dict(model_state)
    inputs = _json(inputs_path)
    days = pd.to_datetime(inputs["date"], errors="raise").strftime("%Y-%m-%d")
    start, end = artifact["valid_target_interval"]
    if not ((days >= start) & (days < end)).all() or not np.isin(inputs["best_of"], [1, 3, 5]).all():
        raise ValueError("Input is outside the saved date/series-format contract")
    p = model.predict_proba(inputs["p_a0"], inputs["p_specialist"])
    if len(days) != len(p) or len(inputs["best_of"]) != len(p):
        raise ValueError("Forecast metadata and probability inputs differ in length")
    return {"origin": artifact["origin"], "alpha": model.alpha, "p": p.tolist(), "research_only": True}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["train", "predict"])
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--research-root", type=Path)
    parser.add_argument("--artifact", type=Path)
    parser.add_argument("--inputs", type=Path)
    args = parser.parse_args()
    if args.stage == "train":
        if args.protocol is None:
            parser.error("train requires --protocol")
        result = run(args.protocol, args.research_root)
        value = result["decision"] if "decision" in result else {
            key: result[key] for key in ("status", "rows", "head_fits", "base_fits")
        }
    else:
        if args.artifact is None or args.inputs is None:
            parser.error("predict requires --artifact and --inputs")
        value = predict(args.artifact, args.inputs)
    print(json.dumps(value, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
