"""Fixed-design integration uncertainty, not confidence in sports-model accuracy.

No sampling design is inferred from a seed, a hit table, or a finite list of
strength vectors. Callers must declare iid sampling and fixed-N stopping.
Hybrid observations are bounded conditional probabilities, not integer hits.
"""
from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from numbers import Integral, Real
from statistics import NormalDist
from typing import Any


_MARGINAL_LAWS = ("champion_prob", "final_prob", "advance_prob", "placement_prob",
                  "swiss_record_prob", "stage_rank_prob")
_JOINT_LAWS = ("joint_final_prob", "joint_advance_prob")
_COVERAGE_SCOPE = "pointwise_fixed_n_numerical_only"
_CAVEATS = [
    "Intervals concern numerical integration of the declared law, not model accuracy, calibration, or empirical tournament uncertainty.",
    "Intervals are pointwise, not simultaneous across teams or targets, and not anytime-valid.",
    "Exactness does not remove floating-point rounding, score-law binning, finite latent integration, or rule/model assumptions.",
]


def _at(mapping: Any, path: tuple[str, ...], default=None):
    for key in path:
        if not isinstance(mapping, Mapping) or key not in mapping:
            return default
        mapping = mapping[key]
    return mapping


def _leaves(value: Any, path: tuple[str, ...]):
    if isinstance(value, Mapping):
        for key, child in value.items():
            yield from _leaves(child, (*path, str(key)))
    elif isinstance(value, (list, tuple)):
        for rank, child in enumerate(value, 1):
            yield from _leaves(child, (*path, str(rank)))
    else:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError("Marginal probabilities must be numeric")
        p = float(value)
        if not math.isfinite(p) or not 0 <= p <= 1:
            raise ValueError("Marginal probabilities must be finite and in [0,1]")
        yield path, p


def _target_reason(metadata: Mapping, path: tuple[str, ...]) -> str | None:
    """Only explicit integration exclusions disable a defined probability target."""
    unsupported = metadata.get("unsupported_targets") or {}
    if not isinstance(unsupported, Mapping):
        raise ValueError("integration.unsupported_targets must map target paths to reasons")
    for length in range(len(path), 0, -1):
        reason = unsupported.get("/".join(path[:length]))
        if reason is not None:
            if not isinstance(reason, str) or not reason:
                raise ValueError("Unsupported numerical targets require explicit reasons")
            return reason
    return None


def _structural_zero(forecast: Mapping, path: tuple[str, ...]) -> bool:
    law, *keys = path
    zeros = forecast.get("structural_zero_categories") or {}
    if law in ("champion_prob", "final_prob"):
        return keys[0] in (zeros.get(law.removesuffix("_prob")) or [])
    if law == "placement_prob":
        possible = forecast.get("possible_placement_categories")
        return isinstance(possible, Mapping) and keys[0] in possible and keys[1] not in possible[keys[0]]
    if law == "stage_rank_prob":
        possible = _at(forecast.get("stage_rank_possible"), tuple(keys[:-1]))
        return possible is not None and int(keys[-1]) not in possible
    if law == "advance_prob":
        return keys[-1] in (_at(zeros, ("advance", keys[0]), []) or [])
    if law == "joint_final_prob":
        return (keys[0] in (zeros.get("joint_final") or []) or
                bool(set(json.loads(keys[0])).intersection(zeros.get("final") or [])))
    if law == "joint_advance_prob":
        return (keys[-1] in (_at(zeros, ("joint_advance", keys[0]), []) or []) or
                bool(set(json.loads(keys[-1])).intersection(_at(zeros, ("advance", keys[0]), []) or [])))
    return False


def _put(mapping: dict, path: tuple[str, ...], value: dict):
    for key in path[:-1]:
        mapping = mapping.setdefault(key, {})
    mapping[path[-1]] = value


def _design_reason(metadata: Mapping, kind: str) -> str | None:
    if kind == "exact":
        return None
    unit = {"monte_carlo": "independent_complete_path", "hybrid": "independent_prefix"}.get(kind)
    if unit is None:
        return "Integration kind missing or unsupported; no sampling design inferred"
    if metadata.get("effective_sampling_unit") != unit or metadata.get("iid") is not True:
        return "Design is not declared iid; finite-vector, stratified, or dependent sampling needs design-specific inference"
    if metadata.get("stopping_rule") != "fixed_n":
        return "Fixed-N stopping not declared; these intervals are not anytime-valid"
    n = metadata.get("fixed_n")
    if isinstance(n, bool) or not isinstance(n, Integral) or n <= 0:
        return "Positive fixed_n is missing; no sample count inferred"
    return None


def _moments(metadata: Mapping, path: tuple[str, ...], p: float, n: int) -> dict:
    values = _at(metadata.get("prefix_moments"), path)
    if values is None:
        values = (metadata.get("numerical_moments") or {}).get("/".join(path))
    if values is None:
        return {}
    if not isinstance(values, Mapping) or values.get("n") != n or isinstance(values.get("n"), bool):
        raise ValueError("Hybrid moments must have the declared fixed_n")
    try:
        total, squares = float(values["sum"]), float(values["sum_squares"])
    except (KeyError, ValueError, TypeError) as exc:
        raise ValueError("Malformed hybrid moments") from exc
    tolerance = 1e-8 * max(1, n)
    if (not math.isfinite(total) or not math.isfinite(squares) or
            abs(total - n * p) > tolerance or total < 0 or total > n or
            squares < total * total / n - tolerance or squares > total + tolerance):
        raise ValueError("Hybrid moments must describe bounded contributions and the issued mean")
    if n == 1:
        return {"standard_error": None, "standard_error_reason": "At least two independent prefixes required"}
    variance = max(0., (squares - total * total / n) / (n - 1))
    return {"standard_error": math.sqrt(variance / n),
            "standard_error_scope": "sample_moment_estimate_not_a_coverage_guarantee"}


def _numerical_leaf(forecast: Mapping, metadata: Mapping, path: tuple[str, ...],
                    p: float, confidence: float, *, empirical_hits: int | None = None,
                    require_empirical_hits: bool = False) -> dict:
    kind = metadata.get("kind")
    reason = _design_reason(metadata, kind)
    structural = _structural_zero(forecast, path)
    if structural and p != 0:
        raise ValueError("Structural-zero category has positive probability")
    zero_kind = ("structural_zero" if structural else
                 "exact_law_zero" if p == 0 and kind == "exact" else
                 "sampling_zero" if p == 0 and kind == "monte_carlo" and reason is None else
                 "zero_sampled_conditional_mean" if p == 0 and kind == "hybrid" and reason is None else
                 "unclassified_zero" if p == 0 else None)
    leaf = {"estimate": p, "interval": None, "zero_kind": zero_kind}
    target_reason = _target_reason(metadata, path)
    if target_reason is not None:
        leaf.update(method="unsupported", unsupported_reason=target_reason)
    elif structural:
        leaf.update(interval=[0., 0.], method="structural_support", sampling_error=0.)
    elif kind == "exact":
        leaf.update(method="exact_integration", sampling_error=0.)
    elif reason is not None:
        leaf.update(method="unsupported", unsupported_reason=reason)
    elif kind == "monte_carlo":
        if require_empirical_hits and empirical_hits is None:
            leaf.update(method="unsupported",
                        unsupported_reason="Actual empirical joint counts unavailable; no binomial count inferred")
            if p == 0:
                leaf["zero_kind"] = "unclassified_zero"
            return leaf
        n = int(metadata["fixed_n"])
        count = empirical_hits if require_empirical_hits else round(p * n)
        if abs(count - p * n) > 1e-7:
            raise ValueError("IID complete-path empirical probabilities require integer hit counts")
        z = NormalDist().inv_cdf((1 + confidence) / 2)
        denominator = 1 + z * z / n
        center = (p + z * z / (2 * n)) / denominator
        radius = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
        leaf.update(interval=[max(0., center - radius), min(1., center + radius)],
                    method="wilson_fixed_n", hits=count, n=n,
                    standard_error=math.sqrt(p * (1 - p) / n))
        if count == 0:
            leaf["zero_hit_upper_bound"] = -math.expm1(math.log(1 - confidence) / n)
            leaf["zero_hit_bound_method"] = "one_sided_exact_binomial_fixed_n"
    else:
        n = int(metadata["fixed_n"])
        radius = math.sqrt(math.log(2 / (1 - confidence)) / (2 * n))
        leaf.update(interval=[max(0., p - radius), min(1., p + radius)],
                    method="bounded_hoeffding_fixed_n", n=n,
                    observation_unit="per_prefix_conditional_probability")
        leaf.update(_moments(metadata, path, p, n))
    return leaf


def _validate_confidence(confidence: float):
    if isinstance(confidence, bool) or not isinstance(confidence, Real) or not 0 < confidence < 1:
        raise ValueError("confidence must lie strictly between zero and one")


def _integration(forecast: Mapping) -> Mapping:
    metadata = forecast.get("integration") or {}
    if not isinstance(metadata, Mapping):
        raise ValueError("integration must declare structured design metadata")
    return metadata


def _canonical_members(members: Any, teams: Sequence[str], cardinality: int) -> str:
    encoded = members if isinstance(members, str) else None
    if encoded is not None:
        try:
            members = json.loads(encoded)
        except (ValueError, TypeError) as exc:
            raise ValueError("Joint category must be a canonical JSON member array") from exc
    if (not isinstance(members, Sequence) or isinstance(members, (str, bytes)) or
            any(not isinstance(member, str) for member in members)):
        raise ValueError("Joint members must be a sequence of entrant identities")
    if len(members) != cardinality or len(set(members)) != cardinality:
        raise ValueError(f"Joint category requires exactly {cardinality} distinct members")
    if not set(members).issubset(teams):
        raise ValueError("Joint members must be known entrant identities")
    key = json.dumps(sorted(members), separators=(",", ":"), ensure_ascii=False)
    if encoded is not None and encoded != key:
        raise ValueError("Joint category encoding must be canonical")
    return key


def _joint_design(metadata: Mapping, target_design: Mapping | None) -> Mapping:
    design = target_design if target_design is not None else metadata.get("joint_target_design")
    if not isinstance(design, Mapping):
        raise ValueError("Authoritative joint target design is required")
    teams = design.get("teams")
    if (not isinstance(teams, Sequence) or isinstance(teams, (str, bytes)) or not teams or
            any(not isinstance(team, str) or not team for team in teams) or len(set(teams)) != len(teams)):
        raise ValueError("Joint target design requires unique known entrant identities")
    counts = design.get("advance_counts")
    if (not isinstance(counts, Mapping) or
            any(not isinstance(stage, str) or not stage or isinstance(count, bool) or
                not isinstance(count, Integral) or not 0 <= count <= len(teams)
                for stage, count in counts.items())):
        raise ValueError("Joint target design requires declared stage advancement cardinalities")
    finalists = design.get("finalist_count")
    if finalists is not None and (isinstance(finalists, bool) or
                                 not isinstance(finalists, Integral) or finalists != 2 or len(teams) < 2):
        raise ValueError("Joint finalist design requires exactly two members")
    return design


def _joint_context(forecast: Mapping, law: str, stage: str | None,
                   design: Mapping) -> tuple[tuple[str, ...], int, Mapping | None]:
    if law == "joint_final_prob":
        if stage is not None:
            raise ValueError("Finalist queries do not accept a stage")
        if design.get("finalist_count") != 2:
            raise ValueError("Joint finalist target not declared by the design")
        return (law,), 2, forecast.get(law)
    if law != "joint_advance_prob":
        raise ValueError("Joint query law must be joint_final_prob or joint_advance_prob")
    if not isinstance(stage, str) or stage not in design["advance_counts"]:
        raise ValueError("Joint qualifier query requires a declared stage")
    stages = forecast.get(law)
    if stages is not None and not isinstance(stages, Mapping):
        raise ValueError("Joint advancement laws must map declared stages")
    return (law, stage), design["advance_counts"][stage], (stages.get(stage) if stages is not None else None)


def _joint_rows(rows: Mapping | None, design: Mapping, cardinality: int) -> dict | None:
    if rows is None or rows == {}:
        return None
    if not isinstance(rows, Mapping):
        raise ValueError("Joint law must map canonical member sets to probabilities")
    clean = {}
    for key, value in rows.items():
        if not isinstance(key, str):
            raise ValueError("Joint law keys must be canonical JSON member arrays")
        _canonical_members(key, design["teams"], cardinality)
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("Joint probabilities must be finite and in [0,1]")
        clean[key] = float(value)
    if not math.isclose(math.fsum(clean.values()), 1., rel_tol=0., abs_tol=1e-8):
        raise ValueError("Joint categorical law must conserve unit probability mass")
    return clean


def _joint_counts(forecast: Mapping, metadata: Mapping, prefix: tuple[str, ...],
                  rows: Mapping, design: Mapping, cardinality: int) -> Mapping | None:
    if metadata.get("kind") != "monte_carlo" or _design_reason(metadata, "monte_carlo") is not None:
        return None
    path = (prefix[0].removesuffix("_prob"), *prefix[1:])
    counts = _at(forecast.get("hits"), path)
    if counts is None:
        return None
    if not isinstance(counts, Mapping):
        raise ValueError("Joint empirical counts must map canonical member sets")
    n = int(metadata["fixed_n"])
    for key, count in counts.items():
        if not isinstance(key, str):
            raise ValueError("Joint counts keys must be canonical JSON member arrays")
        _canonical_members(key, design["teams"], cardinality)
        if isinstance(count, bool) or not isinstance(count, Integral) or not 0 <= count <= n:
            raise ValueError("Joint empirical counts must be integer hits in [0,fixed_n]")
    if sum(counts.values()) != n or any(
            abs(counts.get(key, 0) / n - rows.get(key, 0.)) > 1e-12
            for key in set(rows).union(counts)):
        raise ValueError("Joint empirical counts must describe the issued law and declared fixed_n")
    return counts


def query_tournament_joint_numerics(forecast: Mapping[str, Any], law: str, members: Sequence[str] | str,
                                   *, stage: str | None = None, confidence: float = .95,
                                   target_design: Mapping | None = None) -> dict[str, Any]:
    """Query one joint event, including a legal category absent from sparse counts.

    ``integration.joint_target_design`` must declare ``teams``, ``finalist_count``
    (two or None), and ``advance_counts`` (stage to total qualifier cardinality).
    ``target_design`` supplies that same authoritative design for retained laws.
    Member sequences are canonicalized; encoded JSON category strings must already
    be canonical sorted arrays. Validity never comes from sparse category keys.
    MC inference requires actual ``hits.joint_final`` / ``hits.joint_advance``.
    Hybrid inference treats each iid prefix's conditional probability as bounded,
    even when its weighted mean looks like an integer binomial count.

    No forecast probability is changed. Bounds are pointwise, numerical-only,
    fixed-design diagnostics, not calibration guarantees or loss replacements.
    """
    _validate_confidence(confidence)
    metadata = _integration(forecast)
    design = _joint_design(metadata, target_design)
    prefix, cardinality, raw = _joint_context(forecast, law, stage, design)
    key = _canonical_members(members, design["teams"], cardinality)
    rows = _joint_rows(raw, design, cardinality)
    if rows is None:
        leaf = {"estimate": None, "interval": None, "zero_kind": None, "method": "unsupported",
                "unsupported_reason": "Joint target law unavailable"}
    else:
        counts = _joint_counts(forecast, metadata, prefix, rows, design, cardinality)
        leaf = _numerical_leaf(forecast, metadata, (*prefix, key), rows.get(key, 0.), confidence,
                               empirical_hits=counts.get(key, 0) if counts is not None else None,
                               require_empirical_hits=True)
    return {**leaf, "law": law, "stage": stage, "category": key,
            "target_path": "/".join((*prefix, key)), "members": json.loads(key),
            "confidence": float(confidence), "coverage_scope": _COVERAGE_SCOPE,
            "effective_sampling_unit": metadata.get("effective_sampling_unit"),
            "caveats": list(_CAVEATS)}


def summarize_tournament_numerics(forecast: Mapping[str, Any], confidence: float = .95) -> dict[str, Any]:
    """Summarize pointwise fixed-N numerical uncertainty for marginal and joint laws.

    ``integration`` declares kind, effective_sampling_unit, iid, fixed_n, and
    stopping_rule='fixed_n'. For hybrid, optional numerical_moments maps slash-
    joined marginal paths to {sum,sum_squares,n}; nested prefix_moments is also
    accepted. Both summarize per-prefix conditional contributions, not weights
    on enumerated suffix paths. Hoeffding intervals remain valid without moments.
    Explicit integration.unsupported_targets maps slash-joined paths (or law
    prefixes) to exclusion reasons. Legacy target_resolved flags describe known
    checkpoint outcomes and do not disable forecasting unknown future outcomes.

    Exact integration has no MC error but retains arithmetic/model caveats.
    Only supplied sparse joint categories are summarized; use
    :func:`query_tournament_joint_numerics` for a specified absent member set.
    No exponential category universe or simultaneous coverage is invented.
    """
    _validate_confidence(confidence)
    metadata = _integration(forecast)
    kind = metadata.get("kind")
    reason = _design_reason(metadata, kind)
    targets, unavailable = {}, {}
    for law in _MARGINAL_LAWS:
        values = forecast.get(law)
        if values is None or values == {}:
            unavailable[law] = "Marginal target law unavailable"
            continue
        for path, p in _leaves(values, (law,)):
            _put(targets, path, _numerical_leaf(forecast, metadata, path, p, confidence))
    design = metadata.get("joint_target_design")
    for law in _JOINT_LAWS:
        values = forecast.get(law)
        if values is None or values == {}:
            unavailable[law] = "Joint target law unavailable"
            continue
        if design is None:
            unavailable[law] = "Authoritative joint target design unavailable; sparse keys do not declare validity"
            continue
        design = _joint_design(metadata, design)
        if law == "joint_advance_prob":
            if not isinstance(values, Mapping) or any(stage not in design["advance_counts"] for stage in values):
                raise ValueError("Joint advancement laws must use declared stages")
            stages = list(design["advance_counts"])
        else:
            stages = [None]
        for stage in stages:
            prefix, cardinality, raw = _joint_context(forecast, law, stage, design)
            rows = _joint_rows(raw, design, cardinality)
            if rows is None:
                unavailable["/".join(prefix)] = "Joint target law unavailable"
                continue
            counts = _joint_counts(forecast, metadata, prefix, rows, design, cardinality)
            for key, p in rows.items():
                _put(targets, (*prefix, key), _numerical_leaf(
                    forecast, metadata, (*prefix, key), p, confidence,
                    empirical_hits=counts.get(key, 0) if counts is not None else None,
                    require_empirical_hits=True))
    return {"integration_kind": kind, "effective_sampling_unit": metadata.get("effective_sampling_unit"),
            "confidence": float(confidence), "coverage_scope": _COVERAGE_SCOPE,
            "sampling_design_supported": reason is None, "unsupported_reason": reason,
            "targets": targets, "unavailable_reasons": unavailable,
            "approximation": metadata.get("approximation"), "caveats": list(_CAVEATS)}
