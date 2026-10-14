"""Isolated W1 support-gated historical study; no serving or original-artifact edits.

CLI stages are inventory, freeze, run and saved-query prediction. Membership and
source provenance freeze before reference-score access. Only the declared nine
trajectories and three terminal scalars may run; any failure permanently closes
the create-once execution journal. All conclusions are development-only.
"""
from __future__ import annotations

from datetime import datetime, timezone
import gc
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.analysis.sports_reliability_common import paired_block_contrast, contrast_draw_arrays
from src.features.champion_oov_data import CHANNELS, ARRAY_KEYS, replay_identity_sources
from src.models.champion_history_runner import partitions, CODE_PATHS

STUDY = "champion_oov_reliability"
MONTHS = [f"2026-{month:02d}" for month in range(1, 10)]
SEEDS = (20261007, 20261008, 20261009)
ARMS = ("H", "C", "HC")
PRIMARY = (("HC", "D0", "natural_champion_oov"), ("HC", "H", "natural_champion_oov"),
           ("HC", "C", "natural_champion_oov"), ("HC", "C0", "all"))
SCENARIO = "actual_first_map_roster_conditioned_daily_release; not authenticated issuance"


def _now():
    return datetime.now(timezone.utc).isoformat()


def _sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _clean(value):
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list, np.ndarray)):
        return [_clean(v) for v in value]
    if isinstance(value, np.generic):
        return _clean(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _json(path, value, *, replace=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Journals and evolving artifact indexes are mutable; scientific outputs are create-only.
    with path.open("w" if replace else "x", encoding="utf-8") as handle:
        json.dump(_clean(value), handle, indent=2, allow_nan=False)
        handle.write("\n")


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def build_identity_state(observations, targets, train_end):
    """Small public tabular adapter; production replay uses compact event counts."""
    required = {"observation_id", "match_id", "date", "release_day", "history_eligible", *CHANNELS}
    if not required.issubset(observations.columns) or not {"match_id", "date"}.issubset(targets.columns):
        raise ValueError("missing source identity observation fields")
    if observations.observation_id.isna().any() or observations.observation_id.duplicated().any():
        raise ValueError("duplicate or missing observation_id")
    if targets.match_id.isna().any() or targets.match_id.duplicated().any():
        raise ValueError("duplicate or missing target match_id")
    obs = observations.copy()
    for key in ("date", "release_day"):
        obs[key] = pd.to_datetime(obs[key], errors="raise").dt.strftime("%Y-%m-%d")
    if (obs.release_day < obs.date).any():
        raise ValueError("release precedes forecast day")
    obs = obs.loc[obs.history_eligible.astype(bool)].sort_values(["date", "match_id", "observation_id"])
    vocabularies = {}
    for channel in CHANNELS:
        ids = sorted(str(v) for v in obs.loc[(obs.date < train_end) & (obs.release_day < train_end), channel].dropna().unique())
        vocabularies[channel] = {identity: index + 2 for index, identity in enumerate(ids)}
    result = {}
    for target in targets.sort_values("match_id").itertuples(index=False):
        day = pd.Timestamp(target.date).strftime("%Y-%m-%d")
        prior = obs.loc[obs.release_day < day]
        result[str(target.match_id)] = {
            channel: {str(identity): int(count) for identity, count in prior[channel].dropna().value_counts().sort_index().items()}
            for channel in CHANNELS}
    return {"vocabularies": vocabularies, "targets": result}


def inventory_history_support(series, arrays, *, canonical_identities, schema):
    mask = np.asarray(arrays["mask"])
    if mask.dtype != bool or mask.ndim != 4 or mask.shape[:3] != (len(series), 2, 5):
        raise ValueError("invalid history token masks")
    if series.match_id.duplicated().any():
        raise ValueError("duplicate canonical fixture identity")
    rows = series.copy().reset_index(drop=True)
    rows["month"] = rows.date.astype(str).str[:7]
    rows["valid_tokens"] = mask.sum(axis=(1, 2, 3))
    rows["cold_player_count"] = (~mask.any(axis=-1)).sum(axis=(1, 2))
    if "unknown_player_count" not in rows:
        rows["unknown_player_count"] = 0
    compact = "canonical_identity_indices" in canonical_identities
    token_totals = {}
    for channel in CHANNELS:
        codes = np.asarray(arrays[ARRAY_KEYS[channel]])
        if codes.shape != mask.shape:
            raise ValueError("identity code axes differ")
        vocabulary = schema[channel + "_vocabulary"]
        if compact:
            ids = np.asarray(canonical_identities["canonical_identity_indices"][channel])
            table = canonical_identities["canonical_id_tables"][channel]
            if ids.shape != mask.shape or ids.dtype.kind not in "iu" or np.any(ids < 0) or np.any(ids >= len(table)):
                raise ValueError("invalid canonical identity indices")
            expected = np.asarray([0 if v is None else vocabulary.get(v, 1) for v in table], np.int64)[ids]
            present = ids != 0
        else:
            if channel not in canonical_identities:
                raise ValueError("missing raw canonical identity channel")
            ids = np.asarray(canonical_identities[channel], object)
            if ids.shape != mask.shape:
                raise ValueError("canonical identity axes differ")
            present = np.fromiter((v is not None and not pd.isna(v) for v in ids.flat), bool, count=ids.size).reshape(ids.shape)
            expected = np.fromiter((0 if not p else vocabulary.get(str(v), 1)
                                    for v, p in zip(ids.flat, present.flat)), np.int64, count=ids.size).reshape(ids.shape)
        if np.any(mask & (codes != expected)):
            raise ValueError(channel + " canonical identity/code mismatch")
        for state, selected in (("known", codes >= 2), ("oov", present & (codes == 1)), ("missing", ~present)):
            rows[channel + "_" + state + "_tokens"] = (mask & selected).sum(axis=(1, 2, 3))
        token_totals[channel] = {split: {
            "observed": int(rows.loc[rows.split == split, "valid_tokens"].sum()),
            **{state: int(rows.loc[rows.split == split, channel + "_" + state + "_tokens"].sum())
               for state in ("known", "oov", "missing")}}
            for split in ("TRAIN", "CAL", "ASSESS")}
    rows["natural_champion_oov"] = rows.champion_oov_tokens > 0
    rows["seen_champion_history"] = (rows.champion_known_tokens > 0) & ~rows.natural_champion_oov
    rows["missing_champion_identity"] = rows.champion_missing_tokens > 0
    rows["cold_history"] = rows.valid_tokens == 0
    rows["patch_oov"] = rows.patch_oov_tokens > 0
    rows["competition_oov"] = rows.competition_oov_tokens > 0
    groups = ("natural_champion_oov", "seen_champion_history", "missing_champion_identity", "cold_history", "patch_oov", "competition_oov")
    universe = sorted(rows.loc[rows.split == "ASSESS", "month"].unique().tolist())

    def counts(split):
        result = {}
        for group in groups:
            subset = rows.loc[(rows.split == split) & rows[group]]
            monthly = subset.month.value_counts().sort_index()
            result[group] = {"series_count": len(subset), "months": monthly.index.tolist(),
                             "per_month": {m: int(monthly.get(m, 0)) for m in universe}}
        return result

    return {"series": rows, "assessment": counts("ASSESS"), "CAL": counts("CAL"), "TRAIN": counts("TRAIN"),
            "month_universe": universe, "token_totals": token_totals,
            "counting_units": {"groups": "unique canonical fixture", "tokens": "observed query-window occurrences", "support": "unique source player-map events"}}


def stage0_support_gate(inventory, component_contract):
    reasons = []
    for group in ("natural_champion_oov", "seen_champion_history"):
        support = inventory.get("assessment", {}).get(group, {})
        if support.get("series_count", 0) < 200:
            reasons.append(group + ": fewer than 200 canonical assessment series")
        if len(support.get("months", [])) < 6:
            reasons.append(group + ": fewer than six nonzero assessment months")
    if component_contract.get("available") is not True:
        reasons.append("component fields unavailable or zero TRAIN numeric support")
    if component_contract.get("valid_masks") is not True:
        reasons.append("component/source validity mask contract failed")
    if component_contract.get("phi_dimension") != 36:
        reasons.append("phi_dimension must equal 36")
    if component_contract.get("embedding_capacity_compatible") is False:
        reasons.append("frozen vocabulary exceeds original TRAIN embedding capacity; hard-stop before fits")
    return {"status": "CLOSED_NO_FITS" if reasons else "SUPPORT_PASS", "reasons": reasons,
            "allowed_neural_fits": 0 if reasons else 9, "allowed_CAL_terminal_scalars": 0 if reasons else 3}


def reliability_paired_comparison(candidate, reference, *, mask=None, confidence=.995, one_sided=False,
                                  universe=None, min_rows=200, min_blocks=6, metric="winner_nll"):
    for key in ("match_id", "date", "best_of"):
        if not np.array_equal(candidate[key].to_numpy(), reference[key].to_numpy()):
            raise ValueError("paired comparison immutable " + key + " alignment mismatch")
    return paired_block_contrast(candidate[metric].to_numpy(float) - reference[metric].to_numpy(float),
        candidate.date.astype(str).str[:7].to_numpy(), mask=mask, universe=MONTHS if universe is None else universe,
        confidence=confidence, one_sided=one_sided, min_rows=min_rows, min_blocks=min_blocks, draws=10000, seed=20261009)


def _context(run_id, repo_root=None, research_root=None, evidence_root=None):
    root = Path(repo_root or Path(__file__).resolve().parents[2]).resolve()
    research = Path(research_root).resolve() if research_root else root
    if not run_id or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in run_id):
        raise ValueError("invalid run_id")
    evidence = Path(evidence_root) if evidence_root else root / "data/08_reporting/sports_reliability_improvements" / run_id
    if not evidence.is_absolute():
        evidence = root / evidence
    evidence = evidence.resolve()
    contract = _read_json(evidence / "w1_contract.json")
    paths = {stage: root / "data" / stage / STUDY / run_id for stage in
             ("04_feature", "05_model_input", "06_models", "07_model_output", "08_reporting")}
    return root, research, evidence, contract, paths


def _resolve(relative, root, research):
    path = Path(relative)
    if path.is_absolute():
        if not path.is_file():
            raise FileNotFoundError(path)
        return path
    for origin in (root, research):
        candidate = origin / path
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(relative)


def _locator(path, root):
    return str(path.relative_to(root)) if path.is_relative_to(root) else str(path)


def _protected(root, research, evidence):
    expected = _read_json(evidence / "authorization_and_initial_pins.json")["protected_before"]
    actual = {key: _sha(_resolve(key, root, research)) for key in expected}
    if actual != expected:
        raise ValueError("protected original bytes changed")
    return actual


def _original_data(root, research, contract):
    prepared = contract["input_paths"]["old_prepared_root"]
    source = _resolve(prepared + "/arrays.npz", root, research)
    with np.load(source, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    series = pd.read_parquet(_resolve(prepared + "/series.parquet", root, research))
    maps = pd.read_parquet(_resolve(prepared + "/maps.parquet", root, research))
    schema = _read_json(_resolve(prepared + "/schema.json", root, research))
    return series, maps, arrays, schema


def _metadata(root, research, contract):
    manifest = _read_json(_resolve(contract["input_paths"]["canonical_manifest_read_only"], root, research))
    path = _resolve(manifest["input"]["path"], root, research)
    if _sha(path) != manifest["input"]["sha256"]:
        raise ValueError("benchmark native metadata pin mismatch")
    metadata = pd.read_parquet(path)
    metadata["match_id"] = metadata.golgg_match_id.astype(str)
    if metadata.match_id.duplicated().any():
        raise ValueError("duplicate benchmark metadata IDs")
    return path, metadata.set_index("match_id", drop=False)


def _taxonomy(series, metadata):
    from src.models.competition_tiers import classify_competition
    output = series.copy()
    output["tournament_source_name"] = output.tournament
    profiles = [classify_competition(name, pd.Timestamp(day).date()) for name, day in zip(output.tournament, output.date)]
    output["profile_tier"] = [profile.tier.value for profile in profiles]
    output["major"] = output.profile_tier.eq("major")
    output["major_state"] = np.where(output.profile_tier.eq("unknown"), "UNKNOWN", np.where(output.major, "major", "nonmajor"))
    if "unknown_players" not in metadata:
        raise ValueError("missing benchmark player source-unseen provenance")
    # Metadata has the frozen benchmark scope, not every TRAIN history-only row.
    unknown = pd.to_numeric(metadata.unknown_players, errors="coerce").reindex(output.match_id.astype(str)).to_numpy(float)
    output["benchmark_unknown_players"] = unknown
    output["player_source_unseen"] = unknown > 0
    output["player_source_state"] = np.where(np.isnan(unknown), "UNKNOWN", np.where(unknown > 0, "unseen", "seen"))
    assess = output.split.eq("ASSESS").to_numpy()
    if np.any(assess & np.isnan(unknown)):
        raise ValueError("missing ASSESS frozen benchmark source-unseen metadata")
    return output


def _reference_scores(series, root, research, contract, split):
    inputs = contract["input_paths"]
    if split in ("CAL", "ASSESS"):
        key = "saved_d0_cal_outputs" if split == "CAL" else "saved_d0_assessment_outputs"
        path = next(value for value in inputs[key] if value.endswith("series_scores.parquet"))
        saved = pd.read_parquet(_resolve(path, root, research))
        if saved.match_id.astype(str).duplicated().any():
            raise ValueError("duplicate D0 reference IDs")
        d0 = saved.assign(match_id=saved.match_id.astype(str)).set_index("match_id").loc[series.match_id.astype(str)].reset_index()
        for field in ("date", "best_of", "winner", "score_index"):
            if not np.array_equal(d0[field].to_numpy(), series[field].to_numpy(), equal_nan=field in ("winner", "score_index")):
                raise ValueError("D0 immutable reference alignment mismatch: " + field)
        for field in ("team_a", "team_b", "roster_a", "roster_b"):
            if field not in d0 or _clean(d0[field].tolist()) != _clean(series[field].tolist()):
                raise ValueError("D0 participant/roster lineage mismatch: " + field)
    else:
        raise ValueError("native C0 TRAIN reference unavailable outside frozen benchmark scope")
    _, metadata = _metadata(root, research, contract)
    saved = pd.read_parquet(_resolve(inputs["native_c0_output"], root, research))
    saved["match_id"] = saved.golgg_match_id.astype(str)
    if saved.match_id.duplicated().any():
        raise ValueError("duplicate native C0 outputs")
    saved = saved.set_index("match_id").loc[series.match_id.astype(str)]
    native = metadata.loc[series.match_id.astype(str)]
    a, b = series.team_a.astype(str).to_numpy(), series.team_b.astype(str).to_numpy()
    one, two = native.team1_id.astype(str).to_numpy(), native.team2_id.astype(str).to_numpy()
    direct, reverse = (one == a) & (two == b), (one == b) & (two == a)
    if not np.all(direct ^ reverse) or not np.array_equal(saved.best_of.to_numpy(int), series.best_of.to_numpy(int)):
        raise ValueError("C0 participant-only alignment mismatch")
    if not np.array_equal(pd.to_datetime(native.date).dt.strftime("%Y-%m-%d").to_numpy(), series.date.astype(str).to_numpy()):
        raise ValueError("C0 date alignment mismatch")
    if not np.array_equal(saved.y.to_numpy(float), native.y.to_numpy(float), equal_nan=True):
        raise ValueError("C0 saved-label/native metadata lineage mismatch")
    expected_y = np.where(reverse, 1 - saved.y.to_numpy(float), saved.y.to_numpy(float))
    if not np.array_equal(expected_y, series.winner.to_numpy(float), equal_nan=True):
        raise ValueError("C0 participant-oriented labels differ from canonical labels")
    probability = saved[inputs["native_c0_column"]].to_numpy(float)
    probability = np.where(reverse, 1 - probability, probability)
    c0 = _winner_scores(series, probability)
    c0["native_participant_reversed"] = reverse
    return d0, c0


def _winner_scores(series, probabilities):
    p = np.asarray(probabilities, float)
    if p.shape != (len(series),) or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("invalid full-universe winner probabilities")
    rows = series.copy().reset_index(drop=True)
    rows["p_win"] = p
    y = rows.winner.to_numpy(float)
    with np.errstate(divide="ignore"):
        rows["winner_nll"] = -np.log(np.where(y == 1, p, 1 - p))
    rows.loc[np.isnan(y), "winner_nll"] = np.nan
    rows["winner_brier"] = (p - y) ** 2
    return rows


def _validate_historical_references(root, research, inputs):
    """Check reused bytes against old receipts, before establishing any W1 pins."""
    scoring = _read_json(_resolve(inputs["old_scoring_binding"], root, research))
    for locator in [*inputs["saved_d0_models"], inputs["saved_d0_ensemble"]]:
        expected = scoring["additional_pins"].get(locator)
        if expected is None or scoring["models_before_scoring"].get(locator) != expected:
            raise ValueError("historical D0 model receipts disagree or lack pin: " + locator)
        if _sha(_resolve(locator, root, research)) != expected:
            raise ValueError("historical D0 model bytes changed: " + locator)
    outputs = _read_json(_resolve(inputs["old_decision"], root, research))["output_hashes"]
    for locator in [*inputs["saved_d0_cal_outputs"], *inputs["saved_d0_assessment_outputs"],
                    *inputs["saved_c0_assessment_outputs"]]:
        expected = outputs.get(locator)
        if expected is None or _sha(_resolve(locator, root, research)) != expected:
            raise ValueError("historical reference output bytes changed or lack pin: " + locator)


def _input_pins(root, research, evidence, contract):
    inputs = contract["input_paths"]
    _validate_historical_references(root, research, inputs)
    paths = set(CODE_PATHS) | {"src/features/champion_oov_data.py", "src/models/champion_oov_reliability.py",
        "src/analysis/champion_oov_reliability.py", "src/analysis/sports_reliability_common.py", "scripts/train_champion_oov.py",
        "src/models/competition_tiers.py", "src/analysis/model_reliability.py", "scripts/verify_champion_oov_reliability.py"}
    for key, value in inputs.items():
        if key in ("old_prepared_root", "strength_bank_root", "old_prepared_files", "strength_bank_files", "native_c0_column"):
            continue
        paths.update(value if isinstance(value, list) else [value])
    for name in inputs["old_prepared_files"]:
        paths.add(inputs["old_prepared_root"] + "/" + name)
    for name in inputs["strength_bank_files"]:
        paths.add(inputs["strength_bank_root"] + "/" + name)
    metadata_path, _ = _metadata(root, research, contract)
    resolved = [_resolve(path, root, research) for path in sorted(paths)]
    resolved.extend(evidence / name for name in ("authorization_and_initial_pins.json", "prefit_design_resolutions.json", "common_contract.json", "oracle_contract.json", "w1_contract.json"))
    resolved.append(metadata_path)
    # Preserve the old source-snapshot recipe; original binding pins are not amended.
    for name in ("old_binding", "old_scoring_binding"):
        binding = _read_json(_resolve(inputs[name], root, research))
        for locator, expected in binding.get("pins", {}).items():
            path = _resolve(locator, root, research)
            if locator == "src/models/champion_history_runner.py":
                change = _read_json(_resolve(inputs["old_scoring_binding"], root, research))["runner_change"]
                snapshot = _resolve(change["original_snapshot"], root, research)
                if _sha(snapshot) != expected or expected != change["original_sha256"]:
                    raise ValueError("original runner snapshot recipe changed")
                if _sha(path) != change["current_sha256"]:
                    raise ValueError("original scoring continuation runner changed")
                resolved.extend((snapshot, path))
                continue
            if _sha(path) != expected:
                raise ValueError("original snapshot closure mismatch " + locator)
            resolved.append(path)
    return {_locator(path, root): _sha(path) for path in resolved}


def _sidecars(paths):
    identity_schema = _read_json(paths["04_feature"] / "identity_schema.json")
    with np.load(paths["04_feature"] / "support_arrays.npz", allow_pickle=False) as archive:
        state = {"n_prior": {channel: archive["n_prior_" + channel] for channel in CHANNELS},
                 "canonical_identity_indices": {channel: archive[channel + "_identity_index"] for channel in CHANNELS},
                 "canonical_id_tables": identity_schema["canonical_id_tables"]}
    return state


def _closure(paths, run_id, status, reasons, *, execution=None):
    comparisons = {f"{a}-{b}:{group}:winner_nll": {"status": "NOT_RUN_BLOCKED" if status in ("CLOSED_BY_SUPPORT", "HARD_STOP_BEFORE_FIT") else "NOT_RUN_FIT_FAILED", "reason": reasons}
                   for a, b, group in PRIMARY}
    if not (paths["08_reporting"] / "comparisons.json").exists():
        _json(paths["08_reporting"] / "comparisons.json", comparisons)
    if not (paths["08_reporting"] / "guards.json").exists():
        guards = {f"HC-{ref}:{group}": {"status": "NOT_RUN", "reason": reasons}
                  for group in ("BO3", "BO5", "major", "player_source_unseen") for ref in ("D0", "C0")}
        guards["HC-D0:seen_champion_history"] = {"status": "NOT_RUN", "reason": reasons}
        _json(paths["08_reporting"] / "guards.json", guards)
    decision = {"run_id": run_id, "status": status, "decision": "NO_ADOPTION", "reasons": reasons,
                "no_production_or_betting_adoption": True, "benchmark_candidate": "NOT_RUN", "reserved_primary_slots": [3, 4, 5, 6],
                "operation_trace": [] if execution is None else execution["operation_trace"], "unused_slots_reallocated": False}
    if execution is not None:
        decision.update({key: execution[key] for key in ("neural_fits", "CAL_terminal_scalars", "seed_epochs")})
        decision["failure_reason"] = execution.get("failure_reason")
        decision["failure_phase"] = execution.get("failure_phase")
    _json(paths["08_reporting"] / "decision.json", decision)
    return decision


def _membership_records(rows):
    definitions = {
        "major": "classify_competition(tournament_source_name,target_day).tier.value == 'major'",
        "player_source_unseen": "frozen benchmark unknown_players > 0; null is UNKNOWN",
        "natural_champion_oov": "any reliable mask-valid raw champion outside closed TRAIN",
        "seen_champion_history": "any reliable TRAIN champion and no natural champion OOV",
        "missing_champion_identity": "any observed null or quarantined unreliable champion",
        "patch_oov": "any mask-valid exact patch identity outside closed TRAIN",
        "competition_oov": "any mask-valid exact competition identity outside closed TRAIN",
        "cold_history": "no mask-valid history token on either side"}
    columns = [c for c in ("row_index", "match_id", "date", "split", "month", "best_of", "history_profile",
               "major_state", "player_source_state", *definitions) if c in rows]
    return {"rows": _clean(rows[columns].to_dict("records")), "definitions": definitions,
            "definition_sha256": hashlib.sha256(json.dumps(definitions, sort_keys=True).encode()).hexdigest(),
            "month_universe": MONTHS, "membership_uses_outcomes": False}


def _freeze_artifacts(run_id, root, research, evidence, contract, paths, rows, gate, pins):
    from src.models.champion_history_model import DEFAULT_CONFIG
    from src.models.champion_oov_reliability import held_identity
    destination = paths["05_model_input"]
    destination.mkdir(parents=True, exist_ok=False)
    membership = _membership_records(rows)
    _json(destination / "membership.json", membership)
    schema = _read_json(_resolve(contract["input_paths"]["old_prepared_root"] + "/schema.json", root, research))
    held = {channel: [identity for identity in schema[channel + "_vocabulary"] if held_identity(channel, identity)]
            for channel in CHANNELS}
    _json(destination / "held_identities.json", held)
    normalizer = _read_json(paths["04_feature"] / "composition_preprocessor.json")
    protocol = {"schema_version": 1, "study": "W1", "run_id": run_id, "created_at_utc": _now(),
        "evidence_root": str(evidence.resolve()),
        "status": "FROZEN_BEFORE_FIT" if gate["status"] == "SUPPORT_PASS" else "FROZEN_CLOSED_BY_SUPPORT",
        "scenario": SCENARIO, "support_gate": gate, "config": dict(DEFAULT_CONFIG), "arms": ARMS, "seeds": SEEDS,
        "budget": {"neural": 9, "seed_epochs": 108, "terminal_CAL": 3, "prop_scales": 0, "C0": 0, "D0": 0},
        "primary_slots": [3, 4, 5, 6], "primary_contrasts": PRIMARY, "family_slots": 10, "confidence": .995,
        "draws": 10000, "seed": 20261009, "month_universe": MONTHS, "blocked_slot_reallocation": False,
        "guard_count": 9, "guard_one_sided_confidence": 1 - .05 / 9, "guard_upper_max": .005,
        "guard_support": {"series": 100, "months": 4}, "normalizer": normalizer, "w1_contract": contract,
        "prefit_resolutions": _read_json(evidence / "prefit_design_resolutions.json"),
        "public_sidecar_contract": "Explicit identity_state aligned to current arrays; model applies identical indices",
        "reference_recipe": "Original binding and source_snapshot runner retained verbatim; new closure pinned separately",
        "oracle_schema": _read_json(evidence / "oracle_contract.json")["artifact_schema"],
        "membership_sha256": _sha(destination / "membership.json")}
    _json(destination / "protocol.json", protocol)
    new_pins = dict(pins)
    for stage in ("04_feature", "05_model_input"):
        for path in sorted(paths[stage].iterdir()):
            if path.is_file():
                new_pins[_locator(path, root)] = _sha(path)
    for filename in ("stage0.json", "stage0_comparisons.json", "integrity.json"):
        path = paths["08_reporting"] / filename
        new_pins[_locator(path, root)] = _sha(path)
    binding = {"run_id": run_id, "created_at_utc": _now(), "protocol_sha256": _sha(destination / "protocol.json"),
               "oracle_contract_sha256": _sha(evidence / "oracle_contract.json"), "pins": new_pins,
               "original_binding_amended": False, "formal_fits": 0}
    _json(destination / "binding.json", binding)
    return protocol, binding


def _case_definitions(rows):
    conditions = {**{f"BO{bo}": rows.best_of.eq(bo) for bo in (1, 3, 5)},
                  **{group: rows[group] for group in ("natural_champion_oov", "seen_champion_history",
                     "missing_champion_identity", "cold_history", "patch_oov", "competition_oov")}}
    conditions["masked_statistics"] = rows.get("masked_statistics", pd.Series(False, index=rows.index))
    conditions["unknown_player"] = rows.unknown_player_count.gt(0)
    conditions["held_identity_history"] = rows.get("held_identity_history", pd.Series(False, index=rows.index))
    for profile in ("full", "partial", "cold"):
        conditions["history_" + profile] = rows.history_profile.eq(profile) if "history_profile" in rows else pd.Series(False, index=rows.index)
    conditions["UNKNOWN_rule"] = rows.draft_mode.eq("UNKNOWN") if "draft_mode" in rows else pd.Series(False, index=rows.index)
    cases = []
    for state, mask in conditions.items():
        candidates = rows.loc[mask].sort_values("match_id")
        cases.append({"case_id": "W1:" + state, "study": "W1", "arm": "HC", "seed_or_ensemble": "ensemble",
                      "contract_state": state, "match_id": None if candidates.empty else str(candidates.iloc[0].match_id),
                      "target_origin": None if candidates.empty else str(candidates.iloc[0].date),
                      "scenario": SCENARIO, "status": "ABSENT_IN_SOURCE" if candidates.empty else "SELECTED_BEFORE_FIT",
                      "expected_status": "ABSENT_IN_SOURCE" if candidates.empty else "LEGAL_LAW_AFTER_COMPLETED_RUN",
                      "expected_reason": "state absent in source" if candidates.empty else None,
                      "input_reference_and_sha256": None, "public_result_reference": None,
                      "independent_expected_reference": None, "comparison_kind": "raw_law_and_evaluator",
                      "tolerances": {"probability_replay": {"atol": 2e-6, "rtol": 0},
                                     "logit_replay": {"atol": 2e-6, "rtol": 2e-6},
                                     "replayed_score_propagation": {"atol": 2e-5, "rtol": 2e-5},
                                     "identical_saved_law_scores_and_bootstrap": {"atol": 1e-10, "rtol": 1e-10}},
                      "state_hash_before": None, "state_hash_after": None,
                      "synthetic_contract_case": candidates.empty})
    return cases[:32]


def _oracle_index(run_id, root, research, evidence, contract, paths, branch, *, execution=None):
    import pyarrow.parquet as pq
    descriptors = []
    prepared = contract["input_paths"]["old_prepared_root"]
    files = [(role, _resolve(prepared + "/" + filename, root, research))
             for role, filename in (("series", "series.parquet"), ("maps", "maps.parquet"), ("arrays", "arrays.npz"), ("schema", "schema.json"))]
    for key in ("saved_d0_models", "saved_d0_ensemble", "saved_d0_cal_outputs", "saved_d0_assessment_outputs",
                "saved_c0_assessment_outputs", "native_c0_output"):
        values = contract["input_paths"][key]
        for value in values if isinstance(values, list) else [values]:
            files.append(("reference_" + key, _resolve(value, root, research)))
    for stage in paths.values():
        if stage.exists():
            for path in sorted(stage.iterdir()):
                if path.is_file() and path.name != "oracle_inputs.json":
                    files.append(("operation_trace" if path.name == "execution.json" else path.stem, path))
    for role, path in files:
        descriptor = {"role": role, "repo_relative_path": _locator(path, root), "sha256": _sha(path),
            "format": path.suffix.lstrip("."), "shape_or_row_count": None, "dtype_or_columns": None,
            "units": "See pinned schema/protocol; probabilities are dimensionless",
            "ordered_axis_names": [], "ordered_id_axis_reference": prepared + "/series.parquet",
            "absence_status": None, "absence_reason": None}
        if path.suffix == ".parquet":
            metadata = pq.read_metadata(path)
            descriptor["shape_or_row_count"] = metadata.num_rows
            descriptor["dtype_or_columns"] = str(metadata.schema)
            descriptor["ordered_axis_names"] = ["record"]
        elif path.suffix == ".npz":
            # Shapes/dtypes are from NPY headers, never duplicate full prepared arrays.
            import zipfile
            with zipfile.ZipFile(path) as archive:
                shapes, dtypes = {}, {}
                for member in archive.namelist():
                    with archive.open(member) as handle:
                        version = np.lib.format.read_magic(handle)
                        if version == (1, 0):
                            shape, _, dtype = np.lib.format.read_array_header_1_0(handle)
                        elif version == (2, 0):
                            shape, _, dtype = np.lib.format.read_array_header_2_0(handle)
                        else:
                            raise ValueError("unsupported NPZ header version")
                    shapes[member[:-4]], dtypes[member[:-4]] = list(shape), str(dtype)
            descriptor["shape_or_row_count"], descriptor["dtype_or_columns"] = shapes, dtypes
            descriptor["ordered_axis_names"] = ["row", "side", "player_slot", "history_slot"]
        elif path.suffix == ".pt":
            import torch
            state = torch.load(path, map_location="cpu", weights_only=False)
            flattened = {}
            def tensors(value, prefix=""):
                if hasattr(value, "shape") and hasattr(value, "dtype"):
                    flattened[prefix] = value
                elif isinstance(value, dict):
                    for key, child in value.items():
                        tensors(child, prefix + "/" + str(key))
                elif isinstance(value, (list, tuple)):
                    for index, child in enumerate(value):
                        tensors(child, prefix + "/" + str(index))
            tensors(state)
            descriptor["shape_or_row_count"] = {key: list(value.shape) for key, value in flattened.items()}
            descriptor["dtype_or_columns"] = {key: str(value.dtype) for key, value in flattened.items()}
        elif path.suffix == ".json":
            value = _read_json(path)
            descriptor["shape_or_row_count"] = len(value)
            descriptor["dtype_or_columns"] = list(value) if isinstance(value, dict) else "JSON records"
        elif path.suffix == ".joblib":
            descriptor["shape_or_row_count"] = 1
            descriptor["dtype_or_columns"] = "serialized fitted model; numerical state separately exported"
        tokens = path.stem.split("_")
        if tokens[0] in ARMS:
            descriptor["arm"] = tokens[0]
            descriptor["split"] = next((s for s in ("TRAIN", "CAL", "ASSESS") if s in tokens), None)
            descriptor["seed"] = next((int(s) for s in tokens if s.isdecimal() and len(s) == 8), None)
        descriptors.append(descriptor)
    produced = {Path(item["repo_relative_path"]).name for item in descriptors}
    required = [(f"{arm}_seed_{seed}{suffix}", role) for arm in ARMS for seed in SEEDS for suffix, role in
                ((".joblib", "model"), ("_initial_state.pt", "initial_state"), ("_final_state.pt", "final_state"),
                 ("_optimizer.pt", "optimizer_state"), ("_fit_receipt.json", "fit_receipt"))]
    required += [(f"{arm}_temperature.json", "temperature") for arm in ARMS]
    required += [(f"{arm}_{split}_{kind}.parquet", kind) for arm in ARMS for split in ("TRAIN", "CAL", "ASSESS")
                 for kind in ("seed_raw_terminal", "seed_tempered_terminal", "terminal", "series_scores")]
    required += [("public_query_inputs.npz", "public_query_inputs"), ("public_predictions.parquet", "public_predictions"),
                 ("public_cases.json", "public_cases")]
    required += [("benchmark_predictions.parquet", "benchmark_predictions")]
    required += [(filename, role) for role, filename in (
        ("source_observations", "source_observations.parquet"), ("identity_tokens", "identity_tokens.parquet"),
        ("support_arrays", "support_arrays.npz"), ("identity_schema", "identity_schema.json"),
        ("composition_preprocessor", "composition_preprocessor.json"), ("stage0_series", "stage0_series.parquet"),
        ("stage0", "stage0.json"), ("stage0_comparisons", "stage0_comparisons.json"),
        ("protocol", "protocol.json"), ("binding", "binding.json"))]
    required += [(f"D0_{split}_{kind}.parquet", "D0_" + kind) for split in ("TRAIN", "CAL", "ASSESS")
                 for kind in ("seed_raw_terminal", "seed_tempered_terminal", "terminal")]
    for filename, role in required:
        if filename not in produced:
            descriptors.append({"role": role, "repo_relative_path": None, "sha256": None, "format": None,
                "shape_or_row_count": None, "dtype_or_columns": None, "units": None, "ordered_axis_names": [],
                "ordered_id_axis_reference": prepared + "/series.parquet", "absence_status": "NOT_RUN",
                "absence_reason": branch + ": candidate artifact not produced", "expected_filename": filename})
    for split in ("TRAIN", "CAL"):
        descriptors.append({"role": "C0_" + split + "_terminal", "repo_relative_path": None, "sha256": None,
            "format": None, "shape_or_row_count": None, "dtype_or_columns": None, "units": None,
            "ordered_axis_names": [], "ordered_id_axis_reference": prepared + "/series.parquet",
            "absence_status": "UNAVAILABLE_ORIGINAL_REFERENCE_SCOPE",
            "absence_reason": "Native C0 is the immutable assessment-only benchmark reference; no C0 refit or invented TRAIN/CAL law"})
    stage0 = _read_json(paths["08_reporting"] / "stage0.json") if (paths["08_reporting"] / "stage0.json").exists() else {}
    rows = pd.read_parquet(paths["04_feature"] / "stage0_series.parquet") if (paths["04_feature"] / "stage0_series.parquet").exists() else None
    index = {"schema_version": 1, "study": "W1", "run_id": run_id, "scenario": SCENARIO, "branch_status": branch,
        "created_at_utc": _now(), "protocol_sha256": _sha(paths["05_model_input"] / "protocol.json") if (paths["05_model_input"] / "protocol.json").exists() else None,
        "binding_sha256": _sha(paths["05_model_input"] / "binding.json") if (paths["05_model_input"] / "binding.json").exists() else None,
        "oracle_contract_sha256": _sha(evidence / "oracle_contract.json"), "artifact_descriptors": descriptors,
        "population_ids": {} if rows is None else {split: rows.loc[rows.split == split, "match_id"].astype(str).tolist() for split in ("TRAIN", "CAL", "ASSESS")},
        "operation_trace": [] if execution is None else execution["operation_trace"],
        "missingness_ledger": stage0.get("TRAIN_missing_identity", {}),
        "public_cases": _read_json(paths["07_model_output"] / "public_cases.json") if (paths["07_model_output"] / "public_cases.json").exists() else ([] if rows is None else _case_definitions(rows)),
        "prefit_resolutions": _read_json(evidence / "prefit_design_resolutions.json"),
        "fitted_prediction_oracle_status": "NOT_RUN_NO_FIT" if branch == "CLOSED_BY_SUPPORT" else "NOT_YET_VERIFIED"}
    _json(paths["08_reporting"] / "oracle_inputs.json", index, replace=(paths["08_reporting"] / "oracle_inputs.json").exists())
    return index


def inventory_run(run_id, *, repo_root=None, research_root=None, evidence_root=None):
    from src.models.champion_oov_reliability import CompositionPreprocessor, held_identity
    root, research, evidence, contract, paths = _context(run_id, repo_root, research_root, evidence_root)
    for stage in ("04_feature", "05_model_input", "06_models", "07_model_output", "08_reporting"):
        if paths[stage].exists():
            raise FileExistsError("W1 output already exists: " + str(paths[stage]))
    before, pins = {}, {}
    paths["04_feature"].mkdir(parents=True, exist_ok=False)
    paths["08_reporting"].mkdir(parents=True, exist_ok=False)
    try:
        before = _protected(root, research, evidence)
        pins = _input_pins(root, research, evidence, contract)
        series, maps, arrays, schema = _original_data(root, research, contract)
        if [len(partitions(series)[s]) for s in ("TRAIN", "CAL", "ASSESS")] != [36205, 1328, 3103]:
            raise ValueError("canonical split counts differ from frozen original")
        state, observations, identity_schema = replay_identity_sources(
            _resolve(contract["input_paths"]["raw_source"], root, research), series, arrays, schema,
            output=paths["04_feature"], roots=(root, research))
        inventory = inventory_history_support(series, arrays, canonical_identities=state, schema=schema)
        _, metadata = _metadata(root, research, contract)
        rows = _taxonomy(inventory.pop("series"), metadata)
        rows["masked_statistics"] = (arrays["mask"][..., None] & ~np.isfinite(arrays["history"][..., :12])).any(axis=(1, 2, 3, 4))
        held_tokens = np.zeros(arrays["mask"].shape, bool)
        for channel in CHANNELS:
            table = state["canonical_id_tables"][channel]
            held_lookup = np.asarray([False if identity is None or identity not in schema[channel + "_vocabulary"]
                                      else held_identity(channel, identity) for identity in table], bool)
            held_tokens |= held_lookup[state["canonical_identity_indices"][channel]] & arrays["mask"]
        rows["held_identity_history"] = held_tokens.any(axis=(1, 2, 3))
        rows.to_parquet(paths["04_feature"] / "stage0_series.parquet", index=False)
        train = partitions(series)["TRAIN"]
        capacity_errors, inherited_sizes = [], {}
        for channel in CHANNELS:
            inherited_size = max(2, int(arrays[ARRAY_KEYS[channel]][train].max()) + 1)
            inherited_sizes[channel] = inherited_size
            if any(code >= inherited_size for code in schema[channel + "_vocabulary"].values()):
                capacity_errors.append(channel + " frozen vocabulary exceeds original TRAIN embedding capacity")
        preprocessor = CompositionPreprocessor.fit(arrays, train, history_feature_names=schema["history_feature_names"])
        normalizer = preprocessor.to_dict()
        normalizer["TRAIN_ids"] = series.iloc[train].match_id.astype(str).tolist()
        # Bounded sufficient-statistic exports never duplicate all TRAIN tensors.
        for key in ("history", "players", "context"):
            width = arrays[key].shape[-1]
            counts = np.zeros(width, np.int64)
            sums, squares = np.zeros(width, np.float64), np.zeros(width, np.float64)
            for start in range(0, len(train), 256):
                batch = train[start:start + 256]
                values = arrays[key][batch].astype(np.float64)
                observed = np.isfinite(values)
                if key == "history":
                    observed &= arrays["mask"][batch][..., None]
                flat = np.where(observed, values, 0).reshape(-1, width)
                counts += observed.reshape(-1, width).sum(0)
                sums += flat.sum(0)
                squares += (flat * flat).sum(0)
            normalizer[key + "_sufficient_statistics"] = {
                "counts": counts.tolist(), "sums": sums.tolist(), "sums_of_squares": squares.tolist(),
                "ddof": 0, "rounding": "(raw-mean).astype(float32), divide scale.astype(float32)"}
        _json(paths["04_feature"] / "composition_preprocessor.json", normalizer)
        numeric_counts = np.asarray(normalizer["history_sufficient_statistics"]["counts"])
        required_indices = [*range(17), 18, 19]
        components = {"available": list(schema["history_feature_names"]) == list(__import__("src.features.champion_history_data", fromlist=["HISTORY_FEATURE_NAMES"]).HISTORY_FEATURE_NAMES)
                      and bool(np.all(numeric_counts[required_indices] > 0))
                      and bool(np.any(arrays["mask"][train] & (arrays["roles"][train] > 0))),
                      "valid_masks": identity_schema["all_slots_validated"], "phi_dimension": 36,
                      "numeric_fields": [schema["history_feature_names"][i] for i in required_indices],
                      "stat_masks": schema["history_feature_names"][:12], "roles": ["TOP", "JUNGLE", "MID", "ADC", "SUPPORT"],
                      "TRAIN_observation_counts": numeric_counts.tolist(), "role_observations": int(np.count_nonzero(arrays["mask"][train] & (arrays["roles"][train] > 0)))}
        components["inherited_embedding_sizes"] = inherited_sizes
        components["embedding_capacity_compatible"] = not capacity_errors
        source_train = observations.loc[observations.history_eligible & (observations.date < "2025-07-01") & (observations.release_day < "2025-07-01")]
        missing = {}
        for channel in CHANNELS:
            support = source_train[channel].value_counts(dropna=False)
            table = pd.DataFrame({"canonical_id": support.index, "unique_source_events": support.to_numpy()})
            table.to_parquet(paths["04_feature"] / (channel + "_TRAIN_support.parquet"), index=False)
            missing[channel] = {"source_events": int(source_train[channel].isna().sum()),
                "target_rows": int((rows.loc[rows.split.eq("TRAIN"), channel + "_missing_tokens"] > 0).sum()),
                "target_tokens": int(rows.loc[rows.split.eq("TRAIN"), channel + "_missing_tokens"].sum()),
                "legacy_missing_embedding_support": "UNTRAINED_NO_OBSERVED_TRAIN_TOKENS" if not rows.loc[rows.split.eq("TRAIN"), channel + "_missing_tokens"].sum() else "OBSERVED_TRAIN_TOKENS_NOT_RELIABILITY_CLAIM"}
        inventory["month_universe"] = MONTHS
        for split in ("assessment", "TRAIN", "CAL"):
            for group in inventory[split].values():
                group["per_month"] = {month: group["per_month"].get(month, 0) for month in MONTHS}
        gate = stage0_support_gate(inventory, components)
        stage0 = dict(inventory, run_id=run_id, created_at_utc=_now(), component_contract=components,
                      support_gate=gate, TRAIN_missing_identity=missing, formal_fits=0, initial_pins=pins,
                      public_cases=_case_definitions(rows), scores_read_after_membership=True)
        _json(paths["08_reporting"] / "stage0.json", stage0)
        # This create-only membership receipt must precede reference probability access.
        _json(paths["04_feature"] / "membership_before_scores.json", _membership_records(rows))
        assess = rows.loc[rows.split.eq("ASSESS")].reset_index(drop=True)
        d0, c0 = _reference_scores(assess, root, research, contract, "ASSESS")
        paths["07_model_output"].mkdir(parents=True, exist_ok=False)
        d0.to_parquet(paths["07_model_output"] / "D0_ASSESS_series_scores.parquet", index=False)
        c0.to_parquet(paths["07_model_output"] / "C0_ASSESS_series_scores.parquet", index=False)
        descriptive = {group: reliability_paired_comparison(d0, c0, mask=assess[group].to_numpy(bool))
                       for group in ("natural_champion_oov", "seen_champion_history")}
        for group in ("natural_champion_oov", "seen_champion_history"):
            _save_draws(paths, "stage0_" + group, d0, c0, assess[group].to_numpy(bool))
        _json(paths["08_reporting"] / "stage0_comparisons.json", {"interpretation": "Descriptive feasibility; no causal or training authorization", "comparisons": descriptive})
        _json(paths["08_reporting"] / "integrity.json", {"protected_before": before, "protected_after": _protected(root, research, evidence)})
        if capacity_errors:
            raise ValueError("; ".join(capacity_errors))
        if gate["status"] != "SUPPORT_PASS":
            _freeze_artifacts(run_id, root, research, evidence, contract, paths, rows, gate, pins)
            execution = {"run_id": run_id, "status": "CLOSED_BY_SUPPORT", "operation_trace": [], "neural_fits": 0,
                         "CAL_terminal_scalars": 0, "seed_epochs": 0, "reserved_neural": 9, "reserved_CAL": 3}
            _json(paths["08_reporting"] / "execution.json", execution)
            _closure(paths, run_id, "CLOSED_BY_SUPPORT", gate["reasons"])
            _oracle_index(run_id, root, research, evidence, contract, paths, "CLOSED_BY_SUPPORT", execution=execution)
        else:
            _oracle_index(run_id, root, research, evidence, contract, paths, "SUPPORT_PASS_PENDING_FREEZE")
        return stage0
    except Exception as error:
        if not (paths["08_reporting"] / "decision.json").exists():
            reasons = [type(error).__name__ + ": " + str(error)]
            _freeze_invalid(run_id, root, evidence, contract, paths, reasons, pins)
            _json(paths["08_reporting"] / "execution.json", {"run_id": run_id, "status": "HARD_STOP_BEFORE_FIT",
                  "operation_trace": [], "neural_fits": 0, "CAL_terminal_scalars": 0, "seed_epochs": 0})
            _closure(paths, run_id, "HARD_STOP_BEFORE_FIT", [type(error).__name__ + ": " + str(error)])
            _oracle_index(run_id, root, research, evidence, contract, paths, "HARD_STOP_BEFORE_FIT")
        raise


def freeze_run(run_id, *, repo_root=None, research_root=None, evidence_root=None):
    root, research, evidence, contract, paths = _context(run_id, repo_root, research_root, evidence_root)
    stage0 = _read_json(paths["08_reporting"] / "stage0.json")
    if stage0["support_gate"]["status"] != "SUPPORT_PASS":
        raise ValueError("Stage0 closed; no fit-authorizing freeze permitted")
    _protected(root, research, evidence)
    for locator, expected in stage0["initial_pins"].items():
        if _sha(_resolve(locator, root, research)) != expected:
            raise ValueError("prefit input changed " + locator)
    rows = pd.read_parquet(paths["04_feature"] / "stage0_series.parquet")
    result = _freeze_artifacts(run_id, root, research, evidence, contract, paths, rows,
                              stage0["support_gate"], stage0["initial_pins"])
    _oracle_index(run_id, root, research, evidence, contract, paths, "FROZEN_BEFORE_FIT")
    return {"protocol": result[0], "binding": result[1]}


def _load_bound(run_id, repo_root=None, research_root=None, evidence_root=None):
    root, research, evidence, contract, paths = _context(run_id, repo_root, research_root, evidence_root)
    protocol = _read_json(paths["05_model_input"] / "protocol.json")
    binding = _read_json(paths["05_model_input"] / "binding.json")
    if _sha(paths["05_model_input"] / "protocol.json") != binding["protocol_sha256"]:
        raise ValueError("new protocol binding changed")
    if evidence != Path(protocol["evidence_root"]).resolve():
        raise ValueError("supplied evidence root differs from frozen authorization root")
    if contract != protocol["w1_contract"]:
        raise ValueError("supplied W1 contract differs from frozen contract")
    contract = protocol["w1_contract"]
    _protected(root, research, evidence)
    for locator, expected in binding["pins"].items():
        if _sha(_resolve(locator, root, research)) != expected:
            raise ValueError("bound source/data/import bytes changed: " + locator)
    series, maps, arrays, schema = _original_data(root, research, contract)
    return root, research, evidence, contract, paths, protocol, series, maps, arrays, schema, _sidecars(paths)


def _terminal_frame(prediction, series, *, arm, temperature, seed=None, weight=1.):
    pmf = prediction.score_probs
    bo = series.best_of.to_numpy(int)
    legal = np.tile(np.arange(3), 2)[None, :] <= bo[:, None] // 2
    if pmf.shape != (len(series), 6) or not np.isfinite(pmf).all() or np.any(pmf < 0) or np.any(pmf[~legal] != 0) or not np.allclose(pmf.sum(1), 1, atol=1e-12, rtol=0):
        raise ValueError("invalid normalized legal terminal law")
    frame = series[["row_index", "match_id", "date", "best_of", "winner", "score_index"]].copy()
    frame["arm"], frame["seed"], frame["weight"], frame["temperature"] = arm, seed, weight, temperature
    frame["p_win"] = prediction.p_win
    for bin_index in range(6):
        frame[f"score_p_{bin_index}"] = pmf[:, bin_index]
        if hasattr(prediction, "score_logits"):
            frame[f"logit_{bin_index}"] = prediction.score_logits[:, bin_index]
    return frame


def _export_arm(arm, ensemble, arrays, state, series, maps, splits, paths):
    from src.analysis.champion_history_evaluation import evaluate_prediction
    from src.models.champion_oov_reliability import assemble_terminal_ensemble
    metrics = {}
    for split, indices in splits.items():
        sub = series.iloc[indices].copy().reset_index(drop=True)
        sub_maps = maps.loc[maps.row_index.isin(indices)].copy().reset_index(drop=True)
        raw = [model.predict(arrays, indices, identity_state=state) for model in ensemble.models]
        prediction = assemble_terminal_ensemble(raw, temperature=ensemble.temperature)
        raw_frames, tempered_frames = [], []
        for seed, member, tempered in zip(SEEDS, raw, prediction.components):
            raw_frames.append(_terminal_frame(member, sub, arm=arm, seed=seed, temperature=1., weight=1 / 3))
            tempered_frames.append(_terminal_frame(tempered, sub, arm=arm, seed=seed, temperature=ensemble.temperature, weight=1 / 3))
        pd.concat(raw_frames, ignore_index=True).to_parquet(paths["07_model_output"] / f"{arm}_{split}_seed_raw_terminal.parquet", index=False)
        pd.concat(tempered_frames, ignore_index=True).to_parquet(paths["07_model_output"] / f"{arm}_{split}_seed_tempered_terminal.parquet", index=False)
        _terminal_frame(prediction, sub, arm=arm, temperature=ensemble.temperature).to_parquet(
            paths["07_model_output"] / f"{arm}_{split}_terminal.parquet", index=False)
        # Inherited evaluator retains every canonical row and explicit task absences.
        metric, scored, map_scores, profiles = evaluate_prediction(prediction, sub, sub_maps)
        scored.to_parquet(paths["07_model_output"] / f"{arm}_{split}_series_scores.parquet", index=False)
        map_scores.to_parquet(paths["07_model_output"] / f"{arm}_{split}_map_scores.parquet", index=False)
        profiles.to_parquet(paths["07_model_output"] / f"{arm}_{split}_profiles.parquet", index=False)
        metrics[split] = metric
        del raw, prediction, raw_frames, tempered_frames
    return metrics


def _save_draws(paths, name, candidate, reference, mask, *, blocks=None, universe=None):
    block_labels = candidate.date.astype(str).str[:7].to_numpy() if blocks is None else blocks
    deltas = candidate.winner_nll.to_numpy(float) - reference.winner_nll.to_numpy(float)
    draws = contrast_draw_arrays(deltas, block_labels, mask=mask, universe=MONTHS if universe is None else universe,
                                 draws=10000, seed=20261009)
    arrays = {key: value for key, value in draws.items() if isinstance(value, np.ndarray)}
    arrays.update(match_ids=candidate.match_id.astype(str).to_numpy(dtype=str),
                  block_labels=np.asarray(block_labels, dtype=str), selected=np.asarray(mask, bool),
                  candidate_scores=candidate.winner_nll.to_numpy(float), reference_scores=reference.winner_nll.to_numpy(float),
                  differences=deltas, candidate_missing=candidate.winner_nll.isna().to_numpy(),
                  reference_missing=reference.winner_nll.isna().to_numpy())
    arrays["universe"] = np.asarray(draws["universe"], dtype=str)
    arrays["numpy_version"] = np.asarray(np.__version__)
    np.savez_compressed(paths["08_reporting"] / (name + "_draws.npz"), **arrays)


def _complete_scores(root, research, contract, paths, series, splits):
    assess = pd.read_parquet(paths["04_feature"] / "stage0_series.parquet").iloc[splits["ASSESS"]].reset_index(drop=True)
    scores = {arm: pd.read_parquet(paths["07_model_output"] / f"{arm}_ASSESS_series_scores.parquet") for arm in ARMS}
    scores["D0"], scores["C0"] = _reference_scores(assess, root, research, contract, "ASSESS")
    for reference in ("D0", "C0"):
        destination = paths["07_model_output"] / f"{reference}_ASSESS_series_scores.parquet"
        # Stage0 originals are retained byte-for-byte; never rewrite them.
        if not destination.exists():
            scores[reference].to_parquet(destination, index=False)
    comparisons, brier, proxy = {}, {}, {}
    block_labels = (assess.tournament_source_name.fillna("UNKNOWN").astype(str) + "|" + assess.date.astype(str).str[:4]).to_numpy()
    universe = sorted(set(block_labels))
    for candidate, reference, group in PRIMARY:
        key = f"{candidate}-{reference}:{group}:winner_nll"
        mask = np.ones(len(assess), bool) if group == "all" else assess[group].to_numpy(bool)
        contrast = reliability_paired_comparison(scores[candidate], scores[reference], mask=mask)
        comparisons[key] = contrast
        _save_draws(paths, f"primary_{candidate}_{reference}_{group}", scores[candidate], scores[reference], mask)
        delta = scores[candidate].winner_brier.to_numpy(float) - scores[reference].winner_brier.to_numpy(float)
        brier[key] = {"point": float(delta[mask].mean()), "nonworse": bool(np.isfinite(delta[mask]).all() and delta[mask].mean() <= 0)}
        differences = scores[candidate].winner_nll.to_numpy(float) - scores[reference].winner_nll.to_numpy(float)
        proxy[key] = paired_block_contrast(differences, block_labels, mask=mask, universe=universe,
            confidence=.95, draws=10000, seed=20261009, min_rows=200, min_blocks=6)
        proxy[key]["interpretation"] = "source-name/year tournament proxy sensitivity only; not a causal independent cluster"
        _save_draws(paths, f"proxy_{candidate}_{reference}_{group}", scores[candidate], scores[reference],
                    mask, blocks=block_labels, universe=universe)
    guards = {}
    for group in ("BO3", "BO5", "major", "player_source_unseen", "seen_champion_history"):
        mask = assess.best_of.eq(int(group[2:])).to_numpy() if group.startswith("BO") else assess[group].to_numpy(bool)
        references = ("D0",) if group == "seen_champion_history" else ("D0", "C0")
        for reference in references:
            contrast = reliability_paired_comparison(scores["HC"], scores[reference], mask=mask,
                confidence=1 - .05 / 9, one_sided=True, min_rows=100, min_blocks=4)
            contrast["pass"] = contrast["status"] == "ESTIMABLE" and contrast["upper"] is not None and contrast["upper"] <= .005
            if group == "player_source_unseen":
                contrast["unknown_taxonomy_rows"] = int(assess.player_source_state.eq("UNKNOWN").sum())
            if group == "major":
                contrast["unknown_taxonomy_rows"] = int(assess.major_state.eq("UNKNOWN").sum())
            guards[f"HC-{reference}:{group}"] = contrast
            _save_draws(paths, f"guard_HC_{reference}_{group}", scores["HC"], scores[reference], mask)
    diagnostics = {}
    for group in ("cold_history", "missing_champion_identity", "patch_oov", "competition_oov", "masked_statistics"):
        mask = assess[group].to_numpy(bool)
        diagnostics[group] = {reference: reliability_paired_comparison(scores["HC"], scores[reference], mask=mask)
                              for reference in ("D0", "C0")}
    whole_brier = scores["HC"].winner_brier.to_numpy(float) - scores["C0"].winner_brier.to_numpy(float)
    whole_nonworse = bool(np.isfinite(whole_brier).all() and whole_brier.mean() <= 0)
    passed = all(item["status"] == "ESTIMABLE" and item["upper"] is not None and item["upper"] < 0
                 and item["equal_block"]["point"] is not None and item["equal_block"]["point"] <= 0
                 for item in comparisons.values())
    passed = passed and all(item["nonworse"] for item in brier.values()) and whole_nonworse and all(item["pass"] for item in guards.values())
    _json(paths["08_reporting"] / "comparisons.json", comparisons)
    _json(paths["08_reporting"] / "guards.json", guards)
    _json(paths["08_reporting"] / "diagnostics.json", diagnostics)
    _json(paths["08_reporting"] / "proxy_sensitivity.json", proxy)
    _json(paths["08_reporting"] / "brier_guards.json", {"primary": brier, "whole_HC_C0": {"point": float(whole_brier.mean()), "nonworse": whole_nonworse}})
    benchmark = assess[["match_id", "date", "best_of"]].copy()
    benchmark["y"] = assess.winner
    for arm in ("HC", "C0", "D0"):
        if not np.array_equal(scores[arm].match_id.astype(str), assess.match_id.astype(str)):
            raise ValueError("benchmark immutable full row coverage mismatch")
        benchmark["p_" + arm] = scores[arm].p_win.to_numpy(float)
    benchmark.to_parquet(paths["07_model_output"] / "benchmark_predictions.parquet", index=False)
    _json(paths["07_model_output"] / "benchmark_lineage.json", {
        "scenario": SCENARIO, "n_rows": len(benchmark), "warmup_rows_invented": 0,
        "generation": "saved three-member terminal-temperature HC ensemble; native C0 and unchanged saved D0",
        "protocol_sha256": _sha(paths["05_model_input"] / "protocol.json"),
        "execution_sha256_at_generation": _sha(paths["08_reporting"] / "execution.json"),
        "participant_alignment": "native participant IDs only; no label-selected orientation",
        "fit_lineage": {arm: _sha(paths["06_models"] / f"{arm}_ensemble.joblib") for arm in ARMS},
        "rescoring_scope": "DESCRIPTIVE existing common engine; frozen W1 primary and nine guards remain authoritative"})
    return passed


def _publish_completed(run_id, root, research, evidence, paths, execution, passed):
    decision = {"run_id": run_id, "status": "DEVELOPMENT_CANDIDATE" if passed else "NO_ADOPTION",
                "decision": "DEVELOPMENT_CANDIDATE" if passed else "NO_ADOPTION", "gates_passed": passed,
                "neural_fits": execution["neural_fits"], "CAL_terminal_scalars": execution["CAL_terminal_scalars"],
                "seed_epochs": execution["seed_epochs"],
                "evidence_status": "DEVELOPMENT_ONLY", "no_production_or_betting_adoption": True,
                "original_negative_evidence_preserved": True,
                "output_pins": {_locator(p, root): _sha(p) for stage in ("06_models", "07_model_output")
                                for p in sorted(paths[stage].iterdir()) if p.is_file()}}
    # No completed journal or decision may precede the final protected-byte check.
    protected_after = _protected(root, research, evidence)
    _json(paths["08_reporting"] / "final_integrity.json", {"protected_after": protected_after})
    execution.update(status="COMPLETED", completed_at_utc=_now())
    _json(paths["08_reporting"] / "execution.json", execution, replace=True)
    _json(paths["08_reporting"] / "decision.json", decision)
    return decision


def _record_failure(paths, run_id, execution, current, error, phase):
    reason = type(error).__name__ + ": " + str(error)
    if current is not None and current["status"] == "ATTEMPTED":
        current.update(status="FAILED", error=reason, failure_reason=reason,
                       failed_at_utc=_now(), finished_at_utc=_now())
    execution.update(status="FIT_FAILED_WITHIN_BUDGET", stopped_at_utc=_now(),
                     failure_reason=reason, failure_phase=phase)
    _json(paths["08_reporting"] / "execution.json", execution, replace=True)
    if not (paths["08_reporting"] / "decision.json").exists():
        _closure(paths, run_id, "FIT_FAILED_WITHIN_BUDGET", [reason], execution=execution)


def run_experiment(run_id, *, repo_root=None, research_root=None, evidence_root=None):
    import torch
    from src.models.champion_oov_reliability import fit_oov_model, calibrate_oov_ensemble
    (root, research, evidence, contract, paths, protocol, series, maps, arrays, schema, state) = _load_bound(
        run_id, repo_root, research_root, evidence_root)
    if protocol["support_gate"]["status"] != "SUPPORT_PASS" or protocol["status"] != "FROZEN_BEFORE_FIT":
        raise ValueError("frozen support authorization required")
    execution_path = paths["08_reporting"] / "execution.json"
    if execution_path.exists() or paths["06_models"].exists():
        raise FileExistsError("execution already attempted; no automatic restart")
    paths["06_models"].mkdir(parents=True, exist_ok=False)
    paths["07_model_output"].mkdir(parents=True, exist_ok=True)
    splits = partitions(series)
    execution = {"run_id": run_id, "started_at_utc": _now(), "status": "RUNNING", "operation_trace": [],
                 "neural_fits": 0, "CAL_terminal_scalars": 0, "seed_epochs": 0, "C0_fits": 0, "D0_fits": 0,
                 "prop_scale_fits": 0, "automatic_restart": False}
    _json(execution_path, execution)
    metrics, calibrations = {}, {}
    current = None
    phase = "FIT_OR_EXPORT"
    try:
        for arm in ARMS:
            models = []
            for seed in SEEDS:
                current = _attempt_record(execution, "W1_NEURAL_FIT", arm, seed, "TRAIN", 9,
                                          execution["neural_fits"], protocol["config"], series.iloc[splits["TRAIN"]].match_id)
                current["epochs"] = 12
                execution["operation_trace"].append(current)
                execution["neural_fits"] += 1
                _json(execution_path, execution, replace=True)
                model = fit_oov_model(arrays, series, maps, splits["TRAIN"], seed=seed, arm=arm, schema=schema,
                                      identity_state=state, config=protocol["config"])
                # Completed training progress survives any subsequent export failure.
                current["fit_receipt"] = model.fit_receipt
                execution["seed_epochs"] += int(model.fit_receipt["epochs_completed"])
                _json(execution_path, execution, replace=True)
                destination = paths["06_models"] / f"{arm}_seed_{seed}"
                torch.save(model.initial_state, str(destination) + "_initial_state.pt")
                torch.save(model.final_state, str(destination) + "_final_state.pt")
                torch.save(model.optimizer_state, str(destination) + "_optimizer.pt")
                _json(str(destination) + "_fit_receipt.json", model.fit_receipt)
                # State/optimizer exports are standalone; don't replicate optimizer in joblibs.
                del model.optimizer_state
                joblib.dump(model, str(destination) + ".joblib")
                current.update(status="COMPLETED", completed_at_utc=_now(), finished_at_utc=_now(),
                               iterations=12, function_evaluations=model.fit_receipt["optimizer_steps"], fit_receipt=model.fit_receipt)
                _json(execution_path, execution, replace=True)
                models.append(model)
            current = _attempt_record(execution, "W1_TERMINAL_CAL", arm, None, "CAL", 3,
                                      execution["CAL_terminal_scalars"], {"bounds": [.25, 4.], "penalty": .001},
                                      series.iloc[splits["CAL"]].match_id)
            execution["operation_trace"].append(current)
            execution["CAL_terminal_scalars"] += 1
            _json(execution_path, execution, replace=True)
            ensemble, receipt = calibrate_oov_ensemble(models, arrays, series, splits["CAL"], identity_state=state)
            joblib.dump(ensemble, paths["06_models"] / f"{arm}_ensemble.joblib")
            _json(paths["06_models"] / f"{arm}_temperature.json", receipt)
            calibrations[arm] = receipt
            current.update(status="COMPLETED", completed_at_utc=_now(), finished_at_utc=_now(), receipt=receipt,
                           iterations=receipt["score_temperature"].get("nit"),
                           function_evaluations=receipt["score_temperature"].get("nfev"))
            _json(execution_path, execution, replace=True)
            metrics[arm] = _export_arm(arm, ensemble, arrays, state, series, maps,
                                       {split: splits[split] for split in ("TRAIN", "CAL")}, paths)
            del model, models, ensemble
            gc.collect()
        if (execution["neural_fits"], execution["CAL_terminal_scalars"], execution["seed_epochs"]) != (9, 3, 108):
            raise ValueError("frozen operation budget accounting mismatch")
        # ASSESS is read once only after all nine trajectories and three scalars complete.
        for arm in ARMS:
            ensemble = joblib.load(paths["06_models"] / f"{arm}_ensemble.joblib")
            metrics[arm].update(_export_arm(arm, ensemble, arrays, state, series, maps,
                                          {"ASSESS": splits["ASSESS"]}, paths))
            del ensemble
            gc.collect()
        _export_d0(root, research, contract, paths, arrays, series, splits)
        _save_public_cases(paths, arrays, state, series)
        _json(paths["08_reporting"] / "calibration_receipts.json", calibrations)
        _json(paths["08_reporting"] / "metrics.json", metrics)
        passed = _complete_scores(root, research, contract, paths, series, splits)
        phase = "FINAL_INTEGRITY"
        decision = _publish_completed(run_id, root, research, evidence, paths, execution, passed)
        _oracle_index(run_id, root, research, evidence, contract, paths, decision["status"], execution=execution)
        return decision
    except Exception as error:
        _record_failure(paths, run_id, execution, current, error, phase)
        _oracle_index(run_id, root, research, evidence, contract, paths, "FIT_FAILED_WITHIN_BUDGET", execution=execution)
        raise


def predict_saved(run_id, match_ids, arm="HC", *, repo_root=None, research_root=None, evidence_root=None):
    if arm not in ARMS:
        raise ValueError("prediction arm must be H/C/HC")
    ids = list(map(str, match_ids))
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("match IDs must be nonempty and unique")
    (root, research, evidence, contract, paths, protocol, series, maps, arrays, schema, state) = _load_bound(
        run_id, repo_root, research_root, evidence_root)
    execution = _read_json(paths["08_reporting"] / "execution.json")
    if execution["status"] != "COMPLETED":
        raise ValueError("public prediction requires completed bound execution")
    decision = _read_json(paths["08_reporting"] / "decision.json")
    for locator, expected in decision["output_pins"].items():
        if _sha(_resolve(locator, root, research)) != expected:
            raise ValueError("saved query model/output hash mismatch")
    positions = {str(identity): index for index, identity in enumerate(series.match_id)}
    if any(identity not in positions for identity in ids):
        raise ValueError("requested match ID absent from frozen canonical histories")
    indices = np.asarray([positions[identity] for identity in ids], np.int64)
    query_arrays = {key: value[indices] for key, value in arrays.items()}
    query_state = {"n_prior": {channel: state["n_prior"][channel][indices] for channel in CHANNELS}}
    model = joblib.load(paths["06_models"] / f"{arm}_ensemble.joblib")
    prediction = model.predict(query_arrays, identity_state=query_state)
    return {"run_id": run_id, "arm": arm, "scenario": SCENARIO, "predictions": [
        {"match_id": identity, "best_of": int(series.best_of.iloc[index]), "p_win": float(prediction.p_win[i]),
         "terminal_probabilities": prediction.score_probs[i].tolist(), "target_origin": str(series.date.iloc[index])}
        for i, (identity, index) in enumerate(zip(ids, indices))]}


def _attempt_record(execution, kind, arm, seed, partition, ceiling, before, parameters, input_ids):
    now = _now()
    return {"operation_id": f"W1:{len(execution['operation_trace']) + 1}", "study": "W1",
            "operation": kind, "kind": kind, "arm": arm, "seed": seed, "partition": partition,
            "authorized_ceiling": ceiling, "counter_before": before, "counter_after": before + 1,
            "attempted_at_utc": now, "started_at_utc": now, "finished_at_utc": None, "status": "ATTEMPTED",
            "parameters_sha256": hashlib.sha256(json.dumps(parameters, sort_keys=True).encode()).hexdigest(),
            "input_ids_sha256": hashlib.sha256(json.dumps(list(map(str, input_ids))).encode()).hexdigest(),
            "iterations": None, "function_evaluations": None, "failure_reason": None}


def _freeze_invalid(run_id, root, evidence, contract, paths, reasons, pins):
    """Bind a terminal invalidity closure, never a fit-authorizing fallback."""
    destination = paths["05_model_input"]
    destination.mkdir(parents=True, exist_ok=True)
    if not (destination / "protocol.json").exists():
        _json(destination / "protocol.json", {"run_id": run_id, "study": "W1", "created_at_utc": _now(),
            "evidence_root": str(evidence.resolve()),
            "status": "FROZEN_INVALID_NO_FITS", "scenario": SCENARIO, "reasons": reasons,
            "fit_authorization": False, "allowed_neural_fits": 0, "allowed_CAL_terminal_scalars": 0,
            "reserved_primary_slots": [3, 4, 5, 6], "w1_contract": contract,
            "prefit_resolutions": _read_json(evidence / "prefit_design_resolutions.json")})
    if not (destination / "binding.json").exists():
        _json(destination / "binding.json", {"run_id": run_id, "created_at_utc": _now(),
            "protocol_sha256": _sha(destination / "protocol.json"),
            "oracle_contract_sha256": _sha(evidence / "oracle_contract.json"), "pins": pins,
            "original_binding_amended": False, "fit_authorization": False, "invalidity_reasons": reasons})


def _export_d0(root, research, contract, paths, arrays, series, splits):
    """Replay unchanged saved D0, without fitting or reusing its prior decision."""
    from src.models.champion_history_model import _temper
    model = joblib.load(_resolve(contract["input_paths"]["saved_d0_ensemble"], root, research))
    for split, indices in splits.items():
        sub = series.iloc[indices].reset_index(drop=True)
        raw = [member.predict(arrays, indices) for member in model.models]
        frames = [_terminal_frame(member, sub, arm="D0", seed=seed, temperature=1., weight=1 / 3)
                  for seed, member in zip(SEEDS, raw)]
        pd.concat(frames, ignore_index=True).to_parquet(paths["07_model_output"] / f"D0_{split}_seed_raw_terminal.parquet", index=False)
        tempered = []
        for frame, member in zip(frames, raw):
            frame = frame.copy()
            law = _temper(member.score_probs, model.score_temperature)
            for bin_index in range(6):
                frame[f"score_p_{bin_index}"] = law[:, bin_index]
            frame["p_win"], frame["temperature"] = law[:, :3].sum(1), model.score_temperature
            tempered.append(frame)
        pd.concat(tempered, ignore_index=True).to_parquet(paths["07_model_output"] / f"D0_{split}_seed_tempered_terminal.parquet", index=False)
        from src.models.champion_history_model import HistoryPrediction
        law = np.stack([frame[[f"score_p_{i}" for i in range(6)]].to_numpy() for frame in tempered]).mean(0)
        prediction = HistoryPrediction(law, sub.best_of.to_numpy(int))
        _terminal_frame(prediction, sub, arm="D0", temperature=model.score_temperature).to_parquet(
            paths["07_model_output"] / f"D0_{split}_terminal.parquet", index=False)
        if split != "ASSESS":
            _winner_scores(sub, prediction.p_win).to_parquet(paths["07_model_output"] / f"D0_{split}_series_scores.parquet", index=False)
    del model


def _save_public_cases(paths, arrays, state, series):
    """Export actual saved-adapter API laws; independent expected values are external."""
    from src.models.champion_history_model import _parameter_digest
    rows = pd.read_parquet(paths["04_feature"] / "stage0_series.parquet")
    cases = _case_definitions(rows)
    selected = sorted({case["match_id"] for case in cases if case["match_id"] is not None})[:32]
    positions = {str(match): index for index, match in enumerate(series.match_id)}
    indices = np.asarray([positions[match] for match in selected], np.int64)
    query = {key: value[indices] for key, value in arrays.items()}
    support = {"n_prior": {channel: state["n_prior"][channel][indices] for channel in CHANNELS}}
    input_path = paths["07_model_output"] / "public_query_inputs.npz"
    np.savez_compressed(input_path, **query, **{"n_prior_" + channel: support["n_prior"][channel] for channel in CHANNELS})
    sub = series.iloc[indices].reset_index(drop=True)
    frames, hashes = [], {}
    for arm in ARMS:
        ensemble = joblib.load(paths["06_models"] / f"{arm}_ensemble.joblib")
        hashes[arm] = {}
        for seed, model in zip(SEEDS, ensemble.models):
            before = _parameter_digest(model.network)
            prediction = model.predict(query, identity_state=support)
            after = _parameter_digest(model.network)
            frame = _terminal_frame(prediction, sub, arm=arm, seed=seed, temperature=1.)
            frame["public_kind"], frame["state_hash_before"], frame["state_hash_after"] = "member", before, after
            frames.append(frame)
            hashes[arm][str(seed)] = {"before": before, "after": after}
        prediction = ensemble.predict(query, identity_state=support)
        frame = _terminal_frame(prediction, sub, arm=arm, temperature=ensemble.temperature)
        frame["public_kind"] = "ensemble"
        frames.append(frame)
        del ensemble
    output = paths["07_model_output"] / "public_predictions.parquet"
    pd.concat(frames, ignore_index=True).to_parquet(output, index=False)
    for case in cases:
        case["input_reference_and_sha256"] = {"path": str(input_path), "sha256": _sha(input_path), "ordered_match_ids": selected}
        if case["match_id"] is not None:
            case["public_result_reference"] = {"path": str(output), "sha256": _sha(output),
                                               "match_id": case["match_id"], "arm": "HC"}
            case["state_hash_before"] = {seed: value["before"] for seed, value in hashes["HC"].items()}
            case["state_hash_after"] = {seed: value["after"] for seed, value in hashes["HC"].items()}
            case["expected_reason"] = "Independent expected law to be reconstructed by frozen verifier; not supplied by production"
        else:
            case["expected_reason"] = "ABSENT_IN_SOURCE; synthetic contract case belongs to independent verifier and cannot establish population support"
    _json(paths["07_model_output"] / "public_cases.json", cases)
