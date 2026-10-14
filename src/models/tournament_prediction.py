"""Explicit local all-pair inference; no default model, market, or live state.

Prepared bundles contain every hypothetical pair from the shared chronological-v2
training/export replay. ``chronological`` accepts only separately trained v2 folds,
not frozen/old corrected weights. Availability certification is an independent
gate: semantically matched research inputs are not captured historical forecasts.
"""
from __future__ import annotations

from collections import Counter
import hashlib
from datetime import date, datetime, time, timedelta, timezone
from itertools import combinations
import json
import math
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np

from src.models.siamese_series import SiameseSeriesModel
from src.models.symmetric_series import _PROBABILITY_FIELDS, _REQUIRED_BASE_FIELDS
from src.models.team_order import pair_columns

BUNDLE_VERSION = 'exp081-tournament-pairs-v2'
COMPATIBILITY_GAPS = [
    'old corrected folds use first-target-map rosters; chronological-v2 training uses prior-observed rosters and inference requires explicit prior-known player IDs',
    'old corrected replay batches series lexically on start day; chronological-v2 releases complete series in numeric-ID order after completion day',
    'old corrected replay decays daily participants before snapshots; chronological-v2 shares nonmutating cutoff decay and actual per-map roster updates between training and export',
    'old fold weights/scalers/calibrators have never been fitted on chronological-v2 rating/W20 snapshots; the shared canonical79 algebra does not make those weights compatible',
]


def json_digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def file_digest(path) -> str:
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def utc(value: str) -> datetime:
    if not isinstance(value, str):
        raise ValueError('timestamp must be an explicit timezone-aware ISO string')
    try:
        result = datetime.fromisoformat(value.replace('Z', '+00:00'))
    except ValueError as exc:
        raise ValueError('timestamp must be an explicit timezone-aware ISO string') from exc
    if result.tzinfo is None:
        raise ValueError('timestamp must include its timezone')
    return result.astimezone(timezone.utc)


def _formats(best_ofs) -> list[int]:
    values = list(best_ofs)
    if not values or any(type(value) is not int or value not in (1, 3, 5) for value in values):
        raise ValueError('best_ofs must contain only integer 1, 3, 5')
    return sorted(set(values))


def _teams(teams) -> list[str]:
    values = list(teams)
    if len(values) < 2 or any(not isinstance(t, str) or not t or t != t.strip() for t in values) or len(set(values)) != len(values):
        raise ValueError('teams require at least two unique exact string IDs')
    return values


def previous_observed_rosters(matches: Iterable[dict], teams: list[str], cutoff: date) -> tuple[dict, dict]:
    """Explicit diagnostic proxy from last strictly prior map, never event maps."""
    from scripts.prospective_sports_features import _game_days, _game_roster, _order, _series_team_ids
    wanted = set(_teams(teams))
    latest = {}
    for row in matches:
        day = date.fromisoformat(row['date'])
        if day >= cutoff:
            continue
        try:
            days = _game_days(row, day)
        except ValueError as exc:
            raise ValueError(f"historical roster series {row.get('match_id')!r} on {day}: {exc}") from exc
        if days[-1] >= cutoff:
            continue
        _series_team_ids(row)
        for index, (game, game_day) in enumerate(zip(row['games'], days, strict=True)):
            key = (game_day, _order(str(row['match_id'])), index)
            for side in ('t1', 't2'):
                team = str(game[side + '_id'])
                if team in wanted and (team not in latest or key > latest[team][0]):
                    latest[team] = (key, _game_roster(game, side), str(row['match_id']))
    if wanted != set(latest):
        raise ValueError(f'no prior observed roster for {sorted(wanted - set(latest))}')
    return ({t: latest[t][1] for t in sorted(wanted)},
            {t: {'date': latest[t][0][0].isoformat(), 'match_id': latest[t][2],
                 'game_index': latest[t][0][2]} for t in sorted(wanted)})


def _pair_bundle(frame, audit, rosters, at, *, policy, proxy, input_hashes, roster_provenance):
    from scripts.prospective_sports_features import FEATURE_VERSION, FEATURE_CONTRACT, NATIVE_SUPPLEMENT
    return {
        'version': BUNDLE_VERSION, 'cutoff': at.isoformat(), 'feature_version': FEATURE_VERSION,
        'feature_contract': dict(FEATURE_CONTRACT), 'feature_contract_sha256': json_digest(FEATURE_CONTRACT),
        'native_supplement': dict(NATIVE_SUPPLEMENT),
        'rows': frame.to_dict('records'), 'audit': audit,
        'rosters': {team: list(players) for team, players in rosters.items()},
        'roster_policy': policy, 'roster_provenance': roster_provenance or {},
        'previous_roster_observations': proxy, 'input_hashes': input_hashes or {},
        'prepared_at': datetime.now(timezone.utc).isoformat(),
    }


def prepare_pair_bundle_from_state(state, rosters: dict, *, cutoff: str, best_ofs: Iterable[int],
                                   input_hashes: dict | None = None, roster_provenance: dict | None = None,
                                   previous_roster_observations: dict | None = None) -> dict:
    """Snapshot the canonical prior-day state; never advance it for hypothetical pairs."""
    from scripts.prospective_sports_features import ChronologicalFeatureState
    if not isinstance(state, ChronologicalFeatureState):
        raise ValueError('canonical chronological feature state required')
    at = utc(cutoff)
    frame = state.pairs(rosters, at.date(), _formats(best_ofs), include_native=True)
    audit = state.audit(at.date())
    audit.update(used_series=sum(state.team_series.values()) // 2,
                 replay_policy='incremental shared state; only strictly earlier completed source days consumed')
    policy = 'explicit_roster_ids' if previous_roster_observations is None else 'previous_observed_roster_proxy'
    return _pair_bundle(frame, audit, rosters, at, policy=policy, proxy=previous_roster_observations,
                        input_hashes=input_hashes, roster_provenance=roster_provenance)


def prepare_pair_bundle(matches: Iterable[dict], rosters: dict | None, *, cutoff: str,
                        best_ofs: Iterable[int], previous_roster_teams: list[str] | None = None,
                        input_hashes: dict | None = None, roster_provenance: dict | None = None) -> dict:
    """Reconstruct every counterfactual pair from strictly earlier calendar days.

    For a streamed source and proxy rosters, resolve the proxy in a first pass and
    supply those IDs explicitly here; the CLI does this without retaining raw maps.
    """
    from scripts.prospective_sports_features import build_pair_features
    at = utc(cutoff)
    formats = _formats(best_ofs)
    policy = 'explicit_roster_ids'
    proxy = None
    if previous_roster_teams is not None:
        if rosters is not None:
            raise ValueError('explicit roster IDs and previous-roster proxy are mutually exclusive')
        materialized = list(matches)
        rosters, proxy = previous_observed_rosters(materialized, previous_roster_teams, at.date())
        matches = materialized
        policy = 'previous_observed_roster_proxy'
    if rosters is None:
        raise ValueError('explicit roster IDs or explicit previous-roster proxy required')
    frame, audit = build_pair_features(matches, rosters, at.date(), formats, include_native=True)
    return _pair_bundle(frame, audit, rosters, at, policy=policy, proxy=proxy,
                        input_hashes=input_hashes, roster_provenance=roster_provenance)


def _validated_rows(bundle, teams, best_ofs, cutoff):
    if bundle.get('version') != BUNDLE_VERSION:
        raise ValueError('unsupported prepared pair bundle version')
    if utc(bundle['cutoff']) != cutoff:
        raise ValueError('prepared pair cutoff differs from tournament cutoff')
    expected = {(frozenset((a, b)), bo) for a, b in combinations(teams, 2) for bo in best_ofs}
    rows = bundle.get('rows')
    if not isinstance(rows, list) or not rows:
        raise ValueError('prepared pair rows must be nonempty')
    from scripts.prospective_sports_features import NATIVE_FIELDS, NATIVE_SUPPLEMENT
    native = bundle.get('native_supplement')
    if native is not None and native != NATIVE_SUPPLEMENT:
        raise ValueError('unsupported native supplement contract')
    allowed = set(_REQUIRED_BASE_FIELDS) | {'team_a', 'team_b', 'best_of', 'feature_history_max_at'}
    fields = set(_REQUIRED_BASE_FIELDS) | (set(NATIVE_FIELDS) if native is not None else set())
    allowed.update(fields)
    paired = pair_columns(sorted(fields))
    seen, states, probabilities = set(), {}, {}
    history_max = None
    for row in rows:
        if not isinstance(row, dict) or set(row) != allowed:
            raise ValueError('pair rows require exact canonical sports fields; market/label/extra columns forbidden')
        a, b, bo = row['team_a'], row['team_b'], row['best_of']
        if type(bo) is not int or not isinstance(a, str) or not isinstance(b, str):
            raise ValueError('pair IDs and best_of require exact string/integer types')
        key = (frozenset((a, b)), bo)
        if a == b or key not in expected or key in seen:
            raise ValueError('all unordered pairings/formats must occur exactly once')
        seen.add(key)
        history = utc(row['feature_history_max_at'])
        if history > cutoff:
            raise ValueError('feature history extends after tournament cutoff')
        history_max = max(history, history_max) if history_max else history
        for field in fields:
            if isinstance(row[field], bool) or not isinstance(row[field], (float, int)) or not math.isfinite(row[field]):
                raise ValueError(f'nonfinite or nonnumeric sports feature: {field}')
        if native is not None and any(row[field] <= 0 for field in NATIVE_FIELDS):
            raise ValueError('native supplement TM sigma must be positive')
        for field in _PROBABILITY_FIELDS:
            p = row[field]
            if not 0 <= p <= 1:
                raise ValueError('rating probabilities must be bounded')
            state_key = (tuple(sorted((a, b))), field)
            p = p if a < b else 1 - p
            if state_key in probabilities and not math.isclose(probabilities[state_key], p, abs_tol=1e-12, rel_tol=0):
                raise ValueError('pair rating state is not frozen across formats')
            probabilities[state_key] = p
        for left, right in paired:
            for team, field in ((a, left), (b, right)):
                state_key = (team, left)
                value = row[field]
                if state_key in states and states[state_key] != value:
                    raise ValueError('team/roster state is not frozen across pairings')
                states[state_key] = value
    if seen != expected:
        raise ValueError('missing counterfactual pairing/format; realized fixtures are insufficient')
    from scripts.prospective_sports_features import _roster
    rosters = bundle.get('rosters')
    if not isinstance(rosters, dict) or set(rosters) != set(teams):
        raise ValueError('prepared bundle requires explicit roster IDs for every team')
    players = set()
    for team, roster in rosters.items():
        ids = _roster(roster, f'frozen roster for {team}')
        if players.intersection(ids):
            raise ValueError('frozen rosters share a player ID')
        players.update(ids)
    roster_provenance = bundle.get('roster_provenance')
    if roster_provenance is not None:
        if not isinstance(roster_provenance, Mapping):
            raise ValueError('roster provenance must be a team evidence mapping')
        for team, evidence in roster_provenance.items():
            if not isinstance(evidence, Mapping):
                raise ValueError(f'roster provenance for {team} must be an evidence mapping')
            for clock in ('published_at', 'available_at'):
                value = evidence.get(clock)
                if value is None:
                    continue  # Unknown clocks remain uncertified scenario evidence.
                try:
                    observed = utc(value)
                except ValueError as exc:
                    raise ValueError(f'roster {clock} for {team} requires an aware publication/availability clock') from exc
                if observed > cutoff:
                    raise ValueError(f'roster {clock} for {team} is after tournament cutoff')
    audit = bundle.get('audit', {})
    if audit.get('source_date_max') and date.fromisoformat(audit['source_date_max']) >= cutoff.date():
        raise ValueError('source feature history must be strictly prior-day')
    if audit.get('feature_history_max_at') and utc(audit['feature_history_max_at']) > cutoff:
        raise ValueError('audit feature history extends after tournament cutoff')
    return rows, history_max


def predict_exp081_pairs(artifact: Mapping, bundle: Mapping, *, teams: list[str],
                         best_ofs: Iterable[int], cutoff: str, mode: str = 'verified') -> tuple[dict, dict]:
    """Return direct calibrated SERIES means and explicit qualification provenance."""
    from scripts.prospective_sports_features import FEATURE_VERSION, FEATURE_CONTRACT
    at = utc(cutoff)
    teams, formats = _teams(teams), _formats(best_ofs)
    if mode not in ('verified', 'exploratory', 'chronological'):
        raise ValueError('mode must be verified, exploratory or chronological')
    version = artifact.get('model_version', '')
    aligned = artifact.get('feature_version') == FEATURE_VERSION
    if not isinstance(version, str) or not version:
        raise ValueError('explicit model version required')
    if not aligned and artifact.get('feature_version') != 'ratings-w20-symmetric-series-v1':
        raise ValueError('unsupported artifact feature version')
    if mode == 'chronological' and not aligned:
        raise ValueError('chronological mode requires a newly trained chronological-v2 fold; old weights are incompatible')
    try:
        fold = artifact['provenance']['fold']
        fit_end = date.fromisoformat(fold['fit']['date_max'])
        cal_end = date.fromisoformat(fold['calibration']['date_max'])
        ensemble_end = date.fromisoformat(artifact['ensemble_calibration']['calibration_end'])
        fit_end = max(fit_end, date.fromisoformat(fold['fit'].get('result_day_max', fit_end.isoformat())),
                      date.fromisoformat(artifact.get('training_end', fit_end.isoformat())))
        cal_end = max(cal_end, date.fromisoformat(fold['calibration'].get('result_day_max', cal_end.isoformat())),
                      date.fromisoformat(artifact.get('calibration_end', cal_end.isoformat())))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError('artifact requires explicit training and calibration date provenance') from exc
    if fit_end >= at.date():
        raise ValueError('model training includes tournament cutoff day or future data')
    if max(cal_end, ensemble_end) >= at.date():
        raise ValueError('model calibration includes tournament cutoff day or future data')
    if artifact.get('feature_version') != (FEATURE_VERSION if aligned else 'ratings-w20-symmetric-series-v1'):
        raise ValueError('unsupported artifact feature version')
    if bundle.get('feature_version') != FEATURE_VERSION:
        raise ValueError('unsupported prepared feature version; compatibility cannot be inferred')
    if aligned:
        for source in (artifact, bundle):
            if (source.get('feature_contract') != FEATURE_CONTRACT
                    or source.get('feature_contract_sha256') != json_digest(FEATURE_CONTRACT)):
                raise ValueError('chronological feature contract does not match the shared training/export implementation')
        if not artifact.get('eligible_from') or utc(artifact['eligible_from']) > at:
            raise ValueError('fold is not eligible before the requested tournament origin')
        if fit_end >= utc(artifact['eligible_from']).date():
            raise ValueError('model training includes eligible origin day or future data')
        if max(cal_end, ensemble_end) >= utc(artifact['eligible_from']).date():
            raise ValueError('model calibration includes eligible origin day or future data')
    rows, history_max = _validated_rows(bundle, teams, formats, at)
    if aligned:
        if artifact.get('prediction_semantics', 'direct_series_probability') != 'direct_series_probability':
            raise ValueError('EXP081 requires direct series probability semantics')
        projection_count = artifact.get('series_projection_count', 0)
        if type(projection_count) is not int or projection_count != 0:
            raise ValueError('EXP081 direct series output must not be projected')
        if artifact.get('feature_algebra_version', 'ratings-w20-symmetric-series-v1') != 'ratings-w20-symmetric-series-v1':
            raise ValueError('unsupported feature algebra')
        if artifact.get('ensemble_aggregation', 'sigmoid(mean(member_platt_slope * antisymmetric_logit))') != 'sigmoid(mean(member_platt_slope * antisymmetric_logit))':
            raise ValueError('unsupported ensemble aggregation; calibrated member logits must be averaged')
    gaps = [] if aligned else list(COMPATIBILITY_GAPS)
    availability = [
        'source calendar days bound chronological consumption but do not certify source publication/availability',
        'roster IDs/proxy history do not certify historical announcement availability',
    ]
    created = artifact.get('created_at')
    if created is None:
        availability.append('artifact creation/availability timestamp is missing')
    elif utc(created) > at:
        availability.append('artifact was created after tournament cutoff: retrospective reconstruction, not an available pre-start forecast')
    if not bundle.get('input_hashes'):
        availability.append('prepared bundle omits original source file hashes; only supplied bundle content is reproducibly identified')
    if not bundle.get('roster_provenance'):
        availability.append('roster source/announcement provenance not supplied')
    if mode == 'verified':
        raise ValueError('verified mode blocked by availability certification: ' + '; '.join(gaps + availability))
    model = SiameseSeriesModel.from_mapping(artifact)
    if aligned:
        from src.models.symmetric_series import build_feature_mapping
        if list(model.feature_names) != list(build_feature_mapping(rows[0], best_of=rows[0]['best_of'])):
            raise ValueError('chronological EXP081 requires the complete canonical79 feature order')
    if (not np.isfinite(model.scales).all() or np.any(model.scales <= 0)
            or not np.isfinite(model.means).all() or np.any(model.means != 0)):
        raise ValueError('EXP081 requires finite positive scales and zero centering for symmetry')
    if model.scales.shape != (len(model.feature_names),) or model.means.shape != model.scales.shape:
        raise ValueError('EXP081 scaler dimensions mismatch')
    for member in model.members:
        arrays = (member.w1, member.b1, member.w2, member.b2, member.w3)
        if not all(np.isfinite(value).all() for value in arrays) or not all(
                math.isfinite(value) for value in (member.b3, member.platt_slope)) or member.platt_slope <= 0:
            raise ValueError('ensemble parameters must be finite and calibration slopes positive')
        if (member.w1.ndim != 2 or member.w1.shape[0] != len(model.feature_names)
                or member.b1.shape != (member.w1.shape[1],) or not member.b1.size
                or member.w2.ndim != 2 or member.w2.shape[0] != member.b1.size
                or member.b2.shape != (member.w2.shape[1],) or not member.b2.size
                or member.w3.shape != (member.b2.size, 1)):
            raise ValueError('ensemble parameter dimensions mismatch')
    probabilities = {}
    for row in rows:
        p = model.predict(row, best_of=row['best_of'])
        if not math.isfinite(p) or not 0 <= p <= 1:
            raise ValueError('model emitted a nonfinite or invalid series probability')
        swapped = dict(row)
        for field in _PROBABILITY_FIELDS:
            swapped[field] = 1 - row[field]
        for left, right in pair_columns(sorted(_REQUIRED_BASE_FIELDS)):
            swapped[left], swapped[right] = row[right], row[left]
        reverse = model.predict(swapped, best_of=row['best_of'])
        if not math.isfinite(reverse) or abs(p + reverse - 1) > 1e-6:
            raise ValueError('model pair probabilities violate complementary side symmetry')
        a, b, bo = row['team_a'], row['team_b'], row['best_of']
        probabilities[a, b, bo] = p
    provenance = {
        'qualification': 'chronologically_reconstructed_research_only' if aligned else 'exploratory_retrospective_not_pit_certified', 'mode': mode,
        'probability_unit': 'series_win', 'estimator': 'SiameseSeriesModel.predict calibrated mean',
        'model_version': version, 'model_content_sha256': json_digest(artifact),
        'model_created_at': created, 'model_status': artifact.get('status'),
        'model_feature_version': artifact['feature_version'], 'feature_version': bundle['feature_version'],
        'model_provenance': artifact['provenance'], 'ensemble_calibration': artifact['ensemble_calibration'],
        'training': artifact.get('training'),
        'training_cutoff': datetime.combine(fit_end + timedelta(days=1), time.min, tzinfo=timezone.utc).isoformat(),
        'calibration_cutoff': datetime.combine(max(cal_end, ensemble_end) + timedelta(days=1), time.min, tzinfo=timezone.utc).isoformat(),
        'cutoff_timestamp_semantics': 'exclusive UTC next-midnight upper bound of final consumed source day; not source availability',
        'tournament_cutoff': at.isoformat(),
        'feature_history_max_at': history_max.isoformat(), 'features_sha256': json_digest(bundle),
        'feature_audit': bundle.get('audit'), 'input_hashes': bundle.get('input_hashes', {}),
        'roster_policy': bundle.get('roster_policy'), 'rosters': bundle.get('rosters'),
        'roster_provenance': bundle.get('roster_provenance'),
        'previous_roster_observations': bundle.get('previous_roster_observations'),
        'history_provenance': bundle.get('history_provenance'),
        'features_prepared_at': bundle.get('prepared_at'),
        'compatibility_gaps': gaps,
        'feature_semantic_compatibility': 'matched_chronological_version' if aligned else 'unmatched_legacy_training_semantics',
        'availability_gaps': availability, 'availability_certified': False,
    }
    return probabilities, provenance


def _positive_slope(value):
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)) or not math.isfinite(value) or value <= 0:
        raise ValueError('calibration slopes must be finite and positive')
    return float(value)


def predict_model_pairs(artifact: Mapping, bundle: Mapping, *, model_kind: str,
                        artifact_sha256: str, teams: list[str], best_ofs: Iterable[int],
                        cutoff: str, mode: str = 'chronological') -> tuple[dict, dict]:
    """Infer explicitly selected annual research recipes from identical frozen state.

    ``artifact_sha256`` identifies the trusted serialized file, not a JSON
    rendering of a loaded estimator. Callers must hash the file they loaded.
    The EXP081-specific API separately supports old exploratory JSON folds.
    """
    from scipy.special import expit, logit
    from scripts.prospective_sports_features import FEATURE_CONTRACT, FEATURE_VERSION, NATIVE_SUPPLEMENT
    from src.models.symmetric_series import build_feature_mapping, _independent_series_probability

    if model_kind not in ('exp039', 'linear79', 'exp081', 'glicko', 'glicko_format'):
        raise ValueError('unsupported explicit model kind')
    if (not isinstance(artifact_sha256, str) or len(artifact_sha256) != 64
            or any(c not in '0123456789abcdef' for c in artifact_sha256)):
        raise ValueError('actual serialized artifact SHA256 is required')
    if mode not in ('chronological', 'exploratory', 'verified'):
        raise ValueError('unsupported prediction mode')
    if not isinstance(artifact, Mapping) or not isinstance(bundle, Mapping):
        raise ValueError('artifact and prepared bundle must be mappings')
    at = utc(cutoff)
    teams, formats = _teams(teams), _formats(best_ofs)
    for source in (artifact, bundle):
        if (source.get('feature_version') != FEATURE_VERSION
                or source.get('feature_contract') != FEATURE_CONTRACT
                or source.get('feature_contract_sha256') != json_digest(FEATURE_CONTRACT)):
            raise ValueError('chronological feature contract does not match shared training/export')
    if model_kind == 'exp081':
        probabilities, provenance = predict_exp081_pairs(
            artifact, bundle, teams=teams, best_ofs=formats, cutoff=cutoff, mode=mode)
        provenance.update(model_content_sha256=artifact_sha256, model_artifact_sha256=artifact_sha256,
                          model_kind=model_kind, series_projection_count=0)
        return probabilities, provenance
    if artifact.get('feature_algebra_version') != 'ratings-w20-symmetric-series-v1':
        raise ValueError('unsupported artifact feature algebra')
    if artifact.get('prediction_semantics') != 'direct_series_probability':
        raise ValueError('artifact must declare series probability output')
    projection = 1 if model_kind in ('glicko', 'glicko_format') else 0
    count = artifact.get('series_projection_count')
    if type(count) is not int or count != projection:
        raise ValueError('artifact series projection count conflicts with model kind')
    version = artifact.get('model_version')
    if not isinstance(version, str) or not version:
        raise ValueError('explicit artifact model version required')
    try:
        fold = artifact['provenance']['fold']
        fit_end = max(date.fromisoformat(artifact['training_end']),
                      date.fromisoformat(fold['fit']['date_max']),
                      date.fromisoformat(fold['fit'].get('result_day_max', artifact['training_end'])))
        cal_end = max(date.fromisoformat(artifact['calibration_end']),
                      date.fromisoformat(fold['calibration']['date_max']),
                      date.fromisoformat(fold['calibration'].get('result_day_max', artifact['calibration_end'])))
        eligible = utc(artifact['eligible_from'])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError('explicit training/calibration bounds and eligible_from required') from exc
    if fit_end >= at.date() or fit_end >= eligible.date():
        raise ValueError('model training includes origin day or later outcomes')
    if cal_end >= at.date() or cal_end >= eligible.date():
        raise ValueError('model calibration includes origin day or later outcomes')
    if eligible > at:
        raise ValueError('fold is not eligible at requested origin')
    rows, history_max = _validated_rows(bundle, teams, formats, at)

    availability = [
        'source calendar days do not certify source publication/availability',
        'explicit or proxy roster IDs do not certify historical announcement availability',
    ]
    created = artifact.get('created_at')
    if created is None or utc(created) > at:
        availability.append('artifact creation is missing or later than origin; retrospective reconstruction only')
    if not bundle.get('input_hashes'):
        availability.append('original source file hashes not supplied')
    if not bundle.get('roster_provenance'):
        availability.append('roster source/announcement provenance not supplied')
    if mode == 'verified':
        raise ValueError('verified mode blocked by availability certification: ' + '; '.join(availability))

    if model_kind in ('exp039', 'linear79'):
        import pandas as pd
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        slope = _positive_slope(artifact.get('calibration_slope'))
        if model_kind == 'exp039':
            from scripts.corrected_history_evaluation import build_exp039_features, exp039_probability
            from src.models.team_order import swap_orientation
            from sklearn.impute import SimpleImputer
            from sklearn.preprocessing import StandardScaler
            if bundle.get('native_supplement') != NATIVE_SUPPLEMENT:
                raise ValueError('native46 requires an explicit same-state native supplement')
            frame = pd.DataFrame(rows).assign(BoN=[row['best_of'] for row in rows], y_true=0)
            # y_true is only the orientation helper's bookkeeping column; no
            # observed outcome is consumed by features or inference.
            frame, names, rank = build_exp039_features(frame)
            estimator = artifact.get('pipeline')
            if (not isinstance(estimator, Pipeline)
                    or not isinstance(estimator.named_steps.get('model'), LogisticRegression)
                    or list(artifact.get('rank_probability_features', ())) != list(rank)):
                raise ValueError('native46 requires its fitted logistic pipeline and exact probability features')
            fitted = estimator.named_steps['model']
            if list(getattr(estimator, 'feature_names_in_', ())) != names:
                raise ValueError('native46 fitted pipeline feature order mismatch')
            if list(estimator.named_steps) != ['imputer', 'scaler', 'model']:
                raise ValueError('native46 requires its established imputer/scaler/model pipeline')
            imputer, scaler = estimator.named_steps['imputer'], estimator.named_steps['scaler']
            if not isinstance(imputer, SimpleImputer) or not isinstance(scaler, StandardScaler):
                raise ValueError('unsupported native46 feature transformations')
            for component, field in ((imputer, 'statistics_'), (scaler, 'mean_'), (scaler, 'scale_')):
                value = np.asarray(getattr(component, field, None), dtype=float)
                if value.shape != (len(names),) or not np.isfinite(value).all():
                    raise ValueError('native46 requires complete finite fitted transformations')
            if np.any(scaler.scale_ <= 0):
                raise ValueError('native46 fitted scales must be positive')
            reverse_frame = swap_orientation(frame, names, rank, np.ones(len(frame), dtype=bool))
            def infer(values):
                return np.clip(expit(slope * logit(exp039_probability(estimator, values, names, rank))), 1e-12, 1-1e-12)
            inputs, reverse_inputs = frame, reverse_frame
        else:
            engineered = [build_feature_mapping(row, best_of=row['best_of']) for row in rows]
            names = list(engineered[0])
            scales = np.asarray(artifact.get('scales'), dtype=float)
            if scales.shape != (len(names),) or not np.isfinite(scales).all() or np.any(scales <= 0):
                raise ValueError('Linear79 requires finite positive train-only scales')
            estimator = fitted = artifact.get('model')
            if not isinstance(fitted, LogisticRegression) or fitted.fit_intercept:
                raise ValueError('Linear79 requires a fitted zero-intercept logistic model')
            inputs = np.asarray([list(row.values()) for row in engineered], dtype=float) / scales
            reverse_inputs = -inputs
            def infer(values):
                return np.clip(expit(slope * estimator.decision_function(values)), 1e-12, 1-1e-12)
        if list(artifact.get('features', ())) != names:
            raise ValueError('artifact requires exact complete feature order')
        if (not np.array_equal(getattr(fitted, 'classes_', None), [0, 1])
                or np.shape(getattr(fitted, 'coef_', None)) != (1, len(names))
                or not np.isfinite(fitted.coef_).all()
                or not np.isfinite(fitted.intercept_).all()
                or (model_kind == 'linear79' and np.any(fitted.intercept_ != 0))):
            raise ValueError('invalid fitted binary logistic parameters')
        values, reverse_values = infer(inputs), infer(reverse_inputs)
    else:
        if artifact.get('map_probability_field') != 'player_gl' or artifact.get('map_clip') != [0.001, 0.999]:
            raise ValueError('Glicko requires explicit player_gl map input and shared clipping contract')
        slopes = artifact.get('calibration_slopes')
        if model_kind == 'glicko_format':
            if not isinstance(slopes, Mapping) or set(slopes) != {'1', '3', '5'}:
                raise ValueError('format Glicko requires all three calibrated format slopes')
            slopes = {bo: _positive_slope(slope) for bo, slope in slopes.items()}
        elif slopes is not None:
            raise ValueError('raw Glicko must not contain format calibration')
        def infer(reverse):
            output = []
            for row in rows:
                p = _independent_series_probability(1-row['player_gl'] if reverse else row['player_gl'], row['best_of'])
                if slopes is not None:
                    p = float(np.clip(expit(slopes[str(row['best_of'])] * logit(p)), 1e-12, 1-1e-12))
                output.append(p)
            return np.asarray(output)
        values, reverse_values = infer(False), infer(True)
    if (values.shape != (len(rows),) or not np.isfinite(values).all()
            or np.any((values < 0) | (values > 1))
            or not np.isfinite(reverse_values).all()
            or np.any(abs(values + reverse_values - 1) > 1e-6)):
        raise ValueError('model emitted invalid or noncomplementary series probabilities')
    probabilities = {(row['team_a'], row['team_b'], row['best_of']): float(p)
                     for row, p in zip(rows, values, strict=True)}
    return probabilities, {
        'qualification': 'chronologically_reconstructed_research_only', 'mode': mode,
        'probability_unit': 'series_win', 'series_projection_count': projection,
        'model_kind': model_kind, 'model_version': version,
        'model_content_sha256': artifact_sha256, 'model_artifact_sha256': artifact_sha256,
        'model_created_at': created, 'model_status': artifact.get('status'),
        'model_feature_version': artifact['feature_version'], 'feature_version': bundle['feature_version'],
        'model_provenance': artifact['provenance'], 'training': artifact.get('training'),
        'training_cutoff': datetime.combine(fit_end + timedelta(days=1), time.min, tzinfo=timezone.utc).isoformat(),
        'calibration_cutoff': datetime.combine(cal_end + timedelta(days=1), time.min, tzinfo=timezone.utc).isoformat(),
        'cutoff_timestamp_semantics': 'exclusive UTC next-midnight upper bound of final consumed source day; not source availability',
        'tournament_cutoff': at.isoformat(), 'feature_history_max_at': history_max.isoformat(),
        'features_sha256': json_digest(bundle), 'feature_audit': bundle.get('audit'),
        'input_hashes': bundle.get('input_hashes', {}), 'rosters': bundle.get('rosters'),
        'roster_policy': bundle.get('roster_policy'), 'roster_provenance': bundle.get('roster_provenance'),
        'previous_roster_observations': bundle.get('previous_roster_observations'),
        'history_provenance': bundle.get('history_provenance'), 'features_prepared_at': bundle.get('prepared_at'),
        'compatibility_gaps': [], 'feature_semantic_compatibility': 'matched_chronological_version',
        'availability_gaps': availability, 'availability_certified': False,
    }


def fit_score_distributions(rows: Iterable[dict], *, cutoff: date, best_ofs: Iterable[int]) -> tuple[dict, dict]:
    """Unsmoothed pooled losing-map PMFs, conditional on an independently drawn winner.

    Rows: {id, date, best_of, score_a, score_b}; optional end_date records
    overnight completion. These are empirical score margins, NOT winner forecasts.
    """
    if not isinstance(cutoff, date) or isinstance(cutoff, datetime):
        raise ValueError('score cutoff must be a calendar date')
    formats = _formats(best_ofs)
    counts = {bo: [0] * (bo // 2 + 1) for bo in formats if bo != 1}
    seen, used, excluded = set(), [], Counter()
    for row in rows:
        day = date.fromisoformat(row['date'])
        end = date.fromisoformat(row.get('end_date', row['date']))
        if end < day:
            raise ValueError('score completion precedes series start')
        if max(day, end) >= cutoff:
            excluded['on_or_after_cutoff'] += 1
            continue
        bo = row['best_of']
        if type(bo) is not int:
            raise ValueError('score best_of must be an integer')
        if bo not in counts:
            excluded['unrequested_format'] += 1
            continue
        identity = row.get('id')
        if not isinstance(identity, str) or not identity or identity in seen:
            raise ValueError('historical score IDs must be unique nonempty strings')
        seen.add(identity)
        a, b = row['score_a'], row['score_b']
        if (type(a) is not int or type(b) is not int or min(a, b) < 0
                or max(a, b) != bo // 2 + 1 or min(a, b) > bo // 2):
            raise ValueError('invalid completed historical series score')
        counts[bo][min(a, b)] += 1
        used.append(row)
    for bo, values in counts.items():
        if not sum(values):
            raise ValueError(f'no prior Bo{bo} score data; no smoothing or fallback distribution is supplied')
    distributions = {bo: [value / sum(values) for value in values] for bo, values in counts.items()}
    if 1 in formats:
        distributions[1] = [1.]
    return distributions, {
        'model': 'empirical_pooled_losing_maps_conditional_on_winner', 'cutoff': cutoff.isoformat(),
        'samples': {str(bo): sum(values) for bo, values in counts.items()}, 'counts': counts,
        'used_scores_sha256': json_digest(used), 'used_series_ids': sorted(seen),
        'source_date_max': max((row.get('end_date', row['date']) for row in used), default=None),
        'exclusions': dict(excluded), 'smoothing': None,
        'limitations': ['pooled across supplied competitions/teams; conditional score shape is not team-specific',
                        'calendar dates establish prior play, not point-in-time publication availability'],
    }


def golgg_score_rows(matches: Iterable[dict], cutoff: date):
    """Validate prior GOLGG series before exposing empirical completed score rows."""
    from scripts.prospective_sports_features import _compact_series, _game_days, _order
    from scripts.build_siamese_research_dataset import _history_helpers
    from src.utils import golgg_schema as schema
    helpers, game_ids, match_ids = _history_helpers(), set(), set()
    for row in matches:
        day = date.fromisoformat(row['date'])
        if day >= cutoff:
            continue
        days = _game_days(row, day)
        if days[-1] >= cutoff:
            continue
        bo = schema.best_of(row)
        if bo not in (1, 3, 5) or schema.initial_wins(row):
            continue  # Bo2 and seeded map advantages are not ordinary series margins.
        identity = str(row['match_id'])
        if identity in match_ids:
            raise ValueError('duplicate historical score series ID')
        match_ids.add(identity)
        compact = _compact_series(row, _order(identity), helpers, game_ids, days)
        a = sum(game.score for game in compact.games)
        yield {'id': identity, 'date': day.isoformat(), 'end_date': days[-1].isoformat(),
               'best_of': bo, 'score_a': int(a), 'score_b': int(len(compact.games) - a)}
