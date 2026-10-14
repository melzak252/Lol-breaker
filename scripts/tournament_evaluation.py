"""Offline, manifest-driven tournament forecasts; outcomes are joined only by scoring.

An origin is the persistence transaction: its complete model/target rows are written
once, before any labels are read. Resume verifies both manifest and referenced inputs.
Historical reconstruction, synthetic mechanics and captured forecasts never share a run.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from itertools import combinations
import json
import math
import fcntl
import os
from pathlib import Path
import platform

import tempfile
import shutil
from src.models.tournament_prediction import file_digest, json_digest, utc

VERSION = 'tournament-evaluation-v3'
QUALIFICATIONS = {'synthetic_mechanics', 'retrospective_reconstruction', 'prospective_capture'}
FORECAST_MODES = {'pre_draw', 'post_draw', 'daily_rollforward', 'post_round'}


def _observation_day_end(origin):
    if origin['forecast_mode'] != 'daily_rollforward':
        return None
    return (utc(origin['cutoff']).date() - timedelta(days=1)).isoformat()


def _load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _atomic(path, payload):
    """Replace mutable status/projection, never a frozen input or origin record."""
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w', encoding='utf-8') as stream:
        json.dump(payload, stream, allow_nan=False, sort_keys=True, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _new(path, payload):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(payload, stream, allow_nan=False, sort_keys=True, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())



def _read_origin(directory):
    record = directory / 'record.json'
    if _load(directory / 'commit.json')['sha256'] != file_digest(record):
        raise ValueError('saved origin content hash changed')
    payload = _load(record)
    if directory.name != json_digest([payload['tournament_id'], payload['phase_id'], payload['origin_id']]):
        raise ValueError('saved origin identity changed')
    return payload


def _commit_origin(directory, payload):
    """Publish record and checksum together; interrupted staging is never a commit."""
    pending = Path(tempfile.mkdtemp(prefix='.pending-', dir=directory.parent))
    try:
        _new(pending / 'record.json', payload)
        _new(pending / 'commit.json', {'sha256': file_digest(pending / 'record.json')})
        os.rename(pending, directory)
    finally:
        if pending.exists():
            shutil.rmtree(pending)

def _path(base, value):
    if not isinstance(value, str) or not value:
        raise ValueError('input path must be a nonempty string')
    return (base / value).resolve()


def _source_paths(manifest, base):
    """Intentionally excludes outcomes: unavailable future labels are not inputs."""
    paths = {_path(base, value) for value in manifest.get('evidence_files', [])}
    paths.update(_path(base, value) for value in manifest.get('evidence_hashes', {}))
    for model in manifest['models']:
        if model.get('artifact'):
            paths.add(_path(base, model['artifact']))
    for event in manifest['events']:
        paths.add(_path(base, event['spec']))
        for origin in event['origins']:
            for key in ('state', 'score_model'):
                if origin.get(key):
                    paths.add(_path(base, origin[key]))
            for key in ('pairs', 'tables'):
                paths.update(_path(base, value) for value in origin.get(key, {}).values())
    return sorted(paths)



def _validate_mode_state(origin, spec, state, qualification):
    mode = origin['forecast_mode']
    stages = {stage['id']: stage for stage in spec['stages']}
    selected = origin.get('phase_stages', list(stages))
    if (not isinstance(selected, list) or not selected or len(set(selected)) != len(selected)
            or any(not isinstance(stage, str) or stage not in stages for stage in selected)):
        raise ValueError('phase_stages must identify distinct declared stages')
    has_uniform_draw = any(stages[key].get('draw', {}).get('policy') == 'uniform' for key in selected)
    if mode == 'pre_draw':
        if any(row['stage'] in selected for field in ('completed', 'pairings', 'orders', 'group_orders')
               for row in (state or {}).get(field, [])):
            raise ValueError('pre_draw cannot observe its own phase pairings or results')
        if not has_uniform_draw:
            raise ValueError('pre_draw requires a declared unknown-draw policy in its forecast scope')
        return
    if mode == 'post_draw':
        if any(row['stage'] in selected for row in (state or {}).get('pairings', [])):
            return
        if not has_uniform_draw and origin.get('draw_knowledge') == {'status': 'deterministic_spec'}:
            return
        raise ValueError('post_draw requires published first-round pairings in state')
    if mode == 'post_round':
        checkpoint = origin.get('checkpoint')
        if not isinstance(checkpoint, dict) or set(checkpoint) != {'stage', 'group', 'round'}:
            raise ValueError('post_round requires an explicit checkpoint {stage, group, round}')
        if any(not isinstance(checkpoint[key], (str, int)) or isinstance(checkpoint[key], bool)
               for key in checkpoint):
            raise ValueError('post_round checkpoint requires exact stage, group, and round identifiers')
        completed = [row for row in (state or {}).get('completed', [])
                     if all(row.get(key) == checkpoint[key] for key in checkpoint)]
        if not completed:
            raise ValueError('post_round requires completed records for its declared checkpoint')
        stage = next((item for item in spec['stages'] if item['id'] == checkpoint['stage']), None)
        if stage is None:
            raise ValueError('post_round checkpoint names an unknown stage')
        pairing = next((row for row in (state or {}).get('pairings', [])
                        if all(row.get(key) == checkpoint[key] for key in checkpoint)), None)
        if pairing is not None:
            expected = {frozenset(pair) for pair in pairing['pairs']}
            observed = {frozenset((row['team_a'], row['team_b'])) for row in completed}
            if not expected or expected != observed or len(completed) != len(observed):
                raise ValueError('post_round checkpoint pairing block is partial or mismatched')
            return
        graph_match_ids = {match['id'] for match in stage.get('matches', [])} if stage.get('kind') == 'graph' else set()
        if checkpoint['group'] == 'all' and checkpoint['round'] in graph_match_ids and len(completed) == 1:
            return
        raise ValueError('post_round requires a complete pairing block or one declared graph match')
    if qualification == 'prospective_capture':
        raise ValueError('prospective_capture requires strict_timestamped state, not daily reconstructed state')
    if state is None or state.get('temporal_policy') != 'daily_rollforward':
        raise ValueError('daily_rollforward requires state temporal_policy=daily_rollforward')
    cutoff = utc(origin['cutoff'])
    if cutoff.hour or cutoff.minute or cutoff.second or cutoff.microsecond:
        raise ValueError('daily_rollforward cutoff must be UTC midnight')
    for completed in state.get('completed', []):
        if any(key in completed for key in ('completed_at', 'completed_date', 'available_at')):
            raise ValueError('daily_rollforward completed events use source_date, never invented timestamps')
        try:
            source_day = date.fromisoformat(completed['source_date'])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError('daily_rollforward completed events require source_date provenance') from error
        if source_day >= cutoff.date():
            raise ValueError('daily_rollforward source_date must strictly precede cutoff UTC day')

def _validate_manifest(manifest):
    if manifest.get('version') != VERSION or manifest.get('qualification') not in QUALIFICATIONS:
        raise ValueError('explicit version and evidence qualification required')
    models = manifest.get('models')
    if not isinstance(models, list) or not models:
        raise ValueError('nonempty model list required')
    names = [model['id'] for model in models]
    if len(set(names)) != len(names) or any(not isinstance(name, str) or not name or ':' in name
                                           or name in {'fair_series', 'flat'} for name in names):
        raise ValueError('model IDs must be unique, nonempty and not reserved')
    if any(model.get('kind') not in {'series_table', 'exp081', 'exp039', 'linear79', 'glicko', 'glicko_format'}
           for model in models):
        raise ValueError('model kind must identify a supported serialized recipe or archived series_table')
    seen = set()
    edition_starts = {}
    events = manifest.get('events')
    if not isinstance(events, list):
        raise ValueError('events must be an explicit list, including empty eligible cohorts')
    for event in events:
        identity = (event['tournament_id'], event['phase_id'])
        if any(not isinstance(value, str) or not value for value in identity):
            raise ValueError('edition and phase IDs must be nonempty strings')
        if identity in seen:
            raise ValueError('duplicate tournament phase in manifest')
        seen.add(identity)
        start = utc(event['edition_start_at']) if event.get('edition_start_at') is not None else None
        if start is None and event.get('edition_start_status') != 'unverified_whole_edition_boundary':
            raise ValueError('missing edition start requires an explicit unverified boundary status')
        recorded_start = edition_starts.setdefault(event['tournament_id'], start)
        if recorded_start != start:
            raise ValueError('phases of one edition disagree on edition_start_at')
        phase_start = utc(event['phase_start_at']) if event.get('phase_start_at') else None
        if phase_start is not None and start is not None and phase_start < start:
            raise ValueError('phase_start_at cannot precede edition_start_at')
        origins = event['origins']
        prior_origins = {}
        previous = None
        for origin in origins:
            if origin.get('forecast_scope') not in {
                    'full_tournament', 'phase_start', 'remaining_tournament', 'remaining_phase'}:
                raise ValueError('explicit recognized forecast_scope required')
            if origin.get('forecast_mode') not in FORECAST_MODES:
                raise ValueError('each forecast origin requires an explicit recognized forecast_mode')
            excluded = origin.get('model_exclusions', {})
            if (not isinstance(excluded, dict) or not set(excluded).issubset(names)
                    or any(not isinstance(reason, str) or not reason.strip() for reason in excluded.values())):
                raise ValueError('model_exclusions requires declared model IDs and explicit nonempty reasons')
            origin_id = origin['origin_id']
            if origin_id in prior_origins:
                raise ValueError('duplicate forecast origin in tournament phase')
            at = utc(origin['cutoff'])
            if previous is not None and at < previous:
                raise ValueError('forecast origins must be chronological')
            previous = at
            if origin['axis'] not in {'axis1', 'axis2', 'axis3'}:
                raise ValueError('axis must be axis1, axis2 or axis3; axis4 is diagnostic slicing')
            scope = origin['forecast_scope']
            if scope in {'phase_start', 'remaining_phase'} and phase_start is None:
                raise ValueError('phase-scoped forecasts require explicit phase_start_at')
            if scope in {'phase_start', 'remaining_phase'} and not origin.get('phase_stages'):
                raise ValueError('phase-scoped forecasts require explicit phase_stages')
            boundary = phase_start if scope in {'phase_start', 'remaining_phase'} else start
            if boundary is None:
                raise ValueError('whole-tournament scope requires an independently evidenced edition start')
            if origin['axis'] == 'axis1':
                if scope not in {'full_tournament', 'phase_start'} or at >= boundary:
                    raise ValueError('start origin must precede its edition or phase start')
            if at >= boundary and not origin.get('state'):
                raise ValueError('post-start forecasts require real observed tournament state')
            if origin['axis'] == 'axis3':
                reference = prior_origins.get(origin.get('reference_origin_id'))
                if reference is None or reference['axis'] != 'axis1':
                    raise ValueError('checkpoint requires an explicit earlier reference_origin_id')
                expected_scope = {'full_tournament': 'remaining_tournament', 'phase_start': 'remaining_phase'}[reference['forecast_scope']]
                if scope != expected_scope:
                    raise ValueError('checkpoint and frozen reference scopes differ')
                if origin.get('phase_stages') != reference.get('phase_stages'):
                    raise ValueError('checkpoint and frozen reference phase stages differ')
            targets = origin['targets']
            if not targets or len({target['target_id'] for target in targets}) != len(targets):
                raise ValueError('each origin requires unique explicit targets')
            for target in targets:
                if target.get('resolved', False):
                    continue
                if utc(target['target_start_at']) <= at:
                    raise ValueError('forecast cutoff must precede each unresolved target start')
                if target['source'] == 'series_probability':
                    if utc(target['pairing_available_at']) > at:
                        raise ValueError('pairing was not published by forecast cutoff')
            if manifest['qualification'] == 'prospective_capture':
                now = datetime.now(timezone.utc)
                if at > now:
                    raise ValueError('prospective cutoff cannot include not-yet-available future information')
                if any(not target.get('resolved', False) and utc(target['target_start_at']) <= now
                       for target in targets):
                    raise ValueError('prospective forecast must be captured before target start')
                for key in ('rules_available_at', 'entrants_available_at', 'seeds_available_at'):
                    if utc(event[key]) > at:
                        raise ValueError(f'{key} exceeds forecast cutoff')
            prior_origins[origin_id] = origin


def _probabilities(model, origin, event, spec, formats, base, qualification):
    from scripts.simulate_tournament import _table_probabilities
    from src.models.tournament_prediction import predict_model_pairs
    at = utc(origin['cutoff'])
    if model['kind'] != 'series_table':
        artifact_path = _path(base, model['artifact'])
        if model['kind'] in {'exp039', 'linear79'}:
            import joblib
            artifact = joblib.load(artifact_path)
        else:
            artifact = _load(artifact_path)
        bundle = _load(_path(base, origin['pairs'][model['id']]))
        mode = 'verified' if qualification == 'prospective_capture' else 'chronological'
        return predict_model_pairs(
            artifact, bundle, model_kind=model['kind'], artifact_sha256=file_digest(artifact_path),
            teams=spec['teams'], best_ofs=formats, cutoff=origin['cutoff'], mode=mode)
    table = _load(_path(base, origin['tables'][model['id']]))
    probabilities, provenance = _table_probabilities(table, spec, formats, at)
    supplied = table.get('provenance', {})
    if qualification != 'synthetic_mechanics':
        for key in ('training_cutoff', 'calibration_cutoff', 'feature_history_max_at'):
            if key not in supplied or utc(supplied[key]) > at:
                raise ValueError(f'archived model table requires {key} bounded by forecast cutoff')
        if not supplied.get('model_artifact_sha256') or supplied.get('model_id') != model['id']:
            raise ValueError('table must identify its exact model and immutable artifact hash')
        if qualification == 'prospective_capture' and utc(supplied['created_at']) > at:
            raise ValueError('table was unavailable at forecast cutoff')
    return probabilities, provenance


def _score_model(origin, base):
    if not origin.get('score_model'):
        return None
    model = _load(_path(base, origin['score_model']))
    source = model['provenance']
    cutoff = utc(origin['cutoff']).date().isoformat()
    if source.get('cutoff') != cutoff or not source.get('source_date_max') or source['source_date_max'] >= cutoff:
        raise ValueError('score distribution must use explicitly prior source days at this cutoff')
    return {int(bo): values for bo, values in model['distributions'].items()}


def _series_p(probabilities, a, b, bo):
    if (a, b, bo) in probabilities:
        return probabilities[a, b, bo]
    return 1.0 - probabilities[b, a, bo]


def _target_vector(target, result, probabilities):
    source = target['source']
    kind = target['target_kind']
    if source == 'series_probability':
        if kind != 'series':
            raise ValueError('series probability requires series target kind')
        a, b = target['team_a'], target['team_b']
        p = (float(target['known_winner'] == a) if target.get('resolved', False)
             else _series_p(probabilities, a, b, target['best_of']))
        return {a: p, b: 1-p}
    if source == 'champion_prob':
        if kind != 'champion' or not result.get(source):
            raise ValueError('champion target requires a real championship output')
        return dict(result[source])
    if source in {'advance_prob', 'final_prob'}:
        if kind != 'binary':
            raise ValueError('reach/qualification requires binary target kind')
        vector = result[source]
        if source == 'advance_prob':
            vector = vector[target['stage']]
        p = vector[target['team']]
        return {'0': 1-p, '1': p}
    if source == 'stage_rank_prob':
        if kind != 'placement':
            raise ValueError('stage ranking requires explicit placement bands')
        ranks = result[source][target['stage']][target['group']][target['team']]
        bands = target['bands']
        assigned = [rank for band in bands for rank in band['ranks']]
        if assigned != list(range(1, len(ranks)+1)):
            raise ValueError('official placement bands must partition stage ranks in increasing order')
        if len({band['id'] for band in bands}) != len(bands):
            raise ValueError('duplicate placement band')
        return {band['id']: sum(ranks[rank-1] for rank in band['ranks']) for band in bands}
    if source == 'placement_prob':
        if kind != 'placement':
            raise ValueError('placement output requires placement kind')
        return dict(result[source][target['team']])
    if source == 'joint_final_prob':
        if kind != 'joint':
            raise ValueError('joint finalist output requires joint target kind')
        teams = sorted(result['final_prob'])
        return {json.dumps(list(pair), separators=(',', ':'), ensure_ascii=False): result[source].get(
                    json.dumps(list(pair), separators=(',', ':'), ensure_ascii=False), 0.0)
                for pair in combinations(teams, 2)}
    if source == 'joint_advance_prob':
        if kind != 'joint' or not target.get('categories'):
            raise ValueError('joint qualification requires an explicit enumerable category space')
        distribution = result[source][target['stage']]
        if not set(distribution).issubset(target['categories']):
            raise ValueError('qualification category space omits a simulated legal outcome')
        return {category: distribution.get(category, 0.0) for category in target['categories']}
    if source == 'observable_prob':
        if kind not in {'categorical', 'count'}:
            raise ValueError('observable output requires categorical or count target kind')
        vector = result[source][target['observable_id']]
        if list(vector) != target.get('categories'):
            raise ValueError('observable target requires its complete declared category order')
        return dict(vector)
    raise ValueError(f'unsupported target source {source!r}; never infer from champion rank')


def _condition_target(target, result):
    """Derive resolved facts/flat eligibility from state, never from MC certainty."""
    target = dict(target)
    target['resolved'] = False
    source = target['source']
    resolved = result.get('target_resolved', {})
    if source == 'champion_prob':
        target['resolved'] = resolved.get('champion', False)
        zeros = result.get('structural_zero_categories', {}).get('champion', [])
        target['structural_zero_categories'] = zeros
        target['eligible_teams'] = [team for team in result[source] if team not in zeros]
    elif source in {'advance_prob', 'final_prob'}:
        if source == 'advance_prob':
            flags = resolved.get('advance_by_team', {}).get(target['stage'], {})
            distribution = result[source][target['stage']]
        else:
            flags = resolved.get('final_by_team', {})
            distribution = result[source]
        target['resolved'] = flags.get(target['team'], False)
        target['eligible_teams'] = [team for team in distribution if not flags.get(team, False)]
        known_qualified = sum(flags.get(team, False) and p == 1.0 for team, p in distribution.items())
        target['remaining_slots'] = round(sum(distribution.values())) - known_qualified
    elif source == 'placement_prob':
        target['resolved'] = resolved.get('placement_by_team', {}).get(target['team'], False)
        possibilities = result['possible_placement_categories']
        allowed = set(possibilities[target['team']])
        fixed = [values[0] for values in possibilities.values() if len(values) == 1]
        target['flat_band_sizes'] = {
            band['id']: max(0, len(band['ranks']) - fixed.count(band['id']))
            if band['id'] in allowed else 0 for band in target['bands']}
        target['structural_zero_categories'] = [
            band['id'] for band in target['bands'] if band['id'] not in allowed]
    elif source == 'stage_rank_prob':
        possibilities = result['stage_rank_possible'][target['stage']][target['group']]
        allowed = set(possibilities[target['team']])
        if not allowed:
            raise ValueError('team cannot enter this stage; use global official placement targets')
        bands = target['bands']
        target['resolved'] = any(allowed <= set(band['ranks']) for band in bands)
        target['flat_band_sizes'] = {
            band['id']: max(0, len(band['ranks']) - sum(
                bool(values) and set(values) <= set(band['ranks']) for values in possibilities.values()))
            if allowed.intersection(band['ranks']) else 0 for band in bands}
        target['structural_zero_categories'] = [
            band['id'] for band in bands if not allowed.intersection(band['ranks'])]
    elif source == 'series_probability':
        if any(key not in target for key in ('stage', 'group', 'round')):
            raise ValueError('series target requires stage/group/round node identity')
        matches = [pair for pair in result['known_pairings']
                   if pair['stage'] == target['stage'] and pair['group'] == target['group']
                   and str(pair['round']) == str(target['round'])
                   and {pair['team_a'], pair['team_b']} == {target['team_a'], target['team_b']}
                   and pair['best_of'] == target['best_of']]
        if len(matches) != 1:
            raise ValueError('series target is not a known legal pairing at this checkpoint')
        target['resolved'] = matches[0]['completed']
        if target['resolved']:
            target['known_winner'] = matches[0]['winner']
    elif source == 'joint_final_prob':
        target['resolved'] = resolved.get('final', False)
        target['structural_zero_categories'] = result['structural_zero_categories']['joint_final']
    elif source == 'joint_advance_prob':
        target['resolved'] = resolved.get('advance', {}).get(target['stage'], False)
    elif source == 'observable_prob':
        observable = target['observable_id']
        target['structural_zero_categories'] = result.get(
            'structural_zero_categories', {}).get('observable', {}).get(observable, [])
    return target


def _observable_specs(origin, initial):
    """Bind upset labels to one declared start matrix, independent of candidate."""
    observables = []
    for supplied in origin.get('observables', []):
        observable = dict(supplied)
        if observable.get('kind') == 'upset_count':
            if 'reference_probabilities' in observable:
                raise ValueError('upset reference must name an immutable start-origin model matrix')
            reference_origin = observable.pop('reference_origin_id')
            reference_model = observable.pop('reference_model')
            if reference_origin not in initial or reference_model not in initial[reference_origin]:
                raise ValueError('upset reference model is unavailable at the declared start origin')
            observable['reference_probabilities'] = initial[reference_origin][reference_model][0]
        observables.append(observable)
    return observables


def _flat_vector(target, vector, teams):
    if target['target_kind'] in {'joint', 'categorical', 'count'}:
        return None  # Non-topological flat marginals do not define joint/path/count baselines.
    if target.get('resolved', False):
        return dict(vector)
    if target['target_kind'] == 'binary':
        eligible = target.get('eligible_teams', teams)
        slots = target.get('remaining_slots')
        if slots is None or type(slots) is not int or not 0 <= slots <= len(eligible) or not eligible:
            raise ValueError('flat binary baseline requires explicit remaining_slots and eligible teams')
        if target['team'] not in eligible:
            raise ValueError('unresolved binary target team must be eligible')
        p = slots / len(eligible)
        return {'0': 1-p, '1': p}
    if target['target_kind'] == 'placement':
        bands = target.get('bands')
        if not bands:
            raise ValueError('flat placements require explicit official band sizes')
        sizes = target['flat_band_sizes']
        total = sum(sizes.values())
        if total <= 0:
            raise ValueError('unresolved placement target has no remaining eligible slots')
        return {band['id']: sizes[band['id']] / total for band in bands}
    eligible = target.get('eligible_teams', list(vector))
    if not eligible or not set(eligible).issubset(vector):
        raise ValueError('flat baseline eligibility must be a nonempty subset of target categories')
    return {category: 1 / len(eligible) if category in eligible else 0.0 for category in vector}




def _row(event, origin, target, model, vector, simulation, provenance, *, precision_kwargs=None):
    if not vector or any(isinstance(p, bool) or not math.isfinite(p) or not 0 <= p <= 1 for p in vector.values()):
        raise ValueError('invalid target probability vector')
    if not math.isclose(sum(vector.values()), 1, abs_tol=1e-8):
        raise ValueError('target categorical probabilities must sum to one')
    row = {key: event[key] for key in ('tournament_id', 'phase_id', 'edition_start_at', 'family', 'tier', 'format')}
    row['forecast_scope'] = origin['forecast_scope']
    row['forecast_mode'] = origin['forecast_mode']
    row['phase_start_at'] = event.get('phase_start_at')
    row['edition_start_status'] = event.get('edition_start_status', 'provided_timestamp')
    row['phase_start_precision'] = event.get('phase_start_precision', 'provided_timestamp')
    row['reference_origin_id'] = origin.get('reference_origin_id')
    row['observation_day_end'] = _observation_day_end(origin)
    row['structural_zero_categories'] = target.get('structural_zero_categories', [])
    row['structural_zero_evidence'] = {
        category: f"Validated spec and replay-derived support for {target['source']}:{target['target_id']}"
        for category in row['structural_zero_categories']}
    row.update(origin_id=origin['origin_id'], axis=origin['axis'], cutoff=origin['cutoff'],
               target_id=target['target_id'], target_kind=target['target_kind'],
               target_family=target.get('target_family', target['target_kind']), model=model,
               probabilities=vector, resolved=target.get('resolved', False),
               target_start_at=target['target_start_at'], provenance=provenance)
    for key in ('best_of', 'horizon', 'team'):
        if key in target:
            row[key] = target[key]
    if target['target_kind'] in {'binary', 'series'}:
        row['categories'] = list(vector)
        row['positive_category'] = '1' if target['target_kind'] == 'binary' else target['team_b']
    if target['target_kind'] == 'placement':
        row['categories'] = [band['id'] for band in target['bands']]
    if target['target_kind'] in {'categorical', 'count'}:
        row['categories'] = target['categories']
    if simulation is not None:
        n = simulation['simulations']
        from scripts.score_tournament_forecasts import monte_carlo_precision
        precision = monte_carlo_precision(vector, n, structural_zeros=row['structural_zero_categories'],
                                          **(precision_kwargs or {}))
        precision['cap_reached'] = simulation.get('numerical_precision', {}).get('cap_reached', False)
        row.update(probability_source='monte_carlo', mc_samples=n,
                   mc_hits={key: round(p*n) for key, p in vector.items()},
                   mc_precision_resolved=precision['resolved'], mc_precision=precision)
    else:
        row['probability_source'] = 'analytic'
    return row


def _project_ledger(output):
    count = failures = 0
    temporary = output / 'forecasts.jsonl.tmp'
    with temporary.open('w', encoding='utf-8') as stream:
        for path in sorted((output / 'origins').iterdir()):
            if path.name.startswith('.pending-'):
                continue
            payload = _read_origin(path)
            for row in payload['forecasts']:
                stream.write(json.dumps(row, allow_nan=False, sort_keys=True) + '\n')
                count += 1
                failures += row.get('forecast_status') == 'failed'
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, output / 'forecasts.jsonl')
    return count, failures


def _run_evaluation(manifest_path, output_dir, *, simulations=20000, max_simulations=None, seed=81, resume=False):
    """Forecast explicit eligible origins; preserve partial results on real failures."""
    from src.models.tournament_formats import required_best_ofs, simulate_tournament
    manifest_path = Path(manifest_path).resolve()
    output = Path(output_dir).resolve()
    manifest = _load(manifest_path)
    _validate_manifest(manifest)
    if type(simulations) is not int or simulations < 1 or type(seed) is not int:
        raise ValueError('simulations must be positive and seed an integer')
    limit = simulations if max_simulations is None else max_simulations
    if type(limit) is not int or limit < simulations:
        raise ValueError('max_simulations must be an integer no smaller than simulations')
    looks, planned_samples = 1, simulations
    while planned_samples < limit:
        planned_samples = min(limit, planned_samples * 2)
        looks += 1
    base = manifest_path.parent
    inputs = {str(path): file_digest(path) for path in _source_paths(manifest, base)}
    for value, expected_hash in manifest.get('evidence_hashes', {}).items():
        if inputs[str(_path(base, value))] != expected_hash:
            raise ValueError('pinned source evidence hash changed since manifest compilation')
    code = {name: file_digest(Path(__file__).resolve().parents[1] / name) for name in (
        'scripts/tournament_evaluation.py', 'scripts/simulate_tournament.py',
        'src/models/tournament_formats.py', 'src/models/tournament_prediction.py',
        'scripts/prospective_sports_features.py', 'src/models/siamese_series.py',
        'scripts/score_tournament_forecasts.py')}
    protocol = {'version': VERSION, 'manifest_sha256': file_digest(manifest_path), 'input_sha256': inputs,
                'source_sha256': code, 'simulations': simulations, 'max_simulations': limit, 'seed': seed,
                'qualification': manifest['qualification'], 'python': platform.python_version()}
    protocol['numerical_precision'] = {
        'scope': 'all target categories and planned doubling looks within each origin',
        'alpha': .05, 'absolute_half_width': .005, 'log_half_width': .05,
        'looks': looks, 'primary_paired_delta_log_loss_half_width': .001}
    if output.exists():
        if not resume:
            raise ValueError('output exists; explicit resume is required')
        if _load(output / 'protocol.json') != protocol:
            raise ValueError('manifest, code, parameters or referenced input hashes changed; cannot resume')
    else:
        output.mkdir(parents=True)
        (output / 'origins').mkdir()
        _new(output / 'protocol.json', protocol)
        _new(output / 'input_manifest.json', manifest)
    expected = {json_digest([event['tournament_id'], event['phase_id'], origin['origin_id']])
                for event in manifest['events'] for origin in event['origins']}
    for saved in (output / 'origins').iterdir():
        if saved.name.startswith('.pending-'):
            continue
        if saved.name not in expected or not saved.is_dir():
            raise ValueError('unexpected saved origin record')
        _read_origin(saved)
    _atomic(output / 'status.json', {'status': 'running', 'started_at': datetime.now(timezone.utc).isoformat()})
    completed = 0
    try:
        for event in manifest['events']:
            spec = _load(_path(base, event['spec']))
            formats = sorted(required_best_ofs(spec))
            fair = {(a, b, bo): 0.5 for a, b in combinations(spec['teams'], 2) for bo in formats}
            initial = {}
            model_identity = {}
            for origin in event['origins']:
                key = json_digest([event['tournament_id'], event['phase_id'], origin['origin_id']])
                record_path = output / 'origins' / key
                current = {}
                excluded = origin.get('model_exclusions', {})
                for model in manifest['models']:
                    if model['id'] in excluded:
                        continue
                    probability, provenance = _probabilities(model, origin, event, spec, formats, base, manifest['qualification'])
                    current[model['id']] = (probability, provenance)
                    if model['kind'] == 'series_table' and manifest['qualification'] != 'synthetic_mechanics':
                        identity = provenance['supplied_provenance']['model_artifact_sha256']
                        if model['id'] in model_identity and identity != model_identity[model['id']]:
                            raise ValueError('model artifact changed within tournament; refresh features, not weights')
                        model_identity[model['id']] = identity
                if origin['axis'] == 'axis1':
                    initial[origin['origin_id']] = current
                if record_path.exists():
                    completed += 1
                    continue
                state = _load(_path(base, origin['state'])) if origin.get('state') else None
                if state is not None and utc(state['cutoff']) != utc(origin['cutoff']):
                    raise ValueError('state cutoff must match forecast origin exactly')
                _validate_mode_state(origin, spec, state, manifest['qualification'])
                if state is not None and manifest['qualification'] == 'prospective_capture':
                    if state.get('temporal_policy', 'strict_timestamped') != 'strict_timestamped':
                        raise ValueError('prospective capture requires strict_timestamped, not reconstructed state')
                methods = {'fair_series': (fair, {'baseline': 'same_state_fair_series'})}
                failures = {}
                if origin['axis'] == 'axis3':
                    frozen = initial[origin['reference_origin_id']]
                    for model in manifest['models']:
                        name = model['id']
                        if name in current:
                            methods[name + ':refreshed'] = current[name]
                        else:
                            failures[name + ':refreshed'] = excluded[name]
                        if name in frozen:
                            methods[name + ':frozen'] = frozen[name]
                        else:
                            failures[name + ':frozen'] = 'no eligible model matrix at declared reference origin'
                else:
                    methods.update(current)
                    failures.update(excluded)
                score_model = _score_model(origin, base)
                forecasts, results = [], {}
                observables = _observable_specs(origin, initial)
                cells = None
                for name, (probability, provenance) in methods.items():
                    kwargs = {'simulations': simulations, 'seed': seed, 'score_distributions': score_model,
                              'observables': observables}
                    if state is not None:
                        kwargs['state'] = state
                    while True:
                        from scripts.score_tournament_forecasts import monte_carlo_precision
                        result = simulate_tournament(spec, probability, **kwargs)
                        projected = []
                        for target in origin['targets']:
                            target = _condition_target(target, result)
                            projected.append((target, _target_vector(target, result, probability)))
                        if cells is None:
                            cells = len(methods) * sum(len(vector) for target, vector in projected
                                                       if not target['resolved'] and target['source'] != 'series_probability')
                        precise = all(
                            target['resolved'] or target['source'] == 'series_probability'
                            or monte_carlo_precision(
                                vector, kwargs['simulations'], looks=looks, cells=max(1, cells),
                                structural_zeros=target.get('structural_zero_categories', []))['resolved']
                            for target, vector in projected)
                        if kwargs['simulations'] >= limit or precise:
                            break
                        kwargs['simulations'] = min(limit, kwargs['simulations'] * 2)
                    provenance = {**provenance, 'state_qualification': result['state_qualification'],
                                  'spec_sha256': file_digest(_path(base, event['spec'])),
                                  'state_sha256': file_digest(_path(base, origin['state'])) if state is not None else None}
                    result['numerical_precision'] = {'resolved': precise, 'cap_reached': kwargs['simulations'] >= limit,
                                                     'looks': looks, 'cells': cells}
                    results[name] = result
                    for target, vector in projected:
                        if not target.get('resolved', False) and utc(target['target_start_at']) <= utc(origin['cutoff']):
                            raise ValueError('unresolved target start must follow forecast cutoff')
                        sampled = None if target['source'] == 'series_probability' or target['resolved'] else result
                        row = _row(event, origin, target, name, vector, sampled, provenance,
                                   precision_kwargs={'looks': looks, 'cells': max(1, cells)})
                        forecasts.append(row)
                        if name == 'fair_series':
                            for unavailable, reason in failures.items():
                                failed = {key: value for key, value in row.items() if not key.startswith('mc_')}
                                failed.update(model=unavailable, probabilities={}, probability_source='unspecified',
                                              forecast_status='failed', failure_reason=reason,
                                              provenance={'model_exclusion': reason,
                                                          'state_qualification': result['state_qualification']})
                                forecasts.append(failed)
                            flat = _flat_vector(target, vector, spec['teams'])
                            if flat is not None:
                                forecasts.append(_row(event, origin, target, 'flat', flat, None,
                                                      {'baseline': 'non_topological_flat_marginal',
                                                       'state_qualification': result['state_qualification']}))
                # No outcome file is read anywhere on this path.
                if {str(path): file_digest(path) for path in _source_paths(manifest, base)} != inputs:
                    raise ValueError('referenced inputs changed during evaluation')
                captured_at = datetime.now(timezone.utc)
                for row in forecasts:
                    if manifest['qualification'] == 'prospective_capture' and not row['resolved']:
                        if captured_at >= utc(row['target_start_at']):
                            raise ValueError('target started during forecast generation; capture is ineligible')
                    row['predicted_at'] = captured_at.isoformat()
                _commit_origin(record_path, {'origin_id': origin['origin_id'], 'tournament_id': event['tournament_id'],
                                      'phase_id': event['phase_id'],
                                      'created_at': datetime.now(timezone.utc).isoformat(),
                                      'forecasts': forecasts, 'simulations': results})
                completed += 1
        rows, failures = _project_ledger(output)
        status = {'status': 'completed' if completed else 'no_eligible_origins', 'completed_origins': completed,
                  'forecast_rows': rows, 'forecast_failures': failures, 'qualification': manifest['qualification'],
                  'forecast_sha256': file_digest(output / 'forecasts.jsonl'),
                  'production_qualified': False, 'finished_at': datetime.now(timezone.utc).isoformat(),
                  'exclusions': manifest.get('exclusions', [])}
        _atomic(output / 'status.json', status)
        return status
    except Exception as exc:
        # Persist a failed run and re-raise; this is not a fallback or successful empty result.
        rows, failures = _project_ledger(output)
        _atomic(output / 'status.json', {'status': 'failed', 'completed_origins': completed,
                                         'forecast_rows': rows, 'forecast_failures': failures,
                                         'error_type': type(exc).__name__, 'error': str(exc)})
        raise


def run_evaluation(manifest_path, output_dir, *, simulations=20000, max_simulations=None, seed=81, resume=False):
    """Hold an OS-released writer lock while forecasting or resuming one run."""
    output = Path(output_dir).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with (output.parent / ('.' + output.name + '.lock')).open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError('another writer owns this evaluation run') from exc
        return _run_evaluation(manifest_path, output, simulations=simulations, max_simulations=max_simulations,
                               seed=seed, resume=resume)


def score_evaluation(run_dir, outcomes_path, output_path, *, bootstrap=5000, seed=82):
    """Join outcome labels after the immutable forecast phase, with explicit pending rows."""
    from scripts.score_tournament_forecasts import evaluate_ledger
    run_dir = Path(run_dir)
    status = _load(run_dir / 'status.json')
    if status['status'] not in {'completed', 'no_eligible_origins'}:
        raise ValueError('failed/incomplete run cannot publish completed evaluation scores')
    if status.get('forecast_sha256') != file_digest(run_dir / 'forecasts.jsonl'):
        raise ValueError('persisted forecast ledger changed after completion')
    forecasts = [json.loads(line) for line in (run_dir / 'forecasts.jsonl').read_text().splitlines() if line]
    outcomes = _load(outcomes_path)
    result = evaluate_ledger(forecasts, outcomes, bootstrap=bootstrap, seed=seed)
    result['forecast_sha256'] = file_digest(run_dir / 'forecasts.jsonl')
    result['outcomes_sha256'] = file_digest(outcomes_path)
    result['qualification'] = _load(run_dir / 'protocol.json')['qualification']
    result['production_qualified'] = False
    _new(Path(output_path), result)
    return result
