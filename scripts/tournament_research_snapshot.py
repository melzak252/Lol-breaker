#!/usr/bin/env python3
"""Capture prospective odds-free tournament state; replay its fixed bracket offline.

Input directory: event.json, bracket.json, pair_features.csv and referenced source
files. No database, scraper, current-rating fallback or historical backdating.
"""
from __future__ import annotations

import argparse
from datetime import UTC, datetime
import hashlib
import io
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from betting_app.services.current_roster_service import ROLE_ORDER
from betting_app.services.tournament_service import BracketMatchNode, TournamentBracket
from scripts.confirm_oddsfree_student import _read_sealed, _seal, load_protocol, predict_frozen
from src.models.frozen_tournament import simulate_frozen_bracket, validate_bracket
from src.models.symmetric_series import _PROBABILITY_FIELDS, _REQUIRED_BASE_FIELDS, _W20_FIELDS

VERSION = 'oddsfree-tournament-snapshot-v1'
MODELS = ('outcome', 'market_w10')
QUALIFICATION = 'prospective_local_capture_not_source_certified'
INPUT_FILES = ('event.json', 'bracket.json', 'pair_features.csv')
CODE_FILES = ('scripts/tournament_research_snapshot.py', 'src/models/frozen_tournament.py',
              'betting_app/services/tournament_service.py', 'betting_app/services/current_roster_service.py')


def _utc(value):
    if not isinstance(value, str):
        raise ValueError('timestamp must be an explicit timezone-aware string')
    result = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if result.tzinfo is None or result.utcoffset() is None:
        raise ValueError('naive timestamps are not eligible')
    return result.astimezone(UTC)


def _digest(content):
    return hashlib.sha256(content).hexdigest()




def _local_file(directory, name):
    if not isinstance(name, str) or not name or Path(name).is_absolute() or '..' in Path(name).parts:
        raise ValueError('evidence paths must remain inside the input directory')
    path = directory / name
    if path.is_symlink() or not path.resolve().is_relative_to(directory.resolve()):
        raise ValueError('evidence paths must not traverse symlinks outside the input directory')
    if not path.is_file():
        raise ValueError(f'missing input file: {name}')
    return path


def _read_inputs(directory):
    payloads = {name: _local_file(directory, name).read_bytes() for name in INPUT_FILES}
    event = json.loads(payloads['event.json'])
    evidence = event.get('evidence_files')
    if not isinstance(evidence, list) or not evidence or not all(isinstance(n, str) for n in evidence):
        raise ValueError('source evidence files must be explicitly listed')
    if len(set(evidence)) != len(evidence) or set(evidence) & set(INPUT_FILES):
        raise ValueError('source evidence must be unique and separate from generated inputs')
    for name in evidence:
        payloads[name] = _local_file(directory, name).read_bytes()
    return payloads


def _references(refs, evidence):
    if not isinstance(refs, list) or not refs or not all(isinstance(ref, str) for ref in refs):
        raise ValueError('every bracket, roster and feature state needs source evidence')
    if len(set(refs)) != len(refs) or not set(refs).issubset(evidence):
        raise ValueError('source references must name unique archived evidence files')


def _validate_inputs(payloads, observed_at):
    event = json.loads(payloads['event.json'])
    start = _utc(event['event_start_at'])
    if not observed_at < start:
        raise ValueError('capture must precede tournament start; historical backdating is forbidden')
    definition = json.loads(payloads['bracket.json'])
    nodes = {key: BracketMatchNode(**value) for key, value in definition.pop('matches').items()}
    bracket = TournamentBracket(matches=nodes, **definition)
    validate_bracket(bracket)
    evidence = set(event['evidence_files'])
    _references(event.get('bracket_sources'), evidence)
    _references(event.get('feature_sources'), evidence)
    rosters = event.get('rosters')
    if not isinstance(rosters, dict) or set(rosters) != set(bracket.teams):
        raise ValueError('each bracket entrant requires a frozen roster')
    players = set()
    for roster in rosters.values():
        if not isinstance(roster, dict):
            raise ValueError('roster must declare role-assigned players and source files')
        _references(roster.get('source_files'), evidence)
        roles = roster.get('players')
        if not isinstance(roles, dict) or set(roles) != set(ROLE_ORDER):
            raise ValueError('roster requires exactly TOP, JUNGLE, MID, ADC, SUPPORT')
        values = list(roles.values())
        if any(not isinstance(value, str) or not value.strip() or value != value.strip() for value in values):
            raise ValueError('player identities must be nonempty explicit strings')
        if len(set(values)) != 5 or players.intersection(values):
            raise ValueError('players must be unique across frozen event rosters')
        players.update(values)
    frame = pd.read_csv(io.BytesIO(payloads['pair_features.csv']), dtype={'team_a': str, 'team_b': str})
    required = set(_REQUIRED_BASE_FIELDS) | {'team_a', 'team_b', 'best_of', 'feature_history_max_at'}
    if set(frame.columns) != required or not frame.columns.is_unique or frame.empty:
        raise ValueError('pair features require only the complete canonical sports schema and pair/history identities')
    if frame[['team_a', 'team_b', 'best_of']].isna().any().any():
        raise ValueError('pair identities and formats must be complete')
    formats = {node.best_of for node in bracket.matches.values()}
    expected = {(frozenset((a, b)), bo) for i, a in enumerate(bracket.teams)
                for b in bracket.teams[i+1:] for bo in formats}
    seen, frozen_values, pair_values = set(), {}, {}
    # Raw team/roster summaries and W20 describe the same state in every pairing.
    paired_fields = [(field, field[:-1]+'2') for field in sorted(_REQUIRED_BASE_FIELDS)
                     if field.endswith('1') and field[:-1]+'2' in _REQUIRED_BASE_FIELDS]
    paired_fields += [(f't1_rolling_{field}', f't2_rolling_{field}') for field in _W20_FIELDS]
    for row in frame.to_dict('records'):
        a, b, bo = row['team_a'], row['team_b'], row['best_of']
        key = (frozenset((a, b)), bo)
        if a == b or key not in expected or key in seen:
            raise ValueError('all unordered entrant pairings/formats must occur exactly once')
        seen.add(key)
        history = _utc(row['feature_history_max_at'])
        if history > observed_at:
            raise ValueError('feature history cannot include information after capture')
        raw_values = np.asarray([row[field] for field in sorted(_REQUIRED_BASE_FIELDS)], dtype=float)
        if not np.isfinite(raw_values).all():
            raise ValueError('raw sports features must be finite, never silently imputed')
        ordered_pair = tuple(sorted((a, b)))
        pair_probabilities = np.asarray([row[field] for field in _PROBABILITY_FIELDS], dtype=float)
        if a != ordered_pair[0]:
            pair_probabilities = 1 - pair_probabilities
        if ordered_pair in pair_values and not np.allclose(pair_values[ordered_pair], pair_probabilities, atol=1e-12, rtol=0):
            raise ValueError('pair map probabilities must be frozen across series formats and orientations')
        pair_values[ordered_pair] = pair_probabilities
        for field_a, field_b in paired_fields:
            for team, field in ((a, field_a), (b, field_b)):
                state_key = (team, field_a)
                value = float(row[field])
                if state_key in frozen_values and frozen_values[state_key] != value:
                    raise ValueError('team/roster state differs between frozen pairings')
                frozen_values[state_key] = value
    if seen != expected:
        raise ValueError('missing counterfactual pairing/format; realized fixtures are insufficient')
    return bracket, frame, start


def _probabilities(frame, predictions, model):
    return {(row.team_a, row.team_b, int(row.best_of)): float(probability)
            for row, probability in zip(frame.itertuples(index=False), predictions['p__'+model], strict=True)}


def capture_tournament(protocol_dir: Path, input_dir: Path, output_dir: Path) -> dict:
    protocol_dir, input_dir, output_dir = Path(protocol_dir), Path(input_dir), Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(output_dir)
    protocol = load_protocol(protocol_dir)
    payloads = _read_inputs(input_dir)
    observed_at = datetime.now(UTC)
    if _utc(protocol['locked_at']) > observed_at:
        raise ValueError('model comparison must be frozen before forecast capture')
    bracket, frame, start = _validate_inputs(payloads, observed_at)
    predictions = predict_frozen(protocol_dir, frame[sorted(_REQUIRED_BASE_FIELDS | {'best_of'})])
    predicted_at = datetime.now(UTC)
    if not observed_at <= predicted_at < start:
        raise ValueError('prediction finished after event start or clock moved backwards')
    table = pd.concat([frame[['team_a', 'team_b', 'best_of']], predictions], axis=1)
    encoded = table.to_csv(index=False).encode()
    code_payloads = {name: (ROOT/name).read_bytes() for name in CODE_FILES}
    metadata = {
        'version': VERSION, 'qualification': QUALIFICATION,
        'tournament_id': bracket.id, 'event_start_at': start.isoformat(),
        'feature_version': json.loads(payloads['event.json']).get('feature_version','unversioned_source_supplied'),
        'eligible_for_frozen_confirmation': False,
        'source_available_at': observed_at.isoformat(), 'data_cutoff_at': observed_at.isoformat(),
        'predicted_at': predicted_at.isoformat(), 'probability_unit': 'series_win',
        'protocol_sha256': _digest((protocol_dir/'protocol.json').read_bytes()),
        'input_sha256': {name: _digest(content) for name, content in payloads.items()},
        'predictions_sha256': _digest(encoded),
        'source_sha256': {name: _digest(content) for name, content in code_payloads.items()},
        'assumptions': ['All sports state and rosters remain frozen throughout the event.',
                        'Independent series draws conditional on frozen pair probabilities; no within-event updates.',
                        'Source content observed locally before start; announcement authenticity and feature derivation are not certified.',
                        'Fixed single/double-elimination graph only; no implicit reset, Swiss draw or map-to-series expansion.'],
        'production_qualified': False,
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    for name, content in payloads.items():
        destination = output_dir/'inputs'/name
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open('xb') as handle:
            handle.write(content)
    for name, content in code_payloads.items():
        destination = output_dir/'source_snapshot'/name
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open('xb') as handle:
            handle.write(content)
    with (output_dir/'predictions.csv').open('xb') as handle:
        handle.write(encoded)
    _seal(output_dir, 'snapshot', metadata)
    return metadata


def replay_tournament(protocol_dir: Path, snapshot_dir: Path, output_dir: Path,
                      simulations: int = 20000, seed: int = 89) -> dict:
    protocol_dir, snapshot_dir, output_dir = Path(protocol_dir), Path(snapshot_dir), Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(output_dir)
    protocol = load_protocol(protocol_dir)
    metadata, snapshot_hash = _read_sealed(snapshot_dir, 'snapshot', expected_version=VERSION)
    if metadata['qualification'] != QUALIFICATION:
        raise ValueError('unsupported tournament snapshot contract')
    if metadata['protocol_sha256'] != _digest((protocol_dir/'protocol.json').read_bytes()):
        raise ValueError('frozen model protocol hash mismatch')
    for name, expected in metadata['source_sha256'].items():
        if name not in CODE_FILES or _digest((ROOT/name).read_bytes()) != expected:
            raise ValueError('snapshot inference or graph source changed')
        if _digest(_local_file(snapshot_dir/'source_snapshot', name).read_bytes()) != expected:
            raise ValueError('archived snapshot source hash mismatch')
    if set(metadata['source_sha256']) != set(CODE_FILES):
        raise ValueError('snapshot source manifest is incomplete')
    payloads = _read_inputs(snapshot_dir/'inputs')
    if {name: _digest(content) for name, content in payloads.items()} != metadata['input_sha256']:
        raise ValueError('archived tournament input hash mismatch')
    observed = _utc(metadata['source_available_at'])
    cutoff, predicted = _utc(metadata['data_cutoff_at']), _utc(metadata['predicted_at'])
    if not _utc(protocol['locked_at']) <= observed <= cutoff <= predicted <= datetime.now(UTC):
        raise ValueError('snapshot chronology is invalid')
    bracket, frame, start = _validate_inputs(payloads, observed)
    if bracket.id != metadata['tournament_id'] or start != _utc(metadata['event_start_at']) or predicted >= start:
        raise ValueError('snapshot event identity or pre-start prediction contract failed')
    encoded = (snapshot_dir/'predictions.csv').read_bytes()
    if _digest(encoded) != metadata['predictions_sha256']:
        raise ValueError('archived prediction hash mismatch')
    saved = pd.read_csv(io.BytesIO(encoded), dtype={'team_a': str, 'team_b': str})
    expected_columns = ['team_a', 'team_b', 'best_of', 'p__outcome', 'p__market_w10']
    if saved.columns.tolist() != expected_columns or len(saved) != len(frame):
        raise ValueError('archived predictions have a different pairing schema')
    if not saved[['team_a', 'team_b', 'best_of']].equals(frame[['team_a', 'team_b', 'best_of']]):
        raise ValueError('archived prediction pairing identities changed')
    recomputed = predict_frozen(protocol_dir, frame[sorted(_REQUIRED_BASE_FIELDS | {'best_of'})])
    if not np.allclose(saved[['p__outcome', 'p__market_w10']], recomputed, atol=1e-12, rtol=0):
        raise ValueError('saved probabilities disagree with archived odds-free inputs')
    result = {
        'tournament_id': bracket.id, 'snapshot_sha256': snapshot_hash,
        'protocol_sha256': metadata['protocol_sha256'], 'qualification': QUALIFICATION,
        'models': {model: simulate_frozen_bracket(bracket, _probabilities(frame, recomputed, model), simulations, seed)
                   for model in MODELS},
        'interpretation': 'Replay of a pre-start frozen forecast, not validation against observed tournament outcomes.',
        'feature_version': metadata['feature_version'],
        'eligible_for_frozen_confirmation': False,
        'production_qualified': False,
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    _seal(output_dir, 'simulation', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest='command', required=True)
    capture = subparsers.add_parser('capture')
    capture.add_argument('--protocol-dir', type=Path, required=True)
    capture.add_argument('--input-dir', type=Path, required=True)
    capture.add_argument('--output-dir', type=Path, required=True)
    replay = subparsers.add_parser('replay')
    replay.add_argument('--protocol-dir', type=Path, required=True)
    replay.add_argument('--snapshot-dir', type=Path, required=True)
    replay.add_argument('--output-dir', type=Path, required=True)
    replay.add_argument('--simulations', type=int, default=20000)
    replay.add_argument('--seed', type=int, default=89)
    args = parser.parse_args()
    if args.command == 'capture':
        result = capture_tournament(args.protocol_dir, args.input_dir, args.output_dir)
    else:
        result = replay_tournament(args.protocol_dir, args.snapshot_dir, args.output_dir, args.simulations, args.seed)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
