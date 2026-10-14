#!/usr/bin/env python3
"""Export strict prior-day canonical79 inputs from local history and announcements.

Corrected rating/roster semantics are versioned, NOT certified compatible with the
frozen EXP089 weights. This tool neither scrapes nor fits/promotes a model.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import ExitStack
from datetime import datetime, timezone
import gzip
import hashlib
import importlib.metadata
import io
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from betting_app.services.current_roster_service import ROLE_ORDER
from betting_app.services.tournament_service import BracketMatchNode, TournamentBracket
from scripts.confirm_oddsfree_student import _environment, _read_sealed, _seal
from scripts.prospective_sports_features import FEATURE_VERSION, build_pair_features
from scripts.tournament_research_snapshot import _local_file, _references, _utc, _validate_inputs
from src.models.frozen_tournament import validate_bracket
from src.models.symmetric_series import _PROBABILITY_FIELDS, _REQUIRED_BASE_FIELDS
from src.models.team_order import pair_columns

VERSION = 'prospective-feature-export-v1'
CODE_FILES = (
    'scripts/export_prospective_features.py', 'scripts/prospective_sports_features.py',
    'scripts/05_ratingi_baseline/03_generate_ratings.py',
    'scripts/06_metamodel/06i_best_metamodel_config_search.py',
    'scripts/build_siamese_research_dataset.py', 'scripts/tournament_research_snapshot.py',
    'scripts/confirm_oddsfree_student.py', 'src/utils/golgg_schema.py',
    'src/models/symmetric_series.py', 'src/models/team_order.py', 'src/models/frozen_tournament.py',
    'betting_app/services/current_roster_service.py', 'betting_app/services/tournament_service.py',
    'src/ratings/base.py', 'src/ratings/manager.py', 'src/ratings/elo.py', 'src/ratings/glicko.py',
    'src/ratings/trueskill_rating.py', 'src/ratings/openskill_rating.py',
    'src/ratings/plackett_luce.py', 'src/ratings/thurstone.py',
)
FIXTURE_FIELDS = {'golgg_match_id', 'team_a', 'team_b', 'best_of', 'match_start_at'}


def iter_matches(path: Path):
    """Read a JSON array one object at a time, without loading the multi-GB cache."""
    decoder = json.JSONDecoder()
    with ExitStack() as stack:
        source = stack.enter_context(Path(path).open('rb'))
        if source.peek(2)[:2] == b'\x1f\x8b':
            source = stack.enter_context(gzip.GzipFile(fileobj=source))
        stream = stack.enter_context(io.TextIOWrapper(source, encoding='utf-8-sig'))
        buffer, eof = '', False

        def fill():
            nonlocal buffer, eof
            chunk = stream.read(65536)
            eof = not chunk
            buffer += chunk

        def space():
            nonlocal buffer
            buffer = buffer.lstrip()
            while not buffer and not eof:
                fill()
                buffer = buffer.lstrip()

        space()
        if not buffer.startswith('['):
            raise ValueError('history must be a JSON array of match objects')
        buffer = buffer[1:]
        first = True
        while True:
            space()
            if first and buffer.startswith(']'):
                buffer = buffer[1:]
                break
            while True:
                try:
                    value, end = decoder.raw_decode(buffer)
                    break
                except json.JSONDecodeError as exc:
                    if eof:
                        raise ValueError('malformed or truncated historical JSON array') from exc
                    fill()
            if not isinstance(value, dict):
                raise ValueError('historical array entries must be match objects')
            buffer = buffer[end:]
            yield value
            first = False
            space()
            if buffer.startswith(']'):
                buffer = buffer[1:]
                break
            if not buffer.startswith(','):
                raise ValueError('history objects require a comma or closing array')
            buffer = buffer[1:]
        space()
        if buffer:
            raise ValueError('unexpected content after historical JSON array')


def _hash_file(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def _json_bytes(value):
    return (json.dumps(value, indent=2, allow_nan=False)+'\n').encode()


def _save(directory, relative, content, hashes):
    path = directory/relative
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as handle:
        handle.write(content)
    hashes[relative] = hashlib.sha256(content).hexdigest()


def _archive_history(source, destination):
    # mtime/inode are change detectors only, never claimed source-availability times.
    digest = hashlib.sha256()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open('rb') as original, destination.open('xb') as archived:
        before = source.stat()
        for chunk in iter(lambda: original.read(1024*1024), b''):
            digest.update(chunk)
            archived.write(chunk)
        after = source.stat()
    if (before.st_ino,before.st_size,before.st_mtime_ns) != (after.st_ino,after.st_size,after.st_mtime_ns):
        raise ValueError('history changed during archival; retry into a new directory')
    return digest.hexdigest()


def _announcements(source, now):
    payloads = {name:_local_file(source,name).read_bytes() for name in ('event.json','bracket.json')}
    event = json.loads(payloads['event.json'])
    definition = json.loads(payloads['bracket.json'])
    bracket = TournamentBracket(matches={key:BracketMatchNode(**value) for key,value in definition.pop('matches').items()},**definition)
    validate_bracket(bracket)
    if not now < _utc(event['event_start_at']):
        raise ValueError('feature export must precede the whole event')
    if _utc(event['bracket_announced_at']) > now:
        raise ValueError('bracket announcement is not yet available')
    evidence = event.get('evidence_files')
    if not isinstance(evidence,list) or not evidence or any(not isinstance(name,str) for name in evidence):
        raise ValueError('explicit local announcement evidence is required')
    reserved = {'event.json','bracket.json','pair_features.csv','feature_export.json','fixtures.csv'}
    if len(set(evidence)) != len(evidence) or set(evidence)&reserved:
        raise ValueError('evidence must be unique and separate from generated files')
    _references(event.get('bracket_sources'),evidence)
    if event.get('feature_sources'):
        _references(event['feature_sources'],evidence)
    if not isinstance(event.get('rosters'),dict) or set(event['rosters']) != set(bracket.teams):
        raise ValueError('every exact source team ID requires an announced roster')
    rosters = {}
    for team,roster in event['rosters'].items():
        _references(roster.get('source_files'),evidence)
        if _utc(roster['announced_at']) > now:
            raise ValueError('roster announcement is not yet available')
        if not isinstance(roster.get('players'),dict) or set(roster['players']) != set(ROLE_ORDER):
            raise ValueError('announced rosters require five explicit role assignments')
        rosters[team] = [roster['players'][role] for role in ROLE_ORDER]
    for name in evidence:
        payloads[name] = _local_file(source,name).read_bytes()
    if (source/'fixtures.csv').exists():
        payloads['fixtures.csv'] = _local_file(source,'fixtures.csv').read_bytes()
    return event,bracket,rosters,payloads


def _fixture_outputs(content, frame, bracket, observed, event_start):
    fixtures = pd.read_csv(io.BytesIO(content),dtype={'golgg_match_id':str,'team_a':str,'team_b':str})
    if set(fixtures.columns) != FIXTURE_FIELDS or not fixtures.columns.is_unique or fixtures.empty:
        raise ValueError('fixtures require exact ID, side, format and start columns')
    ids = fixtures.golgg_match_id
    if ids.isna().any() or ids.duplicated().any() or any(not v.strip() or v != v.strip() for v in ids):
        raise ValueError('fixture IDs must be unique nonempty exact strings')
    seeded = Counter((frozenset((node.team1,node.team2)),node.best_of) for node in bracket.matches.values() if node.team1 and node.team2)
    lookup = {(r['team_a'],r['team_b'],r['best_of']):r for r in frame.to_dict('records')}
    raw,metadata = [],[]
    for fixture in fixtures.to_dict('records'):
        a,b,bo = fixture['team_a'],fixture['team_b'],fixture['best_of']
        key = (frozenset((a,b)),bo)
        if seeded[key] == 0:
            raise ValueError('only explicitly seeded actual fixtures may become series forecasts')
        seeded[key] -= 1
        if not observed < _utc(fixture['match_start_at']) or _utc(fixture['match_start_at']) < event_start:
            raise ValueError('fixture must be unstarted and cannot precede its event')
        lo,hi = sorted((a,b))
        values = lookup[lo,hi,bo]
        row = {field:values[field] for field in sorted(_REQUIRED_BASE_FIELDS)}
        if a != lo:
            for field in _PROBABILITY_FIELDS:
                row[field] = 1-row[field]
            for left,right in pair_columns(sorted(_REQUIRED_BASE_FIELDS)):
                row[left],row[right] = row[right],row[left]
        raw.append({'golgg_match_id':fixture['golgg_match_id'],'best_of':bo,**row})
        # Extra contract columns intentionally fail the legacy confirmation reader.
        # Dropping them would misrepresent corrected features as frozen-compatible.
        metadata.append({'golgg_match_id':fixture['golgg_match_id'],'match_start_at':fixture['match_start_at'],'feature_history_max_at':values['feature_history_max_at'],'feature_version':FEATURE_VERSION,'eligible_for_frozen_confirmation':False})
    return {'sports.csv':pd.DataFrame(raw).to_csv(index=False).encode(), 'metadata.csv':pd.DataFrame(metadata).to_csv(index=False).encode()}


def _runtime():
    return {**_environment(), **{name:importlib.metadata.version(name) for name in ('glicko2','trueskill','openskill')}}


def export_features(history_json: Path, event_dir: Path, output_dir: Path) -> dict:
    history_json,event_dir,output_dir = Path(history_json),Path(event_dir),Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(output_dir)
    event,bracket,rosters,payloads = _announcements(event_dir,datetime.now(timezone.utc))
    output_dir.mkdir(parents=True,exist_ok=False)
    hashes = {}
    hashes['source/history.json'] = _archive_history(history_json,output_dir/'source/history.json')
    for name,content in payloads.items():
        _save(output_dir,'source/announcements/'+name,content,hashes)
    source_observed = datetime.now(timezone.utc)
    if not source_observed < _utc(event['event_start_at']):
        raise ValueError('event started during source capture')
    formats = sorted({node.best_of for node in bracket.matches.values()})
    frame,audit = build_pair_features(iter_matches(output_dir/'source/history.json'),rosters,source_observed.date(),formats)
    for name in CODE_FILES:
        _save(output_dir,'source/code/'+name,(ROOT/name).read_bytes(),hashes)
    manifest = {
        'version':VERSION,'feature_version':FEATURE_VERSION,'source_available_at':source_observed.isoformat(),
        'feature_asof_date':source_observed.date().isoformat(),'history':audit,
        'source_sha256':dict(hashes),'environment':_runtime(),
        'eligible_for_frozen_confirmation':False,'production_qualified':False,
        'frozen_model_compatibility':'unverified: corrected Glicko updates and per-game historical rosters differ from legacy feature generation',
        'limitations':['Date-only history uses strictly earlier source calendar days; intraday ordering and external publication times are not certified.',
                       'Announcement timestamps are source assertions backed by locally observed files, not independent publication certification.',
                       'Frozen-model inference on this new feature version is a compatibility diagnostic, not new confirmation evidence.'],
    }
    feature_evidence = _json_bytes(manifest)
    event['evidence_files'].append('feature_export.json')
    event['feature_version'] = FEATURE_VERSION
    event['feature_sources'] = [*event.get('feature_sources',[]),'feature_export.json']
    event_payloads = {name:content for name,content in payloads.items() if name != 'fixtures.csv'}
    event_payloads.update({'event.json':_json_bytes(event),'pair_features.csv':frame.to_csv(index=False).encode(),'feature_export.json':feature_evidence})
    completed = datetime.now(timezone.utc)
    _validate_inputs(event_payloads,completed)
    for name,content in event_payloads.items():
        _save(output_dir,'event_inputs/'+name,content,hashes)
    if 'fixtures.csv' in payloads:
        for name,content in _fixture_outputs(payloads['fixtures.csv'],frame,bracket,completed,_utc(event['event_start_at'])).items():
            _save(output_dir,name,content,hashes)
    manifest.update(exported_at=completed.isoformat(),files=hashes)
    _seal(output_dir,'export',manifest)
    return manifest


def verify_export(export_dir: Path) -> dict:
    export_dir = Path(export_dir)
    manifest,_ = _read_sealed(export_dir,'export',expected_version=VERSION)
    if manifest['feature_version'] != FEATURE_VERSION or manifest['environment'] != _runtime():
        raise ValueError('feature or numerical environment contract changed')
    for name,digest in manifest['files'].items():
        if _hash_file(_local_file(export_dir,name)) != digest:
            raise ValueError(f'archived export bytes changed: {name}')
    for name in CODE_FILES:
        if _hash_file(ROOT/name) != manifest['files']['source/code/'+name]:
            raise ValueError(f'feature construction source changed: {name}')
    observed = _utc(manifest['source_available_at'])
    event,bracket,rosters,payloads = _announcements(export_dir/'source/announcements',observed)
    if manifest['feature_asof_date'] != observed.date().isoformat():
        raise ValueError('feature cutoff differs from actual source observation')
    frame,audit = build_pair_features(iter_matches(export_dir/'source/history.json'),rosters,observed.date(),sorted({n.best_of for n in bracket.matches.values()}))
    saved = pd.read_csv(export_dir/'event_inputs/pair_features.csv',dtype={'team_a':str,'team_b':str},float_precision='round_trip')
    try:
        pd.testing.assert_frame_equal(frame,saved,check_dtype=False,check_exact=True)
    except AssertionError as exc:
        raise ValueError('archived features do not reconstruct from source history') from exc
    if audit != manifest['history']:
        raise ValueError('history audit does not reconstruct')
    if 'fixtures.csv' in payloads:
        for name,content in _fixture_outputs(payloads['fixtures.csv'],frame,bracket,observed,_utc(event['event_start_at'])).items():
            if content != (export_dir/name).read_bytes():
                raise ValueError('fixture features do not reconstruct')
    return {'verified':True,'feature_version':FEATURE_VERSION,'pair_format_rows':len(frame),'eligible_for_frozen_confirmation':False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command',required=True)
    export = commands.add_parser('export')
    export.add_argument('--history-json',type=Path,required=True)
    export.add_argument('--event-dir',type=Path,required=True)
    export.add_argument('--output-dir',type=Path,required=True)
    verify = commands.add_parser('verify')
    verify.add_argument('--export-dir',type=Path,required=True)
    args = vars(parser.parse_args())
    command = args.pop('command')
    result = export_features(**args) if command == 'export' else verify_export(**args)
    print(json.dumps(result,indent=2,allow_nan=False))


if __name__ == '__main__':
    main()
