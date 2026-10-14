"""Map historical odds by independent team identities; reject ambiguous fixtures.

Date-only matching is retrospective. It does not certify quote availability before
prediction or match start. Existing exports and automatically learned aliases must
not be overwritten or silently treated as trustworthy inputs.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
import hashlib
import io
import json
import os
from pathlib import Path
import re
import sys
import unicodedata
from itertools import product

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils import golgg_schema
from src.utils.oddsportal_data import event_id_from_url

MAPPING_VERSION = 'identity-only-v5'
QUOTE_COLUMN = re.compile(r'odds[12]_([a-z0-9_]+)_(open|close)(_at)?')
SOURCE_COLUMNS = (
    'event_id', 'date', 'home_team', 'away_team', 'homeResult', 'awayResult',
    'url', 'tournament', 'odds_status', 'date_status', 'canonical_metadata_source',
    'quote_timing_semantics', 'odds_capture_completed_at', 'shell_capture_completed_at',
    'source_home_participant_id', 'source_away_participant_id', 'source_participant_identity_status',
)

BOOKMAKERS = ('betclic', 'betfan', 'efortuna', 'lv_bet', 'sts', 'superbet', 'fuksiarz')
QUOTE_PAIRS = (
    ('avg_odds_home', 'avg_odds_away'),
    ('avg_open_home', 'avg_open_away'),
    *((f'odds1_{book}_{period}', f'odds2_{book}_{period}')
      for book in BOOKMAKERS for period in ('close', 'open')),
)
PROJECTED_COLUMNS = frozenset((*SOURCE_COLUMNS, *(column for pair in QUOTE_PAIRS for column in pair)))
PROVENANCE_COLUMNS = (
    'source_row', 'source_date', 'source_home_team', 'source_away_team',
    'source_home_result', 'source_away_result', 'source_sides_swapped',
    'date_offset_days', 'mapping_version', 'source_event_id', 'source_date_basis',
    'source_odds_status', 'source_date_status', 'source_canonical_metadata_source',
    'quote_timing_semantics', 'odds_capture_completed_at', 'shell_capture_completed_at',
    'source_home_participant_id', 'source_away_participant_id', 'source_participant_identity_status',
    'source_identity_home', 'source_identity_away', 'source_alias_ids',
    'golgg_team1_id', 'golgg_team2_id', 'golgg_tournament',
)
OUTPUT_COLUMNS = (
    'golgg_match_id', 'odds_date', 'golgg_date', 'golgg_team1', 'golgg_team2',
    't1_score', 't2_score', 't1_win', 't2_win', 'draw', 'match_type', 'tournament',
    *(column for pair in QUOTE_PAIRS for column in pair), 'oddsportal_url',
    *PROVENANCE_COLUMNS,
)


def normalize_team_name(name: str | None) -> str:
    """Normalize generic organization words, retaining academy and squad identity."""
    if not isinstance(name, str):
        return ''
    name = unicodedata.normalize('NFKD', name)
    name = ''.join(character for character in name if not unicodedata.combining(character))
    name = re.sub(r'\b(esports|gaming|team|club|e-sports|esport)\b', '', name.lower())
    return re.sub(r'[^a-z0-9]', '', name)


def load_aliases(filepath: str | Path) -> tuple[dict[str, str], dict[tuple[str, str], list[dict]]]:
    """Load explicitly reviewed aliases; reject the old outcome-learned format.

    Expected JSON: {"reviewed": true, "aliases": {...}, "scoped_aliases": [...]}.
    Review is an explicit attestation, not proof supplied by this program.
    Scoped rules require a source participant ID, inclusive date bounds and evidence.
    """
    with Path(filepath).open(encoding='utf-8') as file:
        payload = json.load(file)
    if not isinstance(payload, dict) or payload.get('reviewed') is not True or not isinstance(payload.get('aliases'), dict):
        raise ValueError('Aliases must be explicitly reviewed; legacy learned aliases are unsafe.')
    aliases = {}
    for source, target in payload['aliases'].items():
        key, value = normalize_team_name(source), normalize_team_name(target)
        if not key or not value:
            raise ValueError('Reviewed aliases require nonempty team identities.')
        if key in aliases and aliases[key] != value:
            raise ValueError(f'Conflicting reviewed aliases for {source!r}.')
        aliases[key] = value
    for value in aliases.values():
        if value in aliases and aliases[value] != value:
            raise ValueError('Reviewed aliases must point directly to canonical identities, not chains or cycles.')
    scoped = defaultdict(list)
    seen_rules = set()
    rules = payload.get('scoped_aliases', [])
    if not isinstance(rules, list):
        raise ValueError('scoped_aliases must be a list of reviewed identity rules.')
    for raw in rules:
        if not isinstance(raw, dict):
            raise ValueError('Each scoped alias must be an object.')
        rule = dict(raw)
        rule_id = rule.get('id')
        source_key = normalize_team_name(rule.get('source_name'))
        target_key = normalize_team_name(rule.get('target_name'))
        participant_id = _participant_id(rule.get('source_participant_id'))
        if not isinstance(rule_id, str) or not rule_id or rule_id in seen_rules:
            raise ValueError('Scoped aliases require unique nonempty rule IDs.')
        if not source_key or not target_key or participant_id is None:
            raise ValueError('Scoped aliases require both names and a source participant ID.')
        start = datetime.strptime(rule['valid_from'], '%Y-%m-%d').date()
        end = datetime.strptime(rule['valid_to'], '%Y-%m-%d').date()
        if start > end:
            raise ValueError('Scoped alias date bounds are reversed.')
        evidence = rule.get('evidence')
        if not isinstance(evidence, list) or not evidence or any(
                not isinstance(url, str) or not url.startswith(('https://', 'http://')) for url in evidence):
            raise ValueError('Scoped aliases require explicit evidence URLs.')
        pattern = rule.get('tournament_pattern')
        if pattern is not None and (not isinstance(pattern, str) or not normalize_team_name(pattern)):
            raise ValueError('Scoped tournament patterns must be nonempty strings.')
        rule.update(source_key=source_key, target_key=target_key,
                    participant_id=participant_id, start=start, end=end)
        scoped[(source_key, participant_id)].append(rule)
        seen_rules.add(rule_id)
    return aliases, dict(scoped)


def apply_alias(name: str, aliases: dict[str, str]) -> str:
    return aliases.get(name, name)


def _participant_id(value) -> str | None:
    if value is None or pd.isna(value):
        return None
    number = _source_score(value)
    if number == 0:
        raise ValueError('OddsPortal participant IDs must be positive integers.')
    return str(number)


def _source_name_variants(name: str, participant_id: str | None, day, tournament: str,
                          aliases: dict[str, str], scoped: dict) -> dict[str, list[str]]:
    """Retain direct spelling plus only source-ID/date-qualified reviewed labels."""
    source_key = normalize_team_name(name)
    variants = {apply_alias(source_key, aliases): []}
    for rule in scoped.get((source_key, participant_id), ()):
        if not rule['start'] <= day <= rule['end']:
            continue
        pattern = rule.get('tournament_pattern')
        if pattern is not None and normalize_team_name(pattern) not in normalize_team_name(tournament):
            continue
        target = apply_alias(rule['target_key'], aliases)
        variants.setdefault(target, []).append(rule['id'])
    return variants


def _identity_candidates(by_identity, day, variants, offset=0) -> list:
    """Deduplicate graph edges without choosing among labels or orientations."""
    found = {}
    for names in product(*variants):
        if names[0] == names[1]:
            continue
        for game, canonical in by_identity.get((day, tuple(sorted(names))), ()):
            key = game['match_id']
            if key not in found:
                found[key] = (game, canonical, offset, set())
            found[key][3].add(names != canonical)
    return [found[key] for key in sorted(found, key=str)]


def _project_record(record: dict) -> dict:
    if not isinstance(record, dict):
        raise ValueError('Every OddsPortal record must be a JSON object.')
    projected = {key: value for key, value in record.items()
                 if key in PROJECTED_COLUMNS or QUOTE_COLUMN.fullmatch(key)}
    archive = record.get('source_archive_row')
    if archive is None and isinstance(record.get('source_identity'), dict):
        identity = record['source_identity']
        archive = {
            'home-name': identity.get('source_archive_home_name'),
            'away-name': identity.get('source_archive_away_name'),
            'home-participant-id': identity.get('source_home_participant_id'),
            'away-participant-id': identity.get('source_away_participant_id'),
        }
    if archive is not None:
        if not isinstance(archive, dict):
            raise ValueError('Source archive participant identity must be an object.')
        names = tuple(normalize_team_name(record.get(key)) for key in ('home_team', 'away_team'))
        archive_names = tuple(normalize_team_name(archive.get(key)) for key in ('home-name', 'away-name'))
        ids = tuple(_participant_id(archive.get(key)) for key in ('home-participant-id', 'away-participant-id'))
        if all(names) and names[0] != names[1] and names == archive_names:
            status = 'archive_direct'
        elif all(names) and names[0] != names[1] and names == archive_names[::-1]:
            ids = ids[::-1]
            status = 'archive_reversed'
        else:
            ids = (None, None)
            status = 'unresolved_archive_order'
    else:
        ids = tuple(_participant_id(record.get(key)) for key in (
            'source_home_participant_id', 'source_away_participant_id'))
        status = 'declared_columns' if any(ids) else 'unavailable'
    projected.update(source_home_participant_id=ids[0], source_away_participant_id=ids[1],
                     source_participant_identity_status=status)
    return projected


def _load_odds(path: Path, snapshot_bytes: int | None) -> tuple[list[dict], dict]:
    """Bound a live journal at opening; only complete newline-delimited rows count."""
    input_format = path.suffix.lower().lstrip('.')
    if input_format not in ('csv', 'json', 'jsonl'):
        raise ValueError('OddsPortal input must be CSV, JSON array, or JSONL.')
    if snapshot_bytes is not None and (
            input_format != 'jsonl' or isinstance(snapshot_bytes, bool)
            or not isinstance(snapshot_bytes, int) or snapshot_bytes < 0):
        raise ValueError('snapshot_bytes must be a nonnegative integer for JSONL input.')
    digest = hashlib.sha256()
    records = []
    complete_bytes = incomplete_bytes = 0
    with path.open('rb') as stream:
        size = os.fstat(stream.fileno()).st_size
        boundary = size if snapshot_bytes is None else snapshot_bytes
        if boundary > size:
            raise ValueError('JSONL snapshot boundary exceeds the current file size.')
        observed_at = datetime.now(timezone.utc).isoformat()
        if input_format == 'jsonl':
            while stream.tell() < boundary:
                line = stream.readline(boundary - stream.tell())
                if not line:
                    raise ValueError('JSONL source was truncated while reading its snapshot.')
                if not line.endswith(b'\n'):
                    incomplete_bytes = len(line)
                    break
                try:
                    record = json.loads(line)
                except (ValueError, UnicodeDecodeError) as exc:
                    raise ValueError(f'Invalid complete JSONL record at byte {complete_bytes}.') from exc
                records.append(_project_record(record))
                digest.update(line)
                complete_bytes += len(line)
        else:
            body = stream.read(boundary)
            digest.update(body)
            complete_bytes = len(body)
            if input_format == 'csv':
                payload = pd.read_csv(io.BytesIO(body)).to_dict('records')
            else:
                payload = json.loads(body)
                if not isinstance(payload, list):
                    raise ValueError('OddsPortal JSON must contain an array of records.')
            records = [_project_record(record) for record in payload]
    return records, {
        'path': str(path.resolve()), 'format': input_format, 'snapshot_at_utc': observed_at,
        'bytes_at_open': size, 'snapshot_bytes': boundary,
        'complete_prefix_bytes': complete_bytes, 'incomplete_tail_bytes': incomplete_bytes,
        'sha256': digest.hexdigest(), 'rows': len(records),
    }


def _golgg_records(stream):
    """Reuse the repository's optional streaming-parser pattern for the 1GB source."""
    try:
        import ijson
    except ImportError:
        payload = json.load(stream)
        if not isinstance(payload, list):
            raise ValueError('GOL.GG JSON must contain an array of matches.')
        yield from payload
    else:
        first = stream.read(1)
        while first and first.isspace():
            first = stream.read(1)
        if first != b'[':
            raise ValueError('GOL.GG JSON must contain an array of matches.')
        stream.seek(0)
        yield from ijson.items(stream, 'item')


def _canonical_game(raw: dict) -> dict:
    first, second = golgg_schema.score1(raw), golgg_schema.score2(raw)
    return {
        'match_id': raw['match_id'], 'date': raw['date'],
        'name_1': golgg_schema.team1_name(raw), 'name_2': golgg_schema.team2_name(raw),
        'tid_1': golgg_schema.team1_id(raw), 'tid_2': golgg_schema.team2_id(raw),
        'score_1': first, 'score_2': second,
        't1_win': first > second, 't2_win': second > first, 'draw': first == second,
        'tournament': golgg_schema.match_tournament(raw),
    }


def _quote_pairs(records: list[dict]) -> list[tuple[str, str]]:
    pairs = list(QUOTE_PAIRS)
    extra = set()
    for row in records:
        for column in row:
            match = QUOTE_COLUMN.fullmatch(column)
            if match:
                suffix = f'{match[1]}_{match[2]}{match[3] or ""}'
                extra.add((f'odds1_{suffix}', f'odds2_{suffix}'))
    pairs.extend(sorted(extra - set(pairs)))
    return pairs


def _source_day(value: str) -> tuple:
    if not isinstance(value, str):
        raise ValueError('Missing source date.')
    start = datetime.fromisoformat(value)
    if start.tzinfo is not None:
        return start.astimezone(timezone.utc).date(), 'utc_start_day'
    return start.date(), 'uncertified_source_calendar_day'


def _source_score(value) -> int:
    if isinstance(value, bool):
        raise ValueError('A boolean is not a series score.')
    try:
        number = Decimal(str(value))
    except InvalidOperation as exc:
        raise ValueError('Invalid series score.') from exc
    if not number.is_finite() or number < 0 or number != number.to_integral_value():
        raise ValueError('Series scores must be nonnegative integers.')
    return int(number)


def _value(row: dict, key: str):
    value = row.get(key)
    return None if pd.isna(value) else value


def map_matches(oddsportal_input: str | Path, golgg_json: str | Path,
                output_csv: str | Path, aliases_json: str | Path | None = None,
                *, snapshot_bytes: int | None = None) -> None:
    """Write unique identity-aligned rows and an audit of every accepted/rejected row.

    CSV, normalized JSON and active JSONL journals share the same identity rules.
    Prefer the same UTC/source-calendar date; only if it has no candidate, inspect
    both adjacent dates together. Resolve the entire candidate graph before writing:
    neither outcomes nor consumption order can disambiguate repeated fixtures.
    Results may reject a uniquely identified fixture, but never choose its identity.
    """
    output = Path(output_csv)
    audit_path = output.with_suffix('.audit.json')
    for path in (output, audit_path):
        if path.exists():
            raise FileExistsError(f'Preserve existing evidence; choose a new output path: {path}')
    aliases, scoped_aliases = ({}, {}) if aliases_json is None else load_aliases(aliases_json)
    if aliases_json is None:
        aliases_digest = None
    else:
        with Path(aliases_json).open('rb') as file:
            aliases_digest = hashlib.file_digest(file, 'sha256').hexdigest()
    odds, source = _load_odds(Path(oddsportal_input), snapshot_bytes)
    quote_pairs = _quote_pairs(odds)
    by_identity = defaultdict(list)
    seen_ids = set()
    with Path(golgg_json).open('rb') as file:
        golgg_digest = hashlib.file_digest(file, 'sha256').hexdigest()
        file.seek(0)
        for raw in _golgg_records(file):
            game = _canonical_game(raw)
            mid = str(game['match_id'])
            if mid in seen_ids:
                raise ValueError(f'Duplicate GOL.GG match id: {mid}')
            seen_ids.add(mid)
            names = tuple(apply_alias(normalize_team_name(game[key]), aliases) for key in ('name_1', 'name_2'))
            if not all(names) or names[0] == names[1]:
                continue
            day = datetime.strptime(game['date'], '%Y-%m-%d').date()
            by_identity[(day, tuple(sorted(names)))].append((game, names))

    audit = []
    proposals = {}
    for index, row in enumerate(odds):
        entry = {'source_row': index, 'source_date': _value(row, 'date'),
                 'source_home_team': _value(row, 'home_team'), 'source_away_team': _value(row, 'away_team'),
                 'oddsportal_url': _value(row, 'url'), 'source_event_id': _value(row, 'event_id'),
                 'source_odds_status': _value(row, 'odds_status'),
                 'source_home_participant_id': _value(row, 'source_home_participant_id'),
                 'source_away_participant_id': _value(row, 'source_away_participant_id'),
                 'source_participant_identity_status': _value(row, 'source_participant_identity_status'),
                 'status': 'unmatched', 'candidate_ids': []}
        audit.append(entry)
        if entry['source_odds_status'] not in (None, 'available'):
            entry['status'] = 'source_odds_unavailable'
            continue
        if entry['source_event_id'] is not None:
            try:
                if event_id_from_url(_value(row, 'url')) != entry['source_event_id']:
                    raise ValueError('URL and declared event identities disagree.')
            except ValueError:
                entry['status'] = 'invalid_source_event_id'
                continue
        names = tuple(apply_alias(normalize_team_name(row.get(key)), aliases) for key in ('home_team', 'away_team'))
        if not all(names):
            entry['status'] = 'invalid_identity'
            continue
        try:
            day, basis = _source_day(_value(row, 'date'))
        except ValueError:
            entry['status'] = 'invalid_date'
            continue
        entry['source_date_basis'] = basis
        variants = tuple(_source_name_variants(
            row.get(name), entry[id_key], day, row.get('tournament'), aliases, scoped_aliases)
            for name, id_key in (('home_team', 'source_home_participant_id'),
                                 ('away_team', 'source_away_participant_id')))
        if names[0] == names[1] and len(variants[0]) == len(variants[1]) == 1:
            entry['status'] = 'invalid_identity'
            continue
        entry['source_identity_variants'] = [sorted(values) for values in variants]
        candidates = _identity_candidates(by_identity, day, variants)
        if not candidates:
            candidates = [candidate for offset in (-1, 1)
                          for candidate in _identity_candidates(
                              by_identity, day + timedelta(days=offset), variants, offset)]
        entry['candidate_ids'] = [game['match_id'] for game, _, _, _ in candidates]
        if len(candidates) != 1:
            if candidates:
                entry['status'] = 'ambiguous_golgg_fixture'
            continue
        game, canonical, offset, orientations = candidates[0]
        if len(orientations) != 1:
            entry['status'] = 'ambiguous_source_orientation'
            continue
        swapped = next(iter(orientations))
        resolved = canonical[::-1] if swapped else canonical
        entry.update(source_identity_home=resolved[0], source_identity_away=resolved[1],
                     source_alias_ids=sorted(set(variants[0][resolved[0]] + variants[1][resolved[1]])))
        proposals[index] = (game, swapped, offset, day)

    # Count all edges, including ambiguous rows: do not make a candidate unique by
    # first discarding or consuming another competing observation.
    claims = Counter(mid for entry in audit for mid in entry['candidate_ids'])
    results = []
    for index, (game, swapped, offset, day) in proposals.items():
        entry = audit[index]
        if claims[game['match_id']] != 1:
            entry['status'] = 'ambiguous_odds_fixture'
            continue
        row = odds[index]
        first, second = (_value(row, 'awayResult'), _value(row, 'homeResult')) if swapped else (
            _value(row, 'homeResult'), _value(row, 'awayResult'))
        if pd.isna(first) or pd.isna(second):
            entry['status'] = 'missing_source_result'
            continue
        try:
            first, second = _source_score(first), _source_score(second)
        except ValueError:
            entry['status'] = 'invalid_source_result'
            continue
        if (first > second) != bool(game['t1_win']) or (second > first) != bool(game['t2_win']):
            entry['status'] = 'result_mismatch'
            continue
        if (first, second) != (game['score_1'], game['score_2']):
            entry['status'] = 'score_mismatch'
            continue
        record = {'golgg_match_id': game['match_id'], 'odds_date': day.isoformat(),
                  'golgg_date': game['date'], 'golgg_team1': game['name_1'], 'golgg_team2': game['name_2'],
                  't1_score': game['score_1'], 't2_score': game['score_2'],
                  't1_win': game['t1_win'], 't2_win': game['t2_win'], 'draw': game['draw'],
                  'match_type': 'identity_reviewed_alias' if entry['source_alias_ids'] else (
                      'identity_exact' if offset == 0 else 'identity_date_offset'),
                  'tournament': _value(row, 'tournament'), 'oddsportal_url': _value(row, 'url'),
                  'source_row': index, 'source_date': _value(row, 'date'),
                  'source_home_team': _value(row, 'home_team'), 'source_away_team': _value(row, 'away_team'),
                  'source_home_result': _value(row, 'homeResult'), 'source_away_result': _value(row, 'awayResult'),
                  'source_sides_swapped': swapped, 'date_offset_days': offset,
                  'source_event_id': entry['source_event_id'], 'source_date_basis': entry['source_date_basis'],
                  'source_odds_status': entry['source_odds_status'],
                  'source_date_status': _value(row, 'date_status'),
                  'source_canonical_metadata_source': _value(row, 'canonical_metadata_source'),
                  'quote_timing_semantics': _value(row, 'quote_timing_semantics'),
                  'odds_capture_completed_at': _value(row, 'odds_capture_completed_at'),
                  'shell_capture_completed_at': _value(row, 'shell_capture_completed_at'),
                  'golgg_team1_id': game['tid_1'], 'golgg_team2_id': game['tid_2'],
                  'golgg_tournament': game['tournament'], 'mapping_version': MAPPING_VERSION}
        record.update({key: entry.get(key) for key in (
            'source_home_participant_id', 'source_away_participant_id', 'source_participant_identity_status',
            'source_identity_home', 'source_identity_away')})
        record['source_alias_ids'] = json.dumps(entry['source_alias_ids'])
        for left, right in quote_pairs:
            record[left], record[right] = (_value(row, right), _value(row, left)) if swapped else (
                _value(row, left), _value(row, right))
        results.append(record)
        entry.update(status='matched', golgg_match_id=game['match_id'], source_sides_swapped=swapped,
                     date_offset_days=offset)
    output.parent.mkdir(parents=True, exist_ok=True)
    columns = list(OUTPUT_COLUMNS)
    columns.extend(column for pair in quote_pairs for column in pair if column not in OUTPUT_COLUMNS)
    with output.open('x', encoding='utf-8', newline='') as file:
        pd.DataFrame(results, columns=columns).to_csv(file, index=False)
    summary = dict(Counter(entry['status'] for entry in audit))
    with audit_path.open('x', encoding='utf-8') as file:
        json.dump({'mapping_version': MAPPING_VERSION, 'summary': summary,
                   'timing': 'Date-only retrospective identity mapping; provider last change is not certified CLOSE.',
                   'source': source, 'golgg_source': {'path': str(Path(golgg_json).resolve()),
                                                    'sha256': golgg_digest, 'matches': len(seen_ids)},
                   'aliases_source': None if aliases_json is None else str(Path(aliases_json).resolve()),
                   'aliases_sha256': aliases_digest,
                   'quote_columns': [column for pair in quote_pairs for column in pair],
                   'rows': audit}, file, indent=2, allow_nan=False)
    print(json.dumps({'output': str(output), 'audit': str(audit_path), 'summary': summary}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--oddsportal-input', required=True, help='CSV, normalized JSON, or an active settled-records.jsonl journal.')
    parser.add_argument('--golgg-json', required=True)
    parser.add_argument('--output-csv', required=True)
    parser.add_argument('--aliases-json', help='Optional explicitly reviewed aliases; no aliases are inferred.')
    parser.add_argument('--snapshot-bytes', type=int, help='Replay a fixed JSONL byte boundary; default is size at opening.')
    args = parser.parse_args()
    map_matches(args.oddsportal_input, args.golgg_json, args.output_csv, args.aliases_json,
                snapshot_bytes=args.snapshot_bytes)
