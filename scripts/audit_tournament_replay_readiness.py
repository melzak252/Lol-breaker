#!/usr/bin/env python3
"""Inventory local historical replay prerequisites without manufacturing a bracket.

--output-dir must be NEW. Default cohort is all snapshot series from 2024 onward,
never a bookmaker intersection. --event-root may be repeated to locate existing
event.json/bracket.json/pair_features.csv bundles. No database or network access.
Outputs distinguish match-level retrospective replay from full-event forecasts.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
import gzip
import hashlib
import json
from pathlib import Path
import re
import sys
from urllib.parse import unquote, urlparse

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.export_prospective_features import iter_matches
from scripts.prospective_sports_features import _series_team_ids
from scripts.score_tournament_forecasts import digest, load_bracket, observed_outcomes, save_json
from scripts.tournament_research_snapshot import _read_inputs, _validate_inputs, _utc
from src.utils import golgg_schema
from src.models.tournament_catalog import DEFAULT_BUNDLE, DEFAULT_CATALOG, instantiate_profile

VERSION = 'tournament-replay-readiness-v3'
DEFAULT_SNAPSHOTS = ROOT/'data/artifacts/corrected039081-20260908/replay/snapshots.csv'
DEFAULT_HISTORY = ROOT/'data/artifacts/golgg-header-repair-20260908-validated/matches.json'
DEFAULT_REQUESTS = ROOT/'data/golgg_corrected_20260908/sources/requests.jsonl'
DEFAULT_PILOT = ROOT/'data/artifacts/tournament-engine-20260909/lck2026-road-to-msi/seeding-and-outcome-evidence.json'
REASONS = ['start_frozen_rosters_not_archived', 'all_pair_start_features_not_archived',
           'announced_bracket_draw_rules_not_archived', 'prestart_forecast_not_archived',
           'historical_publication_times_not_certified']


def _read_csv(path):
    return pd.read_csv(path, dtype={'golgg_match_id': str, 'team1_id': str, 'team2_id': str})


def audit_bundles(roots):
    """Use existing capture validation at its recorded observation time, not now.

    This is verification of existing local evidence, never historical capture or
    independent publication-time certification. Incomplete bundles retain reasons.
    """
    rows, files = [], []
    paths = sorted({path.resolve() for root in roots for path in root.rglob('event.json')})
    for path in paths:
        row = {'event_path': str(path), 'tournament_id': '', 'prestart_local_bundle': 0,
               'complete_observed_results': 0, 'eligible_full_event_replay': 0,
               'eligible_live': 0, 'point_in_time_certified': 0}
        files.append(path)
        reasons = []
        try:
            directory = path.parent
            event = json.loads(path.read_text())
            bracket_path = directory/'bracket.json'
            if bracket_path.is_file():
                row['tournament_id'] = json.loads(bracket_path.read_text())['id']
            snapshot_path = directory.parent/'snapshot.json' if directory.name == 'inputs' else directory/'snapshot.json'
            if not snapshot_path.is_file():
                reasons.append('no_existing_prestart_snapshot_metadata')
            else:
                snapshot = json.loads(snapshot_path.read_text())
                files.append(snapshot_path)
                payloads = _read_inputs(directory)
                files.extend(directory/name for name in payloads if directory/name != path)
                if {name: hashlib.sha256(content).hexdigest() for name, content in payloads.items()} != snapshot['input_sha256']:
                    raise ValueError('snapshot input hash mismatch')
                observed = _utc(snapshot['source_available_at'])
                bracket, frame, start = _validate_inputs(payloads, observed)
                if not observed <= _utc(snapshot['data_cutoff_at']) <= _utc(snapshot['predicted_at']) < start:
                    raise ValueError('prediction did not precede whole event')
                if snapshot['tournament_id'] != bracket.id:
                    raise ValueError('snapshot tournament ID mismatch')
                predictions_path = snapshot_path.parent/'predictions.csv'
                files.append(predictions_path)
                if digest(predictions_path) != snapshot['predictions_sha256']:
                    raise ValueError('prediction hash mismatch')
                row.update(tournament_id=bracket.id, pair_rows=len(frame), prestart_local_bundle=1,
                           event_start_at=event['event_start_at'], source_available_at=observed.isoformat())
                results_path = snapshot_path.parent/'results.json'
                if not results_path.is_file():
                    reasons.append('complete_results_bundle_missing')
                else:
                    files.append(results_path)
                    observed_outcomes(bracket, json.loads(results_path.read_text()))
                    row['complete_observed_results'] = 1
                    row['eligible_full_event_replay'] = 1
            reasons.append('local_capture_is_not_independent_source_certification')
        except (ValueError, TypeError, KeyError, OSError) as exc:
            reasons.append(f'{type(exc).__name__}:{exc}')
        row['reasons'] = ';'.join(reasons)
        rows.append(row)
    return rows, files


def _source_rows(path):
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _date_issue(value, field):
    if not isinstance(value, str) or not value.strip():
        return f'missing_{field}'
    try:
        if datetime.strptime(value, '%Y-%m-%d').date().isoformat() != value:
            raise ValueError('date must use the full ISO calendar form')
    except ValueError:
        return f'invalid_{field}'
    return None


def _source_title(value):
    """MediaWiki's explicit URL/namespace spelling, never a fuzzy title join."""
    value = unquote(str(value or '')).replace('_', ' ').strip()
    if '/wiki/' in value:
        value = value.split('/wiki/', 1)[1]
    return value.removeprefix('Data:').split('#', 1)[0]


def _format_section(content):
    headings = list(re.finditer(r'(?m)^(={2,6})\s*(.*?)\s*\1\s*$', content))
    for index, heading in enumerate(headings):
        if heading[2].strip().casefold() != 'format':
            continue
        end = next((h.start() for h in headings[index + 1:]
                    if len(h[1]) <= len(heading[1])), len(content))
        return content[heading.end():end].strip()
    return ''


def _archived_phase_evidence(events, bundle, expected):
    """Read identity and Format only; results/prize tables cannot create links."""
    cache, evidence, paths, changes = {}, {}, set(), []
    for event in events:
        source = event.get('source') or {}
        raw_name = source.get('raw_path')
        record = {'verified': False, 'source': source, 'format_excerpt': '',
                  'identity': {}, 'format_links': [], 'reasons': []}
        evidence[event['phase_id']] = record
        if not raw_name:
            record['reasons'].append('archived_phase_page_missing')
            continue
        # Relocate the archived bundle as a unit, never borrow a missing raw
        # file from the original frozen bundle when auditing a copy.
        path = bundle / 'raw' / Path(raw_name).name
        paths.add(path)
        if path not in cache:
            actual = digest(path) if path.is_file() else None
            declared = expected.get(f'raw/{path.name}', {}).get('sha256')
            issue = ('missing_referenced_source' if actual is None else
                     'archived_content_hash_changed' if declared and declared != actual else None)
            if issue:
                changes.append({'path': str(path), 'reason': issue,
                                'expected_sha256': declared, 'sha256': actual})
            try:
                payload = json.loads(path.read_text()) if not issue else {}
                cache[path] = (payload.get('query', {}).get('pages', []), issue, actual)
            except (OSError, ValueError, TypeError, AttributeError) as exc:
                cache[path] = ([], f'invalid_archived_page:{type(exc).__name__}', actual)
        pages, issue, actual = cache[path]
        record.update(path=str(path), sha256=actual)
        if issue:
            record['reasons'].append(issue)
            continue
        matches = [page for page in pages if str(page.get('pageid')) == str(source.get('pageid'))]
        if len(matches) != 1:
            record['reasons'].append('archived_page_identity_not_unique')
            continue
        page = matches[0]
        revisions = [rev for rev in page.get('revisions', [])
                     if str(rev.get('revid')) == str(source.get('revid'))]
        if len(revisions) != 1 or _source_title(page.get('title')) != _source_title(event.get('title')):
            record['reasons'].append('archived_page_title_or_revision_changed')
            changes.append({'phase_id': event['phase_id'], 'path': str(path),
                            'reason': 'archived_page_title_or_revision_changed'})
            continue
        content = revisions[0].get('slots', {}).get('main', {}).get('content', '')
        if not isinstance(content, str):
            record['reasons'].append('archived_page_content_missing')
            continue
        excerpt = _format_section(content)
        # Only recognize simple, unambiguous identity parameters in the source
        # infobox. This is not a general wikitext parser or an outcome extractor.
        infobox = re.search(r'(?ms)^\{\{Infobox Tournament\b(.*?)^\}\}', content)
        identity = {}
        if infobox:
            for key in ('CM_StandardLeague', 'CM_StandardName', 'CM_Year', 'split'):
                values = re.findall(rf'(?m)^\|{key}\s*=\s*([^|\n{{}}]*)', infobox[1])
                if len(values) == 1:
                    identity[key] = values[0].strip()
        links = [_source_title(match[1]) for match in
                 re.finditer(r'\[\[([^][|]+)(?:\|[^][]*)?\]\]', excerpt)]
        record.update(verified=True, format_excerpt=excerpt, identity=identity,
                      format_links=sorted(set(links)))
    return evidence, paths, changes


def _event_day_evidence(row, offsets):
    raw = {key: row.get(key) for key in ('source_date', 'source_time',
           'source_timezone', 'source_dst', 'scheduled_at_utc')}
    day, clock = raw['source_date'], raw['source_time']
    issues, changes = [], []
    raw_issue = _date_issue(day, 'source_date')
    if raw_issue:
        # The archive has date=00:00,time=YYYY-MM-DD. Repair the typed
        # fields, not selected match IDs, and retain the original payload.
        if (isinstance(day, str) and re.fullmatch(r'\d{2}:\d{2}(?::\d{2})?', day)
                and not _date_issue(clock, 'source_time_date')):
            day, clock = clock, day
            changes.append('archived_date_time_fields_transposed')
        else:
            day = None
            issues.append(raw_issue)
    scheduled = None
    if raw['scheduled_at_utc']:
        try:
            scheduled = _utc(raw['scheduled_at_utc'])
        except (ValueError, TypeError):
            issues.append('invalid_scheduled_at_utc')
    else:
        issues.append('missing_scheduled_at_utc')
    reconstructed = None
    if day and clock:
        offset = offsets.get(str(raw['source_dst'] or '').lower(), {}).get(raw['source_timezone'])
        if isinstance(offset, (int, float)) and not isinstance(offset, bool):
            try:
                if not re.fullmatch(r'\d{2}:\d{2}(?::\d{2})?', str(clock)):
                    raise ValueError('invalid local clock')
                local = datetime.fromisoformat(f'{day}T{clock}')
                reconstructed = (local + timedelta(hours=offset)).replace(tzinfo=timezone.utc)
            except (TypeError, ValueError, OverflowError):
                issues.append('invalid_archived_local_time')
        else:
            issues.append('archived_timezone_offset_unresolved')
    if reconstructed and scheduled and reconstructed != scheduled:
        issues.append('scheduled_timestamp_disagrees_with_archived_zone')
    if day and scheduled and day != scheduled.date().isoformat():
        changes.append('local_and_utc_event_days_differ')
    days = [value for value in (day, scheduled.date().isoformat() if scheduled else None,
                               reconstructed.date().isoformat() if reconstructed else None) if value]
    conflict = 'scheduled_timestamp_disagrees_with_archived_zone' in issues
    return {'raw': raw, 'raw_date_issue': raw_issue,
            'normalized_local_event_day': day,
            'scheduled_utc_event_day': scheduled.date().isoformat() if scheduled else None,
            'reconstructed_scheduled_at_utc': reconstructed.isoformat() if reconstructed else None,
            'conservative_event_day': max(days) if days and not conflict else None,
            'changes': changes, 'issues': issues,
            'completion_at': None, 'available_at': None, 'completion_certified': False,
            'semantics': 'scheduled_event_day_only; not_result_completion_or_publication',
            'release_policy': 'no_automatic_release; next_day_availability_requires_explicit_research_assumption'}


def _row_families(row):
    explicit = row.get('family')
    families = row.get('families')
    if isinstance(families, str) and families:
        try:
            families = json.loads(families)
        except ValueError:
            return None, 'invalid_source_families'
    if families and (not isinstance(families, list) or
                     any(not isinstance(value, str) or not value for value in families)):
        return None, 'invalid_source_families'
    if explicit and families and explicit not in families:
        return None, 'conflicting_source_family'
    return ({explicit} if explicit else set(families or [])), None


def _stage_supported(stage, text):
    patterns = {'group_stage': r'\bgroups?\b|\bGSL\b',
                'swiss': r'\bswiss\b',
                'playoffs': r'\bplayoffs?\b|\bknockout\b|\belimination\b',
                'champions_round': r'\bchampions[’\']?\s+round\b'}
    return bool(stage in patterns and re.search(patterns[stage], text, re.IGNORECASE))


def _phase_join(row, by_title, archived, event_day, bindings):
    families, issue = _row_families(row)
    candidates = list(by_title.get(_source_title(row.get('event_title')), []))
    if issue:
        return None, {'method': None, 'reason': issue, 'candidate_phase_ids': []}
    if families:
        candidates = [event for event in candidates if event.get('family') in families]
    if row.get('year'):
        candidates = [event for event in candidates if str(event.get('year')) == str(row['year'])]
    stage = row.get('stage')
    binding = bindings.get(row.get('event_title'), {}).get(row.get('tab'))
    if binding and stage and binding != stage:
        return None, {'method': None, 'reason': 'archived_stage_binding_conflict',
                      'candidate_phase_ids': [event['phase_id'] for event in candidates]}
    stage = binding or stage
    exact = [event for event in candidates if event.get('stage') == stage]
    mapping = {'method': None, 'source_columns': ['event_title', 'family', 'families', 'year', 'stage'],
               'candidate_phase_ids': [event['phase_id'] for event in exact],
               'outcomes_used': False}
    if len(exact) == 1:
        mapping.update(method='exact_family_title_stage',
                       rationale='Exactly one phase after explicit source family/year and title/stage constraints.')
        return exact[0]['phase_id'], mapping
    if len(exact) > 1:
        # Family ambiguity is not resolved by chronology or realized teams.
        if len({event.get('family') for event in exact}) > 1:
            mapping['reason'] = 'ambiguous_phase_join'
            return None, mapping
        labels = ' '.join(str(row.get(key) or '') for key in ('section_path', 'tab'))
        label = ('rumble' if re.search(r'\brumble\b', labels, re.I) else
                 'initial' if re.search(r'\bgroups?\b', labels, re.I) else None)
        eligible = []
        for event in exact:
            evidence = archived[event['phase_id']]
            description = ' '.join(str(event.get(key) or '') for key in ('phase_name', 'format_summary'))
            if not evidence['verified'] or not label:
                continue
            if label == 'rumble':
                supported = bool(re.search(r'\brumble\b', description, re.I) and
                                 re.search(r'\brumble\b', evidence['format_excerpt'], re.I))
            else:
                supported = bool(re.search(r'\binitial\s+group', description, re.I) and
                                 re.search(r'\bgroups?\b', evidence['format_excerpt'], re.I))
            if supported:
                eligible.append(event)
        if len(eligible) == 1:
            event = eligible[0]
            days = [day for day in (event_day['normalized_local_event_day'],
                                    event_day['scheduled_utc_event_day']) if day]
            if (not days or event_day['conservative_event_day'] is None or
                    any(_date_issue(event.get(field), field) for field in ('start_date', 'end_date'))):
                mapping['reason'] = 'phase_date_boundary_unresolved'
            elif not all(event['start_date'] <= day <= event['end_date'] for day in days):
                mapping['reason'] = 'phase_label_date_conflict'
            else:
                mapping.update(method='archived_phase_label_and_boundary',
                               source_columns=mapping['source_columns'] + ['tab', 'section_path',
                               'source_date', 'scheduled_at_utc', 'phase.format_summary',
                               'phase.start_date', 'phase.end_date'],
                               rationale='Archived Format corroborates the schedule label; every known local/UTC '
                                         'event day lies inside this phase, without observed-result boundaries.',
                               phase_source=archived[event['phase_id']])
                return event['phase_id'], mapping
        else:
            mapping['reason'] = 'ambiguous_phase_join'
        return None, mapping
    composite = [event for event in candidates if event.get('stage') == 'main'
                 and archived[event['phase_id']]['verified']
                 and _stage_supported(stage, archived[event['phase_id']]['format_excerpt'])
                 and _stage_supported(stage, str(event.get('format_summary') or ''))]
    mapping['candidate_phase_ids'] = [event['phase_id'] for event in composite]
    if len(composite) == 1:
        event = composite[0]
        mapping.update(method='archived_composite_phase',
                       rationale='The exact source title/family identifies a composite main phase; both '
                                 'its archived Format and phase summary explicitly contain the result stage.',
                       phase_source=archived[event['phase_id']])
        return event['phase_id'], mapping
    mapping['reason'] = 'ambiguous_phase_join' if composite else 'no_exact_title_stage_join'
    return None, mapping


def _edition_clusters(events, archived):
    by_title = defaultdict(list)
    parents = {event['phase_id']: event['phase_id'] for event in events}

    def root(identifier):
        while parents[identifier] != identifier:
            identifier = parents[identifier]
        return identifier

    def merge(left, right):
        parents[root(right)] = root(left)

    exact = defaultdict(list)
    for event in events:
        by_title[_source_title(event.get('title'))].append(event)
        # Format prose often uses the page's explicit standard-name alias
        # (e.g. "LPL 2024 Spring"). This is source identity, not fuzzy matching.
        alias = archived[event['phase_id']]['identity'].get('CM_StandardName')
        if alias and _source_title(alias) != _source_title(event.get('title')):
            by_title[_source_title(alias)].append(event)
        exact[(event.get('family'), event.get('year'), event.get('title'))].append(event)
    for members in exact.values():
        for member in members[1:]:
            merge(members[0]['phase_id'], member['phase_id'])
    links, unresolved = [], []
    for event in events:
        evidence = archived[event['phase_id']]
        for title in evidence['format_links']:
            for other in by_title.get(title, []):
                if event['phase_id'] == other['phase_id'] or event.get('title') == other.get('title'):
                    continue
                record = {'source_phase_id': event['phase_id'], 'target_phase_id': other['phase_id'],
                          'source_title': event.get('title'), 'linked_title': title,
                          'evidence': evidence, 'kind': 'format_membership_link',
                          'direction_is_advancement_verified': False}
                left, right = evidence['identity'], archived[other['phase_id']]['identity']
                if event.get('family') != other.get('family') or event.get('year') != other.get('year'):
                    record['reason'] = 'different_source_family_or_year'
                elif left.get('split') != right.get('split'):
                    record['reason'] = 'different_archived_split'
                elif (not archived[other['phase_id']]['verified'] or
                      not left.get('CM_StandardLeague') or
                      left.get('CM_StandardLeague') != right.get('CM_StandardLeague') or
                      left.get('CM_Year') != str(event.get('year')) or
                      right.get('CM_Year') != str(other.get('year'))):
                    record['reason'] = 'archived_edition_identity_unresolved'
                else:
                    merge(event['phase_id'], other['phase_id'])
                    record['rationale'] = ('Explicit archived Format link and equal archived league/year/split; '
                                           'not a winner, placement or a year-wide navigation tab.')
                    links.append(record)
                    continue
                unresolved.append(record)
    clusters = defaultdict(list)
    for event in events:
        clusters[root(event['phase_id'])].append(event)
    editions, phase_cluster = [], {}
    for members in clusters.values():
        family, year = members[0].get('family'), members[0].get('year')
        titles = sorted({member.get('title') for member in members})
        phase_ids = [member['phase_id'] for member in members]
        phase_links = [link for link in links if link['source_phase_id'] in phase_ids]
        key = hashlib.sha256(json.dumps([family, year, titles]).encode()).hexdigest()[:20]
        edition_id = f'{"source-linked-edition" if phase_links else "unverified-title-cluster"}:{key}'
        phase_cluster.update({phase_id: edition_id for phase_id in phase_ids})
        editions.append({'tournament_id': edition_id, 'title': titles[0], 'titles': titles,
                         'family': family, 'year': year, 'phase_ids': phase_ids,
                         'edition_start_at': None,
                         'edition_start_status': 'whole_edition_boundary_not_verified',
                         'candidate_cluster_start_date': min(
                             (member['start_date'] for member in members
                              if not _date_issue(member.get('start_date'), 'start_date')), default=None),
                         'cluster_start_is_edition_start': False,
                         'grouping_evidence': {'source_table': 'events.jsonl',
                                              'source_columns': ['family', 'year', 'title'],
                                              'archived_format_links': phase_links},
                         'identity_status': 'source_linked_phase_cluster' if phase_links else
                                            'unverified_title_cluster',
                         'whole_edition_verified': False, 'phase_links': phase_links,
                         'reasons': ['whole_edition_boundary_and_transition_graph_not_verified']})
    return editions, phase_cluster, unresolved


def _team_identity(name, aliases):
    record = aliases.get(name, {})
    return {'source_name': name,
            'canonical_title': record.get('canonical_title') if record.get('resolved') is True else None,
            'status': 'archived_alias' if record.get('resolved') is True else 'unresolved_exact_identity',
            'source': record, 'golgg_team_id': None,
            'scope': 'Leaguepedia identity only; no outcome-based cross-provider mapping'}


def _group_membership(row, groups, aliases):
    def identity(name):
        return _team_identity(name, aliases)['canonical_title'] or name

    teams = {identity(row.get('team1')), identity(row.get('team2'))}
    matches = []
    if len(teams) != 2 or None in teams or '' in teams:
        return matches
    for group in groups:
        fields = group.get('fields') or {}
        for name in str(fields.get('groups') or '').split(','):
            name = name.strip()
            members = {identity(team.strip()) for team in str(fields.get(name) or '').split(',')
                       if team.strip()}
            if teams <= members:
                matches.append({'group': name, 'title': group.get('title'),
                                'revid': group.get('revid'), 'raw_path': group.get('raw_path'),
                                'source_ordinal': group.get('source_ordinal'),
                                'basis': 'both_exact_archived_team_identities_in_group_definition'})
    return matches


def build_readiness_manifest(bundle=DEFAULT_BUNDLE, *, catalog=DEFAULT_CATALOG,
                             pilot_roots=None):
    """Inventory every phase; eligibility never depends on a captured forecast.

    Title groups are evidence-backed phase clusters, NOT verified whole editions.
    Source results are segregated from seed construction. Candidate state inputs
    do not authorize temporal sports-model forecasts or strict checkpoints.
    """
    bundle, catalog = Path(bundle), Path(catalog)
    events = _source_rows(bundle / 'events.jsonl')
    ids = [e.get('phase_id') for e in events]
    if any(not isinstance(i, str) or not i for i in ids):
        raise ValueError('Every source phase requires a nonempty phase_id')
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate historical phase IDs in bundle')
    config = json.loads(catalog.read_text())
    profiles = defaultdict(list)
    rules = {r['phase_id']: r for r in config['phase_rules']}
    for profile in config['profiles']:
        for phase_id in profile['source_phase_ids']:
            profiles[phase_id].append(profile)
    series_path = bundle / 'series_results.csv'
    required = {'event_title', 'stage', 'wiki_series_id', 'source_date',
                'scheduled_at_utc', 'status', 'team1', 'team2', 'best_of',
                'validation_errors'}
    frame = pd.read_csv(series_path, dtype=str, keep_default_na=False) if series_path.is_file() else pd.DataFrame()
    missing_columns = sorted(required - set(frame.columns))
    source_series = frame.to_dict('records')
    optional_json = {}
    for name in ('source_manifest.json', 'time_offsets.json', 'stage_bindings.json', 'team_aliases.json'):
        path = bundle / name
        optional_json[name] = json.loads(path.read_text()) if path.is_file() else {}
    expected = optional_json['source_manifest.json'].get('files', {})
    archived, referenced_paths, source_changes = _archived_phase_evidence(events, bundle, expected)
    by_title = defaultdict(list)
    for event in events:
        by_title[_source_title(event.get('title'))].append(event)
    groups = _source_rows(bundle / 'source_group_definitions.jsonl')
    group_index = defaultdict(list)
    for group in groups:
        group_index[_source_title(group.get('title'))].append(group)
    payload_index = defaultdict(list)
    for source_row, payload in enumerate(_source_rows(bundle / 'series_results.jsonl'), 1):
        payload_index[payload.get('wiki_series_id')].append((source_row, payload))
    aliases = optional_json['team_aliases.json']
    identifiers = Counter(row.get('wiki_series_id') for row in source_series)
    joined, unjoined, reconciliation = defaultdict(list), [], []
    for index, row in enumerate(source_series, 2):
        identifier = row.get('wiki_series_id')
        event_day = _event_day_evidence(row, optional_json['time_offsets.json'])
        phase_id, mapping = _phase_join(row, by_title, archived, event_day,
                                       optional_json['stage_bindings.json'])
        payloads = payload_index.get(identifier, [])
        payload_record = None
        if payloads:
            if len(payloads) != 1:
                phase_id = None
                mapping['reason'] = 'duplicate_archived_payload_identity'
            else:
                line, payload = payloads[0]
                payload_record = {'path': str(bundle / 'series_results.jsonl'),
                                  'source_line': line, 'payload': payload}
                identity_fields = ('title', 'pageid', 'revid', 'source_ordinal', 'event_title',
                                   'team1', 'team2', 'source_date', 'source_time',
                                   'source_timezone', 'source_dst', 'scheduled_at_utc',
                                   'section_path', 'tab')
                disagreements = [key for key in identity_fields if key in row and key in payload
                                 and str(row[key] or '') != str(payload[key] or '')]
                if disagreements:
                    phase_id = None
                    mapping.update(reason='csv_archived_payload_identity_conflict',
                                   conflicting_columns=disagreements)
                    source_changes.append({'source_row': index, 'wiki_series_id': identifier,
                                           'reason': 'csv_archived_payload_identity_conflict',
                                           'conflicting_columns': disagreements})
        if not identifier or identifiers[identifier] > 1:
            phase_id = None
            mapping['reason'] = 'duplicate_series_identity' if identifier else 'missing_series_identity'
        membership = _group_membership(row, group_index[_source_title(row.get('event_title'))], aliases)
        record = {'source_row': index, 'wiki_series_id': identifier, 'phase_id': phase_id,
                  'status': 'joined' if phase_id else 'excluded', 'mapping': mapping,
                  'event_day': event_day, 'raw_source': row, 'archived_payload': payload_record,
                  'teams': {side: _team_identity(row.get(side), aliases) for side in ('team1', 'team2')},
                  'group_membership_evidence': membership,
                  'group_membership_used_as_seed_or_qualifier': False}
        reconciliation.append(record)
        if phase_id:
            joined[phase_id].append({'source_row': index, **row, 'reconciliation': record})
        else:
            unjoined.append({'source_row': index, **row, 'reason': mapping['reason'],
                             'candidate_phase_ids': mapping.get('candidate_phase_ids', []),
                             'mapping': mapping, 'event_day': event_day})
        for source in (row, *[team['source'] for team in record['teams'].values()]):
            if source.get('raw_path'):
                referenced_paths.add(bundle / 'raw' / Path(source['raw_path']).name)
    for group in groups:
        if group.get('raw_path'):
            referenced_paths.add(bundle / 'raw' / Path(group['raw_path']).name)
        for name in str((group.get('fields') or {}).get('groups') or '').split(','):
            for team in str((group.get('fields') or {}).get(name.strip()) or '').split(','):
                source = aliases.get(team.strip(), {})
                if source.get('raw_path'):
                    referenced_paths.add(bundle / 'raw' / Path(source['raw_path']).name)
    editions, phase_cluster, unresolved_links = _edition_clusters(events, archived)
    for record in reconciliation:
        record['tournament_id'] = phase_cluster.get(record['phase_id'])
    pilot_roots = ([ROOT / 'data/artifacts/tournament-engine-20260909']
                   if pilot_roots is None else [Path(p) for p in pilot_roots])
    pilots, pilot_files = [], []
    for path in sorted({p for root in pilot_roots for p in root.rglob('seeding-and-outcome-evidence.json')}):
        pilot_files.append(path)
        record = {'path': str(path), 'sha256': digest(path), 'status': 'excluded',
                  'eligible_forecast': False, 'historical_certification': False}
        try:
            payload = json.loads(path.read_text())
            source_phase = payload['source_phase']
            phase_id = source_phase['phase_id']
            record['phase_id'] = phase_id
            source = next((e for e in events if e['phase_id'] == phase_id), None)
            reasons = []
            if source is None or source != source_phase:
                reasons.append('pilot_source_phase_join_mismatch')
            teams = payload.get('team_ids_in_seed_order', [])
            prior = payload.get('prior_standings', [])
            prior_titles = sorted({r.get('event_title', '') for r in prior})
            # Rank from a previous completed phase can seed a later phase, never
            # the edition start. Do not use this event's placements or fixtures.
            prior_events = [e for e in events if e.get('title') in prior_titles]
            start = source_phase.get('start_date')
            if (_date_issue(start, 'start_date') or not prior_events or
                    {e['title'] for e in prior_events} != set(prior_titles) or
                    any(_date_issue(e.get('end_date'), 'end_date') or
                        e['end_date'] >= start or e.get('status') != 'completed'
                        for e in prior_events)):
                reasons.append('prior_standings_chronology_unproven')
            names = payload.get('team_names', {})
            rank_names = [r.get('team', '').casefold() for r in prior[:len(teams)]]
            ranks = [str(r.get('placement')) for r in prior[:len(teams)]]
            if (not teams or len(set(teams)) != len(teams)
                    or len(teams) != source_phase.get('participant_count')
                    or any(not isinstance(t, str) or not t for t in teams)
                    or [names.get(t, '').casefold() for t in teams] != rank_names
                    or ranks != [str(i) for i in range(1, len(teams) + 1)]):
                reasons.append('seed_identity_order_not_supported_by_prior_standings')
            if not payload.get('outcomes_used_for_features_or_simulation') is False:
                reasons.append('outcome_segregation_unproven')
            record.update(status='phase_start_reconstruction_candidate' if not reasons else 'excluded',
                          team_ids_in_seed_order=teams if not reasons else [],
                          prior_standings_titles=prior_titles,
                          source=source_phase.get('source'),
                          roster_policy=payload.get('roster_policy'),
                          source_columns=['team_ids_in_seed_order', 'team_names',
                                          'prior_standings.event_title', 'prior_standings.placement',
                                          'prior_standings.team', 'source_phase'],
                          reasons=reasons,
                          forecast_blockers=['chronological_model_fold_and_feature_parity_required',
                                             'previous_observed_roster_is_research_proxy',
                                             'prior_standings_source_revision_is_retrospective',
                                             'phase_start_is_not_whole_edition_start'])
        except (ValueError, TypeError, KeyError, OSError) as exc:
            record['reasons'] = [f'{type(exc).__name__}:{exc}']
        pilots.append(record)
    phases = []
    for event in events:
        phase_id = event['phase_id']
        rows = joined[phase_id]
        reasons = [issue for field in ('start_date', 'end_date')
                   if (issue := _date_issue(event.get(field), field))]
        if not reasons and event['end_date'] < event['start_date']:
            reasons.append('end_date_precedes_start_date')
        if event.get('status') != 'completed':
            reasons.append('phase_not_completed')
        if missing_columns:
            reasons.append('series_source_columns_missing')
        if not rows:
            reasons.append('no_exact_joined_series')
        invalid_rows = []
        for row in rows:
            errors = list(row['reconciliation']['event_day']['issues'])
            if row.get('validation_errors') not in ('[]', '', None):
                errors.append('source_validation_errors')
            if errors:
                invalid_rows.append({'source_row': row['source_row'],
                                     'wiki_series_id': row.get('wiki_series_id'),
                                     'reasons': errors, 'source': row})
        if invalid_rows:
            reasons.append('joined_series_have_date_or_source_validation_gaps')
        source_changed = phase_id in rules and rules[phase_id].get('source') != event.get('source', {})
        configured = bool(profiles[phase_id]) and not source_changed
        if source_changed:
            source_changes.append({'phase_id': phase_id, 'reason': 'catalog_source_revision_changed',
                                   'catalog_source': rules[phase_id].get('source'),
                                   'current_source': event.get('source')})
            reasons.append('catalog_source_revision_changed')
        if not configured:
            reasons.append('no_current_executable_profile')
        candidate = next((p for p in pilots if p.get('phase_id') == phase_id and
                          p['status'] == 'phase_start_reconstruction_candidate'), None)
        common = list(reasons)
        if not candidate:
            common += ['announced_entrant_ids_and_seed_slots_not_compiled']
        common += ['origin_rosters_not_reconstructed', 'chronological_fold_and_all_pair_features_required']
        state_gaps = ['timestamped_completed_results_and_published_pairings_missing',
                      'official_round_block_cutoffs_not_compiled']
        axes = {}
        for axis, extra in (
                ('axis1', ['whole_edition_boundary_and_transition_graph_not_verified']),
                ('phase_start', []), ('axis2', state_gaps),
                ('axis3', state_gaps + ['continuation_state_not_compiled']),
                ('axis4', ['requires_matched_forecast_outcome_ledger'])):
            axes[axis] = {'eligible': False, 'status': 'excluded',
                          'reasons': list(dict.fromkeys(common + extra))}
        phases.append({'phase_id': phase_id, 'tournament_id': phase_cluster[phase_id],
                       'source_event': event, 'event_status': event.get('status'),
                       'source': event.get('source'), 'source_columns': list(event),
                       'phase_start_date': event.get('start_date') if not
                                           _date_issue(event.get('start_date'), 'start_date') else None,
                       'phase_start_at': None,
                       'phase_start_evidence': {
                           'source_column': 'events.jsonl.start_date',
                           'precision': 'source_calendar_day; UTC_instant_not_certified',
                           'is_whole_edition_start': False},
                       'archived_identity_evidence': archived[phase_id],
                       'source_gaps': [
                           {'prerequisite': 'historical_availability',
                            'source_columns': ['source.revision_timestamp', 'source.retrieved_at'],
                            'reason': 'revision_and_retrieval_are_not_original_publication_times'},
                           {'prerequisite': 'checkpoint_results',
                            'source_columns': ['source_date', 'scheduled_at_utc', 'status'],
                            'missing_fields': ['completed_at', 'available_at'],
                            'reason': 'scheduled_start_and_completed_status_do_not_timestamp_completion'},
                           {'prerequisite': 'entrants_and_seeds',
                            'source_columns': ['participant_count', 'advancement_and_seeding',
                                               'draw_constraints'],
                            'reason': 'counts_and_prose_are_not_canonical_seed_slot_assignments'},
                           {'prerequisite': 'origin_rosters',
                            'source_columns': [],
                            'reason': 'phase_source_has_no_timestamped_five_role_roster'},
                       ],
                       'status': 'reconstruction_candidate' if candidate else
                                 'scenario_only' if configured else 'excluded',
                       'historical_certification': {'eligible': False, 'reasons': [
                           'historical_source_availability_not_certified',
                           'captured_forecast_not_archived']},
                       'reconstruction': {'eligible': False, 'state_candidate': candidate,
                                          'reasons': common,
                                          'captured_forecast_required': False},
                       'axes': axes, 'reasons': reasons,
                       'profile_ids': [p['id'] for p in profiles[phase_id]] if configured else [],
                       'catalog_gaps': rules.get(phase_id, {}).get('additional_gaps', []),
                       'profile_assumptions': {p['id']: p.get('assumptions', []) for p in profiles[phase_id]},
                       'group_evidence': {'records': group_index[event.get('title')],
                                          'scope': 'title_not_phase; labels_not_canonical_ids',
                                          'used_as_seed_slots': False},
                       'series_evidence': {'joined_rows': len(rows), 'invalid_rows': invalid_rows,
                                           'source_columns': list(frame.columns),
                                           'source_row_numbers': [r['source_row'] for r in rows],
                                           'join_method_counts': dict(Counter(
                                               row['reconciliation']['mapping']['method'] for row in rows)),
                                           'event_day_changes': dict(Counter(
                                               change for row in rows
                                               for change in row['reconciliation']['event_day']['changes'])),
                                           'completion_and_publication_timestamps_present': False},
                       'metric_exclusions': {
                           'champion': ['explicit_champion_target_and_outcome_link_required'],
                           'binary': ['official_finalist_or_qualification_destination_required'],
                           'placement': ['official_ordered_tied_place_bands_required'],
                           'joint': ['joint_target_and_observed_set_link_required'],
                           'series': ['temporal_origin_pairing_and_canonical_team_join_required']}})
    inputs = [bundle / 'events.jsonl', series_path, bundle / 'series_results.jsonl', catalog,
              bundle / 'source_group_definitions.jsonl', *[bundle / name for name in optional_json],
              *sorted(referenced_paths), *pilot_files, Path(__file__).resolve()]
    input_records = []
    for path in dict.fromkeys(inputs):
        actual = digest(path) if path.is_file() else None
        relative = str(path.relative_to(bundle)) if path.is_relative_to(bundle) else None
        declared = expected.get(relative, {}).get('sha256')
        record = {'path': str(path), 'exists': path.is_file(), 'sha256': actual,
                  'archived_sha256': declared,
                  'content_matches_archive': actual == declared if declared else None}
        input_records.append(record)
        if (declared and actual != declared) or (path in referenced_paths and actual is None):
            change = {'path': str(path), 'reason': 'archived_content_hash_changed' if actual else
                      'missing_referenced_source', 'expected_sha256': declared, 'sha256': actual}
            if change not in source_changes:
                source_changes.append(change)
    assigned_ids = [record['wiki_series_id'] for record in reconciliation if record['phase_id']]
    assigned_rows = [row['source_row'] for rows in joined.values() for row in rows]
    if (len(assigned_rows) != len(set(assigned_rows)) or
            len(assigned_ids) != len(set(assigned_ids)) or
            set(assigned_rows) | {row['source_row'] for row in unjoined} != set(range(2, len(source_series) + 2))):
        raise AssertionError('Readiness reconciliation lost or duplicated source series')
    return {'version': VERSION, 'qualification': 'offline_inventory_not_predictive_evidence',
            'coverage': {'source_phases': len(events),
                         'reconciled_phases': sum(len(edition['phase_ids']) for edition in editions),
                         'duplicate_series_assignments': len(assigned_ids) - len(set(assigned_ids)),
                         'completed_phases': sum(e.get('status') == 'completed' for e in events),
                         'status_counts': dict(Counter(e.get('status', 'missing') for e in events)),
                         'catalog_expected_phases': config.get('phase_count'),
                         'catalogued_phases_missing_from_source': sorted(set(rules) - set(ids)),
                         'source_phases_missing_from_catalog': sorted(set(ids) - set(rules))},
            'reconciliation_counts': {
                'source_series': len(source_series), 'reconciled_series': len(reconciliation),
                'joined_series': len(assigned_ids), 'excluded_series': len(unjoined),
                'join_methods': dict(Counter(record['mapping']['method'] for record in reconciliation
                                            if record['phase_id'])),
                'exclusion_reasons': dict(Counter(row['reason'] for row in unjoined)),
                'raw_date_issues': dict(Counter(record['event_day']['raw_date_issue']
                                               for record in reconciliation
                                               if record['event_day']['raw_date_issue'])),
                'event_day_changes': dict(Counter(change for record in reconciliation
                                                 for change in record['event_day']['changes'])),
                'normalized_event_days': sum(record['event_day']['conservative_event_day'] is not None
                                             for record in reconciliation),
                'completion_certified_series': 0,
                'edition_clusters': len(editions),
                'source_linked_edition_clusters': sum(bool(edition['phase_links']) for edition in editions),
                'verified_whole_editions': 0},
            'source_tables': {'series_results.csv': {'path': str(series_path),
                              'columns': list(frame.columns), 'missing_columns': missing_columns,
                              'rows': len(source_series), 'joined_rows': sum(map(len, joined.values())),
                              'unjoined_rows': len(unjoined)}},
            'editions': editions, 'phases': phases, 'eligible_origins': [],
            'pilot_candidates': pilots, 'unjoined_series': unjoined,
            'series_reconciliation': reconciliation, 'source_changes': source_changes,
            'unresolved_phase_links': unresolved_links,
            'exclusions': [{'phase_id': p['phase_id'], 'axis': axis, **status}
                           for p in phases for axis, status in p['axes'].items()],
            'inputs': input_records,
            'policy': {'invented_seeds': False, 'network_used': False,
                       'results_used_as_start_entrants': False,
                       'identity_matching_uses_winners_scores_or_placing': False,
                       'mechanics_profiles_certify_historical_readiness': False,
                       'retrospective_capture_required': False,
                       'checkpoint_protocol': 'strict_timestamped; day_only_origins_not_certified'}}


def write_readiness_manifest(output_dir, **kwargs):
    """Write a fresh self-contained inventory; never replace frozen artifacts."""
    manifest = build_readiness_manifest(**kwargs)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    save_json(output_dir / 'readiness_manifest.json', manifest)
    save_json(output_dir / 'exclusions.json', manifest['exclusions'])
    return manifest



def compile_pilot_inputs(output_dir, *, bundle=DEFAULT_BUNDLE, evidence_path=DEFAULT_PILOT):
    """Compile the source-backed six-seed qualification pilot, without features.

    Results are used only for graph outcome joins and checkpoint observations.
    The start spec is instantiated exclusively from prior-phase seed evidence.
    Day availability is an explicit reconstruction policy, not source certification.
    """
    output_dir, bundle, evidence_path = Path(output_dir), Path(bundle), Path(evidence_path)
    if output_dir.exists():
        raise FileExistsError(output_dir)
    evidence = json.loads(evidence_path.read_text())
    manifest = build_readiness_manifest(bundle, pilot_roots=[evidence_path.parent])
    candidate = next((p for p in manifest['pilot_candidates'] if p['path'] == str(evidence_path)
                      and p['status'] == 'phase_start_reconstruction_candidate'), None)
    if candidate is None:
        raise ValueError('Pilot requires supported prior-phase seed evidence')
    phase = evidence['source_phase']
    if phase['phase_id'] != '2a0692ac5415ae278b90':
        raise ValueError('Only the reviewed LCK 2026 Road to MSI pilot is authorized')
    spec = instantiate_profile('lck-road-to-msi', candidate['team_ids_in_seed_order'])
    stage = spec['stages'][0]
    aliases_path = bundle / 'team_aliases.json'
    aliases = json.loads(aliases_path.read_text())

    def canonical(name):
        row = aliases.get(name)
        if not row or row.get('resolved') is not True or not row.get('canonical_title'):
            raise ValueError(f'Unresolved canonical team source: {name}')
        return row['canonical_title']

    canonical_ids = {}
    for team_id in spec['teams']:
        title = canonical(evidence['team_names'][team_id])
        if title in canonical_ids:
            raise ValueError('Canonical team identity collision')
        canonical_ids[title] = team_id
    series_path = bundle / 'series_results.csv'
    source = pd.read_csv(series_path, dtype=str, keep_default_na=False)
    required = {'event_title', 'stage', 'wiki_series_id', 'team1', 'team2',
                'team1_score', 'team2_score', 'best_of', 'winner_side',
                'source_date', 'scheduled_at_utc', 'status', 'validation_errors',
                'section_path'}
    if not required.issubset(source.columns):
        raise ValueError(f'Missing pilot source columns: {sorted(required - set(source.columns))}')
    rows = source[(source.event_title == phase['title']) & (source.stage == phase['stage'])].to_dict('records')
    if len(rows) != 5 or len({r['wiki_series_id'] for r in rows}) != 5:
        raise ValueError('Pilot requires exactly five uniquely identified source series')
    for row in rows:
        if row['status'] != 'completed' or row['validation_errors'] != '[]':
            raise ValueError('Pilot has incomplete or invalid source results')
        row['team_a_id'] = canonical_ids.get(canonical(row['team1']))
        row['team_b_id'] = canonical_ids.get(canonical(row['team2']))
        if not row['team_a_id'] or not row['team_b_id']:
            raise ValueError('Source canonical team not in prior-phase entrants')
        a, b, bo = int(row['team1_score']), int(row['team2_score']), int(row['best_of'])
        if bo != 5 or max(a, b) != 3 or min(a, b) not in (0, 1, 2):
            raise ValueError('Illegal completed best-of-five source score')
        if row['winner_side'] != ('1' if a > b else '2'):
            raise ValueError('Source score and reported winner disagree')
        if _date_issue(row['source_date'], 'source_date'):
            raise ValueError('Invalid source_date in pilot')
        scheduled = _utc(row['scheduled_at_utc'])
        # Source dates are local (PST here). Retain the later UTC day so an
        # early local label cannot reveal tomorrow's UTC series.
        row['research_result_date'] = max(row['source_date'], scheduled.date().isoformat())
    results, outputs, used, provenance = [], {}, set(), []

    def resolve(slot):
        if isinstance(slot, str):
            return slot
        kind, node = next(iter(slot.items()))
        return outputs[node][0 if kind == 'winner' else 1]

    for node in stage['matches']:
        a, b = resolve(node['a']), resolve(node['b'])
        matches = [r for r in rows if r['wiki_series_id'] not in used and
                   {r['team_a_id'], r['team_b_id']} == {a, b}]
        if len(matches) != 1:
            raise ValueError(f'No unique actual series for graph node {node["id"]}')
        row = matches[0]
        used.add(row['wiki_series_id'])
        sa, sb = int(row['team1_score']), int(row['team2_score'])
        if row['team_a_id'] != a:
            sa, sb = sb, sa
        winner, loser = (a, b) if sa > sb else (b, a)
        outputs[node['id']] = (winner, loser)
        results.append({'stage': stage['id'], 'group': 'all', 'round': node['id'],
                        'team_a': a, 'team_b': b, 'best_of': 5,
                        'winner': winner, 'loser': loser, 'score_a': sa, 'score_b': sb,
                        'weight': 1, 'completed_date': row['research_result_date']})
        provenance.append({'node_id': node['id'], 'wiki_series_id': row['wiki_series_id'],
                           'source': {key: value for key, value in row.items()
                                      if key not in ('team_a_id', 'team_b_id')},
                           'canonical_mapping': {row['team1']: a if row['team_a_id'] == a else b,
                                                 row['team2']: b if row['team_b_id'] == b else a}})
    ranking = [resolve(slot) for slot in stage['ranking']]
    qualified = ranking[:stage['advance']]
    targets = [{'target_id': f'msi_qualification:{team}', 'target_kind': 'binary',
                'target_family': 'msi_qualification', 'source': 'advance_prob',
                'stage': stage['id'], 'team': team, 'categories': ['0', '1']}
               for team in spec['teams']]
    blocks = defaultdict(list)
    for row in rows:
        blocks[row['section_path']].append(row)
    if set(blocks) != {'Round 1', 'Round 2', 'Round 3', 'Round 4'}:
        raise ValueError('Unreviewed official round labels in pilot source')
    origin_days = [('phase_start', phase['start_date'])]
    for block in ('Round 1', 'Round 2', 'Round 3', 'Round 4'):
        day = max(r['research_result_date'] for r in blocks[block])
        next_day = (datetime.strptime(day, '%Y-%m-%d') + timedelta(days=1)).date().isoformat()
        origin_days.append((f'after_{block.lower().replace(" ", "_")}', next_day))
    origins, states, outcome_rows = [], {}, []
    for origin_id, day in origin_days:
        cutoff = f'{day}T00:00:00+00:00'
        completed = [r for r in results if r['completed_date'] < day]
        known = {r['round']: r for r in completed}
        resolved_yes = {known[n]['winner'] for n in ('upper', 'last') if n in known}
        resolved_no = {r['loser'] for r in completed if r['round'] != 'upper'}
        eligible = [t for t in spec['teams'] if t not in resolved_yes | resolved_no]
        future = [r for r in rows if r['research_result_date'] >= day]
        target_start = min((r['scheduled_at_utc'] for r in future), default=None)
        origin_targets = [{**t, 'resolved': t['team'] in resolved_yes | resolved_no,
                           'eligible_teams': eligible, 'remaining_slots': 2 - len(resolved_yes),
                           'target_start_at': target_start} for t in targets]
        state_name = f'state-{origin_id}.json'
        states[state_name] = {'version': 1, 'cutoff': cutoff,
                             'temporal_policy': 'prior_day_reconstruction', 'completed': completed}
        origins.append({'origin_id': origin_id, 'axis': 'axis1' if origin_id == 'phase_start' else 'axis3',
                        'scope': 'phase_start' if origin_id == 'phase_start' else 'remaining_phase',
                        'cutoff': cutoff, 'state': state_name, 'targets': origin_targets,
                        'terminal': not future})
        outcome_rows.extend({'tournament_id': phase['phase_id'], 'origin_id': origin_id,
                             'target_id': t['target_id'], 'observed': '1' if t['team'] in qualified else '0'}
                            for t in targets)
    input_paths = [evidence_path, bundle / 'events.jsonl', DEFAULT_CATALOG, aliases_path, series_path]
    output = {'version': 'tournament-pilot-inputs-v1', 'scope': 'phase_start',
              'qualification': 'reconstructed_not_certified', 'tournament_id': phase['phase_id'],
              'phase_id': phase['phase_id'], 'title': phase['title'],
              'phase_start_date': phase['start_date'], 'phase_end_date': phase['end_date'],
              'family': 'lck', 'tier': 'Major', 'format': ['graph', 'gauntlet'],
              'spec': 'spec.json', 'targets': targets, 'origins': origins,
              'source_hashes': {str(p): digest(p) for p in input_paths},
              'temporal_policy': {'name': 'prior_day_reconstruction',
                                  'cutoff': 'Declared UTC midnight forecast origins, not event kickoff times.',
                                  'result_day': 'Later of source_date and scheduled_at_utc calendar day.',
                                  'limitation': 'Assumes completed results available on next UTC day; no certified completion/publication timestamp.'},
              'exclusions': {'champion': 'Profile explicitly has two qualification routes and no championship final.',
                             'placement': 'Graph ranks do not establish official ordered place bands.',
                             'axis2': 'Future pairing publication times absent; no realized future-pair forecasts.'}}
    output_dir.mkdir(parents=True, exist_ok=False)
    save_json(output_dir / 'spec.json', spec)
    for filename, state in states.items():
        save_json(output_dir / filename, state)
    save_json(output_dir / 'inputs.json', output)
    save_json(output_dir / 'outcomes.json', {'qualification': output['qualification'],
              'result_rows': results, 'source_series': provenance,
              'qualified_team_ids': qualified, 'outcomes': outcome_rows})
    save_json(output_dir / 'artifact_manifest.json', {
        p.name: digest(p) for p in sorted(output_dir.iterdir()) if p.is_file()})
    return output


def audit(args):
    args.output_dir.mkdir(parents=True, exist_ok=False)
    readiness = build_readiness_manifest(getattr(args, 'bundle', DEFAULT_BUNDLE),
                                        catalog=getattr(args, 'catalog', DEFAULT_CATALOG))
    save_json(args.output_dir/'readiness_manifest.json', readiness)
    roots = args.event_root or [ROOT/'data']
    protocol = {'version': VERSION, 'created_at': datetime.now(timezone.utc).isoformat(),
                'cohort': f'all snapshot series on/after {args.from_date}; no bookmaker filtering',
                'from_date': args.from_date, 'event_search_roots': [str(root) for root in roots],
                'unit': 'source series and exact tournament label, not simulated games',
                'rules': ['No later realized opponents used to construct earlier draws or seeds.',
                          'Per-match retrospective features are not event-start all-pair features.',
                          'Actual game rosters are not announced start-frozen rosters.',
                          'Date-only replay and retrieval timestamps are not publication certification.',
                          'Full-event scores require complete graph-linked observed outcomes.',
                          'No market/Benter teachers or bookmaker cohort intersection.'],
                'input_sha256': {str(path): digest(path) for path in [
                    args.snapshots, args.history, args.requests, args.replay_audit]},
                'source_sha256': digest(__file__)}
    save_json(args.output_dir/'protocol.json', protocol)
    replay_audit = json.loads(args.replay_audit.read_text())
    if (protocol['input_sha256'][str(args.history)] != replay_audit['source']['sha256']
            or protocol['input_sha256'][str(args.snapshots)] != replay_audit['outputs']['snapshots.csv']):
        raise ValueError('history and snapshots must match their originating replay audit')
    frame = _read_csv(args.snapshots)
    required = {'golgg_match_id', 'date', 'tournament', 'team1_id', 'team2_id', 'best_of', 'y_true'}
    if not required.issubset(frame.columns) or frame[list(required)].isna().any().any():
        raise ValueError('snapshot metadata must be complete')
    if frame.golgg_match_id.duplicated().any():
        raise ValueError('snapshot series IDs must be unique')
    dates = pd.to_datetime(frame.date, errors='raise')
    frame = frame.loc[dates >= pd.Timestamp(args.from_date)].copy()
    if frame.empty:
        raise ValueError('no rows in requested modern cohort')
    if not frame.y_true.isin([0, 1]).all():
        raise ValueError('binary series outcomes required')
    ids = set(frame.golgg_match_id)
    raw, historical_events = {}, defaultdict(list)
    for match in iter_matches(args.history):
        match_id = str(match.get('match_id', ''))
        name = match.get('tournament_name')
        day = match.get('date')
        if name and isinstance(day, str) and day >= args.from_date:
            historical_events[name].append({'match_id': match_id, 'date': day,
                                           'team1': match.get('sname_t1'), 'team2': match.get('sname_t2'),
                                           'winner': match.get('won')})
        if match_id in ids:
            if match_id in raw:
                raise ValueError(f'duplicate source match ID: {match_id}')
            raw[match_id] = {key: match.get(key) for key in (
                'date', 'tournament_name', 'best_of')}
            raw[match_id]['team_ids'] = _series_team_ids(match)
            score1, score2 = golgg_schema.score1(match), golgg_schema.score2(match)
            if score1 == score2:
                raise ValueError(f'snapshot references a tied source series: {match_id}')
            raw[match_id]['y_true'] = int(score1 > score2)
    bundles, bundle_files = audit_bundles(roots)
    pd.DataFrame(bundles, columns=sorted(set().union(*(row.keys() for row in bundles))) if bundles else [
        'event_path', 'tournament_id', 'prestart_local_bundle', 'complete_observed_results',
        'eligible_full_event_replay', 'eligible_live', 'point_in_time_certified', 'reasons']).to_csv(args.output_dir/'bundle_eligibility.csv', index=False)
    request_rows, event_pages, game_pages = [], defaultdict(list), defaultdict(list)
    with args.requests.open() as stream:
        for line_number, line in enumerate(stream, 1):
            request = json.loads(line)
            url = request.get('url', '')
            url_path = unquote(urlparse(url).path)
            event_name = ''
            match_id = ''
            kind = 'other'
            if '/tournament/tournament-matchlist/' in url_path:
                event_name = url_path.split('/tournament/tournament-matchlist/', 1)[1].strip('/')
                kind = 'retrospective_matchlist_not_draw_rules'
            elif '/game/stats/' in url_path:
                match_id = url_path.split('/game/stats/', 1)[1].split('/')[0]
                kind = 'retrospective_game_page'
            body = request.get('body_path')
            body_path = args.requests.parent/body if isinstance(body, str) else None
            exists = body_path is not None and body_path.is_file()
            row = {'request_line': line_number, 'kind': kind, 'tournament': event_name,
                   'golgg_match_id': match_id, 'url': url, 'retrieved_at': request.get('retrieved_at'),
                   'status_code': request.get('status_code'), 'body_path': str(body_path) if body_path else '',
                   'body_exists': int(exists), 'recorded_content_sha256': request.get('sha256'),
                   'content_hash_verified': 0, 'publication_time_certified': 0}
            # Verify actual modern event pages, not just references in a request log.
            if event_name in historical_events and exists:
                with gzip.open(body_path, 'rb') as body_stream:
                    actual = hashlib.file_digest(body_stream, 'sha256').hexdigest()
                row['content_hash_verified'] = int(actual == request.get('sha256'))
            request_rows.append(row)
            if event_name:
                event_pages[event_name].append(row)
            if match_id:
                game_pages[match_id].append(row)
    pd.DataFrame(request_rows).to_csv(args.output_dir/'source_request_inventory.csv', index=False)
    row_evidence = []
    for row in frame.itertuples(index=False):
        match = raw.get(row.golgg_match_id)
        reasons = list(REASONS)
        aligned = False
        if match is None:
            reasons.append('source_match_missing')
        else:
            aligned = (match.get('date') == row.date and match.get('tournament_name') == row.tournament
                       and match.get('best_of') == row.best_of
                       and match['team_ids'] == (row.team1_id, row.team2_id)
                       and match['y_true'] == row.y_true)
            if not aligned:
                reasons.append('source_outcome_metadata_mismatch')
        pages = game_pages[row.golgg_match_id]
        row_evidence.append({'golgg_match_id': row.golgg_match_id, 'tournament': row.tournament,
                             'date': row.date, 'team1_id': row.team1_id, 'team2_id': row.team2_id,
                             'best_of': row.best_of, 'y_true': row.y_true,
                             'source_match_present': int(match is not None), 'source_metadata_outcome_aligned': int(aligned),
                             'archived_game_pages': sum(page['body_exists'] for page in pages),
                             'retrospective_match_replay_reachable': int(aligned),
                             'eligible_full_event_replay': 0, 'eligibility_live': 0,
                             'point_in_time_certified': 0, 'reasons': ';'.join(reasons)})
    evidence = pd.DataFrame(row_evidence)
    evidence.to_csv(args.output_dir/'row_eligibility.csv', index=False)
    event_rows = []
    for name in sorted(set(historical_events) | set(frame.tournament)):
        source = historical_events.get(name, [])
        selected = frame[frame.tournament == name]
        pages = event_pages[name]
        teams = sorted({team for match in source for team in (match['team1'], match['team2']) if team})
        formats = sorted(selected.best_of.unique().tolist())
        pairs = {tuple(sorted((row.team1_id, row.team2_id))) for row in selected.itertuples()}
        reasons = list(REASONS) + ['source_matchlist_does_not_establish_complete_graph_outcomes']
        event_rows.append({'tournament': name, 'source_series': len(source), 'snapshot_series': len(selected),
                           'first_observed_series_date': min((match['date'] for match in source), default=''),
                           'last_observed_series_date': max((match['date'] for match in source), default=''),
                           'observed_teams_count_not_announced_entrants': len(teams),
                           'observed_team_names': json.dumps(teams), 'realized_pairs': len(pairs),
                           'observed_best_of': json.dumps(formats),
                           'all_pair_feature_rows_at_start': 0,
                           'verified_archived_matchlists': sum(page['content_hash_verified'] for page in pages),
                           'matchlist_last_retrieved_at': max((page['retrieved_at'] or '' for page in pages), default=''),
                           'retrospective_match_replay_reachable': int(not selected.empty and evidence.loc[evidence.tournament == name, 'source_metadata_outcome_aligned'].all()),
                           'eligible_full_event_replay': 0, 'eligibility_live': 0, 'point_in_time_certified': 0,
                           'reasons': ';'.join(reasons)})
    pd.DataFrame(event_rows).to_csv(args.output_dir/'event_eligibility.csv', index=False)
    cache_rows = []
    cache_files = sorted(args.cache_dir.glob('*.json'))
    for path in cache_files:
        cache = json.loads(path.read_text())
        for node_id, match in cache.get('matches', {}).items():
            complete = bool(match.get('team1') and match.get('team2') and match.get('winner'))
            cache_rows.append({'tournament_id': cache.get('tournament_id'), 'node_id': node_id,
                               **{field: match.get(field) for field in ['team1', 'team2', 'winner', 'score1', 'score2']},
                               'cache_source': cache.get('source'), 'synced_at': cache.get('synced_at'),
                               'observed_result_present': int(complete), 'eligible_full_event_replay': 0,
                               'reasons': 'live_partial_result_cache_not_start_frozen_graph;no_start_rosters;no_all_pair_features;no_prestart_forecast' +
                               ('' if complete else ';outcome_incomplete')})
    pd.DataFrame(cache_rows).to_csv(args.output_dir/'cached_node_eligibility.csv', index=False)
    inventory_paths = [args.snapshots, args.history, args.requests, args.replay_audit, Path(__file__),
                       ROOT/'scripts/tournament_research_snapshot.py', ROOT/'scripts/export_prospective_features.py',
                       ROOT/'scripts/score_tournament_forecasts.py', ROOT/'scripts/prospective_sports_features.py',
                       ROOT/'src/utils/golgg_schema.py',
                       *cache_files, *bundle_files]
    inventory = [{'path': str(path), 'exists': path.is_file(), 'sha256': digest(path) if path.is_file() else None,
                  'bytes': path.stat().st_size if path.is_file() else None} for path in sorted(set(inventory_paths))]
    pd.DataFrame(inventory).to_csv(args.output_dir/'provenance_inventory.csv', index=False)
    reachable = [{'experiment': 'chronological_actual_series_baselines_and_ablations',
                  'status': 'reachable_local_retrospective', 'input': str(args.snapshots),
                  'row_count': int(evidence.retrospective_match_replay_reachable.sum()),
                  'meaning': 'Outcome-supervised past-date match replay only, not tournament-start forecast evidence.'},
                 {'experiment': 'event_start_champion_final_node_scores',
                  'status': 'reachable' if any(row['eligible_full_event_replay'] for row in bundles) else 'missing_prerequisites',
                  'qualified_bundles': [row['event_path'] for row in bundles if row['eligible_full_event_replay']],
                  'requires': 'Existing prestart capture, complete all-pair sports features, announced five-role rosters, authentic graph/draw rules, complete graph-linked results.'},
                 {'experiment': 'retrospective_synthetic_graph_mechanics_and_uncertainty',
                  'status': 'reachable_demonstration_not_historical_validation',
                  'input': 'src/models/frozen_tournament.py; explicit research-only graph',
                  'meaning': 'Analytic/parity/sensitivity proof only; never present synthetic winners or draws as historical evidence.'}]
    save_json(args.output_dir/'reachable_experiments.json', reachable)
    summary = {'version': VERSION, 'snapshot_rows': len(frame), 'historical_event_labels': len(event_rows),
               'source_aligned_rows': int(evidence.source_metadata_outcome_aligned.sum()),
               'existing_event_bundles_found': len(bundles),
               'full_event_replay_eligible_bundles': sum(row['eligible_full_event_replay'] for row in bundles),
               'point_in_time_certified_events': 0, 'cache_events': len(cache_files),
               'cache_completed_nodes': sum(row['observed_result_present'] for row in cache_rows),
               'cache_nodes': len(cache_rows), 'archived_request_rows': len(request_rows),
               'verified_modern_event_matchlists': sum(row['content_hash_verified'] for row in request_rows),
               'qualification': 'retrospective_provenance_audit_not_source_certified',
               'limitations': REASONS + ['Event labels and observed-team union are not an announced entrant list.',
                                         'Request hashes for non-event-page bodies are recorded assertions, not reverified content.',
                                         'Local receipt and file mtimes cannot establish historical publication.',
                                         'Source series result coverage does not establish an event champion or draw rules.'],
               'reachable_experiments': reachable, 'production_qualified': False}
    save_json(args.output_dir/'summary.json', summary)
    save_json(args.output_dir/'manifest.json', {'inputs': inventory, 'files': {
        path.name: digest(path) for path in sorted(args.output_dir.iterdir()) if path.is_file()}})
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', required=True, type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--inventory-only', action='store_true',
                      help='Inventory all archived phases without requiring match replay snapshots.')
    mode.add_argument('--compile-pilot', action='store_true',
                      help='Compile the source-backed LCK 2026 phase-start research inputs and separate outcomes.')
    parser.add_argument('--pilot-evidence', type=Path, default=DEFAULT_PILOT)
    parser.add_argument('--bundle', type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument('--catalog', type=Path, default=DEFAULT_CATALOG)
    parser.add_argument('--snapshots', type=Path, default=DEFAULT_SNAPSHOTS)
    parser.add_argument('--history', type=Path, default=DEFAULT_HISTORY)
    parser.add_argument('--requests', type=Path, default=DEFAULT_REQUESTS)
    parser.add_argument('--replay-audit', type=Path, default=DEFAULT_SNAPSHOTS.parent/'audit.json')
    parser.add_argument('--cache-dir', type=Path, default=ROOT/'data/cache/tournaments')
    parser.add_argument('--event-root', type=Path, action='append')
    parser.add_argument('--from-date', default='2024-01-01')
    args = parser.parse_args()
    if args.compile_pilot:
        print(json.dumps(compile_pilot_inputs(args.output_dir, bundle=args.bundle,
                                             evidence_path=args.pilot_evidence), indent=2))
    elif args.inventory_only:
        result = write_readiness_manifest(args.output_dir, bundle=args.bundle, catalog=args.catalog)
        print(json.dumps({'version': result['version'], 'coverage': result['coverage'],
                          'reconciliation_counts': result['reconciliation_counts'],
                          'source_changes': len(result['source_changes']),
                          'eligible_origins': len(result['eligible_origins']),
                          'pilot_candidates': result['pilot_candidates'],
                          'manifest': str(args.output_dir/'readiness_manifest.json')}, indent=2))
    else:
        print(json.dumps(audit(args), indent=2))


if __name__ == '__main__':
    main()
