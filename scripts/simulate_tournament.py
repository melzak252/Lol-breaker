#!/usr/bin/env python3
"""Offline tournament preparation and simulation with explicit inputs only.

Examples (run from repository root):
  python -m scripts.simulate_tournament profiles
  python -m scripts.simulate_tournament prepare --spec event.json --golgg-json history.json \
      --rosters rosters.json --cutoff 2026-06-06T00:00:00Z --output pairs.json
  python -m scripts.simulate_tournament simulate --spec event.json --artifact fold.json \
      --model-kind exp081 --pairs pairs.json --mode chronological --cutoff 2026-06-06T00:00:00Z --output result.json

Roster JSON is {"teamID": ["playerID", ... exactly five IDs]}. Alternatively,
--previous-observed-roster-proxy opts into the last strictly prior map roster.
Never supply target-event realized rosters as pre-start information. Prepared
bundles use canonical raw all-pair features, not training/realized-fixture rows.

An explicit probability table is {"probability_unit":"series_win", "rows":[
{"team_a":"A", "team_b":"B", "best_of":3, "p":0.6}], "provenance":{...}}.
Every unordered pair and engine-requested format is required. Values are direct
series probabilities, not market odds, map probabilities, or lower-risk scores.
No source scraping, live database access, training, or default artifact loading.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from itertools import combinations
import json
import math
from pathlib import Path
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.tournament_prediction import (
    file_digest, fit_score_distributions, golgg_score_rows, json_digest,
    predict_model_pairs, prepare_pair_bundle, previous_observed_rosters, utc,
)


def _load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _write(path, payload):
    text = json.dumps(payload, indent=2, allow_nan=False) + '\n'
    if path is None:
        print(text, end='')
    else:
        # Never overwrite a frozen artifact or a previous run by accident.
        with Path(path).open('x', encoding='utf-8') as stream:
            stream.write(text)


def _spec(args):
    if args.spec is not None:
        return _load(args.spec), {'spec_file': file_digest(args.spec)}
    if args.profile_input is None:
        raise ValueError('--profile requires --profile-input JSON containing {"teams":[...]}')
    from src.models.tournament_catalog import instantiate_profile
    inputs = _load(args.profile_input)
    return instantiate_profile(args.profile, **inputs), {'profile_input_file': file_digest(args.profile_input)}


def _history(path):
    # Existing streaming reader supports JSON arrays and gzip; no source loaded in RAM.
    from scripts.export_prospective_features import iter_matches
    return iter_matches(path)


def _prepare(args):
    from src.models.tournament_formats import required_best_ofs
    spec, hashes = _spec(args)
    at = utc(args.cutoff)
    formats = sorted(required_best_ofs(spec))
    hashes['history_file'] = file_digest(args.golgg_json)
    observations = None
    if args.previous_observed_roster_proxy:
        rosters, observations = previous_observed_rosters(_history(args.golgg_json), spec['teams'], at.date())
    else:
        rosters = _load(args.rosters)
        hashes['rosters_file'] = file_digest(args.rosters)
    if set(rosters) != set(spec['teams']):
        raise ValueError('roster IDs must exactly match all global spec team IDs')
    roster_provenance = _load(args.roster_provenance) if args.roster_provenance else {}
    if args.roster_provenance:
        hashes['roster_provenance_file'] = file_digest(args.roster_provenance)
    bundle = prepare_pair_bundle(_history(args.golgg_json), rosters, cutoff=args.cutoff,
                                 best_ofs=formats, input_hashes=hashes,
                                 roster_provenance=roster_provenance)
    if observations is not None:
        bundle['roster_policy'] = 'previous_observed_roster_proxy'
        bundle['previous_roster_observations'] = observations
    bundle['spec_sha256'] = json_digest(spec)
    if args.history_provenance:
        bundle['history_provenance'] = _load(args.history_provenance)
        bundle['input_hashes']['history_provenance_file'] = file_digest(args.history_provenance)
    if args.fit_scores:
        distributions, audit = fit_score_distributions(
            golgg_score_rows(_history(args.golgg_json), at.date()), cutoff=at.date(), best_ofs=formats)
        bundle['score_model'] = {'distributions': distributions, 'provenance': audit}
    if file_digest(args.golgg_json) != hashes['history_file']:
        raise ValueError('historical source changed during preparation')
    return bundle


def _table_probabilities(table, spec, formats, cutoff):
    if not isinstance(table, dict) or table.get('probability_unit') != 'series_win':
        raise ValueError('explicit table must declare probability_unit=series_win')
    expected = {(frozenset((a, b)), bo) for a, b in combinations(spec['teams'], 2) for bo in formats}
    seen, probabilities = set(), {}
    for row in table['rows']:
        if set(row) != {'team_a', 'team_b', 'best_of', 'p'}:
            raise ValueError('table rows require exactly team_a, team_b, best_of, p')
        a, b, bo, p = row['team_a'], row['team_b'], row['best_of'], row['p']
        if not isinstance(a, str) or not isinstance(b, str) or type(bo) is not int:
            raise ValueError('invalid probability pair identity or format')
        key = (frozenset((a, b)), bo)
        if key not in expected or key in seen or a == b:
            raise ValueError('series table requires each unordered pair/format exactly once')
        if isinstance(p, bool) or not isinstance(p, (int, float)) or not math.isfinite(p) or not 0 <= p <= 1:
            raise ValueError('series probabilities must be finite within [0,1]')
        seen.add(key)
        probabilities[a, b, bo] = p
    if seen != expected:
        raise ValueError('incomplete series probability table; all counterfactual pairs required')
    provenance = table.get('provenance', {})
    if not isinstance(provenance, dict):
        raise ValueError('table provenance must be an object')
    if provenance.get('feature_history_max_at') and utc(provenance['feature_history_max_at']) > cutoff:
        raise ValueError('table feature history extends after tournament cutoff')
    return probabilities, {
        'qualification': 'explicit_series_table_not_pit_certified', 'probability_unit': 'series_win',
        'table_sha256': json_digest(table), 'supplied_provenance': provenance,
        'limitation': 'table origin and calibration are caller-supplied; not certified by this engine',
    }


def _simulate(args):
    from src.models.tournament_formats import required_best_ofs, simulate_tournament
    spec, hashes = _spec(args)
    at = utc(args.cutoff)
    formats = sorted(required_best_ofs(spec))
    score_model = None
    if args.series_table:
        if args.pairs is not None:
            raise ValueError('--pairs is only valid with --artifact')
        if args.model_kind is not None:
            raise ValueError('--model-kind is only valid with --artifact')
        table = _load(args.series_table)
        probabilities, provenance = _table_probabilities(table, spec, formats, at)
        hashes['series_table_file'] = file_digest(args.series_table)
    else:
        if args.pairs is None:
            raise ValueError('--artifact requires --pairs prepared bundle')
        if args.model_kind is None:
            raise ValueError('--artifact requires explicit --model-kind')
        artifact_hash = file_digest(args.artifact)
        if args.model_kind in ('exp039', 'linear79'):
            import joblib
            artifact = joblib.load(args.artifact)
        else:
            artifact = _load(args.artifact)
        if file_digest(args.artifact) != artifact_hash:
            raise ValueError('artifact changed while loading')
        bundle = _load(args.pairs)
        probabilities, provenance = predict_model_pairs(
            artifact, bundle, model_kind=args.model_kind, artifact_sha256=artifact_hash,
            teams=spec['teams'], best_ofs=formats, cutoff=args.cutoff, mode=args.mode)
        hashes.update(artifact_file=artifact_hash, pairs_file=file_digest(args.pairs))
        score_model = bundle.get('score_model')
    if args.score_model:
        score_model = _load(args.score_model)
        hashes['score_model_file'] = file_digest(args.score_model)
    scores = None
    if score_model is not None:
        score_provenance = score_model['provenance']
        if score_provenance.get('cutoff') != at.date().isoformat():
            raise ValueError('score model cutoff must match tournament cutoff date')
        if score_provenance.get('source_date_max') and score_provenance['source_date_max'] >= at.date().isoformat():
            raise ValueError('score model consumes non-prior historical results')
        scores = {int(bo): values for bo, values in score_model['distributions'].items()}
    state = _load(args.state) if getattr(args, 'state', None) else None
    if state is not None:
        if utc(state['cutoff']) != at:
            raise ValueError('state cutoff must match forecast cutoff')
        hashes['state_file'] = file_digest(args.state)
    result = simulate_tournament(spec, probabilities, simulations=args.simulations,
                                 seed=args.seed, score_distributions=scores, state=state)
    result['provenance'] = {
        **provenance, 'input_file_sha256': hashes, 'spec_sha256': json_digest(spec),
        'spec': spec, 'seed': args.seed, 'tournament_cutoff': at.isoformat(),
        'run_at': datetime.now(timezone.utc).isoformat(), 'score_model': score_model,
        'code_sha256': {name: file_digest(Path(__file__).resolve().parents[1] / name)
                        for name in ('src/models/tournament_prediction.py', 'src/models/tournament_formats.py',
                                     'src/models/siamese_series.py', 'src/models/symmetric_series.py',
                                     'scripts/simulate_tournament.py', 'scripts/prospective_sports_features.py')},
    }
    return result


def _fit_scores(args):
    at = utc(args.cutoff)
    if args.golgg_json:
        rows = golgg_score_rows(_history(args.golgg_json), at.date())
        source = args.golgg_json
    else:
        rows, source = _load(args.score_history), args.score_history
    digest = file_digest(source)
    distributions, audit = fit_score_distributions(rows, cutoff=at.date(), best_ofs=args.best_ofs)
    if file_digest(source) != digest:
        raise ValueError('score source changed during fitting')
    audit['source_file_sha256'] = digest
    return {'distributions': distributions, 'provenance': audit}


def _add_spec(parser):
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument('--spec', type=Path, help='engine spec JSON (global exact team IDs)')
    choice.add_argument('--profile', help='catalog profile ID; scenario assumptions remain in provenance')
    parser.add_argument('--profile-input', type=Path, help='JSON {"teams":[IDs in seed order]} for catalog API')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest='command', required=True)
    profiles = commands.add_parser('profiles', help='list catalog IDs, team counts, sources and assumptions')
    profiles.add_argument('--output', type=Path)
    prepare = commands.add_parser('prepare', help='strictly prior local GOLGG -> frozen complete all-pair sports features')
    _add_spec(prepare)
    prepare.add_argument('--golgg-json', type=Path, required=True, help='local GOLGG JSON array or gzip; no network')
    roster = prepare.add_mutually_exclusive_group(required=True)
    roster.add_argument('--rosters', type=Path, help='explicit {team ID: [five player IDs]} JSON')
    roster.add_argument('--previous-observed-roster-proxy', action='store_true', help='EXPLICIT retrospective diagnostic proxy; last prior map only')
    prepare.add_argument('--roster-provenance', type=Path, help='optional archived roster sources/announcement metadata JSON')
    prepare.add_argument('--history-provenance', type=Path, help='optional source inventory/quarantine/projection audit JSON, preserved without certifying completeness')
    prepare.add_argument('--fit-scores', action='store_true', help='also fit empirical losing-map PMFs from strictly prior GOLGG scores; no fallback if unsupported')
    prepare.add_argument('--cutoff', required=True, help='timezone-aware tournament pre-start cutoff; excludes its entire UTC calendar day')
    prepare.add_argument('--output', type=Path, required=True, help='NEW prepared JSON bundle file')
    simulate = commands.add_parser('simulate', help='engine spec + explicit series table OR local research model and frozen pairs')
    _add_spec(simulate)
    model = simulate.add_mutually_exclusive_group(required=True)
    model.add_argument('--series-table', type=Path, help='complete unordered series probabilities; never map probabilities or market odds')
    model.add_argument('--artifact', type=Path, help='explicit TRUSTED LOCAL JSON/joblib research fold; joblib executes Python, never load untrusted files')
    simulate.add_argument('--model-kind', choices=['exp039', 'linear79', 'exp081', 'glicko', 'glicko_format'],
                          help='required with artifact: joblib for exp039/linear79, JSON otherwise; no production default')
    simulate.add_argument('--pairs', type=Path, help='prepared bundle from prepare (required with --artifact)')
    simulate.add_argument('--mode', choices=['verified', 'exploratory', 'chronological'], default='chronological',
                          help='chronological requires shared feature semantics; verified additionally requires availability certification')
    simulate.add_argument('--cutoff', required=True, help='timezone-aware tournament cutoff, identical to prepared bundle')
    simulate.add_argument('--score-model', type=Path, help='optional empirical PMF bundle from fit-scores; overrides embedded score model')
    simulate.add_argument('--state', type=Path, help='timestamped completed results and published draws; same original spec')
    simulate.add_argument('--simulations', type=int, default=1000)
    simulate.add_argument('--seed', type=int, default=81)
    simulate.add_argument('--output', type=Path, help='NEW JSON output file; omitted writes stdout')
    scores = commands.add_parser('fit-scores', help='fit unsmoothed prior-only losing-map PMFs, separate from series winner model')
    source = scores.add_mutually_exclusive_group(required=True)
    source.add_argument('--golgg-json', type=Path, help='local completed GOLGG source array')
    source.add_argument('--score-history', type=Path, help='JSON [{id,date,best_of,score_a,score_b,end_date?}]')
    scores.add_argument('--best-ofs', type=int, nargs='+', choices=[1, 3, 5], required=True)
    scores.add_argument('--cutoff', required=True)
    scores.add_argument('--output', type=Path, required=True)
    evaluate = commands.add_parser('evaluate', help='forecast immutable manifest origins; never read future outcome labels')
    evaluate.add_argument('--manifest', type=Path, required=True)
    evaluate.add_argument('--output-dir', type=Path, required=True)
    evaluate.add_argument('--simulations', type=int, default=20000)
    evaluate.add_argument('--max-simulations', type=int, default=200000,
                          help='double samples until simultaneous category and log-probability precision passes or this cap')
    evaluate.add_argument('--seed', type=int, default=81)
    evaluate.add_argument('--resume', action='store_true', help='resume only when code, manifest and input hashes match')
    score = commands.add_parser('score', help='join outcomes to a completed immutable four-axis forecast ledger')
    score.add_argument('--run-dir', type=Path, required=True)
    score.add_argument('--outcomes', type=Path, required=True)
    score.add_argument('--output', type=Path, required=True)
    score.add_argument('--bootstrap', type=int, default=5000)
    score.add_argument('--seed', type=int, default=82)
    args = parser.parse_args(argv)
    try:
        if args.command == 'evaluate':
            from scripts.tournament_evaluation import run_evaluation
            result = run_evaluation(args.manifest, args.output_dir, simulations=args.simulations,
                                    max_simulations=args.max_simulations, seed=args.seed, resume=args.resume)
            print(json.dumps(result, allow_nan=False, indent=2))
            return
        if args.command == 'score':
            from scripts.tournament_evaluation import score_evaluation
            result = score_evaluation(args.run_dir, args.outcomes, args.output,
                                      bootstrap=args.bootstrap, seed=args.seed)
            print(json.dumps({'output': str(args.output), 'production_qualified': result['production_qualified']}))
            return
        if args.output is not None and args.output.exists():
            raise ValueError('output already exists; choose a new path to preserve frozen inputs and runs')
        if args.command == 'profiles':
            from src.models.tournament_catalog import list_profiles
            result = list_profiles()
        elif args.command == 'prepare':
            result = _prepare(args)
        elif args.command == 'simulate':
            result = _simulate(args)
        else:
            result = _fit_scores(args)
        _write(args.output, result)
    except (ValueError, KeyError, TypeError, OSError) as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    main()
