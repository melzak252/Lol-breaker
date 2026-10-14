#!/usr/bin/env python3
"""Independent R1 historical-source and saved-law oracle; never estimate a model.

Expected values use source bytes and separately written NumPy/SciPy expressions.
Native snapshot inference is an actual-under-test parity check, not independent
reconstruction of the protected C0 network. Toy algebra is reported separately.

Parent commands (each --output must be a new file):
  python scripts/verify_lineup_law_reliability.py prepare --run-id RUN \\
      --repo-root REPO --research-root ARCHIVE --output PREPARE_RECEIPT.json
  python scripts/verify_lineup_law_reliability.py completed --run-id RUN \\
      --repo-root REPO --research-root ARCHIVE --output COMPLETED_RECEIPT.json
The closed-support branch deliberately does not exercise fitted/native inference.
Hard-stop receipts select forensic partial replay, never scientific completion.
Every unavailable check retains its original failure reason and NOT_RUN status;
unexpected execution/integrity failures remain FAIL. Fitted PASS is finalized
only after scientific decisions and final source/code/protected integrity checks.
Positive OTHER mass, including below one float64 ulp, always withholds a scalar.
"""
from __future__ import annotations

import argparse
import base64
import ast
import bisect
import copy
import hashlib
import json
import math
import os
import platform
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from contextlib import closing
from datetime import date, datetime, timezone
# The verifier and its owned native worker share the authorized CPU ceiling.
for _thread_env in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[_thread_env] = "4"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
from pathlib import Path

import numpy as np
import scipy
import pandas as pd
from scipy.special import logsumexp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
MONTHS = tuple(f"2026-{m:02d}" for m in range(1, 10))
INDEX_FIELDS = ("schema_version", "study", "run_id", "scenario", "branch_status", "created_at_utc", "protocol_sha256", "binding_sha256", "oracle_contract_sha256", "artifact_descriptors", "population_ids", "operation_trace", "missingness_ledger", "public_cases", "prefit_resolutions")
DESCRIPTOR_FIELDS = ("role", "repo_relative_path", "sha256", "format", "shape_or_row_count", "dtype_or_columns", "units", "ordered_axis_names", "ordered_id_axis_reference", "absence_status", "absence_reason")
LIMITS = ["Historical exposed development only; no authentic issuance or unexposed confirmation.", "Source authenticity and causal identification are not established by arithmetic agreement.", "C0 snapshot replay checks the public protected implementation, not an independently reconstructed neural network.", "Side independence is provisional; roster bounds are algebraic missing-mass bounds, not confidence intervals.", "No uncertain-case scalar winner forecast, winner superiority, business/ROI/EV/deployment claim.", "Shared source bytes, NumPy/SciPy and public native contract dependencies remain disclosed."]


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def load(path):
    def reject(value):
        raise ValueError("nonfinite JSON token " + value)
    return json.loads(Path(path).read_text(), parse_constant=reject)


def safe(value):
    if isinstance(value, dict):
        return {str(k): safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list, np.ndarray)):
        return [safe(v) for v in value]
    if isinstance(value, np.generic):
        return safe(value.item())
    if value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def ident(value):
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, (bool, np.bool_)):
        raise ValueError("boolean identity")
    if isinstance(value, (float, np.floating)):
        if math.isnan(value):
            return None
        if not math.isfinite(value) or not value.is_integer():
            raise ValueError("nonintegral identity")
        value = int(value)
    value = str(value).strip()
    return None if value.lower() in ("", "none", "null", "nan", "<na>") else value


def day(value):
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc).date() if value.tzinfo else value.date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value))


def order(value):
    return (0, int(value)) if value.isascii() and value.isdecimal() else (1, value)


def context(raw):
    from src.utils.golgg_schema import match_tournament
    a, b = ident(match_tournament(raw)), ident(raw.get("competition_context"))
    if a and b and a != b:
        raise ValueError("exact competition contexts disagree")
    return a or b


def five(values):
    values = [ident(v) for v in values]
    present = [v for v in values if v is not None]
    if len(values) > 5 or len(present) != len(set(present)):
        raise ValueError("corrupt roster identity")
    return sorted(present) if len(present) == 5 else None


class Audit:
    def __init__(self):
        self.checks, self.failures = [], []
        self.kinds = Counter()
        self.numeric_elements = 0

    def check(self, name, actual, expected, *, atol=None, rtol=0, kind="exact", reference=None):
        item = dict(check_id=name, kind=kind, scope=name, input_reference=reference,
                    independent_method="separate source/NumPy/SciPy reconstruction", expected_reference="independent oracle", observed_reference="study export/public actual",
                    compared_values=1, tolerance_kind="exact" if atol is None else {"atol": atol, "rtol": rtol},
                    maximum_absolute_difference=0., maximum_relative_difference=0., missing_mask_mismatches=0,
                    failure_examples=[])
        if atol is None:
            passed = actual == expected
            if passed:
                if isinstance(actual, (list, tuple, np.ndarray)):
                    item["compared_values"] = len(actual)
            else:
                a, b = safe(actual), safe(expected)
                passed = a == b
                if isinstance(a, list) and isinstance(b, list):
                    item["compared_values"] = len(b)
                    mismatches = [(i, av, bv) for i, (av, bv) in enumerate(zip(a, b)) if av != bv]
                    item["mismatch_count"] = len(mismatches) + abs(len(a)-len(b))
                    item["failure_examples"] = [{"index": i, "actual": av, "expected": bv} for i, av, bv in mismatches[:12]]
                else:
                    item["failure_examples"] = [{"actual": a, "expected": b}]
        else:
            a, b = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
            item["compared_values"] = int(b.size)
            self.numeric_elements += b.size
            passed = a.shape == b.shape
            if passed:
                finite = np.isfinite(a) & np.isfinite(b)
                mismatch = np.isnan(a) != np.isnan(b)
                valid = np.isclose(a, b, atol=atol, rtol=rtol, equal_nan=True)
                valid &= ~(np.isinf(a) | np.isinf(b))
                passed = bool(valid.all())
                d = np.abs(a[finite] - b[finite])
                item["maximum_absolute_difference"] = float(d.max(initial=0))
                item["maximum_relative_difference"] = float(np.max(d / np.maximum(np.abs(b[finite]), np.finfo(float).tiny), initial=0))
                item["missing_mask_mismatches"] = int(mismatch.sum())
                item["mismatch_count"] = int((~valid).sum())
                for ix in np.argwhere(~valid)[:12]:
                    ix = tuple(ix)
                    item["failure_examples"].append({"index": ix, "actual": safe(a[ix]), "expected": safe(b[ix])})
            else:
                item["failure_examples"] = [{"actual_shape": a.shape, "expected_shape": b.shape}]
        item.update(status="PASS" if passed else "FAIL", reason=None if passed else "contract mismatch")
        self.checks.append(item)
        self.kinds[kind] += 1
        if not passed:
            self.failures.append(item)


def source_replay(raw_path, native_map, canonical_map, targets):
    """Stream every map; retain only compact clocks, first labels and final events."""
    from scripts.export_prospective_features import iter_matches
    from src.utils.golgg_schema import team1_id, team2_id, best_of
    events, clocks, first, map_rows = [], [], {}, []
    source_ids, game_ids = set(), set()
    maps = 0
    with closing(iter_matches(raw_path)) as stream:
        for raw in stream:
            sid = ident(raw.get("match_id"))
            if sid is None or sid in source_ids:
                raise ValueError("missing/duplicate source series " + str(sid))
            source_ids.add(sid)
            games = raw.get("games")
            if not isinstance(games, list) or not games or any(not isinstance(g, dict) for g in games):
                raise ValueError("invalid recorded maps " + sid)
            teams = (ident(team1_id(raw)), ident(team2_id(raw)))
            if None in teams or teams[0] == teams[1]:
                raise ValueError("invalid immutable teams " + sid)
            ctx = context(raw)
            dates, rosters = [], []
            for index, game in enumerate(games, 1):
                gid = ident(game.get("game_id"))
                if gid is not None:
                    if gid in game_ids:
                        raise ValueError("duplicate game ID " + gid)
                    game_ids.add(gid)
                if game.get("match_id") is not None and ident(game["match_id"]) != sid:
                    raise ValueError("map/series identity mismatch " + sid)
                played = day(game.get("date"))
                dates.append(played)
                if len(dates) > 1 and dates[-1] < dates[-2]:
                    raise ValueError("map chronology contradiction " + sid)
                sides = (ident(game.get("t1_id")), ident(game.get("t2_id")))
                if set(sides) != set(teams):
                    raise ValueError("map immutable team contradiction " + sid)
                own = {}
                all_ids = []
                for key, team in zip(("t1", "t2"), sides):
                    players = game.get(key + "_players")
                    if not isinstance(players, dict) or len(players) > 5:
                        raise ValueError("invalid player slots " + sid)
                    ids = []
                    for player in players.values():
                        if not isinstance(player, dict):
                            raise ValueError("invalid player identity object " + sid)
                        ids.append(ident(player.get("player_id", player.get("id"))))
                    five(ids)
                    own[team] = ids + [None] * (5 - len(ids))
                    all_ids.extend(x for x in ids if x is not None)
                    map_rows.append(dict(source_match_id=sid, game_id=gid, map_index=index, team_id=team,
                                         competition_context=ctx, original_map_day=played.isoformat(),
                                         roster=sorted(x for x in own[team] if x is not None) + [None] * own[team].count(None)))
                if len(all_ids) != len(set(all_ids)):
                    raise ValueError("cross-side player identity corruption " + sid)
                rosters.append(own)
                maps += 1
            floor = max(dates)
            native_override = day(native_map[sid]) if sid in native_map else None
            native = max(floor, native_override or floor)
            header = day(raw.get("date"))
            override = day(canonical_map[sid]) if sid in canonical_map else None
            row_bound = day(targets[sid]["release_day"]) if sid in targets else None
            canonical_floor = max(header, floor, override or floor)
            if row_bound is not None and row_bound < canonical_floor:
                raise ValueError("canonical target release precedes source bound " + sid)
            canonical = max(canonical_floor, row_bound or canonical_floor)
            effective = max(native, canonical)
            clocks.append(dict(source_match_id=sid, native_floor=floor.isoformat(), native_override=None if native_override is None else native_override.isoformat(), native_release_day=native.isoformat(), canonical_header_day=header.isoformat(), canonical_maps_floor=floor.isoformat(), canonical_override=None if override is None else override.isoformat(), canonical_row_bound=None if row_bound is None else row_bound.isoformat(), canonical_release_day=canonical.isoformat(), effective_release_day=effective.isoformat(), disagreement=native != canonical, unknown_reason=None))
            wins = Counter()
            for game in games:
                flags = [game.get("t1_win"),game.get("t2_win")]
                if sid in targets:
                    if any(v not in (0,1,False,True) for v in flags) or flags[0] == flags[1]:
                        raise ValueError("canonical winner source lineage " + sid)
                    wins[ident(game["t1_id"] if flags[0] else game["t2_id"])] += 1
            first[sid] = dict(teams=teams, context=ctx, rosters={t: sorted(x for x in rosters[0][t] if x is not None)+[None]*rosters[0][t].count(None) for t in teams}, best_of=best_of(raw), native_release_day=native.isoformat(), canonical_release_day=canonical.isoformat(), forecast=min(header,min(dates)).isoformat(), wins=dict(wins))
            for team in teams:
                roster = sorted(x for x in rosters[-1][team] if x is not None) + [None] * rosters[-1][team].count(None)
                reason = "canonical_release_bound" if canonical > native else "override_day" if native_override is not None and native_override >= floor else "source_game_day_floor"
                events.append(dict(source_match_id=sid, team_id=team, competition_context=ctx, roster=roster, map_index=len(games), source_series_order=order(sid), original_map_day=dates[-1].isoformat(), native_release_day=native.isoformat(), canonical_release_day=canonical.isoformat(), effective_release_day=effective.isoformat(), release_source="corrected_release_map_override" if sid in native_map else "latest_source_game_day_fallback", effective_release_reason=reason))
    events.sort(key=lambda e: (e["effective_release_day"], order(e["source_match_id"]), e["team_id"]))
    by_id = {c["source_match_id"]:c for c in clocks}
    for row in map_rows:
        clock = by_id[row["source_match_id"]]
        row.update({k:clock[k] for k in ("native_release_day","canonical_release_day","effective_release_day")})
    return events, clocks, first, maps, map_rows


def candidate_replay(events, target_id, team, ctx, cutoff):
    prior = events.get((team, ctx), []) if ctx is not None else []
    keys = [day(e["effective_release_day"]) for e in prior]
    end = bisect.bisect_left(keys, cutoff)
    window = prior[max(0, end - 10):end]
    if any(e["source_match_id"] == target_id for e in prior[:end]):
        raise ValueError("target already released " + target_id)
    result = dict(history_ids=[e["source_match_id"] for e in window], incomplete_history_count=sum(five(e["roster"]) is None for e in window), candidates=[], absence=None)
    if ctx is None:
        result["absence"] = "MISSING_EXACT_CONTEXT"
    elif not window:
        result["absence"] = "NO_EXACT_CONTEXT_HISTORY"
    elif five(window[-1]["roster"]) is None:
        result["absence"] = "INCOMPLETE_LATEST_APPEARANCE"
    else:
        selected = []
        for e in reversed(window):
            roster = five(e["roster"])
            if roster is not None and roster not in selected:
                selected.append(roster)
                result["candidates"].append(dict(roster=roster, count=sum(five(x["roster"]) == roster for x in window), age_days=(cutoff - day(e["effective_release_day"])).days, last_source_match_id=e["source_match_id"], last_played_day=e["original_map_day"], last_release_day=e["effective_release_day"]))
                if len(selected) == 2:
                    break
    return result


def design(counts, ages, previous, conditional):
    n = len(counts)
    x = np.zeros((n, 3, 6 if conditional else 2))
    x[:, 0, 0] = 1
    if conditional:
        x[:, 0, 1:3] = np.log1p(np.column_stack((counts[:, 0], ages[:, 0])))
        x[:, 1, 3] = 1
        x[:, 1, 4:6] = np.log1p(np.column_stack((counts[:, 1], ages[:, 1])))
    else:
        x[:, 1, 1] = 1
    x[~previous, 1] = 0
    return x


def numerical_law(theta, x, previous, labels=None):
    z = np.einsum("ncp,p->nc", x, theta)
    z[~previous, 1] = -np.inf
    logq = z - logsumexp(z, axis=1, keepdims=True)
    q = np.exp(logq)
    if labels is None:
        return q
    if np.any((labels == 1) & ~previous):
        raise ValueError("structurally impossible previous label")
    nll = -logq[np.arange(len(labels)), labels]
    residual = q.copy()
    residual[np.arange(len(labels)), labels] -= 1
    gradient = np.einsum("nc,ncp->p", residual, x) / len(labels) + .2 * theta
    return q, float(nll.mean() + .1 * np.dot(theta, theta)), gradient


def month_replay(delta, labels, mask, universe=MONTHS, confidence=.995):
    delta, labels, mask = np.asarray(delta, float), np.asarray(labels, str), np.asarray(mask, bool)
    if not np.isfinite(delta[mask]).all():
        raise ValueError("nonfinite proper-score delta")
    counts = np.array([np.sum(mask & (labels == m)) for m in universe], dtype=np.int64)
    sums = np.array([np.sum(delta[mask & (labels == m)]) for m in universe])
    means = np.divide(sums, counts, out=np.full(len(universe), np.nan), where=counts > 0)
    indices = np.random.default_rng(20261009).integers(0, len(universe), size=(10000, len(universe)), dtype=np.int64)
    denominator = counts[indices].sum(axis=1)
    weighted = np.divide(sums[indices].sum(axis=1), denominator, out=np.full(10000, np.nan), where=denominator > 0)
    supported = counts[indices] > 0
    equal_denominator = supported.sum(axis=1)
    equal = np.divide(np.where(supported, means[indices], 0).sum(axis=1), equal_denominator, out=np.full(10000, np.nan), where=equal_denominator > 0)
    undefined = ~np.isfinite(weighted) | ~np.isfinite(equal)
    supported_blocks = int((counts > 0).sum())
    status = "NOT_ESTIMABLE" if undefined.any() else "INSUFFICIENT_SUPPORT" if mask.sum() < 200 or supported_blocks < 6 else "ESTIMABLE"
    tails = [(1-confidence)/2, (1+confidence)/2]
    draw_counts = np.zeros((10000,len(universe)),np.int64)
    for j in range(len(universe)):
        draw_counts[:,j] = (indices == j).sum(axis=1)
    return dict(indices=indices, draw_counts=draw_counts, counts=counts, sums=sums, means=means, denominator=denominator, equal_denominator=equal_denominator, weighted=weighted, equal=equal, undefined=undefined, status=status, population_n=int(mask.sum()), supported_blocks=supported_blocks, match_weighted_delta=float(sums.sum()/counts.sum()) if counts.sum() else None, equal_month_delta=float(means[counts > 0].mean()) if supported_blocks else None, match_weighted_bounds=np.quantile(weighted,tails,method="linear").tolist() if not undefined.any() else None, equal_month_bounds=np.quantile(equal,tails,method="linear").tolist() if not undefined.any() else None)


def partial(q_a, q_b, native):
    q_a, q_b = np.asarray(q_a, float), np.asarray(q_b, float)
    if any(q.shape != (3,) or not np.isfinite(q).all() or (q < 0).any() or abs(q.sum()-1) > 1e-12 for q in (q_a, q_b)):
        raise ValueError("invalid categorical law")
    weights = np.outer(q_a[:2], q_b[:2])
    lower = 0.
    for i, j in np.ndindex(2, 2):
        if weights[i, j] > 0:
            p = native[i, j]
            if not math.isfinite(p) or not 0 <= p <= 1:
                raise ValueError("missing/invalid positive-weight native pair")
            lower += weights[i, j] * p
    supported = float(weights.sum())
    joint = np.outer(q_a, q_b)
    u = float(sum(joint[i, j] for i, j in ((0,2),(1,2),(2,0),(2,1),(2,2))))
    scalar = q_a[2] == 0 and q_b[2] == 0
    return dict(lower=float(lower), upper=float(lower+u), supported_mass=supported, missing_mass=u, pair_weights=weights, prob_a=float(lower) if scalar else None, scalar_absence=None if scalar else "POSITIVE_UNSUPPORTED_ROSTER_MASS")


def table(path, columns=None):
    frame = pd.read_parquet(path,columns=columns)
    return [{k: table_value(v) for k, v in row.items()} for row in frame.to_dict("records")]


def table_value(value):
    if isinstance(value,(float,np.floating)) and math.isinf(value):
        raise ValueError("infinite value in a source/exported table")
    if isinstance(value,(list,tuple,np.ndarray)):
        return [table_value(v) for v in value]
    if isinstance(value,dict):
        return {k:table_value(v) for k,v in value.items()}
    return safe(value)




def compare_rows(audit, name, actual, expected, fields):
    audit.check(name + ".rows", len(actual), len(expected))
    for field in fields:
        audit.check(name + "." + field, [x.get(field) for x in actual], [x.get(field) for x in expected], reference=name)


def decode_snapshot(value):
    output = {"target": copy.deepcopy(value["target"]), "raw": {}, "hist": {}}
    shapes = {"raw": {"players": ((2,5,21), "float32"), "core": ((2,5), "float32"), "team": ((2,11), "float32"), "w20": ((2,10), "float32"), "gates": ((2,2), "float32")}, "hist": {"history": ((2,5,16,33), "float16"), "mask": ((2,5,16), "uint8"), "champions": ((2,5,16), "uint16")}}
    for family, definitions in shapes.items():
        if set(value[family]) != set(definitions):
            raise ValueError("snapshot keys mismatch")
        for key, (shape, dtype) in definitions.items():
            wire = value[family][key]
            if wire["shape"] != list(shape) or wire["dtype"] != dtype:
                raise ValueError("snapshot dtype/shape mismatch")
            data = base64.b64decode(wire["data_base64"], validate=True)
            array = np.frombuffer(data, dtype=np.dtype(dtype)).reshape(shape).copy()
            if not np.isfinite(array).all():
                raise ValueError("nonfinite snapshot")
            if key=="mask" and not np.isin(array,(0,1)).all():
                raise ValueError("snapshot mask must be binary")
            output[family][key] = array
    return output


def tensor_hashes(snapshot):
    return {family: {key: hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest() for key, array in values.items()} for family, values in snapshot.items() if family != "target"}


def swap_snapshot(snapshot):
    result = copy.deepcopy(snapshot)
    target = result["target"]
    target["team1_id"], target["team2_id"] = target["team2_id"], target["team1_id"]
    for a, b in (("roster_a", "roster_b"), ("tensor_player_order_a", "tensor_player_order_b"), ("raw_tensor_player_order_a", "raw_tensor_player_order_b"), ("temporal_tensor_player_order_a", "temporal_tensor_player_order_b")):
        target[a], target[b] = target[b], target[a]
    for family in ("raw", "hist"):
        result[family] = {k: v[::-1].copy() for k, v in result[family].items()}
    return result


def toy_checks(audit):
    q = [.5, .25, .25]
    result = partial(q, q, {(i,j): .75 for i,j in np.ndindex(2,2)})
    for key, expected in (("supported_mass", .5625), ("missing_mass", .4375), ("lower", .421875), ("upper", .859375)):
        audit.check("toy.binary_rational."+key, result[key], expected, kind="toy_exact_algebra")
    reverse = partial(q, q, {(i,j): .25 for i,j in np.ndindex(2,2)})
    audit.check("toy.binary_rational.complement", [reverse["lower"], reverse["upper"]], [1-result["upper"], 1-result["lower"]], kind="toy_exact_algebra")
    near = partial([1-2**-40,0,2**-40], [1,0,0], {(0,0): .75})
    audit.check("toy.no_small_mass_scalar_fallback", near["prob_a"], None, kind="toy_exact_algebra")
    tiny = partial([1-2**-60,0,2**-60], [1,0,0], {(0,0): .75})
    audit.check("toy.sub_ulp_other_mass",tiny["missing_mass"],2**-60,kind="toy_exact_algebra")
    audit.check("toy.sub_ulp_no_scalar",tiny["prob_a"],None,kind="toy_exact_algebra")


def opportunity_replay(canonical, metadata, events, first, clocks, audit):
    indexed = defaultdict(list)
    for event in events:
        indexed[(event["team_id"], event["competition_context"])].append(event)
    native_meta = {ident(row["golgg_match_id"]): row for row in metadata}
    if len(native_meta) != len(metadata):
        raise ValueError("duplicate native metadata")
    output, conflicts = [], []
    conflict_cache = {}
    contradictory = [c for c in clocks if c["native_release_day"] < c["canonical_release_day"]]
    for origin in sorted({str(r["date"]) for r in canonical}):
        conflict_cache[origin] = sorted((c["source_match_id"] for c in contradictory if c["native_release_day"] < origin <= c["canonical_release_day"]), key=order)
    for row in canonical:
        mid = ident(row["match_id"])
        origin = day(row["date"])
        label_release = day(row["release_day"])
        partition = "TRAIN" if origin < date(2025,7,1) and label_release < date(2025,7,1) else "CAL" if date(2025,7,1) <= origin < date(2026,1,1) and label_release < date(2026,1,1) else "ASSESS" if date(2026,1,1) <= origin < date(2027,1,1) else "EXCLUDED"
        source = first[mid]
        a, b = ident(row["team_a"]), ident(row["team_b"])
        if set(source["teams"]) != {a,b} or source["best_of"] != int(row["best_of"]) or source["forecast"] != origin.isoformat():
            raise ValueError("raw/canonical identity, format or origin binding " + mid)
        ctx = source["context"]
        canonical_context = context(row)
        if canonical_context is not None and canonical_context != ctx:
            raise ValueError("raw/canonical exact context mismatch " + mid)
        for team, field in ((a,"roster_a"),(b,"roster_b")):
            known = {ident(p) for p in row[field] if ident(p) is not None}
            actual = source["rosters"][team]
            raw_known = {ident(p) for p in actual if ident(p) is not None}
            if len(known | raw_known) > 5 or (five(row[field]) is not None and five(actual) is not None and five(row[field]) != five(actual)):
                raise ValueError("canonical first-map source identity mismatch " + mid)
        if partition == "ASSESS":
            meta = native_meta[mid]
            direct = (ident(meta["team1_id"]), ident(meta["team2_id"])) == (a,b)
            reverse = (ident(meta["team1_id"]), ident(meta["team2_id"])) == (b,a)
            if not direct ^ reverse or int(meta["best_of"]) != int(row["best_of"]):
                raise ValueError("native canonical ASSESS binding " + mid)
            winner = int(source["wins"].get(a,0) > source["wins"].get(b,0))
            if winner != (1-float(meta["y"]) if reverse else float(meta["y"])):
                raise ValueError("native/canonical/raw ASSESS winner binding " + mid)
            for team, field, canonical_field in ((a,"roster_b" if reverse else "roster_a","roster_a"),(b,"roster_a" if reverse else "roster_b","roster_b")):
                if five(meta[field]) != five(row[canonical_field]):
                    raise ValueError("native canonical first-map roster binding " + mid)
        if day(source["native_release_day"]) < origin or day(source["canonical_release_day"]) < origin:
            raise ValueError("target already released " + mid)
        conflict_ids = conflict_cache[origin.isoformat()]
        conflicts.append(dict(match_id=mid, cutoff_day=origin.isoformat(), conflict_source_ids=conflict_ids,
                              native_absence="NATIVE_HISTORY_RELEASE_CONFLICT" if conflict_ids else None))
        for side, team, opponent in (("A",a,b), ("B",b,a)):
            state = candidate_replay(indexed, mid, team, ctx, origin)
            actual = source["rosters"][team]
            candidates = state["candidates"]
            absence = state["absence"]
            boundary = date(2025,7,1) if partition == "TRAIN" else date(2026,1,1) if partition == "CAL" else date(2027,1,1)
            label_absence = "INCOMPLETE_FIRST_MAP_TARGET" if five(actual) is None else None
            if max(label_release,day(source["native_release_day"]),day(source["canonical_release_day"])) >= boundary:
                label_absence = "UNRELEASED_LABEL"
            if int(row["best_of"]) not in (1,3,5):
                label_absence = "UNSUPPORTED_TARGET_FORMAT"
            absence = absence or label_absence
            label = None
            if absence is None:
                label = next((i for i, candidate in enumerate(candidates) if actual == candidate["roster"]), 2)
            counts = [c["count"] for c in candidates] + [0] * (2-len(candidates))
            ages = [c["age_days"] for c in candidates] + [0] * (2-len(candidates))
            output.append(dict(match_id=mid, side=side, date=origin.isoformat(), month=origin.isoformat()[:7], partition=partition, best_of=int(row["best_of"]), team_id=team, opponent_id=opponent, competition_context=ctx, native_release_day=source["native_release_day"], label_release_day=label_release.isoformat(), actual_roster=actual, state=state, candidates=candidates, history_ids=state["history_ids"], incomplete_history_count=state["incomplete_history_count"], counts=counts, ages=ages, previous_available=len(candidates)==2, label=label, absence=absence))
            output[-1]["label_absence"] = label_absence
            output[-1]["history_state"] = "cold" if not state["history_ids"] else "full" if len(state["history_ids"])==10 else "partial"
            tier = taxonomy_tier(ctx,origin,audit.taxonomy_rules)
            output[-1]["major"] = "UNKNOWN" if tier=="unknown" else "YES" if tier=="major" else "NO"
            unknown = native_meta.get(mid,{}).get("unknown_players")
            if unknown is None and partition=="ASSESS":
                raise ValueError("missing frozen ASSESS source-unseen membership " + mid)
            output[-1]["player_source_unseen"] = "UNKNOWN" if unknown is None else "YES" if float(unknown)>0 else "NO"
    return output, conflicts


def support_replay(rows):
    counts = {}
    for partition, prefix in (("TRAIN","train"), ("CAL","cal"), ("ASSESS","assessment")):
        selected = [r for r in rows if r["partition"] == partition and r["label"] is not None]
        counts[prefix+"_side_labels"] = len(selected)
        counts[prefix+"_series"] = len({r["match_id"] for r in selected})
        counts[prefix+"_months"] = len({r["month"] for r in selected})
        for i, cls in enumerate(("latest","previous","other")):
            counts[prefix+"_"+cls+"_labels"] = sum(r["label"] == i for r in selected)
    floors = dict(train_side_labels=1000, train_previous_labels=100, train_other_labels=100, assessment_series=200, assessment_months=6, cal_side_labels=3, cal_latest_labels=1, cal_previous_labels=1, cal_other_labels=1)
    failed = [{"field":k, "required":v, "observed":counts[k], "reason":"BELOW_FLOOR"} for k,v in floors.items() if counts[k] < v]
    return counts, failed


def saved_fit_replay(audit, model, rows, conditional, name, require_acceptance=True):
    train = [r for r in rows if r["partition"] == "TRAIN" and r["label"] is not None]
    theta = np.asarray(model["theta"], float)
    audit.check(name+".theta_shape", theta.shape, (6 if conditional else 2,))
    if not np.isfinite(theta).all() or np.any(np.abs(theta)>5) or not train:
        raise ValueError("invalid saved theta/TRAIN membership")
    counts = np.array([r["counts"] for r in train], float)
    ages = np.array([r["ages"] for r in train], float)
    previous = np.array([r["previous_available"] for r in train], bool)
    labels = np.array([r["label"] for r in train], int)
    x = design(counts, ages, previous, conditional)
    _, objective, gradient = numerical_law(theta, x, previous, labels)
    receipt = model.get("receipt", model)
    audit.check(name+".objective", receipt["objective"], objective, atol=1e-8, kind="saved_fit")
    audit.check(name+".gradient", receipt["gradient"], gradient, atol=1e-10, rtol=1e-10, kind="saved_fit")
    projected = gradient.copy()
    projected[(theta <= -5) & (gradient > 0)] = 0
    projected[(theta >= 5) & (gradient < 0)] = 0
    audit.check(name+".projected_kkt", receipt["projected_gradient_infinity"], np.max(np.abs(projected)), atol=1e-10, rtol=1e-10, kind="saved_fit")
    accepted = bool(receipt["success"]) and int(receipt["iterations"]) <= 500 and np.max(np.abs(projected)) <= 1e-5
    audit.check(name+".acceptance_receipt",receipt["accepted"],bool(accepted))
    if require_acceptance:
        audit.check(name+".kkt_acceptance",bool(np.max(np.abs(projected)) <= 1e-5),True)
    audit.check(name+".bounds", receipt.get("bounds"), [[-5.,5.]]*len(theta))
    audit.check(name+".zero_start", receipt.get("initial_theta"), [0.]*len(theta))
    audit.check(name+".ridge", receipt.get("ridge"), .1)
    audit.check(name+".conditional_flag",receipt["conditional"],conditional)
    audit.check(name+".theta_order",receipt["theta_order"],["a1","b1","d1","a2","b2","d2"] if conditional else ["a1","a2"])
    for field,value in (("optimizer","L-BFGS-B"),("maxiter",500),("ftol",1e-12),("gtol",1e-8)):
        audit.check(name+".fixed_recipe."+field,receipt[field],value)
    audit.check(name+".boundary_flags",receipt["boundary_hit"],((theta<=-5)|(theta>=5)).tolist())
    audit.check(name+".iteration_limit", int(receipt["iterations"]) <= 500, True)
    if require_acceptance:
        audit.check(name+".optimizer_success",bool(receipt["success"]),True)
    audit.check(name+".membership", model["training_keys"], [[r["match_id"],r["side"]] for r in train])
    membership = copy.deepcopy(model["training_membership"])
    for member in membership:
        if isinstance(member["release_lineage"],str):
            member["release_lineage"] = json.loads(member["release_lineage"])
    expected_membership = [{"match_id":r["match_id"],"side":r["side"],"label":("LATEST","PREVIOUS","OTHER")[r["label"]],"release_lineage":r["release_lineage"],"history_ids":r["history_ids"]} for r in train]
    compare_rows(audit,name+".full_training_lineage",membership,expected_membership,("match_id","side","label","release_lineage","history_ids"))
    fixtures = [(theta,"saved"), (np.zeros_like(theta),"zero"), (np.linspace(-.3,.4,len(theta)),"interior")]
    for point, tag in fixtures:
        _, value, derivative = numerical_law(point, x, previous, labels)
        numeric = []
        for j in range(len(theta)):
            h = 1e-5*max(1., abs(point[j]))
            plus, minus = point.copy(), point.copy()
            if point[j] >= 5:
                minus[j] -= h
                twice = point.copy()
                twice[j] -= 2*h
                numeric.append((3*value-4*numerical_law(minus,x,previous,labels)[1]+numerical_law(twice,x,previous,labels)[1])/(2*h))
            elif point[j] <= -5:
                plus[j] += h
                twice = point.copy()
                twice[j] += 2*h
                numeric.append((-3*value+4*numerical_law(plus,x,previous,labels)[1]-numerical_law(twice,x,previous,labels)[1])/(2*h))
            else:
                plus[j] += h
                minus[j] -= h
                numeric.append((numerical_law(plus,x,previous,labels)[1]-numerical_law(minus,x,previous,labels)[1])/(2*h))
        audit.check(name+".finite_difference."+tag, derivative, numeric, atol=2e-6, rtol=2e-5, kind="gradient_diagnostic")
        audit.checks[-1]["independent_method"] = "central finite differences at smooth interior; second-order feasible one-sided at declared coefficient bounds; h_j=1e-5 max(1,abs(theta_j))"
    return theta


def laws_and_scores(audit, rows, models, exported_laws, exported_scores):
    law_lookup, series, expected_laws = {}, defaultdict(lambda: defaultdict(list)), []
    for arm, theta in models.items():
        available = [r for r in rows if r["state"]["absence"] is None]
        counts = np.array([r["counts"] for r in available], float)
        ages = np.array([r["ages"] for r in available], float)
        previous = np.array([r["previous_available"] for r in available], bool)
        q = numerical_law(theta, design(counts,ages,previous,arm=="conditional"), previous)
        values = {(r["match_id"],r["side"]):law for r,law in zip(available,q)}
        for r in rows:
            law = values.get((r["match_id"],r["side"]))
            label = r["label"]
            nll = -math.log(law[label]) if law is not None and label is not None else None
            brier = (law[2]-(label==2))**2 if law is not None and label is not None else None
            if law is not None:
                law_lookup[(r["match_id"],r["side"],arm)] = law
            expected_laws.append(dict(match_id=r["match_id"],side=r["side"],law=arm,partition=r["partition"],label=label,absence=r["absence"],q_latest=None if law is None else law[0],q_previous=None if law is None else law[1],q_other=None if law is None else law[2],lineup_nll=nll,other_brier=brier))
            if nll is not None:
                series[r["match_id"]][arm].append((nll,brier))
    actual = {(r["match_id"],r["side"],r["law"]):r for r in exported_laws}
    keys = [(r["match_id"],r["side"],r["law"]) for r in expected_laws]
    audit.check("laws.exact_all_opportunity_keys",sorted(actual),sorted(keys))
    audit.check("laws.no_duplicate_keys",len(actual),len(exported_laws))
    ordered = [actual.get(key,{}) for key in keys]
    compare_rows(audit,"laws",ordered,expected_laws,("match_id","side","law","partition","label","absence"))
    for field in ("q_latest","q_previous","q_other","lineup_nll","other_brier"):
        tolerance = 1e-12 if field.startswith("q_") else 1e-10
        audit.check("laws."+field,[r.get(field) for r in ordered],[r[field] for r in expected_laws],atol=tolerance,rtol=tolerance,kind="raw_law_replay")
    zero = [i for i,r in enumerate(expected_laws) if r["q_previous"] == 0]
    audit.check("laws.structural_zero",[ordered[i].get("q_previous") for i in zero],[0.]*len(zero))
    for field in ("lineup_nll","other_brier"):
        expected_metric = []
        for row in ordered:
            label = row.get("label")
            if label is None:
                expected_metric.append(None)
            else:
                q = np.asarray([row["q_latest"],row["q_previous"],row["q_other"]],float)
                if not np.isfinite(q).all() or q[label] <= 0:
                    raise ValueError("invalid observed saved categorical law")
                expected_metric.append(-math.log(q[label]) if field=="lineup_nll" else (q[2]-(label==2))**2)
        audit.check("identical_exported_law_metric_layer."+field,[r.get(field) for r in ordered],expected_metric,atol=1e-10,rtol=1e-10,kind="exported_law_metric_layer")
    del actual, ordered, expected_laws
    import gc; gc.collect()
    if exported_scores is None:
        return law_lookup,[]
    actual_scores = {r["match_id"]:r for r in exported_scores}
    expected_scores = []
    for mid in dict.fromkeys(r["match_id"] for r in rows):
        row = dict(match_id=mid,eligible_side_count=len(series[mid].get("conditional",[])))
        for arm in models:
            scores = series[mid].get(arm,[])
            row[arm+"_nll"] = float(np.mean([x[0] for x in scores])) if scores else None
            row[arm+"_brier"] = float(np.mean([x[1] for x in scores])) if scores else None
        expected_scores.append(row)
    ordered_scores = [actual_scores.get(r["match_id"],{}) for r in expected_scores]
    compare_rows(audit,"series_scores",ordered_scores,expected_scores,("match_id","eligible_side_count"))
    for field in ("conditional_nll","global_nll","conditional_brier","global_brier"):
        audit.check("series_scores."+field,[r.get(field) for r in ordered_scores],[r[field] for r in expected_scores],atol=1e-10,rtol=1e-10,kind="side_to_series")
    audit.check("series.all_parent_keys",sorted(actual_scores),sorted(r["match_id"] for r in expected_scores))
    return law_lookup,expected_scores


def uncertainty_replay(audit, rows, scores, archive, metrics):
    assess_ids = list(dict.fromkeys(r["match_id"] for r in rows if r["partition"] == "ASSESS"))
    meta = {r["match_id"]:r for r in rows if r["partition"] == "ASSESS"}
    lookup = {r["match_id"]:r for r in scores}
    months = np.array([meta[mid]["month"] for mid in assess_ids])
    eligible = np.array([lookup[mid]["eligible_side_count"] > 0 for mid in assess_ids])
    audit.check("bootstrap.ids",archive["match_ids"].astype(str).tolist(),assess_ids)
    audit.check("bootstrap.months",archive["months"].astype(str).tolist(),months.tolist())
    audit.check("bootstrap.eligible",archive["eligible"].tolist(),eligible.tolist())
    primary = {}
    proxy_labels = np.array([meta[mid]["date"][:4]+":"+(meta[mid]["competition_context"] or "UNKNOWN") for mid in assess_ids])
    proxy_universe = sorted(set(proxy_labels.tolist()))
    audit.check("proxy.blocks",archive["proxy_universe"].astype(str).tolist(),proxy_universe)
    audit.check("proxy.membership",archive["proxy_blocks"].astype(str).tolist(),proxy_labels.tolist())
    arrays = {"indices":"draw_indices","draw_counts":"draw_counts","counts":"block_counts","sums":"block_sums","means":"block_means","denominator":"weighted_denominator","equal_denominator":"equal_denominator","weighted":"weighted_delta","equal":"equal_delta","undefined":"undefined"}
    for metric,slot in (("nll",7),("brier",8)):
        candidate = np.array([lookup[mid]["conditional_"+metric] if eligible[i] else np.nan for i,mid in enumerate(assess_ids)])
        control = np.array([lookup[mid]["global_"+metric] if eligible[i] else np.nan for i,mid in enumerate(assess_ids)])
        delta = candidate-control
        for key,value in (("conditional_"+metric,candidate),("global_"+metric,control),("delta_"+metric,delta)):
            audit.check("bootstrap."+key,archive[key],value,atol=1e-10,rtol=1e-10,kind="saved_law_scores")
        expected = None
        for name,labels,universe,confidence,reports in (("month",months,MONTHS,.995,metrics["primary"]),("proxy",proxy_labels,proxy_universe,.95,metrics["source_name_year_proxy"])):
            replay = month_replay(delta,labels,eligible,universe=universe,confidence=confidence)
            for source,target in arrays.items():
                key = name+"_"+metric+"_"+target
                if source in ("indices","draw_counts","counts","denominator","equal_denominator","undefined"):
                    audit.check("bootstrap."+key,archive[key].tolist(),replay[source].tolist(),kind="bootstrap_exact")
                else:
                    audit.check("bootstrap."+key,archive[key],replay[source],atol=1e-10,rtol=1e-10,kind="bootstrap_numerical")
            for target in ("weighted_undefined","equal_undefined"):
                audit.check("bootstrap."+name+"_"+metric+"_"+target,archive[name+"_"+metric+"_"+target].tolist(),replay["undefined"].tolist(),kind="bootstrap_exact")
            report_contrast(audit,name+"."+metric,reports[metric],replay,confidence)
            if name == "month":
                expected = replay
        passing = expected["status"]=="ESTIMABLE" and expected["match_weighted_bounds"][1]<0 and expected["equal_month_delta"]<0
        primary[slot] = {"status":expected["status"],"passes_scientific_gate":passing}
        groups = {"BO1":np.array([meta[mid]["best_of"]==1 for mid in assess_ids]),"BO3":np.array([meta[mid]["best_of"]==3 for mid in assess_ids]),"BO5":np.array([meta[mid]["best_of"]==5 for mid in assess_ids]),"unilateral":np.array([lookup[mid]["eligible_side_count"]==1 for mid in assess_ids]),"bilateral":np.array([lookup[mid]["eligible_side_count"]==2 for mid in assess_ids])}
        for field,names in (("major",("major","nonmajor","major_UNKNOWN")),("player_source_unseen",("source_unseen","source_seen","source_UNKNOWN"))):
            for value,name in zip(("YES","NO","UNKNOWN"),names):
                groups[name] = np.array([meta[mid][field]==value for mid in assess_ids])
        side_lookup = {(r["match_id"],r["side"]):r for r in rows}
        for history in ("full","partial","cold"):
            groups["history_any_"+history] = np.array([any(("cold" if not side_lookup[mid,s]["history_ids"] else "full" if len(side_lookup[mid,s]["history_ids"])==10 else "partial")==history for s in ("A","B")) for mid in assess_ids])
        for pattern in ((0,0),(0,1),(0,2),(1,1),(1,2),(2,2)):
            groups["candidates_"+"_".join(map(str,pattern))] = np.array([tuple(sorted(len(side_lookup[mid,s]["candidates"]) for s in ("A","B")))==pattern for mid in assess_ids])
        groups["incomplete_history"] = np.array([any(side_lookup[mid,s]["incomplete_history_count"]>0 for s in ("A","B")) for mid in assess_ids])
        audit.check("subgroups.frozen_names",sorted(metrics["subgroups"]),sorted(groups))
        for name,group in groups.items():
            audit.check("subgroup."+name+".membership",archive["membership_"+name].tolist(),group.tolist())
            replay = month_replay(delta,months,eligible & group)
            report_contrast(audit,"subgroup."+name+"."+metric,metrics["subgroups"][name][metric],replay,.995)
    return primary


def report_contrast(audit,name,actual,replay,confidence):
    for key,expected in (("status",replay["status"]),("n_rows",replay["population_n"]),("n_blocks",replay["supported_blocks"]),("undefined_draws",int(replay["undefined"].sum())),("draws",10000),("seed",20261009),("confidence",confidence)):
        audit.check(name+"."+key,actual[key],expected,kind="contrast_report")
    audit.check(name+".point",actual["point"],replay["match_weighted_delta"],atol=None if replay["match_weighted_delta"] is None else 1e-10,rtol=1e-10,kind="contrast_report")
    audit.check(name+".equal.point",actual["equal_block"]["point"],replay["equal_month_delta"],atol=None if replay["equal_month_delta"] is None else 1e-10,rtol=1e-10,kind="contrast_report")
    for field,values in (("weighted",replay["match_weighted_bounds"]),("equal",replay["equal_month_bounds"])):
        observed = actual if field=="weighted" else actual["equal_block"]
        if values is None:
            audit.check(name+"."+field+".bounds",[observed["lower"],observed["upper"]],[None,None])
        else:
            audit.check(name+"."+field+".bounds",[observed["lower"],observed["upper"]],values,atol=1e-10,rtol=1e-10,kind="contrast_report")


def native_replay(audit, rows, law_lookup, pair_rows, bound_rows, repo, selected_ids, clocks, pins, blocked_rows, research, *, complete=True, inherited_pairs=()):
    from betting_app.services.c0_inference import C0InferenceClient
    opportunities = {(r["match_id"],r["side"]):r for r in rows if r["partition"]=="ASSESS"}
    pairs, captures = defaultdict(dict), {}
    origins = sorted({r["date"] for r in rows if r["partition"]=="ASSESS"})
    conflict_by_origin = {origin:sorted((c["source_match_id"] for c in clocks if c["native_release_day"]<origin<=c["canonical_release_day"]),key=order) for origin in origins}
    native_days = sorted(c["native_release_day"] for c in clocks)
    expected_pair_keys,expected_blocks = set(),{}
    for a in (r for r in rows if r["partition"]=="ASSESS" and r["side"]=="A"):
        mid = a["match_id"]
        b = opportunities[mid,"B"]
        if a["state"]["absence"] or b["state"]["absence"]:
            continue
        for i,ca in enumerate(a["candidates"]):
            for j,cb in enumerate(b["candidates"]):
                key = (mid,i,j)
                if conflict_by_origin[a["date"]]:
                    expected_blocks[key] = "NATIVE_HISTORY_RELEASE_CONFLICT"
                elif set(ca["roster"]) & set(cb["roster"]):
                    expected_blocks[key] = "CROSS_PAIR_PLAYER_OVERLAP"
                else:
                    expected_pair_keys.add(key)
    actual_blocks = {(r["match_id"],r["candidate_a_index"],r["candidate_b_index"]):r for r in blocked_rows}
    actual_attempts = {(r["match_id"],r["candidate_a_index"],r["candidate_b_index"]) for r in pair_rows}
    audit.check("native.no_duplicate_attempts",len(actual_attempts),len(pair_rows))
    audit.check("native.no_duplicate_blocks",len(actual_blocks),len(blocked_rows))
    deferred = {key for key,r in actual_blocks.items() if r["reason"]=="NOT_REQUESTED_AFTER_NATIVE_PAIR_FAILURE"}
    failed_targets = {r["match_id"] for r in pair_rows if r.get("failure_reason") is not None}
    for key in deferred:
        audit.check("native.deferred."+str(key),key[0] in failed_targets,True)
    audit.check("native.exact_attempted_or_deferred_pairs" if complete else "native.partial_legal_pairs",
                sorted(actual_attempts|deferred) if complete else sorted((actual_attempts|deferred)-expected_pair_keys),
                sorted(expected_pair_keys) if complete else [])
    audit.check("native.exact_blocked_pair_population" if complete else "native.partial_legal_blocks",
                sorted(set(actual_blocks)-deferred) if complete else sorted(set(actual_blocks)-deferred-set(expected_blocks)),
                sorted(expected_blocks) if complete else [])
    for key,reason in expected_blocks.items():
        if not complete and key not in actual_blocks:
            continue
        observed = actual_blocks.get(key,{})
        audit.check("native.blocked."+str(key)+".reason",observed.get("reason"),reason)
        audit.check("native.blocked."+str(key)+".uncalled",observed.get("native_requested"),False)
        audit.check("native.blocked."+str(key)+".conflicts",observed.get("conflict_source_ids"),conflict_by_origin[opportunities[key[0],"A"]["date"]] if reason=="NATIVE_HISTORY_RELEASE_CONFLICT" else [])
    public_count = 0
    unique_targets = set()
    for row in pair_rows:
        mid = row["match_id"]
        i,j = int(row["candidate_a_index"]), int(row["candidate_b_index"])
        if (i,j) in pairs[mid] or i not in (0,1) or j not in (0,1):
            raise ValueError("duplicate/illegal native pair")
        a,b = opportunities[mid,"A"], opportunities[mid,"B"]
        if a["partition"] != "ASSESS" or row.get("failure_reason") is not None:
            if row.get("failure_reason") is not None:
                if complete and row not in inherited_pairs:
                    audit.check("native.failure.authorized",row.get("failure_kind"),"AVAILABILITY")
                    audit.check("native.failure.timeout",row.get("failure_type"),"C0WorkerTimeout")
                continue
            raise ValueError("native pre2026 pair")
        capture_path = repo / row["evidence_path"]
        audit.check("native."+mid+f".{i}.{j}.capture_sha256",sha(capture_path),row["evidence_sha256"],kind="artifact_hash")
        capture = load(capture_path)
        wire = capture["encoded_snapshot"]
        snapshot = decode_snapshot(wire)
        target = snapshot["target"]
        for key, expected in (("team1_id",a["team_id"]),("team2_id",b["team_id"]),("competition_context",a["competition_context"]),("cutoff_day",a["date"]),("best_of",a["best_of"]),("roster_a",a["candidates"][i]["roster"]),("roster_b",b["candidates"][j]["roster"])):
            audit.check("native."+mid+f".{i}.{j}."+key,target.get(key),expected,kind="native_identity")
        for side,op,index in (("a",a,i),("b",b,j)):
            for prefix in ("tensor_player_order_","raw_tensor_player_order_","temporal_tensor_player_order_"):
                audit.check("native."+mid+f".{i}.{j}."+prefix+side,target.get(prefix+side),op["candidates"][index]["roster"],kind="native_identity")
        hashes = tensor_hashes(snapshot)
        flat = {k:v for family in hashes.values() for k,v in family.items()}
        audit.check("native."+mid+f".{i}.{j}.digests",capture["prediction"]["diagnostics"]["input_hashes"],flat,kind="native_input_digest")
        audit.check("native."+mid+f".{i}.{j}.row_digests",json.loads(row["input_hashes"]) if isinstance(row["input_hashes"],str) else row["input_hashes"],flat,kind="native_input_digest")
        original = capture["native_history_provenance"]
        audit.check("native."+mid+f".{i}.{j}.original_provenance",original,capture["prediction"]["diagnostics"]["feature_provenance"])
        audit.check("native."+mid+f".{i}.{j}.source_hash",original["raw_history_sha256"],pins["raw_source"]["sha256"],kind="native_source_pin")
        audit.check("native."+mid+f".{i}.{j}.release_hash",original["release_map_sha256"],pins["native_release_map"]["sha256"],kind="native_source_pin")
        audit.check("native."+mid+f".{i}.{j}.consumed_count",original["history_series_consumed"],bisect.bisect_left(native_days,a["date"]))
        audit.check("native."+mid+f".{i}.{j}.complete_source_count",original["history_series_total"],len(clocks))
        actual_clocks = [datetime.fromisoformat(v) for v in (row["requested_at"],capture["generated_at"],capture["received_at"])]
        audit.check("native."+mid+f".{i}.{j}.actual_execution_clocks",all(c.tzinfo is not None for c in actual_clocks) and all(a<=b for a,b in zip(actual_clocks,actual_clocks[1:])),True)
        audit.check("native."+mid+f".{i}.{j}.series_probability_complement",row["prob_b"],1-row["prob_a"],atol=1e-12,rtol=1e-12,kind="native_capture")
        audit.check("native."+mid+f".{i}.{j}.no_known_unreleased_native_history",conflict_by_origin[a["date"]],[])
        cutoff = day(a["date"])
        if target.get("feature_source_max_day") is not None and day(target["feature_source_max_day"]) >= cutoff:
            raise ValueError("native feature boundary violation")
        audit.check("native."+mid+f".{i}.{j}.mode",capture["prediction"]["diagnostics"]["mode"],"full")
        audit.check("native."+mid+f".{i}.{j}.probability_target",capture["prediction"]["diagnostics"]["probability_target"],"series_win_already_projected")
        audit.check("native."+mid+f".{i}.{j}.identity",capture["prediction"]["model_version"],"c0-native-2026-w32-e12-v1")
        audit.check("native."+mid+f".{i}.{j}.feature_identity",capture["prediction"]["feature_version"],"native-c0-history16-v1")
        audit.check("native."+mid+f".{i}.{j}.prob_a",row["prob_a"],capture["prediction"]["prob_a"],atol=0,kind="native_capture")
        pairs[mid][i,j] = float(row["prob_a"])
        if mid in selected_ids:
            captures[mid,i,j] = (capture_path, float(row["prob_a"]))
        unique_targets.add(mid)
        del capture, wire, snapshot
    for mid in unique_targets:
        audit.check("native."+mid+".pair_ceiling",len(pairs[mid]) <= 4,True)
    import gc; gc.collect()
    original_environment = os.environ.get("C0_RESEARCH_ROOT")
    os.environ["C0_RESEARCH_ROOT"] = str(research)
    client = None
    actual_forward,actual_reverse = defaultdict(dict),defaultdict(dict)
    try:
        client = C0InferenceClient(config_path=repo/"conf/base/c0_serving.json") if captures else None
        for (mid,i,j),(path,p) in sorted(captures.items(),key=lambda x:(opportunities[x[0][0],"A"]["date"],x[0])):
            snapshot = decode_snapshot(load(path)["encoded_snapshot"])
            before = tensor_hashes(snapshot)
            prediction = client.predict_snapshot(snapshot,mode="full")
            reverse = swap_snapshot(snapshot)
            reverse_before = tensor_hashes(reverse)
            swapped = client.predict_snapshot(reverse,mode="full")
            public_count += 2
            actual_forward[mid][i,j] = prediction
            actual_reverse[mid][j,i] = swapped
            audit.check("public_native."+mid+f".{i}.{j}.parity",prediction.prob_a,p,atol=2e-6,kind="public_native_actual")
            audit.check("public_native."+mid+f".{i}.{j}.swap",swapped.prob_a,1-prediction.prob_a,atol=2e-6,kind="public_native_actual")
            audit.check("public_native."+mid+f".{i}.{j}.nonmutation",tensor_hashes(snapshot),before)
            audit.check("public_native."+mid+f".{i}.{j}.swap_nonmutation",tensor_hashes(reverse),reverse_before)
        from src.analysis.lineup_law_reliability import supported_roster_bounds
        for mid,matrix in actual_forward.items():
            for arm in ("conditional","global"):
                if (mid,"A",arm) not in law_lookup or (mid,"B",arm) not in law_lookup:
                    continue
                qa,qb = law_lookup[mid,"A",arm],law_lookup[mid,"B",arm]
                needed = {(i,j) for i,j in np.ndindex(2,2) if qa[i]*qb[j]>0}
                if set(matrix) != needed:
                    continue
                before = (qa.copy(),qb.copy())
                forward = supported_roster_bounds(qa,qb,matrix,best_of=opportunities[mid,"A"]["best_of"])
                reversed_result = supported_roster_bounds(qb,qa,actual_reverse[mid],best_of=opportunities[mid,"A"]["best_of"])
                expected = partial(qa,qb,pairs[mid])
                audit.check("public_native_bounds."+mid+"."+arm+".captured_expected",[forward["lower"],forward["upper"],forward["missing_mass"]],[expected["lower"],expected["upper"],expected["missing_mass"]],atol=2e-6,kind="public_native_actual")
                audit.check("public_native_bounds."+mid+"."+arm+".complement",[reversed_result["lower"],reversed_result["upper"],reversed_result["missing_mass"]],[1-forward["upper"],1-forward["lower"],forward["missing_mass"]],atol=2e-6,kind="public_native_actual")
                audit.check("public_native_bounds."+mid+"."+arm+".nonmutation",[qa.tolist(),qb.tolist()],[x.tolist() for x in before])
    finally:
        if client is not None:
            client.close()
        if original_environment is None:
            os.environ.pop("C0_RESEARCH_ROOT",None)
        else:
            os.environ["C0_RESEARCH_ROOT"] = original_environment
    seen = set()
    audit.bound_expectations = []
    for row in bound_rows:
        mid,arm = row["match_id"],row["law"]
        if (mid,arm) in seen:
            raise ValueError("duplicate bounds")
        seen.add((mid,arm))
        a,b = opportunities[mid,"A"],opportunities[mid,"B"]
        reason = a["state"]["absence"] or b["state"]["absence"]
        if reason is None and conflict_by_origin[a["date"]]:
            reason = "NATIVE_HISTORY_RELEASE_CONFLICT"
        if reason is None and any(key[0]==mid and value=="CROSS_PAIR_PLAYER_OVERLAP" for key,value in expected_blocks.items()):
            reason = "CROSS_PAIR_PLAYER_OVERLAP"
        if mid in failed_targets:
            reason = "INHERITED_NATIVE_FAILURE_NO_RETRY" if any(r["match_id"]==mid for r in inherited_pairs) else "NATIVE_PAIR_UNAVAILABLE"
        audit.check("bounds."+mid+"."+arm+".absence",row.get("absence"),reason)
        expected_row = dict(match_id=mid,law=arm,date=a["date"],month=a["month"],best_of=a["best_of"],competition_context=a["competition_context"],major=a["major"],player_source_unseen=a["player_source_unseen"],candidate_pattern="/".join(str(len(r["candidates"])) for r in (a,b)),history_pattern="/".join("cold" if not r["history_ids"] else "full" if len(r["history_ids"])==10 else "partial" for r in (a,b)),incomplete_pattern="/".join(str(r["incomplete_history_count"]) for r in (a,b)),absence=reason)
        for field in ("month","best_of","competition_context","major","player_source_unseen","candidate_pattern","history_pattern","incomplete_pattern","provisional_independence","algebraic_not_confidence_interval"):
            audit.check("bounds."+mid+"."+arm+".membership."+field,row[field],True if field in ("provisional_independence","algebraic_not_confidence_interval") else expected_row[field])
        if reason is not None:
            expected_row.update(lower=None,upper=None,missing_mass=None,supported_mass=None,prob_a=None,scalar_absence="NATIVE_PAIR_UNAVAILABLE")
            audit.bound_expectations.append(expected_row)
            for field in ("lower","upper","missing_mass","supported_mass","prob_a"):
                audit.check("bounds."+mid+"."+arm+".absent."+field,row[field],None)
            continue
        qa,qb = (np.array([1.,0.,0.]),np.array([1.,0.,0.])) if arm=="latest_point" else (law_lookup[mid,"A",arm],law_lookup[mid,"B",arm])
        expected = partial(qa,qb,pairs[mid])
        expected_row.update(expected)
        audit.bound_expectations.append(expected_row)
        reverse = partial(qb,qa,{(j,i):1-p for (i,j),p in pairs[mid].items()})
        for key in ("lower","upper","supported_mass","missing_mass"):
            audit.check("bounds."+mid+"."+arm+"."+key,row[key],expected[key],atol=1e-12,rtol=1e-12,kind="full_real_partial_mass")
        weights = json.loads(row["pair_weights"]) if isinstance(row["pair_weights"],str) else row["pair_weights"]
        expected_weights = {f"{i},{j}":float(expected["pair_weights"][i,j]) for i,j in np.ndindex(2,2)}
        audit.check("bounds."+mid+"."+arm+".unconditional_weights",sorted(weights),sorted(expected_weights))
        audit.check("bounds."+mid+"."+arm+".weight_values",[weights[k] for k in sorted(expected_weights)],[expected_weights[k] for k in sorted(expected_weights)],atol=1e-12,rtol=1e-12,kind="full_real_partial_mass")
        audit.check("bounds."+mid+"."+arm+".scalar",row["prob_a"],expected["prob_a"])
        audit.check("bounds."+mid+"."+arm+".scalar_reason",row["scalar_absence"],expected["scalar_absence"])
        audit.check("bounds."+mid+"."+arm+".algebraic_complement",[reverse["lower"],reverse["upper"],reverse["missing_mass"]],[1-expected["upper"],1-expected["lower"],expected["missing_mass"]],atol=1e-12,rtol=1e-12,kind="full_real_partial_mass")
    expected_keys = {(r["match_id"],arm) for r in rows if r["partition"]=="ASSESS" for arm in ("conditional","global","latest_point")}
    audit.check("bounds.complete_arm_pair_population" if complete else "bounds.partial_legal_population",sorted(seen) if complete else sorted(seen-expected_keys),sorted(expected_keys) if complete else [])
    return {"unique_native_pairs":sum(map(len,pairs.values())),"unique_native_targets":len(unique_targets),"native_public_calls":public_count,"unique_native_public_pairs":len(captures)}


def public_contract_replay(audit, rows, models, selected_ids):
    from src.models.lineup_law_reliability import lineup_probabilities
    from src.analysis.lineup_law_reliability import supported_roster_bounds
    from betting_app.core.models.contract import UnifiedPredictionResult
    eligible = [r for r in rows if r["state"]["absence"] is None]
    counts = np.array([r["counts"] for r in eligible],float)
    ages = np.array([r["ages"] for r in eligible],float)
    previous = np.array([r["previous_available"] for r in eligible],bool)
    for arm,theta in models.items():
        expected = numerical_law(theta,design(counts,ages,previous,arm=="conditional"),previous)
        actual = lineup_probabilities(theta,counts,ages,previous,conditional=arm=="conditional")
        audit.check("public_law."+arm+".full_population",actual,expected,atol=1e-12,rtol=1e-12,kind="public_law_actual")
        for i,row in enumerate(eligible):
            if row["match_id"] in selected_ids:
                before = (theta.copy(),counts[i:i+1].copy(),ages[i:i+1].copy(),previous[i:i+1].copy())
                single = lineup_probabilities(theta,counts[i:i+1],ages[i:i+1],previous[i:i+1],conditional=arm=="conditional")
                audit.check("public_law."+arm+"."+row["match_id"]+row["side"],single,expected[i:i+1],atol=1e-12,rtol=1e-12,kind="public_law_actual")
                audit.check("public_law.nonmutation."+arm+"."+row["match_id"]+row["side"],[theta.tolist(),counts[i:i+1].tolist(),ages[i:i+1].tolist(),previous[i:i+1].tolist()],[x.tolist() for x in before])
    q = [.5,.25,.25]
    for bo in (1,3,5):
        predictions = {(i,j):UnifiedPredictionResult(canonical_match_id=None,prob_a=.75,prob_b=.25,model_name="Causal-C0",model_version="c0-native-2026-w32-e12-v1",feature_version="native-c0-history16-v1",best_of=bo,diagnostics={"mode":"full","probability_target":"series_win_already_projected"}) for i,j in np.ndindex(2,2)}
        actual = supported_roster_bounds(q,q,predictions,best_of=bo)
        expected = partial(q,q,{key:.75 for key in predictions})
        for key in ("lower","upper","missing_mass","supported_mass","prob_a","scalar_absence"):
            audit.check("toy.public_bounds.BO"+str(bo)+"."+key,actual[key],expected[key],kind="toy_public_actual")
        near = supported_roster_bounds([1-2**-40,0,2**-40],[1,0,0],{(0,0):predictions[0,0]},best_of=bo)
        audit.check("toy.public_bounds.BO"+str(bo)+".no_1e8_scalar_fallback",near["prob_a"],None,kind="toy_public_actual")
        audit.check("toy.public_bounds.BO"+str(bo)+".near_mass",near["missing_mass"],2**-40,kind="toy_public_actual")
        point = supported_roster_bounds([1,0,0],[1,0,0],{(0,0):predictions[0,0]},best_of=bo)
        audit.check("toy.public_bounds.BO"+str(bo)+".conditional_point",[point["lower"],point["upper"],point["prob_a"],point["missing_mass"]],[.75,.75,.75,0.],kind="toy_public_actual")
        from dataclasses import replace
        reverse_predictions = {key:replace(value,prob_a=.25,prob_b=.75) for key,value in predictions.items()}
        reverse_fixture = supported_roster_bounds(q,q,reverse_predictions,best_of=bo)
        audit.check("toy.public_bounds.BO"+str(bo)+".exact_reversal",[reverse_fixture["lower"],reverse_fixture["upper"]],[.140625,.578125],kind="toy_public_actual")
        for invalid in ([-.1,.1,1.],[.2,.2,.2],[float("nan"),0.,1.],[1.,0.]):
            rejected = False
            try:
                supported_roster_bounds(invalid,q,predictions,best_of=bo)
            except ValueError:
                rejected = True
            audit.check("toy.public_bounds.BO"+str(bo)+".reject."+repr(invalid),rejected,True,kind="toy_public_actual")


def selected_cases(rows):
    """Outcome-blind lexical coverage of frozen ASSESS structural/history/BO cells."""
    first = {}
    for row in sorted((r for r in rows if r["partition"]=="ASSESS"),key=lambda r:(r["match_id"],r["side"])):
        cells = ["BO"+str(row["best_of"]),"candidates_"+str(len(row["candidates"])),"absence_"+(row["absence"] or "AVAILABLE"),"history_"+("cold" if not row["history_ids"] else "full" if len(row["history_ids"])==10 else "partial")]
        for cell in cells:
            first.setdefault(cell,row["match_id"])
    chosen = sorted(set(first.values()))[:32]
    return chosen,[{"match_id":mid,"states":sorted(cell for cell,value in first.items() if value==mid)} for mid in chosen]


def finalize_statuses(audit, receipt):
    if audit.failures:
        receipt["status"] = "FAIL"
        if receipt.get("fitted_prediction_oracle_status") in ("PASS","PENDING_REQUIRED_CHECKS"):
            receipt["fitted_prediction_oracle_status"] = "FAIL"
    elif receipt.get("fitted_prediction_oracle_status") == "PENDING_REQUIRED_CHECKS":
        receipt["fitted_prediction_oracle_status"] = "PASS"
        receipt["status"] = "PASS"


def no_fit_outputs_replay(audit, roots):
    for name in ("conditional.json","global.json"):
        audit.check("prepare.no_saved_model."+name,(roots["06_models"]/name).exists(),False)
    for stage,name in (("07_model_output","side_laws.parquet"),("07_model_output","series_scores.parquet"),
                       ("07_model_output","native_pairs.parquet"),("07_model_output","roster_bounds.parquet"),
                       ("08_reporting","contrasts.npz")):
        audit.check("prepare.no_predictions."+name,(roots[stage]/name).exists(),False)


def partial_hard_stop(reporting, index):
    stops = [reporting/name for name in ("prepare_hard_stop.json","run_hard_stop.json","recover_evaluation_hard_stop.json") if (reporting/name).is_file()]
    if len(stops) > 1:
        raise ValueError("multiple command hard-stop receipts")
    return load(stops[0]) if stops else None


def artifact_available(audit, repo, descriptors, path):
    matches = [d for d in descriptors if (repo/d["repo_relative_path"]).resolve() == path.resolve()]
    audit.check("partial.descriptor."+path.name+".unique",len(matches),1)
    if len(matches) != 1:
        return False
    descriptor = matches[0]
    available = descriptor["absence_status"] == "AVAILABLE"
    audit.check("partial.descriptor."+path.name+".presence",path.is_file(),available)
    if not available:
        audit.check("partial.descriptor."+path.name+".reason",bool(descriptor["absence_reason"]),True)
    return available


def not_run(audit, receipt, name, reason):
    audit.check(name,None,None,kind="not_run")
    audit.checks[-1].update(status="NOT_RUN",reason=reason)
    receipt["scope_not_exercised"].append(name+": "+reason)


def native_execution_replay(audit, execution, pairs, blocked, hard_stop=None):
    sessions = execution["worker_sessions"]
    startups = execution["worker_startup_attempts"]
    established = [r for r in startups if r["status"] == "ESTABLISHED"]
    failed = [r for r in startups if r["status"] == "FAILED"]
    audit.check("native.lifecycle.startup_statuses",len(established)+len(failed),len(startups))
    audit.check("native.lifecycle.session_indices",[s["session_index"] for s in sessions],list(range(len(sessions))))
    audit.check("native.lifecycle.established_sessions",[s["session_index"] for s in established],list(range(len(sessions))))
    audit.check("native.lifecycle.worker_pids",execution["worker_pids"],[s["pid"] for s in sessions])
    audit.check("native.lifecycle.startup_indices",[s["attempt_index"] for s in startups],list(range(len(startups))))
    audit.check("native.lifecycle.startup_count",execution["native_worker_startup_attempts"],len(startups))
    audit.check("native.lifecycle.startup_failures",execution["native_worker_startup_failures"],len(failed))
    audit.check("native.lifecycle.failed_startup_has_no_session",[s["session_index"] for s in failed],[None]*len(failed))
    for startup in failed:
        audit.check("native.lifecycle.failed_startup_reason",bool(startup["failure_type"]) and bool(startup["failure_reason"]),True)
    advances = set()
    for row in pairs:
        if row["native_requested"]:
            audit.check("native.lifecycle.request_session",row["session_index"] in range(len(sessions)),True)
            advances.add((row["session_index"],row["cutoff_day"]))
        else:
            audit.check("native.lifecycle.uncalled_startup",row["failure_phase"],"STARTUP")
    confirmed = {(r["session_index"],r["cutoff_day"]) for r in pairs if r["status"]=="AVAILABLE"}
    audit.check("native.lifecycle.origin_advances",execution["state_advance_origin_count"],len(confirmed))
    audit.check("native.lifecycle.origin_advance_attempts",execution["state_advance_origin_attempt_count"],len(advances))
    audit.check("native.lifecycle.origin_advance_unconfirmed",execution["state_advance_origin_unconfirmed_count"],len(advances-confirmed))
    for session in sessions:
        audit.check("native.lifecycle."+str(session["session_index"])+".pid",isinstance(session["pid"],int) and session["pid"]>0,True)
        clocks = [datetime.fromisoformat(session[k]) for k in ("established_at","closed_at")]
        audit.check("native.lifecycle."+str(session["session_index"])+".closed",all(c.tzinfo is not None for c in clocks) and clocks[0]<=clocks[1] and bool(session["close_reason"]),True)
    for row in pairs:
        if row.get("failure_reason") is not None:
            availability = row.get("failure_kind") == "AVAILABILITY"
            audit.check("native.failure."+str((row["match_id"],row["candidate_a_index"],row["candidate_b_index"]))+".authorized",
                        availability and row.get("failure_type") == "C0WorkerTimeout" or not availability and hard_stop is not None,True)
            if not availability:
                audit.check("native.integrity_failure_is_fatal",row.get("status"),"NATIVE_PAIR_FATAL")
                audit.check("native.integrity_failure_matches_stop",row.get("failure_kind"),hard_stop.get("failure_kind") if hard_stop else None)
    for field,expected in (("native_worker_sessions",len(sessions)),("native_worker_recycles",max(0,len(sessions)-1)),
                           ("native_request_count",sum(r.get("native_requested",True) for r in pairs)),
                           ("native_valid_pairs",sum(r["status"]=="AVAILABLE" for r in pairs)),
                           ("native_failed_pairs",sum(r["status"]!="AVAILABLE" for r in pairs)),
                           ("native_blocked_pairs",len(blocked)),("raw_snapshots_persisted",sum(r.get("evidence_path") is not None for r in pairs)),
                           ("worker_closed",True)):
        audit.check("native.execution."+field,execution[field],expected)
    audit.check("native.execution.requests_by_series",execution["requests_by_series"],dict(Counter(r["match_id"] for r in pairs if r.get("native_requested",True))))


def partial_outputs_replay(audit, receipt, stop, repo, roots, index, trace, rows, protocol, binding, selected, clocks, pins, research, lineage=None, recovery=None):
    reason = stop["failure_type"]+": "+stop["failure_reason"]
    receipt["fitted_prediction_oracle_status"] = "NOT_RUN_PARTIAL_HARD_STOP"
    receipt["primary_slot_statuses"] = {"7":"NOT_RUN_PARTIAL_HARD_STOP","8":"NOT_RUN_PARTIAL_HARD_STOP"}
    audit.check("partial.no_restart",stop["no_restart"],True)
    audit.check("partial.operation_ceiling",len(trace)<=2,True)
    audit.check("partial.actual_attempt_count",stop["formal_fits"],len(trace))
    decision = load(roots["08_reporting"]/"decision.json")
    # A late owned failure may follow an immutable scientific decision write.
    # The hard-stop failure, not that earlier decision, controls oracle success.
    audit.check("partial.hard_stop_status",stop["status"],"BLOCKED_IDENTITY_RELEASE" if stop["failure_kind"]=="SOURCE_VALIDATION" else "INCONCLUSIVE")
    audit.check("partial.retained_decision_status",decision["status"] in ("BLOCKED_IDENTITY_RELEASE","INSUFFICIENT_FIT_SUPPORT","INVALID_FIT","FIT_FAILED_WITHIN_BUDGET","NO_LINEUP_LAW_GAIN","INCONCLUSIVE","LINEUP_LAW_DEVELOPMENT_SUPPORTED_NOT_WINNER_MODEL"),True)
    for field,value in (("formal_fits",len(trace)),("completed_fits",sum(r["status"]=="COMPLETED" for r in trace)),
                        ("unused_TRAIN_slots",2-len(trace)),("CAL_fits",0),("C0_fits",0),("ASSESS_fits",0)):
        audit.check("partial.decision."+field,decision[field],value)
    for field,value in (("completed_fits",sum(r["status"]=="COMPLETED" for r in trace)),("unused_TRAIN_slots",2-len(trace))):
        audit.check("partial.hard_stop."+field,stop[field],value)
    if not trace:
        no_fit_outputs_replay(audit,roots)
    budget_replay(audit,trace,rows,*(lineage[:3] if lineage else (roots,protocol,binding)))
    audit.check("partial.execution_failure",stop["failure_kind"],"SOURCE_VALIDATION",kind="hard_stop")
    available = lambda stage,name: artifact_available(audit,repo,index["artifact_descriptors"],roots[stage]/name)
    models = {}
    for arm in ("conditional","global"):
        if available("06_models",arm+".json"):
            models[arm] = saved_fit_replay(audit,load(roots["06_models"]/(arm+".json")),rows,arm=="conditional",arm,require_acceptance=False)
            receipt["raw_state_count"] += 1
        else:
            not_run(audit,receipt,"partial.saved_fit."+arm,reason)
    laws, scores = {}, []
    if available("07_model_output","side_laws.parquet"):
        exported = table(roots["07_model_output"]/"side_laws.parquet")
        for row in exported:
            if isinstance(row.get("label"),str):
                row["label"] = ("LATEST","PREVIOUS","OTHER").index(row["label"])
        score_available = available("07_model_output","series_scores.parquet")
        laws,scores = laws_and_scores(audit,rows,models,exported,table(roots["07_model_output"]/"series_scores.parquet") if score_available else None)
        if not score_available:
            not_run(audit,receipt,"partial.series_scores",reason)
    else:
        not_run(audit,receipt,"partial.side_laws_and_scores",reason)
        audit.check("partial.scores_require_laws",available("07_model_output","series_scores.parquet"),False)
    metrics_available = available("08_reporting","metrics.json")
    contrasts_available = available("08_reporting","contrasts.npz")
    if metrics_available:
        metrics = load(roots["08_reporting"]/"metrics.json")
        if laws:
            diagnostics_replay(audit,rows,laws,metrics["diagnostics"])
        if contrasts_available and scores:
            with np.load(roots["08_reporting"]/"contrasts.npz",allow_pickle=False) as arrays:
                receipt["primary_slot_statuses"] = uncertainty_replay(audit,rows,scores,arrays,metrics)
    if not (metrics_available and contrasts_available and scores):
        not_run(audit,receipt,"partial.contrasts",reason)
    if available("08_reporting","cal_audit.json"):
        cal = load(roots["08_reporting"]/"cal_audit.json")
        if laws:
            diagnostics_replay(audit,rows,laws,cal["diagnostics"],partitions=("CAL",))
        audit.check("partial.CAL.once",cal["audit_number"],1)
        audit.check("partial.CAL.no_fit",cal["formal_fits"],0)
    else:
        not_run(audit,receipt,"partial.CAL",reason)
    pairs = table(roots["07_model_output"]/"native_pairs.parquet") if available("07_model_output","native_pairs.parquet") else []
    for pair in pairs:
        if pair.get("failure_reason") is not None and pair.get("evidence_path"):
            name = "partial.failed_capture."+str((pair["match_id"],pair["candidate_a_index"],pair["candidate_b_index"]))
            try:
                capture = load(repo/pair["evidence_path"])
                snapshot = decode_snapshot(capture["encoded_snapshot"])
                hashes = {k:v for values in tensor_hashes(snapshot).values() for k,v in values.items()}
                prediction = capture["prediction"]
                audit.check(name+".input_hashes",prediction["diagnostics"]["input_hashes"],hashes,kind="native_input_digest")
                audit.check(name+".identity",prediction["model_version"],"c0-native-2026-w32-e12-v1",kind="native_identity")
                audit.check(name+".feature_identity",prediction["feature_version"],"native-c0-history16-v1",kind="native_identity")
                audit.check(name+".provenance",prediction["diagnostics"]["feature_provenance"],capture["native_history_provenance"])
                for field,role in (("raw_history_sha256","raw_source"),("release_map_sha256","native_release_map")):
                    audit.check(name+"."+field,capture["native_history_provenance"][field],pins[role]["sha256"])
                audit.check(name+".probability",math.isfinite(prediction["prob_a"]) and 0<=prediction["prob_a"]<=1,True)
                op_a = next(r for r in rows if r["match_id"]==pair["match_id"] and r["side"]=="A")
                op_b = next(r for r in rows if r["match_id"]==pair["match_id"] and r["side"]=="B")
                for field,expected in (("team1_id",op_a["team_id"]),("team2_id",op_b["team_id"]),
                                       ("cutoff_day",op_a["date"]),("competition_context",op_a["competition_context"]),
                                       ("roster_a",op_a["candidates"][int(pair["candidate_a_index"])]["roster"]),
                                       ("roster_b",op_b["candidates"][int(pair["candidate_b_index"])]["roster"])):
                    audit.check(name+"."+field,snapshot["target"][field],expected,kind="native_identity")
            except (ValueError,KeyError,TypeError,IndexError,StopIteration) as error:
                audit.check(name+".schema",type(error).__name__+": "+str(error),None,kind="hard_stop")
    blocks = table(roots["07_model_output"]/"native_blocked_pairs.parquet") if available("07_model_output","native_blocked_pairs.parquet") else []
    bounds_available = available("07_model_output","roster_bounds.parquet")
    if pairs or bounds_available:
        native_replay(audit,rows,laws,pairs,table(roots["07_model_output"]/"roster_bounds.parquet") if bounds_available else [],repo,[],clocks,pins,blocks,research,complete=False,
                      inherited_pairs=[r for r in pairs if r.get("execution_scope")=="PARENT_INHERITED"] if recovery else [])
    if available("08_reporting","native_execution.json"):
        execution = load(roots["08_reporting"]/"native_execution.json")
        if recovery:
            recovery_native_replay(audit,receipt,repo,roots,lineage,recovery,pairs,blocks,execution,hard_stop=stop)
        else:
            native_execution_replay(audit,execution,pairs,blocks,stop)
    else:
        not_run(audit,receipt,"partial.native_execution",reason)
    if not pairs and not bounds_available:
        not_run(audit,receipt,"partial.native_pairs_and_bounds",reason)
    if available("08_reporting","query_nonmutation.json"):
        mutation = load(roots["08_reporting"]/"query_nonmutation.json")
        audit.check("partial.query_nonmutation.status",mutation["status"],"UNCHANGED")
        audit.check("partial.query_nonmutation.before_after",mutation["after"],mutation["before"])
        audit.check("partial.query_nonmutation.candidates",mutation["after"]["candidate_ledger_sha256"],sha(roots["04_feature"]/"opportunities.parquet"))
        expected_theta_hashes = {arm:hashlib.sha256(np.asarray(theta,dtype=np.float64).tobytes(order="C")).hexdigest() for arm,theta in models.items()}
        audit.check("partial.query_nonmutation.theta",mutation["after"]["theta_sha256"],expected_theta_hashes)
    else:
        not_run(audit,receipt,"partial.query_nonmutation",reason)
    if available("08_reporting","saved_output_binding.json"):
        output_binding = load(roots["08_reporting"]/"saved_output_binding.json")
        expected_paths = {str(p) for stage in ("06_models","07_model_output") for p in roots[stage].rglob("*") if p.is_file()}
        audit.check("partial.output_binding.paths",sorted(output_binding["hashes"]),sorted(expected_paths))
        for path,digest in output_binding["hashes"].items():
            audit.check("partial.output_binding."+path,sha(path),digest,kind="artifact_hash")
    else:
        not_run(audit,receipt,"partial.saved_output_binding",reason)
    if available("08_reporting","public_cases.json"):
        audit.check("partial.saved_public_cases",load(roots["08_reporting"]/"public_cases.json"),load(roots["05_model_input"]/"public_cases.json"))
    else:
        not_run(audit,receipt,"partial.saved_public_cases",reason)
    not_run(audit,receipt,"partial.public_native_and_saved_contract",reason)
    receipt["status"] = "FAIL" if stop["failure_kind"]!="SOURCE_VALIDATION" else "PASS_CLOSURE_CONTRACT"


def partial_preparation_replay(audit, receipt, stop, repo, roots, index, canonical, pins, protocol, binding, research):
    reason = stop["failure_type"]+": "+stop["failure_reason"]
    rows, events, clocks, first = [], [], [], {}
    source_failed = False
    try:
        events,clocks,first,map_count,map_rows = source_replay(Path(pins["raw_source"]["path"]),load(pins["native_release_map"]["path"]),load(pins["canonical_release_override"]["path"]),{r["match_id"]:r for r in canonical})
        rows,conflicts = opportunity_replay(canonical,table(pins["native_participants"]["path"]),events,first,clocks,audit)
    except ValueError as error:
        source_failed = True
        audit.check("partial.source_validation_failure_class",stop["failure_kind"],"SOURCE_VALIDATION",kind="hard_stop")
        receipt["scope_exercised"].append("independent source validation rejection: "+str(error))
    if stop["failure_kind"] == "SOURCE_VALIDATION":
        audit.check("partial.source_validation_independently_rejected",source_failed,True)
    expected_tables = [("04_feature","appearances.parquet",events),
                       ("04_feature","source_maps.parquet",map_rows if not source_failed else None),
                       ("08_reporting","clock_comparison.parquet",clocks)]
    if not source_failed:
        expected_tables += [("08_reporting","origin_conflicts.parquet",conflicts)]
        native_order = sorted(clocks,key=lambda c:(c["native_release_day"],order(c["source_match_id"])))
        expected_tables.append(("04_feature","native_source_order.parquet",[{"native_rank":i,**clock} for i,clock in enumerate(native_order)]))
        side_lookup = {(r["match_id"],r["side"]):r for r in rows}
        series = []
        for row in canonical:
            a,b = side_lookup[row["match_id"],"A"],side_lookup[row["match_id"],"B"]
            expected = dict(match_id=row["match_id"],roster_a_ids=a["actual_roster"],roster_b_ids=b["actual_roster"],
                            lineup_label_availability="ALL_SIDES" if a["label_absence"] is None and b["label_absence"] is None else "ABSENT" if a["label_absence"] is not None and b["label_absence"] is not None else "ONE_SIDE",
                            lineup_label_absence_reasons=[a["label_absence"],b["label_absence"]])
            for side in ("a","b"):
                roster = [ident(p) for p in row["roster_"+side]]
                expected["canonical_roster_"+side+"_ids"] = sorted(p for p in roster if p is not None)+[None]*roster.count(None)
            series.append(expected)
        expected_tables.append(("04_feature","series.parquet",series))
    for stage,name,expected in expected_tables:
        if artifact_available(audit,repo,index["artifact_descriptors"],roots[stage]/name):
            if expected:
                actual = table(roots[stage]/name)
                fields = tuple(expected[0])
                if name == "origin_conflicts.parquet":
                    fields = ("match_id","cutoff_day","conflict_source_ids","native_absence")
                if name == "source_maps.parquet":
                    actual.sort(key=lambda r:(r["source_match_id"],r["map_index"],r["team_id"]))
                    expected.sort(key=lambda r:(r["source_match_id"],r["map_index"],r["team_id"]))
                compare_rows(audit,"partial.prepare."+name,actual,expected,fields)
            else:
                not_run(audit,receipt,"partial.prepare."+name,reason)
        else:
            not_run(audit,receipt,"partial.prepare."+name,reason)
    if source_failed:
        support = load(roots["08_reporting"]/"support.json")
        audit.check("partial.support.invalid_identity",support["counts"]["identity_release_valid"],False)
        audit.check("partial.support.source_failure_evidence",bool(support.get("source_backed_failure")),True)
    if not source_failed:
        if artifact_available(audit,repo,index["artifact_descriptors"],roots["04_feature"]/"opportunities.parquet"):
            actual = table(roots["04_feature"]/"opportunities.parquet")
            for row in actual:
                if isinstance(row["candidates"],str):
                    row["candidates"] = json.loads(row["candidates"])
                if isinstance(row["label"],str):
                    row["label"] = ("LATEST","PREVIOUS","OTHER").index(row["label"])
            compare_rows(audit,"partial.prepare.opportunities",actual,rows,("match_id","side","partition","date","team_id","actual_roster","history_ids","candidates","counts","ages","previous_available","label","label_absence","absence"))
        if artifact_available(audit,repo,index["artifact_descriptors"],roots["08_reporting"]/"opportunity_inventory.json"):
            inventory_replay(audit,rows,load(roots["08_reporting"]/"opportunity_inventory.json"))
        support = load(roots["08_reporting"]/"support.json")
        if support.get("counts_unavailable_reason"):
            audit.check("partial.support.unestablished_identity",support["counts"]["identity_release_valid"],None)
        else:
            counts,failed = support_replay(rows)
            for field,value in support["counts"].items():
                audit.check("partial.support."+field,value,True if field=="identity_release_valid" else counts[field])
            audit.check("partial.support.failed_floors",support["failed_floors"],failed)
    trace = load(roots["08_reporting"]/"operation_trace.json")
    audit.check("partial.prepare.zero_operations",trace,[])
    audit.check("partial.index.operation_trace",index["operation_trace"],trace)
    receipt["formal_operation_counts"] = {"TRAIN":len(trace),"CAL":0,"C0":0,"ASSESS":0,"restarts":0}
    partial_outputs_replay(audit,receipt,stop,repo,roots,index,trace,rows,protocol,binding,[],clocks,pins,research)
    final_integrity_replay(audit,receipt,repo,roots,pins,binding)
    finalize_statuses(audit,receipt)


def final_integrity_replay(audit, receipt, repo, roots, pins, binding, archived_authorization=None):
    decision_path = roots["08_reporting"]/"decision.json"
    if decision_path.is_file():
        decision = load(decision_path)
        for field,pinned in (("source_after",pins),("code_after",binding["import_origins"])):
            audit.check("decision."+field+".keys",sorted(decision[field]),sorted(pinned))
            for role,pin in pinned.items():
                relative = str(Path(pin["path"]).resolve().relative_to(repo)) if Path(pin["path"]).resolve().is_relative_to(repo) else None
                archived = (archived_authorization or {}).get("old_runtime_snapshots",{}).get(relative)
                expected = archived["sha256"] if archived and relative in RECOVERY_RUNTIME_PATHS else sha(pin["path"])
                audit.check("decision."+field+"."+role,decision[field][role],expected,kind="artifact_hash")
    for path,entry in receipt["protected_before_after"].items():
        entry["after"] = sha(repo/path)
        audit.check("protected.after."+path,entry["after"],entry["before"],kind="protected_hash")



RECOVERY_RUNTIME_PATHS = {
    "src/analysis/lineup_law_reliability.py",
    "scripts/train_lineup_law.py",
    "scripts/verify_lineup_law_reliability.py",
}
ZERO_RECOVERY_FITS = {"TRAIN":0,"CAL":0,"C0":0,"ASSESS":0}


def runtime_pin_replay(audit, repo, name, pin, authorization=None, *, allow_archived=False):
    path = Path(pin["path"]).resolve()
    relative = str(path.relative_to(repo)) if path.is_relative_to(repo) else None
    archived = (authorization or {}).get("old_runtime_snapshots",{}).get(relative)
    if allow_archived and archived and relative in RECOVERY_RUNTIME_PATHS and relative in authorization["allowed_source_changes"]:
        audit.check("pin."+name+".archived",sha(repo/archived["snapshot"]),archived["sha256"],kind="artifact_hash")
        audit.check("pin."+name+".authorized_old_digest",pin["sha256"],archived["sha256"],kind="artifact_hash")
    else:
        audit.check("pin."+name,sha(path),pin["sha256"],kind="artifact_hash")


def recovery_counts_replay(audit, new_trace, inherited_pairs, new_pairs, forbidden):
    audit.check("recovery.zero_new_fit_trace",new_trace,[])
    audit.check("recovery.no_original_native_retry",sorted({r["match_id"] for r in new_pairs}&set(forbidden)),[])
    audit.check("recovery.original_attempted_ids",sorted({r["match_id"] for r in inherited_pairs if r["native_requested"]}),sorted(forbidden))


def recovery_lineage_replay(audit, repo, roots, protocol, binding, authorization, descriptor):
    parent_id = authorization["parent_run_id"]
    parent_roots = {s:repo/"data"/s/"lineup_law_reliability"/parent_id for s in roots}
    audit.check("recovery.schema",descriptor["schema"],"lineup_law_evaluation_recovery_v1")
    audit.check("recovery.parent_run_id",descriptor["parent_run_id"],parent_id)
    audit.check("recovery.run_id",descriptor["run_id"],authorization["run_id"])
    audit.check("recovery.authorization_digest",descriptor["authorization"]["sha256"],sha(descriptor["authorization"]["path"]),kind="artifact_hash")
    audit.check("recovery.authorization_contents",load(descriptor["authorization"]["path"]),authorization)
    audit.check("recovery.archived_runtime",descriptor["archived_runtime"],authorization["old_runtime_snapshots"])
    audit.check("recovery.inherited_fits",descriptor["inherited_TRAIN_fits"],2)
    audit.check("recovery.fit_budget",descriptor["new_fit_budget"],ZERO_RECOVERY_FITS)
    audit.check("recovery.forbidden_ids",descriptor["no_retry_native_match_ids"],authorization["no_retry_native_match_ids"])
    for role,stage,name in (("parent_protocol","05_model_input","protocol.json"),("parent_binding","05_model_input","binding.json"),
                            ("parent_hard_stop","08_reporting","run_hard_stop.json"),("parent_decision","08_reporting","decision.json")):
        expected = parent_roots[stage]/name
        audit.check("recovery."+role+".path",Path(descriptor[role]["path"]).resolve(),expected.resolve())
        audit.check("recovery."+role+".sha256",descriptor[role]["sha256"],sha(expected),kind="artifact_hash")
    parent_protocol = load(parent_roots["05_model_input"]/"protocol.json")
    approval = load(parent_roots["05_model_input"]/"prefit_approval.json")
    for field,value in (("status","APPROVED"),("study","R1"),("run_id",parent_id),
                        ("protocol_sha256",sha(parent_roots["05_model_input"]/"protocol.json")),
                        ("binding_sha256",sha(parent_roots["05_model_input"]/"binding.json")),
                        ("support_sha256",sha(parent_roots["08_reporting"]/"support.json")),
                        ("fit_budget",{"TRAIN":2,"CAL":0,"C0":0,"ASSESS":0,"restarts":0})):
        audit.check("recovery.parent_approval."+field,approval[field],value)
    prepare_oracle = approval["independent_prepare_oracle"]
    audit.check("recovery.parent_prepare_oracle.digest",sha(repo/prepare_oracle["path"]),prepare_oracle["sha256"],kind="artifact_hash")
    audit.check("recovery.parent_prepare_oracle.status",load(repo/prepare_oracle["path"])["status"],"PASS_PREPARE_CONTRACT")
    approved_at = datetime.fromisoformat(approval["created_at_utc"])
    original_trace = load(parent_roots["08_reporting"]/"operation_trace.json")
    audit.check("recovery.parent_approval.before_fits",approved_at.tzinfo is not None and
                all(approved_at<=datetime.fromisoformat(item["started_at_utc"]) for item in original_trace),True)
    parent_binding = load(parent_roots["05_model_input"]/"binding.json")
    audit.check("recovery.binding.source_pins",binding["pins"],protocol["hashes"])
    audit.check("recovery.binding.import_origins",binding["import_origins"],protocol["import_origins"])
    for key,pin in {**parent_binding["pins"],**parent_binding["import_origins"]}.items():
        runtime_pin_replay(audit,repo,"parent."+key,pin,authorization,allow_archived=True)
    allowed_protocol_changes = {"run_id","output_paths","hashes","import_origins","recovery"}
    audit.check("recovery.scientific_protocol",{k:v for k,v in protocol.items() if k not in allowed_protocol_changes},
                {k:v for k,v in parent_protocol.items() if k not in allowed_protocol_changes})
    for field in ("hashes","import_origins"):
        audit.check("recovery.protocol."+field+".keys",sorted(protocol[field]),sorted(parent_protocol[field]))
        for key,old in parent_protocol[field].items():
            origin = parent_binding["import_origins"].get(key,parent_binding["pins"].get(key,{}))
            relative = str(Path(origin["path"]).resolve().relative_to(repo)) if origin.get("path") and Path(origin["path"]).resolve().is_relative_to(repo) else None
            if relative not in RECOVERY_RUNTIME_PATHS:
                audit.check("recovery.protocol."+field+"."+key,protocol[field][key],old)
            else:
                audit.check("recovery.protocol."+field+"."+key+".current",protocol[field][key],
                            {"path":str(repo/relative),"sha256":sha(repo/relative)},kind="artifact_hash")
    audit.check("recovery.current_runtime.keys",sorted(descriptor["current_runtime"]),sorted(RECOVERY_RUNTIME_PATHS))
    for relative,pin in descriptor["current_runtime"].items():
        audit.check("recovery.current_runtime."+relative+".path",Path(pin["path"]).resolve(),(repo/relative).resolve())
        audit.check("recovery.current_runtime."+relative+".digest",pin["sha256"],sha(repo/relative),kind="artifact_hash")
    inherited = descriptor["inherited_artifacts"]
    audit.check("recovery.binding.descriptor_sha256",binding["recovery_sha256"],sha(roots["05_model_input"]/"recovery.json"),kind="artifact_hash")
    required = {(stage,name) for stage,names in {
        "05_model_input":("prefit_approval.json","public_case_contracts.json","public_cases.json"),
        "06_models":("conditional.json","global.json"),
        "07_model_output":("side_laws.parquet","series_scores.parquet"),
        "08_reporting":("operation_trace.json","support.json","cal_audit.json","clock_comparison.parquet","origin_conflicts.parquet"),
    }.items() for name in names}
    required |= {("04_feature",p.name) for p in parent_roots["04_feature"].iterdir() if p.is_file()}
    for stage,name in required:
        destination = roots[stage]/name
        entry = inherited[str(destination)]
        audit.check("recovery.inherited."+name+".path",Path(entry["path"]).resolve(),(parent_roots[stage]/name).resolve())
    for destination,entry in inherited.items():
        target,source = Path(destination).resolve(),Path(entry["path"]).resolve()
        audit.check("recovery.inherited.destination_scope."+destination,any(target.is_relative_to(root) for root in roots.values()),True)
        audit.check("recovery.inherited.parent_scope."+destination,any(source.is_relative_to(root) for root in parent_roots.values()),True)
        audit.check("recovery.inherited.parent."+destination,sha(source),entry["sha256"],kind="artifact_hash")
        audit.check("recovery.inherited.copy."+destination,sha(target),entry["sha256"],kind="artifact_hash")
    stop,decision = load(parent_roots["08_reporting"]/"run_hard_stop.json"),load(parent_roots["08_reporting"]/"decision.json")
    for name,value in (("parent_status",stop["status"]),("parent_decision",decision["status"])):
        audit.check("recovery."+name,value,"INCONCLUSIVE")
    audit.check("recovery.parent_original_failure",stop["failure_type"],"AttributeError")
    audit.check("recovery.parent_failure_not_source_validation",stop["failure_kind"]!="SOURCE_VALIDATION",True)
    audit.check("recovery.parent_no_metrics",(parent_roots["08_reporting"]/"metrics.json").exists(),False)
    audit.check("recovery.parent_no_intervals",(parent_roots["08_reporting"]/"contrasts.npz").exists(),False)
    return parent_roots,parent_protocol,parent_binding,stop


def recovery_native_replay(audit, receipt, repo, roots, lineage, recovery, pairs, blocked, execution, hard_stop=None):
    parent_roots,_,_,stop = lineage
    inherited = [r for r in pairs if r.get("execution_scope")=="PARENT_INHERITED"]
    new = [r for r in pairs if r.get("execution_scope")=="RECOVERY_NEW"]
    audit.check("recovery.native.exact_execution_scopes",len(inherited)+len(new),len(pairs))
    parent_pairs = table(parent_roots["07_model_output"]/"native_pairs.parquet")
    parent_block_path = parent_roots["07_model_output"]/"native_blocked_pairs.parquet"
    parent_blocks = table(parent_block_path) if parent_block_path.is_file() else []
    inherited_clean = [{k: r.get(k) for k in p} for r, p in zip(inherited, parent_pairs)]
    audit.check("recovery.native.inherited_pairs", inherited_clean, parent_pairs)
    audit.check("recovery.native.inherited_pairs_extra_null", all(r.get(k) is None for r, p in zip(inherited, parent_pairs) for k in r if k not in p and k != "execution_scope"), True)
    parent_execution = load(parent_roots["08_reporting"]/"native_execution.json")
    audit.check("recovery.native.inherited_execution",execution["inherited_execution"],parent_execution)
    inherited_blocks = [r for r in blocked if r.get("execution_scope")=="PARENT_INHERITED"]
    audit.check("recovery.native.inherited_blocks",[{k:v for k,v in r.items() if k!="execution_scope"} for r in inherited_blocks],parent_blocks)
    audit.check("recovery.native.exact_block_scopes",len(inherited_blocks)+sum(r.get("execution_scope")=="RECOVERY_NEW" for r in blocked),len(blocked))
    native_execution_replay(audit,parent_execution,parent_pairs,parent_blocks,stop)
    new_blocks = [r for r in blocked if r.get("execution_scope")=="RECOVERY_NEW"]
    native_execution_replay(audit,execution,new,new_blocks,hard_stop)
    recovery_counts_replay(audit,load(roots["08_reporting"]/"new_operation_trace.json"),inherited,new,recovery["no_retry_native_match_ids"])
    for field,value in (("inherited_request_count",parent_execution["native_request_count"]),
                        ("new_native_request_count",execution["native_request_count"]),
                        ("no_retry_native_match_ids",recovery["no_retry_native_match_ids"])):
        audit.check("recovery.native."+field,execution[field],value)
    decision = load(roots["08_reporting"]/"decision.json")
    audit.check("recovery.decision.inherited_fits",decision["inherited_TRAIN_fits"],2)
    audit.check("recovery.decision.new_fits",decision["new_fit_counts"],ZERO_RECOVERY_FITS)
    receipt["native_operation_counts"] = {"inherited_requests":parent_execution["native_request_count"],
                                          "new_requests":execution["native_request_count"],
                                          "inherited_sessions":parent_execution["native_worker_sessions"],
                                          "new_sessions":execution["native_worker_sessions"]}
    return inherited

def run_oracle(args, audit, receipt):
    repo = args.repo_root.resolve()
    sys.path.insert(0,str(repo))
    research = args.research_root
    if research is None:
        pointer = repo/"data/research_root.txt"
        research = Path(os.environ.get("ENSEMBLE_RESEARCH_ROOT") or (pointer.read_text().strip() if pointer.is_file() else repo))
    research = research.expanduser().resolve()
    roots = {s:repo/"data"/s/"lineup_law_reliability"/args.run_id for s in ("04_feature","05_model_input","06_models","07_model_output","08_reporting")}
    protocol_path, binding_path = roots["05_model_input"]/"protocol.json", roots["05_model_input"]/"binding.json"
    protocol, binding = load(protocol_path), load(binding_path)
    recovery_path = roots["05_model_input"]/"recovery.json"
    recovery = load(recovery_path) if recovery_path.is_file() else None
    authorization_path = getattr(args,"recovery_authorization",None)
    if recovery and authorization_path is None:
        authorization_path = Path(recovery["authorization"]["path"])
    recovery_authorization = load(authorization_path) if authorization_path else None
    lineage = None
    if recovery_authorization:
        expected_authorization_path = repo/"data/08_reporting/sports_reliability_improvements"/recovery_authorization["parent_portfolio"]/"r1_recovery_authorization.json"
        audit.check("recovery.authorization.exact_evidence_path",Path(authorization_path).resolve(),expected_authorization_path.resolve())
        audit.check("recovery.authorization.schema",recovery_authorization["schema"],"lineup_law_technical_recovery_authorization_v1")
        audit.check("recovery.authorization.allowed_paths",sorted(recovery_authorization["allowed_source_changes"]),sorted(RECOVERY_RUNTIME_PATHS))
        audit.check("recovery.authorization.snapshot_paths",sorted(recovery_authorization["old_runtime_snapshots"]),sorted(RECOVERY_RUNTIME_PATHS))
        audit.check("recovery.authorization.run_scope",args.run_id in (recovery_authorization["run_id"],recovery_authorization["parent_run_id"]),True)
        audit.check("recovery.authorization.fit_budget",recovery_authorization["new_fit_budget"],ZERO_RECOVERY_FITS)
        for field in ("scientific_contract_changed","extra_recipe_or_hyperparameter_choice","new_lineup_assessment_metrics_or_intervals_read"):
            audit.check("recovery.authorization."+field,recovery_authorization[field],False)
        audit.check("recovery.authorization.inherited_fits",recovery_authorization["inherited_TRAIN_fits"],2)
        audit.check("recovery.authorization.original_requests",recovery_authorization["original_failed_requests"],2)
        for path,digest in recovery_authorization["parent_file_pins"].items():
            audit.check("recovery.authorized_parent."+path,sha(repo/path),digest,kind="artifact_hash")
        for relative,entry in recovery_authorization["old_runtime_snapshots"].items():
            audit.check("recovery.archived."+relative,sha(repo/entry["snapshot"]),entry["sha256"],kind="artifact_hash")
        for path,digest in recovery_authorization["protected_before"].items():
            audit.check("recovery.protected."+path,sha(repo/path),digest,kind="protected_hash")
    if recovery:
        if recovery_authorization is None:
            raise ValueError("recovery requires exact authorization")
        lineage = recovery_lineage_replay(audit,repo,roots,protocol,binding,recovery_authorization,recovery)
    audit.check("binding.protocol_sha256",sha(protocol_path),binding["protocol_sha256"],kind="artifact_hash")
    audit.check("binding.repo_root",protocol["repo_root"],str(repo))
    audit.check("binding.research_root",protocol["research_root"],str(research))
    pins = binding["pins"]
    audit.taxonomy_rules = read_taxonomy(repo/"src/models/competition_tiers.py")
    receipt["input_and_output_hashes"] = {}
    for name,pin in {**pins,**binding["import_origins"]}.items():
        actual = sha(pin["path"])
        runtime_pin_replay(audit,repo,name,pin,recovery_authorization,allow_archived=recovery is None and recovery_authorization is not None)
        receipt["input_and_output_hashes"][pin["path"]] = actual
    from importlib.metadata import version
    for name,expected in protocol["library_versions"].items():
        actual = version(name)
        audit.check("library_version."+name,actual,expected,kind="artifact_environment")
        receipt["environment_versions"][name] = actual
    authorization = load(pins["authorization_and_initial_pins.json"]["path"])
    frozen_at = datetime.fromisoformat(protocol["created_at"])
    authorized_at = datetime.fromisoformat(authorization["started_at"])
    audit.check("chronology.freeze_actual_authorized_utc",frozen_at.tzinfo is not None and frozen_at>=authorized_at,True)
    for path,expected in authorization["protected_before"].items():
        observed = sha(repo/path)
        audit.check("protected.before."+path,observed,expected,kind="protected_hash")
        receipt["protected_before_after"][path] = {"before":observed,"expected":expected}
    index_path = roots["08_reporting"]/("oracle_inputs_prepared.json" if args.mode=="prepare" and (roots["08_reporting"]/"oracle_inputs_prepared.json").is_file() else "oracle_inputs.json")
    index = load(index_path)
    receipt["oracle_contract_sha256"] = sha(pins["oracle_contract.json"]["path"])
    receipt["implementation_hashes"] = binding["import_origins"]
    audit.check("index.required_fields",sorted(set(INDEX_FIELDS)-set(index)),[])
    audit.check("index.study",index["study"],"R1")
    audit.check("index.run_id",index["run_id"],args.run_id)
    indexed_at = datetime.fromisoformat(index["created_at_utc"])
    audit.check("chronology.index_actual_utc",indexed_at.tzinfo is not None and frozen_at<=indexed_at<=datetime.now(timezone.utc),True)
    for key,path in (("protocol_sha256",protocol_path),("binding_sha256",binding_path),("oracle_contract_sha256",Path(pins["oracle_contract.json"]["path"]))):
        audit.check("index."+key,index[key],sha(path),kind="artifact_hash")
    import pyarrow.parquet as pq
    import zipfile
    descriptors = index["artifact_descriptors"]
    stop = partial_hard_stop(roots["08_reporting"],index)
    produced_paths = set()
    for descriptor in descriptors:
        audit.check("descriptor."+descriptor["role"]+".fields",sorted(set(DESCRIPTOR_FIELDS)-set(descriptor)),[])
        if descriptor["absence_status"] == "AVAILABLE":
            path = repo/descriptor["repo_relative_path"]
            audit.check("descriptor."+descriptor["role"]+".sha256",sha(path),descriptor["sha256"],kind="artifact_hash")
            receipt["input_and_output_hashes"][str(path)] = sha(path)
            produced_paths.add(path.resolve())
            if isinstance(descriptor["dtype_or_columns"],dict) and path.suffix==".parquet":
                metadata = pq.ParquetFile(path)
                audit.check("descriptor."+descriptor["role"]+".row_count",descriptor["shape_or_row_count"],metadata.metadata.num_rows)
                audit.check("descriptor."+descriptor["role"]+".columns",descriptor["dtype_or_columns"],{f.name:str(f.type) for f in metadata.schema_arrow})
            elif path.suffix==".npz":
                shapes,dtypes = {},{}
                with zipfile.ZipFile(path) as container:
                    for member in container.namelist():
                        if not member.endswith(".npy"):
                            raise ValueError("unexpected non-array member in immutable NPZ")
                        with container.open(member) as handle:
                            version = np.lib.format.read_magic(handle)
                            if version==(1,0):
                                shape,fortran,dtype = np.lib.format.read_array_header_1_0(handle)
                            elif version==(2,0):
                                shape,fortran,dtype = np.lib.format.read_array_header_2_0(handle)
                            else:
                                raise ValueError("unsupported NPZ header version")
                            shapes[member[:-4]],dtypes[member[:-4]] = list(shape),str(dtype)
                audit.check("descriptor."+descriptor["role"]+".array_shapes",descriptor["shape_or_row_count"],shapes)
                audit.check("descriptor."+descriptor["role"]+".array_dtypes",descriptor["dtype_or_columns"],dtypes)
        else:
            audit.check("descriptor."+descriptor["role"]+".absence_reason",bool(descriptor["absence_reason"]),True)
    for stage,path in roots.items():
        for artifact in path.rglob("*"):
            if artifact.is_file() and not artifact.name.startswith("oracle_inputs") and artifact.resolve()!=args.output.resolve():
                audit.check("index.describes."+str(artifact.relative_to(repo)),artifact.resolve() in produced_paths,True)
    canonical_fields = ["match_id","date","release_day","best_of","team_a","team_b","roster_a","roster_b"]
    available_fields = pq.read_schema(pins["canonical_dataset"]["path"]).names
    canonical_fields += [k for k in ("winner","y","tournament","tournament_name","competition_context","major","is_major","player_source_unseen","source_unseen") if k in available_fields]
    canonical = table(pins["canonical_dataset"]["path"],canonical_fields)
    for row in canonical:
        for field in ("match_id","team_a","team_b"):
            row[field] = ident(row[field])
        for field in ("date","release_day"):
            row[field] = day(row[field]).isoformat()
    audit.check("canonical.unique_ids",len({r["match_id"] for r in canonical}),len(canonical))
    audit.check("canonical.total",len(canonical),40636)
    if stop is not None and stop.get("command") == "prepare":
        partial_preparation_replay(audit,receipt,stop,repo,roots,index,canonical,pins,protocol,binding,research)
        return
    events,clocks,first,map_count,map_rows = source_replay(Path(pins["raw_source"]["path"]),load(pins["native_release_map"]["path"]),load(pins["canonical_release_override"]["path"]),{r["match_id"]:r for r in canonical})
    actual_events = table(roots["04_feature"]/"appearances.parquet")
    compare_rows(audit,"appearances",actual_events,events,("source_match_id","team_id","competition_context","roster","map_index","original_map_day","native_release_day","canonical_release_day","effective_release_day","release_source","effective_release_reason"))
    del actual_events
    actual_maps = table(roots["04_feature"]/"source_maps.parquet")
    actual_maps.sort(key=lambda r:(r["source_match_id"],r["map_index"],r["team_id"]))
    map_rows.sort(key=lambda r:(r["source_match_id"],r["map_index"],r["team_id"]))
    compare_rows(audit,"source_maps",actual_maps,map_rows,("source_match_id","map_index","team_id","competition_context","original_map_day","roster","native_release_day","canonical_release_day","effective_release_day"))
    del actual_maps, map_rows
    actual_clocks = table(roots["08_reporting"]/"clock_comparison.parquet")
    compare_rows(audit,"all_source_clocks",actual_clocks,clocks,tuple(clocks[0]))
    del actual_clocks
    rows,conflicts = opportunity_replay(canonical,table(pins["native_participants"]["path"]),events,first,clocks,audit)
    event_count = len(events)
    del events
    import gc; gc.collect()
    for row in rows:
        row["release_lineage"] = {"canonical_label_release_day":row["label_release_day"],"canonical_source_release_day":first[row["match_id"]]["canonical_release_day"],"native_release_day":row["native_release_day"],"source_sha256":pins["raw_source"]["sha256"],"native_release_map_sha256":pins["native_release_map"]["sha256"],"canonical_release_override_sha256":pins["canonical_release_override"]["sha256"]}
    actual_rows = table(roots["04_feature"]/"opportunities.parquet")
    for row in actual_rows:
        if isinstance(row["candidates"],str):
            row["candidates"] = json.loads(row["candidates"])
        if isinstance(row["release_lineage"],str):
            row["release_lineage"] = json.loads(row["release_lineage"])
        if row["label"] is not None:
            row["label"] = ("LATEST","PREVIOUS","OTHER").index(row["label"]) if isinstance(row["label"],str) else int(row["label"])
    compare_rows(audit,"opportunities",actual_rows,rows,("match_id","side","partition","date","month","best_of","team_id","opponent_id","competition_context","native_release_day","label_release_day","actual_roster","history_ids","incomplete_history_count","history_state","candidates","counts","ages","previous_available","label","label_absence","absence","major","player_source_unseen"))
    compare_rows(audit,"opportunities.release_lineage",actual_rows,rows,("release_lineage",))
    del actual_rows
    gc.collect()
    inventory_replay(audit,rows,load(roots["08_reporting"]/"opportunity_inventory.json"))
    compare_rows(audit,"all_origin_conflicts",table(roots["08_reporting"]/"origin_conflicts.parquet"),conflicts,("match_id","cutoff_day","conflict_source_ids","native_absence"))
    native_order = sorted(clocks,key=lambda c:(c["native_release_day"],order(c["source_match_id"])))
    expected_native_order = [{"native_rank":i,**clock} for i,clock in enumerate(native_order)]
    compare_rows(audit,"complete_native_source_order",table(roots["04_feature"]/"native_source_order.parquet"),expected_native_order,tuple(expected_native_order[0]))
    native_days = [c["native_release_day"] for c in native_order]
    for conflict in conflicts:
        conflict["native_consumed_prefix_length"] = bisect.bisect_left(native_days,conflict["cutoff_day"])
    compare_rows(audit,"all_origin_consumed_prefixes",table(roots["08_reporting"]/"origin_conflicts.parquet"),conflicts,("native_consumed_prefix_length",))
    side_lookup = {(r["match_id"],r["side"]):r for r in rows}
    expected_series = []
    for canonical_row in canonical:
        mid = str(canonical_row["match_id"])
        a,b = side_lookup[mid,"A"],side_lookup[mid,"B"]
        expected = {"match_id":mid,"roster_a_ids":a["actual_roster"],"roster_b_ids":b["actual_roster"],"lineup_label_availability":"ALL_SIDES" if a["label_absence"] is None and b["label_absence"] is None else "ABSENT" if a["label_absence"] is not None and b["label_absence"] is not None else "ONE_SIDE","lineup_label_absence_reasons":[a["label_absence"],b["label_absence"]]}
        for side in ("a","b"):
            roster = [ident(p) for p in canonical_row["roster_"+side]]
            expected["canonical_roster_"+side+"_ids"] = sorted(p for p in roster if p is not None)+[None]*roster.count(None)
        expected_series.append(expected)
    compare_rows(audit,"series_first_vs_canonical_rosters",table(roots["04_feature"]/"series.parquet"),expected_series,tuple(expected_series[0]))
    del expected_series, side_lookup
    gc.collect()
    support = load(roots["08_reporting"]/"support.json")
    counts,failed = support_replay(rows)
    for key,actual in support["counts"].items():
        audit.check("support.count."+key,actual,True if key=="identity_release_valid" else counts[key])
    audit.check("support.failed_floors",support["failed_floors"],failed)
    audit.check("support.status",support["status"],"INSUFFICIENT_FIT_SUPPORT" if failed else "READY_FOR_TWO_TRAIN_FITS")
    audit.check("support.initial_formal_fits",support["formal_fits"],0)
    partitions = Counter(r["partition"] for r in rows)
    audit.check("support.all_partition_opportunities",dict(partitions),{"TRAIN":72410,"CAL":2656,"ASSESS":6206})
    audit.check("support.absence_counts",support["absence_counts"],dict(Counter(r["absence"] or "AVAILABLE" for r in rows)))
    audit.check("support.source_series",support["source_series"],len(first))
    audit.check("support.source_maps",support["source_maps"],map_count)
    audit.check("support.clock_disagreements",support["clock_disagreements"],sum(c["disagreement"] for c in clocks))
    for partition in ("TRAIN","CAL","ASSESS"):
        expected_ids = list(dict.fromkeys(r["match_id"] for r in rows if r["partition"]==partition))
        audit.check("population."+partition,protocol["population_ids"][partition],expected_ids)
        audit.check("index.population."+partition,index["population_ids"][partition],expected_ids)
    trace_path = roots["08_reporting"]/("prepare_operation_trace.json" if args.mode=="prepare" and (roots["08_reporting"]/"prepare_operation_trace.json").is_file() else "operation_trace.json")
    trace = load(trace_path)
    audit.check("index.operation_trace",index["operation_trace"],trace)
    receipt["formal_operation_counts"] = {"TRAIN":len(trace),"CAL":0,"C0":0,"ASSESS":0,"restarts":0}
    if recovery:
        new_trace = load(roots["08_reporting"]/"new_operation_trace.json")
        audit.check("recovery.index.descriptor",index["recovery"],recovery)
        audit.check("recovery.index.new_trace",index["new_operation_trace"],new_trace)
        audit.check("recovery.index.inherited_fits",index["inherited_TRAIN_fits"],2)
        audit.check("recovery.index.new_fits",index["new_fit_counts"],ZERO_RECOVERY_FITS)
        audit.check("recovery.zero_new_fit_trace",new_trace,[])
        receipt["formal_operation_counts"].update(inherited_TRAIN=2,new_fit_counts=ZERO_RECOVERY_FITS)
    receipt["raw_state_count"] = 0
    receipt["population_counts"] = {"canonical_series":len(canonical),"canonical_sides":len(rows),"unique_source_series":len(first),"unique_source_maps":map_count,"historical_side_appearances":event_count}
    receipt["membership_counts"] = counts
    receipt["month_block_counts"] = {m:sum(r["side"]=="A" and r["partition"]=="ASSESS" and r["month"]==m for r in rows) for m in MONTHS}
    receipt["missingness_statuses"] = dict(Counter(r["absence"] or "AVAILABLE" for r in rows))
    receipt["scope_exercised"] += ["full raw source/map/final appearance/first label reconstruction","all conservative/native/canonical source clocks and all-origin contradictions","full canonical side opportunities, candidate ordering/counts/ages/labels/absence","exact support inventories and all protected/source/import hashes"]
    selected,states = selected_cases(rows)
    audit.check("prefit.public_cases.frozen_ids",sorted(c["match_id"] for c in load(roots["05_model_input"]/"public_cases.json")),selected)
    coverage = load(roots["05_model_input"]/"public_case_contracts.json")
    required_states = ("BO1","BO3","BO5","candidates_0","candidates_1","candidates_2","history_cold","history_partial","history_full","absence_AVAILABLE","absence_MISSING_EXACT_CONTEXT","absence_NO_EXACT_CONTEXT_HISTORY","absence_INCOMPLETE_LATEST_APPEARANCE")
    expected_states = {state:"R1_"+entry["match_id"] for entry in states for state in entry["states"]}
    audit.check("prefit.public_contract.coverage_keys",sorted(coverage["coverage"]),sorted(required_states))
    for state in required_states:
        case = coverage["coverage"][state]
        audit.check("prefit.public_contract."+state+".status",case["status"],"AVAILABLE" if state in expected_states else "ABSENT_IN_SOURCE")
        audit.check("prefit.public_contract."+state+".case_id",case["case_id"],expected_states.get(state))
        audit.check("prefit.public_contract."+state+".synthetic_reference",case["synthetic_contract_reference"] is None,state in expected_states)
    audit.check("prefit.public_contract.real_case_hash",coverage["real_cases_sha256"],sha(roots["05_model_input"]/"public_cases.json"),kind="artifact_hash")
    fixture = coverage["synthetic_fixture_source"]
    audit.check("prefit.public_contract.fixture_hash",fixture["sha256"],sha(fixture["path"]),kind="artifact_hash")
    audit.check("prefit.public_contract.fixture_is_not_observation",fixture["observed_population_support"],False)
    audit.check("prefit.public_contract.fixture_not_executed",fixture["status"],"INPUT_ONLY_NOT_EXECUTED")
    toy = coverage["bounds_format_inputs"]
    audit.check("prefit.public_contract.binary_rational_inputs",[toy["best_of"],toy["q_a"],toy["q_b"],toy["supported_native_probabilities"],toy["reversed_probabilities"]],[[1,3,5],[.5,.25,.25],[.5,.25,.25],{str(i)+","+str(j):.75 for i,j in np.ndindex(2,2)},{str(i)+","+str(j):.25 for i,j in np.ndindex(2,2)}],kind="toy_input_inventory")
    receipt["public_case_selection"] = states
    if stop is not None:
        partial_outputs_replay(audit,receipt,stop,repo,roots,index,trace,rows,protocol,binding,selected,clocks,pins,research,lineage,recovery)
    elif failed or args.mode == "prepare":
        audit.check("prepare.zero_fit_trace",trace,[])
        no_fit_outputs_replay(audit,roots)
        receipt["fitted_prediction_oracle_status"] = "NOT_RUN_NO_FIT" if failed else "NOT_RUN_PREPARE_ONLY"
        receipt["scope_not_exercised"] += ["saved fitted law, optimizer, native public replay and statistical performance: no fit"]
        receipt["primary_slot_statuses"] = {str(s):"NOT_RUN_BLOCKED" if failed else "NOT_RUN_PREPARE_ONLY" for s in (7,8)}
        if failed:
            decision = load(roots["08_reporting"]/"decision.json")
            audit.check("closure.decision",decision["status"],"INSUFFICIENT_FIT_SUPPORT")
            for field,expected in (("formal_fits",0),("completed_fits",0),("CAL_fits",0),("C0_fits",0),("ASSESS_fits",0),("unused_TRAIN_slots",2)):
                audit.check("closure."+field,decision[field],expected)
            audit.check("closure.slot_keys",sorted(decision["primary_slots"]),["7","8"])
            for slot in ("7","8"):
                audit.check("closure.slot."+slot,decision["primary_slots"][slot]["status"],"NOT_RUN_BLOCKED")
        receipt["status"] = "PASS_CLOSURE_CONTRACT" if failed else "PASS_PREPARE_CONTRACT"
    elif index["branch_status"] in ("INVALID_FIT","FIT_FAILED_WITHIN_BUDGET"):
        audit.check("failed_fit.budget",len(trace)<=2,True)
        budget_replay(audit,trace,rows,roots,protocol,binding)
        for arm in ("conditional","global"):
            model_path = roots["06_models"]/(arm+".json")
            if model_path.is_file():
                saved_fit_replay(audit,load(model_path),rows,arm=="conditional",arm,require_acceptance=False)
                receipt["raw_state_count"] += 1
        audit.check("failed_fit.last_attempt_failed",bool(trace) and trace[-1]["status"] in ("FAILED","INVALID_FIT","ATTEMPTED"),True)
        for slot in ("7","8"):
            decision = load(roots["08_reporting"]/"decision.json")
            audit.check("failed_fit.slot."+slot,decision["primary_slots"][slot]["status"],"NOT_RUN_FIT_FAILED")
        receipt["primary_slot_statuses"] = {"7":"NOT_RUN_FIT_FAILED","8":"NOT_RUN_FIT_FAILED"}
        receipt["fitted_prediction_oracle_status"] = "NOT_RUN_FIT_FAILED"
        receipt["scope_exercised"].append("failed fixed recipe retained state and immutable attempt/budget forensic replay")
        receipt["scope_not_exercised"].append("unproduced fitted laws/public native/contrasts: fixed recipe failed without restart")
        if len(trace)<2:
            audit.check("failed_fit.unused_global_has_no_state",(roots["06_models"]/"global.json").is_file(),False)
        receipt["status"] = "PASS_FAILED_FIT_CLOSURE_CONTRACT"
    else:
        receipt["fitted_prediction_oracle_status"] = "PENDING_REQUIRED_CHECKS"
        receipt["status"] = "PENDING_REQUIRED_CHECKS"
        audit.check("budget.exact_two_attempts",len(trace),2)
        for i,item in enumerate(trace):
            for field in ("operation_id","study","kind","arm","seed","partition","authorized_ceiling","counter_before","counter_after","started_at_utc","finished_at_utc","status","parameters_sha256","input_ids_sha256","iterations","function_evaluations","failure_reason"):
                audit.check("budget.trace."+str(i)+"."+field+".present",field in item,True)
            audit.check("budget.trace."+str(i)+".before",item["counter_before"],i)
            audit.check("budget.trace."+str(i)+".after",item["counter_after"],i+1)
            audit.check("budget.trace."+str(i)+".partition",item["partition"],"TRAIN")
        budget_replay(audit,trace,rows,*(lineage[:3] if recovery else (roots,protocol,binding)))
        models = {}
        for arm in ("conditional","global"):
            model_path = roots["06_models"]/(arm+".json")
            model = load(model_path)
            models[arm] = saved_fit_replay(audit,model,rows,arm=="conditional",arm)
            receipt["raw_state_count"] += 1
        law_cols = ("match_id","side","law","partition","label","absence","q_latest","q_previous","q_other","lineup_nll","other_brier")
        exported_laws = table(roots["07_model_output"]/"side_laws.parquet", columns=law_cols)
        for row in exported_laws:
            if isinstance(row.get("label"),str):
                row["label"] = ("LATEST","PREVIOUS","OTHER").index(row["label"])
        score_cols = ("match_id","eligible_side_count","conditional_nll","conditional_brier","global_nll","global_brier","partition")
        exported_laws_count = len(exported_laws)
        laws,scores = laws_and_scores(audit,rows,models,exported_laws,table(roots["07_model_output"]/"series_scores.parquet", columns=score_cols))
        del exported_laws
        gc.collect()
        metrics = load(roots["08_reporting"]/"metrics.json")
        with np.load(roots["08_reporting"]/"contrasts.npz",allow_pickle=False) as arrays:
            receipt["primary_slot_statuses"] = uncertainty_replay(audit,rows,scores,arrays,metrics)
        diagnostics_replay(audit,rows,laws,metrics["diagnostics"])
        cal = load(roots["08_reporting"]/"cal_audit.json")
        diagnostics_replay(audit,rows,laws,cal["diagnostics"],partitions=("CAL",))
        audit.check("CAL.once_only",cal["audit_number"],1)
        audit.check("CAL.no_fit",cal["formal_fits"],0)
        audit.check("CAL.classes",cal["class_counts"],dict(Counter(("LATEST","PREVIOUS","OTHER")[r["label"]] for r in rows if r["partition"]=="CAL" and r["label"] is not None)))
        public_contract_replay(audit,rows,models,selected)
        saved_public_replay(audit,args,rows,models,laws,selected,index)
        output_binding = load(roots["08_reporting"]/"saved_output_binding.json")
        expected_paths = {str(p) for stage in ("06_models","07_model_output") for p in roots[stage].rglob("*") if p.is_file()}
        audit.check("saved_output_binding.exact_paths",sorted(output_binding["hashes"]),sorted(expected_paths))
        for path,digest in output_binding["hashes"].items():
            audit.check("saved_output_binding."+path,sha(path),digest,kind="artifact_hash")
        pairs = table(roots["07_model_output"]/"native_pairs.parquet")
        blocks = table(roots["07_model_output"]/"native_blocked_pairs.parquet")
        execution = load(roots["08_reporting"]/"native_execution.json")
        inherited_pairs = recovery_native_replay(audit,receipt,repo,roots,lineage,recovery,pairs,blocks,execution) if recovery else []
        receipt["population_counts"].update(native_replay(audit,rows,laws,pairs,table(roots["07_model_output"]/"roster_bounds.parquet"),repo,selected,clocks,pins,blocks,research,inherited_pairs=inherited_pairs))
        bound_diagnostics_replay(audit,audit.bound_expectations,metrics["diagnostics"])
        receipt["population_counts"].update(unique_saved_feature_laws=len(laws),saved_side_law_records=exported_laws_count,unique_public_cases=len(selected),unique_primary_slots=2,unique_formal_guards=0)
        audit.check("metrics.portfolio_family",metrics["portfolio_slots"],10)
        if not recovery:
            native_execution_replay(audit,execution,pairs,blocks)
        audit.check("metrics.reserved_r1_slots",metrics["reserved_R1_slots"],[7,8])
        for arm in ("conditional","global"):
            model_path = roots["06_models"]/(arm+".json")
            audit.check("public.nonmutation.saved_model."+arm,sha(model_path),next(item["parameters_sha256"] for item in trace if item["arm"]==arm),kind="saved_state_nonmutation")
        receipt["independent_calculation_counts"] = {"saved_objective_gradient_reconstructions":2,"finite_difference_diagnostic_points":6,"optimizer_steps":0,"fits":0,"bootstrap_draws_per_contrast":10000}
        receipt["scope_exercised"] += ["all saved side softmax/likelihood/gradient/projected KKT and series scores","all nine-month paired draws and source-name/year proxy sensitivity","all native pair identities/tensors/input digests and unconditional missing-mass bounds","deterministic saved-public batch/singleton laws/candidates/abstentions"]
        if receipt["population_counts"]["native_public_calls"]:
            receipt["scope_exercised"].append("selected captured native singleton and A/B public parity with owned worker closure")
        else:
            receipt["scope_not_exercised"].append("native public replay: no available pairs among frozen selected real cases")
        receipt["scope_not_exercised"] += ["independent reconstruction of the complete protected native learned network","native worker replay outside the frozen <=32-target public case set","source authenticity, genuine prospective issuance, unconditional uncertain-case scalar winner scoring or business/deployment certification"]
        decision = load(roots["08_reporting"]/"decision.json")
        passing = all(s["passes_scientific_gate"] for s in receipt["primary_slot_statuses"].values())
        estimable = all(s["status"]=="ESTIMABLE" for s in receipt["primary_slot_statuses"].values())
        audit.check("decision.scientific_status",decision["status"],"LINEUP_LAW_DEVELOPMENT_SUPPORTED_NOT_WINNER_MODEL" if passing else "NO_LINEUP_LAW_GAIN" if estimable else "INCONCLUSIVE")
    final_integrity_replay(audit,receipt,repo,roots,pins,binding,recovery_authorization if recovery is None else None)
    receipt["guard_statuses"] = {"R1":"NO_FORMAL_GUARD_FAMILY"}
    receipt["comparison_population_counting"] = {"unique_series":len(canonical),"unique_sides":len(rows),"unique_source_series":len(first),"unique_source_maps":map_count,"repeated_numeric_elements":audit.numeric_elements,"repeated_check_entries":len(audit.checks),"draws_are_not_independent_observations":True}
    receipt["comparison_population_counting"]["unique_artifacts"] = len(receipt["input_and_output_hashes"])
    receipt["comparison_population_counting"]["unique_real_public_cases"] = len(selected) if receipt["raw_state_count"]==2 else 0
    receipt["comparison_population_counting"]["synthetic_toy_checks"] = sum(c["kind"].startswith("toy_") for c in audit.checks)
    finalize_statuses(audit,receipt)


def read_taxonomy(path):
    """Read declarative regex/date rules as data, without executing classifier code."""
    tree = ast.parse(Path(path).read_text())
    declarations = next(node.value for node in tree.body if isinstance(node,ast.AnnAssign) and isinstance(node.target,ast.Name) and node.target.id=="_RULES")
    output = []
    for call in declarations.elts:
        if not isinstance(call,ast.Call) or not isinstance(call.func,ast.Name) or call.func.id!="_rule":
            raise ValueError("unrecognized frozen taxonomy declaration")
        pattern = ast.literal_eval(call.args[0])
        tier = call.args[2].attr.lower()
        dates = {}
        for keyword in call.keywords:
            if keyword.arg in ("before","not_before"):
                dates[keyword.arg] = date(*(ast.literal_eval(a) for a in keyword.value.args))
        output.append((re.compile(pattern),tier,dates))
    return output


def taxonomy_tier(name,origin,rules):
    if name is None:
        return "unknown"
    decomposed = unicodedata.normalize("NFKD",str(name)).casefold()
    text = re.sub("[^a-z0-9]+"," ","".join(c for c in decomposed if not unicodedata.combining(c))).strip()
    if text in ("","na","nan","nat","none","null"):
        return "unknown"
    for pattern,tier,dates in rules:
        if dates.get("not_before",date.min) <= origin < dates.get("before",date.max) and pattern.search(text):
            return tier
    return "unknown"


def summary_values(values):
    if not values:
        return {"n":0,"mean":None,"min":None,"max":None,"quantiles":None,"reason":"NO_OBSERVATIONS"}
    values = np.asarray(values,float)
    return {"n":len(values),"mean":float(values.mean()),"min":float(values.min()),"max":float(values.max()),"quantiles":dict(zip(("p0","p10","p25","p50","p75","p90","p100"),np.quantile(values,[0,.1,.25,.5,.75,.9,1],method="linear").tolist())),"reason":None}


def recursive_compare(audit,name,actual,expected):
    if isinstance(expected,dict):
        audit.check(name+".keys",sorted(actual),sorted(str(k) for k in expected))
        for key,value in expected.items():
            recursive_compare(audit,name+"."+str(key),actual[str(key)],value)
    elif isinstance(expected,list) and expected and isinstance(expected[0],dict):
        audit.check(name+".length",len(actual),len(expected))
        for i,value in enumerate(expected):
            recursive_compare(audit,name+"."+str(i),actual[i],value)
    elif isinstance(expected,float):
        audit.check(name,actual,expected,atol=1e-10,rtol=1e-10,kind="secondary_diagnostic")
    else:
        audit.check(name,actual,expected,kind="secondary_diagnostic")


def diagnostics_replay(audit,rows,laws,actual,partitions=("TRAIN","CAL","ASSESS")):
    for partition in partitions:
        all_rows = [r for r in rows if r["partition"]==partition]
        eligible = [r for r in all_rows if r["label"] is not None]
        classes = Counter(("LATEST","PREVIOUS","OTHER")[r["label"]] for r in eligible)
        for arm in ("conditional","global"):
            qs = [laws[r["match_id"],r["side"],arm] for r in eligible]
            nll = [-math.log(q[r["label"]]) for r,q in zip(eligible,qs)]
            brier = [(q[2]-(r["label"]==2))**2 for r,q in zip(eligible,qs)]
            reliability = []
            for bin_id in range(10):
                selected = [(r,q) for r,q in zip(eligible,qs) if min(9,int(q[2]*10))==bin_id]
                reliability.append({"bin":bin_id,"n_sides":len(selected),"mean_q_other":float(np.mean([q[2] for _,q in selected])) if selected else None,"realized_other":float(np.mean([r["label"]==2 for r,_ in selected])) if selected else None,"absence_reason":None if selected else "EMPTY_FIXED_BIN"})
            expected = {"opportunity_sides":len(all_rows),"eligible_sides":len(eligible),"eligible_series":len({r["match_id"] for r in eligible}),"class_counts":dict(classes),"absence_counts":dict(Counter(r["absence"] or "AVAILABLE" for r in all_rows)),"candidate_recall":float(np.mean([r["label"]!=2 for r in eligible])) if eligible else None,"realized_other_rate":float(np.mean([r["label"]==2 for r in eligible])) if eligible else None,"mean_nll_side":float(np.mean(nll)) if nll else None,"mean_other_brier_side":float(np.mean(brier)) if brier else None,"other_reliability":reliability,"candidate_counts":{str(k):v for k,v in Counter(len(r["candidates"]) for r in all_rows).items()},"history_lengths":{str(k):v for k,v in Counter(len(r["history_ids"]) for r in all_rows).items()},"incomplete_history_counts":{str(k):v for k,v in Counter(r["incomplete_history_count"] for r in all_rows).items()},"release_age_days":summary_values([c["age_days"] for r in all_rows for c in r["candidates"]]),"played_age_days":summary_values([(day(r["date"])-day(c["last_played_day"])).days for r in all_rows for c in r["candidates"]])}
            diag = dict(actual[partition+":"+arm])
            if "absence_counts" in diag and "nan" in diag["absence_counts"]:
                counts_copy = dict(diag["absence_counts"])
                counts_copy["AVAILABLE"] = counts_copy.pop("nan")
                diag["absence_counts"] = counts_copy
            diag.pop("mean_nll_series", None)
            diag.pop("mean_other_brier_series", None)
            recursive_compare(audit,"diagnostics."+partition+"."+arm,diag,expected)


def saved_public_replay(audit,args,rows,models,laws,selected,index):
    from src.analysis.lineup_law_reliability import predict_run
    lookup = {(r["match_id"],r["side"]):r for r in rows}
    bound_path = args.repo_root/"data/07_model_output/lineup_law_reliability"/args.run_id/"roster_bounds.parquet"
    saved_bounds = {(r["match_id"],r["law"]):r for r in table(bound_path)}
    audit.check("public_cases.lexical_ids",sorted(c["match_id"] for c in index["public_cases"]),selected)
    for arm,theta in models.items():
        batch = predict_run(args.run_id,selected,law=arm,repo_root=args.repo_root,research_root=args.research_root)
        audit.check("public_saved."+arm+".batch_ids",[r["match_id"] for r in batch],selected)
        for actual in batch:
            mid = actual["match_id"]
            singleton = predict_run(args.run_id,[mid],law=arm,repo_root=args.repo_root,research_root=args.research_root)[0]
            audit.check("public_saved."+arm+"."+mid+".singleton_batch",singleton,actual,kind="public_saved_actual")
            expected_bound = saved_bounds[mid,arm]
            for field in ("lower","upper","supported_mass","missing_mass","prob_a"):
                actual_value,expected_value = actual["bounds"].get(field),expected_bound[field]
                audit.check("public_saved."+arm+"."+mid+".bounds."+field,actual_value,expected_value,
                            atol=1e-12 if expected_value is not None else None,rtol=1e-12,kind="public_saved_actual")
            for side_result in actual["side_laws"]:
                side = side_result["side"]
                row = lookup[mid,side]
                audit.check("public_saved."+arm+"."+mid+side+".absence",side_result["absence"],row["state"]["absence"])
                for field in ("history_ids","incomplete_history_count","candidates","absence"):
                    audit.check("public_saved."+arm+"."+mid+side+".state."+field,side_result["state"][field],row["state"][field])
                if row["state"]["absence"]:
                    audit.check("public_saved."+arm+"."+mid+side+".law",side_result["q"],None)
                else:
                    audit.check("public_saved."+arm+"."+mid+side+".law",side_result["q"],laws[mid,side,arm],atol=1e-12,rtol=1e-12,kind="public_saved_actual")
            if actual["bounds"].get("missing_mass",0)>0:
                audit.check("public_saved."+arm+"."+mid+".no_scalar",actual["bounds"]["prob_a"],None)


def budget_replay(audit,trace,rows,roots,protocol,binding):
    train = [r for r in rows if r["partition"]=="TRAIN" and r["label"] is not None]
    training_keys = [[r["match_id"],r["side"]] for r in train]
    expected_input_hash = hashlib.sha256(json.dumps(training_keys).encode()).hexdigest()
    audit.check("budget.attempt_ceiling",len(trace)<=2,True)
    for i,item in enumerate(trace):
        if i >= 2:
            break
        for field in ("operation_id","study","kind","arm","seed","partition","authorized_ceiling","counter_before","counter_after","started_at_utc","finished_at_utc","status","parameters_sha256","input_ids_sha256","iterations","function_evaluations","failure_reason"):
            audit.check("budget.prefix."+str(i)+"."+field+".present",field in item,True)
        audit.check("budget.prefix."+str(i)+".status",item["status"] in ("COMPLETED","FAILED","INVALID_FIT","ATTEMPTED","INTERRUPTED_OR_FAILED"),True)
        arm = ("conditional","global")[i]
        for field,value in (("operation_id","R1_TRAIN_"+arm),("study","R1"),("kind","bounded_TRAIN_lineup_law"),("arm",arm),("partition","TRAIN"),("seed",None),("authorized_ceiling",2),("counter_before",i),("counter_after",i+1),("input_ids_sha256",expected_input_hash)):
            audit.check("budget."+arm+"."+field,item[field],value)
        attempt = load(roots["08_reporting"]/(arm+"_attempt.json"))
        for field in ("started_at_utc","counter_before","counter_after","input_ids_sha256","operation_id"):
            audit.check("budget."+arm+".immutable_attempt."+field,attempt[field],item[field])
        audit.check("budget."+arm+".before_invoke_status",attempt["status"],"ATTEMPTED")
        audit.check("budget."+arm+".before_invoke_unfinished",attempt["finished_at_utc"],None)
        model_path = roots["06_models"]/(arm+".json")
        if model_path.is_file():
            model = load(model_path)
            audit.check("budget."+arm+".parameter_digest",item["parameters_sha256"],sha(model_path),kind="artifact_hash")
            audit.check("budget."+arm+".frozen_inputs",model["frozen_input_hashes"],protocol["hashes"])
            audit.check("budget."+arm+".code_pins",model["code_hashes"],protocol["import_origins"])
            audit.check("budget."+arm+".protocol",model["protocol_sha256"],binding["protocol_sha256"])
            clocks = [datetime.fromisoformat(v) for v in (attempt["started_at_utc"],model["started_at_utc"],model["finished_at_utc"],item["finished_at_utc"])]
            audit.check("budget."+arm+".recorded_before_optimizer",all(c.tzinfo is not None for c in clocks) and all(a<=b for a,b in zip(clocks,clocks[1:])),True)
            audit.check("budget."+arm+".iterations",item["iterations"],model["iterations"])
            audit.check("budget."+arm+".function_evaluations",item["function_evaluations"],model["function_evaluations"])
    if trace:
        decision = load(roots["08_reporting"]/"decision.json")
        for field,value in (("formal_fits",len(trace)),("completed_fits",sum(r["status"]=="COMPLETED" for r in trace)),("CAL_fits",0),("C0_fits",0),("ASSESS_fits",0),("unused_TRAIN_slots",2-len(trace))):
            audit.check("decision.budget."+field,decision[field],value)


def bound_diagnostics_replay(audit,bounds,actual):
    for law in ("conditional","global","latest_point"):
        rows = [r for r in bounds if r["law"]==law]
        valid = [r for r in rows if r["missing_mass"] is not None]
        expected = {"opportunities":len(rows),"valid_series":len(valid),"absence_counts":dict(Counter(r["absence"] or "AVAILABLE" for r in rows)),"missing_mass":summary_values([r["missing_mass"] for r in valid]),"width":summary_values([r["upper"]-r["lower"] for r in valid]),"scalar_available":sum(r["prob_a"] is not None for r in rows),"scalar_absence_counts":dict(Counter(r["scalar_absence"] or r["absence"] or "AVAILABLE" for r in rows))}
        profiles = {}
        for dimension in ("month","best_of","competition_context","major","player_source_unseen","candidate_pattern","history_pattern","incomplete_pattern","absence"):
            grouped = defaultdict(list)
            for row in rows:
                grouped[str(row.get(dimension) or "AVAILABLE_OR_UNKNOWN")].append(row)
            profiles[dimension] = {}
            for key,members in grouped.items():
                available = [r for r in members if r["missing_mass"] is not None]
                profiles[dimension][key] = {"series":len(members),"valid_series":len(available),"width":summary_values([r["upper"]-r["lower"] for r in available]),"missing_mass":summary_values([r["missing_mass"] for r in available]),"scalar_available":sum(r["prob_a"] is not None for r in members),"absences":dict(Counter(r["absence"] or "AVAILABLE" for r in members))}
        expected["profiles"] = profiles
        recursive_compare(audit,"bounds.diagnostics."+law,actual["bounds:"+law],expected)


def inventory_replay(audit,rows,actual):
    expected_cells = {}
    for dimension in ("partition","month","best_of","competition_context","major","player_source_unseen","history_state","incomplete_history_count","previous_available","absence","label_absence"):
        grouped = defaultdict(list)
        for row in rows:
            grouped[str(row[dimension]) if row[dimension] is not None else "UNKNOWN_OR_NONE"].append(row)
        cells = {}
        for key,members in grouped.items():
            eligible = [r for r in members if r["label"] is not None]
            cells[key] = {"opportunity_sides":len(members),"parent_series":len({r["match_id"] for r in members}),"eligible_sides":len(eligible),"eligible_series":len({r["match_id"] for r in eligible}),"class_counts":dict(Counter(("LATEST","PREVIOUS","OTHER")[r["label"]] for r in eligible)),"all_nine_month_counts":{m:sum(r["month"]==m for r in members) for m in MONTHS},"candidate_recall":float(np.mean([r["label"]!=2 for r in eligible])) if eligible else None}
        expected_cells[dimension] = cells
    recursive_compare(audit,"inventory.cells",actual["cells"],expected_cells)
    for field,expected in (("opportunity_sides",len(rows)),("one_two_absent_candidates",{str(k):v for k,v in Counter(len(r["candidates"]) for r in rows).items()}),("window_appearance_counts",{str(k):v for k,v in Counter(len(r["history_ids"]) for r in rows).items()}),("latest_incomplete_withheld",sum(r["absence"]=="INCOMPLETE_LATEST_APPEARANCE" for r in rows))):
        audit.check("inventory."+field,actual[field],expected,kind="inventory")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode",choices=("prepare","completed"))
    parser.add_argument("--run-id",required=True)
    parser.add_argument("--repo-root",type=Path,default=ROOT)
    parser.add_argument("--research-root",type=Path)
    parser.add_argument("--recovery-authorization",type=Path,help="Exact archived-runtime authorization, also for original hard-stop audit")
    parser.add_argument("--output",type=Path,required=True,help="Fresh JSON receipt path; never overwritten")
    args = parser.parse_args(argv)
    if not re.fullmatch(r"[A-Za-z0-9_-]+",args.run_id):
        parser.error("run-id must be a safe single path component")
    # Reserve before any work, including an exception or interrupt; no overwrite.
    args.output.parent.mkdir(parents=True,exist_ok=True)
    handle = args.output.open("x",encoding="utf-8")
    audit = Audit()
    receipt = {"schema_version":1,"run_id":args.run_id,"study":"R1","status":"FAIL","execution_created_at_utc":datetime.now(timezone.utc).isoformat(),"oracle_contract_sha256":None,"verifier_code_hashes":{str(Path(__file__).resolve()):sha(__file__)},"input_and_output_hashes":{},"implementation_hashes":{},"environment_versions":{"python":platform.python_version(),"numpy":np.__version__,"pandas":pd.__version__},"hardware_and_threads":{"platform":platform.platform(),"cpu_only":True,"thread_environment":{k:os.environ.get(k) for k in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS")}},"scope_exercised":[],"scope_not_exercised":[],"formal_operation_counts":{},"independent_calculation_counts":{"fits":0,"optimizer_steps":0},"raw_state_count":0,"population_counts":{},"membership_counts":{},"month_block_counts":{},"primary_slot_statuses":{},"guard_statuses":{},"missingness_statuses":{},"protected_before_after":{},"scientific_limitations":LIMITS}
    receipt["environment_versions"]["scipy"] = scipy.__version__
    try:
        toy_checks(audit)
        run_oracle(args,audit,receipt)
    except (Exception,KeyboardInterrupt) as error:
        audit.check("fatal.oracle_contract",{"exception":type(error).__name__,"message":str(error)},None,kind="hard_stop")
    finally:
        receipt["checks"],receipt["failures"] = audit.checks,audit.failures
        receipt["comparisons_by_kind"] = dict(audit.kinds)
        receipt["max_absolute_and_relative_differences"] = {kind:{"absolute":max((c["maximum_absolute_difference"] for c in audit.checks if c["kind"]==kind),default=0.),"relative":max((c["maximum_relative_difference"] for c in audit.checks if c["kind"]==kind),default=0.)} for kind in audit.kinds}
        finalize_statuses(audit,receipt)
        json.dump(safe(receipt),handle,indent=2,allow_nan=False)
        handle.write("\n")
        handle.close()
    return 1 if audit.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
