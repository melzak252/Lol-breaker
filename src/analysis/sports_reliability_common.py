"""Paired whole-block uncertainty for the authorized sports studies only.

Empty subgroup blocks remain in the population. Numerical agreement with an
expanded-row oracle is tolerance-based, not a floating-point bitwise claim.
"""
from functools import lru_cache
import hashlib
import math
from numbers import Integral

import numpy as np


@lru_cache(maxsize=4)
def _sampling_counts(blocks, draws, seed):
    """Cache immutable multiplicities and the exact int64 index-stream hash."""
    rng = np.random.default_rng(seed)
    digest = hashlib.sha256()
    counts = np.zeros((draws, blocks), dtype=np.int64)
    sampled = np.empty((draws, blocks), dtype=np.int64)
    if blocks:
        batch = max(1, 1_000_000 // blocks)
        for start in range(0, draws, batch):
            end = min(draws, start + batch)
            indices = rng.integers(0, blocks, size=(end - start, blocks), dtype=np.int64)
            digest.update(indices.tobytes(order='C'))
            sampled[start:end] = indices
            # Flattened row/block bins avoid a draws x blocks x blocks tensor.
            indices += np.arange(end - start, dtype=np.int64)[:, None] * blocks
            counts[start:end] = np.bincount(
                indices.ravel(), minlength=(end - start) * blocks
            ).reshape(end - start, blocks)
    # A bytes-backed view cannot be made writable by a subsequent caller.
    frozen = np.frombuffer(counts.tobytes(order='C'), dtype=np.int64).reshape(draws, blocks)
    frozen_indices = np.frombuffer(sampled.tobytes(order='C'), dtype=np.int64).reshape(draws, blocks)
    return frozen, frozen_indices, digest.hexdigest()


def _integer(name, value, minimum):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')
    return int(value)


def _labels(values, name):
    array = np.asarray(values, dtype=object)
    if array.ndim != 1:
        raise ValueError(f'{name} must be one-dimensional')
    result = []
    for value in array:
        if isinstance(value, np.generic):
            value = value.item()
        try:
            hash(value)
            valid = value is not None and bool(value == value)
        except (TypeError, ValueError):
            valid = False
        if not valid:
            raise ValueError(f'{name} must contain nonmissing hashable scalar block identities')
        if isinstance(value, (float, complex)) and not np.isfinite(value):
            raise ValueError(f'{name} must contain finite block identities')
        result.append(value)
    return result


def _sum(values):
    # Sorting makes the reduction independent of input row order. fsum reduces
    # cancellation error; it does not promise cross-platform bitwise identity.
    try:
        value = math.fsum(sorted(values))
    except OverflowError:
        return None
    return value if math.isfinite(value) else None


def contrast_draw_arrays(differences, block_labels, *, mask=None, universe=None,
                         draws=10000, seed=20261009):
    """Export the same whole-block calculation used by paired_block_contrast.

    Raw arrays retain all replicates; undefined deltas and empty block means
    are NaN (persist in NPZ, not JSON). Sampling arrays are immutable cached
    views. Block summaries and nonfinite counts preserve invalidity evidence.
    """
    try:
        values = np.asarray(differences, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError('differences must be a one-dimensional numeric array') from exc
    labels = _labels(block_labels, 'block_labels')
    if values.ndim != 1 or len(values) != len(labels):
        raise ValueError('differences and block_labels must be one-dimensional with equal lengths')
    if mask is None:
        selected = np.ones(len(values), dtype=bool)
    else:
        selected = np.asarray(mask)
        if selected.ndim != 1 or len(selected) != len(values) or selected.dtype.kind != 'b':
            raise ValueError('mask must be a one-dimensional boolean array matching differences')
    draws = _integer('draws', draws, 1)
    seed = _integer('seed', seed, 0)
    if universe is None:
        try:
            blocks = sorted(set(labels))
        except TypeError as exc:
            raise ValueError('default block identities must be sortable; supply an explicit universe') from exc
    else:
        blocks = _labels(universe, 'universe')
        if len(set(blocks)) != len(blocks):
            raise ValueError('universe must contain unique block identities')
    lookup = {block: index for index, block in enumerate(blocks)}
    if any(label not in lookup for label in labels):
        raise ValueError('universe must contain every full-cohort block label, including unselected rows')
    grouped = [[] for _ in blocks]
    for value, label, keep in zip(values, labels, selected):
        if keep:
            grouped[lookup[label]].append(float(value))
    sizes = np.array([len(group) for group in grouped], dtype=np.int64)
    occupied = sizes > 0
    nonfinite = int(np.count_nonzero(~np.isfinite(values[selected])))
    sums = [None if any(not math.isfinite(v) for v in group) else _sum(group)
            for group in grouped]
    means = [total / len(group) if group and total is not None else None
             for total, group in zip(sums, grouped)]
    counts, indices, digest = _sampling_counts(len(blocks), draws, seed)
    denominators = counts @ sizes
    equal_denominators = counts @ occupied.astype(np.int64)
    undefined = denominators == 0
    block_sums = np.asarray([s if s is not None else np.nan for s in sums], dtype=float)
    block_means = np.asarray([m if m is not None else np.nan for m in means], dtype=float)
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        weighted = (counts @ block_sums) / denominators
        equal = (counts @ np.where(occupied, block_means, 0.)) / equal_denominators
    weighted[undefined] = np.nan
    equal[undefined] = np.nan
    return dict(
        universe=tuple(blocks), draw_indices=indices, draw_counts=counts,
        weighted_denominator=denominators, equal_denominator=equal_denominators,
        weighted_delta=weighted, equal_delta=equal, undefined=undefined,
        weighted_undefined=undefined, equal_undefined=equal_denominators == 0,
        block_counts=sizes, block_sums=block_sums, block_means=block_means,
        nonfinite_selected_rows=nonfinite, bootstrap_index_sha256=digest,
        block_summary=[dict(block=block, count=int(size), sum=total, mean_or_null=mean)
                       for block, size, total, mean in zip(blocks, sizes, sums, means)])


def paired_block_contrast(differences, block_labels, *, mask=None, universe=None,
                          confidence=0.995, one_sided=False, min_rows=200,
                          min_blocks=6, draws=10000, seed=20261009):
    """Return the exact common-contract schema without embedding raw arrays."""
    try:
        confidence = float(confidence)
    except (TypeError, ValueError) as exc:
        raise ValueError('confidence must be finite and strictly between 0 and 1') from exc
    if not math.isfinite(confidence) or not 0 < confidence < 1:
        raise ValueError('confidence must be finite and strictly between 0 and 1')
    if not isinstance(one_sided, (bool, np.bool_)):
        raise ValueError('one_sided must be boolean')
    one_sided = bool(one_sided)
    min_rows = _integer('min_rows', min_rows, 0)
    min_blocks = _integer('min_blocks', min_blocks, 0)
    draws = _integer('draws', draws, 1)
    seed = _integer('seed', seed, 0)
    raw = contrast_draw_arrays(differences, block_labels, mask=mask,
                               universe=universe, draws=draws, seed=seed)
    blocks = raw['universe']
    sizes = raw['block_counts']
    occupied = sizes > 0
    nonfinite = raw['nonfinite_selected_rows']
    sums = [entry['sum'] for entry in raw['block_summary']]
    means = [entry['mean_or_null'] for entry in raw['block_summary']]
    digest = raw['bootstrap_index_sha256']
    undefined = int(np.count_nonzero(raw['undefined']))
    n_rows, n_blocks = int(sizes.sum()), int(occupied.sum())
    result = dict(
        status='ESTIMABLE', reason=None, n_rows=n_rows, n_blocks=n_blocks,
        n_universe_blocks=len(blocks), point=None, lower=None, upper=None,
        equal_block=dict(point=None, lower=None, upper=None),
        undefined_draws=undefined, nonfinite_selected_rows=nonfinite,
        confidence=confidence, one_sided=one_sided, draws=draws, seed=seed,
        bootstrap_index_sha256=digest,
        block_summary=raw['block_summary'],
        min_rows=min_rows, min_blocks=min_blocks)
    if nonfinite:
        result.update(status='NOT_ESTIMABLE', reason='nonfinite selected loss differences')
        return result
    if any(total is None for total in sums):
        result.update(status='NOT_ESTIMABLE', reason='nonfinite block loss aggregate')
        return result
    if not n_rows:
        result.update(status='NOT_ESTIMABLE', reason='empty selection')
        return result
    total = _sum(sums)
    equal_total = _sum([mean for mean in means if mean is not None])
    if total is None or equal_total is None:
        result.update(status='NOT_ESTIMABLE', reason='nonfinite point aggregate')
        return result
    result['point'] = total / n_rows
    result['equal_block']['point'] = equal_total / n_blocks
    if undefined:
        result.update(status='NOT_ESTIMABLE', reason='zero eligible denominator in bootstrap draws')
        return result
    weighted = raw['weighted_delta']
    equal = raw['equal_delta']
    if not np.all(np.isfinite(weighted)) or not np.all(np.isfinite(equal)):
        result.update(status='NOT_ESTIMABLE', reason='nonfinite bootstrap aggregate')
        return result
    quantiles = [confidence] if one_sided else [(1 - confidence) / 2, 1 - (1 - confidence) / 2]
    for output, samples in ((result, weighted), (result['equal_block'], equal)):
        with np.errstate(over='ignore', invalid='ignore'):
            bounds = np.quantile(samples, quantiles, method='linear')
        if not np.all(np.isfinite(bounds)):
            result.update(status='NOT_ESTIMABLE', reason='nonfinite bootstrap quantile')
            result['lower'] = result['upper'] = None
            result['equal_block']['lower'] = result['equal_block']['upper'] = None
            return result
        output['upper'] = float(bounds[-1])
        if not one_sided:
            output['lower'] = float(bounds[0])
    if n_rows < min_rows or n_blocks < min_blocks:
        result.update(status='INSUFFICIENT_SUPPORT',
                      reason=f'support below floors: rows {n_rows}/{min_rows}, occupied blocks {n_blocks}/{min_blocks}')
    return result
