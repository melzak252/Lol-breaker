#!/usr/bin/env python3
"""Render supplied offline validation evidence without fitting or selecting models."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import html
import json
import math
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.models.tournament_prediction import file_digest, json_digest
from scripts.score_tournament_forecasts import LEDGER_COHORT


def load(path):
    return json.loads(Path(path).read_text())


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _table(title, headers, rows):
    def cell(value):
        if isinstance(value, (dict, list)):
            value = json.dumps(value, sort_keys=True, allow_nan=False)
        return html.escape(str(value) if value is not None else 'unavailable')

    body = ''.join('<tr>' + ''.join(f'<td>{cell(value)}</td>' for value in row) + '</tr>' for row in rows)
    return (f'<details><summary>{html.escape(title)}</summary><div class="scroll"><table><thead><tr>'
            + ''.join(f'<th>{html.escape(name)}</th>' for name in headers)
            + '</tr></thead><tbody>' + body + '</tbody></table></div></details>')


def _interval_cells(row, interval):
    return [interval['ci_low'], interval['ci_high'], interval['confidence_level'], interval['status'],
            row.get('cells', row.get('targets')), row.get('editions'), row.get('occupied_months'),
            row.get('editions_with_unknown_start')]


def _plot_interval(ax, x, interval, offset=0):
    if finite(interval['ci_low']) and finite(interval['ci_high']):
        ax.vlines(x, offset + interval['ci_low'], offset + interval['ci_high'], color='tab:orange', linewidth=2)


def render(output, *, readiness, scores, benchmark, oracles, precision=None, admission=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    inputs = [Path(readiness), Path(benchmark)/'benchmark_summary.json', Path(oracles)/'results.json',
              *map(Path, scores)]
    if precision:
        inputs.append(Path(precision))
    if admission:
        inputs.append(Path(admission))
    pinned = {str(path.resolve()): file_digest(path) for path in inputs}
    figures = []
    score_tables = []
    interval_headers = ('interval low', 'interval high', 'confidence level', 'interval status',
                        'cells / targets', 'edition IDs', 'occupied months', 'unknown-start edition IDs')

    def save(fig, title, qualification, *, source=None):
        source_path = str(Path(source).resolve()) if source is not None else None
        identifier = json_digest([title, qualification, source_path, pinned.get(source_path)])[:16]
        name = f'{identifier}.png'
        fig.suptitle(title, fontsize=11, wrap=True)
        fig.text(.02, .01, qualification, fontsize=8, wrap=True)
        fig.tight_layout(rect=(0, .06, 1, .95))
        fig.savefig(output/name, dpi=145)
        plt.close(fig)
        figures.append({'file': name, 'title': title, 'qualification': qualification})

    inventory = load(readiness)
    families = sorted({row['source_event']['family'] for row in inventory['phases']})
    statuses = sorted({row['status'] for row in inventory['phases']})
    counts = Counter((row['source_event']['family'], row['status']) for row in inventory['phases'])
    fig, ax = plt.subplots(figsize=(10, 5))
    matrix = np.array([[counts[family, status] for status in statuses] for family in families])
    ax.imshow(matrix, cmap='Blues', aspect='auto')
    ax.set_xticks(range(len(statuses)), statuses, rotation=25, ha='right')
    ax.set_yticks(range(len(families)), families)
    for y, x in np.ndindex(matrix.shape):
        ax.text(x, y, str(matrix[y, x]), ha='center', va='center')
    save(fig, f"Source readiness: all {len(inventory['phases'])} phases",
         'Inventory statuses only; source-linked clusters are not verified whole editions. Admission evidence remains separate.')

    if admission:
        admitted = load(admission)
        scopes = ('phase_start', 'remaining_phase', 'full_tournament', 'remaining_tournament')
        matrix = np.array([[sum(row['family'] == family and row['scopes'][scope]['eligible']
                                for row in admitted['phases']) for scope in scopes] for family in families])
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(matrix, cmap='Blues', aspect='auto')
        ax.set_xticks(range(len(scopes)), scopes, rotation=25, ha='right')
        ax.set_yticks(range(len(families)), families)
        for y, x in np.ndindex(matrix.shape):
            denominator = sum(row['family'] == families[y] for row in admitted['phases'])
            ax.text(x, y, f'{matrix[y, x]}/{denominator}', ha='center', va='center')
        save(fig, 'Source-backed admission by scope — full inventory denominator',
             'Rules/state admission only, before model-feature and numerical gates. Unknown whole-edition boundaries remain excluded.')

    summary = load(Path(benchmark)/'benchmark_summary.json')
    for cohort, key in (('full_history', 'overall'), ('2024_plus', '2024_plus')):
        rows = list(summary[key].values())
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        for ax, metric in zip(axes, ('log_loss', 'brier')):
            ax.barh([row['model'] for row in rows], [row[metric] for row in rows])
            ax.set_xlabel(metric)
        save(fig, f'Same-row annual recipes — {cohort}',
             'Retrospective series benchmark; report-informed recipes, prior-observed roster proxies; no tournament or promotion claim.')
    paired = summary['paired_monthly']
    for cohort in ('full_history', '2024_plus'):
        rows = [row for row in paired if row['cohort'] == cohort and row.get('metric') == 'log_loss']
        fig, ax = plt.subplots(figsize=(12, max(4, len(rows)*.35)))
        for i, row in enumerate(rows):
            mean = row['mean']
            low, high = row['ci95_low'], row['ci95_high']
            if finite(mean):
                ax.plot(mean, i, 'o', color='tab:blue')
                if finite(low) and finite(high):
                    ax.plot([low, high], [i, i], color='tab:blue')
        ax.axvline(0, color='black', linewidth=.8)
        ax.set_yticks(range(len(rows)), [f"{row['candidate']} − {row['reference']}" for row in rows])
        ax.set_xlabel('Paired LogLoss difference; negative favors candidate')
        save(fig, f'Series monthly-block comparisons — {cohort}',
             '5,000 monthly resamples; nominal retrospective intervals, not adjusted selection/promotion evidence.')

    for score_path in scores:
        report = load(score_path)
        label = f'{Path(score_path).parent.name}/{Path(score_path).stem} [{pinned[str(Path(score_path).resolve())][:8]}]'
        analysis = report.get('analysis', 'full_cohort')
        label += f' — {analysis}'
        score_tables.append(_table(
            f'{label} — model summaries (including unpaired methods)',
            (*LEDGER_COHORT, 'model', 'metric', 'mean', 'ci_low', 'ci_high', 'inference_status',
             'forecast rows', 'eligible targets', 'scored targets', 'forecast failures', 'occupied months'),
            ([*(row[key] for key in LEDGER_COHORT), row['model'], row['metric'], row['mean'],
              row['ci_low'], row['ci_high'], row['inference_status'],
              *(row['counts'].get(key) for key in ('forecast_rows', 'eligible_targets', 'scored_targets',
                                                  'forecast_failed_rows', 'occupied_months'))]
             for row in report['summaries'])))
        if 'removed_methods' in report:
            score_tables.append(_table(
                f'{label} — removed methods and original failures',
                (*LEDGER_COHORT, 'model', 'removed rows', 'failed rows', 'failure reasons'),
                ([*(row[key] for key in LEDGER_COHORT), row['model'], row['removed_rows'],
                  row['failed_rows'], row['reasons']] for row in report['removed_methods'])))
        group_keys = LEDGER_COHORT
        groups = defaultdict(list)
        comparison_sets = [(analysis, report['comparisons'])]
        comparison_sets.extend((f"{item['dimension']}={item['value']}", item['comparisons'])
                               for item in report['diagnostics'])
        headers = (*group_keys, 'metric', 'model', 'reference', 'delta', 'ci_low', 'ci_high',
                   'inference_status', 'finite / eligible paired targets', 'edition IDs',
                   'failed candidate / reference rows')
        for scope, comparisons in comparison_sets:
            table_rows = []
            for row in comparisons:
                values = [*(row[key] for key in headers[:12]),
                          f"{row['paired_finite_targets']} / {row['paired_eligible_targets']}",
                          row['counts']['editions'],
                          f"{row['counts']['forecast_failed_rows']} / {row['reference_counts']['forecast_failed_rows']}"]
                table_rows.append('<tr>' + ''.join(
                    f'<td>{html.escape(str(value) if value is not None else "unavailable")}</td>'
                    for value in values) + '</tr>')
                if scope == analysis or scope.startswith(('family=', 'format=')):
                    groups[(scope, *(row[key] for key in (*group_keys, 'metric')))].append(row)
            heading = html.escape(f'{label} — {scope}')
            score_tables.append(f'<details><summary>{heading}: {len(comparisons)} declared contrasts</summary>'
                                '<div class="scroll"><table><thead><tr>'
                                + ''.join(f'<th>{html.escape(key)}</th>' for key in headers)
                                + '</tr></thead><tbody>' + ''.join(table_rows) + '</tbody></table></div></details>')
        for key, rows in sorted(groups.items()):
            fig, ax = plt.subplots(figsize=(12, max(4, len(rows)*.32)))
            unavailable = []
            for i, row in enumerate(rows):
                if finite(row['delta']):
                    ax.plot(row['delta'], i, 'o', color='tab:blue')
                    if finite(row['ci_low']) and finite(row['ci_high']):
                        ax.plot([row['ci_low'], row['ci_high']], [i, i], color='tab:blue')
                else:
                    unavailable.append(f"{row['model']} vs {row['reference']}")
                    ax.text(.01, i, 'unavailable', transform=ax.get_yaxis_transform(), fontsize=8)
            ax.set_yticks(range(len(rows)), [
                f"{row['model']} − {row['reference']} "
                f"(n={row['paired_finite_targets']}/{row['paired_eligible_targets']}; ids={row['counts']['editions']})"
                for row in rows])
            ax.axvline(0, color='black', linewidth=.8)
            ax.set_xlabel('Paired loss difference')
            save(fig, f"{label} — {' / '.join(key)}",
                 f"All {len(rows)} slice contrasts retained; {len(unavailable)} unavailable. n=finite/eligible paired targets; ids=edition IDs, not certified independent editions. Null intervals are inconclusive.", source=score_path)

        for diagnostic in report['calibration']:
            bins = diagnostic['bins']
            title = ' / '.join(str(diagnostic[key]) for key in (*group_keys, 'model'))
            fig, ax = plt.subplots(figsize=(7, 5))
            populated = [row for row in bins if finite(row.get('mean_probability'))]
            ax.plot([0, 1], [0, 1], '--', color='gray')
            if populated:
                ax.plot([row['mean_probability'] for row in populated], [row['observed_frequency'] for row in populated], 'o-')
                for row in populated:
                    ax.annotate(f"n={row['cells']}; e={row['editions']}",
                                (row['mean_probability'], row['observed_frequency']), fontsize=7)
                    _plot_interval(ax, row['mean_probability'], row['uncertainty'], row['mean_probability'])
            else:
                ax.text(.5, .5, 'No precision-eligible calibration cohort', ha='center', transform=ax.transAxes)
            ax.set(xlim=(0, 1), ylim=(0, 1), xlabel='Mean probability', ylabel='Observed frequency')
            save(fig, f'{label} reliability — {title}',
                 f"{diagnostic['status']}; included={diagnostic['included_targets']}/{diagnostic['forecast_rows']}; exclusions={diagnostic['exclusions']}. Bands show observed-minus-predicted intervals shifted by mean probability; per-bin support and unavailable reasons are tabulated. No calibration certification.", source=score_path)
            score_tables.append(_table(
                f'{label} — reliability interval support — {title}',
                ('bin', 'lower', 'upper', 'mean probability', 'observed frequency',
                 'observed minus predicted', *interval_headers),
                ([row['bin'], row['lower'], row['upper'], row['mean_probability'],
                  row['observed_frequency'], row['observed_minus_predicted'],
                  *_interval_cells(row, row['uncertainty'])] for row in bins)))

        for diagnostic in report['distribution_diagnostics']:
            title = ' / '.join(str(diagnostic[key]) for key in (*group_keys, 'model'))
            frequencies = diagnostic['category_frequencies']
            fig, axes = plt.subplots(1, 3, figsize=(17, 5))
            if frequencies:
                x = np.arange(len(frequencies))
                axes[0].bar(x-.18, [row['mean_probability'] for row in frequencies], .36, label='predicted')
                axes[0].bar(x+.18, [row['observed_frequency'] for row in frequencies], .36, label='observed')
                axes[0].set_xticks(x, [row['category'] for row in frequencies], rotation=90, fontsize=6)
                axes[0].legend()
                for position, row in zip(x, frequencies):
                    _plot_interval(axes[0], position+.18, row['observed_minus_predicted_interval'],
                                   row['mean_probability'])
            else:
                axes[0].text(.5, .5, 'No precision-eligible targets', ha='center', transform=axes[0].transAxes)
            axes[0].set_title('Score / count / joint category frequencies')
            sets = [row for row in diagnostic['predictive_sets'] if finite(row['coverage'])]
            axes[1].plot([0, 1], [0, 1], '--', color='gray')
            if sets:
                axes[1].plot([row['actual_included_mass'] for row in sets], [row['coverage'] for row in sets], 'o-')
                for row in sets:
                    axes[1].annotate(f"{row['nominal_level']:.0%}: size={row['mean_set_size']:.1f}",
                                     (row['actual_included_mass'], row['coverage']), fontsize=7)
                    _plot_interval(axes[1], row['actual_included_mass'], row['coverage_minus_mass_interval'],
                                   row['actual_included_mass'])
            axes[1].set(xlim=(0, 1), ylim=(0, 1), xlabel='Actual predictive-set mass', ylabel='Coverage')
            pit = diagnostic['pit']
            if pit['bins']:
                axes[2].bar([row['lower'] for row in pit['bins']], [row['edition_weighted_fraction'] for row in pit['bins']], .095, align='edge')
                axes[2].axhline(.1, linestyle='--', color='gray')
                for row in pit['bins']:
                    _plot_interval(axes[2], (row['lower']+row['upper'])/2, row['interval'])
            else:
                axes[2].text(.5, .5, pit['status'].replace('_', ' '), ha='center', wrap=True, transform=axes[2].transAxes)
            axes[2].set(xlim=(0, 1), xlabel='Randomized PIT (ordered targets only)', ylabel='Edition-weighted frequency')
            save(fig, f'{label} distributions — {title}',
                 f"{diagnostic['status']}; editions={diagnostic['editions']}, targets={diagnostic['included_targets']}. Exclusions: {diagnostic['exclusions']}. Orange bands are discrepancy intervals shifted by predicted probability/mass, or direct PIT-frequency intervals. Exact support/status remains in the diagnostic tables.", source=score_path)
            score_tables.append(_table(
                f'{label} — category discrepancy intervals — {title}',
                ('category', 'mean probability', 'observed frequency', *interval_headers),
                ([row['category'], row['mean_probability'], row['observed_frequency'],
                  *_interval_cells(row, row['observed_minus_predicted_interval'])] for row in frequencies)))
            score_tables.append(_table(
                f'{label} — predictive-set coverage-minus-mass intervals — {title}',
                ('nominal level', 'actual included mass', 'coverage', 'coverage minus mass', 'mean set size',
                 *interval_headers),
                ([row['nominal_level'], row['actual_included_mass'], row['coverage'],
                  row['coverage_minus_mass'], row['mean_set_size'],
                  *_interval_cells(row, row['coverage_minus_mass_interval'])]
                 for row in diagnostic['predictive_sets'])))
            score_tables.append(_table(
                f"{label} — PIT frequency intervals ({pit['status']}) — {title}",
                ('bin', 'lower', 'upper', 'forecast count', 'edition weighted fraction', *interval_headers),
                ([row['bin'], row['lower'], row['upper'], row['forecast_count'], row['edition_weighted_fraction'],
                  *_interval_cells(row, row['interval'])] for row in pit['bins'])))

        trajectories = defaultdict(list)
        for row in report['target_scores']:
            if row['target_kind'] not in ('binary', 'placement') or not row.get('probabilities'):
                continue
            trajectories[(row['tournament_id'], row['phase_id'], row['model'],
                          row['target_id'], *(row[field] for field in group_keys))].append(row)
        for key, rows in sorted(trajectories.items()):
            rows.sort(key=lambda row: (row['cutoff'], row['origin_id']))
            if len({row['cutoff'] for row in rows}) < 2:
                continue
            fig, ax = plt.subplots(figsize=(10, 4))
            for category in rows[0]['probabilities']:
                ax.plot(range(len(rows)), [row['probabilities'].get(category, np.nan) for row in rows], 'o-', label=category)
            ax.set_xticks(range(len(rows)), [row['cutoff'][:10] for row in rows], rotation=45, ha='right')
            ax.set(ylim=(0, 1), ylabel='Forecast probability')
            ax.legend(fontsize=7)
            save(fig, f"{label} trajectory — {' / '.join(key)}",
                 'Raw immutable forecast vectors, including numerically unresolved rows. No monotonic-sharpness or calibration claim.', source=score_path)

    oracle_rows = load(Path(oracles)/'results.json')
    grouped = defaultdict(list)
    oracle_aggregates = []
    for row in oracle_rows:
        grouped[row['fixture'], row['source']].append(row)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for key, rows in sorted(grouped.items()):
        for ax, metric in zip(axes, ('maximum_category_error', 'total_variation', 'expected_log_score_excess')):
            by_size = defaultdict(list)
            for row in rows:
                by_size[row['simulations']].append(row)
            sizes, means = [], []
            for size, samples in sorted(by_size.items()):
                values = [row[metric] for row in samples if finite(row.get(metric))]
                if len(values) == len(samples):
                    status, mean = 'finite_all_seeds', float(np.mean(values))
                elif metric == 'expected_log_score_excess' and any(
                        row.get('expected_log_score_status') == 'infinite_unsampled_reference_support'
                        for row in samples):
                    status, mean = 'infinite_unsampled_reference_support', None
                elif all(row.get('reason') and row.get(metric) is None for row in samples):
                    status, mean = 'not_applicable_marginal_vector', None
                else:
                    status, mean = 'unavailable_incomplete_seed_support', None
                oracle_aggregates.append({'fixture': key[0], 'source': key[1], 'metric': metric,
                    'simulations': size, 'seed_records': len(samples), 'finite_seed_records': len(values),
                    'status': status, 'mean': mean})
                sizes.append(size)
                means.append(mean if mean is not None else np.nan)
                if mean is None and status != 'not_applicable_marginal_vector':
                    ax.plot(size, .97, 'x', color='tab:red', transform=ax.get_xaxis_transform())
                    ax.annotate(f"{'∞' if status.startswith('infinite') else 'unavailable'} "
                                f"({len(values)}/{len(samples)} finite)", (size, .97),
                                xycoords=ax.get_xaxis_transform(), fontsize=6, rotation=90, va='top')
            ax.plot(sizes, means, 'o-', label=' / '.join(key))
            ax.set_xscale('log')
            ax.set(xlabel='Independent rollouts M', ylabel=metric)
    axes[0].legend(fontsize=5)
    save(fig, 'Numerical error versus fixed rollout count — independent exact oracles',
         'Finite means require all declared seeds. Red markers retain infinite/unavailable cases; support counts are tabulated below. Numerical error only, not historical uncertainty.')
    seed_precision = load(precision) if precision else None
    if seed_precision is not None:
        score_tables.append(
            '<h2>Repeated-seed numerical precision</h2><p>Conditional on the same editions and outcomes; '
            'not historical confidence intervals or independent predictive observations. Complete and '
            'available-method denominators remain separate. The declared comparison model is '
            + html.escape(seed_precision['protocol']['model'])
            + '; this does not answer the frozen academic primary for a different artifact.</p>')
        for key in ('complete_cohort_mc', 'available_method_mc'):
            score_tables.append(_table(
                f'Repeated-seed precision — {key}',
                (*LEDGER_COHORT, 'model', 'reference', 'metric', 'status', 'seeds', 'required seeds',
                 'same matched targets', 'mean paired delta', 'MC mean delta 95% half-width',
                 'primary numerical budget applicable', 'maximum half-width', 'budget status'),
                ([*(row[field] for field in LEDGER_COHORT), row['model'], row['reference'], row['metric'],
                  row['status'], row['seed_count'], row['required_seed_count'],
                  row['same_matched_targets_across_seeds'], row['mean_paired_delta'],
                  row['mc_mean_delta_95_half_width'],
                  *(row['locked_primary_delta_ll_budget'][field]
                    for field in ('applicable', 'maximum_95_half_width', 'status'))]
                 for row in seed_precision[key])))
        score_tables.append(_table(
            'Per-seed coverage and unresolved target precision',
            ('seed', 'actual MC sample counts', 'forecast rows', 'forecast failures',
             'unresolved non-failed target precision statuses'),
            ([run['seed'], run['actual_mc_sample_counts'], run['status']['forecast_rows'],
              run['status']['forecast_failures'],
              dict(Counter(row['precision_status'] for row in run['target_numerical_precision']
                           if row['forecast_status'] == 'ok' and not row['resolved_at_origin']))]
             for run in seed_precision['runs'])))
    manifest = {'version': 'tournament-validation-figures-v1', 'inputs': pinned, 'figures': figures,
                'oracle_aggregates': oracle_aggregates,
                'admission': load(admission) if admission else None,
                'seed_precision': seed_precision,
                'production_qualified': False}
    if any(file_digest(path) != value for path, value in pinned.items()):
        raise ValueError('report input changed during rendering')
    (output/'figures.json').write_text(json.dumps(manifest, indent=2, allow_nan=False)+'\n')
    cards = ''.join(f'<section><h2>{html.escape(row["title"])}</h2><img loading="lazy" src="{row["file"]}" alt="{html.escape(row["title"])}"><p>{html.escape(row["qualification"])}</p></section>' for row in figures)
    tables = '<section><h2>Scoreboards, cohort exclusions and diagnostic support</h2><p>Every supplied diagnostic slice remains separate. Full-cohort ledgers retain failures. Available-method ledgers explicitly remove whole failed methods; their removal tables preserve the original failures. Unpaired model summaries remain visible. Finite slice effects do not replace the complete cohort. Edition IDs are not certified independent editions.</p>' + ''.join(score_tables) + '</section>'
    oracle_headers = ('fixture', 'source', 'metric', 'simulations', 'seed_records',
                      'finite_seed_records', 'status', 'mean')
    tables += '<section><h2>Oracle seed support — no finite-only averages</h2><details><summary>All numerical aggregate cells</summary><div class="scroll"><table><thead><tr>' + ''.join(
        f'<th>{html.escape(key)}</th>' for key in oracle_headers) + '</tr></thead><tbody>' + ''.join(
        '<tr>' + ''.join(f'<td>{html.escape(str(row[key]) if row[key] is not None else "unavailable")}</td>'
                        for key in oracle_headers) + '</tr>' for row in oracle_aggregates) + '</tbody></table></div></details></section>'
    (output/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Offline tournament validation</title>'
        '<style>body{font:15px system-ui;margin:2rem;background:#f4f5f7;color:#17212b}section{background:white;padding:1rem;margin:1rem 0}img{max-width:100%;height:auto}p{max-width:100ch}.scroll{overflow:auto}table{border-collapse:collapse;font-size:12px}td,th{padding:.4rem;border:1px solid #ccd2d8;white-space:nowrap}summary{padding:.6rem;cursor:pointer}</style>'
        '<h1>Offline tournament validation</h1><p>Retrospective research. Explicit unavailable cells remain unavailable. No model promotion. <a href="figures.json">Source hashes, precision and oracle evidence</a>.</p>'+tables+cards)
    print(json.dumps({'output': str(output), 'figures': len(figures), 'production_qualified': False}))
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--readiness', type=Path, required=True)
    parser.add_argument('--scores', type=Path, nargs='+', required=True)
    parser.add_argument('--benchmark', type=Path, required=True)
    parser.add_argument('--oracles', type=Path, required=True)
    parser.add_argument('--precision', type=Path)
    parser.add_argument('--admission', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = vars(parser.parse_args())
    render(**args)


if __name__ == '__main__':
    main()
