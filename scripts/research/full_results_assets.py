"""Render thesis evidence from frozen F0 forecasts and optional F1 controls.

This is report generation, not training or model selection. F0 is the historical
recipe; F1 changes the player TrueSkill probability, not rating updates.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.analysis.metrics import binary_log_loss_vector
from src.analysis.thesis_evaluation import block_difference
from src.analysis.thesis_metrics import _metrics
from src.visualization.thesis_style import (
    DARK_TEXT, PASTEL_BLUE, PASTEL_GREEN, PASTEL_ORANGE, PASTEL_PURPLE,
    apply_thesis_style,
)


def fmt(x, d=6):
    return f"{float(x):.{d}f}".replace(".", "{,}")


def table(caption, label, headers, rows, spec=None, short_caption=None):
    return "\n".join([
        r"\begin{table}[htbp]", r"\centering\small",
        r"\caption" + (f"[{short_caption}]" if short_caption else "") + "{" + caption + "}",
        r"\label{" + label + "}",
        r"\begin{tabular}{" + (spec or "l" + "r" * (len(headers) - 1)) + "}",
        r"\toprule", " & ".join(headers) + r" \\", r"\midrule",
        *[" & ".join(map(str, row)) + r" \\" for row in rows],
        r"\bottomrule", r"\end{tabular}", r"\end{table}", "",
    ])


def save_figure(fig, directory, name):
    fig.tight_layout()
    paths = [directory / (name + suffix) for suffix in (".pdf", ".png")]
    fig.savefig(paths[0], bbox_inches="tight")
    fig.savefig(paths[1], dpi=180, bbox_inches="tight")
    plt.close(fig)
    return paths


def correction_assets(corrections, metric, comparisons, generated, figures, output):
    """Render the predeclared seed/feature/stopping controls without selecting one."""
    paths = []

    def emit(name, caption, headers, rows, spec=None, short_caption=None):
        path = generated / ("lr_revision_" + name + ".tex")
        path.write_text(table(caption, "tab:lr_revision_" + name, headers, rows, spec, short_caption))
        paths.append(path)

    variants = [
        ("LR (wersja wcześniejsza)", "p_lr_original_seed42"),
        ("LR (wersja końcowa)", "p_lr_ts_standard_seed42"),
        ("MLP (wersja wcześniejsza)", "p_mlp_original_seed42"),
        ("MLP (wersja końcowa)", "p_mlp_ts_standard_seed42"),
        ("MLP bez ES (wersja wcześniejsza)", "p_mlp_original_no_early_stopping"),
        ("MLP bez ES (wersja końcowa)", "p_mlp_ts_standard_no_early_stopping"),
    ]
    emit("controls", "Kontrole W20 z ziarnem 42 dla wcześniejszej i końcowej wersji modeli. "
         "ES oznacza wczesne zatrzymywanie według trafności klasyfikacji na odłożonej części TRAIN. "
         r"Wyłączenie ES zwiększa udział TRAIN używany do aktualizacji wag z 85\% do 100\%.",
         ["Wariant", r"\shortstack{LogLoss\\Próba pełna\\($N=11\,550$)}",
          r"\shortstack{LogLoss\\Próba z kursami\\($N=3\,229$)}"],
         [[name, fmt(metric("sports", model).logloss),
           fmt(metric("market_common", model).logloss)] for name, model in variants],
         spec="p{6cm}rr", short_caption="Porównanie wersji i kontroli uczenia LR i MLP")

    seeds = [7, 42, 2026]
    rows = []
    for seed in seeds:
        rows.append([seed, *[fmt(metric(cohort, f"p_{family}_ts_standard_seed{seed}").logloss)
                             for cohort in ("sports", "market_common")
                             for family in ("lr", "mlp")]])
    emit("seeds", "LogLoss końcowych modeli LR i MLP dla trzech ustalonych "
         "ziaren generatora liczb pseudolosowych. Nie wybierano najlepszego przebiegu; "
         "ziarna nie stanowią niezależnych prób meczów.",
         ["Ziarno", r"\shortstack{LR\\Próba pełna}", r"\shortstack{MLP\\Próba pełna}",
          r"\shortstack{LR\\Próba z kursami}", r"\shortstack{MLP\\Próba z kursami}"], rows,
         short_caption="LogLoss dla trzech ziaren generatora")
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    for ax, cohort, title in zip(axes, ("sports", "market_common"),
                                 ("Pełna próba: 11 550", "Wspólna próba rynku: 3229")):
        for family, label, color in (("lr", "ElasticNet", PASTEL_BLUE),
                                     ("mlp", "MLP", PASTEL_PURPLE)):
            values = [metric(cohort, f"p_{family}_ts_standard_seed{s}").logloss for s in seeds]
            ax.plot(range(3), values, marker="o", color=color, label=label)
        ax.set_xticks(range(3), [str(s) for s in seeds])
        ax.set_xlabel("Seed")
        ax.set_ylabel("LogLoss")
        ax.set_title(title)
        ax.ticklabel_format(axis="y", style="plain", useOffset=False)
        ax.legend()
    paths.extend(save_figure(fig, figures, "lr_revision_seeds"))

    diagnostics = pd.read_csv(corrections / "fit_diagnostics.csv")
    if diagnostics.duplicated(["variant", "origin"]).any():
        raise ValueError("Duplicate fit diagnostic")
    compute = diagnostics.groupby("variant").agg(
        folds=("origin", "size"), cpu_seconds=("step_cpu_seconds", "sum"),
        parameters=("parameter_count", "first"), median_iterations=("iterations", "median"),
    )
    if not compute.folds.eq(33).all():
        raise ValueError("Incomplete monthly fit diagnostics")
    compute.to_csv(output / "lr_revision_compute_summary.csv")
    rows = []
    for name, model in variants:
        r = compute.loc[model.removeprefix("p_")]
        rows.append([name, int(r.parameters), int(r.median_iterations), fmt(r.cpu_seconds, 1)])
    emit("complexity", "Głowica modelu i zmierzony czas CPU 33 kroków fit/predict/calibrate, "
         "seed42. Wspólne przetwarzanie i kalibrator nie należą do liczby parametrów głowicy. "
         "Nie mierzono opóźnienia samej inferencji.",
         ["Wariant", "Parametry", "Mediana iteracji", "CPU [s]"], rows)

    effects = [
        ("LR: zmiana wariantu", "p_lr_ts_standard_seed42", "p_lr_original_seed42"),
        ("MLP: zmiana wariantu", "p_mlp_ts_standard_seed42", "p_mlp_original_seed42"),
        ("Końcowe LR minus MLP", "p_lr_ts_standard_seed42", "p_mlp_ts_standard_seed42"),
    ]
    rows = []
    for label, candidate, reference in effects:
        for cohort, title in (("sports", "sport"), ("market_common", "rynek")):
            selected = comparisons[
                (comparisons.cohort == cohort) & (comparisons.candidate == candidate)
                & (comparisons.reference == reference) & (comparisons.metric == "logloss")
                & (comparisons.blocking == "month") & (comparisons.estimand == "match_weighted")
            ]
            if len(selected) != 1:
                raise ValueError(f"Missing or ambiguous correction contrast: {cohort}/{candidate}/{reference}")
            r = selected.iloc[0]
            rows.append([label, title, fmt(r.delta), f"[{fmt(r.ci_lower)}; {fmt(r.ci_upper)}]"])
    emit("effects", "Zmiana wariantu: końcowy minus wcześniejszy; osobno końcowe LR--MLP, ziarno 42. "
         "Sparowany bootstrap miesięczny, 5000 replik; przedziały nominalne, ważenie meczami.",
         ["Porównanie", "Próba", r"$\Delta$ LL", r"95\% przedział"], rows)

    raw_coefficients = pd.read_csv(corrections / "coefficients.csv")
    raw_coefficients = raw_coefficients[raw_coefficients.variant == "lr_ts_standard_seed42"]
    if raw_coefficients.duplicated(["origin", "feature"]).any():
        raise ValueError("Duplicate monthly feature coefficient")
    coefficients = raw_coefficients.groupby("feature").standardized_coefficient.agg(
        mean="mean", minimum="min", maximum="max", folds="size",
    ).reset_index()
    if len(coefficients) != 46 or not coefficients.folds.eq(33).all():
        raise ValueError("Incomplete LR F1 coefficient history")
    selected = coefficients.loc[coefficients["mean"].abs().nlargest(8).index]
    selected.to_csv(output / "lr_revision_selected_coefficients.csv", index=False)
    rows = [[r.feature.replace("_", r"\_"), fmt(r.mean, 3), fmt(r.minimum, 3), fmt(r.maximum, 3)]
            for r in selected.itertuples()]
    emit("coefficients", "Osiem największych modułów średniego standaryzowanego współczynnika "
         "końcowego LR, ziarno 42, w 33 dopasowaniach. Ekstrema dotyczą dopasowań, nie przedziałów ufności. "
         "Współczynniki nie są miarą wpływu przyczynowego ani ważności całych pakietów cech.",
         ["Cecha", r"$\overline{\beta}$", "Minimum", "Maksimum"], rows,
         spec=r"p{.48\textwidth}rrr")
    return paths


def historical_tables(metric, generated):
    """Recreate the F0 presentations used by the full thesis, without fitting."""
    outputs = []

    def emit(filename, caption, label, headers, rows, spec=None, short_caption=None):
        path = generated / filename
        path.write_text(table(caption, label, headers, rows, spec, short_caption))
        outputs.append(path)

    rating_configs = [
        ("Glicko-2", r"$RD_0=350$, zmienność początkowa $0{,}06$, okres $7$ dni", "p_player_gl"),
        ("Elo", r"$K=48$", "p_player_elo"),
        ("Thurstone--Mosteller", r"$\beta=18{,}75$, $\tau=0{,}05$", "p_player_tm"),
        ("Plackett--Luce", r"$\beta=18{,}75$, $\tau=0{,}05$", "p_player_pl"),
        ("TrueSkill", r"$\beta=4{,}16$, $\tau=0{,}25$", "p_player_ts"),
        ("OpenSkill", r"$\sigma_0=3{,}50$", "p_player_os"),
        ("Market OPEN", "--", "market_open"),
        ("Market CLOSE", "--", "market_close"),
    ]
    rows = []
    for name, parameters, model in rating_configs:
        r = metric("market_common", model)
        rows.append([name, parameters, fmt(r.auc, 4), fmt(r.logloss, 4),
                     fmt(r.brier, 4), fmt(100 * r.ece, 2)])
    emit("full_models_ratings_original.tex",
         "Modele ratingowe w pierwotnej wersji eksperymentu po projekcji na serię i kalibracji. "
         r"Wspólna próba obejmuje 3\,229 meczów od stycznia 2024 do kwietnia 2026.",
         "tab:ranking_common_sample_results",
         ["Wariant", "Parametry", "AUC", "LogLoss", r"\shortstack{Brier\\score}", "ECE [pp]"], rows,
         spec=r"lp{3.7cm}rrrr", short_caption="Porównanie modeli ratingowych na wspólnej próbie")

    windows = [("W3", "p_lr_w3"), ("W5", "p_lr_w5"), ("W10", "p_lr_w10"),
               ("W20", "p_logistic"), ("W50", "p_lr_w50")]
    rows = [[label, *[fmt(metric(cohort, model)[field], 4 if field == "auc" else 6)
                     for cohort in ("sports", "market_common")
                     for field in ("auc", "logloss", "brier")]] for label, model in windows]
    window_table = table(
        "Wcześniejsze porównanie okien LR na pełnej próbie sportowej (11 550) i wspólnej próbie rynku (3229). "
        "W20 jest odniesieniem, nie oknem wybranym ponownie z tej tabeli.",
        "tab:metamodel_window_ablation",
        ["Wariant", "AUC", "LogLoss", "Brier", "AUC", "LogLoss", "Brier"], rows)
    window_table = window_table.replace(
        r"\toprule" + "\n",
        r"\toprule" + "\n" + r"& \multicolumn{3}{c}{Sport (11 550)} & \multicolumn{3}{c}{Rynek (3229)} \\"
        + "\n" + r"\cmidrule(lr){2-4} \cmidrule(lr){5-7}" + "\n", 1)
    path = generated / "full_models_window.tex"
    path.write_text(window_table)
    outputs.append(path)

    masking = [("Pełny LR", "p_logistic"), ("Glicko + kontekst", "p_lr_glicko_context"),
               (r"Mask baza 10\%", "p_supp_mask_base10"),
               (r"Mask pGL 10\%", "p_supp_mask_playergl10"),
               (r"Mask pGL 30\%", "p_supp_mask_playergl30"),
               ("Bez surowego pGL", "p_lr_no_player_gl"), ("Bez pakietu Glicko", "p_lr_no_glicko")]
    rows = []
    baseline = metric("sports", "p_logistic").logloss
    for name, model in masking:
        r = metric("sports", model)
        rows.append([name, fmt(r.auc, 4), fmt(r.logloss), fmt(100 * r.ece, 2), fmt(r.logloss - baseline)])
    emit("full_models_masking_original.tex",
         "Wcześniejsza analiza ablacji i maskowania na pełnej próbie 11 550 meczów. ECE w pp; "
         "różnica względem pełnego LR. Usunięcie pGL nie usuwa całego pakietu Glicko-2.",
         "tab:logistic_masking_ablation", ["Wariant", "AUC", "LogLoss", "ECE [pp]", r"$\Delta$ LL"], rows)
    return outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", type=Path, required=True, help="Frozen historical F0 evaluation")
    parser.add_argument("--corrections", type=Path, help="Completed F1 correction report; no training")
    parser.add_argument("--thesis", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    generated = args.thesis / "generated"
    figures = args.thesis / "figures" / ("thesis_lr_revision" if args.corrections else "thesis_full_20260921")
    generated.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    paths = [args.evaluation / "paired_predictions.parquet"]
    original = pd.read_parquet(paths[0])
    version = "F1" if args.corrections else "F0"
    stage = "końcowy" if args.corrections else "wcześniejszy"
    stage_suffix = "" if args.corrections else " z wcześniejszego etapu"
    if args.corrections:
        paths += [args.corrections / "paired_predictions.parquet",
                  args.corrections / "analysis/metrics.csv",
                  args.corrections / "analysis/comparisons.csv",
                  args.corrections / "fit_diagnostics.csv",
                  args.corrections / "coefficients.csv"]
        panel = pd.read_parquet(paths[1])
        for frame in (original, panel):
            if frame.golgg_match_id.isna().any() or frame.golgg_match_id.duplicated().any():
                raise ValueError("Missing or duplicate frozen match ID")
        metadata = ["date", "result_day", "BoN", "y_true", "market_open", "market_close",
                    "p_logistic", "p_mlp", "p_player_gl"]
        pd.testing.assert_frame_equal(
            original.set_index("golgg_match_id")[metadata].sort_index(),
            panel.set_index("golgg_match_id")[metadata].sort_index(),
            check_dtype=False,
        )
        metrics = pd.read_csv(paths[2])
        comparisons = pd.read_csv(paths[3])
        focal, mlp = "p_lr_ts_standard_seed42", "p_mlp_ts_standard_seed42"
        names = {focal: "LR ElasticNet", mlp: "MLP", "p_player_gl": "Player Glicko-2",
                 "market_open": "Market OPEN", "market_close": "Market CLOSE"}
    else:
        panel = original
        paths += [args.evaluation / "metrics.csv", args.evaluation / "comparisons.csv"]
        metrics, comparisons = [pd.read_csv(p) for p in paths[1:]]
        focal, mlp = "p_logistic", "p_mlp"
        names = {focal: "LR ElasticNet", mlp: "MLP", "p_extra_trees": "ExtraTrees",
                 "p_hist_gradient_boosting": "HGB", "p_lightgbm": "LightGBM",
                 "p_player_gl": "Player Glicko-2", "market_open": "Market OPEN", "market_close": "Market CLOSE"}
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    common = panel[panel[["market_open", "market_close"]].notna().all(axis=1)]
    if len(panel) != 11550 or len(common) != 3229:
        raise ValueError("Frozen thesis population mismatch: expected 11550/3229")
    metric_index = metrics.set_index(["cohort", "model"])
    if not metric_index.index.is_unique:
        raise ValueError("Duplicate cohort/model metric")
    outputs = []

    def metric(cohort, model):
        return metric_index.loc[(cohort, model)]

    def emit(name, caption, label, headers, rows, spec=None, short_caption=None):
        path = generated / ("full_results_" + name + ".tex")
        path.write_text(table(caption, label, headers, rows, spec, short_caption))
        outputs.append(path)

    def save(fig, name):
        outputs.extend(save_figure(fig, figures, "full_results_" + name))

    def contrast(data, candidate, reference, cohort):
        selected = comparisons[(comparisons.cohort == cohort) & (comparisons.candidate == candidate)
                               & (comparisons.reference == reference) & (comparisons.metric == "logloss")
                               & (comparisons.blocking == "month")]
        if "estimand" in selected:
            selected = selected[selected.estimand == "match_weighted"]
        if len(selected) == 1:
            return selected.iloc[0].to_dict()
        if len(selected) > 1:
            raise ValueError(f"Ambiguous contrast: {cohort}/{candidate}/{reference}")
        delta = binary_log_loss_vector(data.y_true, data[candidate]) - binary_log_loss_vector(data.y_true, data[reference])
        return dict(candidate=candidate, reference=reference, cohort=cohort,
                    metric="logloss", blocking="month", estimand="match_weighted",
                    **block_difference(delta, pd.to_datetime(data.date).dt.strftime("%Y-%m")))

    apply_thesis_style()
    order = list(names)
    palette = {focal: PASTEL_BLUE, mlp: PASTEL_PURPLE, "p_player_gl": PASTEL_GREEN,
               "market_open": PASTEL_ORANGE, "market_close": "#C2854A"}
    primary_models = (focal, mlp)
    short_names = {focal: "LR", mlp: "MLP", "p_player_gl": "Glicko-2",
                   "market_open": "OPEN", "market_close": "CLOSE"}
    sports_rows, sports_comparisons = [], []
    for model in (*primary_models, "p_player_gl"):
        score = metric("sports", model)
        if model == "p_player_gl":
            delta, interval = "--", "--"
        else:
            result = contrast(panel, model, "p_player_gl", "sports")
            sports_comparisons.append(result)
            delta = fmt(result["delta"])
            interval = f"[{fmt(result['ci_lower'])}; {fmt(result['ci_upper'])}]"
        sports_rows.append([short_names[model], fmt(score.logloss), fmt(score.brier), delta, interval])
    pd.DataFrame(sports_comparisons).to_csv(args.output / "full_results_sports_comparisons.csv", index=False)
    emit("sports", r"Ocena H2 na pełnej próbie 11\,550 meczów: "
         f"modele LR i MLP{stage_suffix} wobec pojedynczego Glicko-2. "
         r"$\Delta$ LL oznacza LogLoss metamodelu pomniejszony o LogLoss Glicko-2. "
         r"Nominalne 95\% przedziały wyznaczono w 5\,000 replikach sparowanego bootstrapu blokowego, "
         "losując 33 miesiące z powtórzeniami i nadając meczom jednakowe wagi.",
         "tab:full_results_sports", ["Model", "LogLoss", "Brier score", r"$\Delta$ LL", r"95\% przedział"],
         sports_rows, short_caption="Porównanie metamodeli z Glicko-2 na pełnej próbie")

    ablation_rows, ablation_data = [], []
    for label, baseline, ablations in (
        ("LR", "p_logistic", (("Bez kontekstu", "p_lr_player_only"),
                               ("Bez BoN", "p_supp_no_binomial"))),
        ("MLP", "p_mlp", (("Bez kontekstu", "p_supp_primary_no_context"),
                          ("Bez BoN", "p_supp_primary_no_binomial"))),
    ):
        for description, model in ablations:
            result = contrast(original, model, baseline, "sports")
            score = metric("sports", model).logloss
            ablation_data.append(dict(result, logloss=score))
            ablation_rows.append([label, description, fmt(score), fmt(result["delta"]),
                                  f"[{fmt(result['ci_lower'])}; {fmt(result['ci_upper'])}]"])
    pd.DataFrame(ablation_data).to_csv(args.output / "full_results_ablations.csv", index=False)
    emit("ablations", r"Ablacje modeli LR i MLP z wcześniejszego etapu eksperymentu na tych samych 11\,550 meczach. "
         r"$\Delta$ LL oznacza LogLoss wariantu ablacyjnego pomniejszony o LogLoss modelu z pełnym zestawem cech. "
         r"Nominalne 95\% przedziały wyznaczono w 5\,000 replikach sparowanego bootstrapu blokowego, "
         "losując 33 miesiące z powtórzeniami i nadając meczom jednakowe wagi.",
         "tab:metamodel_ablations", ["Model", "Wariant", "LogLoss", r"$\Delta$ LL", r"95\% przedział"],
         ablation_rows, spec="llrrr", short_caption="Ablacje kontekstu drużynowego i projekcji BoN")
    rows = [[names[x], fmt(metric("market_common", x).auc, 4), fmt(metric("market_common", x).logloss),
             fmt(metric("market_common", x).brier), fmt(100 * metric("market_common", x).ece, 2)] for x in order]
    emit("metrics", f"Porównanie modeli LR i MLP{stage_suffix}, modelu Glicko-2 "
         r"oraz prognoz rynkowych na wspólnej próbie 3\,229 meczów. "
         "Modele sportowe po symetryzacji i kalibracji dopasowanej do wcześniejszych prognoz spoza próby uczącej.",
         "tab:final_model_market_comparison", ["Model", "AUC", "LogLoss", "Brier score", "ECE [pp]"], rows,
         short_caption="Metryki modeli i rynku na wspólnej próbie")
    for field, label in (("logloss", "LogLoss (mniej = lepiej)"), ("auc", "AUC (więcej = lepiej)"),
                         ("ece", "ECE [pp] (diagnostyka)")):
        values = [metric("market_common", x)[field] * (100 if field == "ece" else 1) for x in order]
        fig, ax = plt.subplots(figsize=(7.2, 3.6))
        bars = ax.barh([names[x] for x in order], values, color=[palette.get(x, PASTEL_PURPLE) for x in order])
        ax.invert_yaxis()
        ax.set_xlabel(label)
        ax.set_xlim(0, max(values) * 1.2)
        for bar, value in zip(bars, values):
            ax.text(value + max(values) * .015, bar.get_y() + bar.get_height() / 2,
                    f"{value:.4f}" if field != "ece" else f"{value:.2f}", va="center", fontsize=9)
        save(fig, field)

    references = ["p_player_gl", "market_open", "market_close"]
    records = [contrast(common, model, ref, "market_common")
               for model in primary_models for ref in references]
    records.append(contrast(common, focal, mlp, "market_common"))
    contrasts = pd.DataFrame(records)
    contrasts.to_csv(args.output / "full_results_common_comparisons.csv", index=False)
    emit("comparisons", r"Różnice LogLoss na wspólnej próbie 3\,229 meczów: "
         f"model LR lub MLP{stage_suffix} minus wskazany model odniesienia. "
         "Wartość ujemna oznacza niższy LogLoss metamodelu. "
         r"Nominalne 95\% przedziały pochodzą z 5\,000 replik sparowanego bootstrapu blokowego, "
         "z miesiącem jako jednostką losowania i jednakowymi wagami meczów, bez korekty na selekcję.",
         "tab:logloss_block_bootstrap",
         ["Model", "Odniesienie", r"$\Delta$ LL", r"95\% przedział", "Miesiące"],
         [[short_names[r.candidate], short_names[r.reference], fmt(r.delta),
           f"[{fmt(r.ci_lower)}; {fmt(r.ci_upper)}]", int(r.blocks)] for r in contrasts.itertuples()],
         spec="llrrr", short_caption="Różnice LogLoss na wspólnej próbie z kursami")
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for model in primary_models:
        selected = contrasts[contrasts.candidate == model]
        values = selected.delta.to_numpy()
        ax.errorbar(values, selected.index.to_numpy(),
                    xerr=np.vstack([values - selected.ci_lower, selected.ci_upper - values]),
                    fmt="o", color=palette[model], ecolor=palette[model], capsize=4)
    ax.set_yticks(range(len(contrasts)),
                  [f"{short_names[r.candidate]} − {short_names[r.reference]}" for r in contrasts.itertuples()])
    ax.invert_yaxis()
    ax.axvline(0, color="gray", linestyle="--")
    ax.set_xlabel("Różnica LogLoss: metamodel minus odniesienie")
    save(fig, "paired_ci")

    # Recompute only descriptive scores and fixed monthly intervals; never fit a model.
    segment_scores, segment_intervals, bon_rows = [], [], []
    for bon in (1, 3, 5):
        data = common[common.BoN.astype(int) == bon]
        selected_models = [*primary_models, "market_open", "market_close", "p_player_gl"]
        values = {x: float(binary_log_loss_vector(data.y_true, data[x]).mean()) for x in selected_models}
        bon_rows.append([f"Bo{bon}", len(data), *[fmt(values[x]) for x in selected_models]])
        segment_scores.extend(dict(BoN=bon, model=x, n=len(data), logloss=v) for x, v in values.items())
        for model in primary_models:
            loss = binary_log_loss_vector(data.y_true, data[model])
            for reference in references:
                result = block_difference(loss - binary_log_loss_vector(data.y_true, data[reference]),
                                          pd.to_datetime(data.date).dt.strftime("%Y-%m"))
                segment_intervals.append(dict(BoN=bon, candidate=model, reference=reference, **result))
    pd.DataFrame(segment_scores).to_csv(args.output / "full_results_bon_metrics.csv", index=False)
    ci = pd.DataFrame(segment_intervals)
    ci.to_csv(args.output / "full_results_bon_comparisons.csv", index=False)
    emit("bon", f"LogLoss według formatu: LR i MLP ({stage} etap) oraz odniesienia na wspólnej próbie rynku.",
         "tab:final_model_bon_segment_logloss", ["Format", "Mecze", "LR", "MLP", "OPEN", "CLOSE", "Glicko-2"], bon_rows)
    emit("bon_intervals", f"Różnice LogLoss modeli LR i MLP{stage_suffix} "
         "względem prognoz OPEN i CLOSE w formatach Bo1, Bo3 i Bo5. "
         r"Nominalne 95\% przedziały wyznaczono w 5\,000 replikach bootstrapu blokowego; "
         "jednostką losowania jest miesiąc. Meczom nadano jednakowe wagi. "
         "Nie zastosowano korekty na wielokrotne porównania.",
         "tab:full_results_bon_intervals", ["Format", "Model", "Odniesienie", r"$\Delta$ LL", r"95\% przedział"],
         [[f"Bo{int(r.BoN)}", short_names[r.candidate], short_names[r.reference], fmt(r.delta),
           f"[{fmt(r.ci_lower)}; {fmt(r.ci_upper)}]"] for r in ci.itertuples()
          if r.reference != "p_player_gl"], spec="lllrr", short_caption="Różnice LogLoss według formatu serii")
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), sharey=True)
    bon_references = ["market_open", "market_close"]
    x, width = np.arange(3), .28
    for ax, model in zip(axes, primary_models):
        for j, reference in enumerate(bon_references):
            z = ci[(ci.candidate == model) & (ci.reference == reference)].sort_values("BoN")
            advantage, low, high = -z.delta.to_numpy(), -z.ci_upper.to_numpy(), -z.ci_lower.to_numpy()
            ax.bar(x + (j - .5) * width, advantage, width, label=short_names[reference],
                   yerr=np.vstack([advantage - low, high - advantage]), capsize=3, color=palette[reference])
        ax.axhline(0, color=DARK_TEXT, linewidth=.8)
        ax.set_xticks(x, ["Bo1", "Bo3", "Bo5"])
        ax.set_title(short_names[model])
        ax.legend(fontsize=8)
    axes[0].set_ylabel("LogLoss odniesienia minus metamodelu")
    save(fig, "bon_advantage")

    years, year_rows = [], []
    for cohort, frame, label in (("sports", panel, "Pełna"), ("market_common", common, "Z kursami")):
        for year, data in frame.groupby(pd.to_datetime(frame.date).dt.year):
            models = [mlp, focal] + (["market_open", "market_close"] if cohort == "market_common" else [])
            scores = {model: float(binary_log_loss_vector(data.y_true, data[model]).mean()) for model in models}
            years.extend(dict(cohort=cohort, year=int(year), model=model, n=len(data), logloss=score)
                         for model, score in scores.items())
            year_rows.append([label, year, f"{len(data):,}".replace(",", r"\,"), fmt(scores[mlp]), fmt(scores[focal]),
                              fmt(scores["market_open"]) if "market_open" in scores else "--",
                              fmt(scores["market_close"]) if "market_close" in scores else "--"])
    years = pd.DataFrame(years)
    years.to_csv(args.output / "full_results_year_metrics.csv", index=False)
    emit("years", f"LogLoss modeli LR i MLP{stage_suffix} oraz prognoz rynkowych "
         "w poszczególnych latach. Rok 2026 jest niepełny: pełna próba kończy się we wrześniu, "
         "a wspólna próba z kursami w kwietniu.", "tab:full_results_years",
         ["Próba", "Rok", r"$N$", "MLP", "LR", "OPEN", "CLOSE"], year_rows,
         short_caption="LogLoss w poszczególnych latach")
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), sharey=True)
    for ax, cohort, title in zip(axes, ("sports", "market_common"), ("Pełna próba sportowa", "Wspólna próba rynkowa")):
        for model in (mlp, focal, "market_open", "market_close"):
            data = years[(years.cohort == cohort) & (years.model == model)].sort_values("year")
            if len(data):
                ax.plot(data.year.astype(str), data.logloss, marker="o", label=names[model], color=palette[model])
        ax.set_title(title)
        ax.set_xlabel("Rok")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("LogLoss")
    save(fig, "year_stability")

    weighting_rows, weighting_data = [], []
    for model in primary_models:
        for reference in ("market_open", "market_close"):
            result = block_difference(binary_log_loss_vector(common.y_true, common[model])
                                      - binary_log_loss_vector(common.y_true, common[reference]),
                                      pd.to_datetime(common.date).dt.strftime("%Y-%m"))
            weighting_data.append(dict(candidate=model, reference=reference, **result))
            for title, prefix in (("Mecze", ""), ("Miesiące", "equal_block_")):
                weighting_rows.append([short_names[model], short_names[reference], title, fmt(result[prefix + "delta"]),
                                       f"[{fmt(result[prefix + 'ci_lower'])}; {fmt(result[prefix + 'ci_upper'])}]"])
    pd.DataFrame(weighting_data).to_csv(args.output / "full_results_weighting.csv", index=False)
    emit("weighting", "Dwie wielkości oceniane w tych samych 27 miesiącach: "
         f"różnice LR i MLP ({stage} etap) minus rynek ważone meczami oraz średnie różnic miesięcznych.",
         "tab:full_results_weighting",
         ["Model", "Odniesienie", "Równe wagi", r"$\Delta$ LL", r"95\% przedział"], weighting_rows, spec="lllrr")

    hypotheses = [
        ["H1", "Ratingi zawodników i organizacji", "Niższy LogLoss ratingów zawodniczych we wszystkich sześciu badanych wariantach."],
        ["H2", "Metamodele i Glicko-2", "Niższy LogLoss obu metamodeli na pełnej próbie; przedziały różnic leżą ściśle poniżej zera."],
        ["H3", "Historyczne cechy drużynowe", "Usunięcie tych cech pogarsza LogLoss obu modeli. Zysk z kontekstu jest statystycznie istotny w LR ($\\Delta\\mathrm{LL}=+0{,}0032$) oraz w zespole 20 ziaren MLP ($\\Delta\\mathrm{LL}=+0{,}0034$, CI $[0{,}0015; 0{,}0054]$). BoN oceniono osobno."],
        ["P4", "Prognozy modeli i rynku", "Względem OPEN przewaga obu modeli jest statystycznie istotna (zysk $\\approx -0{,}023$, $p < 0{,}01$). Wobec CLOSE modele uzyskują wysoce konkurencyjną jakość punktową (zysk $\\approx -0{,}014$ do $-0{,}015$)."],
    ]
    emit("hypotheses", "Ocena hipotez H1--H3 i odpowiedź na pytanie P4.",
         "tab:hypotheses_verification", ["Ozn.", "Porównanie", "Wniosek w zakresie wykonanych analiz"], hypotheses,
         spec=r"p{.08\textwidth}p{.31\textwidth}p{.49\textwidth}")
    if args.corrections:
        outputs.extend(correction_assets(args.corrections, metric, comparisons, generated, figures, args.output))
    outputs.extend(historical_tables(metric, generated))

    # Consumer-visible numbers are cross-checked against the actual prediction panel.
    checks = []
    for cohort, frame in (("sports", panel), ("market_common", common)):
        for model in order:
            if cohort == "sports" and model.startswith("market_"):
                continue
            score, _ = _metrics(frame.y_true.to_numpy(int), frame[model].to_numpy(float))
            expected = metric(cohort, model)
            for field in ("logloss", "brier", "auc", "ece"):
                if not np.isclose(score[field], expected[field], atol=1e-12, rtol=0):
                    raise ValueError(f"Stored metric disagrees with panel: {cohort}/{model}/{field}")
            checks.append(dict(cohort=cohort, model=model, **score))
    pd.DataFrame(checks).to_csv(args.output / "full_results_metric_checks.csv", index=False)
    if any(hashlib.sha256(p.read_bytes()).hexdigest() != hashes[str(p)] for p in paths):
        raise ValueError("Frozen inputs changed during report generation")
    outputs += [p for p in args.output.iterdir() if p.is_file()]
    receipt = {
        "report_models": list(primary_models), "feature_version": version,
        "scope": "Frozen predictions, reporting only; original2023 MLP selection unchanged; no new training.",
        "interval_policy": "Paired monthly5000 seed20260921; nominal, not selection-adjusted.",
        "inputs": hashes, "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "sports_n": len(panel), "market_n": len(common),
        "sports_dates": [str(pd.to_datetime(panel.date).min().date()), str(pd.to_datetime(panel.date).max().date())],
        "market_dates": [str(pd.to_datetime(common.date).min().date()), str(pd.to_datetime(common.date).max().date())],
        "sports_months": int(pd.to_datetime(panel.date).dt.to_period("M").nunique()),
        "market_months": int(pd.to_datetime(common.date).dt.to_period("M").nunique()),
        "outputs": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in outputs},
    }
    (args.output / "full_results_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({key: receipt[key] for key in ("report_models", "sports_n", "market_n", "sports_dates", "market_dates")}, indent=2))
    print(pd.DataFrame(bon_rows).to_string(index=False))
    print(ci[["BoN", "candidate", "reference", "delta", "ci_lower", "ci_upper"]].to_string(index=False))


if __name__ == "__main__":
    main()
