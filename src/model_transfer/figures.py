"""Draw Figures 2 to 8 of the article, with the same data and a simplified layout."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.patches import Patch, Rectangle
from scipy import stats

from . import config, summary
from .data import get_domains

TRAIT_LABELS = ("Dry matter [dt/ha]", "N content [% in DM]", "N-uptake [kg/ha]")
DATE_COLORS = {
    "20140515": plt.get_cmap("Greens")(0.70),
    "20141015": plt.get_cmap("Greens")(0.50),
    "20170510": "blue",
    "20170830": "dodgerblue",
    "20171026": "lightskyblue",
}
SCATTER_LIMITS = ((0, 60), (0.9, 3.9), (-15, 105))
VALIDATION_SOURCE = "red"
VALIDATION_TARGET = "goldenrod"


def _date_label(date):
    return f"DOY {config.DOY[date]} ({date[:4]})"


def _date_legend(extra=()):
    handles = [Patch(color=DATE_COLORS[date], label=_date_label(date)) for date in config.DATES]
    return handles + list(extra)


def _save(figure, name):
    config.FIGURES_DIR.mkdir(exist_ok=True)
    figure.savefig(config.FIGURES_DIR / f"{name}.png", dpi=200, bbox_inches="tight", facecolor="w")
    plt.close(figure)


def _full_iqr(tables, trait_index):
    source, _ = get_domains(tables, trait_index)
    return stats.iqr(source.y)


def _cumulative_iqr(tables, trait_index, year, dates, date_index):
    """Return the IQR of the source samples plus the target dates up to date_index."""
    source, target = get_domains(tables, trait_index, year)
    in_target = np.isin(target.dates, dates[: date_index + 1])
    return stats.iqr(np.concatenate((source.y, target.y[in_target])))


def _best_step(case, date_index):
    return case["steps"][date_index][case["best"][date_index]]


def figure_2(tables, baseline):
    """R2 and relative RMSE of each dataset, on its own year and on the other year."""
    names = ("2014", "2017", "All data")
    figure, axes = plt.subplots(1, 3, figsize=(12, 4.4), constrained_layout=True)
    for trait_index, ax in enumerate(axes):
        iqr = _full_iqr(tables, trait_index)
        image = np.full((3, 3), np.nan)
        for dataset in range(3):
            case = baseline[dataset][trait_index]
            cells = [(dataset, case["r2_validation"], case["rmse_validation"])]
            if dataset < 2:
                _, target = get_domains(tables, trait_index, config.SOURCE_YEARS[dataset])
                sse = ((target.y[:, None] - case["pred_target"]) ** 2).sum(axis=0)
                sst = ((target.y - target.y.mean()) ** 2).sum()
                cells.append((1 - dataset, 1 - sse / sst, np.sqrt(sse / len(target.y))))
            for column, r2, rmse in cells:
                value = max(np.mean(r2), 0.0)
                image[dataset, column] = value
                ax.text(
                    column,
                    dataset,
                    f"{value:.2f}\n[{np.mean(rmse) / iqr:.2f}\n±{np.std(rmse) / iqr:.2f}]",
                    ha="center",
                    va="center",
                    color="white" if value > 0.55 else "black",
                )
        mesh = ax.imshow(image, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(3), names)
        ax.set_yticks(range(3), names)
        ax.set_xlabel("Target data")
        ax.set_title(TRAIT_LABELS[trait_index])
    axes[0].set_ylabel("Source data")
    figure.colorbar(mesh, ax=axes, shrink=0.8, label="$R^2$ [RMSE/IQR ± std]")
    _save(figure, "figure_2")


def figure_3(tables, transfer):
    """Performance after transfer, with the ratio between optimal and selected steps."""
    figure, axes = plt.subplots(2, 3, figsize=(12, 7.5), constrained_layout=True)
    for trait_index in range(3):
        labels, r2, source_rmse, source_std, full_rmse, full_std, ratios = [], [], [], [], [], [], []
        full_iqr = _full_iqr(tables, trait_index)
        for dataset, year in enumerate(config.SOURCE_YEARS):
            case = transfer[dataset][trait_index]
            for date_index, date in enumerate(case["dates"]):
                outputs = _best_step(case, date_index)
                iqr = _cumulative_iqr(tables, trait_index, year, case["dates"], date_index)
                factors = stats.mode(outputs["factors"], keepdims=False).mode
                labels.append(f"{date[:4]} DOY {config.DOY[date]} [{factors}]")
                r2.append(np.mean(outputs["r2_independent_validation"]))
                source_rmse.append(np.mean(outputs["rmse_validation"]) / full_iqr)
                source_std.append(np.std(outputs["rmse_validation"]) / full_iqr)
                full_rmse.append(np.mean(outputs["rmse_independent_validation"]) / iqr)
                full_std.append(np.std(outputs["rmse_independent_validation"]) / iqr)
                ratios.append(summary.optimal_ratios(case, date_index)["std_ratio"])

        positions = np.arange(len(labels))
        ax = axes[0, trait_index]
        ax.barh(positions, r2, color="deepskyblue", edgecolor="k")
        ax.set_yticks(positions, labels)
        ax.invert_yaxis()
        ax.set_xlim(0, 1)
        ax.set_xlabel("Average $R^2$")
        ax.set_title(TRAIT_LABELS[trait_index])

        ax = axes[1, trait_index]
        ax.barh(positions - 0.2, source_rmse, 0.38, xerr=source_std, color="lime", edgecolor="k",
                label="Source validation samples")
        ax.barh(positions + 0.2, full_rmse, 0.38, xerr=full_std, color="deepskyblue", edgecolor="k",
                label="Source and target validation samples")
        ax.axvline(1.02, color="k", linewidth=0.8)
        for position, ratio in zip(positions, ratios):
            ax.text(1.05, position, f"{ratio:.1%}", va="center", color="crimson")
        ax.set_yticks(positions, labels)
        ax.invert_yaxis()
        ax.set_xlim(0, 1.4)
        ax.set_xticks(np.arange(0, 1.01, 0.25))
        ax.set_xlabel("Average RMSE / IQR")
    handles, names = axes[1, 0].get_legend_handles_labels()
    handles.append(Patch(color="crimson", label="Ratio published in Figure 3 (see the README)"))
    figure.legend(handles=handles, loc="outside lower center", ncol=3)
    _save(figure, "figure_3")


def figure_4(transfer):
    """RMSE as samples are added, marking the selected (Sel.) and optimal (Opt.) numbers."""
    figure = plt.figure(figsize=(12, 8), constrained_layout=True)
    grid = figure.add_gridspec(2, 2, height_ratios=[1, 1])
    placements = (grid[0, :], grid[1, 0], grid[1, 1])
    for trait_index, placement in enumerate(placements):
        inner = placement.subgridspec(3, 1, height_ratios=[1, 1, 7], hspace=0.05)
        optimal_ax, selected_ax, ax = (figure.add_subplot(inner[row]) for row in range(3))
        n_dates = sum(len(transfer[dataset][trait_index]["dates"]) for dataset in range(2))
        offset = 0
        for dataset in range(2):
            case = transfer[dataset][trait_index]
            for date_index, date in enumerate(case["dates"]):
                x_shift = (offset - (n_dates - 1) / 2) * 0.28
                means, deviations = [], []
                for step in case["steps"][date_index]:
                    means.append(np.mean(step["rmse_independent_validation"]))
                    deviations.append(np.std(step["rmse_independent_validation"]))
                samples = (np.arange(len(means)) + 1) * config.N_CLUSTERS
                ax.errorbar(samples + x_shift, means, yerr=deviations, fmt="none",
                            color=DATE_COLORS[date], capsize=3, linewidth=1.8)
                for strip, chosen in (
                    (optimal_ax, int(np.argmin(means))),
                    (selected_ax, case["best"][date_index]),
                ):
                    strip.bar(samples[chosen] + x_shift, 1, width=0.26, color=DATE_COLORS[date])
                offset += 1
        ax.set_xlabel("Number of samples from the target domain added to the training data")
        ax.set_ylabel(f"RMSE - {TRAIT_LABELS[trait_index]}")
        ax.set_xticks(samples)
        if trait_index == 0:
            ax.set_ylim(2, 8)
        for strip, label in ((optimal_ax, "Opt."), (selected_ax, "Sel.")):
            strip.set_xlim(ax.get_xlim())
            strip.set_ylim(0, 1)
            strip.set_xticks([])
            strip.set_yticks([0.5], [label])
    figure.legend(handles=_date_legend(), loc="outside lower center", ncol=5, title="Target dataset")
    _save(figure, "figure_4")


def figure_5(transfer_uncertainty, transfer_centroid):
    """RMSE when selecting samples with (y) and without (x) prediction uncertainty."""
    figure, axes = plt.subplots(1, 3, figsize=(12, 4.6), constrained_layout=True)
    for trait_index, ax in enumerate(axes):
        limits = []
        for dataset in range(2):
            with_unc = transfer_uncertainty[dataset][trait_index]
            without = transfer_centroid[dataset][trait_index]
            for date_index, date in enumerate(with_unc["dates"]):
                y = _best_step(with_unc, date_index)["rmse_independent_validation"]
                x = _best_step(without, date_index)["rmse_independent_validation"]
                ax.errorbar(np.mean(x), np.mean(y), xerr=np.std(x), yerr=np.std(y), fmt="none",
                            color=DATE_COLORS[date], capsize=3, linewidth=1.8)
                limits += [np.mean(x) + np.std(x), np.mean(y) + np.std(y),
                           np.mean(x) - np.std(x), np.mean(y) - np.std(y)]
        low, high = min(limits) * 0.95, max(limits) * 1.05
        ax.plot([low, high], [low, high], "k-", linewidth=0.8)
        ax.set_xlim(low, high)
        ax.set_ylim(low, high)
        ax.set_aspect("equal")
        ax.grid(alpha=0.3, linestyle=":")
        ax.set_xlabel(f"RMSE - {TRAIT_LABELS[trait_index]}\n(diversity)")
        ax.set_ylabel(f"RMSE - {TRAIT_LABELS[trait_index]}\n(diversity and uncertainty)")
    figure.legend(handles=_date_legend(), loc="outside lower center", ncol=5, title="Target dataset")
    _save(figure, "figure_5")


def _prediction_bars(ax, predictions, observed, rows, colors):
    means = np.nanmean(predictions[rows], axis=1)
    deviations = np.nanstd(predictions[rows], axis=1)
    for mean, deviation, value, color in zip(means, deviations, observed[rows], colors):
        ax.errorbar(mean, value, xerr=deviation, fmt="none", color=color, capsize=1.4, linewidth=1.1)
    return means


def _scatter_panel(ax, limits, title, r2, rmse, predicted, observed):
    slope, intercept = np.polyfit(predicted, observed, 1)
    line = np.array(limits)
    ax.plot(line, line, "k-", linewidth=0.8)
    ax.plot(line, intercept + slope * line, "--", color="red", linewidth=0.8)
    ax.set_xlim(limits)
    ax.set_ylim(limits)
    ax.set_aspect("equal")
    ax.grid(alpha=0.3, linestyle=":")
    # Rounding twice, as in the published figures, so that the labels show the same values.
    ax.set_title(f"{title}\n$R^2$ = {round(max(r2, 0), 3):.2f} ; RMSE = {round(rmse, 3):.2f}", fontsize=10)


def _scatter_figure(tables, transferred, benchmark, name):
    """Sample-wise predictions; transferred holds [dataset][trait] -> the model of the last date."""
    figure, axes = plt.subplots(3, 3, figsize=(11, 13), constrained_layout=True)
    figure.get_layout_engine().set(h_pad=0.1)
    for trait_index in range(3):
        for dataset, year in enumerate(config.SOURCE_YEARS):
            ax = axes[trait_index, dataset]
            outputs = transferred[dataset][trait_index]
            source, target = get_domains(tables, trait_index, year)
            validation = np.where(~np.isnan(outputs["pred_validation"]).all(axis=1))[0]
            calibration = np.setdiff1d(np.arange(len(source)), validation)
            unused = np.where(~np.isnan(outputs["pred_target"]).all(axis=1))[0]
            _prediction_bars(ax, outputs["pred_fit"], source.y, calibration,
                             [DATE_COLORS[date] for date in source.dates[calibration]])
            val_means = _prediction_bars(ax, outputs["pred_validation"], source.y, validation,
                                         [VALIDATION_SOURCE] * len(validation))
            target_means = _prediction_bars(ax, outputs["pred_target"], target.y, unused,
                                            [VALIDATION_TARGET] * len(unused))
            other = config.SOURCE_YEARS[1 - dataset]
            _scatter_panel(
                ax,
                SCATTER_LIMITS[trait_index],
                f"Source {year} - target {other}",
                np.mean(outputs["r2_independent_validation"]),
                np.mean(outputs["rmse_independent_validation"]),
                np.append(val_means, target_means),
                np.append(source.y[validation], target.y[unused]),
            )

        ax = axes[trait_index, 2]
        case = benchmark[2][trait_index]
        source, _ = get_domains(tables, trait_index)
        calibration, validation = case["calibration"], case["validation"]
        _prediction_bars(ax, case["pred_fit"], source.y, calibration,
                         [DATE_COLORS[date] for date in source.dates[calibration]])
        val_means = _prediction_bars(ax, case["pred_validation"], source.y, validation,
                                     [VALIDATION_SOURCE] * len(validation))
        _scatter_panel(
            ax,
            SCATTER_LIMITS[trait_index],
            "All data - reference",
            np.mean(case["r2_validation"]),
            np.mean(case["rmse_validation"]),
            val_means,
            source.y[validation],
        )
        for column in range(3):
            axes[trait_index, column].set_xlabel(f"{TRAIT_LABELS[trait_index]} - predicted")
            axes[trait_index, column].set_ylabel(f"{TRAIT_LABELS[trait_index]} - observed")
    extra = (Patch(color=VALIDATION_SOURCE, label="Validation (source)"),
             Patch(color=VALIDATION_TARGET, label="Validation (target)"))
    figure.legend(handles=_date_legend(extra), loc="outside lower center", ncol=4, title="Dataset")
    _save(figure, name)


def figure_6(tables, transfer, baseline):
    """Sample-wise accuracy and uncertainty using all the bands."""
    transferred = [
        [_best_step(case, len(case["dates"]) - 1) for case in per_trait] for per_trait in transfer
    ]
    _scatter_figure(tables, transferred, baseline, "figure_6")


def figure_7(baseline, baseline_selection, transfer, transfer_selection):
    """Average VIP per band, with the retained bands marked on the left of each column."""
    names = ("2014", "2017", "All data", "2014+", "2017+")
    wavelengths = config.WAVELENGTHS_NM
    # Symmetric around VIP = 1, the usual threshold, which falls on the neutral color.
    norm = Normalize(vmin=-0.3, vmax=2.3)
    edges = np.append(wavelengths - 2.5, wavelengths[-1] + 2.5)
    figure, axes = plt.subplots(1, 3, figsize=(12, 5.5), constrained_layout=True)
    for trait_index, ax in enumerate(axes):
        columns = []
        for dataset in range(3):
            columns.append(
                (
                    np.nanmean(baseline[dataset][trait_index]["vip"], axis=1),
                    baseline_selection[dataset][trait_index]["bands"],
                )
            )
        for dataset in range(2):
            case = transfer[dataset][trait_index]
            last = len(case["dates"]) - 1
            columns.append(
                (
                    np.nanmean(_best_step(case, last)["vip"], axis=1),
                    transfer_selection[dataset][trait_index][last]["bands"],
                )
            )
        for position, (mean_vip, bands) in enumerate(columns):
            mesh = ax.pcolormesh([position, position + 0.62], edges, mean_vip[:, None],
                                 cmap="PuOr_r", norm=norm)
            ax.hlines(wavelengths[bands], position - 0.22, position - 0.02, color="blue", linewidth=1.5)
        ax.add_patch(Rectangle((1.72, edges[0]), 1.0, edges[-1] - edges[0], fill=False, edgecolor="magenta",
                               linewidth=1.5))
        ax.set_xticks(np.arange(5) + 0.31, names, rotation=45)
        ax.set_xlim(-0.3, 4.75)
        ax.set_xlabel("Training dataset (source)")
        ax.set_ylabel("Wavelength [nm]")
        ax.set_title(TRAIT_LABELS[trait_index])
    colorbar = figure.colorbar(mesh, ax=axes, shrink=0.5, label="VIP")
    colorbar.ax.set_ylim(0.3, 2.3)
    colorbar.set_ticks([0.3, 1.0, 2.3])
    _save(figure, "figure_7")


def figure_8(tables, transfer_selection, baseline_selection):
    """Sample-wise accuracy and uncertainty using only the retained bands."""
    transferred = [[per_date[-1] for per_date in per_trait] for per_trait in transfer_selection]
    _scatter_figure(tables, transferred, baseline_selection, "figure_8")
