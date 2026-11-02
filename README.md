<div align="justify">

# Uncertainty-based model transfer

Code and data to reproduce the analysis of the article:

> Franceschini, M.H.D., Becker, R., Wichern, F. and Kooistra, L., 2022. Quantification of Grassland Biomass and Nitrogen Content through UAV Hyperspectral Imagery - Active Sample Selection for Model Transfer. *Drones*, 6(3), 73. https://doi.org/10.3390/drones6030073

The study estimates grassland dry matter, nitrogen content and nitrogen uptake from drone (UAV) hyperspectral images. Models fitted in one year perform poorly in another year, so a small number of samples from the new year (the target domain) is added to the data of the original year (the source domain). This code selects those samples using their spectral diversity and the uncertainty of their predictions.

The values in this repository include two corrections, explained at the end of this page. They do not change the conclusions of the article.

## How the method works

1. **Baseline models.** Partial Least Squares Regression (PLSR, a regression suited to many correlated bands) is fitted 100 times on random subsets of the data. The spread of the 100 predictions of a sample is its prediction uncertainty.
2. **Sample selection.** For each target date, k-means clustering splits the samples into two spectrally different groups, and the sample with the largest uncertainty in each group is added to the training data. Up to half of the target samples are added, two at a time, refitting the models after each addition.
3. **Number of samples.** The number retained is taken from an interval around the average uncertainty across all additions.
4. **Feature selection.** Bands are retained when the lower limit of the confidence interval of their Variable Importance in the Projection (VIP, a score of how much a band contributes to the model) is above one, and the models are refitted with those bands.

## Main results

Values for the models transferred to each of the 5 target dates, at the selected number of samples, compared with the reference model trained on all dates. RMSE (root mean square error) is in the unit of each trait.

| Trait | Reference R² / RMSE | Samples added | R² | RMSE | RMSE reduction from using uncertainty | RMSE with retained bands |
|---|---|---|---|---|---|---|
| Dry matter (dt/ha) | 0.92 / 3.25 | 8-14 | 0.91-0.97 | 2.49-3.28 | 0.29-2.67 | 3.02-3.84 |
| N content (% of dry matter) | 0.58 / 0.27 | 4-6 | 0.52-0.77 | 0.16-0.28 | 0.03-0.07 | 0.18-0.26 |
| N uptake (kg/ha) | 0.91 / 6.50 | 4-6 | 0.87-0.95 | 4.74-7.75 | 0.98-6.17 | 6.18-9.58 |

- Selecting samples with uncertainty gives a lower average RMSE than using spectral diversity alone in all 15 cases.
- The RMSE of the best possible number of samples, known only after validation, is 0-17% lower than with the selected number (5% on average), so little accuracy is lost.

The complete values are in `results/` and Figures 2 to 8 in `figures/`.

## Reproducing the results

Create the environment with [conda](https://docs.conda.io/) and run the scripts in order from the repository folder. The analysis took about 5 minutes when tested.

```text
conda env create -f environment.yml
conda activate model-transfer
python scripts/01_baseline.py
python scripts/02_transfer.py
python scripts/03_feature_selection.py
python scripts/04_results_and_figures.py
python -m unittest discover -s tests -t tests
```

| Script | Article | Output |
|---|---|---|
| `01_baseline.py` | Section 2.4 | Models for 2014, 2017 and all dates |
| `02_transfer.py` | Section 2.5 | Transferred models, with and without uncertainty |
| `03_feature_selection.py` | Section 2.6 | Models using only the retained bands |
| `04_results_and_figures.py` | Figures 2 to 8 | Tables in `results/` and images in `figures/` |

The figures use a simpler layout than the article. Figure 1 (the study site) and Figures 9 and 10 (the maps, which need the images in the dataset) are not included.

## Repository contents

| Folder | Content |
|---|---|
| `data/` | The two tables used by the models, from the published dataset, explained in `data/README.md` |
| `src/model_transfer/` | The analysis code |
| `scripts/` | The steps to run |
| `results/`, `figures/` | The tables and figures obtained with this code |
| `results/original_code/` | The tables obtained with the code used for the article |
| `tests/` | Checks, with the values of the original run in `tests/expected/` |
| `tools/` | The script that exported those values, which needs the original files |
| `notebooks/` | The notebooks used for the article, reorganized and kept for reference only |

The complete dataset, including the hyperspectral images, is available at https://doi.org/10.4121/19188872.v1.

## Citation and license

Please cite the article above when using this code. Citation details are in `CITATION.cff`.

The code is released under the [MIT license](LICENSE). The data are released under the CC BY-SA 4.0 license by their creators, as described in `data/README.md`.

Parts of the code, tests and documentation were written with the help of an AI coding assistant (Claude, by Anthropic). The author directed the work, checked the results and is responsible for the content.

## Differences from the published article

**The conclusions of the article do not change, and the corrected values are, on balance, equal to or more favourable than the published ones.** Preparing this repository revealed two issues in the original analysis, both corrected here.

| Statement | Article | This repository |
|---|---|---|
| Best possible RMSE compared with the selected number of samples | 20-30% lower | **0-17% lower** (5% on average) |
| Smallest number of samples added (dry matter / N traits) | 8 / 4 | 8 / 4 |
| Uncertainty lowers the RMSE compared with diversity alone | All cases | All 15 cases |
| Cases where the two error bars in Figure 5 do not overlap | 8 of 15 | 10 of 15 |
| RMSE with the retained bands | - | Lower in 13 of 15 cases (5% on average) |
| RMSE with all bands | - | 1% lower on average |

**Percentages in Figure 3.** The published values, and the 20-30% in the text, are ratios of the standard deviation of the RMSE, not of the RMSE described in the caption. The RMSE itself is much closer to the best possible value, so the loss of accuracy is smaller than the article states.

**Number of PLSR factors of the transferred models.** A coding error located the lowest summed RMSE in a sorted list and then read its position as if the list were not sorted, which changed 27% of these choices. The baseline models and the reference model are not affected. With the correction, the RMSE with all bands is on average 4% lower for N content (lower in all 5 cases) and 2% lower for dry matter (lower in 2 of 5 cases), and 3% higher for N uptake (higher in 4 of 5 cases). The maps (Figures 9 and 10) were not recalculated.

**Checking the published values.** Running scripts 02 to 04 with the option `--original-code` restores the original behaviour and writes the tables to `results/original_code/`. The tests compare them with the saved outputs of the original run: R², RMSE and predictions differ by less than 1e-8. The only exception is the comparison without uncertainty, where a cluster of two samples can be an exact tie, which numerical noise decided in the original run. Here the first sample is taken, so 2 steps differ, and no published result uses them. The earlier way of starting k-means is also kept, because scikit-learn 1.3 changed it and the same random seed would otherwise select other samples.

</div>
