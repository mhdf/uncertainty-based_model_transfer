<div align="justify">

# Uncertainty-based model transfer

Code and data to reproduce the analysis of the article:

> Franceschini, M.H.D., Becker, R., Wichern, F. and Kooistra, L., 2022. Quantification of Grassland Biomass and Nitrogen Content through UAV Hyperspectral Imagery - Active Sample Selection for Model Transfer. *Drones*, 6(3), 73. https://doi.org/10.3390/drones6030073

The study estimates grassland dry matter, nitrogen content and nitrogen uptake from drone (UAV) hyperspectral images. Models fitted in one year perform poorly in another year, so a small number of samples from the new year (the target domain) is added to the data of the original year (the source domain). This code selects those samples using their spectral diversity and the uncertainty of their predictions.

## How the method works

1. **Baseline models.** Partial Least Squares Regression (PLSR, a regression suited to many correlated bands) is fitted 100 times on random subsets of the data. The spread of the 100 predictions of a sample is its prediction uncertainty.
2. **Sample selection.** For each target date, k-means clustering splits the samples into two spectrally different groups, and the sample with the largest uncertainty in each group is added to the training data. Up to half of the target samples are added, two at a time, refitting the models after each addition.
3. **Number of samples.** The number retained is taken from an interval around the average uncertainty across all additions.
4. **Feature selection.** Bands are retained when the lower limit of the confidence interval of their Variable Importance in the Projection (VIP, a score of how much a band contributes to the model) is above one, and the models are refitted with those bands.

## Reproducing the results

Create the environment with [conda](https://docs.conda.io/) and run the scripts in order from the repository folder. The full analysis took about 5 minutes when tested.

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

The figures show the same data as the article with a simpler layout. Figure 1 (the study site) and Figures 9 and 10 (the maps, which need the images in the dataset) are not included.

## Repository contents

| Folder | Content |
|---|---|
| `data/` | The two tables used by the models, from the published dataset |
| `src/model_transfer/` | The analysis code |
| `scripts/` | The steps to run |
| `results/`, `figures/` | The tables and figures obtained with this code |
| `tests/` | Checks, with the values of the original run in `tests/expected/` |
| `tools/` | The script that exported those values, which needs the original files |
| `original_notebooks/` | The original notebooks, kept for reference only |

The complete dataset, including the hyperspectral images, is available at https://doi.org/10.4121/19188872.v1. `data/README.md` explains the tables.

## How close is the reproduction?

This code is a rewrite of the original notebooks for current Python libraries. It was compared with the saved outputs of the run used in the article (internally named v106):

| Stage | Result |
|---|---|
| Baseline, 9 models | Identical, including the number of PLSR factors of each repetition |
| Transfer with uncertainty, 100 selection steps | Identical samples at every step |
| Feature selection, 24 models | Identical bands |
| Transfer without uncertainty (Figure 5), 100 selection steps | Identical at all the numbers of samples used in the article; 2 later steps differ |

"Identical" means that R², RMSE, VIP and predictions differ by less than 1e-8, which is numerical noise. The tests check the tables in `results/` against the original values.

Two details were needed to obtain the same samples:

- **Start of k-means.** scikit-learn 1.3 changed how k-means starts, which alters some clusters for the same random seed, so `transfer.py` keeps the earlier procedure.
- **Ties in the comparison without uncertainty.** When a cluster has two samples, both are equally distant from its centroid, so numerical noise decided between them in the original run. Here the first sample is always taken, so that the result does not depend on the computer. This is why 2 steps differ (N content, source 2017, DOY 288 of 2014). They come after the number of samples used for that date and do not enter any published result.

## Differences between the code and the text of the article

Two behaviours of the original code are kept so that the published values are reproduced, although they differ from what the text suggests.

**Number of PLSR factors of the transferred models.** The factors are meant to minimize the summed RMSE of source and added target samples. In the original code these sums are sorted before the minimum is located, and its position is then used as if they were not sorted (`published_factor_choice` in `transfer.py`). Using the unsorted sums changes 27% of these decisions. At the selected number of samples, RMSE changes between -10% and +5% (-1% on average) and the number of samples changes in 4 of the 15 cases. This comparison covers the transfer with uncertainty only.

**Percentages in Figure 3.** The caption describes the ratio between the RMSE of the optimal and of the selected number of samples. The published values are that ratio for the standard deviation of the RMSE. The RMSE ratio, between 81% and 100%, is in `results/optimal_ratios.csv`. This means the selected number of samples comes closer to the lowest RMSE than the "20-30%" stated in Section 3.1 suggests.

## Citation and license

Please cite the article above when using this code. Citation details are in `CITATION.cff`.

The code is released under the [MIT license](LICENSE). The data are released under the CC BY-SA 4.0 license by their creators, as described in `data/README.md`.

Parts of the code, tests and documentation were written with the help of an AI coding assistant (Claude, by Anthropic). The author directed the work, checked the results and is responsible for the content.

</div>
