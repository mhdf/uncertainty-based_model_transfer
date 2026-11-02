<div align="justify">

# Analysis notebooks

These are the notebooks used for the article, reorganized for reading. They are kept for reference only and do not run as they are: they need Python 3.6, library versions of that time, files that are not shared and, in places, manual changes between runs. To reproduce the results, use the scripts in the main folder.

They keep the behaviour of the code used for the article, including the two issues explained at the end of the [main README](../README.md).

| Notebook | Original file in the author's archive (`1_Spectors/1_JypiterLabProcessing/`) | Content |
|---|---|---|
| `01_spectra_extraction.ipynb` | `1_Correction_extraction_FullData.ipynb` | Selection of flight lines and extraction of the spectra of each plot |
| `02_baseline_models.ipynb` | `01_SpectralAnalysis_Part00.ipynb` | Baseline models, their feature selection, Figure 2 and reference panels of Figures 6 and 8 |
| `03_model_transfer.ipynb` | `01_SpectralAnalysis_Part02_01.ipynb` | Model transfer, feature selection, Figures 3 to 9 and files of the maps |

Each code cell is preceded by a short description with its position in the original file (for example, *Original cell 13*).

## What changed

- **Removed:** cells that were fully commented out, earlier versions of figures, a visual check that produced no output, commented-out code, and outdated notes. The removed cells are listed below.
- **Added:** headings, a description of each cell, and spelling corrections in comments.
- **Restored:** steps that had been turned off by commenting them out, namely saving the model outputs and exporting the tables of the published dataset. They now sit behind `SAVE_OUTPUTS` and `EXPORT_SHARED_TABLES`, which are off by default.
- **Replaced:** local folders, by `<PROJECT_ROOT>`, `<RAW_DATA>` and `<CONDA_ENVS>`. Saved cell outputs were also cleared.

The kept code itself was not rewritten, so variable names and long cells remain as in the original. The notebooks were not run again. Instead, each kept cell was compared automatically with the original using its abstract syntax tree (the parsed structure of the code, which ignores comments and layout). All of them are identical, apart from the restored blocks and from `02_baseline_models.ipynb`, where only two settings remain of an earlier figure (original cell 11), because later cells read them. Removed cells were also checked for effects on later cells, such as variables, imports and the working folder. They only change plotting styles, so some figures could look slightly different if run again.

## Removed cells

The numbers give the position among the code cells of the original file, counting from 1.

| Notebook | Cells | Reason |
|---|---|---|
| `01_spectra_extraction` | 4 | Visual check of the images, with no output |
| | 6, 10 | Fully commented out: manual adjustment of flight lines, and comparison of extraction versions |
| `02_baseline_models` | 14, 22 | VIP figures replaced by Figure 7 in notebook 03 |
| | 15 | Fully commented out: earlier version of the maps |
| `03_model_transfer` | 9, 11, 15, 16, 19 | Fully commented out: earlier sample selection, baseline summary, tool to rename saved versions, earlier comparison of criteria, and earlier version of Figure 3 |
| | 23 | VIP figure replaced by Figure 7 |
| | 26, 27, 28, 29 | Fully commented out: selection by regression coefficients, not used in the article |
| | 36, 37, 49 | Earlier versions of Figures 7 and 9 |
| | 43, 44, 45, 47 | Fully commented out: unused comparison, earlier legend ranges, and maps from smoothed images |

The helper modules that the notebooks import (`FakeParulaPalette`, `RemoveIrregularPad_v3` and `VIs_formulation_update20190402`) are not included.

</div>
