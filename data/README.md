<div align="justify">

# Data

These two tables are copies of files from the published dataset:

> Kooistra, L., Franceschini, M.H.D., Becker, R., Wichern, F., Suomalainen, J., Bartholomeus, H. and Capolupo, A., 2022. Haus Riswick grassland experiment with N fertilization and plant growth monitoring based on Unmanned Aerial Vehicle (UAV) hyperspectral imagery - campaigns of 2014 and 2017. 4TU.ResearchData. https://doi.org/10.4121/19188872.v1

They are shared under the [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) license, which also applies to changes made to them. The MIT license of the code does not cover them.

| File | Level | Rows | Used for |
|---|---|---|---|
| `20220215_Traits_Spectra_PlotLevel.csv` | Plot | 245 (49 plots, 5 dates) | Dry matter |
| `20220215_Traits_Spectra_TreatLevel.csv` | Treatment | 75 (15 treatments, 5 dates) | N content and N uptake |

## Columns

Columns are separated by semicolons.

| Column | Meaning |
|---|---|
| `Date` | Harvest date, as year, month and day |
| `Year` | Year of the data acquisition |
| `Treatment` | N fertilization treatment, explained in the README of the dataset |
| `Replicate` | Repetition of the treatment (plot level only) |
| `DM_dt_ha` | Dry above-ground biomass, in dt/ha |
| `N_perc_DM` | Nitrogen content, in % of the dry matter |
| `N_kg_ha` | Nitrogen uptake, in kg/ha |
| `Band_*` | Reflectance from 0 to 1, in sequence from 450 nm |

## Before using the tables

- **Bands.** The tables have 99 bands, from 450 to 940 nm every 5 nm, smoothed to a bandwidth (full width at half maximum) of 30 nm. The images in the dataset have 101 bands, from 450 to 950 nm, without this smoothing.
- **Column name.** `Band_010` is the tenth band, between `Band_09` and `Band_11`.
- **Dates.** Three dates are one day later than the UAV flights of the dataset, which took place on 2014-10-14, 2017-05-09 and 2017-08-29.

</div>
