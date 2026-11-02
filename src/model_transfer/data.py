"""Load the published tables and arrange them as source and target domains."""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config


@dataclass
class Domain:
    """Spectra and one trait for a group of samples, in table order."""

    X: np.ndarray
    y: np.ndarray
    dates: np.ndarray
    groups: list

    def __len__(self):
        return len(self.y)


def load_tables(data_dir=config.DATA_DIR):
    """Return the plot-level and treatment-level tables."""
    dtype = {"Date": str, "Treatment": str}
    plot = pd.read_csv(data_dir / config.PLOT_TABLE, sep=";", dtype=dtype)
    treatment = pd.read_csv(data_dir / config.TREATMENT_TABLE, sep=";", dtype=dtype)
    return {"plot": plot, "treatment": treatment}


def band_columns(table):
    """Return the spectral columns of a table."""
    return [column for column in table.columns if column.startswith("Band_")]


def table_of_trait(tables, trait_index):
    """Return the table at the spatial level used for a trait."""
    return tables["plot"] if trait_index == 0 else tables["treatment"]


def _domain(table, trait_index):
    if "Replicate" in table.columns:
        groups = (table["Replicate"].astype(str) + "_" + table["Treatment"]).tolist()
    else:
        groups = table["Treatment"].tolist()
    return Domain(
        X=table[band_columns(table)].to_numpy(),
        y=table[config.TRAITS[trait_index]].to_numpy(),
        dates=table["Date"].to_numpy(),
        groups=groups,
    )


def get_domains(tables, trait_index, source_year=None):
    """Return (source, target); without a source year, source holds every date."""
    table = table_of_trait(tables, trait_index)
    if source_year is None:
        return _domain(table, trait_index), None
    in_source = table["Year"] == source_year
    return _domain(table[in_source], trait_index), _domain(table[~in_source], trait_index)


def target_dates(source_year):
    """Return the acquisition dates of the other year."""
    return [date for date in config.DATES if not date.startswith(str(source_year))]
