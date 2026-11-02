"""Check the pipeline against the original run and the statements made in the README."""
import sys
import unittest
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from model_transfer import config, summary  # noqa: E402
from model_transfer.baseline import run_baseline  # noqa: E402
from model_transfer.data import band_columns, load_tables  # noqa: E402
from model_transfer.feature_selection import vip_confidence_interval  # noqa: E402
from model_transfer.transfer import choose_factors, choose_number_of_steps, run_transfer  # noqa: E402

EXPECTED_DIR = ROOT / "tests" / "expected"
ORIGINAL_DIR = config.RESULTS_DIR / "original_code"
TOLERANCE = 1e-7
TEXT_COLUMNS = {"target_date": str, "treatment": str}
quiet = lambda message: None


def read(folder, name):
    return pd.read_csv(folder / f"{name}.csv", dtype=TEXT_COLUMNS)


class TableAssertions(unittest.TestCase):
    def assert_same_table(self, new, old):
        self.assertEqual(list(new.columns), list(old.columns))
        self.assertEqual(len(new), len(old))
        for column in new.columns:
            if pd.api.types.is_numeric_dtype(old[column]) and old[column].dtype != bool:
                np.testing.assert_allclose(
                    new[column], old[column], atol=TOLERANCE, rtol=0, equal_nan=True, err_msg=column
                )
            else:
                self.assertEqual(
                    new[column].fillna("").astype(str).tolist(),
                    old[column].fillna("").astype(str).tolist(),
                    column,
                )


class PublishedData(unittest.TestCase):
    def test_table_sizes(self):
        tables = load_tables()
        self.assertEqual(len(tables["plot"]), 245)
        self.assertEqual(len(tables["treatment"]), 75)
        self.assertEqual(len(band_columns(tables["plot"])), len(config.WAVELENGTHS_NM))
        self.assertEqual(len(band_columns(tables["treatment"])), len(config.WAVELENGTHS_NM))


class Criteria(unittest.TestCase):
    def test_step_closest_to_the_average_uncertainty(self):
        self.assertEqual(choose_number_of_steps([10.0, 6.0, 5.0, 4.5, 4.0, 3.5]), 2)

    def test_highest_uncertainty_is_dropped_until_a_step_is_inside(self):
        self.assertEqual(choose_number_of_steps([100.0, 2.0, 1.0]), 1)

    def test_vip_interval_keeps_the_bands_above_one(self):
        values = np.vstack((np.full(100, 2.0), np.full(100, 0.5)))
        values[0] += np.random.RandomState(0).normal(0, 0.1, 100)
        mean, lower, upper = vip_confidence_interval(values)
        self.assertGreater(lower[0], 1)
        self.assertLess(lower[1], 1)
        self.assertTrue((upper >= mean).all())

    def test_factor_rules(self):
        # Target errors 5, 6 and 4 sort as 6, 5, 4. The lowest sum is at the second factor
        # number, but it comes first after sorting.
        rmse_added = [5.0, 6.0, 4.0]
        rmse_source = [3.0, 1.0, 4.0]
        self.assertEqual(choose_factors(rmse_source, rmse_added, "corrected"), 1)
        self.assertEqual(choose_factors(rmse_source, rmse_added, "original"), 0)


class Recalculation(TableAssertions):
    """Repeat the baseline and one transfer case with each rule, which takes under a minute."""

    @classmethod
    def setUpClass(cls):
        warnings.simplefilter("ignore")
        cls.tables = load_tables()
        cls.baseline = run_baseline(cls.tables, progress=quiet)

    def transfer_case(self, factor_rule):
        results = run_transfer(
            self.tables, self.baseline, cases={(1, 2)}, factor_rule=factor_rule, progress=quiet
        )
        return summary.transfer_steps_table(results).reset_index(drop=True)

    @staticmethod
    def same_case(table):
        chosen = (table["source_year"] == 2017) & (table["trait"] == "N_kg_ha")
        return table[chosen].reset_index(drop=True)

    def test_baseline(self):
        self.assert_same_table(
            summary.baseline_table(self.tables, self.baseline), read(EXPECTED_DIR, "baseline")
        )

    def test_transfer_case_with_the_original_rule(self):
        expected = self.same_case(read(EXPECTED_DIR, "transfer_steps_uncertainty"))
        self.assert_same_table(self.transfer_case("original"), expected)

    def test_transfer_case_with_the_corrected_rule(self):
        stored = self.same_case(read(config.RESULTS_DIR, "transfer_steps_uncertainty"))
        self.assert_same_table(self.transfer_case("corrected"), stored)


class OriginalRun(TableAssertions):
    """Compare the tables of results/original_code with the saved outputs of the original run."""

    def test_all_uncertainty_steps(self):
        for name in ("transfer_steps_uncertainty", "selected_samples_uncertainty"):
            self.assert_same_table(read(ORIGINAL_DIR, name), read(EXPECTED_DIR, name))

    def test_centroid_comparison_at_the_selected_steps(self):
        new = read(ORIGINAL_DIR, "transfer_steps_centroid")
        old = read(EXPECTED_DIR, "transfer_steps_centroid")
        self.assert_same_table(
            new[new["selected_as_optimal"]].reset_index(drop=True),
            old[old["selected_as_optimal"]].reset_index(drop=True),
        )

    def test_only_the_known_centroid_steps_differ(self):
        # A tie in a two-sample cluster, explained at the end of the README.
        keys = ["source_year", "trait", "target_date", "samples_added"]
        new = read(ORIGINAL_DIR, "transfer_steps_centroid").set_index(keys)["rmse_independent"]
        old = read(EXPECTED_DIR, "transfer_steps_centroid").set_index(keys)["rmse_independent"]
        differing = new[(new - old).abs() > TOLERANCE].index.tolist()
        self.assertEqual(
            differing, [(2017, "N_perc_DM", "20141015", 6), (2017, "N_perc_DM", "20141015", 8)]
        )

    def test_feature_selection(self):
        for name in ("baseline_feature_selection", "transfer_feature_selection", "retained_bands"):
            self.assert_same_table(read(ORIGINAL_DIR, name), read(EXPECTED_DIR, name))

    def test_values_printed_in_the_article(self):
        # Figure 4: samples selected per target date. Figure 3: percentages in the pie charts.
        samples = {
            "DM_dt_ha": [14, 12, 10, 8, 12],
            "N_perc_DM": [6, 6, 4, 4, 4],
            "N_kg_ha": [6, 4, 8, 4, 4],
        }
        percentages = {
            "DM_dt_ha": [73.6, 73.2, 99.3, 80.8, 73.0],
            "N_perc_DM": [100.0, 100.0, 84.5, 100.0, 77.5],
            "N_kg_ha": [85.6, 87.7, 100.0, 82.9, 52.5],
        }
        steps = read(ORIGINAL_DIR, "transfer_steps_uncertainty")
        steps = steps[steps["selected_as_optimal"]]
        ratios = read(ORIGINAL_DIR, "optimal_ratios")
        for trait in config.TRAITS:
            self.assertEqual(steps.loc[steps["trait"] == trait, "samples_added"].tolist(), samples[trait])
            new = 100 * ratios.loc[ratios["trait"] == trait, "std_ratio"].to_numpy()
            np.testing.assert_allclose(new, percentages[trait], atol=0.05)


class CorrectedResults(unittest.TestCase):
    """Check the statements of the README about the corrected values in results/."""

    def setUp(self):
        self.uncertainty = read(config.RESULTS_DIR, "transfer_steps_uncertainty")
        self.centroid = read(config.RESULTS_DIR, "transfer_steps_centroid")
        self.selected = self.uncertainty[self.uncertainty["selected_as_optimal"]].reset_index(drop=True)

    def test_uncertainty_lowers_the_rmse_for_all_cases(self):
        centroid = self.centroid[self.centroid["selected_as_optimal"]].reset_index(drop=True)
        self.assertEqual(len(self.selected), 15)
        self.assertTrue((self.selected["rmse_independent"] < centroid["rmse_independent"]).all())

    def test_smallest_numbers_of_samples(self):
        samples = self.selected.groupby("trait")["samples_added"].min()
        self.assertEqual(samples["DM_dt_ha"], 8)
        self.assertEqual(samples["N_perc_DM"], 4)
        self.assertEqual(samples["N_kg_ha"], 4)

    def test_gap_to_the_lowest_rmse(self):
        gaps = 100 * (1 - read(config.RESULTS_DIR, "optimal_ratios")["rmse_ratio"])
        self.assertLess(gaps.max(), 20)
        self.assertLess(gaps.mean(), 6)


if __name__ == "__main__":
    unittest.main()
