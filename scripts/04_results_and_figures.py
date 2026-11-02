"""Write the summary tables and draw Figures 2 to 8 from the outputs of the previous scripts."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from model_transfer import figures, summary
from model_transfer.data import load_tables
from model_transfer.storage import load_output

tables = load_tables()
baseline = load_output("baseline")
uncertainty = load_output("transfer_uncertainty")
centroid = load_output("transfer_centroid")
selection_baseline = load_output("selection_baseline")
selection_transfer = load_output("selection_transfer")

summary.write_csv(summary.baseline_table(tables, baseline), "baseline")
summary.write_csv(summary.baseline_table(tables, selection_baseline), "baseline_feature_selection")
summary.write_csv(summary.transfer_steps_table(uncertainty), "transfer_steps_uncertainty")
summary.write_csv(summary.transfer_steps_table(centroid), "transfer_steps_centroid")
summary.write_csv(summary.optimal_ratios_table(uncertainty), "optimal_ratios")
summary.write_csv(summary.selected_samples_table(tables, uncertainty), "selected_samples_uncertainty")
summary.write_csv(summary.selected_samples_table(tables, centroid), "selected_samples_centroid")
summary.write_csv(summary.transfer_selection_table(uncertainty, selection_transfer), "transfer_feature_selection")
summary.write_csv(
    summary.retained_bands_table(selection_baseline, uncertainty, selection_transfer), "retained_bands"
)

figures.figure_2(tables, baseline)
figures.figure_3(tables, uncertainty)
figures.figure_4(uncertainty)
figures.figure_5(uncertainty, centroid)
figures.figure_6(tables, uncertainty, baseline)
figures.figure_7(baseline, selection_baseline, uncertainty, selection_transfer)
figures.figure_8(tables, selection_transfer, selection_baseline)
