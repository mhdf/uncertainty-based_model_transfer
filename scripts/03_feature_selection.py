"""Refit the baseline and transferred models using only the selected bands (Section 2.6)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from model_transfer.data import load_tables
from model_transfer.feature_selection import run_baseline_selection, run_transfer_selection
from model_transfer.storage import load_output, save_output

tables = load_tables()
save_output("selection_baseline", run_baseline_selection(tables, load_output("baseline")))
save_output("selection_transfer", run_transfer_selection(tables, load_output("transfer_uncertainty")))
