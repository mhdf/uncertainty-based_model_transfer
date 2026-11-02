"""Refit the baseline and transferred models using only the selected bands (Section 2.6).

With --original-code, the number of PLS factors is chosen as in the code used for the article.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from model_transfer.data import load_tables
from model_transfer.feature_selection import run_baseline_selection, run_transfer_selection
from model_transfer.storage import load_output, save_output

original = "--original-code" in sys.argv
rule = "original" if original else "corrected"
suffix = "_original" if original else ""

tables = load_tables()
save_output("selection_baseline", run_baseline_selection(tables, load_output("baseline")))
transfer = load_output(f"transfer_uncertainty{suffix}")
save_output(f"selection_transfer{suffix}", run_transfer_selection(tables, transfer, factor_rule=rule))
