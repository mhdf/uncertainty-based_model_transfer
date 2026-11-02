"""Transfer the models by selecting samples with and without uncertainty (Section 2.5).

With --original-code, the number of PLS factors is chosen as in the code used for the article.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from model_transfer.data import load_tables
from model_transfer.storage import load_output, save_output
from model_transfer.transfer import run_transfer

original = "--original-code" in sys.argv
rule = "original" if original else "corrected"
suffix = "_original" if original else ""

tables = load_tables()
baseline = load_output("baseline")

uncertainty = run_transfer(tables, baseline, selection="uncertainty", factor_rule=rule)
save_output(f"transfer_uncertainty{suffix}", uncertainty)

best = [[case["best"] for case in dataset] for dataset in uncertainty]
centroid = run_transfer(tables, baseline, selection="centroid", best=best, factor_rule=rule)
save_output(f"transfer_centroid{suffix}", centroid)
