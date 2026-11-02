"""Transfer the models by selecting samples with and without uncertainty (Section 2.5)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from model_transfer.data import load_tables
from model_transfer.storage import load_output, save_output
from model_transfer.transfer import run_transfer

tables = load_tables()
baseline = load_output("baseline")

uncertainty = run_transfer(tables, baseline, selection="uncertainty")
save_output("transfer_uncertainty", uncertainty)

best = [[case["best"] for case in dataset] for dataset in uncertainty]
save_output("transfer_centroid", run_transfer(tables, baseline, selection="centroid", best=best))
