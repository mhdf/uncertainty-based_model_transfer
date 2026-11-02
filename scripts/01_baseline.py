"""Fit the bootstrap PLSR models of each year and of all dates (Section 2.4)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from model_transfer.baseline import run_baseline
from model_transfer.data import load_tables
from model_transfer.storage import save_output

save_output("baseline", run_baseline(load_tables()))
