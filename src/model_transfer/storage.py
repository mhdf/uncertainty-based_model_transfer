"""Save and load the full outputs of each stage, which are generated locally and not shared."""
import pickle

from . import config


def save_output(name, content):
    config.OUTPUT_DIR.mkdir(exist_ok=True)
    with open(config.OUTPUT_DIR / f"{name}.pkl", "wb") as file:
        pickle.dump(content, file)


def load_output(name):
    path = config.OUTPUT_DIR / f"{name}.pkl"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found: run the previous script first")
    with open(path, "rb") as file:
        return pickle.load(file)
