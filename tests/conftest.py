import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def pytest_configure(config):
    if getattr(config.option, "basetemp", None):
        return
    root = Path(__file__).resolve().parents[1] / "outputs" / "pytest_tmp"
    root.mkdir(parents=True, exist_ok=True)
    config.option.basetemp = str(root)
