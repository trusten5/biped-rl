"""Load the robot config from robot.yaml."""

from pathlib import Path

import yaml

CONFIG_PATH = Path(__file__).resolve().parent / "robot.yaml"


def load_config(path: Path | str = CONFIG_PATH) -> dict:
    """Return the config as a nested dict."""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)
