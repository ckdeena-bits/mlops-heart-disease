from pathlib import Path
from typing import Any, Dict

import yaml

CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "config.yaml"

def load_config(config_path: Path = CONFIG_PATH) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
