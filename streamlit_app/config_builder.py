"""Configuration builder for Streamlit interactive demo.

Loads the default YAML config, applies slider overrides, and returns
a Config object ready for the simulation runner.
"""

import copy
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from src.config import Config


_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_CONFIG_PATH = _PROJECT_ROOT / "config" / "default_config.yaml"


def load_default_config_dict() -> Dict[str, Any]:
    """Load the default config YAML as a raw dictionary.

    Returns
    -------
    dict
        Raw configuration dictionary.
    """
    with open(_DEFAULT_CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


# Slider definitions: (label, config_path, default, min, max, step)
SLIDER_DEFS = [
    ("Attraction gain (k_att)", "planner.k_att", 20.0, 1.0, 100.0, 1.0),
    ("Repulsion gain (k_rep)", "planner.k_rep", 80.0, 10.0, 300.0, 5.0),
    ("Influence range", "planner.influence_range", 120.0, 30.0, 300.0, 10.0),
    ("Vortex weight", "planner.vortex_weight", 0.7, 0.0, 1.0, 0.05),
    ("Cruise fraction", "planner.cruise_fraction", 0.4, 0.1, 1.0, 0.05),
    ("Max speed", "planner.max_speed", 80.0, 20.0, 200.0, 5.0),
    ("PID Kp", "controller.kp", 4.0, 0.5, 10.0, 0.5),
]


def _set_nested(d: Dict[str, Any], path: str, value: Any) -> None:
    """Set a value in a nested dict using dot-separated path."""
    keys = path.split(".")
    for key in keys[:-1]:
        d = d.setdefault(key, {})
    d[keys[-1]] = value


def _get_nested(d: Dict[str, Any], path: str, default: Any = None) -> Any:
    """Get a value from a nested dict using dot-separated path."""
    keys = path.split(".")
    for key in keys[:-1]:
        d = d.get(key, {})
        if not isinstance(d, dict):
            return default
    return d.get(keys[-1], default)


def build_config(overrides: Dict[str, Any]) -> Config:
    """Build a Config object from defaults + slider overrides.

    Parameters
    ----------
    overrides : dict
        Keys are dot-separated config paths (e.g. "planner.k_att"),
        values are the slider values.

    Returns
    -------
    Config
        Fully built configuration object.
    """
    config_dict = copy.deepcopy(load_default_config_dict())

    for path, value in overrides.items():
        _set_nested(config_dict, path, value)

    return Config(config_dict)


def get_scenario_names(config_dict: Optional[Dict[str, Any]] = None) -> List[str]:
    """Get available scenario names from config.

    Parameters
    ----------
    config_dict : dict or None
        Raw config dict. Loads default if None.

    Returns
    -------
    list of str
        Available scenario names.
    """
    if config_dict is None:
        config_dict = load_default_config_dict()
    scenarios = config_dict.get("environment", {}).get("scenarios", {})
    return list(scenarios.keys())
