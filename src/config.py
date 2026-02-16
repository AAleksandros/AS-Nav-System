"""Configuration management for the autonomous navigation system.

This module provides configuration loading and validation from YAML files,
with support for default values and CLI overrides.
"""

from pathlib import Path
from typing import Any, Dict, Optional

import yaml


class ConfigError(Exception):
    """Raised when configuration is invalid or missing required fields."""
    pass


class Config:
    """Configuration container with nested attribute access.

    Allows accessing nested config values using dot notation:
        config.model.name
        config.planning.obstacle_distance_threshold

    Attributes are dynamically created from the loaded YAML structure.
    """

    def __init__(self, config_dict: Dict[str, Any]):
        """Initialize config from dictionary.

        Args:
            config_dict: Configuration dictionary from YAML
        """
        for key, value in config_dict.items():
            if isinstance(value, dict):
                setattr(self, key, Config(value))
            else:
                setattr(self, key, value)

    def __repr__(self) -> str:
        """String representation of config."""
        attrs = {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
        return f"Config({attrs})"

    def get(self, key: str, default: Any = None) -> Any:
        """Get config value with optional default.

        Args:
            key: Configuration key
            default: Default value if key not found

        Returns:
            Configuration value or default
        """
        return getattr(self, key, default)


def load_config(config_path: Optional[str] = None) -> Config:
    """Load configuration from YAML file.

    Args:
        config_path: Path to config file. If None, uses default config.

    Returns:
        Config object with nested attribute access

    Raises:
        ConfigError: If config file not found or invalid
    """
    # Determine config path
    if config_path is None:
        # Use default config
        project_root = Path(__file__).parent.parent
        config_path = project_root / "config" / "default_config.yaml"
    else:
        config_path = Path(config_path)

    # Check if file exists
    if not config_path.exists():
        raise ConfigError(f"Configuration file not found: {config_path}")

    # Load YAML
    try:
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ConfigError(f"Failed to parse YAML config: {e}")

    # Validate required sections
    required_sections = [
        'environment', 'sensor', 'planner', 'controller',
        'control', 'visualization', 'simulation', 'logging',
    ]
    missing_sections = [
        s for s in required_sections if s not in config_dict
    ]
    if missing_sections:
        raise ConfigError(f"Missing required config sections: {missing_sections}")

    # Create config object
    return Config(config_dict)


def merge_config_overrides(config: Config, overrides: Dict[str, Any]) -> Config:
    """Merge CLI overrides into config object.

    Args:
        config: Base configuration
        overrides: Dictionary of override values (e.g., {'frame_skip': 3})

    Returns:
        Updated config object
    """
    # Convert config back to dict, apply overrides, reconstruct
    config_dict = config_to_dict(config)

    for key, value in overrides.items():
        # Support nested keys with dot notation (e.g., "model.device")
        if '.' in key:
            parts = key.split('.')
            current = config_dict
            for part in parts[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]
            current[parts[-1]] = value
        else:
            # Try to find the key in top-level sections
            found = False
            for section in config_dict:
                if isinstance(config_dict[section], dict) \
                        and key in config_dict[section]:
                    config_dict[section][key] = value
                    found = True
                    break
            if not found:
                # Add to processing section by default
                if 'simulation' in config_dict:
                    config_dict['simulation'][key] = value

    return Config(config_dict)


def config_to_dict(config: Config) -> Dict[str, Any]:
    """Convert Config object back to dictionary.

    Args:
        config: Config object

    Returns:
        Dictionary representation
    """
    result = {}
    for key, value in config.__dict__.items():
        if isinstance(value, Config):
            result[key] = config_to_dict(value)
        else:
            result[key] = value
    return result


def print_config(config: Config, indent: int = 0) -> None:
    """Pretty print configuration.

    Args:
        config: Config object to print
        indent: Current indentation level
    """
    for key, value in config.__dict__.items():
        if isinstance(value, Config):
            print("  " * indent + f"{key}:")
            print_config(value, indent + 1)
        else:
            print("  " * indent + f"{key}: {value}")
