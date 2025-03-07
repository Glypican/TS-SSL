"""
Configuration utilities for the tumor_highlighter package.
Uses OmegaConf/Hydra for configuration management.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional, Union

import yaml
from omegaconf import OmegaConf, DictConfig


def get_default_config_path() -> Path:
    """Get the path to the default configuration directory."""
    # Look in project root first (development mode)
    project_root = Path(__file__).parent.parent.parent
    config_dir = project_root / "config"
    
    if config_dir.exists():
        return config_dir
    
    # Look in package directory (installed mode)
    package_dir = Path(__file__).parent.parent
    config_dir = package_dir / "config"
    
    if config_dir.exists():
        return config_dir
    
    # Create default config directory if it doesn't exist
    os.makedirs(config_dir, exist_ok=True)
    return config_dir


def load_config(config_path: Optional[Union[str, Path]] = None) -> DictConfig:
    """
    Load configuration from a YAML file.
    
    Args:
        config_path: Path to configuration file. If None, loads default.
        
    Returns:
        OmegaConf configuration object
    """
    if config_path is None:
        # Use default config
        config_dir = get_default_config_path()
        config_path = config_dir / "default.yaml"
        
        # Create default config if it doesn't exist
        if not config_path.exists():
            create_default_config(config_path)
    
    # Load and return config
    return OmegaConf.load(config_path)


def create_default_config(config_path: Union[str, Path]) -> None:
    """
    Create a default configuration file.
    
    Args:
        config_path: Path where default config will be saved
    """
    default_config = {
        "data": {
            "wsi_table_path": "data/wsi_metadata.csv",
            "tissue_mask": {
                "downsample_factor": 64,
                "hsv_lower": [0, 30, 0],
                "hsv_upper": [180, 255, 255],
                "opening_kernel_size": 5,
                "closing_kernel_size": 5,
                "min_tissue_size": 1000
            },
            "patch_extraction": {
                "patch_size": 256,
                "overlap": 0,
                "level": 0,
                "save_patches": False
            }
        },
        "models": {
            "ts_ssl": {
                "input_channels": 3,
                "hidden_dim": 128,
                "latent_dim": 64,
                "attention_heads": 4
            },
            "feature_extractors": {
                "resnet18": {"enabled": True, "pretrained": True},
                "resnet50": {"enabled": True, "pretrained": True},
                "ctranspath": {"enabled": False, "weights_path": "pretrained/ctranspath.pth"}
            }
        },
        "training": {
            "ts_ssl": {
                "batch_size": 64,
                "epochs": 100,
                "learning_rate": 1e-4,
                "weight_decay": 1e-5,
                "scheduler": {
                    "name": "cosine",
                    "warmup_epochs": 10
                },
                "loss": {
                    "name": "mse",
                    "params": {}
                }
            },
            "mil": {
                "batch_size": 1,
                "epochs": 50,
                "learning_rate": 1e-4,
                "weight_decay": 1e-5
            }
        },
        "experiment": {
            "output_dir": "experiments",
            "mlflow": {
                "tracking_uri": "mlruns",
                "experiment_name": "tumor_highlighter"
            },
            "tensorboard": {
                "log_dir": "runs"
            }
        }
    }
    
    # Save as YAML
    with open(config_path, "w") as f:
        yaml.dump(default_config, f, default_flow_style=False)


def save_config(config: DictConfig, save_path: Union[str, Path]) -> None:
    """
    Save configuration to a YAML file.
    
    Args:
        config: Configuration to save
        save_path: Path where config will be saved
    """
    with open(save_path, "w") as f:
        yaml.dump(OmegaConf.to_container(config, resolve=True), f, default_flow_style=False)


def merge_configs(base_config: DictConfig, override_config: Dict[str, Any]) -> DictConfig:
    """
    Merge base configuration with overrides.
    
    Args:
        base_config: Base configuration
        override_config: Configuration values to override
        
    Returns:
        Merged configuration
    """
    override_conf = OmegaConf.create(override_config)
    return OmegaConf.merge(base_config, override_conf)


def get_subconfig(config: DictConfig, key_path: str) -> Any:
    """
    Get a nested configuration value using dot notation.
    
    Args:
        config: Configuration object
        key_path: Dot-separated path to config value (e.g., "data.patch_size")
        
    Returns:
        Configuration value at the specified path
    """
    return OmegaConf.select(config, key_path)
