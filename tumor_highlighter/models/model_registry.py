"""Model Registry for managing trained models."""

import json
import os
import shutil
from pathlib import Path
from dataclasses import dataclass
import torch

from tumor_highlighter import logger
from tumor_highlighter.models.ts_ssl.autoencoder import SpatialChannelAttentionAutoencoder


@dataclass
class ModelInfo:
    """Model information for the registry."""
    
    name: str
    type: str
    path: str
    description: str = ""


class ModelRegistry:
    """Model Registry for managing trained models."""
    
    def __init__(self, registry_path="~/.tumor_highlighter/models"):
        """Initialize the model registry.
        
        Args:
            registry_path (str): Path to the registry directory
        """
        self.registry_path = Path(registry_path).expanduser()
        self.registry_file = self.registry_path / "registry.json"
        self.registry = {}
        
        # Create registry directory if it doesn't exist
        os.makedirs(self.registry_path, exist_ok=True)
        
        # Load registry if it exists
        if self.registry_file.exists():
            self._load_registry()
        else:
            self._save_registry()
    
    def _load_registry(self):
        """Load the registry from file."""
        try:
            with open(self.registry_file, "r") as f:
                registry_data = json.load(f)
            
            self.registry = {}
            for name, data in registry_data.items():
                self.registry[name] = ModelInfo(
                    name=name,
                    type=data["type"],
                    path=data["path"],
                    description=data.get("description", ""),
                )
            
            logger.info(f"Loaded {len(self.registry)} models from registry")
        except Exception as e:
            logger.error(f"Error loading registry: {e}")
            self.registry = {}
    
    def _save_registry(self):
        """Save the registry to file."""
        try:
            registry_data = {}
            for name, info in self.registry.items():
                registry_data[name] = {
                    "type": info.type,
                    "path": info.path,
                    "description": info.description,
                }
            
            with open(self.registry_file, "w") as f:
                json.dump(registry_data, f, indent=2)
            
            logger.debug(f"Saved {len(self.registry)} models to registry")
        except Exception as e:
            logger.error(f"Error saving registry: {e}")
    
    def register_model(self, model_path, model_name, model_type, description=""):
        """Register a model in the registry.
        
        Args:
            model_path (str): Path to the model file
            model_name (str): Name for the registered model
            model_type (str): Type of the model (ts-ssl, mil, custom)
            description (str): Description of the model
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Validate model path
            model_path = Path(model_path)
            if not model_path.exists():
                logger.error(f"Model file {model_path} does not exist")
                return False
            
            # Create a copy of the model in the registry
            registry_model_path = self.registry_path / f"{model_name}.pth"
            
            # If the file is not already in the registry, copy it
            if str(model_path.resolve()) != str(registry_model_path.resolve()):
                shutil.copy2(model_path, registry_model_path)
            
            # Add model to registry
            self.registry[model_name] = ModelInfo(
                name=model_name,
                type=model_type,
                path=str(registry_model_path),
                description=description,
            )
            
            # Save registry
            self._save_registry()
            
            logger.info(f"Registered model {model_name} of type {model_type}")
            return True
        except Exception as e:
            logger.error(f"Error registering model: {e}")
            return False
    
    def get_model(self, model_name):
        """Get a model from the registry.
        
        Args:
            model_name (str): Name of the model to get
        
        Returns:
            object: The loaded model or None if not found
        """
        if model_name not in self.registry:
            logger.error(f"Model {model_name} not found in registry")
            return None
        
        model_info = self.registry[model_name]
        model_path = model_info.path
        model_type = model_info.type
        
        try:
            if model_type == "ts-ssl":
                # Initialize the model with the correct architecture
                model = SpatialChannelAttentionAutoencoder()
                # Load the model weights
                model.load_state_dict(torch.load(model_path, map_location="cpu"))
                model.eval()  # Set to evaluation mode
                return model
            elif model_type == "mil":
                # Implement MIL model loading
                logger.error("MIL model loading not implemented yet")
                return None
            elif model_type == "custom":
                # Load custom model
                model = torch.load(model_path, map_location="cpu")
                model.eval()  # Set to evaluation mode
                return model
            else:
                logger.error(f"Unknown model type: {model_type}")
                return None
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return None
    
    def list_models(self, model_type=None):
        """List all registered models.
        
        Args:
            model_type (str): Filter models by type (ts-ssl, mil, custom)
        
        Returns:
            list: List of ModelInfo objects
        """
        if model_type:
            return [info for info in self.registry.values() if info.type == model_type]
        else:
            return list(self.registry.values())
    
    def delete_model(self, model_name):
        """Delete a model from the registry.
        
        Args:
            model_name (str): Name of the model to delete
        
        Returns:
            bool: True if successful, False otherwise
        """
        if model_name not in self.registry:
            logger.error(f"Model {model_name} not found in registry")
            return False
        
        try:
            model_info = self.registry[model_name]
            model_path = Path(model_info.path)
            
            # Delete model file if it exists and is in the registry directory
            if model_path.exists() and str(model_path).startswith(str(self.registry_path)):
                os.remove(model_path)
            
            # Remove from registry
            del self.registry[model_name]
            
            # Save registry
            self._save_registry()
            
            logger.info(f"Deleted model {model_name}")
            return True
        except Exception as e:
            logger.error(f"Error deleting model: {e}")
            return False
