"""
Feature extraction module using TIAToolbox's DeepFeatureExtractor.

This module provides a wrapper for TIAToolbox's DeepFeatureExtractor class,
enabling extraction of deep features from whole slide images for use in
downstream machine learning tasks.
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Union, Optional, Any, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from omegaconf import DictConfig
from tiatoolbox.models.engine.semantic_segmentor import DeepFeatureExtractor

# Configure logging
logger = logging.getLogger(__name__)


class FeatureExtractor:
    """
    Wrapper for TIAToolbox's DeepFeatureExtractor.
    
    This class provides a simplified interface for extracting features
    from whole slide images using various models supported by TIAToolbox.
    """
    
    def __init__(
        self,
        model_name: str,
        model_weights: Optional[str] = None,
        batch_size: int = 16,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        num_workers: int = 4,
        patch_size: int = 256,
        stride: Optional[int] = None,
        resolution: float = 1.0,
        units: str = "mpp",
        output_dir: Optional[str] = None,
    ):
        """
        Initialize the feature extractor.
        
        Args:
            model_name: Name of the model to use (e.g., 'resnet50', 'densenet121')
            model_weights: Path to model weights (if None, uses default weights)
            batch_size: Batch size for feature extraction
            device: Device to run the model on ('cuda' or 'cpu')
            num_workers: Number of workers for data loading
            patch_size: Size of patches to extract features from
            stride: Stride between patches (if None, uses patch_size)
            resolution: Resolution for patch extraction (in the specified units)
            units: Units for resolution ('mpp', 'power', or 'level')
            output_dir: Directory to save extracted features
        """
        self.model_name = model_name
        self.model_weights = model_weights
        self.batch_size = batch_size
        self.device = device
        self.num_workers = num_workers
        self.patch_size = patch_size
        self.stride = stride or patch_size
        self.resolution = resolution
        self.units = units
        self.output_dir = output_dir
        
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Initialize the TIAToolbox DeepFeatureExtractor
        self.extractor = self._initialize_extractor()
        
    def _initialize_extractor(self) -> DeepFeatureExtractor:
        """
        Initialize the TIAToolbox DeepFeatureExtractor.
        
        Returns:
            Initialized DeepFeatureExtractor
        """
        logger.info(f"Initializing DeepFeatureExtractor with model {self.model_name}")
        
        # Create the extractor
        extractor = DeepFeatureExtractor(
            pretrained_model=self.model_name,
            pretrained_weights=self.model_weights,
            batch_size=self.batch_size,
            num_loader_workers=self.num_workers,
            verbose=True,
            auto_generate_mask=False,  # We'll use our custom mask
        )
        
        return extractor
    
    def extract_features(
        self,
        slide_path: Union[str, Path],
        mask_path: Optional[Union[str, Path]] = None,
        save_path: Optional[Union[str, Path]] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Extract features from a whole slide image.
        
        Args:
            slide_path: Path to the WSI
            mask_path: Path to the tissue mask (optional)
            save_path: Path to save extracted features (optional)
            
        Returns:
            Dictionary with 'features' and 'coordinates' arrays
        """
        logger.info(f"Extracting features from {slide_path}")
        
        # Convert paths to strings
        slide_path = str(slide_path)
        mask_path = str(mask_path) if mask_path else None
        
        # Set up save path if not provided
        if not save_path and self.output_dir:
            slide_name = Path(slide_path).stem
            save_path = Path(self.output_dir) / f"{slide_name}_{self.model_name}_features"
        
        # Extract features using DeepFeatureExtractor
        try:
            results = self.extractor.predict(
                imgs=[slide_path],
                masks=[mask_path] if mask_path else None,
                mode="wsi",
                patch_input_shape=(self.patch_size, self.patch_size),
                stride_shape=(self.stride, self.stride),
                resolution=self.resolution,
                units=self.units,
                device=self.device,
                save_dir=str(save_path) if save_path else None,
                crash_on_exception=True,
            )
            
            # Get the output files
            if isinstance(results, list) and len(results) > 0:
                # results is a list of tuples (input_path, output_path)
                output_path = results[0][1]
                
                # Load the position and features files
                position_file = f"{output_path}.position.npy"
                features_file = f"{output_path}.features.0.npy"  # Assuming single output head
                
                if os.path.exists(position_file) and os.path.exists(features_file):
                    positions = np.load(position_file)
                    features = np.load(features_file)
                    
                    logger.info(f"Extracted {len(features)} feature vectors of dimension {features.shape[1]}")
                    
                    return {
                        'features': features,
                        'coordinates': positions,
                    }
                else:
                    logger.error(f"Output files not found: {position_file} or {features_file}")
            
            logger.error("Failed to extract features: unexpected results format")
            return {
                'features': np.array([]),
                'coordinates': np.array([]),
            }
            
        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            return {
                'features': np.array([]),
                'coordinates': np.array([]),
            }
    
    def batch_extract_features(
        self,
        wsi_table: Union[str, pd.DataFrame],
        wsi_column: str = "slide_path",
        mask_column: Optional[str] = "mask_path",
        id_column: Optional[str] = "wsi_id",
        save_dir: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Extract features from multiple WSIs listed in a table.
        
        Args:
            wsi_table: Path to CSV file or DataFrame with WSI information
            wsi_column: Column name containing WSI paths
            mask_column: Column name containing mask paths (optional)
            id_column: Column name containing WSI IDs (optional)
            save_dir: Directory to save extracted features (overrides self.output_dir)
            
        Returns:
            DataFrame with paths to extracted features for each WSI
        """
        # Load WSI table if it's a path
        if isinstance(wsi_table, str):
            wsi_df = pd.read_csv(wsi_table)
        else:
            wsi_df = wsi_table
        
        # Set output directory
        output_dir = save_dir or self.output_dir
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Add feature paths column to track output
        feature_paths_column = f"{self.model_name}_feature_path"
        wsi_df[feature_paths_column] = None
        
        # Extract features for each WSI
        for idx, row in tqdm(wsi_df.iterrows(), total=len(wsi_df), desc="Extracting features"):
            # Get WSI path and mask path (if available)
            wsi_path = row[wsi_column]
            mask_path = row[mask_column] if mask_column in row and pd.notna(row[mask_column]) else None
            
            # Get WSI ID
            wsi_id = row[id_column] if id_column in row else Path(wsi_path).stem
            
            # Set up save path
            if output_dir:
                save_path = Path(output_dir) / f"{wsi_id}_{self.model_name}_features"
            else:
                save_path = None
            
            # Extract features
            result = self.extract_features(wsi_path, mask_path, save_path)
            
            # Update DataFrame with feature path
            if len(result['features']) > 0 and save_path:
                wsi_df.loc[idx, feature_paths_column] = str(save_path)
        
        # Save updated WSI table
        if output_dir:
            wsi_df.to_csv(Path(output_dir) / f"wsi_table_with_{self.model_name}_features.csv", index=False)
        
        return wsi_df


class TorchModelFeatureExtractor:
    """
    Feature extractor for any PyTorch model.
    
    This class allows extraction of features from a custom PyTorch model
    that can be used with TIAToolbox's DeepFeatureExtractor.
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        preprocess_fn: Optional[callable] = None,
        layer_name: Optional[str] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Initialize the PyTorch model feature extractor.
        
        Args:
            model: PyTorch model to use for feature extraction
            preprocess_fn: Function to preprocess inputs before passing to model
            layer_name: Name of the layer to extract features from (if None, uses final output)
            device: Device to run the model on ('cuda' or 'cpu')
        """
        self.model = model.to(device)
        self.model.eval()
        self.preprocess_fn = preprocess_fn
        self.layer_name = layer_name
        self.device = device
        
        # Set up for intermediate layer extraction if needed
        self.activation = {}
        if layer_name:
            self._register_hooks()
    
    def _register_hooks(self):
        """Register forward hooks to capture intermediate activations."""
        def get_activation(name):
            def hook(model, input, output):
                self.activation[name] = output.detach()
            return hook
        
        # Try to find the layer by name
        for name, module in self.model.named_modules():
            if name == self.layer_name:
                module.register_forward_hook(get_activation(name))
                break
    
    def extract_features(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Extract features from input tensor.
        
        Args:
            input_tensor: Input tensor to extract features from
            
        Returns:
            Feature tensor
        """
        with torch.no_grad():
            # Apply preprocessing if available
            if self.preprocess_fn:
                input_tensor = self.preprocess_fn(input_tensor)
            
            # Move to device
            input_tensor = input_tensor.to(self.device)
            
            # Forward pass
            output = self.model(input_tensor)
            
            # Return intermediate activation if specified
            if self.layer_name and self.layer_name in self.activation:
                return self.activation[self.layer_name]
            
            return output


def extract_features_from_patch_folder(
    patch_folder: Union[str, Path],
    extractor: Union[FeatureExtractor, TorchModelFeatureExtractor],
    patch_extension: str = "png",
    batch_size: int = 32,
    num_workers: int = 4,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    save_path: Optional[Union[str, Path]] = None,
) -> Dict[str, np.ndarray]:
    """
    Extract features from a folder of patch images.
    
    Args:
        patch_folder: Path to folder containing patch images
        extractor: Feature extractor to use
        patch_extension: File extension of patch images
        batch_size: Batch size for feature extraction
        num_workers: Number of workers for data loading
        device: Device to run extraction on
        save_path: Path to save extracted features
        
    Returns:
        Dictionary with 'features' and 'filenames' arrays
    """
    from torch.utils.data import Dataset, DataLoader
    from torchvision import transforms
    from PIL import Image
    
    class PatchDataset(Dataset):
        def __init__(self, patch_folder, extension, transform=None):
            self.patch_folder = Path(patch_folder)
            self.patch_files = sorted(list(self.patch_folder.glob(f"*.{extension}")))
            self.transform = transform
        
        def __len__(self):
            return len(self.patch_files)
        
        def __getitem__(self, idx):
            patch_path = self.patch_files[idx]
            patch = Image.open(patch_path).convert('RGB')
            
            if self.transform:
                patch = self.transform(patch)
            
            return patch, str(patch_path)
    
    # Basic transform for 224x224 input (adjust as needed)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Create dataset and dataloader
    dataset = PatchDataset(patch_folder, patch_extension, transform)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    features = []
    filenames = []
    
    # Extract features
    for batch, paths in tqdm(dataloader, desc="Extracting features"):
        batch = batch.to(device)
        batch_features = extractor.extract_features(batch).cpu().numpy()
        
        features.append(batch_features)
        filenames.extend(paths)
    
    features = np.concatenate(features, axis=0)
    
    # Save features if requested
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        np.savez(
            save_path,
            features=features,
            filenames=np.array(filenames),
        )
    
    return {
        'features': features,
        'filenames': np.array(filenames),
    }
