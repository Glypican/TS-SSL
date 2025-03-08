"""
Feature Garden implementation for aggregating features from multiple models.

This module provides a unified interface to extract and combine features
from various models for use in downstream tasks like MIL attention-based
tumor highlighting.
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Union, Optional, Any, Tuple
import pickle
import json

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from tumor_highlighter.features.extractor import FeatureExtractor
from tumor_highlighter.utils.config import load_config

# Configure logging
logger = logging.getLogger(__name__)


class FeatureGarden:
    """
    Feature Garden for aggregating features from multiple models.
    
    This class manages multiple feature extractors and provides a unified
    interface to extract features from various models, combine them, and
    use them for downstream tasks.
    """
    
    def __init__(
        self,
        config: Optional[Union[str, Path, DictConfig]] = None,
        output_dir: Optional[Union[str, Path]] = None,
    ):
        """
        Initialize the Feature Garden.
        
        Args:
            config: Configuration file path or OmegaConf DictConfig
            output_dir: Directory to save extracted features
        """
        # Load configuration
        if config is None:
            self.config = load_config()
        elif isinstance(config, (str, Path)):
            self.config = OmegaConf.load(config)
        else:
            self.config = config
        
        # Set output directory
        self.output_dir = output_dir or self.config.experiment.output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize feature extractors
        self.extractors = {}
        self._initialize_extractors()
        
        # Dictionary to track available features
        self.available_features = {}
    
    def _initialize_extractors(self) -> None:
        """Initialize feature extractors based on configuration."""
        logger.info("Initializing feature extractors")
        
        # Check which feature extractors are enabled in config
        enabled_extractors = {}
        
        for extractor_name, extractor_config in self.config.models.feature_extractors.items():
            if extractor_config.enabled:
                enabled_extractors[extractor_name] = extractor_config
        
        logger.info(f"Found {len(enabled_extractors)} enabled feature extractors: {list(enabled_extractors.keys())}")
        
        # Initialize each enabled extractor
        for extractor_name, extractor_config in enabled_extractors.items():
            try:
                # Get parameters from config
                model_name = extractor_name
                weights_path = extractor_config.get("weights_path", None)
                batch_size = extractor_config.get("batch_size", 16)
                
                # Create extractor output directory
                extractor_output_dir = os.path.join(self.output_dir, f"features_{model_name}")
                os.makedirs(extractor_output_dir, exist_ok=True)
                
                # Initialize extractor
                extractor = FeatureExtractorWrapper(
                    model_name=model_name,
                    model_weights=weights_path,
                    batch_size=batch_size,
                    device=extractor_config.get("device", "cuda" if torch.cuda.is_available() else "cpu"),
                    num_workers=extractor_config.get("num_workers", 4),
                    patch_size=self.config.data.patch_extraction.patch_size,
                    stride=self.config.data.patch_extraction.get("stride", None),
                    resolution=extractor_config.get("resolution", 1.0),
                    units=extractor_config.get("units", "mpp"),
                    output_dir=extractor_output_dir,
                )
                
                # Add to extractors dictionary
                self.extractors[model_name] = extractor
                logger.info(f"Initialized feature extractor for {model_name}")
                
            except Exception as e:
                logger.error(f"Failed to initialize feature extractor {extractor_name}: {e}")
    
    def extract_features(
        self,
        wsi_table: Union[str, pd.DataFrame],
        models: Optional[List[str]] = None,
        wsi_column: str = "slide_path",
        mask_column: Optional[str] = "mask_path",
        id_column: Optional[str] = "wsi_id",
        save_results: bool = True,
    ) -> pd.DataFrame:
        """
        Extract features from WSIs listed in the table using multiple models.
        
        Args:
            wsi_table: Path to CSV file or DataFrame with WSI information
            models: List of model names to use (None uses all available models)
            wsi_column: Column name containing WSI paths
            mask_column: Column name containing mask paths
            id_column: Column name containing WSI IDs
            save_results: Whether to save results to disk
            
        Returns:
            DataFrame with paths to extracted features for each WSI
        """
        # Load WSI table if it's a path
        if isinstance(wsi_table, str):
            wsi_df = pd.read_csv(wsi_table)
        else:
            wsi_df = wsi_table.copy()
        
        # Determine models to use
        if models is None:
            models = list(self.extractors.keys())
        
        # Filter to only use available models
        models = [model for model in models if model in self.extractors]
        
        if not models:
            logger.error("No valid feature extractors specified or available")
            return wsi_df
        
        logger.info(f"Extracting features using {len(models)} models: {models}")
        
        # Extract features for each model
        for model_name in models:
            logger.info(f"Extracting features with {model_name}")
            extractor = self.extractors[model_name]
            
            # Create output directory for this model
            model_output_dir = os.path.join(self.output_dir, f"features_{model_name}")
            os.makedirs(model_output_dir, exist_ok=True)
            
            # Extract features
            wsi_df = extractor.batch_extract_features(
                wsi_table=wsi_df,
                wsi_column=wsi_column,
                mask_column=mask_column,
                id_column=id_column,
                save_dir=model_output_dir,
            )
        
        # Save updated WSI table
        if save_results:
            output_path = os.path.join(self.output_dir, "wsi_table_with_features.csv")
            wsi_df.to_csv(output_path, index=False)
            logger.info(f"Saved updated WSI table to {output_path}")
        
        return wsi_df
    
    def load_features(
        self,
        wsi_id: str,
        models: Optional[List[str]] = None,
        combine: bool = False,
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Load extracted features for a specific WSI.
        
        Args:
            wsi_id: ID of the WSI to load features for
            models: List of model names to load features from (None loads all available)
            combine: Whether to combine features from different models
            
        Returns:
            Dictionary of features by model (or combined features if combine=True)
        """
        # Determine models to use
        if models is None:
            models = list(self.extractors.keys())
        
        # Load features for each model
        features = {}
        
        for model_name in models:
            feature_path = os.path.join(self.output_dir, f"features_{model_name}", f"{wsi_id}_{model_name}_features")
            position_file = f"{feature_path}.position.npy"
            feature_file = f"{feature_path}.features.0.npy"
            
            if os.path.exists(position_file) and os.path.exists(feature_file):
                positions = np.load(position_file)
                feature_data = np.load(feature_file)
                
                features[model_name] = {
                    'features': feature_data,
                    'coordinates': positions,
                }
                
                logger.info(f"Loaded {len(feature_data)} feature vectors from {model_name} for {wsi_id}")
        
        # Combine features if requested and multiple models are available
        if combine and len(features) > 1:
            combined = self.combine_features(features)
            return {'combined': combined}
        
        return features
    
    def combine_features(
        self,
        features_dict: Dict[str, Dict[str, np.ndarray]],
        method: str = 'concatenate',
    ) -> Dict[str, np.ndarray]:
        """
        Combine features from multiple models.
        
        Args:
            features_dict: Dictionary of features by model
            method: Method to combine features ('concatenate', 'average', 'max')
            
        Returns:
            Dictionary with combined features and coordinates
        """
        # Verify features can be combined (must have same coordinates)
        coords_list = [f['coordinates'] for f in features_dict.values()]
        
        # Check if all coordinates match
        coords_match = all(np.array_equal(coords_list[0], coords) for coords in coords_list[1:])
        
        if not coords_match:
            logger.error("Cannot combine features with different coordinates")
            return {}
        
        # Extract feature arrays
        feature_arrays = [f['features'] for f in features_dict.values()]
        
        # Combine features based on specified method
        if method == 'concatenate':
            combined_features = np.concatenate(feature_arrays, axis=1)
        elif method == 'average':
            combined_features = np.mean(feature_arrays, axis=0)
        elif method == 'max':
            combined_features = np.max(feature_arrays, axis=0)
        else:
            logger.error(f"Unknown combination method: {method}")
            return {}
        
        return {
            'features': combined_features,
            'coordinates': coords_list[0],
        }
    
    def save_feature_cache(
        self,
        wsi_id: str,
        features: Dict[str, np.ndarray],
        cache_path: Optional[Union[str, Path]] = None,
    ) -> Path:
        """
        Save combined or processed features to cache for quick access.
        
        Args:
            wsi_id: ID of the WSI
            features: Dictionary with feature arrays
            cache_path: Path to save cache (if None, uses default)
            
        Returns:
            Path to saved cache file
        """
        if cache_path is None:
            cache_dir = os.path.join(self.output_dir, "feature_cache")
            os.makedirs(cache_dir, exist_ok=True)
            cache_path = os.path.join(cache_dir, f"{wsi_id}_feature_cache.npz")
        
        np.savez(cache_path, **features)
        logger.info(f"Saved feature cache to {cache_path}")
        
        return Path(cache_path)
    
    def load_feature_cache(
        self,
        wsi_id: str,
        cache_path: Optional[Union[str, Path]] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Load cached features for a WSI.
        
        Args:
            wsi_id: ID of the WSI
            cache_path: Path to cache file (if None, uses default)
            
        Returns:
            Dictionary with feature arrays
        """
        if cache_path is None:
            cache_path = os.path.join(self.output_dir, "feature_cache", f"{wsi_id}_feature_cache.npz")
        
        if not os.path.exists(cache_path):
            logger.error(f"Feature cache not found at {cache_path}")
            return {}
        
        cache = np.load(cache_path)
        features = {key: cache[key] for key in cache.files}
        
        logger.info(f"Loaded feature cache from {cache_path}")
        
        return features
    
    def extract_and_cache_all(
        self,
        wsi_table: Union[str, pd.DataFrame],
        combine_method: str = 'concatenate',
        wsi_column: str = "slide_path",
        mask_column: Optional[str] = "mask_path",
        id_column: Optional[str] = "wsi_id",
    ) -> pd.DataFrame:
        """
        Extract features from all models, combine them, and cache the results.
        
        Args:
            wsi_table: Path to CSV file or DataFrame with WSI information
            combine_method: Method to combine features
            wsi_column: Column name containing WSI paths
            mask_column: Column name containing mask paths
            id_column: Column name containing WSI IDs
            
        Returns:
            DataFrame with paths to extracted and cached features
        """
        # Extract features from all models
        wsi_df = self.extract_features(
            wsi_table=wsi_table,
            wsi_column=wsi_column,
            mask_column=mask_column,
            id_column=id_column,
        )
        
        # Create cache directory
        cache_dir = os.path.join(self.output_dir, "feature_cache")
        os.makedirs(cache_dir, exist_ok=True)
        
        # Add cache path column
        wsi_df['feature_cache_path'] = None
        
        # Combine and cache features for each WSI
        for idx, row in tqdm(wsi_df.iterrows(), total=len(wsi_df), desc="Combining and caching features"):
            try:
                # Get WSI ID
                wsi_id = row[id_column]
                
                # Load features for all models
                features = self.load_features(wsi_id, combine=False)
                
                if not features:
                    logger.warning(f"No features found for {wsi_id}")
                    continue
                
                # Combine features
                combined = self.combine_features(features, method=combine_method)
                
                if not combined:
                    logger.warning(f"Failed to combine features for {wsi_id}")
                    continue
                
                # Save to cache
                cache_path = os.path.join(cache_dir, f"{wsi_id}_feature_cache.npz")
                self.save_feature_cache(wsi_id, combined, cache_path)
                
                # Update DataFrame
                wsi_df.loc[idx, 'feature_cache_path'] = cache_path
                
            except Exception as e:
                logger.error(f"Error processing {wsi_id}: {e}")
        
        # Save updated WSI table
        output_path = os.path.join(self.output_dir, "wsi_table_with_feature_cache.csv")
        wsi_df.to_csv(output_path, index=False)
        
        return wsi_df
