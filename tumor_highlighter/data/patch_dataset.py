"""
Patch dataset module for the tumor_highlighter package.

This module provides classes for working with patch-level datasets,
including loading pre-extracted patches and extracting patches from WSIs.
"""

import os
import glob
import logging
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Callable, Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from tiatoolbox.wsicore.wsireader import WSIReader
from tiatoolbox.tools.patchextraction import SlidingWindowPatchExtractor
from skimage.color import rgb2hsv

from tumor_highlighter.data.mask_generator import HSVTissueMaskGenerator

logger = logging.getLogger(__name__)


class PatchDataset(Dataset):
    """
    Dataset for working with pre-extracted image patches.
    
    This dataset loads image patches from directories or CSV files
    for training and inference.
    """
    
    def __init__(
        self,
        patches: Union[List[Union[str, Path]], str, Path, pd.DataFrame],
        labels: Optional[Union[List[Any], str]] = None,
        transform: Optional[Callable] = None,
        patch_path_column: str = 'patch_path',
        label_column: Optional[str] = 'label',
        filter_by: Optional[Dict[str, Any]] = None,
        min_tissue_percentage: Optional[float] = None,
        max_patches_per_class: Optional[int] = None,
        balance_classes: bool = False,
        class_weights: Optional[Dict[Any, float]] = None,
        seed: int = 42,
    ):
        """
        Initialize the patch dataset.
        
        Args:
            patches: List of patch paths, directory containing patches, 
                    or DataFrame with patch information
            labels: List of labels or column name in DataFrame
            transform: Transform to apply to patches
            patch_path_column: Column name for patch paths (if DataFrame)
            label_column: Column name for labels (if DataFrame)
            filter_by: Dictionary of {column: value} to filter DataFrame
            min_tissue_percentage: Minimum tissue percentage for patches
            max_patches_per_class: Maximum number of patches per class
            balance_classes: Whether to balance classes
            class_weights: Dictionary of {class: weight} for sampling
            seed: Random seed for reproducibility
        """
        self.transform = transform
        self.patch_path_column = patch_path_column
        self.label_column = label_column
        random.seed(seed)
        
        # Initialize patch information based on input type
        if isinstance(patches, pd.DataFrame):
            # Use provided DataFrame
            self.df = patches.copy()
            
            # Check for patch_path_column
            if self.patch_path_column not in self.df.columns:
                raise ValueError(f"Column '{self.patch_path_column}' not found in DataFrame")
            
            # Use provided label_column if available
            if labels is not None and isinstance(labels, str):
                self.label_column = labels
            
            # Check for label_column if specified
            if self.label_column is not None and self.label_column not in self.df.columns:
                logger.warning(f"Label column '{self.label_column}' not found in DataFrame")
                self.label_column = None
        
        elif isinstance(patches, (str, Path)) and os.path.isdir(patches):
            # Directory containing patches
            patch_dir = Path(patches)
            patch_paths = []
            patch_labels = []
            
            # Check if subdirectories are class labels
            subdirs = [d for d in patch_dir.iterdir() if d.is_dir()]
            
            if subdirs and labels is None:
                # Assume subdirectories are class labels
                for subdir in subdirs:
                    class_name = subdir.name
                    class_patches = list(subdir.glob('*.png')) + list(subdir.glob('*.jpg')) + list(subdir.glob('*.tif'))
                    patch_paths.extend(class_patches)
                    patch_labels.extend([class_name] * len(class_patches))
                
                # Create DataFrame
                self.df = pd.DataFrame({
                    self.patch_path_column: [str(p) for p in patch_paths],
                    self.label_column: patch_labels
                })
            
            else:
                # Flat directory of patches
                patch_paths = list(patch_dir.glob('*.png')) + list(patch_dir.glob('*.jpg')) + list(patch_dir.glob('*.tif'))
                
                # Create DataFrame
                self.df = pd.DataFrame({
                    self.patch_path_column: [str(p) for p in patch_paths]
                })
                
                # Add labels if provided
                if labels is not None:
                    if len(labels) != len(patch_paths):
                        raise ValueError(f"Number of labels ({len(labels)}) does not match number of patches ({len(patch_paths)})")
                    self.df[self.label_column] = labels
                else:
                    self.label_column = None
        
        elif isinstance(patches, list):
            # List of patch paths
            # Create DataFrame
            self.df = pd.DataFrame({
                self.patch_path_column: [str(p) for p in patches]
            })
            
            # Add labels if provided
            if labels is not None:
                if len(labels) != len(patches):
                    raise ValueError(f"Number of labels ({len(labels)}) does not match number of patches ({len(patches)})")
                self.df[self.label_column] = labels
            else:
                self.label_column = None
        
        else:
            raise ValueError("Invalid input type for patches")
        
        # Filter DataFrame if requested
        if filter_by is not None:
            for column, value in filter_by.items():
                if column in self.df.columns:
                    if isinstance(value, list):
                        self.df = self.df[self.df[column].isin(value)]
                    else:
                        self.df = self.df[self.df[column] == value]
                else:
                    logger.warning(f"Column '{column}' not found in DataFrame, skipping filter")
        
        # Filter by tissue percentage if requested
        if min_tissue_percentage is not None and min_tissue_percentage > 0:
            if 'tissue_percentage' in self.df.columns:
                self.df = self.df[self.df['tissue_percentage'] >= min_tissue_percentage]
            else:
                logger.warning("Column 'tissue_percentage' not found in DataFrame, calculating on the fly")
                self._calculate_tissue_percentage()
                self.df = self.df[self.df['tissue_percentage'] >= min_tissue_percentage]
        
        # Balance classes if requested
        if balance_classes and self.label_column is not None:
            self.df = self._balance_classes(max_patches_per_class, class_weights)
        elif max_patches_per_class is not None and self.label_column is not None:
            self.df = self._limit_patches_per_class(max_patches_per_class)
        
        # Reset index
        self.df = self.df.reset_index(drop=True)
        
        # Verify all patch paths exist
        self._verify_paths()
        
        logger.info(f"Initialized dataset with {len(self.df)} patches")
        
        # Log class distribution if labels are available
        if self.label_column is not None:
            class_counts = self.df[self.label_column].value_counts()
            logger.info(f"Class distribution: {class_counts.to_dict()}")
    
    def _verify_paths(self) -> None:
        """Verify that all patch paths exist."""
        valid_paths = []
        
        for path in self.df[self.patch_path_column]:
            if os.path.exists(path):
                valid_paths.append(True)
            else:
                valid_paths.append(False)
                logger.warning(f"Patch path not found: {path}")
        
        # Filter out invalid paths
        if not all(valid_paths):
            logger.warning(f"Found {sum(valid_paths)}/{len(valid_paths)} valid patch paths")
            self.df = self.df[valid_paths]
            self.df = self.df.reset_index(drop=True)
    
    def _calculate_tissue_percentage(self) -> None:
        """Calculate tissue percentage for each patch."""
        tissue_percentages = []
        
        for path in tqdm(self.df[self.patch_path_column], desc="Calculating tissue percentages"):
            try:
                img = Image.open(path).convert('RGB')
                tissue_percentage = self._estimate_tissue_percentage(np.array(img))
                tissue_percentages.append(tissue_percentage)
            except Exception as e:
                logger.error(f"Error processing {path}: {e}")
                tissue_percentages.append(0.0)
        
        self.df['tissue_percentage'] = tissue_percentages
    
    @staticmethod
    def _estimate_tissue_percentage(img: np.ndarray) -> float:
        """
        Estimate the percentage of tissue in an image using HSV thresholds.
        
        Args:
            img: RGB image as numpy array
            
        Returns:
            Estimated tissue percentage (0.0 to 1.0)
        """
        # Convert to HSV color space
        hsv_img = rgb2hsv(img)
        
        # Apply thresholds based on the same parameters as HSVTissueMaskGenerator
        hue_mask = (hsv_img[:, :, 0] > 0.6) & (hsv_img[:, :, 0] < 0.98)
        sat_mask = (hsv_img[:, :, 1] > 0.04)
        val_mask = (hsv_img[:, :, 2] > 0.2)
        
        # Combine masks
        tissue_mask = hue_mask & sat_mask & val_mask
        
        # Calculate percentage
        tissue_percentage = np.mean(tissue_mask)
        
        return tissue_percentage
    
    def _balance_classes(
        self,
        max_patches_per_class: Optional[int] = None,
        class_weights: Optional[Dict[Any, float]] = None
    ) -> pd.DataFrame:
        """
        Balance classes by under/oversampling.
        
        Args:
            max_patches_per_class: Maximum number of patches per class
            class_weights: Dictionary of {class: weight} for sampling
            
        Returns:
            Balanced DataFrame
        """
        # Get class counts
        class_counts = self.df[self.label_column].value_counts()
        
        # Determine target count (minimum if max_patches_per_class is not specified)
        if max_patches_per_class is None:
            target_count = class_counts.min()
        else:
            target_count = min(max_patches_per_class, class_counts.max())
        
        # Adjust target count based on class weights
        target_counts = {}
        if class_weights is not None:
            for class_name in class_counts.index:
                weight = class_weights.get(class_name, 1.0)
                target_counts[class_name] = int(target_count * weight)
        else:
            target_counts = {class_name: target_count for class_name in class_counts.index}
        
        # Create balanced DataFrame
        balanced_df = pd.DataFrame()
        
        for class_name, count in class_counts.items():
            class_df = self.df[self.df[self.label_column] == class_name]
            target = target_counts[class_name]
            
            if count > target:
                # Undersample
                balanced_df = pd.concat([balanced_df, class_df.sample(target, random_state=42)])
            else:
                # Oversample (with replacement if necessary)
                balanced_df = pd.concat([
                    balanced_df,
                    class_df.sample(target, replace=(count < target), random_state=42)
                ])
        
        return balanced_df
    
    def _limit_patches_per_class(self, max_patches: int) -> pd.DataFrame:
        """
        Limit the number of patches per class.
        
        Args:
            max_patches: Maximum number of patches per class
            
        Returns:
            Limited DataFrame
        """
        # Group by class
        grouped = self.df.groupby(self.label_column)
        
        # Limit each class
        limited_df = pd.DataFrame()
        
        for _, group in grouped:
            if len(group) > max_patches:
                limited_df = pd.concat([limited_df, group.sample(max_patches, random_state=42)])
            else:
                limited_df = pd.concat([limited_df, group])
        
        return limited_df
    
    def __len__(self) -> int:
        """Get the number of patches in the dataset."""
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a patch from the dataset.
        
        Args:
            idx: Index of the patch
            
        Returns:
            Dictionary containing the patch and metadata
        """
        # Get patch information
        patch_info = self.df.iloc[idx]
        patch_path = patch_info[self.patch_path_column]
        
        # Load patch
        try:
            patch = Image.open(patch_path).convert('RGB')
        except Exception as e:
            logger.error(f"Error loading patch {patch_path}: {e}")
            # Return a dummy patch if loading fails
            patch = Image.new('RGB', (256, 256), color=(0, 0, 0))
        
        # Apply transform if available
        if self.transform is not None:
            patch = self.transform(patch)
        
        # Convert to tensor if not already
        if not isinstance(patch, torch.Tensor):
            patch = torch.tensor(np.array(patch)).permute(2, 0, 1).float() / 255.0
        
        # Create result dictionary
        result = {'patch': patch, 'path': patch_path}
        
        # Add metadata
        for column in self.df.columns:
            if column != self.patch_path_column:
                result[column] = patch_info[column]
        
        return result
    
    def get_labels(self) -> Optional[List[Any]]:
        """Get the list of labels for each patch."""
        if self.label_column is not None:
            return self.df[self.label_column].tolist()
        return None
    
    def get_patch_paths(self) -> List[str]:
        """Get the list of patch paths."""
        return self.df[self.patch_path_column].tolist()
    
    def get_subset(self, indices: List[int]) -> 'PatchDataset':
        """
        Create a subset of the dataset.
        
        Args:
            indices: List of indices to include in the subset
            
        Returns:
            Subset of the dataset
        """
        subset_df = self.df.iloc[indices].reset_index(drop=True)
        
        return PatchDataset(
            patches=subset_df,
            transform=self.transform,
            patch_path_column=self.patch_path_column,
            label_column=self.label_column,
        )
    
    def train_val_split(
        self,
        val_size: float = 0.2,
        stratify: bool = True,
        random_state: int = 42,
    ) -> Tuple['PatchDataset', 'PatchDataset']:
        """
        Split the dataset into training and validation sets.
        
        Args:
            val_size: Fraction of data to use for validation
            stratify: Whether to stratify by class
            random_state: Random seed for reproducibility
            
        Returns:
            Tuple of (train_dataset, val_dataset)
        """
        # Determine stratification
        stratify_col = None
        if stratify and self.label_column is not None:
            stratify_col = self.df[self.label_column]
        
        # Split indices
        train_idx, val_idx = train_test_split(
            np.arange(len(self.df)),
            test_size=val_size,
            stratify=stratify_col,
            random_state=random_state,
        )
        
        # Create subsets
        train_df = self.df.iloc[train_idx].reset_index(drop=True)
        val_df = self.df.iloc[val_idx].reset_index(drop=True)
        
        # Create datasets
        train_dataset = PatchDataset(
            patches=train_df,
            transform=self.transform,
            patch_path_column=self.patch_path_column,
            label_column=self.label_column,
        )
        
        val_dataset = PatchDataset(
            patches=val_df,
            transform=self.transform,
            patch_path_column=self.patch_path_column,
            label_column=self.label_column,
        )
        
        return train_dataset, val_dataset
    
    def save_metadata(self, path: Union[str, Path]) -> None:
        """
        Save dataset metadata to CSV.
        
        Args:
            path: Path to save the CSV file
        """
        self.df.to_csv(path, index=False)


class PatchFromWSIDataset(Dataset):
    """
    Dataset for extracting patches from WSIs on-the-fly.
    
    This class provides a patch-level interface to WSIs,
    extracting patches during iteration for training or inference.
    """
    
    def __init__(
        self,
        wsi_paths: List[Union[str, Path]],
        patch_size: int = 256,
        level: int = 0,
        stride: Optional[int] = None,
        transform: Optional[Callable] = None,
        mask_paths: Optional[List[Union[str, Path]]] = None,
        labels: Optional[List[Any]] = None,
        tissue_threshold: float = 0.5,
        min_tissue_percentage: float = 0.25,
        extraction_method: str = 'grid',
        num_patches_per_wsi: Optional[int] = None,
        cache_dir: Optional[Union[str, Path]] = None,
        save_extracted_patches: bool = False,
    ):
        """
        Initialize the patch extraction dataset.
        
        Args:
            wsi_paths: List of paths to whole slide images
            patch_size: Size of patches to extract
            level: Pyramid level to extract patches from
            stride: Stride between patches (if None, uses patch_size)
            transform: Transform to apply to patches
            mask_paths: List of paths to tissue masks
            labels: Labels for each WSI (for supervised learning)
            tissue_threshold: Threshold for binarizing tissue masks
            min_tissue_percentage: Minimum percentage of tissue required in a patch
            extraction_method: Patch extraction method ('grid' or 'random')
            num_patches_per_wsi: Number of patches to extract per WSI
            cache_dir: Directory to cache extracted patches
            save_extracted_patches: Whether to save extracted patches to disk
        """
        self.wsi_paths = [Path(path) for path in wsi_paths]
        self.patch_size = patch_size
        self.level = level
        self.stride = stride if stride is not None else patch_size
        self.transform = transform
        self.tissue_threshold = tissue_threshold
        self.min_tissue_percentage = min_tissue_percentage
        self.extraction_method = extraction_method
        self.num_patches_per_wsi = num_patches_per_wsi
        self.save_extracted_patches = save_extracted_patches
        
        # Set up masks
        if mask_paths is not None:
            self.mask_paths = [Path(path) if path else None for path in mask_paths]
            assert len(self.mask_paths) == len(self.wsi_paths), "Number of masks must match number of WSIs"
        else:
            self.mask_paths = [None] * len(self.wsi_paths)
        
        # Set up labels
        if labels is not None:
            assert len(labels) == len(self.wsi_paths), "Number of labels must match number of WSIs"
            self.labels = labels
        else:
            self.labels = [None] * len(self.wsi_paths)
        
        # Set up caching
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        if self.cache_dir is not None:
            os.makedirs(self.cache_dir, exist_ok=True)
        
        # Extract metadata for each WSI
        self.wsi_metadata = []
        self._extract_wsi_metadata()
        
        # Build coordinate list for patches
        self.coordinates = []
        self._build_coordinate_list()
    
    def _extract_wsi_metadata(self) -> None:
        """Extract metadata for each WSI."""
        for i, wsi_path in enumerate(tqdm(self.wsi_paths, desc="Processing WSIs")):
            try:
                # Open WSI
                wsi = WSIReader.open(wsi_path)
                
                # Get slide dimensions at specified level
                dims = wsi.slide_dimensions(level=self.level)
                
                # Add to metadata
                self.wsi_metadata.append({
                    'wsi_path': wsi_path,
                    'mask_path': self.mask_paths[i],
                    'width': dims[0],
                    'height': dims[1],
                    'label': self.labels[i],
                })
            except Exception as e:
                logger.error(f"Error processing WSI {wsi_path}: {e}")
                # Add placeholder
                self.wsi_metadata.append({
                    'wsi_path': wsi_path,
                    'mask_path': self.mask_paths[i],
                    'width': 0,
                    'height': 0,
                    'label': self.labels[i],
                    'error': str(e),
                })
    
    def _build_coordinate_list(self) -> None:
        """Build list of coordinates for each patch."""
        for i, metadata in enumerate(tqdm(self.wsi_metadata, desc="Building coordinate list")):
            # Skip WSIs with errors
            if 'error' in metadata:
                continue
            
            wsi_path = metadata['wsi_path']
            mask_path = metadata['mask_path']
            width = metadata['width']
            height = metadata['height']
            
            # Skip empty or invalid WSIs
            if width <= 0 or height <= 0:
                continue
            
            # Get mask if available
            mask = None
            if mask_path is not None and os.path.exists(mask_path):
                try:
                    mask = np.array(Image.open(mask_path).convert('L')) / 255.0
                except Exception as e:
                    logger.error(f"Error loading mask {mask_path}: {e}")
            
            # Extract coordinates based on method
            if self.extraction_method == 'grid':
                coords = self._get_grid_coordinates(i, width, height, mask)
            elif self.extraction_method == 'random':
                coords = self._get_random_coordinates(i, width, height, mask)
            else:
                raise ValueError(f"Invalid extraction method: {self.extraction_method}")
            
            # Add coordinates to list
            self.coordinates.extend(coords)
            
            # Log number of patches
            logger.info(f"Found {len(coords)} patches for WSI {wsi_path.name}")
        
        # Shuffle coordinates
        random.shuffle(self.coordinates)
        
        logger.info(f"Built coordinate list with {len(self.coordinates)} patches from {len(self.wsi_metadata)} WSIs")
    
    def _get_grid_coordinates(
        self,
        wsi_idx: int,
        width: int,
        height: int,
        mask: Optional[np.ndarray],
    ) -> List[Dict[str, Any]]:
        """
        Get grid-based coordinates for a WSI.
        
        Args:
            wsi_idx: Index of the WSI
            width: Width of the WSI at the specified level
            height: Height of the WSI at the specified level
            mask: Tissue mask as numpy array
            
        Returns:
            List of coordinate dictionaries
        """
        # Calculate the number of patches in each dimension
        num_patches_x = (width - self.patch_size) // self.stride + 1
        num_patches_y = (height - self.patch_size) // self.stride + 1
        
        # Ensure at least one patch
        num_patches_x = max(1, num_patches_x)
        num_patches_y = max(1, num_patches_y)
        
        # Get mask dimensions and scaling factors
        if mask is not None:
            mask_height, mask_width = mask.shape
            scale_x = width / mask_width
            scale_y = height / mask_height
        
        # Generate coordinates
        coords = []
        
        for y in range(num_patches_y):
            for x in range(num_patches_x):
                # Calculate patch coordinates
                x_coord = x * self.stride
                y_coord = y * self.stride
                
                # Check if patch is within mask
                if mask is not None:
                    # Convert to mask coordinates
                    mask_x = int(x_coord / scale_x)
                    mask_y = int(y_coord / scale_y)
                    mask_width = int(self.patch_size / scale_x)
                    mask_height = int(self.patch_size / scale_y)
                    
                    # Ensure coordinates are within mask
                    mask_x = min(mask_x, mask.shape[1] - 1)
                    mask_y = min(mask_y, mask.shape[0] - 1)
                    mask_width = min(mask_width, mask.shape[1] - mask_x)
                    mask_height = min(mask_height, mask.shape[0] - mask_y)
                    
                    # Check if patch has enough tissue
                    if mask_width > 0 and mask_height > 0:
                        mask_region = mask[mask_y:mask_y+mask_height, mask_x:mask_x+mask_width]
                        tissue_percentage = np.mean(mask_region > self.tissue_threshold)
                        
                        if tissue_percentage < self.min_tissue_percentage:
                            continue
                
                # Add to list
                coords.append({
                    'wsi_idx': wsi_idx,
                    'x': x_coord,
                    'y': y_coord,
                })
        
        # Limit number of patches if specified
        if self.num_patches_per_wsi is not None and len(coords) > self.num_patches_per_wsi:
            coords = random.sample(coords, self.num_patches_per_wsi)
        
        return coords
    
    def _get_random_coordinates(
        self,
        wsi_idx: int,
        width: int,
        height: int,
        mask: Optional[np.ndarray],
    ) -> List[Dict[str, Any]]:
        """
        Get random coordinates for a WSI.
        
        Args:
            wsi_idx: Index of the WSI
            width: Width of the WSI at the specified level
            height: Height of the WSI at the specified level
            mask: Tissue mask as numpy array
            
        Returns:
            List of coordinate dictionaries
        """
        # Determine number of patches
        num_patches = self.num_patches_per_wsi or 100  # Default to 100 if not specified
        
        # Get mask dimensions and scaling factors
        if mask is not None:
            mask_height, mask_width = mask.shape
            scale_x = width / mask_width
            scale_y = height / mask_height
        
        # Generate coordinates
        coords = []
        max_attempts = num_patches * 10
        attempts = 0
        
        while len(coords) < num_patches and attempts < max_attempts:
            attempts += 1
            
            # Generate random coordinates
            x_coord = random.randint(0, max(0, width - self.patch_size))
            y_coord = random.randint(0, max(0, height - self.patch_size))
            
            # Check if patch is within mask
            if mask is not None:
                # Convert to mask coordinates
                mask_x = int(x_coord / scale_x)
                mask_y = int(y_coord / scale_y)
                mask_width = int(self.patch_size / scale_x)
                mask_height = int(self.patch_size / scale_y)
                
                # Ensure coordinates are within mask
                mask_x = min(mask_x, mask.shape[1] - 1)
                mask_y = min(mask_y, mask.shape[0] - 1)
                mask_width = min(mask_width, mask.shape[1] - mask_x)
                mask_height = min(mask_height, mask.shape[0] - mask_y)
                
                # Check if patch has enough tissue
                if mask_width > 0 and mask_height > 0:
                    mask_region = mask[mask_y:mask_y+mask_height, mask_x:mask_x+mask_width]
                    tissue_percentage = np.mean(mask_region > self.tissue_threshold)
                    
                    if tissue_percentage < self.min_tissue_percentage:
                        continue
            
            # Add to list
            coords.append({
                'wsi_idx': wsi_idx,
                'x': x_coord,
                'y': y_coord,
            })
        
        return coords
    
    def __len__(self) -> int:
        """Get the total number of patches in the dataset."""
        return len(self.coordinates)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a patch from the dataset.
        
        Args:
            idx: Index of the patch
            
        Returns:
            Dictionary containing the patch and metadata
        """
        # Get coordinate information
        coord = self.coordinates[idx]
        wsi_idx = coord['wsi_idx']
        x_coord = coord['x']
        y_coord = coord['y']
        
        # Get WSI information
        metadata = self.wsi_metadata[wsi_idx]
        wsi_path = metadata['wsi_path']
        label = metadata['label']
        
        # Check if patch is cached
        cache_path = None
        if self.cache_dir is not None:
            cache_filename = f"{wsi_path.stem}_x{x_coord}_y{y_coord}_level{self.level}.png"
            cache_path = self.cache_dir / cache_filename
            
            if cache_path.exists():
                # Load from cache
                patch = Image.open(cache_path).convert('RGB')
            else:
                # Extract from WSI
                patch = self._extract_patch(wsi_path, x_coord, y_coord)
                
                # Save to cache
                patch.save(cache_path)
        else:
            # Extract from WSI
            patch = self._extract_patch(wsi_path, x_coord, y_coord)
            
            # Save if requested
            if self.save_extracted_patches:
                save_dir = wsi_path.parent / "patches" / wsi_path.stem
                os.makedirs(save_dir, exist_ok=True)
                save_path = save_dir / f"x{x_coord}_y{y_coord}_level{self.level}.png"
                patch.save(save_path)
        
        # Apply transform if available
        if self.transform is not None:
            patch = self.transform(patch)
        
        # Convert to tensor if not already
        if not isinstance(patch, torch.Tensor):
            patch = torch.tensor(np.array(patch)).permute(2, 0, 1).float() / 255.0
        
        # Return patch and metadata
        return {
            'patch': patch,
            'label': label,
            'wsi_path': str(wsi_path),
            'x': x_coord,
            'y': y_coord,
            'level': self.level,
            'cache_path': str(cache_path) if cache_path else None,
        }
    
    def _extract_patch(self, wsi_path: Path, x: int, y: int) -> Image.Image:
        """
        Extract a patch from a WSI.
        
        Args:
            wsi_path: Path to the WSI
            x: X-coordinate (left)
            y: Y-coordinate (top)
            
        Returns:
            Extracted patch as PIL Image
        """
        # Open WSI
        wsi = WSIReader.open(wsi_path)
        
        # Extract patch
        patch = wsi.read_region(
            location=(x, y),
            size=(self.patch_size, self.patch_size),
            level=self.level,
        )
        
        # Convert to RGB if necessary
        if patch.shape[-1] == 4:  # RGBA
            patch = patch[:, :, :3]  # Remove alpha channel
        
        # Convert to PIL Image
        patch = Image.fromarray(patch)
        
        return patch
    
    def get_dataset_df(self) -> pd.DataFrame:
        """
        Get a DataFrame representation of the dataset.
        
        Returns:
            DataFrame with patch information
        """
        # Create DataFrame
        data = []
        
        for coord in self.coordinates:
            wsi_idx = coord['wsi_idx']
            metadata = self.wsi_metadata[wsi_idx]
            
            data.append({
                'wsi_path': str(metadata['wsi_path']),
                'x': coord['x'],
                'y': coord['y'],
                'level': self.level,
                'patch_size': self.patch_size,
                'label': metadata['label'],
            })
        
        return pd.DataFrame(data)
    
    def extract_all_patches(
        self,
        output_dir: Union[str, Path],
        batch_size: int = 16,
        num_workers: int = 4,
    ) -> pd.DataFrame:
        """
        Extract all patches and save to disk.
        
        Args:
            output_dir: Directory to save patches
            batch_size: Batch size for extraction
            num_workers: Number of workers for extraction
            
        Returns:
            DataFrame with patch information
        """
        output_dir = Path(output_dir)
        os.makedirs(output_dir, exist_ok=True)
        
        # Create DataLoader
        dataloader = DataLoader(
            self,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )
        
        # Extract patches
        patch_info = []
        
        for batch in tqdm(dataloader, desc="Extracting patches"):
            patches = batch['patch']
            labels = batch['label']
            wsi_paths = batch['wsi_path']
            x_coords = batch['x']
            y_coords = batch['y']
            levels = batch['level']
            
            for i in range(len(patches)):
                # Get patch information
                patch = patches[i]
                label = labels[i] if labels[i] is not None else "unlabeled"
                wsi_path = wsi_paths[i]
                x = x_coords[i].item()
                y = y_coords[i].item()
                level = levels[i].item()
                
                # Create filename
                wsi_name = Path(wsi_path).stem
                filename = f"{wsi_name}_x{x}_y{y}_level{level}.png"
                
                # Create directory for label
                label_dir = output_dir / str(label)
                os.makedirs(label_dir, exist_ok=True)
                
                # Save patch
                save_path = label_dir / filename
                
                # Convert tensor to PIL Image
                patch_img = (patch.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                Image.fromarray(patch_img).save(save_path)
                
                # Add to patch info
                patch_info.append({
                    'patch_path': str(save_path),
                    'wsi_path': wsi_path,
                    'x': x,
                    'y': y,
                    'level': level,
                    'label': label,
                })
        
        # Create DataFrame
        df = pd.DataFrame(patch_info)
        
        # Save metadata
        df.to_csv(output_dir / "patch_metadata.csv", index=False)
        
        return df
