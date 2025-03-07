"""
Whole Slide Image (WSI) dataset for the tumor_highlighter package.

This module provides classes for loading and processing WSIs,
extracting patches, and creating datasets for training and inference.
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Callable, Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm

from tiatoolbox.wsicore.wsireader import WSIReader
from tiatoolbox.tools.patchextraction import SlidingWindowPatchExtractor

from tumor_highlighter.data.mask_generator import HSVTissueMaskGenerator

logger = logging.getLogger(__name__)


class WSIDataset(Dataset):
    """
    Dataset for whole slide images using TIAToolbox.
    
    This dataset loads WSIs using TIAToolbox's WSIReader and extracts
    patches on-the-fly for training or inference.
    """
    
    def __init__(
        self,
        wsi_paths: List[Union[str, Path]],
        patch_size: int = 256,
        level: int = 0,
        stride: Optional[int] = None,
        mask_paths: Optional[List[Union[str, Path]]] = None,
        generate_masks: bool = False,
        transform: Optional[Callable] = None,
        labels: Optional[List[Any]] = None,
        mask_threshold: float = 0.5,
        tissue_percentage_threshold: float = 0.25,
        num_patches_per_slide: Optional[int] = None,
        random_sampling: bool = False,
        cache_dir: Optional[Union[str, Path]] = None,
    ):
        """
        Initialize the WSI dataset.
        
        Args:
            wsi_paths: List of paths to whole slide images
            patch_size: Size of patches to extract
            level: Pyramid level to extract patches from
            stride: Stride between patches (if None, uses patch_size)
            mask_paths: List of paths to tissue masks (if None and generate_masks=True, 
                        masks will be generated using HSVTissueMaskGenerator)
            generate_masks: Whether to generate tissue masks if not provided
            transform: Transform to apply to patches
            labels: Optional labels for each WSI (for supervised learning)
            mask_threshold: Threshold for binarizing masks
            tissue_percentage_threshold: Minimum percentage of tissue required in a patch
            num_patches_per_slide: Number of patches to extract per slide 
                                   (if None, extracts all patches)
            random_sampling: Whether to randomly sample patches (if True, 
                            ignores stride and extracts random patches)
            cache_dir: Directory to cache extracted patches
        """
        self.wsi_paths = [Path(path) for path in wsi_paths]
        self.patch_size = patch_size
        self.level = level
        self.stride = stride if stride is not None else patch_size
        self.transform = transform
        self.mask_threshold = mask_threshold
        self.tissue_percentage_threshold = tissue_percentage_threshold
        self.num_patches_per_slide = num_patches_per_slide
        self.random_sampling = random_sampling
        
        # Set up mask paths or generate masks
        if mask_paths is not None:
            self.mask_paths = [Path(path) for path in mask_paths]
        elif generate_masks:
            logger.info("Generating tissue masks for WSIs")
            self.mask_paths = self._generate_masks()
        else:
            self.mask_paths = [None] * len(self.wsi_paths)
        
        # Set up labels
        if labels is not None:
            assert len(labels) == len(wsi_paths), "Number of labels must match number of WSIs"
            self.labels = labels
        else:
            self.labels = [None] * len(self.wsi_paths)
        
        # Set up caching
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        if self.cache_dir is not None:
            os.makedirs(self.cache_dir, exist_ok=True)
        
        # Initialize patch indices for each WSI
        self.patch_indices = []
        self._initialize_patch_indices()
    
    def _generate_masks(self) -> List[Path]:
        """
        Generate tissue masks for WSIs using HSVTissueMaskGenerator.
        
        Returns:
            List of paths to generated masks
        """
        mask_generator = HSVTissueMaskGenerator(downsample_factor=64)
        mask_paths = []
        
        for i, wsi_path in enumerate(tqdm(self.wsi_paths, desc="Generating masks")):
            # Generate mask filename
            mask_filename = wsi_path.stem + "_tissue_mask.png"
            mask_dir = wsi_path.parent / "masks"
            os.makedirs(mask_dir, exist_ok=True)
            mask_path = mask_dir / mask_filename
            
            # Generate mask if it doesn't exist
            if not mask_path.exists():
                try:
                    # Generate mask
                    mask_generator.generate_mask(wsi_path, mask_path)
                    logger.info(f"Generated mask for {wsi_path}")
                except Exception as e:
                    logger.error(f"Error generating mask for {wsi_path}: {e}")
                    mask_path = None
            
            mask_paths.append(mask_path)
        
        return mask_paths
    
    def _initialize_patch_indices(self) -> None:
        """Initialize patch indices for each WSI based on tissue masks."""
        total_patches = 0
        self.slide_info = []
        
        for i, (wsi_path, mask_path) in enumerate(zip(self.wsi_paths, self.mask_paths)):
            slide_patches = []
            
            try:
                # Open WSI
                wsi = WSIReader.open(wsi_path)
                
                # Get slide dimensions at specified level
                dims = wsi.slide_dimensions(level=self.level)
                width, height = dims
                
                # Load mask if available
                mask = None
                if mask_path is not None and os.path.exists(mask_path):
                    mask = np.array(Image.open(mask_path).convert('L')) / 255.0
                
                # Store slide info
                self.slide_info.append({
                    'wsi_path': wsi_path,
                    'mask_path': mask_path,
                    'width': width,
                    'height': height,
                    'label': self.labels[i],
                })
                
                if self.random_sampling and self.num_patches_per_slide is not None:
                    # Generate random patch coordinates within tissue regions
                    slide_patches = self._get_random_patches(wsi, mask, i)
                else:
                    # Generate grid-based patch coordinates
                    slide_patches = self._get_grid_patches(wsi, mask, i)
                
                # Limit number of patches if specified
                if self.num_patches_per_slide is not None and not self.random_sampling:
                    if len(slide_patches) > self.num_patches_per_slide:
                        # Randomly select patches
                        indices = np.random.choice(
                            len(slide_patches),
                            self.num_patches_per_slide,
                            replace=False
                        )
                        slide_patches = [slide_patches[j] for j in indices]
                
                total_patches += len(slide_patches)
                
            except Exception as e:
                logger.error(f"Error processing WSI {wsi_path}: {e}")
                slide_patches = []
            
            self.patch_indices.append(slide_patches)
        
        logger.info(f"Initialized dataset with {len(self.wsi_paths)} WSIs and {total_patches} patches")
    
    def _get_grid_patches(
        self,
        wsi: WSIReader,
        mask: Optional[np.ndarray],
        slide_idx: int
    ) -> List[Dict[str, Any]]:
        """
        Get grid-based patch coordinates for a WSI.
        
        Args:
            wsi: TIAToolbox WSIReader for the WSI
            mask: Tissue mask as numpy array
            slide_idx: Index of the slide in the dataset
            
        Returns:
            List of patch information dictionaries
        """
        # Get slide dimensions at the specified level
        dims = wsi.slide_dimensions(level=self.level)
        width, height = dims
        
        # Calculate the number of patches in each dimension
        num_patches_x = (width - self.patch_size) // self.stride + 1
        num_patches_y = (height - self.patch_size) // self.stride + 1
        
        # Ensure at least one patch
        num_patches_x = max(1, num_patches_x)
        num_patches_y = max(1, num_patches_y)
        
        # Get the downsample factor for the mask
        if mask is not None:
            mask_height, mask_width = mask.shape
            downsample_x = width / mask_width
            downsample_y = height / mask_height
        
        # Generate patch coordinates
        patch_info_list = []
        
        for y in range(num_patches_y):
            for x in range(num_patches_x):
                # Calculate patch coordinates
                x_coord = x * self.stride
                y_coord = y * self.stride
                
                # Check if patch is within mask (if available)
                if mask is not None:
                    # Convert patch coordinates to mask coordinates
                    mask_x = int(x_coord / downsample_x)
                    mask_y = int(y_coord / downsample_y)
                    mask_width = int(self.patch_size / downsample_x)
                    mask_height = int(self.patch_size / downsample_y)
                    
                    # Ensure coordinates are within mask
                    mask_x = min(mask_x, mask.shape[1] - 1)
                    mask_y = min(mask_y, mask.shape[0] - 1)
                    mask_width = min(mask_width, mask.shape[1] - mask_x)
                    mask_height = min(mask_height, mask.shape[0] - mask_y)
                    
                    # Check if patch has enough tissue
                    if mask_width > 0 and mask_height > 0:
                        mask_region = mask[mask_y:mask_y+mask_height, mask_x:mask_x+mask_width]
                        tissue_percentage = np.mean(mask_region > self.mask_threshold)
                        
                        if tissue_percentage < self.tissue_percentage_threshold:
                            continue
                
                # Add patch to list
                patch_info_list.append({
                    'slide_idx': slide_idx,
                    'x': x_coord,
                    'y': y_coord,
                    'level': self.level,
                    'size': self.patch_size,
                })
        
        return patch_info_list
    
    def _get_random_patches(
        self,
        wsi: WSIReader,
        mask: Optional[np.ndarray],
        slide_idx: int
    ) -> List[Dict[str, Any]]:
        """
        Get random patch coordinates for a WSI.
        
        Args:
            wsi: TIAToolbox WSIReader for the WSI
            mask: Tissue mask as numpy array
            slide_idx: Index of the slide in the dataset
            
        Returns:
            List of patch information dictionaries
        """
        # Get slide dimensions at the specified level
        dims = wsi.slide_dimensions(level=self.level)
        width, height = dims
        
        # Get the downsample factor for the mask
        if mask is not None:
            mask_height, mask_width = mask.shape
            downsample_x = width / mask_width
            downsample_y = height / mask_height
        
        patch_info_list = []
        attempts = 0
        max_attempts = self.num_patches_per_slide * 10  # Limit the number of attempts
        
        while len(patch_info_list) < self.num_patches_per_slide and attempts < max_attempts:
            attempts += 1
            
            # Generate random coordinates
            x_coord = np.random.randint(0, max(1, width - self.patch_size))
            y_coord = np.random.randint(0, max(1, height - self.patch_size))
            
            # Check if patch is within mask (if available)
            if mask is not None:
                # Convert patch coordinates to mask coordinates
                mask_x = int(x_coord / downsample_x)
                mask_y = int(y_coord / downsample_y)
                mask_width = int(self.patch_size / downsample_x)
                mask_height = int(self.patch_size / downsample_y)
                
                # Ensure coordinates are within mask
                mask_x = min(mask_x, mask.shape[1] - 1)
                mask_y = min(mask_y, mask.shape[0] - 1)
                mask_width = min(mask_width, mask.shape[1] - mask_x)
                mask_height = min(mask_height, mask.shape[0] - mask_y)
                
                # Check if patch has enough tissue
                if mask_width > 0 and mask_height > 0:
                    mask_region = mask[mask_y:mask_y+mask_height, mask_x:mask_x+mask_width]
                    tissue_percentage = np.mean(mask_region > self.mask_threshold)
                    
                    if tissue_percentage < self.tissue_percentage_threshold:
                        continue
            
            # Add patch to list
            patch_info_list.append({
                'slide_idx': slide_idx,
                'x': x_coord,
                'y': y_coord,
                'level': self.level,
                'size': self.patch_size,
            })
        
        return patch_info_list
    
    def __len__(self) -> int:
        """Get the total number of patches in the dataset."""
        return sum(len(patches) for patches in self.patch_indices)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a patch from the dataset.
        
        Args:
            idx: Global index of the patch
            
        Returns:
            Dictionary containing the patch, its coordinates, and slide information
        """
        # Find which slide this patch belongs to
        slide_idx = 0
        patch_idx = idx
        
        while slide_idx < len(self.patch_indices):
            num_patches = len(self.patch_indices[slide_idx])
            if patch_idx < num_patches:
                break
            patch_idx -= num_patches
            slide_idx += 1
        
        # Get patch info
        patch_info = self.patch_indices[slide_idx][patch_idx]
        wsi_path = self.wsi_paths[slide_idx]
        label = self.labels[slide_idx]
        
        # Check if patch is cached
        if self.cache_dir is not None:
            cache_filename = f"{wsi_path.stem}_x{patch_info['x']}_y{patch_info['y']}_level{patch_info['level']}.png"
            cache_path = self.cache_dir / cache_filename
            
            if cache_path.exists():
                # Load patch from cache
                patch = Image.open(cache_path).convert('RGB')
            else:
                # Extract patch from WSI
                patch = self._extract_patch(wsi_path, patch_info)
                
                # Save to cache
                patch.save(cache_path)
        else:
            # Extract patch from WSI
            patch = self._extract_patch(wsi_path, patch_info)
        
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
            'slide_idx': slide_idx,
            'x': patch_info['x'],
            'y': patch_info['y'],
            'level': patch_info['level'],
            'size': patch_info['size'],
        }
    
    def _extract_patch(self, wsi_path: Path, patch_info: Dict[str, Any]) -> Image.Image:
        """
        Extract a patch from a WSI.
        
        Args:
            wsi_path: Path to the WSI
            patch_info: Dictionary containing patch coordinates and level
            
        Returns:
            Extracted patch as PIL Image
        """
        # Open WSI
        wsi = WSIReader.open(wsi_path)
        
        # Extract patch
        x = patch_info['x']
        y = patch_info['y']
        level = patch_info['level']
        size = patch_info['size']
        
        patch = wsi.read_region(
            location=(x, y),
            size=(size, size),
            level=level,
        )
        
        # Convert to RGB if necessary
        if patch.shape[-1] == 4:  # RGBA
            patch = patch[:, :, :3]  # Remove alpha channel
        
        # Convert to PIL Image
        patch = Image.fromarray(patch)
        
        return patch
    
    def get_slide_patches(self, slide_idx: int) -> List[Dict[str, Any]]:
        """
        Get all patches for a specific slide.
        
        Args:
            slide_idx: Index of the slide
            
        Returns:
            List of patches from the specified slide
        """
        # Check if slide index is valid
        if slide_idx < 0 or slide_idx >= len(self.wsi_paths):
            raise ValueError(f"Invalid slide index: {slide_idx}")
        
        # Get all patches for this slide
        patches = []
        
        # Get global start index for this slide
        start_idx = 0
        for i in range(slide_idx):
            start_idx += len(self.patch_indices[i])
        
        # Get all patches
        for i in range(len(self.patch_indices[slide_idx])):
            patches.append(self[start_idx + i])
        
        return patches
    
    def get_wsi_dataloader(
        self,
        slide_idx: int,
        batch_size: int = 16,
        shuffle: bool = False,
        num_workers: int = 4,
    ) -> DataLoader:
        """
        Get a DataLoader for a specific WSI.
        
        Args:
            slide_idx: Index of the slide
            batch_size: Batch size for the DataLoader
            shuffle: Whether to shuffle the patches
            num_workers: Number of workers for the DataLoader
            
        Returns:
            DataLoader for the specified slide
        """
        # Create a subset dataset for this slide
        class SlideSubset(Dataset):
            def __init__(self, dataset, slide_idx):
                self.dataset = dataset
                self.slide_idx = slide_idx
                self.patch_indices = dataset.patch_indices[slide_idx]
                
                # Get global start index for this slide
                self.start_idx = 0
                for i in range(slide_idx):
                    self.start_idx += len(dataset.patch_indices[i])
            
            def __len__(self):
                return len(self.patch_indices)
            
            def __getitem__(self, idx):
                return self.dataset[self.start_idx + idx]
        
        # Create subset
        subset = SlideSubset(self, slide_idx)
        
        # Create DataLoader
        dataloader = DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
        )
        
        return dataloader


class WSITable:
    """
    Helper class for working with tables of WSI information.
    
    This class provides utilities for loading, filtering, and processing
    tables of WSI information (typically CSV files) for use with the
    WSIDataset class.
    """
    
    def __init__(
        self,
        table_path: Union[str, Path, pd.DataFrame],
        wsi_path_column: str = 'slide_path',
        label_column: Optional[str] = None,
        mask_path_column: Optional[str] = None,
        id_column: Optional[str] = None,
    ):
        """
        Initialize the WSI table.
        
        Args:
            table_path: Path to CSV file or DataFrame containing WSI information
            wsi_path_column: Column name containing WSI paths
            label_column: Column name containing labels
            mask_path_column: Column name containing mask paths
            id_column: Column name containing unique identifiers for WSIs
        """
        # Load table
        if isinstance(table_path, (str, Path)):
            self.table = pd.read_csv(table_path)
        else:
            self.table = table_path
        
        # Set column names
        self.wsi_path_column = wsi_path_column
        self.label_column = label_column
        self.mask_path_column = mask_path_column
        self.id_column = id_column
        
        # Validate table
        if self.wsi_path_column not in self.table.columns:
            raise ValueError(f"WSI path column '{self.wsi_path_column}' not found in table")
        
        if self.label_column is not None and self.label_column not in self.table.columns:
            logger.warning(f"Label column '{self.label_column}' not found in table")
            self.label_column = None
        
        if self.mask_path_column is not None and self.mask_path_column not in self.table.columns:
            logger.warning(f"Mask path column '{self.mask_path_column}' not found in table")
            self.mask_path_column = None
        
        if self.id_column is not None and self.id_column not in self.table.columns:
            logger.warning(f"ID column '{self.id_column}' not found in table")
            self.id_column = None
        
        # Validate WSI paths
        valid_paths = []
        for path in self.table[self.wsi_path_column]:
            if os.path.exists(path):
                valid_paths.append(True)
            else:
                valid_paths.append(False)
                logger.warning(f"WSI path not found: {path}")
        
        self.table['valid_path'] = valid_paths
        logger.info(f"Found {sum(valid_paths)}/{len(valid_paths)} valid WSI paths")
    
    def filter(self, **kwargs) -> 'WSITable':
        """
        Filter the WSI table based on column values.
        
        Args:
            **kwargs: Keyword arguments for filtering (column_name=value)
            
        Returns:
            New WSITable with filtered rows
        """
        # Create a copy of the filtered table
        filtered_table = self.table.copy()
        
        # Apply filters
        for column, value in kwargs.items():
            if column in filtered_table.columns:
                if isinstance(value, list):
                    filtered_table = filtered_table[filtered_table[column].isin(value)]
                else:
                    filtered_table = filtered_table[filtered_table[column] == value]
            else:
                logger.warning(f"Column '{column}' not found in table, skipping filter")
        
        # Create new WSITable with filtered table
        return WSITable(
            filtered_table,
            wsi_path_column=self.wsi_path_column,
            label_column=self.label_column,
            mask_path_column=self.mask_path_column,
            id_column=self.id_column,
        )
    
    def create_dataset(
        self,
        patch_size: int = 256,
        level: int = 0,
        stride: Optional[int] = None,
        generate_masks: bool = False,
        transform: Optional[Callable] = None,
        mask_threshold: float = 0.5,
        tissue_percentage_threshold: float = 0.25,
        num_patches_per_slide: Optional[int] = None,
        random_sampling: bool = False,
        cache_dir: Optional[Union[str, Path]] = None,
        valid_only: bool = True,
    ) -> WSIDataset:
        """
        Create a WSIDataset from the table.
        
        Args:
            patch_size: Size of patches to extract
            level: Pyramid level to extract patches from
            stride: Stride between patches (if None, uses patch_size)
            generate_masks: Whether to generate tissue masks if not provided
            transform: Transform to apply to patches
            mask_threshold: Threshold for binarizing masks
            tissue_percentage_threshold: Minimum percentage of tissue required in a patch
            num_patches_per_slide: Number of patches to extract per slide
            random_sampling: Whether to randomly sample patches
            cache_dir: Directory to cache extracted patches
            valid_only: Whether to only include WSIs with valid paths
            
        Returns:
            WSIDataset created from the table
        """
        # Filter valid paths if requested
        if valid_only:
            table = self.table[self.table['valid_path']].copy()
        else:
            table = self.table.copy()
        
        # Get WSI paths
        wsi_paths = table[self.wsi_path_column].tolist()
        
        # Get mask paths if available
        mask_paths = None
        if self.mask_path_column is not None:
            mask_paths = table[self.mask_path_column].tolist()
        
        # Get labels if available
        labels = None
        if self.label_column is not None:
            labels = table[self.label_column].tolist()
        
        # Create dataset
        dataset = WSIDataset(
            wsi_paths=wsi_paths,
            patch_size=patch_size,
            level=level,
            stride=stride,
            mask_paths=mask_paths,
            generate_masks=generate_masks,
            transform=transform,
            labels=labels,
            mask_threshold=mask_threshold,
            tissue_percentage_threshold=tissue_percentage_threshold,
            num_patches_per_slide=num_patches_per_slide,
            random_sampling=random_sampling,
            cache_dir=cache_dir,
        )
        
        return dataset
    
    def save(self, path: Union[str, Path]) -> None:
        """
        Save the WSI table to CSV.
        
        Args:
            path: Path to save the CSV file
        """
        self.table.to_csv(path, index=False)
    
    def __len__(self) -> int:
        """Get the number of WSIs in the table."""
        return len(self.table)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get information for a specific WSI.
        
        Args:
            idx: Index of the WSI
            
        Returns:
            Dictionary containing WSI information
        """
        row = self.table.iloc[idx]
        return row.to_dict()
    
    @property
    def wsi_paths(self) -> List[str]:
        """Get list of WSI paths."""
        return self.table[self.wsi_path_column].tolist()
    
    @property
    def mask_paths(self) -> Optional[List[str]]:
        """Get list of mask paths if available."""
        if self.mask_path_column is not None:
            return self.table[self.mask_path_column].tolist()
        return None
    
    @property
    def labels(self) -> Optional[List[Any]]:
        """Get list of labels if available."""
        if self.label_column is not None:
            return self.table[self.label_column].tolist()
        return None
    
    @property
    def ids(self) -> Optional[List[str]]:
        """Get list of WSI IDs if available."""
        if self.id_column is not None:
            return self.table[self.id_column].tolist()
        return None
