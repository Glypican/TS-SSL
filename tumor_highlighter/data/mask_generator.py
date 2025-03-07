"""
Module for generating tissue masks from whole slide images.
Implements custom HSV-based masking technique at 64x downsample following
the parameters from the original implementation.
"""

import os
import numpy as np
import cv2
from tiatoolbox.wsicore.wsireader import WSIReader
from typing import Tuple, Union, Optional
from pathlib import Path
import logging
from PIL import Image
from skimage.color import rgb2hsv

# Configure logging
logger = logging.getLogger(__name__)


class HSVTissueMaskGenerator:
    """
    Tissue mask generator using HSV color space thresholding.
    
    This generator creates binary masks indicating tissue regions by:
    1. Reading a thumbnail of the WSI at specified downsample (default 64x)
    2. Converting to HSV color space
    3. Applying optimized thresholds to identify tissue regions
    4. (Optional) Performing morphological operations to clean up the mask
    
    The HSV thresholds have been optimized for H&E stained slides:
    - Hue: between 0.6 and 0.98
    - Saturation: greater than 0.04
    - Value: greater than 0.2
    """
    
    def __init__(
        self,
        downsample_factor: int = 64,
        apply_morphology: bool = False,
        opening_kernel_size: int = 5,
        closing_kernel_size: int = 5,
        min_tissue_size: int = 1000,
    ):
        """
        Initialize the HSV tissue mask generator.
        
        Args:
            downsample_factor: Factor by which to downsample the WSI (default 64)
            apply_morphology: Whether to apply morphological operations
            opening_kernel_size: Size of kernel for morphological opening
            closing_kernel_size: Size of kernel for morphological closing
            min_tissue_size: Minimum contour area to be considered tissue
        """
        self.downsample_factor = downsample_factor
        self.apply_morphology = apply_morphology
        self.opening_kernel = np.ones((opening_kernel_size, opening_kernel_size), np.uint8)
        self.closing_kernel = np.ones((closing_kernel_size, closing_kernel_size), np.uint8)
        self.min_tissue_size = min_tissue_size
    
    def generate_mask(
        self, 
        slide_path: Union[str, Path], 
        save_path: Optional[Union[str, Path]] = None
    ) -> np.ndarray:
        """
        Generate a binary tissue mask for the given slide.
        
        Args:
            slide_path: Path to the whole slide image
            save_path: Optional path to save the generated mask
            
        Returns:
            Binary mask as numpy array where 1 indicates tissue
        """
        # Open the slide
        slide = WSIReader.open(slide_path)
        
        # Get a thumbnail at the specified downsample factor
        thumb = slide.slide_thumbnail(resolution=1.0, units="level", 
                                     level=int(np.log2(self.downsample_factor)))
        
        # Convert to HSV color space
        hsv = rgb2hsv(thumb)
        
        # Create masks based on hue, saturation, and value
        # These parameters were optimized in the original code
        hue_mask = (hsv[:, :, 0] > 0.6) & (hsv[:, :, 0] < 0.98)
        sat_mask = (hsv[:, :, 1] > 0.04)
        val_mask = (hsv[:, :, 2] > 0.2)
        
        # Combine masks to get final tissue mask
        binary_mask = hue_mask & sat_mask & val_mask
        
        # Apply morphological operations if requested
        if self.apply_morphology:
            # Convert to uint8 for OpenCV
            mask_uint8 = (binary_mask * 255).astype(np.uint8)
            
            # Perform morphological operations to clean up the mask
            mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_OPEN, self.opening_kernel)
            mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, self.closing_kernel)
            
            # Find contours and filter out small artifacts
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            filtered_mask = np.zeros_like(mask_uint8)
            
            for contour in contours:
                if cv2.contourArea(contour) >= self.min_tissue_size:
                    cv2.drawContours(filtered_mask, [contour], -1, 255, -1)
            
            # Convert back to binary 0-1 mask
            binary_mask = (filtered_mask > 0).astype(np.uint8)
        
        # Calculate tissue percentage
        tissue_percentage = np.mean(binary_mask)
        logger.info(f"Tissue area percentage: {tissue_percentage:.2%}")
        
        # Save the mask if requested
        if save_path is not None:
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            mask_image = Image.fromarray((binary_mask * 255).astype(np.uint8))
            mask_image.save(save_path)
            logger.info(f"Saved tissue mask to {save_path}")
        
        return binary_mask
    
    def generate_thumbnail_and_mask(
        self,
        slide_path: Union[str, Path],
        thumb_path: Optional[Union[str, Path]] = None,
        mask_path: Optional[Union[str, Path]] = None
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Generate both a thumbnail and tissue mask for the given slide.
        
        Args:
            slide_path: Path to the whole slide image
            thumb_path: Optional path to save the generated thumbnail
            mask_path: Optional path to save the generated mask
            
        Returns:
            Tuple of (thumbnail, mask, tissue_percentage)
        """
        # Open the slide
        slide = WSIReader.open(slide_path)
        
        # Find closest downsample level
        downsample_factors = slide.info.level_downsamples
        closest_level = np.argmin(np.abs(np.array(downsample_factors) - self.downsample_factor))
        actual_downsample = downsample_factors[closest_level]
        
        # Read thumbnail at selected level
        thumb = slide.read_region(
            location=(0, 0),
            level=closest_level,
            size=slide.info.level_dimensions[closest_level]
        )
        
        # Remove alpha channel if present
        if thumb.shape[-1] == 4:
            thumb = thumb[:, :, :3]
        
        # Save thumbnail if requested
        if thumb_path is not None:
            os.makedirs(os.path.dirname(os.path.abspath(thumb_path)), exist_ok=True)
            Image.fromarray(thumb).save(thumb_path)
            logger.info(f"Saved thumbnail with actual downsample {actual_downsample} to {thumb_path}")
        
        # Convert to HSV color space
        hsv = rgb2hsv(thumb)
        
        # Create masks based on hue, saturation, and value
        # These parameters were optimized in the reference code
        hue_mask = (hsv[:, :, 0] > 0.6) & (hsv[:, :, 0] < 0.98)
        sat_mask = (hsv[:, :, 1] > 0.04)
        val_mask = (hsv[:, :, 2] > 0.2)
        
        # Combine masks to get final tissue mask
        binary_mask = hue_mask & sat_mask & val_mask
        
        # Apply morphological operations if requested
        if self.apply_morphology:
            # Convert to uint8 for OpenCV
            mask_uint8 = (binary_mask * 255).astype(np.uint8)
            
            # Perform morphological operations to clean up the mask
            mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_OPEN, self.opening_kernel)
            mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, self.closing_kernel)
            
            # Find contours and filter out small artifacts
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            filtered_mask = np.zeros_like(mask_uint8)
            
            for contour in contours:
                if cv2.contourArea(contour) >= self.min_tissue_size:
                    cv2.drawContours(filtered_mask, [contour], -1, 255, -1)
            
            # Convert back to binary 0-1 mask
            binary_mask = (filtered_mask > 0).astype(np.uint8)
        
        # Calculate tissue percentage
        tissue_percentage = np.mean(binary_mask)
        logger.info(f"Tissue area percentage: {tissue_percentage:.2%}")
        
        # Save the mask if requested
        if mask_path is not None:
            os.makedirs(os.path.dirname(os.path.abspath(mask_path)), exist_ok=True)
            mask_image = Image.fromarray((binary_mask * 255).astype(np.uint8))
            mask_image.save(mask_path)
            logger.info(f"Saved tissue mask to {mask_path}")
        
        return thumb, binary_mask, tissue_percentage
    
    def calculate_tissue_percentage(self, patch: Union[np.ndarray, Image.Image]) -> float:
        """
        Calculate the percentage of tissue in a patch using HSV thresholds.
        
        Args:
            patch: Image patch as numpy array or PIL Image
            
        Returns:
            Tissue percentage (0.0 to 1.0)
        """
        # Convert to numpy array if PIL Image
        if isinstance(patch, Image.Image):
            patch_array = np.array(patch)
        else:
            patch_array = patch
        
        # Convert to HSV color space
        hsv_arr = rgb2hsv(patch_array)
        
        # Create masks based on hue, saturation, and value
        # Use the same parameters as in create_tissue_mask for consistency
        hue_mask = (hsv_arr[:, :, 0] > 0.6) & (hsv_arr[:, :, 0] < 0.98)
        sat_mask = (hsv_arr[:, :, 1] > 0.04)
        val_mask = (hsv_arr[:, :, 2] > 0.2)
        
        # Combine masks to get final tissue mask
        is_tissue = hue_mask & sat_mask & val_mask
        
        # Calculate percentage of tissue pixels
        return np.mean(is_tissue)
