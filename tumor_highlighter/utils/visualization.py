"""
Visualization utilities for the tumor_highlighter package.

This module provides functions for visualizing patches, WSIs, model predictions,
attention heatmaps, and training metrics for diagnostic and presentation purposes.
"""

import os
import io
import logging
import math
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Callable, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from PIL import Image
import cv2
import torch
from tqdm import tqdm

from tiatoolbox.wsicore.wsireader import WSIReader

logger = logging.getLogger(__name__)


def visualize_patches(
    patches: Union[List[np.ndarray], List[torch.Tensor], np.ndarray, torch.Tensor],
    labels: Optional[List[Any]] = None,
    predictions: Optional[List[Any]] = None,
    max_patches: int = 25,
    cols: int = 5,
    figsize: Tuple[int, int] = None,
    title: str = "Patch Visualization",
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True,
    cmap: str = None,
    denormalize: bool = True,
) -> Optional[plt.Figure]:
    """
    Visualize a grid of image patches.
    
    Args:
        patches: List of patches as numpy arrays or torch tensors
        labels: Optional list of labels for each patch
        predictions: Optional list of predictions for each patch
        max_patches: Maximum number of patches to display
        cols: Number of columns in the grid
        figsize: Figure size (width, height) in inches
        title: Title for the plot
        save_path: Path to save the visualization
        show: Whether to display the plot
        cmap: Colormap for grayscale images
        denormalize: Whether to denormalize patch values from [0, 1] to [0, 255]
        
    Returns:
        Matplotlib figure if show=False, otherwise None
    """
    # Ensure patches is a list or convert from tensor/array
    if isinstance(patches, torch.Tensor):
        # Convert batch tensor to list of numpy arrays
        if patches.dim() == 4:  # B x C x H x W
            patches = [p.cpu().numpy() for p in patches]
        else:  # Single tensor
            patches = [patches.cpu().numpy()]
    elif isinstance(patches, np.ndarray):
        # Convert batch array to list of numpy arrays
        if patches.ndim == 4:  # B x C x H x W or B x H x W x C
            # Determine if it's in BCHW or BHWC format
            if patches.shape[1] <= 3:  # Likely BCHW
                patches = [p for p in patches]
            else:  # Likely BHWC
                patches = [p for p in patches]
        else:  # Single array
            patches = [patches]
    
    # Limit number of patches
    num_patches = min(len(patches), max_patches)
    patches = patches[:num_patches]
    
    # Get labels and predictions if provided
    if labels is not None:
        labels = labels[:num_patches]
    if predictions is not None:
        predictions = predictions[:num_patches]
    
    # Calculate rows needed
    rows = math.ceil(num_patches / cols)
    
    # Create figsize if not provided
    if figsize is None:
        figsize = (3 * cols, 3 * rows)
    
    # Create figure
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    fig.suptitle(title, fontsize=16)
    
    # Ensure axes is iterable for single subplot case
    if rows == 1 and cols == 1:
        axes = np.array([axes])
    elif rows == 1 or cols == 1:
        axes = axes.flatten()
    
    # Plot each patch
    for i in range(rows * cols):
        ax = axes.flatten()[i] if rows > 1 or cols > 1 else axes[i]
        
        if i < num_patches:
            patch = patches[i]
            
            # Convert torch tensor to numpy if needed
            if isinstance(patch, torch.Tensor):
                patch = patch.cpu().numpy()
            
            # Handle different formats and denormalize if needed
            if patch.ndim == 3:
                if patch.shape[0] <= 3:  # Likely CHW format
                    patch = np.transpose(patch, (1, 2, 0))
                
                if patch.shape[2] == 1:  # Grayscale
                    patch = patch[:, :, 0]
            
            # Denormalize if requested
            if denormalize and patch.max() <= 1.0:
                patch = (patch * 255).astype(np.uint8)
            
            # Show patch
            if patch.ndim == 2 or (patch.ndim == 3 and patch.shape[2] == 1):
                # Grayscale
                ax.imshow(patch, cmap=cmap or 'gray')
            else:
                # RGB
                ax.imshow(patch)
            
            # Add label and/or prediction
            if labels is not None and predictions is not None:
                label = str(labels[i])
                pred = str(predictions[i])
                ax.set_title(f"Label: {label}\nPred: {pred}")
            elif labels is not None:
                label = str(labels[i])
                ax.set_title(f"Label: {label}")
            elif predictions is not None:
                pred = str(predictions[i])
                ax.set_title(f"Pred: {pred}")
        
        # Turn off axis
        ax.axis('off')
    
    # Adjust layout
    plt.tight_layout()
    if title:
        plt.subplots_adjust(top=0.9)
    
    # Save figure if requested
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    # Show or return figure
    if show:
        plt.show()
        return None
    else:
        return fig


def visualize_reconstructions(
    inputs: Union[List[np.ndarray], np.ndarray, torch.Tensor],
    reconstructions: Union[List[np.ndarray], np.ndarray, torch.Tensor],
    max_images: int = 8,
    figsize: Tuple[int, int] = None,
    title: str = "Original vs. Reconstruction",
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True,
    denormalize: bool = True,
) -> Optional[plt.Figure]:
    """
    Visualize original images alongside their reconstructions.
    
    Args:
        inputs: Original input images
        reconstructions: Reconstructed images
        max_images: Maximum number of image pairs to display
        figsize: Figure size (width, height) in inches
        title: Title for the plot
        save_path: Path to save the visualization
        show: Whether to display the plot
        denormalize: Whether to denormalize image values from [0, 1] to [0, 255]
        
    Returns:
        Matplotlib figure if show=False, otherwise None
    """
    # Convert inputs and reconstructions to lists of numpy arrays
    if isinstance(inputs, torch.Tensor):
        inputs = [i.cpu().numpy() for i in inputs]
    elif isinstance(inputs, np.ndarray) and inputs.ndim == 4:
        inputs = [i for i in inputs]
    elif not isinstance(inputs, list):
        inputs = [inputs]
    
    if isinstance(reconstructions, torch.Tensor):
        reconstructions = [r.cpu().numpy() for r in reconstructions]
    elif isinstance(reconstructions, np.ndarray) and reconstructions.ndim == 4:
        reconstructions = [r for r in reconstructions]
    elif not isinstance(reconstructions, list):
        reconstructions = [reconstructions]
    
    # Ensure equal length
    assert len(inputs) == len(reconstructions), "Inputs and reconstructions must have the same length"
    
    # Limit number of images
    num_images = min(len(inputs), max_images)
    inputs = inputs[:num_images]
    reconstructions = reconstructions[:num_images]
    
    # Create figsize if not provided
    if figsize is None:
        figsize = (3 * num_images, 6)
    
    # Create figure with 2 rows (original and reconstruction)
    fig, axes = plt.subplots(2, num_images, figsize=figsize)
    fig.suptitle(title, fontsize=16)
    
    # Ensure axes is 2D for single image case
    if num_images == 1:
        axes = axes.reshape(2, 1)
    
    # Plot each image pair
    for i in range(num_images):
        for j, (img, row_title) in enumerate(zip([inputs[i], reconstructions[i]], ["Input", "Reconstruction"])):
            ax = axes[j, i]
            
            # Convert torch tensor to numpy if needed
            if isinstance(img, torch.Tensor):
                img = img.cpu().numpy()
            
            # Handle different formats and denormalize if needed
            if img.ndim == 3:
                if img.shape[0] <= 3:  # Likely CHW format
                    img = np.transpose(img, (1, 2, 0))
                
                if img.shape[2] == 1:  # Grayscale
                    img = img[:, :, 0]
            
            # Denormalize if requested
            if denormalize and img.max() <= 1.0:
                img = (img * 255).astype(np.uint8)
            
            # Show image
            if img.ndim == 2 or (img.ndim == 3 and img.shape[2] == 1):
                # Grayscale
                ax.imshow(img, cmap='gray')
            else:
                # RGB
                ax.imshow(img)
            
            # Add row label for first column
            if i == 0:
                ax.set_ylabel(row_title, fontsize=12)
            
            # Turn off axis
            ax.axis('off')
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    
    # Save figure if requested
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    # Show or return figure
    if show:
        plt.show()
        return None
    else:
        return fig


def visualize_attention_heatmap(
    attention: Union[np.ndarray, torch.Tensor],
    image: Optional[Union[np.ndarray, torch.Tensor, Image.Image]] = None,
    figsize: Tuple[int, int] = (10, 8),
    alpha: float = 0.5,
    colormap: str = 'jet',
    title: str = "Attention Heatmap",
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True,
) -> Optional[plt.Figure]:
    """
    Visualize attention weights as a heatmap, optionally overlaid on an image.
    
    Args:
        attention: Attention weights as numpy array or torch tensor
        image: Optional background image to overlay heatmap on
        figsize: Figure size (width, height) in inches
        alpha: Transparency of heatmap overlay (0 = transparent, 1 = opaque)
        colormap: Colormap for heatmap
        title: Title for the plot
        save_path: Path to save the visualization
        show: Whether to display the plot
        
    Returns:
        Matplotlib figure if show=False, otherwise None
    """
    # Convert attention to numpy if needed
    if isinstance(attention, torch.Tensor):
        attention = attention.cpu().numpy()
    
    # Squeeze singleton dimensions
    attention = np.squeeze(attention)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    if image is not None:
        # Convert image to numpy if needed
        if isinstance(image, torch.Tensor):
            image = image.cpu().numpy()
        elif isinstance(image, Image.Image):
            image = np.array(image)
        
        # Handle different formats
        if image.ndim == 3:
            if image.shape[0] <= 3:  # Likely CHW format
                image = np.transpose(image, (1, 2, 0))
            
            if image.shape[2] == 1:  # Grayscale
                image = image[:, :, 0]
        
        # Normalize to [0, 1] if needed
        if image.max() > 1.0:
            image = image / 255.0
        
        # Display image
        ax.imshow(image)
        
        # If attention doesn't match image dimensions, resize it
        if attention.shape != image.shape[:2]:
            attention_resized = cv2.resize(
                attention, 
                (image.shape[1], image.shape[0]),
                interpolation=cv2.INTER_LINEAR
            )
        else:
            attention_resized = attention
        
        # Normalize attention
        if attention_resized.min() < 0 or attention_resized.max() > 1:
            attention_resized = (attention_resized - attention_resized.min()) / (attention_resized.max() - attention_resized.min())
        
        # Create heatmap overlay
        cmap = plt.get_cmap(colormap)
        heatmap = cmap(attention_resized)
        heatmap[..., 3] = alpha * attention_resized  # Set alpha channel
        
        # Display heatmap overlay
        ax.imshow(heatmap)
    else:
        # Just display heatmap
        im = ax.imshow(attention, cmap=colormap)
        plt.colorbar(im, ax=ax)
    
    # Add title
    ax.set_title(title, fontsize=14)
    
    # Turn off axis
    ax.axis('off')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure if requested
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    # Show or return figure
    if show:
        plt.show()
        return None
    else:
        return fig


def visualize_wsi_attention(
    wsi_path: Union[str, Path],
    attention_weights: Union[np.ndarray, torch.Tensor],
    coordinates: Union[np.ndarray, torch.Tensor],
    patch_size: int,
    level: int = 0,
    alpha: float = 0.5,
    colormap: str = 'jet',
    title: str = "WSI Attention Heatmap",
    figsize: Tuple[int, int] = (12, 10),
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True,
    threshold: Optional[float] = None,
    thumbnail_size: int = 2000,
) -> Optional[plt.Figure]:
    """
    Visualize attention weights on a WSI thumbnail.
    
    Args:
        wsi_path: Path to the WSI
        attention_weights: Attention weights for each patch
        coordinates: Coordinates of each patch as [x, y, width, height] or [x, y]
        patch_size: Size of patches in pixels
        level: Pyramid level for the thumbnail
        alpha: Transparency of heatmap overlay
        colormap: Colormap for heatmap
        title: Title for the plot
        figsize: Figure size (width, height) in inches
        save_path: Path to save the visualization
        show: Whether to display the plot
        threshold: Optional threshold for filtering attention weights
        thumbnail_size: Maximum size of thumbnail in pixels
        
    Returns:
        Matplotlib figure if show=False, otherwise None
    """
    # Convert to numpy if needed
    if isinstance(attention_weights, torch.Tensor):
        attention_weights = attention_weights.cpu().numpy()
    if isinstance(coordinates, torch.Tensor):
        coordinates = coordinates.cpu().numpy()
    
    # Squeeze singleton dimensions
    attention_weights = np.squeeze(attention_weights)
    
    # Threshold attention weights if requested
    if threshold is not None:
        attention_mask = attention_weights >= threshold
        attention_weights = attention_weights * attention_mask
    
    # Normalize attention weights to [0, 1]
    if attention_weights.min() < 0 or attention_weights.max() > 1:
        attention_weights = (attention_weights - attention_weights.min()) / (attention_weights.max() - attention_weights.min() + 1e-8)
    
    # Open WSI
    wsi = WSIReader.open(wsi_path)
    
    # Get dimensions
    wsi_dims = wsi.slide_dimensions(level=level)
    wsi_w, wsi_h = wsi_dims
    
    # Calculate downsampling for thumbnail
    max_dim = max(wsi_w, wsi_h)
    if max_dim > thumbnail_size:
        thumb_scale = thumbnail_size / max_dim
        thumb_w = int(wsi_w * thumb_scale)
        thumb_h = int(wsi_h * thumb_scale)
    else:
        thumb_w, thumb_h = wsi_w, wsi_h
        thumb_scale = 1.0
    
    # Read thumbnail
    thumbnail = wsi.read_region(
        (0, 0), level, (wsi_w, wsi_h)
    )
    
    # Resize thumbnail
    thumbnail = cv2.resize(thumbnail, (thumb_w, thumb_h))
    
    # Create empty heatmap
    heatmap = np.zeros((thumb_h, thumb_w), dtype=np.float32)
    
    # Fill heatmap with attention weights
    for i, (coord, weight) in enumerate(zip(coordinates, attention_weights)):
        # Extract coordinates (handle both [x, y, w, h] and [x, y] formats)
        if len(coord) >= 4:  # [x, y, w, h] format
            x, y, w, h = coord[:4]
        else:  # [x, y] format
            x, y = coord[:2]
            w = h = patch_size
        
        # Scale coordinates for thumbnail
        x_scaled = int(x * thumb_scale)
        y_scaled = int(y * thumb_scale)
        w_scaled = max(1, int(w * thumb_scale))
        h_scaled = max(1, int(h * thumb_scale))
        
        # Ensure within bounds
        if (x_scaled >= 0 and y_scaled >= 0 and 
            x_scaled < thumb_w and y_scaled < thumb_h):
            
            # Create patch mask
            patch_mask = np.zeros((thumb_h, thumb_w), dtype=np.float32)
            end_x = min(x_scaled + w_scaled, thumb_w)
            end_y = min(y_scaled + h_scaled, thumb_h)
            patch_mask[y_scaled:end_y, x_scaled:end_x] = weight
            
            # Add to heatmap
            heatmap += patch_mask
    
    # Normalize heatmap
    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Display thumbnail
    ax.imshow(thumbnail)
    
    # Create colormap with transparency
    cmap = plt.get_cmap(colormap)
    heatmap_rgba = cmap(heatmap)
    heatmap_rgba[..., 3] = alpha * heatmap  # Set alpha channel
    
    # Display heatmap overlay
    ax.imshow(heatmap_rgba)
    
    # Add title
    ax.set_title(title, fontsize=14)
    
    # Turn off axis
    ax.axis('off')
    
    # Add colorbar
    norm = plt.Normalize(0, 1)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label='Attention Weight')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure if requested
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    # Show or return figure
    if show:
        plt.show()
        return None
    else:
        return fig


def visualize_patch_locations(
    wsi_path: Union[str, Path],
    coordinates: Union[np.ndarray, List[List[int]]],
    patch_size: Optional[int] = None,
    level: int = 0,
    figsize: Tuple[int, int] = (12, 10),
    colors: Optional[List[str]] = None,
    labels: Optional[List[Any]] = None,
    title: str = "Patch Locations",
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True,
    max_patches: Optional[int] = None,
    thumbnail_size: int = 2000,
) -> Optional[plt.Figure]:
    """
    Visualize patch locations on a WSI thumbnail.
    
    Args:
        wsi_path: Path to the WSI
        coordinates: Coordinates of each patch as [x, y, width, height] or [x, y]
        patch_size: Size of patches in pixels (required if coordinates are [x, y])
        level: Pyramid level for the thumbnail
        figsize: Figure size (width, height) in inches
        colors: Optional list of colors for each patch or group
        labels: Optional list of labels for legend
        title: Title for the plot
        save_path: Path to save the visualization
        show: Whether to display the plot
        max_patches: Maximum number of patches to visualize
        thumbnail_size: Maximum size of thumbnail in pixels
        
    Returns:
        Matplotlib figure if show=False, otherwise None
    """
    # Convert to numpy if needed
    if isinstance(coordinates, torch.Tensor):
        coordinates = coordinates.cpu().numpy()
    
    # Limit number of patches if requested
    if max_patches is not None and len(coordinates) > max_patches:
        indices = np.random.choice(len(coordinates), max_patches, replace=False)
        coordinates = coordinates[indices]
        if colors is not None:
            colors = [colors[i] for i in indices]
        if labels is not None:
            labels = [labels[i] for i in indices]
    
    # Open WSI
    wsi = WSIReader.open(wsi_path)
    
    # Get dimensions
    wsi_dims = wsi.slide_dimensions(level=level)
    wsi_w, wsi_h = wsi_dims
    
    # Calculate downsampling for thumbnail
    max_dim = max(wsi_w, wsi_h)
    if max_dim > thumbnail_size:
        thumb_scale = thumbnail_size / max_dim
        thumb_w = int(wsi_w * thumb_scale)
        thumb_h = int(wsi_h * thumb_scale)
    else:
        thumb_w, thumb_h = wsi_w, wsi_h
        thumb_scale = 1.0
    
    # Read thumbnail
    thumbnail = wsi.read_region(
        (0, 0), level, (wsi_w, wsi_h)
    )
    
    # Resize thumbnail
    thumbnail = cv2.resize(thumbnail, (thumb_w, thumb_h))
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Display thumbnail
    ax.imshow(thumbnail)
    
    # Default colors if not provided
    if colors is None:
        colors = ['r']
    
    # Ensure colors is a list
    if not isinstance(colors, list):
        colors = [colors]
    
    # Create unique label mapping if needed
    unique_labels = None
    label_colors = None
    if labels is not None:
        unique_labels = sorted(set(labels))
        label_colors = {label: colors[i % len(colors)] for i, label in enumerate(unique_labels)}
    
    # Draw patch rectangles
    for i, coord in enumerate(coordinates):
        # Extract coordinates (handle both [x, y, w, h] and [x, y] formats)
        if len(coord) >= 4:  # [x, y, w, h] format
            x, y, w, h = coord[:4]
        else:  # [x, y] format
            if patch_size is None:
                raise ValueError("patch_size must be provided if coordinates are [x, y]")
            x, y = coord[:2]
            w = h = patch_size
        
        # Scale coordinates for thumbnail
        x_scaled = int(x * thumb_scale)
        y_scaled = int(y * thumb_scale)
        w_scaled = max(1, int(w * thumb_scale))
        h_scaled = max(1, int(h * thumb_scale))
        
        # Determine color
        if labels is not None:
            color = label_colors[labels[i]]
        else:
            color = colors[i % len(colors)]
        
        # Draw rectangle
        rect = patches.Rectangle(
            (x_scaled, y_scaled),
            w_scaled,
            h_scaled,
            linewidth=1,
            edgecolor=color,
            facecolor='none'
        )
        ax.add_patch(rect)
    
    # Add legend if labels provided
    if unique_labels is not None:
        legend_elements = [
            patches.Patch(facecolor='none', edgecolor=label_colors[label], label=label)
            for label in unique_labels
        ]
        ax.legend(handles=legend_elements, loc='upper right')
    
    # Add title
    ax.set_title(title, fontsize=14)
    
    # Turn off axis
    ax.axis('off')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure if requested
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    # Show or return figure
    if show:
        plt.show()
        return None
    else:
        return fig


def visualize_training_metrics(
    metrics: Dict[str, List[float]],
    figsize: Tuple[int, int] = (12, 8),
    title: str = "Training Metrics",
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True,
    smooth: int = 1,
) -> Optional[plt.Figure]:
    """
    Visualize training metrics.
    
    Args:
        metrics: Dictionary of metric names and values
        figsize: Figure size (width, height) in inches
        title: Title for the plot
        save_path: Path to save the visualization
        show: Whether to display the plot
        smooth: Window size for smoothing (1 = no smoothing)
        
    Returns:
        Matplotlib figure if show=False, otherwise None
    """
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot each metric
    for name, values in metrics.items():
        # Smooth values if requested
        if smooth > 1 and len(values) > smooth:
            weights = np.ones(smooth) / smooth
            values_smooth = np.convolve(values, weights, mode='valid')
            # Plot original values with low alpha
            ax.plot(values, alpha=0.3, label=f"{name} (raw)")
            # Plot smoothed values
            ax.plot(np.arange(smooth-1, len(values)), values_smooth, label=f"{name} (smoothed)")
        else:
            # Plot raw values
            ax.plot(values, label=name)
    
    # Add labels and legend
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Value", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend()
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure if requested
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    # Show or return figure
    if show:
        plt.show()
        return None
    else:
        return fig


def visualize_confusion_matrix(
    confusion_matrix: np.ndarray,
    class_names: List[str],
    figsize: Tuple[int, int] = (10, 8),
    title: str = "Confusion Matrix",
    normalize: bool = True,
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True,
    colormap: str = "Blues",
) -> Optional[plt.Figure]:
    """
    Visualize a confusion matrix.
    
    Args:
        confusion_matrix: Confusion matrix as numpy array
        class_names: Names of the classes
        figsize: Figure size (width, height) in inches
        title: Title for the plot
        normalize: Whether to normalize by row (true class)
        save_path: Path to save the visualization
        show: Whether to display the plot
        colormap: Colormap for the confusion matrix
        
    Returns:
        Matplotlib figure if show=False, otherwise None
    """
    # Normalize if requested
    if normalize:
        cm = confusion_matrix.astype('float') / (confusion_matrix.sum(axis=1, keepdims=True) + 1e-8)
    else:
        cm = confusion_matrix
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot confusion matrix
    im = ax.imshow(cm, interpolation='nearest', cmap=colormap)
    plt.colorbar(im, ax=ax)
    
    # Add axis labels
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    
    # Add title and axis labels
    ax.set_title(title, fontsize=14)
    ax.set_xlabel('Predicted label', fontsize=12)
    ax.set_ylabel('True label', fontsize=12)
    
    # Add text annotations
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure if requested
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    # Show or return figure
    if show:
        plt.show()
        return None
    else:
        return fig


def generate_experiment_report(
    experiment_dir: Union[str, Path],
    metrics: Dict[str, List[float]],
    example_patches: Optional[List[np.ndarray]] = None,
    example_reconstructions: Optional[Tuple[List[np.ndarray], List[np.ndarray]]] = None,
    confusion_matrix: Optional[np.ndarray] = None,
    class_names: Optional[List[str]] = None,
    attention_heatmap: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    description: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Path:
    """
    Generate a comprehensive experiment report with visualizations.
    
    Args:
        experiment_dir: Directory to save the report
        metrics: Dictionary of training metrics
        example_patches: Optional list of example patches to visualize
        example_reconstructions: Optional tuple of (inputs, reconstructions)
        confusion_matrix: Optional confusion matrix
        class_names: Optional class names for confusion matrix
        attention_heatmap: Optional tuple of (attention, image) for heatmap
        description: Optional text description of the experiment
        config: Optional configuration dictionary for the experiment
        
    Returns:
        Path to the generated report
    """
    experiment_dir = Path(experiment_dir)
    os.makedirs(experiment_dir, exist_ok=True)
    
    # Create HTML report
    html_content = [
        "<!DOCTYPE html>",
        "<html>",
        "<head>",
        f"<title>Experiment Report</title>",
        "<style>",
        "body { font-family: Arial, sans-serif; margin: 20px; }",
        "h1, h2, h3 { color: #333; }",
        ".section { margin-bottom: 30px; }",
        ".figure { margin-bottom: 20px; text-align: center; }",
        ".figure img { max-width: 100%; border: 1px solid #ddd; }",
        ".metrics { display: flex; flex-wrap: wrap; }",
        ".config { font-family: monospace; white-space: pre; background-color: #f5f5f5; padding: 10px; border-radius: 5px; overflow-x: auto; }",
        "</style>",
        "</head>",
        "<body>",
        f"<h1>Experiment Report</h1>",
    ]
    
    # Add description
    if description:
        html_content.extend([
            "<div class='section'>",
            "<h2>Description</h2>",
            f"<p>{description}</p>",
            "</div>"
        ])
    
    # Add configuration
    if config:
        import json
        html_content.extend([
            "<div class='section'>",
            "<h2>Configuration</h2>",
            "<div class='config'>",
            json.dumps(config, indent=2),
            "</div>",
            "</div>"
        ])
    
    # Add metrics
    html_content.extend([
        "<div class='section'>",
        "<h2>Training Metrics</h2>",
        "<div class='figure'>",
    ])
    
    # Create and save metrics figure
    metrics_fig = visualize_training_metrics(
        metrics, 
        title="Training Metrics",
        show=False
    )
    metrics_path = experiment_dir / "metrics.png"
    metrics_fig.savefig(metrics_path, bbox_inches='tight', dpi=150)
    plt.close(metrics_fig)
    
    html_content.extend([
        f"<img src='{metrics_path.name}' alt='Training Metrics'>",
        "</div>",
        "</div>"
    ])
    
    # Add example patches if provided
    if example_patches is not None:
        html_content.extend([
            "<div class='section'>",
            "<h2>Example Patches</h2>",
            "<div class='figure'>",
        ])
        
        patches_fig = visualize_patches(
            example_patches, 
            title="Example Patches",
            show=False
        )
        patches_path = experiment_dir / "example_patches.png"
        patches_fig.savefig(patches_path, bbox_inches='tight', dpi=150)
        plt.close(patches_fig)
        
        html_content.extend([
            f"<img src='{patches_path.name}' alt='Example Patches'>",
            "</div>",
            "</div>"
        ])
    
    # Add reconstructions if provided
    if example_reconstructions is not None:
        inputs, reconstructions = example_reconstructions
        html_content.extend([
            "<div class='section'>",
            "<h2>Reconstructions</h2>",
            "<div class='figure'>",
        ])
        
        recon_fig = visualize_reconstructions(
            inputs,
            reconstructions,
            title="Original vs. Reconstruction",
            show=False
        )
        recon_path = experiment_dir / "reconstructions.png"
        recon_fig.savefig(recon_path, bbox_inches='tight', dpi=150)
        plt.close(recon_fig)
        
        html_content.extend([
            f"<img src='{recon_path.name}' alt='Reconstructions'>",
            "</div>",
            "</div>"
        ])
    
    # Add confusion matrix if provided
    if confusion_matrix is not None and class_names is not None:
        html_content.extend([
            "<div class='section'>",
            "<h2>Confusion Matrix</h2>",
            "<div class='figure'>",
        ])
        
        cm_fig = visualize_confusion_matrix(
            confusion_matrix,
            class_names,
            title="Confusion Matrix",
            show=False
        )
        cm_path = experiment_dir / "confusion_matrix.png"
        cm_fig.savefig(cm_path, bbox_inches='tight', dpi=150)
        plt.close(cm_fig)
        
        html_content.extend([
            f"<img src='{cm_path.name}' alt='Confusion Matrix'>",
            "</div>",
            "</div>"
        ])
    
    # Add attention heatmap if provided
    if attention_heatmap is not None:
        attention, image = attention_heatmap
        html_content.extend([
            "<div class='section'>",
            "<h2>Attention Heatmap</h2>",
            "<div class='figure'>",
        ])
        
        heatmap_fig = visualize_attention_heatmap(
            attention,
            image,
            title="Attention Heatmap",
            show=False
        )
        heatmap_path = experiment_dir / "attention_heatmap.png"
        heatmap_fig.savefig(heatmap_path, bbox_inches='tight', dpi=150)
        plt.close(heatmap_fig)
        
        html_content.extend([
            f"<img src='{heatmap_path.name}' alt='Attention Heatmap'>",
            "</div>",
            "</div>"
        ])
    
    # Finish HTML
    html_content.extend([
        "</body>",
        "</html>"
    ])
    
    # Write HTML to file
    report_path = experiment_dir / "report.html"
    with open(report_path, 'w') as f:
        f.write('\n'.join(html_content))
    
    logger.info(f"Generated experiment report at {report_path}")
    
    return report_path


def create_wsi_thumbnail(
    wsi_path: Union[str, Path],
    output_path: Union[str, Path],
    max_size: int = 2000,
    level: Optional[int] = None,
) -> Path:
    """
    Create a thumbnail for a WSI.
    
    Args:
        wsi_path: Path to the WSI
        output_path: Path to save the thumbnail
        max_size: Maximum dimension of the thumbnail in pixels
        level: Pyramid level to extract thumbnail from (None = automatic)
        
    Returns:
        Path to the saved thumbnail
    """
    # Open WSI
    wsi = WSIReader.open(wsi_path)
    
    # Determine level if not specified
    if level is None:
        # Get dimensions of level 0
        dims_0 = wsi.slide_dimensions(level=0)
        max_dim_0 = max(dims_0)
        
        # Calculate optimal level
        target_downsample = max_dim_0 / max_size
        
        # Find closest level
        level_downsamples = wsi.info.level_downsamples
        level = min(range(len(level_downsamples)), 
                   key=lambda i: abs(level_downsamples[i] - target_downsample))
    
    # Get dimensions
    dims = wsi.slide_dimensions(level=level)
    width, height = dims
    
    # Calculate downsampling for thumbnail
    max_dim = max(width, height)
    if max_dim > max_size:
        scale = max_size / max_dim
        new_width = int(width * scale)
        new_height = int(height * scale)
    else:
        new_width, new_height = width, height
    
    # Read thumbnail
    thumbnail = wsi.read_region(
        (0, 0), level, (width, height)
    )
    
    # Resize if needed
    if new_width != width or new_height != height:
        thumbnail = cv2.resize(thumbnail, (new_width, new_height))
    
    # Ensure thumbnail is RGB
    if thumbnail.shape[-1] == 4:  # RGBA
        thumbnail = thumbnail[:, :, :3]  # Remove alpha channel
    
    # Save thumbnail
    output_path = Path(output_path)
    os.makedirs(output_path.parent, exist_ok=True)
    Image.fromarray(thumbnail).save(output_path)
    
    return output_path
