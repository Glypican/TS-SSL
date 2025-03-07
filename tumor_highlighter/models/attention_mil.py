"""
Multiple Instance Learning (MIL) with attention mechanism for tumor highlighting.

This module implements the attention-based MIL approach for highlighting tumor
regions in whole slide images using deep features extracted from patches.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, List, Union
import numpy as np


class AttentionMIL(nn.Module):
    """
    Attention-based Multiple Instance Learning for tumor highlighting.
    
    This implementation follows the approach described in "Attention-based Deep Multiple
    Instance Learning" by Ilse et al., using a two-layer attention mechanism to
    identify relevant instances (patches) in a bag (WSI).
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        output_dim: int = 2,
        dropout: float = 0.25,
    ):
        """
        Initialize the attention-based MIL model.
        
        Args:
            input_dim: Dimension of input feature vectors
            hidden_dim: Dimension of hidden attention representation
            output_dim: Dimension of output (number of classes)
            dropout: Dropout probability
        """
        super().__init__()
        
        # Attention mechanism: two-layer neural network
        # First attention layer transforms to hidden dimension
        self.attention_a = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout)
        )
        
        # Second attention layer transforms to scalar importance score
        self.attention_b = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        # Classifier that operates on the attention-weighted features
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # Attention weights for visualization
        self.attention_weights = None
    
    def forward(self, x: torch.Tensor, return_attention: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass of the attention-based MIL model.
        
        Args:
            x: Input tensor of shape (batch_size, num_instances, feature_dim)
            return_attention: Whether to return attention weights
            
        Returns:
            model output of shape (batch_size, output_dim)
            or tuple of (output, attention_weights) if return_attention=True
        """
        # Calculate attention scores and normalize
        a = self.attention_a(x)  # shape: (batch_size, num_instances, hidden_dim)
        e = self.attention_b(a)  # shape: (batch_size, num_instances, 1)
        
        # Apply softmax to get attention weights
        alpha = F.softmax(e, dim=1)  # shape: (batch_size, num_instances, 1)
        self.attention_weights = alpha
        
        # Apply attention weights to input
        x_weighted = torch.sum(alpha * x, dim=1)  # shape: (batch_size, feature_dim)
        
        # Apply classifier to weighted representation
        output = self.classifier(x_weighted)  # shape: (batch_size, output_dim)
        
        if return_attention:
            return output, alpha
        
        return output
    
    def get_attention_weights(self) -> torch.Tensor:
        """
        Get the attention weights from the last forward pass.
        
        Returns:
            Attention weights tensor
        """
        if self.attention_weights is None:
            raise RuntimeError("No attention weights available. Run forward pass first.")
            
        return self.attention_weights


class GatedAttentionMIL(nn.Module):
    """
    Gated Attention MIL model for tumor highlighting.
    
    This implementation follows the approach described in "Attention-based Deep Multiple
    Instance Learning" by Ilse et al., using a gated attention mechanism instead of
    the simple two-layer attention.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        output_dim: int = 2,
        dropout: float = 0.25,
    ):
        """
        Initialize the gated attention MIL model.
        
        Args:
            input_dim: Dimension of input feature vectors
            hidden_dim: Dimension of hidden attention representation
            output_dim: Dimension of output (number of classes)
            dropout: Dropout probability
        """
        super().__init__()
        
        # Attention mechanism: gated attention
        # Attention U - contextual information
        self.attention_U = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh()
        )
        
        # Attention V - gating mechanism
        self.attention_V = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Sigmoid()
        )
        
        # Attention weights
        self.attention_weights = nn.Linear(hidden_dim, 1)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # Store attention scores
        self.alphas = None
    
    def forward(self, x: torch.Tensor, return_attention: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass of the gated attention MIL model.
        
        Args:
            x: Input tensor of shape (batch_size, num_instances, feature_dim)
            return_attention: Whether to return attention weights
            
        Returns:
            model output of shape (batch_size, output_dim)
            or tuple of (output, attention_weights) if return_attention=True
        """
        # Calculate attention scores using gated mechanism
        a_U = self.attention_U(x)  # shape: (batch_size, num_instances, hidden_dim)
        a_V = self.attention_V(x)  # shape: (batch_size, num_instances, hidden_dim)
        
        # Element-wise multiplication for gating
        gated = a_U * a_V  # shape: (batch_size, num_instances, hidden_dim)
        
        # Calculate attention scores
        e = self.attention_weights(gated)  # shape: (batch_size, num_instances, 1)
        
        # Apply softmax to get attention weights
        alpha = F.softmax(e, dim=1)  # shape: (batch_size, num_instances, 1)
        self.alphas = alpha
        
        # Apply attention weights to input
        x_weighted = torch.sum(alpha * x, dim=1)  # shape: (batch_size, feature_dim)
        
        # Apply classifier to weighted representation
        output = self.classifier(x_weighted)  # shape: (batch_size, output_dim)
        
        if return_attention:
            return output, alpha
        
        return output
    
    def get_attention_weights(self) -> torch.Tensor:
        """
        Get the attention weights from the last forward pass.
        
        Returns:
            Attention weights tensor
        """
        if self.alphas is None:
            raise RuntimeError("No attention weights available. Run forward pass first.")
            
        return self.alphas


class TumorHighlighter:
    """
    Tumor highlighter using Multiple Instance Learning with attention.
    
    This class provides utilities for highlighting tumor regions in whole slide
    images using attention weights from MIL models.
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize the tumor highlighter.
        
        Args:
            model: Trained MIL attention model
            device: Device to run the model on ('cuda' or 'cpu')
        """
        self.model = model.to(device)
        self.model.eval()
        self.device = device
    
    def highlight_tumor(
        self,
        features: torch.Tensor,
        coordinates: torch.Tensor,
        thumbnail: np.ndarray,
        downsample_factor: float,
        threshold: Optional[float] = None,
        kernel_size: int = 3,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate a tumor heatmap for a WSI.
        
        Args:
            features: Feature tensor of shape (num_patches, feature_dim)
            coordinates: Coordinates tensor of shape (num_patches, 4) with [x1, y1, x2, y2]
            thumbnail: Thumbnail image of the WSI
            downsample_factor: Downsample factor between original WSI and thumbnail
            threshold: Threshold for binarizing attention weights (None = no thresholding)
            kernel_size: Size of the Gaussian kernel for smoothing
            
        Returns:
            Tuple of (heatmap, blended_image) with tumor highlighting
        """
        # Add batch dimension to features and get attention weights
        features_batch = features.unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            _, attention = self.model(features_batch, return_attention=True)
        
        # Remove batch dimension
        attention = attention.squeeze(0).squeeze(-1).cpu().numpy()
        
        # Create empty heatmap with same dimensions as thumbnail
        h, w = thumbnail.shape[:2]
        heatmap = np.zeros((h, w), dtype=np.float32)
        
        # Scale coordinates to thumbnail dimensions
        scaled_coords = coordinates.cpu().numpy() / downsample_factor
        scaled_coords = scaled_coords.astype(np.int32)
        
        # Fill heatmap with attention values
        for i, (x1, y1, x2, y2) in enumerate(scaled_coords):
            # Ensure coordinates are within thumbnail dimensions
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            if x2 > x1 and y2 > y1:
                heatmap[y1:y2, x1:x2] = attention[i]
        
        # Smooth heatmap with Gaussian filter
        import cv2
        heatmap = cv2.GaussianBlur(heatmap, (kernel_size, kernel_size), 0)
        
        # Normalize heatmap
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        
        # Apply threshold if specified
        if threshold is not None:
            heatmap = (heatmap > threshold).astype(np.float32)
        
        # Create color heatmap
        heatmap_color = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
        
        # Blend with original thumbnail
        alpha = 0.5
        blended = cv2.addWeighted(thumbnail, 1 - alpha, heatmap_color, alpha, 0)
        
        return heatmap, blended
    
    def process_wsi(
        self,
        features: np.ndarray,
        coordinates: np.ndarray,
        thumbnail: np.ndarray,
        downsample_factor: float,
        threshold: Optional[float] = None,
        save_path: Optional[str] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Process a whole slide image and generate tumor heatmap.
        
        Args:
            features: Feature array of shape (num_patches, feature_dim)
            coordinates: Coordinates array of shape (num_patches, 4) with [x1, y1, x2, y2]
            thumbnail: Thumbnail image of the WSI
            downsample_factor: Downsample factor between original WSI and thumbnail
            threshold: Threshold for binarizing attention weights
            save_path: Path to save the results
            
        Returns:
            Dictionary with heatmap, blended image, and attention weights
        """
        # Convert numpy arrays to torch tensors
        features_tensor = torch.from_numpy(features).float()
        coordinates_tensor = torch.from_numpy(coordinates).float()
        
        # Generate heatmap
        heatmap, blended = self.highlight_tumor(
            features_tensor,
            coordinates_tensor,
            thumbnail,
            downsample_factor,
            threshold
        )
        
        # Get raw attention weights
        with torch.no_grad():
            _, attention = self.model(features_tensor.unsqueeze(0).to(self.device), return_attention=True)
        
        attention = attention.squeeze(0).squeeze(-1).cpu().numpy()
        
        # Save results if requested
        if save_path:
            import cv2
            import os
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            
            # Save heatmap
            cv2.imwrite(f"{save_path}_heatmap.png", (heatmap * 255).astype(np.uint8))
            
            # Save blended image
            cv2.imwrite(f"{save_path}_blended.png", blended)
            
            # Save attention weights
            np.save(f"{save_path}_attention.npy", attention)
        
        return {
            'heatmap': heatmap,
            'blended': blended,
            'attention': attention
        }
