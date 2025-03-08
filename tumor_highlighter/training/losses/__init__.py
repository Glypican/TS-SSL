"""
Loss functions for training models in the tumor_highlighter package.

This module provides different loss functions that can be used for training
the Task-Specific Self-Supervised Learning (TS-SSL) autoencoder and other models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Union, Callable

class ReconstructionLoss(nn.Module):
    """
    Reconstruction loss for autoencoders.
    
    This class provides a standardized interface for autoencoder 
    reconstruction loss, defaulting to MSESSIMLoss which combines
    pixel-wise accuracy with structural similarity.
    """
    
    def __init__(
        self,
        loss_type: str = 'mse_ssim',
        alpha: float = 0.5,
        **kwargs
    ):
        """
        Initialize reconstruction loss.
        
        Args:
            loss_type: Type of loss to use ('mse', 'l1', 'ssim', 'mse_ssim', etc.)
            alpha: Weight between losses if using a combined loss like 'mse_ssim'
            **kwargs: Additional parameters for the underlying loss function
        """
        super().__init__()
        
        # Use the existing loss registry and get_loss_function utility
        if loss_type == 'mse_ssim' and 'alpha' not in kwargs:
            kwargs['alpha'] = alpha
            
        self.loss_fn = get_loss_function(loss_type, **kwargs)
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Calculate reconstruction loss.
        
        Args:
            predictions: Predicted images from the autoencoder
            targets: Target images (typically the input images)
            
        Returns:
            Loss value
        """
        return self.loss_fn(predictions, targets)


class SSIMLoss(nn.Module):
    """
    Structural Similarity Index (SSIM) loss function.
    
    This loss measures the structural similarity between two images,
    which is more aligned with human perception than pixel-wise losses.
    """
    
    def __init__(
        self,
        window_size: int = 11,
        size_average: bool = True,
        val_range: Optional[float] = None,
    ):
        """
        Initialize the SSIM loss function.
        
        Args:
            window_size: Size of the gaussian window
            size_average: Whether to average over batch
            val_range: Value range of the input (max - min)
        """
        super().__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.val_range = val_range
        
        # Initialize gaussian window
        self.window = self._create_window(window_size)
    
    def _create_window(self, window_size: int) -> torch.Tensor:
        """
        Create a 2D gaussian window.
        
        Args:
            window_size: Size of the window
            
        Returns:
            Gaussian window tensor
        """
        # Create 1D gaussian
        sigma = 1.5
        gauss = torch.Tensor([np.exp(-(x - window_size//2)**2 / (2 * sigma**2)) 
                             for x in range(window_size)])
        
        # Normalize
        gauss = gauss / gauss.sum()
        
        # Create 2D gaussian
        _1D_window = gauss.unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(1, 1, window_size, window_size).contiguous()
        
        return window
    
    def _ssim(
        self,
        img1: torch.Tensor,
        img2: torch.Tensor,
        window: torch.Tensor,
        window_size: int,
        channel: int,
    ) -> torch.Tensor:
        """
        Calculate SSIM between two images.
        
        Args:
            img1: First image
            img2: Second image
            window: Gaussian window
            window_size: Size of the window
            channel: Number of channels
            
        Returns:
            SSIM value
        """
        # Constants to stabilize division
        C1 = (0.01 * self.val_range) ** 2
        C2 = (0.03 * self.val_range) ** 2
        
        # Expand window to match input channels
        window = window.expand(channel, 1, window_size, window_size)
        window = window.to(img1.device)
        
        # Calculate means
        mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)
        
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        
        # Calculate variances and covariance
        sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size//2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size//2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=window_size//2, groups=channel) - mu1_mu2
        
        # Calculate SSIM
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        
        # Average over spatial dimensions
        if self.size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)
    
    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        """
        Calculate 1 - SSIM (as a loss function).
        
        Args:
            img1: Predicted image
            img2: Target image
            
        Returns:
            1 - SSIM (to minimize)
        """
        # Check dimensions
        _, channel, _, _ = img1.size()
        
        # Set val_range if not provided
        if self.val_range is None:
            self.val_range = 1.0 if img1.max() <= 1 else 255.0
        
        # Calculate SSIM
        ssim_value = self._ssim(img1, img2, self.window, self.window_size, channel)
        
        # Return 1 - SSIM (to minimize)
        return 1 - ssim_value


class MSESSIMLoss(nn.Module):
    """
    Combined MSE and SSIM loss function.
    
    This loss combines pixel-wise Mean Squared Error (MSE) with
    structure-aware Structural Similarity Index (SSIM) loss.
    """
    
    def __init__(self, alpha: float = 0.5):
        """
        Initialize combined MSE and SSIM loss.
        
        Args:
            alpha: Weight for MSE loss (1-alpha for SSIM loss)
        """
        super().__init__()
        self.alpha = alpha
        self.mse_loss = nn.MSELoss()
        self.ssim_loss = SSIMLoss()
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Calculate combined MSE and SSIM loss.
        
        Args:
            pred: Predicted image
            target: Target image
            
        Returns:
            Combined loss
        """
        mse = self.mse_loss(pred, target)
        ssim = self.ssim_loss(pred, target)
        
        return self.alpha * mse + (1 - self.alpha) * ssim


class PerceptualLoss(nn.Module):
    """
    Perceptual loss using VGG features.
    
    This loss compares activations of a pretrained network (usually VGG)
    for predicted and target images, capturing perceptual differences.
    """
    
    def __init__(
        self,
        feature_extractor: nn.Module,
        feature_layers: list = [0, 1, 2, 3],
        layer_weights: Optional[list] = None,
        normalize: bool = True,
    ):
        """
        Initialize perceptual loss.
        
        Args:
            feature_extractor: Pretrained model for feature extraction
            feature_layers: List of layer indices to extract features from
            layer_weights: Weights for each layer (None = equal weights)
            normalize: Whether to normalize input images
        """
        super().__init__()
        self.feature_extractor = feature_extractor
        self.feature_layers = feature_layers
        self.normalize = normalize
        
        # Default to equal weights if not provided
        if layer_weights is None:
            self.layer_weights = [1.0] * len(feature_layers)
        else:
            assert len(layer_weights) == len(feature_layers), "Layer weights must match feature layers"
            self.layer_weights = layer_weights
        
        # Set feature extractor to eval mode
        self.feature_extractor.eval()
        
        # Normalization for ImageNet trained models
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
    
    def _normalize(self, img: torch.Tensor) -> torch.Tensor:
        """
        Normalize image to ImageNet range.
        
        Args:
            img: Input image (value range 0-1)
            
        Returns:
            Normalized image
        """
        return (img - self.mean) / self.std
    
    def _get_features(self, img: torch.Tensor) -> list:
        """
        Extract features from specified layers.
        
        Args:
            img: Input image
            
        Returns:
            List of feature maps
        """
        if self.normalize:
            img = self._normalize(img)
            
        features = []
        x = img
        
        # Extract features from specified layers
        for i, module in enumerate(self.feature_extractor.features):
            x = module(x)
            if i in self.feature_layers:
                features.append(x)
                
        return features
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Calculate perceptual loss.
        
        Args:
            pred: Predicted image
            target: Target image
            
        Returns:
            Perceptual loss value
        """
        # Check if inputs are in range [0, 1]
        if pred.max() > 1 or target.max() > 1:
            pred = pred / 255.0
            target = target / 255.0
        
        # Get features
        pred_features = self._get_features(pred)
        target_features = self._get_features(target)
        
        # Calculate loss for each layer
        loss = 0.0
        for i, (p_feat, t_feat) in enumerate(zip(pred_features, target_features)):
            loss += self.layer_weights[i] * F.mse_loss(p_feat, t_feat)
            
        return loss


class FocalLoss(nn.Module):
    """
    Focal Loss for binary classification.
    
    This loss function addresses class imbalance by down-weighting
    well-classified examples.
    """
    
    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = 'mean',
    ):
        """
        Initialize focal loss.
        
        Args:
            alpha: Weighting factor for positive class
            gamma: Focusing parameter (higher = more focus on hard examples)
            reduction: Reduction method ('mean', 'sum', or 'none')
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Calculate focal loss.
        
        Args:
            inputs: Predicted logits (before sigmoid)
            targets: Target binary labels (0 or 1)
            
        Returns:
            Focal loss value
        """
        # Apply sigmoid if not already done
        inputs = inputs.view(-1, 1)
        targets = targets.view(-1, 1)
        
        # Get probabilities
        prob = torch.sigmoid(inputs)
        
        # Calculate BCE loss per-sample
        pt = targets * prob + (1 - targets) * (1 - prob)
        bce_loss = -torch.log(pt + 1e-8)  # Add epsilon for stability
        
        # Apply focal weights
        alpha_t = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        focal_weights = alpha_t * (1 - pt) ** self.gamma
        
        # Apply weights to BCE loss
        focal_loss = focal_weights * bce_loss
        
        # Apply reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:  # 'none'
            return focal_loss


class DiceLoss(nn.Module):
    """
    Dice Loss for segmentation tasks.
    
    This loss function is based on the Dice coefficient, which measures
    the overlap between predicted and target segmentations.
    """
    
    def __init__(
        self,
        smooth: float = 1.0,
        reduction: str = 'mean',
    ):
        """
        Initialize Dice loss.
        
        Args:
            smooth: Smoothing factor to avoid division by zero
            reduction: Reduction method ('mean', 'sum', or 'none')
        """
        super().__init__()
        self.smooth = smooth
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Calculate Dice loss.
        
        Args:
            inputs: Predicted probabilities (after sigmoid/softmax)
            targets: Target binary masks
            
        Returns:
            Dice loss value
        """
        # Flatten inputs and targets
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        # Calculate Dice coefficient
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)
        
        # Return 1 - Dice (to minimize)
        loss = 1 - dice
        
        # Apply reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


# Registry of available loss functions
LOSS_REGISTRY = {
    'mse': nn.MSELoss,
    'l1': nn.L1Loss,
    'ssim': SSIMLoss,
    'mse_ssim': MSESSIMLoss,
    'focal': FocalLoss,
    'dice': DiceLoss,
    # 'perceptual': PerceptualLoss,  # Requires a feature extractor, registered separately
    'bce': nn.BCEWithLogitsLoss,
    'ce': nn.CrossEntropyLoss,
}


def get_loss_function(loss_name: str, **kwargs) -> nn.Module:
    """
    Get a loss function by name.
    
    Args:
        loss_name: Name of the loss function
        **kwargs: Additional arguments for the loss function
        
    Returns:
        Instantiated loss function
        
    Raises:
        ValueError: If loss function is not found in registry
    """
    if loss_name not in LOSS_REGISTRY:
        raise ValueError(f"Loss function '{loss_name}' not found. Available losses: {list(LOSS_REGISTRY.keys())}")
    
    return LOSS_REGISTRY[loss_name](**kwargs)
