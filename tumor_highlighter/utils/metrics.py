"""
Metrics utilities for evaluating model performance in tumor highlighting.

This module provides functions for calculating various metrics to assess
the performance of models for tumor highlighting, segmentation, and feature quality.
"""

import numpy as np
import torch
from typing import Dict, List, Union, Optional, Tuple, Any
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    roc_auc_score,
    confusion_matrix
)
from skimage.metrics import structural_similarity, peak_signal_noise_ratio


def calculate_metrics(
    predictions: Union[np.ndarray, torch.Tensor],
    targets: Union[np.ndarray, torch.Tensor],
    metric_types: List[str] = ["segmentation"],
    threshold: float = 0.5,
    class_names: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    Calculate various metrics for model evaluation.
    
    Args:
        predictions: Model predictions (probability maps, segmentation masks, etc.)
        targets: Ground truth labels or targets
        metric_types: Types of metrics to calculate ("segmentation", "reconstruction", "classification")
        threshold: Threshold for converting probability maps to binary masks (for segmentation)
        class_names: Optional list of class names for multi-class metrics
        
    Returns:
        Dictionary of metrics and their values
    """
    # Convert torch tensors to numpy if needed
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()
    
    metrics = {}
    
    # Calculate metrics based on requested types
    for metric_type in metric_types:
        if metric_type.lower() == "segmentation":
            segmentation_metrics = calculate_segmentation_metrics(
                predictions, targets, threshold
            )
            metrics.update(segmentation_metrics)
            
        elif metric_type.lower() == "reconstruction":
            reconstruction_metrics = calculate_reconstruction_metrics(
                predictions, targets
            )
            metrics.update(reconstruction_metrics)
            
        elif metric_type.lower() == "classification":
            classification_metrics = calculate_classification_metrics(
                predictions, targets, class_names
            )
            metrics.update(classification_metrics)
            
    return metrics


def calculate_segmentation_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Calculate metrics for segmentation tasks.
    
    Args:
        predictions: Predicted probability maps or binary masks
        targets: Ground truth binary masks
        threshold: Threshold for converting probability maps to binary masks
        
    Returns:
        Dictionary of segmentation metrics
    """
    # Convert probabilities to binary masks if needed
    if predictions.max() > 1.0 or predictions.min() < 0.0:
        # Normalize if not in [0,1] range
        predictions = (predictions - predictions.min()) / (predictions.max() - predictions.min() + 1e-8)
    
    if np.any((predictions > 0) & (predictions < 1)):
        binary_preds = (predictions > threshold).astype(np.uint8)
    else:
        binary_preds = predictions.astype(np.uint8)
    
    binary_targets = targets.astype(np.uint8)
    
    # Calculate intersection and union
    intersection = np.logical_and(binary_preds, binary_targets).sum()
    union = np.logical_or(binary_preds, binary_targets).sum()
    
    # Calculate dice coefficient (F1)
    smooth = 1e-8  # Avoid division by zero
    dice = (2.0 * intersection + smooth) / (binary_preds.sum() + binary_targets.sum() + smooth)
    
    # Calculate IoU (Jaccard index)
    iou = (intersection + smooth) / (union + smooth)
    
    # Calculate precision and recall
    true_positives = intersection
    false_positives = binary_preds.sum() - true_positives
    false_negatives = binary_targets.sum() - true_positives
    
    precision = true_positives / (true_positives + false_positives + smooth)
    recall = true_positives / (true_positives + false_negatives + smooth)
    
    # Calculate accuracy
    true_negatives = binary_preds.size - true_positives - false_positives - false_negatives
    accuracy = (true_positives + true_negatives) / binary_preds.size
    
    return {
        "dice": float(dice),
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
        "accuracy": float(accuracy)
    }


def calculate_reconstruction_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
) -> Dict[str, float]:
    """
    Calculate metrics for reconstruction tasks (e.g., autoencoders).
    
    Args:
        predictions: Reconstructed images
        targets: Original images
        
    Returns:
        Dictionary of reconstruction metrics
    """
    # Ensure images are in the correct range [0, 1] or [0, 255]
    if predictions.max() > 1.0 and targets.max() > 1.0:
        # Assume [0, 255] range
        data_range = 255
    else:
        # Assume [0, 1] range
        data_range = 1.0
        
    # Calculate Mean Squared Error
    mse = np.mean((predictions - targets) ** 2)
    
    # Calculate Peak Signal-to-Noise Ratio
    if mse == 0:
        psnr = 100  # Avoid log(0)
    else:
        psnr = 10 * np.log10((data_range ** 2) / mse)
    
    # Calculate Structural Similarity Index
    # Handle multi-channel images
    if predictions.ndim == 4:  # Batch of multi-channel images
        ssim_values = []
        for i in range(predictions.shape[0]):
            ssim_val = structural_similarity(
                predictions[i].transpose(1, 2, 0), 
                targets[i].transpose(1, 2, 0),
                data_range=data_range,
                multichannel=True  # Use multichannel for RGB
            )
            ssim_values.append(ssim_val)
        ssim = np.mean(ssim_values)
    elif predictions.ndim == 3 and predictions.shape[0] in [1, 3]:  # Single multi-channel image
        ssim = structural_similarity(
            predictions.transpose(1, 2, 0), 
            targets.transpose(1, 2, 0),
            data_range=data_range,
            multichannel=True
        )
    else:  # Grayscale images
        ssim = structural_similarity(
            predictions, 
            targets,
            data_range=data_range,
            multichannel=False
        )
    
    return {
        "mse": float(mse),
        "psnr": float(psnr),
        "ssim": float(ssim)
    }


def calculate_classification_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    class_names: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    Calculate metrics for classification tasks.
    
    Args:
        predictions: Predicted class probabilities or class indices
        targets: Ground truth class indices
        class_names: Optional list of class names
        
    Returns:
        Dictionary of classification metrics
    """
    # Convert probabilities to class indices if needed
    if predictions.ndim > 1 and predictions.shape[1] > 1:
        pred_classes = np.argmax(predictions, axis=1)
    else:
        pred_classes = (predictions > 0.5).astype(int)
    
    # Calculate metrics
    acc = accuracy_score(targets, pred_classes)
    
    # Handle binary vs multi-class
    if len(np.unique(targets)) <= 2:
        # Binary classification
        prec = precision_score(targets, pred_classes, zero_division=0)
        rec = recall_score(targets, pred_classes, zero_division=0)
        f1 = f1_score(targets, pred_classes, zero_division=0)
        
        # AUC if probabilities are available
        auc = 0.5  # Default if probabilities not available
        if predictions.ndim > 1 and predictions.shape[1] > 1:
            try:
                auc = roc_auc_score(targets, predictions[:, 1])
            except:
                pass
        elif np.any((predictions > 0) & (predictions < 1)):
            try:
                auc = roc_auc_score(targets, predictions)
            except:
                pass
        
        metrics = {
            "accuracy": float(acc),
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1),
            "auc": float(auc)
        }
    else:
        # Multi-class classification
        prec = precision_score(targets, pred_classes, average='macro', zero_division=0)
        rec = recall_score(targets, pred_classes, average='macro', zero_division=0)
        f1 = f1_score(targets, pred_classes, average='macro', zero_division=0)
        
        metrics = {
            "accuracy": float(acc),
            "precision_macro": float(prec),
            "recall_macro": float(rec),
            "f1_macro": float(f1)
        }
        
        # Add per-class metrics if class names are provided
        if class_names is not None:
            cm = confusion_matrix(targets, pred_classes)
            class_metrics = {}
            
            for i, class_name in enumerate(class_names):
                # For each class, calculate TP, FP, FN, TN
                tp = cm[i, i]
                fp = cm[:, i].sum() - tp
                fn = cm[i, :].sum() - tp
                tn = cm.sum() - tp - fp - fn
                
                # Calculate class metrics
                class_acc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
                class_prec = tp / (tp + fp) if (tp + fp) > 0 else 0
                class_rec = tp / (tp + fn) if (tp + fn) > 0 else 0
                class_f1 = 2 * class_prec * class_rec / (class_prec + class_rec) if (class_prec + class_rec) > 0 else 0
                
                class_metrics[f"{class_name}_accuracy"] = float(class_acc)
                class_metrics[f"{class_name}_precision"] = float(class_prec)
                class_metrics[f"{class_name}_recall"] = float(class_rec)
                class_metrics[f"{class_name}_f1"] = float(class_f1)
            
            metrics.update(class_metrics)
    
    return metrics


def calculate_patch_statistics(
    features: np.ndarray,
    labels: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Calculate statistics for feature patches.
    
    Args:
        features: Feature vectors extracted from patches
        labels: Optional ground truth labels for supervised evaluation
        
    Returns:
        Dictionary of feature statistics
    """
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    from sklearn.metrics import silhouette_score
    
    # Calculate basic statistics
    feature_mean = np.mean(features, axis=0)
    feature_std = np.std(features, axis=0)
    feature_min = np.min(features, axis=0)
    feature_max = np.max(features, axis=0)
    
    # Calculate feature norms
    feature_l2_norms = np.linalg.norm(features, axis=1)
    
    statistics = {
        "mean_feature_norm": float(np.mean(feature_l2_norms)),
        "std_feature_norm": float(np.std(feature_l2_norms)),
        "min_feature_norm": float(np.min(feature_l2_norms)),
        "max_feature_norm": float(np.max(feature_l2_norms)),
        "num_features": features.shape[1],
        "num_samples": features.shape[0]
    }
    
    # If labels are provided, calculate clustering metrics
    if labels is not None and len(np.unique(labels)) > 1:
        try:
            # Calculate silhouette score
            s_score = silhouette_score(features, labels)
            statistics["silhouette_score"] = float(s_score)
        except:
            pass
    
    return statistics
