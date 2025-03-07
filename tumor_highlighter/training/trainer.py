"""
Training utilities for the tumor_highlighter models.

This module provides a generic Trainer class that can be used to train
different kinds of models (autoencoder, classifier, MIL) with support
for experiment tracking, checkpointing, and visualization.
"""

import os
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Callable, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from tqdm import tqdm
import matplotlib.pyplot as plt

from tumor_highlighter.utils.experiment import ExperimentTracker
from tumor_highlighter.training.losses import get_loss_function
from tumor_highlighter.training.schedulers import create_scheduler

logger = logging.getLogger(__name__)


class Trainer:
    """
    Generic model trainer with experiment tracking.
    
    This class provides a unified interface for training different kinds
    of models, with support for checkpointing, visualization, and
    experiment tracking via MLflow and TensorBoard.
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: Dict[str, Any],
        experiment_tracker: Optional[ExperimentTracker] = None,
        device: Optional[str] = None,
    ):
        """
        Initialize trainer.
        
        Args:
            model: Model to train
            config: Training configuration
            experiment_tracker: Experiment tracker for logging metrics
            device: Device to train on (if None, use CUDA if available)
        """
        self.model = model
        self.config = config
        self.experiment_tracker = experiment_tracker
        
        # Set device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Initialize optimizer, loss function, and scheduler
        self._initialize_training_components()
        
        # Track best validation metrics
        self.best_val_loss = float('inf')
        self.best_val_metric = -float('inf')
        self.best_model_path = None
        
        # Set paths for saving models
        if experiment_tracker:
            self.output_dir = Path(experiment_tracker.output_dir)
        else:
            self.output_dir = Path(config.get('output_dir', 'output'))
        
        os.makedirs(self.output_dir, exist_ok=True)
        self.checkpoint_dir = self.output_dir / 'checkpoints'
        os.makedirs(self.checkpoint_dir, exist_ok=True)
    
    def _initialize_training_components(self) -> None:
        """Initialize optimizer, loss function, and scheduler."""
        # Get optimizer configuration
        optim_config = self.config.get('optimizer', {'name': 'adam', 'lr': 1e-4})
        optim_name = optim_config.get('name', 'adam').lower()
        lr = optim_config.get('lr', 1e-4)
        weight_decay = optim_config.get('weight_decay', 0)
        
        # Create optimizer
        if optim_name == 'adam':
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        elif optim_name == 'adamw':
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        elif optim_name == 'sgd':
            momentum = optim_config.get('momentum', 0.9)
            self.optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=lr,
                momentum=momentum,
                weight_decay=weight_decay
            )
        else:
            raise ValueError(f"Unsupported optimizer: {optim_name}")
        
        # Get loss function configuration
        loss_config = self.config.get('loss', {'name': 'mse'})
        loss_name = loss_config.get('name', 'mse')
        loss_params = loss_config.get('params', {})
        
        # Create loss function
        self.criterion = get_loss_function(loss_name, **loss_params)
        
        # Get scheduler configuration
        scheduler_config = self.config.get('scheduler', {'name': 'cosine'})
        scheduler_name = scheduler_config.get('name', 'cosine')
        
        # Create scheduler if specified
        if scheduler_name and scheduler_name != 'none':
            num_epochs = self.config.get('epochs', 100)
            
            # Extract steps_per_epoch from config or use default
            steps_per_epoch = self.config.get('steps_per_epoch', None)
            
            # Create scheduler
            self.scheduler = create_scheduler(
                name=scheduler_name,
                optimizer=self.optimizer,
                num_epochs=num_epochs,
                num_steps_per_epoch=steps_per_epoch,
                **scheduler_config.get('params', {})
            )
        else:
            self.scheduler = None
    
    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int,
        print_freq: int = 10,
        grad_clip: Optional[float] = None,
    ) -> Dict[str, float]:
        """
        Train the model for one epoch.
        
        Args:
            train_loader: DataLoader for training data
            epoch: Current epoch number
            print_freq: How often to print progress
            grad_clip: Maximum gradient norm for gradient clipping
            
        Returns:
            Dictionary of training metrics
        """
        # Set model to training mode
        self.model.train()
        
        # Initialize metrics
        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')
        
        # Additional metrics (to be tracked if available)
        metrics = {}
        
        # Setup progress bar
        progress = tqdm(
            enumerate(train_loader),
            desc=f"Epoch {epoch}",
            total=len(train_loader),
            unit="batch"
        )
        
        end = time.time()
        
        # Train on batches
        for i, batch in progress:
            # Measure data loading time
            data_time.update(time.time() - end)
            
            # Get data and move to device
            inputs, targets = self._prepare_batch(batch)
            
            # Forward pass
            outputs = self._forward_pass(inputs)
            
            # Compute loss
            loss = self._compute_loss(outputs, targets)
            
            # Update metrics
            losses.update(loss.item(), inputs.size(0))
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping (if specified)
            if grad_clip is not None:
                nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
            
            # Optimizer step
            self.optimizer.step()
            
            # Measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            
            # Update progress bar
            progress.set_postfix({'loss': f'{losses.avg:.4f}'})
            
            # Print progress
            if i % print_freq == 0:
                logger.info(
                    f'Epoch: [{epoch}][{i}/{len(train_loader)}] '
                    f'Loss: {losses.val:.4f} ({losses.avg:.4f})'
                )
        
        # Return metrics
        metrics = {
            'train_loss': losses.avg,
            'batch_time': batch_time.avg,
            'data_time': data_time.avg,
        }
        
        return metrics
    
    def validate(
        self,
        val_loader: DataLoader,
        epoch: int,
        print_freq: int = 10,
        visualization_func: Optional[Callable] = None,
    ) -> Dict[str, float]:
        """
        Validate the model on the validation set.
        
        Args:
            val_loader: DataLoader for validation data
            epoch: Current epoch number
            print_freq: How often to print progress
            visualization_func: Function to visualize model outputs
            
        Returns:
            Dictionary of validation metrics
        """
        # Set model to evaluation mode
        self.model.eval()
        
        # Initialize metrics
        batch_time = AverageMeter('Time', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')
        
        # Additional metrics
        metrics = {}
        
        # Setup progress bar
        progress = tqdm(
            enumerate(val_loader),
            desc=f"Validation Epoch {epoch}",
            total=len(val_loader),
            unit="batch"
        )
        
        # Visualization data
        vis_inputs = None
        vis_targets = None
        vis_outputs = None
        
        end = time.time()
        
        # No gradient computation for validation
        with torch.no_grad():
            for i, batch in progress:
                # Get data and move to device
                inputs, targets = self._prepare_batch(batch)
                
                # Forward pass
                outputs = self._forward_pass(inputs)
                
                # Compute loss
                loss = self._compute_loss(outputs, targets)
                
                # Update metrics
                losses.update(loss.item(), inputs.size(0))
                
                # Store data for visualization (first batch only)
                if i == 0 and visualization_func is not None:
                    vis_inputs = inputs
                    vis_targets = targets
                    vis_outputs = outputs
                
                # Measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()
                
                # Update progress bar
                progress.set_postfix({'loss': f'{losses.avg:.4f}'})
                
                # Print progress
                if i % print_freq == 0:
                    logger.info(
                        f'Validation: [{epoch}][{i}/{len(val_loader)}] '
                        f'Loss: {losses.val:.4f} ({losses.avg:.4f})'
                    )
        
        # Create visualizations if requested
        if visualization_func is not None and vis_inputs is not None:
            vis_images = visualization_func(vis_inputs, vis_outputs, vis_targets)
            
            # Log visualizations to experiment tracker
            if self.experiment_tracker and vis_images:
                self.experiment_tracker.log_images(vis_images, epoch)
        
        # Return metrics
        metrics = {
            'val_loss': losses.avg,
            'batch_time': batch_time.avg,
        }
        
        return metrics
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        visualization_func: Optional[Callable] = None,
        resume_from: Optional[str] = None,
        early_stopping_patience: Optional[int] = None,
    ) -> Dict[str, List[float]]:
        """
        Train the model for the specified number of epochs.
        
        Args:
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            epochs: Number of epochs to train
            visualization_func: Function to visualize model outputs
            resume_from: Path to checkpoint to resume from
            early_stopping_patience: Patience for early stopping
            
        Returns:
            Dictionary of training history
        """
        # Resume from checkpoint if specified
        start_epoch = 0
        if resume_from:
            start_epoch = self.load_checkpoint(resume_from)
        
        # Initialize history
        history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': [],
        }
        
        # Initialize early stopping counter
        if early_stopping_patience:
            early_stopping_counter = 0
        
        # Train for specified number of epochs
        for epoch in range(start_epoch, epochs):
            logger.info(f"Starting epoch {epoch + 1}/{epochs}")
            
            # Get current learning rate
            lr = self.optimizer.param_groups[0]['lr']
            history['learning_rate'].append(lr)
            
            # Log learning rate
            if self.experiment_tracker:
                self.experiment_tracker.log_metrics({'learning_rate': lr}, epoch)
            
            # Train for one epoch
            train_metrics = self.train_epoch(train_loader, epoch)
            history['train_loss'].append(train_metrics['train_loss'])
            
            # Validate
            val_metrics = self.validate(val_loader, epoch, visualization_func=visualization_func)
            history['val_loss'].append(val_metrics['val_loss'])
            
            # Log metrics
            if self.experiment_tracker:
                self.experiment_tracker.log_metrics({
                    'train_loss': train_metrics['train_loss'],
                    'val_loss': val_metrics['val_loss'],
                }, epoch)
            
            # Update learning rate scheduler if using epoch-based scheduler
            if self.scheduler is not None:
                if hasattr(self.scheduler, 'step'):
                    if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        self.scheduler.step(val_metrics['val_loss'])
                    else:
                        self.scheduler.step()
            
            # Save checkpoint
            self.save_checkpoint(epoch, val_metrics['val_loss'])
            
            # Check if this is the best model
            if val_metrics['val_loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['val_loss']
                self.save_model(epoch, is_best=True)
                
                # Reset early stopping counter
                if early_stopping_patience:
                    early_stopping_counter = 0
            else:
                # Increment early stopping counter
                if early_stopping_patience:
                    early_stopping_counter += 1
                    if early_stopping_counter >= early_stopping_patience:
                        logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                        break
        
        # Save final model
        self.save_model(epochs - 1, is_best=False)
        
        # End experiment run
        if self.experiment_tracker:
            self.experiment_tracker.end_run()
        
        return history
    
    def _prepare_batch(self, batch: Any) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare batch for training/validation.
        
        Override this method in subclasses to handle specific data formats.
        
        Args:
            batch: Batch from DataLoader
            
        Returns:
            Tuple of (inputs, targets) tensors
        """
        # Default implementation assumes batch is a tuple of (inputs, targets)
        if isinstance(batch, (list, tuple)) and len(batch) >= 2:
            inputs, targets = batch[0], batch[1]
        else:
            # For autoencoder training, inputs and targets might be the same
            inputs = batch
            targets = batch
        
        # Move to device
        if isinstance(inputs, torch.Tensor):
            inputs = inputs.to(self.device)
        
        if isinstance(targets, torch.Tensor):
            targets = targets.to(self.device)
        
        return inputs, targets
    
    def _forward_pass(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Perform forward pass through the model.
        
        Override this method in subclasses to handle specific models.
        
        Args:
            inputs: Input tensor
            
        Returns:
            Output tensor
        """
        # Default implementation just passes inputs through the model
        return self.model(inputs)
    
    def _compute_loss(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute loss between outputs and targets.
        
        Override this method in subclasses to handle specific loss calculations.
        
        Args:
            outputs: Model outputs
            targets: Target values
            
        Returns:
            Loss value
        """
        # Default implementation uses the criterion initialized in __init__
        return self.criterion(outputs, targets)
    
    def save_checkpoint(self, epoch: int, val_loss: float) -> str:
        """
        Save training checkpoint.
        
        Args:
            epoch: Current epoch
            val_loss: Validation loss
            
        Returns:
            Path to saved checkpoint
        """
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'config': self.config,
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved to {checkpoint_path}")
        
        return str(checkpoint_path)
    
    def load_checkpoint(self, checkpoint_path: str) -> int:
        """
        Load training checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint
            
        Returns:
            Epoch number from checkpoint
        """
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'scheduler_state_dict' in checkpoint and self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['val_loss']
        
        return epoch + 1  # Return the next epoch
    
    def save_model(self, epoch: int, is_best: bool = False) -> str:
        """
        Save model weights.
        
        Args:
            epoch: Current epoch
            is_best: Whether this is the best model so far
            
        Returns:
            Path to saved model
        """
        if is_best:
            model_path = self.output_dir / "best_model.pt"
            self.best_model_path = model_path
        else:
            model_path = self.output_dir / f"model_epoch_{epoch}.pt"
        
        torch.save(self.model.state_dict(), model_path)
        logger.info(f"Model saved to {model_path}")
        
        return str(model_path)
    
    def load_model(self, model_path: Optional[str] = None) -> None:
        """
        Load model weights.
        
        Args:
            model_path: Path to model weights (if None, load best model)
        """
        if model_path is None:
            if self.best_model_path is not None:
                model_path = str(self.best_model_path)
            else:
                raise ValueError("No best model path available")
        
        logger.info(f"Loading model from {model_path}")
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))


class AverageMeter:
    """
    Computes and stores the average and current value.
    
    Useful for tracking metrics during training.
    """
    
    def __init__(self, name: str, fmt: str = ':f'):
        """
        Initialize average meter.
        
        Args:
            name: Name of the metric
            fmt: Format string for printing
        """
        self.name = name
        self.fmt = fmt
        self.reset()
    
    def reset(self) -> None:
        """Reset all metrics."""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val: float, n: int = 1) -> None:
        """
        Update metrics with new value.
        
        Args:
            val: New value
            n: Number of samples this value represents
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    
    def __str__(self) -> str:
        """String representation."""
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class AutoencoderTrainer(Trainer):
    """
    Specialized trainer for autoencoder models.
    
    This trainer includes utilities for visualizing reconstructions
    and handling autoencoder-specific training workflows.
    """
    
    def _prepare_batch(self, batch: Any) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare batch for autoencoder training.
        
        For autoencoders, the input and target are the same.
        
        Args:
            batch: Batch from DataLoader
            
        Returns:
            Tuple of (inputs, targets) tensors
        """
        # For autoencoders, inputs and targets are the same
        if isinstance(batch, (list, tuple)) and len(batch) >= 1:
            inputs = batch[0]
            targets = inputs  # Use inputs as targets
        else:
            inputs = batch
            targets = inputs
        
        # Move to device
        if isinstance(inputs, torch.Tensor):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
        
        return inputs, targets
    
    @staticmethod
    def visualize_reconstructions(
        inputs: torch.Tensor,
        outputs: torch.Tensor,
        targets: torch.Tensor,
        num_samples: int = 8,
    ) -> Dict[str, np.ndarray]:
        """
        Visualize autoencoder reconstructions.
        
        Args:
            inputs: Input tensor (B, C, H, W)
            outputs: Output tensor (B, C, H, W)
            targets: Target tensor (B, C, H, W)
            num_samples: Number of samples to visualize
            
        Returns:
            Dictionary of visualization images
        """
        # Use only a subset of samples
        batch_size = inputs.size(0)
        num_samples = min(num_samples, batch_size)
        
        # Convert tensors to numpy
        inputs_np = inputs[:num_samples].detach().cpu().numpy()
        outputs_np = outputs[:num_samples].detach().cpu().numpy()
        
        # Transpose from (B, C, H, W) to (B, H, W, C)
        inputs_np = np.transpose(inputs_np, (0, 2, 3, 1))
        outputs_np = np.transpose(outputs_np, (0, 2, 3, 1))
        
        # Clip to [0, 1] range
        inputs_np = np.clip(inputs_np, 0, 1)
        outputs_np = np.clip(outputs_np, 0, 1)
        
        # Create visualization grid
        fig, axes = plt.subplots(2, num_samples, figsize=(2 * num_samples, 4))
        
        for i in range(num_samples):
            # Display input
            if inputs_np.shape[-1] == 1:  # Grayscale
                axes[0, i].imshow(inputs_np[i, :, :, 0], cmap='gray')
                axes[1, i].imshow(outputs_np[i, :, :, 0], cmap='gray')
            else:  # RGB
                axes[0, i].imshow(inputs_np[i])
                axes[1, i].imshow(outputs_np[i])
            
            # Remove axis ticks
            axes[0, i].set_xticks([])
            axes[0, i].set_yticks([])
            axes[1, i].set_xticks([])
            axes[1, i].set_yticks([])
        
        # Add row labels
        axes[0, 0].set_ylabel('Input')
        axes[1, 0].set_ylabel('Reconstruction')
        
        # Save figure to numpy array
        fig.tight_layout()
        fig.canvas.draw()
        
        # Convert figure to numpy array
        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        
        plt.close(fig)
        
        return {'reconstructions': data}


class MILTrainer(Trainer):
    """
    Specialized trainer for Multiple Instance Learning (MIL) models.
    
    This trainer includes utilities for handling MIL-specific training
    workflows, such as bag-level predictions and attention visualization.
    """
    
    def _prepare_batch(self, batch: Any) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare batch for MIL training.
        
        Args:
            batch: Batch from DataLoader, expected to be a tuple of
                  (bags, labels) where bags is a tensor of shape
                  (batch_size, num_instances, feature_dim)
            
        Returns:
            Tuple of (bags, labels) tensors
        """
        # MIL batch should be a tuple of (bags, labels)
        if isinstance(batch, (list, tuple)) and len(batch) >= 2:
            bags, labels = batch[0], batch[1]
        else:
            raise ValueError("MIL batch should be a tuple of (bags, labels)")
        
        # Move to device
        if isinstance(bags, torch.Tensor):
            bags = bags.to(self.device)
        
        if isinstance(labels, torch.Tensor):
            labels = labels.to(self.device)
        
        return bags, labels
    
    def _forward_pass(self, bags: torch.Tensor) -> torch.Tensor:
        """
        Perform forward pass for MIL model.
        
        Args:
            bags: Input tensor of shape (batch_size, num_instances, feature_dim)
            
        Returns:
            Model predictions and attention weights
        """
        # MIL models typically return both predictions and attention weights
        if hasattr(self.model, 'get_attention_weights'):
            predictions, attention = self.model(bags, return_attention=True)
            return predictions, attention
        else:
            return self.model(bags)
    
    def _compute_loss(
        self,
        outputs: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute loss for MIL model.
        
        Args:
            outputs: Model outputs, either predictions or a tuple of
                    (predictions, attention_weights)
            targets: Target labels
            
        Returns:
            Loss value
        """
        # Extract predictions if outputs is a tuple
        if isinstance(outputs, tuple) and len(outputs) >= 1:
            predictions = outputs[0]
        else:
            predictions = outputs
        
        # Compute loss
        return self.criterion(predictions, targets)
    
    @staticmethod
    def visualize_attention(
        outputs: Tuple[torch.Tensor, torch.Tensor],
        num_samples: int = 4,
    ) -> Dict[str, np.ndarray]:
        """
        Visualize attention weights from MIL model.
        
        Args:
            outputs: Tuple of (predictions, attention_weights)
            num_samples: Number of samples to visualize
            
        Returns:
            Dictionary of visualization images
        """
        # Extract attention weights
        if isinstance(outputs, tuple) and len(outputs) >= 2:
            attention = outputs[1]
        else:
            return {}
        
        # Use only a subset of samples
        batch_size = attention.size(0)
        num_samples = min(num_samples, batch_size)
        
        # Convert attention to numpy
        attention_np = attention[:num_samples].detach().cpu().numpy()
        
        # Create visualization grid
        fig, axes = plt.subplots(1, num_samples, figsize=(3 * num_samples, 3))
        
        # Ensure axes is an array even if num_samples=1
        if num_samples == 1:
            axes = np.array([axes])
        
        for i in range(num_samples):
            # Reshape attention for visualization
            att = attention_np[i].squeeze()
            
            # Plot as heatmap
            if att.ndim == 1:
                # 1D attention (instance-level)
                im = axes[i].imshow(att.reshape(1, -1), cmap='viridis', aspect='auto')
                axes[i].set_title(f'Sample {i+1}')
                axes[i].set_yticks([])
                axes[i].set_xlabel('Instance')
            elif att.ndim == 2:
                # 2D attention (spatial)
                im = axes[i].imshow(att, cmap='viridis')
                axes[i].set_title(f'Sample {i+1}')
                axes[i].set_xticks([])
                axes[i].set_yticks([])
        
        # Add colorbar
        fig.colorbar(im, ax=axes.ravel().tolist())
        
        # Save figure to numpy array
        fig.tight_layout()
        fig.canvas.draw()
        
        # Convert figure to numpy array
        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        
        plt.close(fig)
        
        return {'attention_weights': data}
