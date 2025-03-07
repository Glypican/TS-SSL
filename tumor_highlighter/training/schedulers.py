"""
Learning rate schedulers for training models in the tumor_highlighter package.

This module provides different learning rate schedulers that can be used for training
models with dynamic learning rate adjustment strategies.
"""

import math
from typing import Dict, List, Optional, Union, Callable, Tuple

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import (
    _LRScheduler,
    CosineAnnealingLR,
    ReduceLROnPlateau,
    OneCycleLR,
    ExponentialLR,
)


class LinearWarmupScheduler(_LRScheduler):
    """
    Linear learning rate warmup scheduler.
    
    This scheduler linearly increases the learning rate from a low value
    to the base learning rate over a specified number of steps.
    """
    
    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        max_lr: Optional[float] = None,
        min_lr: float = 0.0,
        last_epoch: int = -1,
    ):
        """
        Initialize linear warmup scheduler.
        
        Args:
            optimizer: Optimizer to adjust learning rate for
            warmup_steps: Number of warmup steps
            max_lr: Maximum learning rate (if None, uses initial lr from optimizer)
            min_lr: Minimum learning rate to start from
            last_epoch: The index of the last epoch
        """
        self.warmup_steps = warmup_steps
        self.max_lr = max_lr
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self) -> List[float]:
        """
        Calculate learning rate for the current step.
        
        Returns:
            List of learning rates for each parameter group
        """
        if self.last_epoch >= self.warmup_steps:
            return [group['lr'] for group in self.optimizer.param_groups]
        
        # Linear warmup
        alpha = self.last_epoch / self.warmup_steps
        
        return [self.min_lr + alpha * (base_lr if self.max_lr is None else self.max_lr) 
                for base_lr in self.base_lrs]


class WarmupCosineScheduler(_LRScheduler):
    """
    Warmup cosine learning rate scheduler.
    
    This scheduler combines linear warmup with cosine annealing,
    providing smooth transitions for the learning rate.
    """
    
    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr: float = 0.0,
        last_epoch: int = -1,
    ):
        """
        Initialize warmup cosine scheduler.
        
        Args:
            optimizer: Optimizer to adjust learning rate for
            warmup_steps: Number of warmup steps
            total_steps: Total number of training steps
            min_lr: Minimum learning rate
            last_epoch: The index of the last epoch
        """
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self) -> List[float]:
        """
        Calculate learning rate for the current step.
        
        Returns:
            List of learning rates for each parameter group
        """
        if self.last_epoch < 0:
            return [group['lr'] for group in self.optimizer.param_groups]
        
        # Warmup phase
        if self.last_epoch < self.warmup_steps:
            alpha = self.last_epoch / self.warmup_steps
            return [self.min_lr + alpha * base_lr for base_lr in self.base_lrs]
        
        # Cosine annealing phase
        step = self.last_epoch - self.warmup_steps
        total_cosine_steps = self.total_steps - self.warmup_steps
        
        # Ensure we don't go beyond total steps
        progress = min(step / total_cosine_steps, 1.0)
        
        # Cosine schedule
        cosine_factor = 0.5 * (1 + math.cos(math.pi * progress))
        
        return [self.min_lr + cosine_factor * (base_lr - self.min_lr) 
                for base_lr in self.base_lrs]


class StepWarmupScheduler(_LRScheduler):
    """
    Step decay with warmup scheduler.
    
    This scheduler combines linear warmup with step decay,
    reducing the learning rate by a factor at specified milestones.
    """
    
    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        milestones: List[int],
        gamma: float = 0.1,
        min_lr: float = 0.0,
        last_epoch: int = -1,
    ):
        """
        Initialize step warmup scheduler.
        
        Args:
            optimizer: Optimizer to adjust learning rate for
            warmup_steps: Number of warmup steps
            milestones: List of epoch indices to reduce learning rate
            gamma: Factor by which to reduce learning rate
            min_lr: Minimum learning rate
            last_epoch: The index of the last epoch
        """
        self.warmup_steps = warmup_steps
        self.milestones = sorted(milestones)
        self.gamma = gamma
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self) -> List[float]:
        """
        Calculate learning rate for the current step.
        
        Returns:
            List of learning rates for each parameter group
        """
        if self.last_epoch < 0:
            return [group['lr'] for group in self.optimizer.param_groups]
        
        # Warmup phase
        if self.last_epoch < self.warmup_steps:
            alpha = self.last_epoch / self.warmup_steps
            return [self.min_lr + alpha * base_lr for base_lr in self.base_lrs]
        
        # Step decay phase
        decay_factor = self.gamma ** len([m for m in self.milestones if m <= self.last_epoch])
        
        return [max(self.min_lr, base_lr * decay_factor) for base_lr in self.base_lrs]


class PlateauWarmupScheduler:
    """
    ReduceLROnPlateau with warmup scheduler.
    
    This scheduler combines linear warmup with ReduceLROnPlateau,
    which reduces learning rate when a metric has stopped improving.
    """
    
    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        factor: float = 0.1,
        patience: int = 10,
        min_lr: float = 0.0,
        mode: str = 'min',
        **plateau_kwargs,
    ):
        """
        Initialize plateau warmup scheduler.
        
        Args:
            optimizer: Optimizer to adjust learning rate for
            warmup_steps: Number of warmup steps
            factor: Factor by which to reduce learning rate
            patience: Number of epochs with no improvement after which to reduce lr
            min_lr: Minimum learning rate
            mode: 'min' or 'max' depending on whether the metric should be minimized or maximized
            **plateau_kwargs: Additional arguments for ReduceLROnPlateau
        """
        self.warmup_steps = warmup_steps
        self.min_lr = min_lr
        self.optimizer = optimizer
        self.current_step = 0
        
        # Get base learning rates
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        
        # Initialize plateau scheduler (used after warmup)
        self.plateau_scheduler = ReduceLROnPlateau(
            optimizer,
            mode=mode,
            factor=factor,
            patience=patience,
            min_lr=min_lr,
            **plateau_kwargs,
        )
    
    def step(self, metrics=None, epoch=None) -> None:
        """
        Step the scheduler based on the current step or metrics.
        
        Args:
            metrics: Validation metrics (only used by plateau scheduler)
            epoch: Current epoch (not used)
        """
        self.current_step += 1
        
        # Warmup phase
        if self.current_step < self.warmup_steps:
            alpha = self.current_step / self.warmup_steps
            for i, param_group in enumerate(self.optimizer.param_groups):
                param_group['lr'] = self.min_lr + alpha * self.base_lrs[i]
        else:
            # Plateau phase
            self.plateau_scheduler.step(metrics)
    
    def state_dict(self) -> Dict:
        """Get state dictionary for checkpointing."""
        return {
            'current_step': self.current_step,
            'base_lrs': self.base_lrs,
            'plateau_scheduler': self.plateau_scheduler.state_dict(),
        }
    
    def load_state_dict(self, state_dict: Dict) -> None:
        """Load state dictionary from checkpoint."""
        self.current_step = state_dict['current_step']
        self.base_lrs = state_dict['base_lrs']
        self.plateau_scheduler.load_state_dict(state_dict['plateau_scheduler'])


# Registry of available schedulers
SCHEDULER_REGISTRY = {
    'cosine': CosineAnnealingLR,
    'linear_warmup': LinearWarmupScheduler,
    'warmup_cosine': WarmupCosineScheduler,
    'step_warmup': StepWarmupScheduler,
    'plateau_warmup': PlateauWarmupScheduler,
    'one_cycle': OneCycleLR,
    'exponential': ExponentialLR,
    'reduce_on_plateau': ReduceLROnPlateau,
}


def create_scheduler(
    name: str,
    optimizer: Optimizer,
    num_epochs: int,
    num_steps_per_epoch: int = None,
    **kwargs,
) -> Union[_LRScheduler, ReduceLROnPlateau, PlateauWarmupScheduler]:
    """
    Create a learning rate scheduler by name.
    
    Args:
        name: Name of the scheduler
        optimizer: Optimizer to adjust learning rate for
        num_epochs: Total number of training epochs
        num_steps_per_epoch: Number of steps per epoch (needed for some schedulers)
        **kwargs: Additional arguments for the scheduler
        
    Returns:
        Instantiated scheduler
        
    Raises:
        ValueError: If scheduler is not found in registry or required args are missing
    """
    if name not in SCHEDULER_REGISTRY:
        raise ValueError(f"Scheduler '{name}' not found. Available schedulers: {list(SCHEDULER_REGISTRY.keys())}")
    
    scheduler_cls = SCHEDULER_REGISTRY[name]
    
    # Compute total steps if needed (for step-based schedulers)
    total_steps = num_epochs * num_steps_per_epoch if num_steps_per_epoch else num_epochs
    
    # Configure scheduler based on name
    if name == 'cosine':
        return scheduler_cls(optimizer, T_max=num_epochs, **kwargs)
    
    elif name == 'one_cycle':
        if num_steps_per_epoch is None:
            raise ValueError("num_steps_per_epoch is required for one_cycle scheduler")
        return scheduler_cls(optimizer, total_steps=total_steps, **kwargs)
    
    elif name == 'warmup_cosine':
        return scheduler_cls(optimizer, total_steps=total_steps, **kwargs)
    
    elif name == 'linear_warmup':
        return scheduler_cls(optimizer, **kwargs)
    
    elif name == 'step_warmup':
        return scheduler_cls(optimizer, **kwargs)
    
    elif name == 'plateau_warmup':
        return scheduler_cls(optimizer, **kwargs)
    
    elif name == 'exponential':
        return scheduler_cls(optimizer, **kwargs)
    
    elif name == 'reduce_on_plateau':
        return scheduler_cls(optimizer, **kwargs)
    
    # Default case: just pass kwargs
    return scheduler_cls(optimizer, **kwargs)


def get_warmup_scheduler_config(config: Dict) -> Dict:
    """
    Get scheduler configuration with warmup from a general config.
    
    Args:
        config: Configuration dictionary with scheduler parameters
        
    Returns:
        Dictionary with scheduler parameters
    """
    scheduler_config = {
        'name': config.get('name', 'warmup_cosine'),
    }
    
    # Add other parameters based on scheduler type
    if scheduler_config['name'] == 'warmup_cosine':
        scheduler_config.update({
            'warmup_steps': config.get('warmup_epochs', 10),
            'min_lr': config.get('min_lr', 0.0),
        })
    
    elif scheduler_config['name'] == 'linear_warmup':
        scheduler_config.update({
            'warmup_steps': config.get('warmup_epochs', 10),
            'min_lr': config.get('min_lr', 0.0),
        })
    
    elif scheduler_config['name'] == 'step_warmup':
        scheduler_config.update({
            'warmup_steps': config.get('warmup_epochs', 10),
            'milestones': config.get('milestones', [30, 60, 90]),
            'gamma': config.get('gamma', 0.1),
            'min_lr': config.get('min_lr', 0.0),
        })
    
    elif scheduler_config['name'] == 'plateau_warmup':
        scheduler_config.update({
            'warmup_steps': config.get('warmup_epochs', 10),
            'factor': config.get('factor', 0.1),
            'patience': config.get('patience', 10),
            'min_lr': config.get('min_lr', 0.0),
            'mode': config.get('mode', 'min'),
        })
    
    return scheduler_config
