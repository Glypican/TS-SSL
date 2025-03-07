"""Trainer for Task-Specific Self-Supervised Learning (TS-SSL) models."""

import os
from pathlib import Path
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

from tumor_highlighter import logger
from tumor_highlighter.models.ts_ssl.autoencoder import SpatialChannelAttentionAutoencoder
from tumor_highlighter.data.wsi_dataset import PatchDataset
from tumor_highlighter.training.losses import ReconstructionLoss
from tumor_highlighter.models.model_registry import ModelRegistry

# Optional MLflow tracking
try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False


class TSSSLTrainer:
    """Trainer for TS-SSL autoencoder models."""
    
    def __init__(
        self,
        model_name,
        input_shape=(256, 256),
        batch_size=16,
        learning_rate=0.001,
        output_dir="./models",
        track_mlflow=False,
    ):
        """Initialize the TS-SSL trainer.
        
        Args:
            model_name (str): Name for the trained model
            input_shape (tuple): Input shape for the model (height, width)
            batch_size (int): Batch size for training
            learning_rate (float): Learning rate for training
            output_dir (str): Directory to save the trained model
            track_mlflow (bool): Whether to track the experiment with MLflow
        """
        self.model_name = model_name
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.output_dir = Path(output_dir)
        self.track_mlflow = track_mlflow and MLFLOW_AVAILABLE
        
        # Initialize model
        self.model = SpatialChannelAttentionAutoencoder(input_shape=input_shape)
        
        # Initialize loss function
        self.criterion = ReconstructionLoss()
        
        # Initialize optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Get device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        logger.info(f"Initialized TS-SSL trainer with model: {model_name}")
        logger.info(f"Using device: {self.device}")
    
    def train(self, data_path, epochs=50, subsample=0.1, validation_split=0.1):
        """Train the TS-SSL model.
        
        Args:
            data_path (str): Path to the training data directory
            epochs (int): Number of epochs for training
            subsample (float): Fraction of patches to use for training (0.0-1.0)
            validation_split (float): Fraction of data to use for validation (0.0-1.0)
        
        Returns:
            dict: Dictionary with training metrics
        """
        logger.info(f"Training TS-SSL model: {self.model_name}")
        logger.info(f"Data path: {data_path}")
        logger.info(f"Epochs: {epochs}")
        logger.info(f"Subsample: {subsample}")
        
        # Create dataset
        dataset = PatchDataset(data_path, subsample=subsample)
        
        # Split dataset into training and validation
        dataset_size = len(dataset)
        val_size = int(validation_split * dataset_size)
        train_size = dataset_size - val_size
        
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )
        
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4
        )
        
        val_loader = DataLoader(
            val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4
        )
        
        logger.info(f"Training dataset size: {train_size}")
        logger.info(f"Validation dataset size: {val_size}")
        
        # Set up MLflow tracking if enabled
        if self.track_mlflow:
            try:
                mlflow.set_experiment(f"TS-SSL-{self.model_name}")
                mlflow.start_run(run_name=f"{self.model_name}-{time.strftime('%Y%m%d-%H%M%S')}")
                
                # Log parameters
                mlflow.log_params({
                    "model_name": self.model_name,
                    "input_shape": self.input_shape,
                    "batch_size": self.batch_size,
                    "learning_rate": self.learning_rate,
                    "epochs": epochs,
                    "subsample": subsample,
                    "validation_split": validation_split,
                    "dataset_size": dataset_size,
                    "train_size": train_size,
                    "val_size": val_size,
                })
            except Exception as e:
                logger.warning(f"Failed to set up MLflow tracking: {e}")
                self.track_mlflow = False
        
        # Training loop
        best_loss = float('inf')
        metrics = {
            "train_loss": [],
            "val_loss": [],
            "best_epoch": 0,
        }
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = self._train_epoch(train_loader, epoch, epochs)
            metrics["train_loss"].append(train_loss)
            
            # Validation phase
            self.model.eval()
            val_loss = self._validate_epoch(val_loader)
            metrics["val_loss"].append(val_loss)
            
            # Log metrics
            logger.info(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
            
            if self.track_mlflow:
                try:
                    mlflow.log_metric("train_loss", train_loss, step=epoch)
                    mlflow.log_metric("val_loss", val_loss, step=epoch)
                except Exception as e:
                    logger.warning(f"Failed to log metrics to MLflow: {e}")
            
            # Save best model
            if val_loss < best_loss:
                best_loss = val_loss
                metrics["best_epoch"] = epoch
                self._save_model(is_best=True)
                logger.info(f"New best model saved (val_loss: {val_loss:.6f})")
        
        # Save final model
        self._save_model(is_best=False)
        
        # Register model
        model_registry = ModelRegistry()
        model_registry.register_model(
            model_path=str(self.output_dir / f"{self.model_name}_best.pth"),
            model_name=self.model_name,
            model_type="ts-ssl",
            description=f"TS-SSL model trained on {data_path} for {epochs} epochs",
        )
        
        # Close MLflow run if enabled
        if self.track_mlflow:
            try:
                mlflow.log_metric("best_val_loss", best_loss)
                mlflow.log_artifact(str(self.output_dir / f"{self.model_name}_best.pth"))
                mlflow.end_run()
            except Exception as e:
                logger.warning(f"Failed to finalize MLflow run: {e}")
        
        logger.info(f"Training completed with best validation loss: {best_loss:.6f} at epoch {metrics['best_epoch']+1}")
        
        return metrics
    
    def _train_epoch(self, dataloader, epoch, total_epochs):
        """Train for one epoch.
        
        Args:
            dataloader: PyTorch dataloader with training data
            epoch: Current epoch number
            total_epochs: Total number of epochs
            
        Returns:
            float: Average loss for the epoch
        """
        running_loss = 0.0
        total_samples = 0
        
        progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch+1}/{total_epochs}")
        
        for batch_idx, batch in progress_bar:
            # Move data to device
            inputs = batch["image"].to(self.device)
            
            # Zero the parameter gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs, encoded_features = self.model(inputs)
            
            # Calculate loss
            loss = self.criterion(outputs, inputs)
            
            # Backward pass and optimize
            loss.backward()
            self.optimizer.step()
            
            # Update statistics
            batch_size = inputs.size(0)
            running_loss += loss.item() * batch_size
            total_samples += batch_size
            
            # Update progress bar
            progress_bar.set_postfix({"loss": loss.item()})
        
        epoch_loss = running_loss / total_samples
        return epoch_loss
    
    def _validate_epoch(self, dataloader):
        """Validate for one epoch.
        
        Args:
            dataloader: PyTorch dataloader with validation data
            
        Returns:
            float: Average validation loss for the epoch
        """
        running_loss = 0.0
        total_samples = 0
        
        with torch.no_grad():
            for batch in dataloader:
                # Move data to device
                inputs = batch["image"].to(self.device)
                
                # Forward pass
                outputs, encoded_features = self.model(inputs)
                
                # Calculate loss
                loss = self.criterion(outputs, inputs)
                
                # Update statistics
                batch_size = inputs.size(0)
                running_loss += loss.item() * batch_size
                total_samples += batch_size
        
        epoch_loss = running_loss / total_samples
        return epoch_loss
    
    def _save_model(self, is_best=False):
        """Save the model.
        
        Args:
            is_best (bool): Whether this is the best model so far
        """
        suffix = "_best" if is_best else ""
        model_path = self.output_dir / f"{self.model_name}{suffix}.pth"
        
        # Create directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Save model state dict
        torch.save(self.model.state_dict(), model_path)
