"""
Experiment tracking utilities for tumor_highlighter.
Implements MLflow tracking with TensorBoard visualization support.
"""

import os
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Union, List

import mlflow
import numpy as np
from omegaconf import DictConfig, OmegaConf
import torch
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt


class ExperimentTracker:
    """
    Experiment tracker that integrates MLflow and TensorBoard.
    
    This class provides utilities for:
    - Tracking hyperparameters, metrics, and artifacts
    - Logging model reconstructions and visualizations
    - Creating experiment reports for sharing
    """
    
    def __init__(
        self,
        config: DictConfig,
        experiment_name: Optional[str] = None,
        run_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
    ):
        """
        Initialize experiment tracker.
        
        Args:
            config: Configuration for the experiment
            experiment_name: Name of the MLflow experiment (overrides config)
            run_name: Name of the MLflow run
            tags: Tags to add to the MLflow run
        """
        self.config = config
        
        # Set up MLflow
        tracking_uri = config.experiment.mlflow.tracking_uri
        mlflow.set_tracking_uri(tracking_uri)
        
        # Set experiment name
        self.experiment_name = experiment_name or config.experiment.mlflow.experiment_name
        mlflow.set_experiment(self.experiment_name)
        
        # Generate run name if not provided
        if run_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_name = f"{self.experiment_name}_{timestamp}"
        self.run_name = run_name
        
        # Set up run
        self.run = mlflow.start_run(run_name=run_name)
        
        # Set tags
        if tags:
            mlflow.set_tags(tags)
        
        # Add default tags
        mlflow.set_tag("timestamp", datetime.now().isoformat())
        
        # Log config as parameters
        flat_config = self._flatten_dict(OmegaConf.to_container(config, resolve=True))
        for key, value in flat_config.items():
            if isinstance(value, (int, float, str, bool)):
                mlflow.log_param(key, value)
        
        # Set up TensorBoard
        tensorboard_dir = Path(config.experiment.tensorboard.log_dir) / run_name
        self.tensorboard = SummaryWriter(log_dir=tensorboard_dir)
        
        # Create output directory
        self.output_dir = Path(config.experiment.output_dir) / run_name
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Save config to output directory
        self._save_config()
        
        # Store run ID for easy access
        self.run_id = self.run.info.run_id
        
        # Save run info for easy access
        run_info = {
            "experiment_name": self.experiment_name,
            "run_name": self.run_name,
            "run_id": self.run_id,
            "start_time": datetime.now().isoformat(),
            "output_dir": str(self.output_dir),
            "tensorboard_dir": str(tensorboard_dir)
        }
        with open(self.output_dir / "run_info.json", "w") as f:
            json.dump(run_info, f, indent=2)
    
    def log_metrics(self, metrics: Dict[str, float], step: int) -> None:
        """
        Log metrics to MLflow and TensorBoard.
        
        Args:
            metrics: Dictionary of metric names and values
            step: Current step (epoch or iteration)
        """
        # Log to MLflow
        mlflow.log_metrics(metrics, step=step)
        
        # Log to TensorBoard
        for name, value in metrics.items():
            self.tensorboard.add_scalar(name, value, step)
    
    def log_images(
        self, 
        images_dict: Dict[str, np.ndarray], 
        step: int,
        log_to_mlflow: bool = False
    ) -> None:
        """
        Log images to TensorBoard and optionally MLflow.
        
        Args:
            images_dict: Dictionary of image names and numpy arrays
            step: Current step (epoch or iteration)
            log_to_mlflow: Whether to also log images to MLflow
        """
        # Log to TensorBoard
        for name, image in images_dict.items():
            # Ensure image is in the right format
            if image.ndim == 2:  # Grayscale
                image = np.expand_dims(image, axis=0)  # Add channel dim
            elif image.ndim == 3 and image.shape[2] in [1, 3]:  # HWC format
                # Convert to CHW format expected by TensorBoard
                image = np.transpose(image, (2, 0, 1))
            
            # Add batch dimension if needed
            if image.ndim == 3:
                image = np.expand_dims(image, axis=0)
                
            self.tensorboard.add_images(name, image, step)
        
        # Log to MLflow if requested
        if log_to_mlflow:
            for name, image in images_dict.items():
                # Create figure and log it
                fig, ax = plt.subplots(figsize=(10, 10))
                if image.ndim == 2 or (image.ndim == 3 and image.shape[2] == 1):
                    ax.imshow(image, cmap='gray')
                else:
                    ax.imshow(image)
                ax.axis('off')
                
                # Save and log
                img_path = self.output_dir / f"{name}_{step}.png"
                plt.savefig(img_path, bbox_inches='tight')
                plt.close(fig)
                
                mlflow.log_artifact(str(img_path), f"images/{name}")
    
    def log_model(
        self, 
        model: torch.nn.Module, 
        name: str, 
        save_format: str = "pytorch"
    ) -> None:
        """
        Log model to MLflow and save locally.
        
        Args:
            model: PyTorch model to save
            name: Name to identify the model
            save_format: Format to save the model ("pytorch" or "onnx")
        """
        # Save model locally
        model_dir = self.output_dir / "models"
        os.makedirs(model_dir, exist_ok=True)
        
        if save_format.lower() == "pytorch":
            model_path = model_dir / f"{name}.pt"
            torch.save(model.state_dict(), model_path)
            
            # Log to MLflow
            mlflow.pytorch.log_model(model, f"models/{name}")
            
        elif save_format.lower() == "onnx":
            model_path = model_dir / f"{name}.onnx"
            # Requires dummy input for tracing
            dummy_input = torch.randn(1, 3, 256, 256)
            torch.onnx.export(model, dummy_input, model_path)
            
            # Log to MLflow
            mlflow.log_artifact(str(model_path), f"models/{name}")
            
        else:
            raise ValueError(f"Unsupported save format: {save_format}")
    
    def log_artifact(self, local_path: Union[str, Path], artifact_path: Optional[str] = None) -> None:
        """
        Log artifact to MLflow.
        
        Args:
            local_path: Path to artifact file
            artifact_path: Path within artifact directory
        """
        mlflow.log_artifact(str(local_path), artifact_path)
    
    def create_report(
        self, 
        metrics: Optional[List[str]] = None,
        images: Optional[List[str]] = None,
        include_config: bool = True,
        output_format: str = "html"
    ) -> Path:
        """
        Create and save an experiment report.
        
        Args:
            metrics: List of metrics to include
            images: List of image names to include
            include_config: Whether to include configuration
            output_format: Report format (html or md)
            
        Returns:
            Path to the generated report
        """
        # Generate report content
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        if output_format == "html":
            report_path = self.output_dir / "report.html"
            
            # Generate HTML report
            # This is a simple template - customize as needed
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Experiment Report: {self.run_name}</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    h1, h2, h3 {{ color: #333; }}
                    pre {{ background-color: #f5f5f5; padding: 10px; border-radius: 5px; }}
                    .metric {{ margin-bottom: 20px; }}
                    .config {{ margin-bottom: 20px; }}
                    .images {{ display: flex; flex-wrap: wrap; }}
                    .image-container {{ margin: 10px; }}
                </style>
            </head>
            <body>
                <h1>Experiment Report: {self.run_name}</h1>
                <p>Generated on: {timestamp}</p>
                <p>Run ID: {self.run_id}</p>
                <p>Experiment: {self.experiment_name}</p>
                
                <h2>Metrics</h2>
                <div class="metrics">
                    <!-- Metrics will be added here -->
                </div>
                
                <h2>Configuration</h2>
                <div class="config">
                    <pre>{json.dumps(OmegaConf.to_container(self.config, resolve=True), indent=2)}</pre>
                </div>
                
                <h2>Images</h2>
                <div class="images">
                    <!-- Images will be added here -->
                </div>
            </body>
            </html>
            """
            
            with open(report_path, "w") as f:
                f.write(html_content)
            
        else:  # markdown
            report_path = self.output_dir / "report.md"
            
            # Generate Markdown report
            md_content = f"""
            # Experiment Report: {self.run_name}
            
            Generated on: {timestamp}
            
            Run ID: {self.run_id}
            
            Experiment: {self.experiment_name}
            
            ## Metrics
            
            <!-- Metrics will be added here -->
            
            ## Configuration
            
            ```json
            {json.dumps(OmegaConf.to_container(self.config, resolve=True), indent=2)}
            ```
            
            ## Images
            
            <!-- Images will be added here -->
            """
            
            with open(report_path, "w") as f:
                f.write(md_content)
        
        # Log report as artifact
        mlflow.log_artifact(str(report_path))
        
        return report_path
    
    def end_run(self) -> None:
        """End the MLflow run and close TensorBoard."""
        self.tensorboard.close()
        mlflow.end_run()
    
    def _save_config(self) -> None:
        """Save configuration to output directory and MLflow."""
        # Save as YAML
        config_path = self.output_dir / "config.yaml"
        OmegaConf.save(self.config, str(config_path))
        
        # Log to MLflow
        mlflow.log_artifact(str(config_path))
    
    @staticmethod
    def _flatten_dict(d: Dict[str, Any], parent_key: str = '', sep: str = '.') -> Dict[str, Any]:
        """
        Flatten a nested dictionary.
        
        Args:
            d: Dictionary to flatten
            parent_key: Parent key for recursion
            sep: Separator between keys
            
        Returns:
            Flattened dictionary
        """
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(ExperimentTracker._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)


def load_run(run_id: str) -> Dict[str, Any]:
    """
    Load a previous run by ID.
    
    Args:
        run_id: MLflow run ID
        
    Returns:
        Dictionary with run information
    """
    run = mlflow.get_run(run_id)
    return {
        "run_id": run_id,
        "run_name": run.data.tags.get("mlflow.runName", "unknown"),
        "experiment_id": run.info.experiment_id,
        "experiment_name": mlflow.get_experiment(run.info.experiment_id).name,
        "start_time": run.info.start_time,
        "end_time": run.info.end_time,
        "status": run.info.status,
        "artifacts_uri": run.info.artifact_uri,
        "metrics": run.data.metrics,
        "params": run.data.params,
        "tags": run.data.tags
    }


def create_experiment_archive(
    run_id: str, 
    output_path: Optional[Union[str, Path]] = None,
    include_models: bool = True
) -> Path:
    """
    Create a zip archive of experiment results for sharing.
    
    Args:
        run_id: MLflow run ID
        output_path: Path to save the archive (defaults to current directory)
        include_models: Whether to include model files
        
    Returns:
        Path to the created archive
    """
    import shutil
    import tempfile
    
    # Get run information
    run_info = load_run(run_id)
    run_name = run_info["run_name"]
    
    # Create temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir)
        
        # Download artifacts
        artifact_path = temp_dir_path / "artifacts"
        os.makedirs(artifact_path, exist_ok=True)
        
        client = mlflow.tracking.MlflowClient()
        artifacts = client.list_artifacts(run_id)
        
        for artifact in artifacts:
            # Skip model files if not requested
            if not include_models and artifact.path.startswith("models/"):
                continue
                
            # Download artifact
            local_path = client.download_artifacts(run_id, artifact.path, str(artifact_path))
            print(f"Downloaded {artifact.path} to {local_path}")
        
        # Get metrics
        metrics_path = temp_dir_path / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(run_info["metrics"], f, indent=2)
        
        # Get parameters
        params_path = temp_dir_path / "params.json"
        with open(params_path, "w") as f:
            json.dump(run_info["params"], f, indent=2)
        
        # Create run info
        run_info_path = temp_dir_path / "run_info.json"
        with open(run_info_path, "w") as f:
            json.dump(run_info, f, indent=2)
        
        # Create archive
        if output_path is None:
            output_path = Path(f"{run_name}_{int(time.time())}.zip")
        else:
            output_path = Path(output_path)
            
        shutil.make_archive(
            str(output_path.with_suffix('')),
            'zip',
            root_dir=str(temp_dir_path)
        )
        
        return output_path.with_suffix('.zip')
