"""Command Line Interface for Tumor Highlighter."""

import argparse
import logging
import sys
from pathlib import Path

from tumor_highlighter import __version__, logger
from tumor_highlighter.models.model_registry import ModelRegistry
from tumor_highlighter.models.ts_ssl.trainer import TSSSLTrainer
from tumor_highlighter.features.garden import FeatureGarden
from tumor_highlighter.features.extractor import FeatureExtractor
from tumor_highlighter.data.tissue_mask import HSVTissueMaskGenerator
from tumor_highlighter.utils.visualization import create_heatmap, overlay_heatmap


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description=f"Tumor Highlighter {__version__} - A toolkit for highlighting tumors in H&E WSIs "
                    f"using task-specific self-supervised learning."
    )
    parser.add_argument(
        "--version", "-v", action="version", version=f"Tumor Highlighter {__version__}"
    )
    parser.add_argument(
        "--debug", action="store_true", help="Enable debug logging"
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Create a parser for the 'train' command
    train_parser = subparsers.add_parser("train", help="Train a TS-SSL model")
    train_parser.add_argument(
        "--data", type=str, required=True, help="Path to the training data directory"
    )
    train_parser.add_argument(
        "--model-name", type=str, required=True, help="Name for the trained model"
    )
    train_parser.add_argument(
        "--batch-size", type=int, default=16, help="Batch size for training"
    )
    train_parser.add_argument(
        "--epochs", type=int, default=50, help="Number of epochs for training"
    )
    train_parser.add_argument(
        "--learning-rate", type=float, default=0.001, help="Learning rate for training"
    )
    train_parser.add_argument(
        "--output-dir", type=str, default="./models", help="Directory to save the trained model"
    )
    train_parser.add_argument(
        "--input-shape", type=int, nargs=2, default=[256, 256], help="Input shape for the model (height, width)"
    )
    train_parser.add_argument(
        "--subsample", type=float, default=0.1, help="Fraction of patches to use for training (0.0-1.0)"
    )
    train_parser.add_argument(
        "--track-mlflow", action="store_true", help="Track experiment with MLflow"
    )

    # Create a parser for the 'extract' command
    extract_parser = subparsers.add_parser("extract", help="Extract features from WSIs")
    extract_parser.add_argument(
        "--wsi", type=str, required=True, help="Path to the WSI file or directory of WSIs"
    )
    extract_parser.add_argument(
        "--model", type=str, required=True, help="Name of the model to use for feature extraction"
    )
    extract_parser.add_argument(
        "--output-dir", type=str, default="./features", help="Directory to save the extracted features"
    )
    extract_parser.add_argument(
        "--batch-size", type=int, default=16, help="Batch size for feature extraction"
    )
    extract_parser.add_argument(
        "--resolution", type=float, default=1.0, help="Resolution for feature extraction"
    )
    extract_parser.add_argument(
        "--units", type=str, default="mpp", choices=["mpp", "power", "level"], help="Units for resolution"
    )
    extract_parser.add_argument(
        "--mask", type=str, help="Path to tissue mask or 'auto' to generate automatically"
    )

    # Create a parser for the 'highlight' command
    highlight_parser = subparsers.add_parser("highlight", help="Create tumor highlight heatmap")
    highlight_parser.add_argument(
        "--wsi", type=str, required=True, help="Path to the WSI file"
    )
    highlight_parser.add_argument(
        "--features", type=str, required=True, help="Path to the extracted features directory"
    )
    highlight_parser.add_argument(
        "--output", type=str, required=True, help="Path to save the heatmap"
    )
    highlight_parser.add_argument(
        "--threshold", type=float, default=0.5, help="Threshold for tumor highlighting"
    )
    highlight_parser.add_argument(
        "--resolution", type=float, default=1.0, help="Resolution for the heatmap"
    )
    highlight_parser.add_argument(
        "--units", type=str, default="mpp", choices=["mpp", "power", "level"], help="Units for resolution"
    )
    highlight_parser.add_argument(
        "--overlay", action="store_true", help="Overlay heatmap on the WSI"
    )

    # Create a parser for the 'register' command
    register_parser = subparsers.add_parser("register", help="Register a model in the model registry")
    register_parser.add_argument(
        "--model-path", type=str, required=True, help="Path to the model file"
    )
    register_parser.add_argument(
        "--model-name", type=str, required=True, help="Name for the registered model"
    )
    register_parser.add_argument(
        "--model-type", type=str, required=True, choices=["ts-ssl", "mil", "custom"], help="Type of the model"
    )
    register_parser.add_argument(
        "--description", type=str, help="Description of the model"
    )

    # Create a parser for the 'mask' command
    mask_parser = subparsers.add_parser("mask", help="Generate tissue mask for WSI")
    mask_parser.add_argument(
        "--wsi", type=str, required=True, help="Path to the WSI file"
    )
    mask_parser.add_argument(
        "--output", type=str, required=True, help="Path to save the mask"
    )
    mask_parser.add_argument(
        "--method", type=str, default="hsv", choices=["hsv", "otsu", "adaptive"], help="Method for tissue masking"
    )
    mask_parser.add_argument(
        "--resolution", type=float, default=1.0, help="Resolution for the mask"
    )
    mask_parser.add_argument(
        "--units", type=str, default="mpp", choices=["mpp", "power", "level"], help="Units for resolution"
    )

    # Create a parser for the 'list' command
    list_parser = subparsers.add_parser("list", help="List registered models")
    list_parser.add_argument(
        "--model-type", type=str, choices=["ts-ssl", "mil", "custom", "all"], default="all", help="Type of models to list"
    )
    list_parser.add_argument(
        "--verbose", action="store_true", help="Show detailed information"
    )

    return parser.parse_args()


def setup_logging(debug=False):
    """Set up logging configuration."""
    level = logging.DEBUG if debug else logging.INFO
    logger.setLevel(level)
    # Clear existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Add new handler
    handler = logging.StreamHandler()
    handler.setLevel(level)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)


def train_model(args):
    """Train a TS-SSL model."""
    logger.info(f"Training TS-SSL model with name: {args.model_name}")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    trainer = TSSSLTrainer(
        model_name=args.model_name,
        input_shape=tuple(args.input_shape),
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        output_dir=output_dir,
        track_mlflow=args.track_mlflow,
    )
    
    trainer.train(
        data_path=args.data,
        epochs=args.epochs,
        subsample=args.subsample,
    )
    
    logger.info(f"Model trained and saved to {output_dir / args.model_name}")


def extract_features(args):
    """Extract features from WSIs using a trained model."""
    logger.info(f"Extracting features from {args.wsi} using model: {args.model}")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    wsi_path = Path(args.wsi)
    wsi_paths = [wsi_path] if wsi_path.is_file() else list(wsi_path.glob("*.svs"))
    
    if not wsi_paths:
        logger.error(f"No WSI files found at {args.wsi}")
        return 1
    
    model_registry = ModelRegistry()
    model = model_registry.get_model(args.model)
    
    if model is None:
        logger.error(f"Model {args.model} not found in registry")
        return 1
    
    feature_extractor = FeatureExtractor(model=model)
    
    for wsi in wsi_paths:
        logger.info(f"Processing {wsi.name}")
        mask_path = None
        if args.mask:
            if args.mask.lower() == "auto":
                mask_generator = HSVTissueMaskGenerator()
                mask_path = output_dir / f"{wsi.stem}_mask.png"
                mask_generator.generate(
                    wsi_path=str(wsi),
                    output_path=str(mask_path),
                    resolution=args.resolution,
                    units=args.units,
                )
            else:
                mask_path = Path(args.mask)
        
        feature_extractor.extract(
            wsi_path=str(wsi),
            output_dir=output_dir / wsi.stem,
            mask_path=str(mask_path) if mask_path else None,
            batch_size=args.batch_size,
            resolution=args.resolution,
            units=args.units,
        )
    
    logger.info(f"Features extracted and saved to {output_dir}")
    return 0


def create_highlight(args):
    """Create tumor highlight heatmap from extracted features."""
    logger.info(f"Creating tumor highlight heatmap for {args.wsi}")
    
    feature_garden = FeatureGarden()
    feature_garden.load(args.features)
    
    heatmap = feature_garden.create_heatmap(
        threshold=args.threshold,
        resolution=args.resolution,
        units=args.units,
    )
    
    if args.overlay:
        overlay_heatmap(
            wsi_path=args.wsi,
            heatmap=heatmap,
            output_path=args.output,
            resolution=args.resolution,
            units=args.units,
        )
    else:
        create_heatmap(
            heatmap=heatmap,
            output_path=args.output,
        )
    
    logger.info(f"Heatmap created and saved to {args.output}")
    return 0


def register_model(args):
    """Register a model in the model registry."""
    logger.info(f"Registering model: {args.model_name}")
    
    model_registry = ModelRegistry()
    success = model_registry.register_model(
        model_path=args.model_path,
        model_name=args.model_name,
        model_type=args.model_type,
        description=args.description,
    )
    
    if success:
        logger.info(f"Model {args.model_name} registered successfully")
        return 0
    else:
        logger.error(f"Failed to register model {args.model_name}")
        return 1


def generate_mask(args):
    """Generate tissue mask for WSI."""
    logger.info(f"Generating tissue mask for {args.wsi}")
    
    if args.method == "hsv":
        mask_generator = HSVTissueMaskGenerator()
        mask_generator.generate(
            wsi_path=args.wsi,
            output_path=args.output,
            resolution=args.resolution,
            units=args.units,
        )
    else:
        logger.error(f"Mask method {args.method} not implemented")
        return 1
    
    logger.info(f"Mask generated and saved to {args.output}")
    return 0


def list_models(args):
    """List registered models."""
    logger.info("Listing registered models")
    
    model_registry = ModelRegistry()
    models = model_registry.list_models(model_type=args.model_type if args.model_type != "all" else None)
    
    if not models:
        logger.info("No models found")
        return 0
    
    for model in models:
        if args.verbose:
            logger.info(f"Name: {model.name}")
            logger.info(f"Type: {model.type}")
            logger.info(f"Path: {model.path}")
            logger.info(f"Description: {model.description}")
            logger.info("---")
        else:
            logger.info(f"{model.name} ({model.type})")
    
    return 0


def main():
    """Main entry point for the CLI."""
    args = parse_args()
    setup_logging(args.debug)
    
    if args.command == "train":
        return train_model(args)
    elif args.command == "extract":
        return extract_features(args)
    elif args.command == "highlight":
        return create_highlight(args)
    elif args.command == "register":
        return register_model(args)
    elif args.command == "mask":
        return generate_mask(args)
    elif args.command == "list":
        return list_models(args)
    else:
        logger.error("No command specified")
        return 1


if __name__ == "__main__":
    sys.exit(main())
