"""
Command-line interface for the tumor_highlighter package.

This module provides a command-line interface for the main functionality
of the tumor_highlighter package, including:
- Extracting patches from WSIs
- Training TS-SSL models
- Extracting features using the Feature Garden
- Training MIL models for tumor highlighting
- Generating tumor heatmaps

Example usage:
    # Extract patches from WSIs
    python -m tumor_highlighter extract-patches --wsi-table data/wsi_table.csv --output-dir data/patches

    # Train TS-SSL model
    python -m tumor_highlighter train-tsssl --patch-dir data/patches --output-dir models/tsssl

    # Extract features
    python -m tumor_highlighter extract-features --wsi-table data/wsi_table.csv --models resnet50 tsssl

    # Train MIL model
    python -m tumor_highlighter train-mil --feature-dir data/features --output-dir models
