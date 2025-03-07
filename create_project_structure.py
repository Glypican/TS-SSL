#!/usr/bin/env python3
"""
Script to create the basic project structure for tumor_highlighter.
Run this once to set up the directory structure.
"""

import os

def create_directory(path):
    """Create directory if it doesn't exist."""
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Created directory: {path}")
    else:
        print(f"Directory already exists: {path}")

def create_file(path, content=""):
    """Create file with optional content if it doesn't exist."""
    if not os.path.exists(path):
        with open(path, "w") as f:
            f.write(content)
        print(f"Created file: {path}")
    else:
        print(f"File already exists: {path}")

# Main project directories
create_directory("tumor_highlighter")
create_directory("tumor_highlighter/data")
create_directory("tumor_highlighter/models")
create_directory("tumor_highlighter/models/ts_ssl")
create_directory("tumor_highlighter/features")
create_directory("tumor_highlighter/training")
create_directory("tumor_highlighter/training/losses")
create_directory("tumor_highlighter/utils")
create_directory("config")
create_directory("notebooks")
create_directory("scripts")

# Create __init__.py files for Python packages
create_file("tumor_highlighter/__init__.py", "__version__ = '0.1.0'\n")
create_file("tumor_highlighter/data/__init__.py")
create_file("tumor_highlighter/models/__init__.py")
create_file("tumor_highlighter/models/ts_ssl/__init__.py")
create_file("tumor_highlighter/features/__init__.py")
create_file("tumor_highlighter/training/__init__.py")
create_file("tumor_highlighter/training/losses/__init__.py")
create_file("tumor_highlighter/utils/__init__.py")

print("Project structure created successfully!")
