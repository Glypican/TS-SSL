"""Tumor Highlighter - A toolkit for highlighting tumors in H&E WSIs using task-specific self-supervised learning."""

__version__ = "0.1.0"
__author__ = "Glypican"
__description__ = "A toolkit for highlighting tumors in H&E WSIs using task-specific self-supervised learning"

import logging

# Set up logger
logger = logging.getLogger("tumor_highlighter")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
