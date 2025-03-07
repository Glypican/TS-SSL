from setuptools import setup, find_packages

setup(
    name="tumor_highlighter",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "torch",
        "torchvision",
        "tiatoolbox",
        "pandas",
        "matplotlib",
        "pillow",
        "scikit-learn",
        "scipy",
        "tqdm",
        "mlflow",
        "omegaconf",
        "hydra-core",
        "tensorboard",
        "opencv-python",
    ],
    python_requires=">=3.7",
    author="",
    author_email="",
    description="A system for highlighting tumor regions in H&E whole slide images",
    keywords="digital pathology, tumor detection, deep learning",
    extras_require={
        "dev": [
            "pytest",
            "black",
            "flake8",
            "isort",
        ],
    },
)
