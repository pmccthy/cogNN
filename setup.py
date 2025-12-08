from setuptools import setup, find_packages

setup(
    name="cog_nn",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        # Core dependencies are managed via mamba/conda environment.yml
        # Listed here for pip-only installs if needed
        "torch",
        "numpy",
        "matplotlib",
        "gymnasium",
    ],
    python_requires=">=3.8",
)


