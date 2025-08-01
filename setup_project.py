# setup_project.py
import os
from pathlib import Path
import subprocess
import sys

def create_directory_structure():
    """Create all required directories"""
    directories = [
        "configs",
        "src", 
        "data/raw",
        "data/processed",
        "models",
        "metrics",
        "plots",
        "experiments",
        "mlruns",
        "mlflow-artifacts"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created directory: {directory}")

def create_init_files():
    """Create __init__.py files"""
    init_dirs = ["configs", "src"]
    
    for directory in init_dirs:
        init_file = Path(directory) / "__init__.py"
        init_file.touch()
        print(f"✅ Created {init_file}")

def install_requirements():
    """Install required packages"""
    requirements = [
        "mlflow>=2.10.0",
        "scikit-learn>=1.3.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "jupyter>=1.0.0",
        "pyyaml>=6.0",
        "dvc[all]>=3.0.0"
    ]
    
    print("Installing requirements...")
    for req in requirements:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", req])
            print(f"✅ Installed {req}")
        except subprocess.CalledProcessError:
            print(f"❌ Failed to install {req}")

def main():
    """Setup the project"""
    print("🚀 Setting up MLOps Day 1 Project")
    
    create_directory_structure()
    create_init_files()
    
    print("\n📦 Project structure created successfully!")
    print("Now copy the fixed code files and run:")
    print("python src/data_pipeline.py")
    print("python src/train_multiple_models.py")

if __name__ == "__main__":
    main()