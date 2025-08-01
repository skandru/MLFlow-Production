# run_pipeline.py
import subprocess
import sys
from pathlib import Path

def run_command(command, description):
    """Run command and handle errors"""
    print(f"\n{'='*50}")
    print(f"Running: {description}")
    print(f"Command: {command}")
    print('='*50)
    
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✅ SUCCESS")
        if result.stdout:
            print("Output:", result.stdout)
    else:
        print("❌ ERROR")
        print("Error:", result.stderr)
        return False
    
    return True

def main():
    """Run complete MLOps pipeline"""
    print("🚀 Starting MLOps Day 1 Pipeline")
    
    # Ensure directories exist
    Path("data/raw").mkdir(parents=True, exist_ok=True)
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    Path("models").mkdir(exist_ok=True)
    Path("metrics").mkdir(exist_ok=True)
    Path("plots").mkdir(exist_ok=True)
    Path("experiments").mkdir(exist_ok=True)
    
    steps = [
        ("python src/data_pipeline.py", "Preparing raw and processed data"),
        ("dvc repro prepare_data", "Running DVC data preparation stage"),
        ("python src/train_multiple_models.py", "Training multiple models with MLFlow"),
        ("dvc repro train_models", "Running DVC training stage"),
        ("python src/evaluate_models.py", "Evaluating and comparing models"),
        ("dvc repro evaluate_models", "Running DVC evaluation stage"),
        ("dvc metrics show", "Showing DVC metrics"),
        ("dvc plots show", "Generating DVC plots")
    ]
    
    failed_steps = []
    
    for command, description in steps:
        if not run_command(command, description):
            failed_steps.append(description)
    
    print(f"\n🎉 Pipeline Execution Complete!")
    
    if failed_steps:
        print(f"⚠️  Some steps failed: {', '.join(failed_steps)}")
    else:
        print("✅ All steps completed successfully!")
    
    print(f"\n📊 Results Summary:")
    print(f"- MLFlow UI: http://localhost:5000")
    print(f"- Model metrics: metrics/model_metrics.json")
    print(f"- Evaluation results: metrics/evaluation_metrics.json")
    print(f"- Comparison plots: plots/model_comparison.png")
    print(f"- DVC pipeline: dvc dag")

if __name__ == "__main__":
    main()