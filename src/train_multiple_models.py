# src/train_multiple_models.py
import yaml
import json
from pathlib import Path
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
import mlflow
import sys
import os

# Add current directory to Python path
sys.path.append(os.getcwd())

# Import config - handle if not available
try:
    from configs.mlflow_config import MLFlowConfig
except ImportError:
    print("Warning: MLFlow config not found, using defaults")
    import mlflow
    class MLFlowConfig:
        @staticmethod
        def setup_mlflow():
            mlflow.set_tracking_uri("file:./mlruns")
            try:
                experiment_id = mlflow.create_experiment("ml-pipeline-experiments")
            except mlflow.exceptions.MlflowException:
                experiment_id = mlflow.get_experiment_by_name("ml-pipeline-experiments").experiment_id
            mlflow.set_experiment("ml-pipeline-experiments")
            return experiment_id

def load_params():
    """Load parameters from params.yaml"""
    if not Path("params.yaml").exists():
        # Create default params if file doesn't exist
        default_params = {
            "models": {
                "random_forest": {
                    "n_estimators": 100,
                    "max_depth": 7,
                    "min_samples_split": 5,
                    "min_samples_leaf": 2,
                    "random_state": 42
                },
                "logistic_regression": {
                    "C": 1.0,
                    "penalty": "l2",
                    "solver": "liblinear",
                    "random_state": 42,
                    "max_iter": 1000
                }
            }
        }
        
        with open("params.yaml", "w") as f:
            yaml.dump(default_params, f, indent=2)
        print("Created default params.yaml")
    
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)
    return params

def load_processed_data():
    """Load processed data"""
    try:
        X_train = pd.read_csv("data/processed/X_train.csv")
        X_test = pd.read_csv("data/processed/X_test.csv")
        y_train = pd.read_csv("data/processed/y_train.csv")['target']
        y_test = pd.read_csv("data/processed/y_test.csv")['target']
        
        return X_train, X_test, y_train, y_test
    except FileNotFoundError as e:
        print(f"Error loading processed data: {e}")
        print("Running data pipeline first...")
        
        # Run data pipeline
        from data_pipeline import DataPipeline
        pipeline = DataPipeline("breast_cancer")
        X_train, X_test, y_train, y_test = pipeline.process_data()
        
        return X_train, X_test, y_train, y_test

def scale_data(X_train, X_test):
    """Scale the data"""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Ensure models directory exists
    Path("models").mkdir(exist_ok=True)
    joblib.dump(scaler, "models/scaler.joblib")
    
    return X_train_scaled, X_test_scaled, scaler

def calculate_metrics(y_true, y_pred, prefix=""):
    """Calculate comprehensive metrics"""
    metrics = {
        f"{prefix}accuracy": float(accuracy_score(y_true, y_pred)),
        f"{prefix}precision": float(precision_score(y_true, y_pred, average='weighted')),
        f"{prefix}recall": float(recall_score(y_true, y_pred, average='weighted')),
        f"{prefix}f1": float(f1_score(y_true, y_pred, average='weighted'))
    }
    return metrics

def train_model(model_name, model_params, X_train, X_test, y_train, y_test):
    """Train a single model with MLFlow tracking"""
    print(f"Training {model_name}...")
    
    # Initialize MLFlow
    experiment_id = MLFlowConfig.setup_mlflow()
    
    with mlflow.start_run(run_name=f"DVC_{model_name}"):
        # Initialize model
        if model_name == "random_forest":
            model = RandomForestClassifier(**model_params)
        elif model_name == "logistic_regression":
            model = LogisticRegression(**model_params)
        elif model_name == "svm":
            model = SVC(**model_params)
        else:
            raise ValueError(f"Unknown model type: {model_name}")
        
        # Train model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        # Calculate metrics
        train_metrics = calculate_metrics(y_train, y_train_pred, "train_")
        test_metrics = calculate_metrics(y_test, y_test_pred, "test_")
        
        all_metrics = {**train_metrics, **test_metrics}
        
        # Log parameters and metrics to MLFlow
        mlflow.log_params({"model_type": model_name, **model_params})
        mlflow.log_metrics(all_metrics)
        
        # Save model
        model_path = f"models/{model_name}_model.joblib"
        joblib.dump(model, model_path)
        
        try:
            mlflow.sklearn.log_model(model, f"{model_name}_model")
        except Exception as e:
            print(f"Warning: Could not log model to MLFlow: {e}")
        
        print(f"Completed {model_name}: Test accuracy = {all_metrics['test_accuracy']:.4f}")
        
        return model, all_metrics

def main():
    """Main training pipeline"""
    print("Starting training pipeline...")
    
    # Load parameters
    params = load_params()
    
    # Load and scale data
    X_train, X_test, y_train, y_test = load_processed_data()
    X_train_scaled, X_test_scaled, scaler = scale_data(X_train, X_test)
    
    # Train all models
    all_metrics = {}
    
    for model_name, model_params in params["models"].items():
        try:
            model, metrics = train_model(
                model_name, model_params, 
                X_train_scaled, X_test_scaled, 
                y_train, y_test
            )
            all_metrics[model_name] = metrics
        except Exception as e:
            print(f"Error training {model_name}: {e}")
            continue
    
    # Save metrics
    Path("metrics").mkdir(exist_ok=True)
    with open("metrics/model_metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)
    
    print("Training completed!")
    print(f"Results saved to metrics/model_metrics.json")

if __name__ == "__main__":
    main()