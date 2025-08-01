# src/experiment_tracker.py
import mlflow
import mlflow.sklearn
import mlflow.pytorch
from mlflow.models import infer_signature
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from configs.mlflow_config import MLFlowConfig

class ExperimentTracker:
    def __init__(self):
        self.experiment_id = MLFlowConfig.setup_mlflow()
        self.run_id = None
        
    def start_run(self, run_name=None, description=None):
        """Start a new MLFlow run"""
        self.run = mlflow.start_run(run_name=run_name, description=description)
        self.run_id = self.run.info.run_id
        return self.run
    
    def log_params(self, params_dict):
        """Log hyperparameters"""
        mlflow.log_params(params_dict)
    
    def log_metrics(self, metrics_dict, step=None):
        """Log metrics with optional step"""
        for metric_name, value in metrics_dict.items():
            mlflow.log_metric(metric_name, value, step=step)
    
    def log_model(self, model, model_name, X_sample=None, y_sample=None):
        """Log model with signature inference"""
        if X_sample is not None:
            signature = infer_signature(X_sample, y_sample)
            mlflow.sklearn.log_model(model, model_name, signature=signature)
        else:
            mlflow.sklearn.log_model(model, model_name)
    
    def log_artifacts(self, artifact_path, local_path):
        """Log artifacts (plots, data, etc.)"""
        mlflow.log_artifacts(local_path, artifact_path)
    
    def end_run(self):
        """End current MLFlow run"""
        mlflow.end_run()