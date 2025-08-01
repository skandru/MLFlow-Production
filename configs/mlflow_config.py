# configs/mlflow_config.py
import os
from pathlib import Path

class MLFlowConfig:
    # Local MLFlow setup
    MLFLOW_TRACKING_URI = "file:./mlruns"
    MLFLOW_EXPERIMENT_NAME = "ml-pipeline-experiments"
    MLFLOW_ARTIFACT_ROOT = "./mlflow-artifacts"
    
    # Model registry
    MODEL_REGISTRY_NAME = "ml-models"
    
    @staticmethod
    def setup_mlflow():
        import mlflow
        mlflow.set_tracking_uri(MLFlowConfig.MLFLOW_TRACKING_URI)
        
        # Create experiment if it doesn't exist
        try:
            experiment_id = mlflow.create_experiment(
                MLFlowConfig.MLFLOW_EXPERIMENT_NAME,
                artifact_location=MLFlowConfig.MLFLOW_ARTIFACT_ROOT
            )
        except mlflow.exceptions.MlflowException:
            experiment_id = mlflow.get_experiment_by_name(
                MLFlowConfig.MLFLOW_EXPERIMENT_NAME
            ).experiment_id
        
        mlflow.set_experiment(MLFlowConfig.MLFLOW_EXPERIMENT_NAME)
        return experiment_id