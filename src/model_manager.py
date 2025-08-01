# src/model_manager.py
import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd
from datetime import datetime
import json

class ModelManager:
    def __init__(self):
        self.client = MlflowClient()
        
    def register_model(self, run_id, model_name, stage="Staging"):
        """Register model in MLFlow Model Registry"""
        model_uri = f"runs:/{run_id}/model"
        
        # Register model
        model_version = mlflow.register_model(
            model_uri=model_uri,
            name=model_name
        )
        
        # Transition to stage
        self.client.transition_model_version_stage(
            name=model_name,
            version=model_version.version,
            stage=stage
        )
        
        return model_version
    
    def compare_models(self, experiment_name, metric="test_accuracy"):
        """Compare models from experiment"""
        experiment = self.client.get_experiment_by_name(experiment_name)
        runs = self.client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=[f"metrics.{metric} DESC"]
        )
        
        comparison_data = []
        for run in runs:
            comparison_data.append({
                "run_id": run.info.run_id,
                "run_name": run.data.tags.get("mlflow.runName", "Unknown"),
                "model_type": run.data.params.get("model_type", "Unknown"),
                metric: run.data.metrics.get(metric, 0),
                "start_time": datetime.fromtimestamp(run.info.start_time / 1000)
            })
        
        return pd.DataFrame(comparison_data)
    
    def promote_best_model(self, experiment_name, model_name, metric="test_accuracy"):
        """Promote best performing model to Production"""
        comparison_df = self.compare_models(experiment_name, metric)
        best_run_id = comparison_df.iloc[0]["run_id"]
        
        # Register and promote to production
        model_version = self.register_model(best_run_id, model_name, "Production")
        
        print(f"Promoted model version {model_version.version} to Production")
        print(f"Best {metric}: {comparison_df.iloc[0][metric]:.4f}")
        
        return model_version