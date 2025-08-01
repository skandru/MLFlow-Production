# src/wandb_integration.py
import wandb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

class WandBIntegration:
    def __init__(self, project_name="mlops-day1", entity=None):
        self.project_name = project_name
        self.entity = entity
        
    def init_run(self, config, run_name=None, tags=None):
        """Initialize W&B run"""
        wandb.init(
            project=self.project_name,
            entity=self.entity,
            config=config,
            name=run_name,
            tags=tags,
            reinit=True
        )
        
    def log_metrics(self, metrics_dict, step=None):
        """Log metrics to W&B"""
        wandb.log(metrics_dict, step=step)
        
    def log_model(self, model_path, model_name):
        """Log model as W&B artifact"""
        artifact = wandb.Artifact(model_name, type="model")
        artifact.add_file(model_path)
        wandb.log_artifact(artifact)
        
    def log_dataset(self, dataset_path, dataset_name):
        """Log dataset as W&B artifact"""
        artifact = wandb.Artifact(dataset_name, type="dataset")
        artifact.add_file(dataset_path)
        wandb.log_artifact(artifact)
        
    def log_confusion_matrix(self, y_true, y_pred, class_names):
        """Log confusion matrix visualization"""
        wandb.log({
            "confusion_matrix": wandb.plot.confusion_matrix(
                probs=None,
                y_true=y_true,
                preds=y_pred,
                class_names=class_names
            )
        })
        
    def finish_run(self):
        """Finish W&B run"""
        wandb.finish()