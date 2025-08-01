# src/ml_pipeline.py
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer, load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from experiment_tracker import ExperimentTracker
import joblib
import json
from datetime import datetime

class MLPipeline:
    def __init__(self, dataset_name="breast_cancer"):
        self.dataset_name = dataset_name
        self.tracker = ExperimentTracker()
        self.load_data()
        
    def load_data(self):
        """Load and prepare dataset"""
        if self.dataset_name == "breast_cancer":
            data = load_breast_cancer()
        elif self.dataset_name == "iris":
            data = load_iris()
        else:
            raise ValueError("Unsupported dataset")
            
        self.X = pd.DataFrame(data.data, columns=data.feature_names)
        self.y = data.target
        self.target_names = data.target_names
        
    def prepare_data(self, test_size=0.2, random_state=42, scale=True):
        """Prepare training and testing data"""
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state, stratify=self.y
        )
        
        if scale:
            self.scaler = StandardScaler()
            self.X_train_scaled = self.scaler.fit_transform(self.X_train)
            self.X_test_scaled = self.scaler.transform(self.X_test)
        else:
            self.X_train_scaled = self.X_train
            self.X_test_scaled = self.X_test
            
        return {
            "train_size": len(self.X_train),
            "test_size": len(self.X_test),
            "features": len(self.X.columns),
            "classes": len(np.unique(self.y))
        }
    
    def train_model(self, model_type="random_forest", **model_params):
        """Train model with MLFlow tracking"""
        run_name = f"{model_type}_{self.dataset_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        with self.tracker.start_run(run_name=run_name, description=f"Training {model_type} on {self.dataset_name}"):
            # Log dataset info
            data_info = self.prepare_data()
            self.tracker.log_params({
                "dataset": self.dataset_name,
                "model_type": model_type,
                **data_info,
                **model_params
            })
            
            # Initialize model
            if model_type == "random_forest":
                model = RandomForestClassifier(**model_params)
            elif model_type == "logistic_regression":
                model = LogisticRegression(**model_params)
            elif model_type == "svm":
                model = SVC(**model_params)
            else:
                raise ValueError("Unsupported model type")
            
            # Train model
            model.fit(self.X_train_scaled, self.y_train)
            
            # Make predictions
            y_train_pred = model.predict(self.X_train_scaled)
            y_test_pred = model.predict(self.X_test_scaled)
            
            # Calculate metrics
            train_metrics = self.calculate_metrics(self.y_train, y_train_pred, "train")
            test_metrics = self.calculate_metrics(self.y_test, y_test_pred, "test")
            
            # Cross-validation score
            cv_scores = cross_val_score(model, self.X_train_scaled, self.y_train, cv=5)
            cv_metrics = {
                "cv_mean_accuracy": cv_scores.mean(),
                "cv_std_accuracy": cv_scores.std()
            }
            
            # Log all metrics
            self.tracker.log_metrics({**train_metrics, **test_metrics, **cv_metrics})
            
            # Create and log visualizations
            self.create_visualizations(self.y_test, y_test_pred, model_type)
            
            # Log model
            self.tracker.log_model(
                model, 
                f"{model_type}_model",
                self.X_test_scaled[:5],
                y_test_pred[:5]
            )
            
            # Save model locally for DVC
            model_path = f"models/{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.joblib"
            joblib.dump(model, model_path)
            
            return model, {**train_metrics, **test_metrics, **cv_metrics}
    
    def calculate_metrics(self, y_true, y_pred, prefix):
        """Calculate comprehensive metrics"""
        return {
            f"{prefix}_accuracy": accuracy_score(y_true, y_pred),
            f"{prefix}_precision": precision_score(y_true, y_pred, average='weighted'),
            f"{prefix}_recall": recall_score(y_true, y_pred, average='weighted'),
            f"{prefix}_f1": f1_score(y_true, y_pred, average='weighted')
        }
    
    def create_visualizations(self, y_true, y_pred, model_type):
        """Create and save visualizations"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', ax=axes[0], cmap='Blues')
        axes[0].set_title(f'Confusion Matrix - {model_type}')
        axes[0].set_xlabel('Predicted')
        axes[0].set_ylabel('Actual')
        
        # Feature Importance (if available)
        if hasattr(self.model, 'feature_importances_'):
            feature_imp = pd.DataFrame({
                'feature': self.X.columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False).head(10)
            
            axes[1].barh(feature_imp['feature'], feature_imp['importance'])
            axes[1].set_title(f'Top 10 Feature Importances - {model_type}')
        
        plt.tight_layout()
        plot_path = f"experiments/{model_type}_analysis.png"
        plt.savefig(plot_path)
        mlflow.log_artifact(plot_path)
        plt.close()