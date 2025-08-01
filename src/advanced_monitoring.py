# src/advanced_monitoring.py
import wandb
import mlflow
from ml_pipeline import MLPipeline
from wandb_integration import WandBIntegration
import time
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

class AdvancedMonitoring:
    def __init__(self, dataset_name="breast_cancer"):
        self.pipeline = MLPipeline(dataset_name)
        self.wandb_integration = WandBIntegration()
        
    def run_comprehensive_experiment(self):
        """Run multiple models with both MLFlow and W&B tracking"""
        models_to_test = [
            ("random_forest", RandomForestClassifier, {
                "n_estimators": [50, 100, 200],
                "max_depth": [3, 5, 7]
            }),
            ("logistic_regression", LogisticRegression, {
                "C": [0.1, 1, 10],
                "penalty": ["l1", "l2"]
            }),
            ("svm", SVC, {
                "C": [0.1, 1, 10],
                "kernel": ["linear", "rbf"]
            })
        ]
        
        results = []
        
        for model_name, model_class, param_combinations in models_to_test:
            for param_set in self.generate_param_combinations(param_combinations):
                result = self.run_single_experiment(model_name, model_class, param_set)
                results.append(result)
                
        return results
    
    def generate_param_combinations(self, param_dict):
        """Generate parameter combinations for testing"""
        from itertools import product
        
        keys = param_dict.keys()
        values = param_dict.values()
        combinations = []
        
        for combination in product(*values):
            combinations.append(dict(zip(keys, combination)))
            
        return combinations[:6]  # Limit to 6 combinations per model
    
    def run_single_experiment(self, model_name, model_class, params):
        """Run single experiment with dual tracking"""
        run_name = f"{model_name}_{hash(str(params)) % 10000}"
        
        # Start both MLFlow and W&B runs
        mlflow_run = self.pipeline.tracker.start_run(run_name=run_name)
        
        config = {
            "model_type": model_name,
            "dataset": self.pipeline.dataset_name,
            **params
        }
        
        self.wandb_integration.init_run(
            config=config,
            run_name=run_name,
            tags=[model_name, self.pipeline.dataset_name]
        )
        
        try:
            # Prepare data
            data_info = self.pipeline.prepare_data()
            
            # Log parameters to both platforms
            self.pipeline.tracker.log_params(config)
            
            # Train model
            model = model_class(**params)
            start_time = time.time()
            model.fit(self.pipeline.X_train_scaled, self.pipeline.y_train)
            training_time = time.time() - start_time
            
            # Make predictions
            y_train_pred = model.predict(self.pipeline.X_train_scaled)
            y_test_pred = model.predict(self.pipeline.X_test_scaled)
            
            # Calculate metrics
            train_metrics = self.pipeline.calculate_metrics(self.pipeline.y_train, y_train_pred, "train")
            test_metrics = self.pipeline.calculate_metrics(self.pipeline.y_test, y_test_pred, "test")
            
            all_metrics = {
                **train_metrics,
                **test_metrics,
                "training_time": training_time
            }
            
            # Log to both platforms
            self.pipeline.tracker.log_metrics(all_metrics)
            self.wandb_integration.log_metrics(all_metrics)
            
            # Log confusion matrix to W&B
            self.wandb_integration.log_confusion_matrix(
                self.pipeline.y_test, 
                y_test_pred, 
                self.pipeline.target_names
            )
            
            return {
                "model_name": model_name,
                "params": params,
                "metrics": all_metrics,
                "run_id": mlflow_run.info.run_id
            }
            
        finally:
            self.pipeline.tracker.end_run()
            self.wandb_integration.finish_run()