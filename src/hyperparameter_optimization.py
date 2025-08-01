# src/hyperparameter_optimization.py
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from ml_pipeline import MLPipeline
import mlflow
import numpy as np

class HyperparameterOptimizer:
    def __init__(self, dataset_name="breast_cancer"):
        self.pipeline = MLPipeline(dataset_name)
        
    def run_grid_search(self, model_type="random_forest", param_grid=None, cv=5):
        """Run grid search with MLFlow tracking"""
        if param_grid is None:
            param_grid = self.get_default_param_grid(model_type)
        
        with mlflow.start_run(run_name=f"GridSearch_{model_type}"):
            # Log search parameters
            mlflow.log_params({
                "search_type": "grid_search",
                "model_type": model_type,
                "cv_folds": cv,
                "param_combinations": len(param_grid)
            })
            
            # Prepare data
            data_info = self.pipeline.prepare_data()
            
            # Get base model
            if model_type == "random_forest":
                from sklearn.ensemble import RandomForestClassifier
                base_model = RandomForestClassifier()
            elif model_type == "logistic_regression":
                from sklearn.linear_model import LogisticRegression
                base_model = LogisticRegression()
            
            # Run grid search
            grid_search = GridSearchCV(
                base_model, 
                param_grid, 
                cv=cv, 
                scoring='accuracy',
                n_jobs=-1,
                verbose=1
            )
            
            grid_search.fit(self.pipeline.X_train_scaled, self.pipeline.y_train)
            
            # Log best parameters and score
            mlflow.log_params(grid_search.best_params_)
            mlflow.log_metric("best_cv_score", grid_search.best_score_)
            
            # Log all results
            results_df = pd.DataFrame(grid_search.cv_results_)
            results_df.to_csv("experiments/grid_search_results.csv", index=False)
            mlflow.log_artifact("experiments/grid_search_results.csv")
            
            return grid_search.best_estimator_, grid_search.best_params_
    
    def get_default_param_grid(self, model_type):
        """Default parameter grids for different models"""
        if model_type == "random_forest":
            return {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 7, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        elif model_type == "logistic_regression":
            return {
                'C': [0.1, 1, 10, 100],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga']
            }