# src/wandb_sweeps.py
import wandb
from ml_pipeline import MLPipeline
import yaml

class WandBSweeps:
    def __init__(self):
        self.pipeline = MLPipeline()
        
    def create_sweep_config(self, model_type="random_forest"):
        """Create sweep configuration"""
        if model_type == "random_forest":
            sweep_config = {
                'method': 'bayes',  # or 'grid', 'random'
                'metric': {
                    'name': 'test_accuracy',
                    'goal': 'maximize'
                },
                'parameters': {
                    'n_estimators': {
                        'distribution': 'int_uniform',
                        'min': 50,
                        'max': 300
                    },
                    'max_depth': {
                        'distribution': 'int_uniform',
                        'min': 3,
                        'max': 20
                    },
                    'min_samples_split': {
                        'distribution': 'int_uniform',
                        'min': 2,
                        'max': 20
                    },
                    'min_samples_leaf': {
                        'distribution': 'int_uniform',
                        'min': 1,
                        'max': 10
                    }
                }
            }
        
        return sweep_config
    
    def train_sweep_model(self):
        """Training function for W&B sweep"""
        with wandb.init() as run:
            config = wandb.config
            
            # Prepare data
            self.pipeline.prepare_data()
            
            # Train model with sweep parameters
            model, metrics = self.pipeline.train_model(
                model_type="random_forest",
                n_estimators=config.n_estimators,
                max_depth=config.max_depth,
                min_samples_split=config.min_samples_split,
                min_samples_leaf=config.min_samples_leaf
            )
            
            # Log metrics to W&B
            wandb.log(metrics)
    
    def run_sweep(self, count=20):
        """Run hyperparameter sweep"""
        sweep_config = self.create_sweep_config()
        
        # Initialize sweep
        sweep_id = wandb.sweep(
            sweep_config, 
            project="mlops-day1"
        )
        
        # Run sweep
        wandb.agent(sweep_id, self.train_sweep_model, count=count)
        
        return sweep_id