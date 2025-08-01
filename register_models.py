# register_models.py
import mlflow
from mlflow.tracking import MlflowClient
import json
from pathlib import Path

def setup_model_registry():
    """Register all trained models in MLFlow Model Registry"""
    
    # Initialize MLFlow
    mlflow.set_tracking_uri("file:./mlruns")
    client = MlflowClient()
    
    # Get the experiment
    experiment_name = "ml-pipeline-experiments"
    experiment = client.get_experiment_by_name(experiment_name)
    
    if not experiment:
        print(f"❌ Experiment '{experiment_name}' not found!")
        return
    
    print(f"📊 Found experiment: {experiment_name}")
    
    # Get all runs from the experiment
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"]
    )
    
    print(f"🔍 Found {len(runs)} runs")
    
    # Register models from the latest runs
    model_registry = {}
    
    for run in runs:
        run_name = run.data.tags.get("mlflow.runName", "Unknown")
        model_type = run.data.params.get("model_type", "Unknown")
        test_accuracy = run.data.metrics.get("test_accuracy", 0)
        
        print(f"\n🤖 Processing run: {run_name}")
        print(f"   Model Type: {model_type}")
        print(f"   Test Accuracy: {test_accuracy:.4f}")
        
        # Register model if it has a recognizable type
        if model_type in ["random_forest", "logistic_regression", "svm"]:
            model_name = f"breast_cancer_{model_type}"
            
            try:
                # Register the model
                model_uri = f"runs:/{run.info.run_id}/{model_type}_model"
                
                print(f"   📝 Registering model: {model_name}")
                
                model_version = mlflow.register_model(
                    model_uri=model_uri,
                    name=model_name,
                    tags={
                        "model_type": model_type,
                        "test_accuracy": str(test_accuracy),
                        "dataset": "breast_cancer"
                    }
                )
                
                print(f"   ✅ Registered {model_name} version {model_version.version}")
                
                # Store for comparison
                model_registry[model_name] = {
                    "version": model_version.version,
                    "accuracy": test_accuracy,
                    "run_id": run.info.run_id
                }
                
            except Exception as e:
                print(f"   ❌ Failed to register {model_name}: {e}")
    
    # Promote best model to Production
    if model_registry:
        promote_best_model(client, model_registry)
    
    return model_registry

def promote_best_model(client, model_registry):
    """Promote the best performing model to Production stage"""
    
    # Find best model by accuracy
    best_model = max(model_registry.items(), key=lambda x: x[1]["accuracy"])
    best_model_name, best_model_info = best_model
    
    print(f"\n🏆 Best Model: {best_model_name}")
    print(f"   Accuracy: {best_model_info['accuracy']:.4f}")
    print(f"   Version: {best_model_info['version']}")
    
    try:
        # Transition to Production
        client.transition_model_version_stage(
            name=best_model_name,
            version=best_model_info['version'],
            stage="Production",
            archive_existing_versions=False
        )
        
        print(f"   ✅ Promoted to Production stage")
        
        # Add description
        client.update_model_version(
            name=best_model_name,
            version=best_model_info['version'],
            description=f"Best performing model with {best_model_info['accuracy']:.4f} test accuracy"
        )
        
    except Exception as e:
        print(f"   ❌ Failed to promote model: {e}")

def list_registered_models():
    """List all registered models"""
    client = MlflowClient()
    
    print("\n📋 Registered Models:")
    print("=" * 60)
    
    registered_models = client.search_registered_models()
    
    if not registered_models:
        print("No models registered yet.")
        return
    
    for model in registered_models:
        print(f"\n🤖 Model: {model.name}")
        
        # Get latest versions
        latest_versions = client.get_latest_versions(model.name, stages=["None", "Staging", "Production"])
        
        for version in latest_versions:
            print(f"   Version {version.version} ({version.current_stage})")
            print(f"   Created: {version.creation_timestamp}")
            if version.description:
                print(f"   Description: {version.description}")

def main():
    """Main function to register and list models"""
    print("🚀 Setting up MLFlow Model Registry")
    
    # Register models
    model_registry = setup_model_registry()
    
    # List registered models
    list_registered_models()
    
    print(f"\n✅ Model Registry setup complete!")
    print(f"🌐 View models at: http://localhost:5000")
    print(f"📁 Navigate to 'Models' tab in MLFlow UI")

if __name__ == "__main__":
    main()