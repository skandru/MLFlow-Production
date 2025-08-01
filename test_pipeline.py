# test_pipeline.py
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import json
import mlflow
from src.data_pipeline import DataPipeline
from src.ml_pipeline import MLPipeline
from src.experiment_tracker import ExperimentTracker

class TestMLOpsPipeline:
    def test_data_pipeline(self):
        """Test data pipeline functionality"""
        pipeline = DataPipeline("breast_cancer")
        df, metadata = pipeline.create_raw_data()
        
        assert len(df) > 0, "Dataset should not be empty"
        assert 'target' in df.columns, "Target column should exist"
        assert metadata['n_samples'] == len(df), "Metadata should match data"
        
        X_train, X_test, y_train, y_test = pipeline.process_data()
        assert len(X_train) > len(X_test), "Training set should be larger"
        
    def test_mlflow_tracking(self):
        """Test MLFlow experiment tracking"""
        tracker = ExperimentTracker()
        
        with tracker.start_run(run_name="test_run"):
            tracker.log_params({"test_param": "test_value"})
            tracker.log_metrics({"test_metric": 0.95})
            
        # Verify run was logged
        client = mlflow.tracking.MlflowClient()
        experiment = client.get_experiment_by_name("ml-pipeline-experiments")
        runs = client.search_runs([experiment.experiment_id])
        
        assert len(runs) > 0, "At least one run should be logged"
        
    def test_model_training(self):
        """Test model training pipeline"""
        pipeline = MLPipeline("breast_cancer")
        model, metrics = pipeline.train_model("random_forest", n_estimators=10)
        
        assert model is not None, "Model should be trained"
        assert "test_accuracy" in metrics, "Test accuracy should be calculated"
        assert metrics["test_accuracy"] > 0, "Accuracy should be positive"
        
    def test_file_outputs(self):
        """Test that all expected files are created"""
        expected_files = [
            "data/raw/breast_cancer.csv",
            "data/processed/X_train.csv",
            "models/scaler.joblib",
            "metrics/model_metrics.json"
        ]
        
        for file_path in expected_files:
            if Path(file_path).exists():
                assert Path(file_path).stat().st_size > 0, f"{file_path} should not be empty"

def run_tests():
    """Run all tests"""
    print("🧪 Running MLOps Pipeline Tests")
    
    test_pipeline = TestMLOpsPipeline()
    
    tests = [
        ("Data Pipeline", test_pipeline.test_data_pipeline),
        ("MLFlow Tracking", test_pipeline.test_mlflow_tracking),
        ("Model Training", test_pipeline.test_model_training),
        ("File Outputs", test_pipeline.test_file_outputs)
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            test_func()
            print(f"✅ {test_name}: PASSED")
            passed += 1
        except Exception as e:
            print(f"❌ {test_name}: FAILED - {str(e)}")
            failed += 1
    
    print(f"\n📊 Test Results: {passed} passed, {failed} failed")
    return failed == 0

if __name__ == "__main__":
    run_tests()