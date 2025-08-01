# src/data_pipeline.py
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer, load_iris
from sklearn.model_selection import train_test_split
import joblib
import json
from pathlib import Path
import sys
import os

# Add current directory to Python path
sys.path.append(os.getcwd())

class DataPipeline:
    def __init__(self, dataset_name="breast_cancer"):
        self.dataset_name = dataset_name
        
    def create_raw_data(self):
        """Create and save raw dataset"""
        print(f"Creating raw data for {self.dataset_name}...")
        
        if self.dataset_name == "breast_cancer":
            data = load_breast_cancer()
        elif self.dataset_name == "iris":
            data = load_iris()
        else:
            raise ValueError(f"Unsupported dataset: {self.dataset_name}")
            
        # Create DataFrames
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['target'] = data.target
        
        # Ensure directories exist
        Path("data/raw").mkdir(parents=True, exist_ok=True)
        
        # Save raw data
        raw_file = f"data/raw/{self.dataset_name}.csv"
        df.to_csv(raw_file, index=False)
        print(f"Saved raw data to {raw_file}")
        
        # Save metadata
        metadata = {
            "dataset_name": self.dataset_name,
            "features": data.feature_names.tolist(),
            "target_names": data.target_names.tolist(),
            "n_samples": len(df),
            "n_features": len(data.feature_names)
        }
        
        metadata_file = f"data/raw/{self.dataset_name}_metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"Saved metadata to {metadata_file}")
            
        return df, metadata
    
    def process_data(self, test_size=0.2, random_state=42):
        """Process and split data"""
        print("Processing data...")
        
        # Load raw data
        raw_file = f"data/raw/{self.dataset_name}.csv"
        if not Path(raw_file).exists():
            print(f"Raw data file {raw_file} not found. Creating it first...")
            self.create_raw_data()
        
        df = pd.read_csv(raw_file)
        
        X = df.drop('target', axis=1)
        y = df['target']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Ensure processed directory exists
        Path("data/processed").mkdir(parents=True, exist_ok=True)
        
        # Save processed data
        X_train.to_csv("data/processed/X_train.csv", index=False)
        X_test.to_csv("data/processed/X_test.csv", index=False)
        pd.Series(y_train).to_csv("data/processed/y_train.csv", index=False, header=['target'])
        pd.Series(y_test).to_csv("data/processed/y_test.csv", index=False, header=['target'])
        
        print("Saved processed data files:")
        print("- data/processed/X_train.csv")
        print("- data/processed/X_test.csv") 
        print("- data/processed/y_train.csv")
        print("- data/processed/y_test.csv")
        
        # Save split metadata
        split_info = {
            "test_size": test_size,
            "random_state": random_state,
            "train_samples": len(X_train),
            "test_samples": len(X_test)
        }
        
        with open("data/processed/split_info.json", "w") as f:
            json.dump(split_info, f, indent=2)
            
        return X_train, X_test, y_train, y_test

def main():
    """Main function to run data pipeline"""
    print("Starting Data Pipeline...")
    
    # Create pipeline
    pipeline = DataPipeline("breast_cancer")
    
    # Create raw data
    df, metadata = pipeline.create_raw_data()
    print(f"Created raw dataset with {len(df)} samples")
    
    # Process data
    X_train, X_test, y_train, y_test = pipeline.process_data()
    print(f"Split data: {len(X_train)} train, {len(X_test)} test samples")
    
    print("Data pipeline completed successfully!")

if __name__ == "__main__":
    main()