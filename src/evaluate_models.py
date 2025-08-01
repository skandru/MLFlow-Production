# src/evaluate_models.py
import json
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

def load_models_and_data():
    """Load all trained models and test data"""
    models = {}
    model_files = Path("models").glob("*_model.joblib")
    
    for model_file in model_files:
        model_name = model_file.stem.replace("_model", "")
        models[model_name] = joblib.load(model_file)
    
    # Load test data
    X_test = pd.read_csv("data/processed/X_test.csv")
    y_test = pd.read_csv("data/processed/y_test.csv").iloc[:, 0]
    
    # Load and apply scaler
    scaler = joblib.load("models/scaler.joblib")
    X_test_scaled = scaler.transform(X_test)
    
    return models, X_test_scaled, y_test

def evaluate_all_models(models, X_test, y_test):
    """Evaluate all models and create comparison"""
    evaluation_results = {}
    
    for model_name, model in models.items():
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, average='weighted'),
            "recall": recall_score(y_test, y_pred, average='weighted'),
            "f1": f1_score(y_test, y_pred, average='weighted')
        }
        
        evaluation_results[model_name] = {
            "metrics": metrics,
            "predictions": y_pred.tolist()
        }
    
    return evaluation_results

def create_comparison_plots(evaluation_results, y_test):
    """Create model comparison visualizations"""
    # Extract metrics for comparison
    metrics_df = pd.DataFrame({
        model_name: results["metrics"] 
        for model_name, results in evaluation_results.items()
    }).T
    
    # Create comparison plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Model Performance Comparison', fontsize=16)
    
    # Metrics comparison
    metrics_df.plot(kind='bar', ax=axes[0, 0])
    axes[0, 0].set_title('Metrics Comparison')
    axes[0, 0].set_ylabel('Score')
    axes[0, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # Best model per metric
    best_models = metrics_df.idxmax()
    best_scores = metrics_df.max()
    
    bars = axes[0, 1].bar(best_models.index, best_scores.values)
    axes[0, 1].set_title('Best Model per Metric')
    axes[0, 1].set_ylabel('Best Score')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, score in zip(bars, best_scores.values):
        height = bar.get_height()
        axes[0, 1].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{score:.3f}', ha='center', va='bottom')
    
    # Model ranking by accuracy
    accuracy_ranking = metrics_df['accuracy'].sort_values(ascending=False)
    axes[1, 0].barh(range(len(accuracy_ranking)), accuracy_ranking.values)
    axes[1, 0].set_yticks(range(len(accuracy_ranking)))
    axes[1, 0].set_yticklabels(accuracy_ranking.index)
    axes[1, 0].set_title('Models Ranked by Accuracy')
    axes[1, 0].set_xlabel('Accuracy Score')
    
    # Confusion matrix for best model
    best_model_name = accuracy_ranking.index[0]
    best_predictions = evaluation_results[best_model_name]["predictions"]
    
    cm = confusion_matrix(y_test, best_predictions)
    sns.heatmap(cm, annot=True, fmt='d', ax=axes[1, 1], cmap='Blues')
    axes[1, 1].set_title(f'Confusion Matrix - {best_model_name}')
    axes[1, 1].set_xlabel('Predicted')
    axes[1, 1].set_ylabel('Actual')
    
    plt.tight_layout()
    
    # Save plot
    Path("plots").mkdir(exist_ok=True)
    plt.savefig("plots/model_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """Main evaluation pipeline"""
    # Load models and data
    models, X_test, y_test = load_models_and_data()
    
    # Evaluate all models
    evaluation_results = evaluate_all_models(models, X_test, y_test)
    
    # Create visualizations
    create_comparison_plots(evaluation_results, y_test)
    
    # Save evaluation results
    Path("metrics").mkdir(exist_ok=True)
    with open("metrics/evaluation_metrics.json", "w") as f:
        json.dump(evaluation_results, f, indent=2, default=str)
    
    # Print summary
    print("\n=== Model Evaluation Summary ===")
    for model_name, results in evaluation_results.items():
        metrics = results["metrics"]
        print(f"\n{model_name.upper()}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
    
    print(f"\nPlots saved to: plots/model_comparison.png")
    print(f"Detailed results saved to: metrics/evaluation_metrics.json")

if __name__ == "__main__":
    main()