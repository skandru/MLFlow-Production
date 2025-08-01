# enhanced_evaluation.py
import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import json
from pathlib import Path

class EnhancedModelEvaluator:
    def __init__(self):
        mlflow.set_tracking_uri("file:./mlruns")
        self.client = MlflowClient()
        
    def load_test_data(self):
        """Load test data for evaluation"""
        X_test = pd.read_csv("data/processed/X_test.csv")
        y_test = pd.read_csv("data/processed/y_test.csv")['target']
        
        # Load and apply scaler
        scaler = joblib.load("models/scaler.joblib")
        X_test_scaled = scaler.transform(X_test)
        
        return X_test_scaled, y_test, X_test.columns
    
    def load_models(self):
        """Load all trained models"""
        models = {}
        model_files = {
            'random_forest': 'models/random_forest_model.joblib',
            'logistic_regression': 'models/logistic_regression_model.joblib', 
            'svm': 'models/svm_model.joblib'
        }
        
        for name, filepath in model_files.items():
            if Path(filepath).exists():
                models[name] = joblib.load(filepath)
        
        return models
    
    def evaluate_all_models(self):
        """Comprehensive evaluation of all models"""
        X_test, y_test, feature_names = self.load_test_data()
        models = self.load_models()
        
        results = {}
        predictions = {}
        
        for model_name, model in models.items():
            print(f"📊 Evaluating {model_name}...")
            
            # Make predictions
            y_pred = model.predict(X_test)
            predictions[model_name] = y_pred
            
            # Calculate metrics
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, average='weighted'),
                'recall': recall_score(y_test, y_pred, average='weighted'),
                'f1': f1_score(y_test, y_pred, average='weighted')
            }
            
            # Get probability predictions for ROC curve (if available)
            if hasattr(model, 'predict_proba'):
                y_proba = model.predict_proba(X_test)[:, 1]
                fpr, tpr, _ = roc_curve(y_test, y_proba)
                roc_auc = auc(fpr, tpr)
                metrics['roc_auc'] = roc_auc
                results[model_name] = {
                    'metrics': metrics,
                    'predictions': y_pred,
                    'probabilities': y_proba,
                    'roc_data': (fpr, tpr)
                }
            else:
                results[model_name] = {
                    'metrics': metrics,
                    'predictions': y_pred
                }
        
        return results, y_test, feature_names
    
    def create_comprehensive_visualizations(self, results, y_test, feature_names):
        """Create comprehensive visualizations"""
        
        # Set up the plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create a large figure with subplots
        fig = plt.figure(figsize=(20, 15))
        
        # 1. Metrics Comparison
        ax1 = plt.subplot(3, 3, 1)
        metrics_df = pd.DataFrame({name: data['metrics'] for name, data in results.items()}).T
        metrics_df.plot(kind='bar', ax=ax1)
        ax1.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Score')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.tick_params(axis='x', rotation=45)
        
        # 2. Accuracy Ranking
        ax2 = plt.subplot(3, 3, 2)
        accuracy_data = {name: data['metrics']['accuracy'] for name, data in results.items()}
        sorted_acc = dict(sorted(accuracy_data.items(), key=lambda x: x[1], reverse=True))
        
        bars = ax2.bar(sorted_acc.keys(), sorted_acc.values(), 
                      color=['gold', 'silver', '#CD7F32'][:len(sorted_acc)])
        ax2.set_title('Accuracy Ranking', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Accuracy Score')
        ax2.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, acc in zip(bars, sorted_acc.values()):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 3. ROC Curves (if available)
        ax3 = plt.subplot(3, 3, 3)
        for name, data in results.items():
            if 'roc_data' in data:
                fpr, tpr = data['roc_data']
                auc_score = data['metrics']['roc_auc']
                ax3.plot(fpr, tpr, label=f'{name} (AUC = {auc_score:.3f})', linewidth=2)
        
        ax3.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        ax3.set_xlabel('False Positive Rate')
        ax3.set_ylabel('True Positive Rate')
        ax3.set_title('ROC Curves', fontsize=14, fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4-6. Confusion Matrices
        model_names = list(results.keys())
        for i, (name, data) in enumerate(results.items()):
            ax = plt.subplot(3, 3, 4 + i)
            cm = confusion_matrix(y_test, data['predictions'])
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_title(f'Confusion Matrix - {name.replace("_", " ").title()}', 
                        fontsize=12, fontweight='bold')
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
        
        # 7. Feature Importance (for Random Forest)
        if 'random_forest' in results:
            ax7 = plt.subplot(3, 3, 7)
            rf_model = joblib.load('models/random_forest_model.joblib')
            feature_importance = pd.DataFrame({
                'feature': feature_names,
                'importance': rf_model.feature_importances_
            }).sort_values('importance', ascending=False).head(10)
            
            ax7.barh(feature_importance['feature'], feature_importance['importance'])
            ax7.set_title('Top 10 Feature Importances (Random Forest)', 
                         fontsize=12, fontweight='bold')
            ax7.set_xlabel('Importance')
        
        # 8. Model Complexity vs Performance
        ax8 = plt.subplot(3, 3, 8)
        model_complexity = {
            'logistic_regression': 1,  # Linear model
            'svm': 2,                  # Non-linear kernel
            'random_forest': 3         # Ensemble method
        }
        
        complexity_data = []
        for name, data in results.items():
            if name in model_complexity:
                complexity_data.append({
                    'model': name.replace('_', ' ').title(),
                    'complexity': model_complexity[name],
                    'accuracy': data['metrics']['accuracy']
                })
        
        complexity_df = pd.DataFrame(complexity_data)
        scatter = ax8.scatter(complexity_df['complexity'], complexity_df['accuracy'], 
                             s=100, alpha=0.7, c=range(len(complexity_df)), cmap='viridis')
        
        for i, row in complexity_df.iterrows():
            ax8.annotate(row['model'], 
                        (row['complexity'], row['accuracy']),
                        xytext=(5, 5), textcoords='offset points')
        
        ax8.set_xlabel('Model Complexity')
        ax8.set_ylabel('Accuracy')
        ax8.set_title('Model Complexity vs Performance', fontsize=12, fontweight='bold')
        ax8.grid(True, alpha=0.3)
        
        # 9. Summary Statistics
        ax9 = plt.subplot(3, 3, 9)
        ax9.axis('off')
        
        # Create summary text
        best_model = max(results.items(), key=lambda x: x[1]['metrics']['accuracy'])
        summary_text = f"""
        📊 EVALUATION SUMMARY
        ═══════════════════════════
        
        🏆 Best Model: {best_model[0].replace('_', ' ').title()}
        📈 Best Accuracy: {best_model[1]['metrics']['accuracy']:.4f}
        
        📋 Model Rankings:
        """
        
        # Add rankings
        sorted_models = sorted(results.items(), 
                             key=lambda x: x[1]['metrics']['accuracy'], 
                             reverse=True)
        
        for i, (name, data) in enumerate(sorted_models, 1):
            summary_text += f"\n        {i}. {name.replace('_', ' ').title()}: {data['metrics']['accuracy']:.4f}"
        
        summary_text += f"""
        
        📊 Dataset Info:
        • Test Samples: {len(y_test)}
        • Features: {len(feature_names)}
        • Classes: {len(np.unique(y_test))}
        """
        
        ax9.text(0.1, 0.9, summary_text, transform=ax9.transAxes, 
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
        
        plt.tight_layout()
        
        # Save the comprehensive plot
        Path("plots").mkdir(exist_ok=True)
        plt.savefig("plots/comprehensive_model_evaluation.png", 
                   dpi=300, bbox_inches='tight', facecolor='white')
        
        print("📊 Comprehensive evaluation plot saved to: plots/comprehensive_model_evaluation.png")
        
        # Log to MLFlow
        with mlflow.start_run(run_name="Model_Comparison_Analysis"):
            mlflow.log_artifact("plots/comprehensive_model_evaluation.png")
            
            # Log comparison metrics
            comparison_metrics = {}
            for name, data in results.items():
                for metric, value in data['metrics'].items():
                    comparison_metrics[f"{name}_{metric}"] = value
            
            mlflow.log_metrics(comparison_metrics)
        
        plt.show()
        
        return metrics_df
    
    def generate_detailed_report(self, results, y_test):
        """Generate detailed evaluation report"""
        report = {
            "evaluation_timestamp": pd.Timestamp.now().isoformat(),
            "dataset_info": {
                "test_samples": len(y_test),
                "class_distribution": pd.Series(y_test).value_counts().to_dict()
            },
            "model_performance": {}
        }
        
        for name, data in results.items():
            report["model_performance"][name] = {
                "metrics": data['metrics'],
                "classification_report": classification_report(
                    y_test, data['predictions'], output_dict=True
                )
            }
        
        # Save detailed report
        Path("reports").mkdir(exist_ok=True)
        with open("reports/detailed_evaluation_report.json", "w") as f:
            json.dump(report, f, indent=2, default=str)
        
        print("📋 Detailed report saved to: reports/detailed_evaluation_report.json")
        
        return report

def main():
    """Main evaluation function"""
    print("🚀 Starting Enhanced Model Evaluation")
    
    evaluator = EnhancedModelEvaluator()
    
    # Run comprehensive evaluation
    results, y_test, feature_names = evaluator.evaluate_all_models()
    
    # Create visualizations
    metrics_df = evaluator.create_comprehensive_visualizations(results, y_test, feature_names)
    
    # Generate detailed report
    report = evaluator.generate_detailed_report(results, y_test)
    
    # Print summary
    print("\n" + "="*60)
    print("📊 EVALUATION COMPLETE")
    print("="*60)
    
    print("\n🏆 Model Performance Summary:")
    for name, data in results.items():
        metrics = data['metrics']
        print(f"\n{name.upper()}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
    
    print(f"\n📁 Outputs generated:")
    print(f"  📊 Comprehensive plot: plots/comprehensive_model_evaluation.png")
    print(f"  📋 Detailed report: reports/detailed_evaluation_report.json")
    print(f"  🌐 MLFlow tracking: http://localhost:5000")

if __name__ == "__main__":
    main()