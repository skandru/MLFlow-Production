# MLOps Day 1: Foundations - Complete Experiment Tracking System

A comprehensive MLOps implementation demonstrating professional-grade experiment tracking, model versioning, and pipeline automation using MLFlow, Weights & Biases, and DVC.
<img width="1740" height="850" alt="Model_Results" src="https://github.com/user-attachments/assets/9183f2b1-9f2d-4d3e-afc1-c1055667bd38" />

<img width="1515" height="847" alt="Model_comparison_results" src="https://github.com/user-attachments/assets/b5385266-1f6c-479a-ae0a-cb0c42da1680" />

<img width="1843" height="752" alt="ML-pipeline-experiments_day1-charts" src="https://github.com/user-attachments/assets/06965895-8077-4d60-b2b9-bc2ec2fc3579" />

<img width="1844" height="704" alt="ML-pipeline-experiments_day1-models" src="https://github.com/user-attachments/assets/363ff04d-ac87-4997-b1f4-cd680b575456" />

## 🎯 Project Overview

This project implements a complete MLOps foundation that demonstrates:

- **Experiment Tracking** with MLFlow and Weights & Biases
- **Model Versioning** and Registry management
- **Data Version Control** with DVC
- **Automated Pipeline** orchestration
- **Comprehensive Model Evaluation** with visualizations
- **Production-Ready** code structure

### 🏆 Results Achieved

| Model | Test Accuracy | Precision | Recall | F1-Score |
|-------|---------------|-----------|---------|----------|
| **Logistic Regression** | **98.25%** | 98.28% | 98.25% | 98.25% |
| **SVM** | **98.25%** | 98.28% | 98.25% | 98.25% |
| **Random Forest** | **95.61%** | 95.74% | 95.61% | 95.63% |

## 📁 Project Structure

```
mlops-day1-project/
├── 📊 data/
│   ├── raw/                    # Original datasets
│   └── processed/              # Cleaned and split data
├── ⚙️ configs/
│   ├── __init__.py
│   └── mlflow_config.py        # MLFlow configuration
├── 🧠 models/
│   ├── random_forest_model.joblib
│   ├── logistic_regression_model.joblib
│   ├── svm_model.joblib
│   └── scaler.joblib           # Data preprocessing scaler
├── 📈 metrics/
│   ├── model_metrics.json      # Training metrics
│   └── evaluation_metrics.json # Evaluation results
├── 📊 plots/
│   └── comprehensive_model_evaluation.png
├── 📋 reports/
│   └── detailed_evaluation_report.json
├── 🔧 src/
│   ├── __init__.py
│   ├── data_pipeline.py        # Data processing pipeline
│   ├── train_multiple_models.py # Model training
│   ├── experiment_tracker.py   # MLFlow tracking utilities
│   ├── ml_pipeline.py          # Core ML pipeline
│   └── wandb_integration.py    # W&B integration
├── 🚀 mlruns/                  # MLFlow tracking store
├── 📝 dvc.yaml                 # DVC pipeline definition
├── ⚙️ params.yaml              # Model hyperparameters
├── 📋 requirements.txt         # Python dependencies
├── 🔧 register_models.py       # Model registry setup
├── 📊 enhanced_evaluation.py   # Comprehensive evaluation
├── 🚀 run_pipeline.py          # Pipeline orchestration
├── 🧪 test_pipeline.py         # Testing framework
└── 📖 README.md               # This file
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Git
- 4GB+ RAM recommended

### 1. Clone and Setup

```bash
# Clone the repository
git clone <your-repo-url>
cd mlops-day1-project

# Create virtual environment
python -m venv mlops-env
source mlops-env/bin/activate  # On Windows: mlops-env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Initialize MLFlow

```bash
# Start MLFlow UI (in separate terminal)
mlflow ui --host 0.0.0.0 --port 5000

# Access at: http://localhost:5000
```

### 3. Run the Complete Pipeline

```bash
# Option 1: Run individual steps
python src/data_pipeline.py
python src/train_multiple_models.py
python register_models.py
python enhanced_evaluation.py

# Option 2: Run automated pipeline
python run_pipeline.py
```

### 4. View Results

- **MLFlow UI**: http://localhost:5000
- **Model Metrics**: `metrics/model_metrics.json`
- **Evaluation Plots**: `plots/comprehensive_model_evaluation.png`
- **Detailed Reports**: `reports/detailed_evaluation_report.json`

## 🔧 Core Components

### Data Pipeline (`src/data_pipeline.py`)

```python
# Automated data processing and splitting
pipeline = DataPipeline("breast_cancer")
df, metadata = pipeline.create_raw_data()        # Load dataset
X_train, X_test, y_train, y_test = pipeline.process_data()  # Split and save
```

**Features:**
- ✅ Automated dataset loading (Breast Cancer dataset)
- ✅ Train/test splitting with stratification
- ✅ Metadata tracking and validation
- ✅ Robust error handling

### Model Training (`src/train_multiple_models.py`)

```python
# Multi-model training with MLFlow tracking
models = ["random_forest", "logistic_regression", "svm"]
for model_type in models:
    model, metrics = train_model(model_type, params)
```

**Features:**
- ✅ Multiple algorithm support
- ✅ Hyperparameter configuration via `params.yaml`
- ✅ Comprehensive metrics tracking
- ✅ Model artifact versioning
- ✅ MLFlow experiment logging

### Model Registry (`register_models.py`)

```python
# Register models in MLFlow Model Registry
model_registry = setup_model_registry()
promote_best_model(client, model_registry)
```

**Features:**
- ✅ Automatic model registration
- ✅ Version management
- ✅ Stage transitions (Staging → Production)
- ✅ Model comparison and promotion

### Enhanced Evaluation (`enhanced_evaluation.py`)

```python
# Comprehensive model evaluation
evaluator = EnhancedModelEvaluator()
results, y_test, features = evaluator.evaluate_all_models()
metrics_df = evaluator.create_comprehensive_visualizations(results, y_test, features)
```

**Features:**
- ✅ 9 different visualization types
- ✅ ROC curves and confusion matrices
- ✅ Feature importance analysis
- ✅ Model complexity vs performance
- ✅ Detailed JSON reports

## 📊 Experiment Tracking

### MLFlow Integration

- **Experiments**: All runs tracked in `ml-pipeline-experiments`
- **Parameters**: Model hyperparameters and dataset info
- **Metrics**: Training/test accuracy, precision, recall, F1-score
- **Artifacts**: Trained models, plots, and metadata
- **Model Registry**: Named, versioned models with stage management

### Weights & Biases (Optional)

```python
# W&B integration for advanced tracking
wandb_integration = WandBIntegration(project_name="mlops-day1")
wandb_integration.init_run(config=params, run_name="experiment_1")
```

### Data Version Control (DVC)

```yaml
# dvc.yaml - Pipeline definition
stages:
  prepare_data:
    cmd: python src/data_pipeline.py
    outs: [data/raw/, data/processed/]
  
  train_models:
    cmd: python src/train_multiple_models.py
    deps: [data/processed/]
    outs: [models/]
    metrics: [metrics/model_metrics.json]
```

## 🎯 Model Performance

### Hyperparameter Configuration

```yaml
# params.yaml
models:
  random_forest:
    n_estimators: 100
    max_depth: 7
    min_samples_split: 5
    min_samples_leaf: 2
    random_state: 42
  
  logistic_regression:
    C: 1.0
    penalty: l2
    solver: liblinear
    max_iter: 1000
    random_state: 42
  
  svm:
    C: 1.0
    kernel: rbf
    random_state: 42
```

### Model Metrics

| Metric | Random Forest | Logistic Regression | SVM |
|--------|---------------|---------------------|-----|
| **Accuracy** | 95.61% | **98.25%** | **98.25%** |
| **Precision** | 95.74% | **98.28%** | **98.28%** |
| **Recall** | 95.61% | **98.25%** | **98.25%** |
| **F1-Score** | 95.63% | **98.25%** | **98.25%** |
| **Training Time** | ~0.15s | ~0.05s | ~0.08s |

## 🔍 Advanced Features

### Automated Model Promotion

```python
# Best performing model automatically promoted to Production
best_model = max(models, key=lambda x: x['test_accuracy'])
client.transition_model_version_stage(
    name=best_model_name,
    version=best_model_version,
    stage="Production"
)
```

### Comprehensive Visualizations

The evaluation pipeline generates 9 different visualization types:

1. **📊 Metrics Comparison** - Side-by-side performance comparison
2. **🏆 Accuracy Ranking** - Models ranked by performance
3. **📈 ROC Curves** - Receiver Operating Characteristic analysis
4. **🔍 Confusion Matrices** - Per-model classification analysis
5. **🌟 Feature Importance** - Random Forest feature analysis
6. **⚖️ Complexity vs Performance** - Model trade-off analysis
7. **📋 Summary Statistics** - Comprehensive performance summary
8. **📊 Class Distribution** - Dataset balance analysis
9. **🎯 Model Recommendations** - Best use-case scenarios

### Testing Framework

```python
# Automated testing pipeline
python test_pipeline.py

# Tests include:
# ✅ Data pipeline functionality
# ✅ MLFlow tracking verification
# ✅ Model training validation
# ✅ File output verification
```

## 🛠️ Technology Stack

| Category | Technology | Purpose |
|----------|------------|---------|
| **🧠 ML Framework** | scikit-learn | Model training and evaluation |
| **📊 Experiment Tracking** | MLFlow | Experiment management and model registry |
| **🌐 Advanced Tracking** | Weights & Biases | Enhanced monitoring and collaboration |
| **📝 Data Versioning** | DVC | Data and pipeline version control |
| **📈 Visualization** | matplotlib, seaborn | Plot generation and analysis |
| **🔧 Data Processing** | pandas, numpy | Data manipulation and analysis |
| **💾 Model Persistence** | joblib | Model serialization |
| **⚙️ Configuration** | YAML | Parameter management |

## 📚 Usage Examples

### Training a Single Model

```python
from src.ml_pipeline import MLPipeline

# Initialize pipeline
pipeline = MLPipeline("breast_cancer")

# Train specific model
model, metrics = pipeline.train_model(
    model_type="random_forest",
    n_estimators=100,
    max_depth=7
)

print(f"Model accuracy: {metrics['test_accuracy']:.4f}")
```

### Comparing Models

```python
from enhanced_evaluation import EnhancedModelEvaluator

# Run comprehensive evaluation
evaluator = EnhancedModelEvaluator()
results, y_test, features = evaluator.evaluate_all_models()

# Generate visualizations
metrics_df = evaluator.create_comprehensive_visualizations(
    results, y_test, features
)
```

### Accessing MLFlow Data

```python
import mlflow
from mlflow.tracking import MlflowClient

# Initialize client
client = MlflowClient()

# Get best run
experiment = client.get_experiment_by_name("ml-pipeline-experiments")
runs = client.search_runs([experiment.experiment_id])
best_run = max(runs, key=lambda x: x.data.metrics.get('test_accuracy', 0))

print(f"Best accuracy: {best_run.data.metrics['test_accuracy']:.4f}")
```

## 🧪 Testing

### Run All Tests

```bash
python test_pipeline.py
```

### Test Coverage

- ✅ Data pipeline functionality
- ✅ Model training workflow
- ✅ MLFlow tracking integration
- ✅ File output validation
- ✅ Error handling verification

## 📈 Monitoring and Alerts

### MLFlow Tracking

- **Real-time metrics**: Track training progress live
- **Parameter comparison**: Compare hyperparameter effects
- **Artifact storage**: Automatic model and plot storage
- **Version control**: Complete model lineage tracking

### Performance Monitoring

```python
# Monitor model performance over time
def monitor_model_drift():
    latest_metrics = load_latest_metrics()
    baseline_metrics = load_baseline_metrics()
    
    drift_threshold = 0.05
    accuracy_drift = abs(latest_metrics['accuracy'] - baseline_metrics['accuracy'])
    
    if accuracy_drift > drift_threshold:
        send_alert(f"Model drift detected: {accuracy_drift:.3f}")
```

## 🚀 Deployment Considerations

### Model Serving

```python
# Load production model
import mlflow.sklearn

model_uri = "models:/breast_cancer_logistic_regression/Production"
model = mlflow.sklearn.load_model(model_uri)

# Make predictions
predictions = model.predict(new_data)
```

### API Integration

```python
# FastAPI endpoint example
from fastapi import FastAPI
import mlflow.sklearn

app = FastAPI()
model = mlflow.sklearn.load_model("models:/best_model/Production")

@app.post("/predict")
def predict(features: List[float]):
    prediction = model.predict([features])
    return {"prediction": int(prediction[0])}
```

## 🔒 Security and Compliance

### Data Privacy

- ✅ No sensitive data in version control
- ✅ Configurable data encryption
- ✅ Access control integration ready
- ✅ Audit trail through MLFlow

### Model Governance

- ✅ Model approval workflows
- ✅ Performance monitoring
- ✅ Bias detection capabilities
- ✅ Regulatory compliance tracking

## 🤝 Contributing

### Development Setup

```bash
# Clone and setup development environment
git clone <repo-url>
cd mlops-day1-project
python -m venv dev-env
source dev-env/bin/activate
pip install -r requirements.txt
pip install -r requirements-dev.txt  # Additional dev dependencies
```

### Code Standards

- **PEP 8** compliance for Python code
- **Type hints** for function signatures
- **Docstrings** for all classes and functions
- **Unit tests** for critical functionality


**Problem**: MLFlow UI not showing models
```bash
# Solution: Register models first
python register_models.py
```

**Problem**: Import errors
```bash
# Solution: Ensure proper Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

**Problem**: Memory issues with large datasets
```bash
# Solution: Reduce batch size or use data sampling
# Edit params.yaml to adjust model parameters
```

## 📝 License

This project is licensed under the MIT License 

## 🙏 Acknowledgments

- **scikit-learn** team for excellent ML algorithms
- **MLFlow** community for experiment tracking tools
- **DVC** team for data version control
- **Weights & Biases** for advanced experiment monitoring

## 📊 Project Metrics

![GitHub last commit](https://img.shields.io/github/last-commit/username/mlops-day1-project)
![GitHub issues](https://img.shields.io/github/issues/username/mlops-day1-project)
![GitHub stars](https://img.shields.io/github/stars/username/mlops-day1-project)
![GitHub forks](https://img.shields.io/github/forks/username/mlops-day1-project)

---

**🎯 Ready for Production | 🔧 Enterprise-Grade | 📊 Fully Tracked**

> This project demonstrates professional MLOps practices essential for senior ML engineering roles. It showcases end-to-end pipeline automation, comprehensive experiment tracking, and production-ready code architecture.

---

*Built with ❤️ for the MLOps community*
