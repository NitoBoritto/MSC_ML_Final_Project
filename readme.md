# Customer Churn Prediction - ML Pipeline

Production-grade ML pipeline for predicting customer churn using XGBoost, with FastAPI backend and Streamlit dashboard.

## 📊 Overview

- **Task:** Binary classification (Churn / No Churn)
- **Dataset:** 11 features + 1 target (customer behavior metrics)
- **Model:** XGBoost with Optuna hyperparameter tuning
- **Training:** MLflow experiment tracking
- **Serving:** FastAPI REST API
- **UI:** Interactive Streamlit dashboard
- **Deployment:** Docker containerization + GitHub Actions CI/CD

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the Model

```bash
python scripts/run_pipeline.py
```

This will:
- Load train.csv and test.csv
- Preprocess data (type conversion, encoding)
- Split into 80/20 train/test
- Train XGBoost with Optuna (50 trials, 5-fold CV)
- Save model to `src/serving/model/xgboost_model.joblib`
- Log artifacts to MLflow

**Output:**
```
✓ Model saved: src/serving/model/xgboost_model.joblib
✓ MLflow tracking: ./mlruns
✓ Run ID: <uuid>
```

### 3. Start API Server

```bash
uvicorn src.app.main:app --reload --port 8000
```

Navigate to `http://localhost:8000/docs` for interactive API documentation.

### 4. Start Streamlit Dashboard

In a new terminal:

```bash
streamlit run src/app/ui.py
```

Navigate to `http://localhost:8501`

## 📁 Project Structure

```
src/
├── config/
│   ├── __init__.py
│   └── settings.py           # Centralized config (features, paths, MLflow)
├── data/
│   ├── __init__.py
│   ├── load_data.py          # CSV loading with validation
│   └── preprocess.py         # Type conversion, encoding
├── features/
│   ├── __init__.py
│   └── build_features.py     # sklearn ColumnTransformer (OHE + MinMaxScaler)
├── models/
│   ├── __init__.py
│   └── train.py              # XGBoost + Optuna + MLflow
├── serving/
│   ├── __init__.py
│   ├── model/                # Local model artifacts (created after training)
│   ├── inference.py          # Model loading & prediction
│   └── api_client.py         # HTTP client for FastAPI
├── utils/
│   ├── __init__.py
│   └── validators.py         # Pydantic input validation
└── app/
    ├── __init__.py
    ├── main.py               # FastAPI application
    └── ui.py                 # Streamlit dashboard

scripts/
└── run_pipeline.py           # Full pipeline orchestration

data/
├── train.csv                 # Training data (800 rows)
└── test.csv                  # Test data (200 rows)

tests/                         # (Optional) Unit tests
├── test_validators.py
└── test_inference.py

.github/
└── workflows/
    └── ci.yml                # GitHub Actions CI/CD pipeline

Dockerfile                     # Multi-stage Docker build
.dockerignore                  # Docker build exclusions
requirements.txt               # Python dependencies
README.md                      # This file
```

## 🔧 Configuration

All settings are in `src/config/settings.py`:

```python
# Feature definitions
FEATURE_NAMES = [
    "Age", "Tenure", "Usage Frequency", "Support Calls", "Payment Delay",
    "Subscription Type", "Contract Length", "Total Spend", "Last Interaction", "Gender"
]

NUMERIC_FEATURES = ["Age", "Tenure", "Usage Frequency", ...]
CATEGORICAL_FEATURES = ["Subscription Type", "Contract Length", "Gender"]

# Paths
MODEL_PATH = "src/serving/model/xgboost_model.joblib"
MLFLOW_TRACKING_URI = "./mlruns"

# API
API_URL = "http://localhost:8000"
PREDICT_ENDPOINT = "/predict"
```

Override via environment variables (e.g., `API_URL=http://prod-api:8000`).

## 📡 API Endpoints

### Health Check
```bash
GET /health
```

Response:
```json
{
  "status": "ok",
  "timestamp": "2024-05-05T10:30:00.000000"
}
```

### Make Prediction
```bash
POST /predict
```

Request:
```json
{
  "Age": 30,
  "Tenure": 12,
  "Usage Frequency": 14,
  "Support Calls": 5,
  "Payment Delay": 18,
  "Subscription Type": "Standard",
  "Contract Length": "Annual",
  "Total Spend": 932.0,
  "Last Interaction": 17,
  "Gender": "Female"
}
```

Response:
```json
{
  "prediction": 0,
  "churn_probability": 0.23,
  "label": "No Churn"
}
```

### Model Info
```bash
GET /model-info
```

Response:
```json
{
  "algorithm": "XGBoost",
  "model_path": "src/serving/model/xgboost_model.joblib",
  "feature_count": 10,
  "numeric_features": 7,
  "categorical_features": 3,
  "version": "1.0.0"
}
```

## 🐳 Docker Deployment

### Build Image

```bash
docker build -t churn-ml:latest .
```

### Run Container

```bash
docker run -p 8000:8000 \
  -e API_URL="http://0.0.0.0:8000" \
  churn-ml:latest
```

Test health:
```bash
curl http://localhost:8000/health
```

## 🔄 CI/CD Pipeline (GitHub Actions)

Workflow: `.github/workflows/ci.yml`

### Jobs

1. **Lint & Import Check** (~5 min)
   - flake8 code style checks
   - Module import validation

2. **Train & Validate Model** (~20 min)
   - Run `scripts/run_pipeline.py`
   - Verify model file exists
   - Upload artifact

3. **Build & Push Docker** (~15 min)
   - Build multi-stage Docker image
   - Push to Docker Hub (main branch only)

### Required Secrets

Add to GitHub repository settings → Secrets and variables:

```
DOCKER_USERNAME=<your-dockerhub-username>
DOCKER_PASSWORD=<your-dockerhub-token>
```

Generate token at: https://hub.docker.com/settings/security

## 🧪 Validation & Testing

### Input Validation

Pydantic validates all inputs:

```python
from src.utils import validate_input

data = {"Age": 30, "Tenure": 12, ...}
is_valid, error = validate_input(data)

if not is_valid:
    print(f"Error: {error}")
```

### Inference Test

```python
from src.serving import predict

result = predict({
    "Age": 30, "Tenure": 12, "Usage Frequency": 14,
    "Support Calls": 5, "Payment Delay": 18,
    "Subscription Type": "Standard", "Contract Length": "Annual",
    "Total Spend": 932, "Last Interaction": 17, "Gender": "Female"
})

print(result)  # {"prediction": 0, "churn_probability": 0.23, "label": "No Churn"}
```

## 📊 Model Details

### Training Strategy

- **Algorithm:** XGBoost Classifier
- **Hyperparameter Tuning:** Optuna (50 trials, TPE sampler)
- **Cross-Validation:** StratifiedKFold (5 splits)
- **Scoring Metric:** Recall (prioritize catching churners)
- **Class Weighting:** Balanced

### Preprocessing Pipeline

```
Input (raw features)
    ↓
ColumnTransformer
    ├─ Numeric: MinMaxScaler
    │   (Age, Tenure, Usage Frequency, Support Calls, Payment Delay, Total Spend, Last Interaction)
    └─ Categorical: OneHotEncoder(drop='first')
        (Subscription Type, Contract Length, Gender)
    ↓
XGBoost Classifier
    ↓
Prediction (0/1) + Probability
```

### MLflow Tracking

All runs tracked in `./mlruns/`:

```bash
# View experiments
mlflow ui --backend-store-uri ./mlruns

# Open browser to http://localhost:5000
```

## 🛠 Development

### Add New Features

1. Update feature list in `src/config/settings.py`
2. Update Pydantic model in `src/utils/validators.py`
3. Retrain model: `python scripts/run_pipeline.py`

### Modify Preprocessing

Edit `src/data/preprocess.py` or `src/features/build_features.py`, then retrain.

### Change Model

Replace XGBoost with LightGBM/LogisticRegression in `src/models/train.py`.

## 🐛 Troubleshooting

### Model Not Found
```
FileNotFoundError: src/serving/model/xgboost_model.joblib not found
```
**Solution:** Run `python scripts/run_pipeline.py` to train the model.

### API Connection Error (Streamlit)
```
API unreachable at http://localhost:8000
```
**Solution:** Ensure FastAPI is running: `uvicorn src.app.main:app --reload`

### Docker Build Fails
```
failed to solve with frontend dockerfile.v0
```
**Solution:** Ensure `requirements.txt` is in project root and `data/` folder exists.

## 📝 Logging

Logs are controlled via `LOG_LEVEL` setting:

```python
LOG_LEVEL = "INFO"  # Options: DEBUG, INFO, WARNING, ERROR
```

All modules use standard Python `logging`:

```python
import logging
logger = logging.getLogger(__name__)
logger.info("Message")
```

## 📦 Dependencies

Key packages (see `requirements.txt`):

- **sklearn:** preprocessing, pipelines, model selection
- **xgboost:** gradient boosting classifier
- **optuna:** hyperparameter tuning
- **mlflow:** experiment tracking & artifact storage
- **fastapi:** REST API framework
- **pydantic:** data validation
- **streamlit:** web dashboard
- **joblib:** model serialization

## 📜 License

TBD

## 👥 Contributors

- ML Engineer (Training & Serving)
- Backend Engineer (FastAPI)
- Frontend Engineer (Streamlit)

## 📞 Support

For issues or questions:
1. Check logs: `LOG_LEVEL=DEBUG python ...`
2. Review `.github/workflows/ci.yml` for pipeline commands
3. Check `src/config/settings.py` for configuration options

---

**Last Updated:** May 5, 2026  
**Model Version:** 1.0.0  
**Python Version:** 3.11+
