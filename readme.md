# Customer Churn Prediction System

A production-oriented machine learning project for customer churn prediction using XGBoost, Optuna, MLflow, FastAPI, and Streamlit.

## Executive Summary

This project predicts whether a customer is likely to churn based on demographics, engagement, subscription profile, and billing behavior. It includes:

- End-to-end training pipeline from CSV data to serialized model
- Hyperparameter tuning with Optuna
- Experiment tracking with MLflow
- REST API with FastAPI for service-style inference
- Streamlit application for interactive business usage
- Strong validation and error handling across UI, API, and model layers

## Business Goal

Churn prediction supports retention strategy by identifying high-risk customers early. The model output includes both class prediction and churn probability so teams can prioritize interventions.

## Technical Stack

- Python, pandas, NumPy
- scikit-learn pipelines and preprocessing
- XGBoost classifier
- Optuna for HPO
- MLflow for experiment and model tracking
- FastAPI for API endpoints
- Streamlit for user-facing dashboard
- Pydantic for input/config validation

## Current Repository Structure

```text
.
├── data/
│   ├── train.csv
│   └── test.csv
├── notebooks/
│   └── expirements.ipynb
├── scripts/
│   └── run_pipeline.py
├── src/
│   ├── app/
│   │   ├── main.py
│   │   └── ui.py
│   ├── config/
│   │   └── settings.py
│   ├── data/
│   │   ├── load_data.py
│   │   └── preprocess.py
│   ├── features/
│   │   └── build_features.py
│   ├── models/
│   │   └── train.py
│   ├── serving/
│   │   ├── api_client.py
│   │   ├── inference.py
│   │   └── model/
│   └── utils/
│       └── validators.py
├── mlruns/
├── .gitignore
├── readme.md
└── requirements.txt
```

## How the System Works

### 1. Data Loading and Preprocessing

- CSV files are loaded with file and type safety checks.
- Preprocessing handles:
  - optional `CustomerID` removal
  - numeric type coercion
  - target mapping (`Yes/No` to `1/0`)
  - missing value dropping

### 2. Feature Engineering

- Numeric features are scaled with `MinMaxScaler`.
- Categorical features are encoded with `OneHotEncoder(drop="first", handle_unknown="ignore")`.
- A single sklearn `ColumnTransformer` guarantees consistent train/inference transformations.

### 3. Model Training

- The classifier is `XGBClassifier`.
- Hyperparameter search uses Optuna with `TPESampler`.
- Cross-validation uses `StratifiedKFold`.
- Objective prioritizes recall with variance penalty for robustness.

### 4. Artifact and Experiment Tracking

- Best trial parameters and metrics are logged to MLflow.
- Model is saved both:
  - locally (joblib) for fast direct inference
  - in MLflow model registry for experiment lineage

### 5. Inference Paths

- API path: FastAPI `/predict` endpoint validates and predicts.
- Direct path: Streamlit can call local inference directly via `predict()`.

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train model

```bash
python scripts/run_pipeline.py
```

### 3. Run API (optional)

```bash
uvicorn src.app.main:app --reload --port 8000
```

### 4. Run Streamlit dashboard

```bash
streamlit run src/app/ui.py
```

## API Endpoints

- `GET /health`: service heartbeat
- `POST /predict`: churn prediction
- `GET /model-info`: model metadata
- `GET /`: quick endpoint map

## Data Contract for Prediction

Expected input fields:

- `Age`
- `Gender`
- `Subscription Type`
- `Contract Length`
- `Usage Frequency`
- `Support Calls`
- `Tenure`
- `Total Spend`
- `Last Interaction`
- `Payment Delay`

## Project Timeline and What Happened

This repository evolved through real deployment and integration fixes. Major milestones:

1. Core pipeline implementation
- Built modular training pipeline (`load -> preprocess -> feature build -> train -> evaluate -> save`).
- Integrated MLflow logging and model registration.

2. Import reliability improvements
- Encountered `ModuleNotFoundError: No module named 'src'` in script/UI entrypoints.
- Fixed with project-root `sys.path` bootstrap for direct execution contexts.

3. Validation schema alignment
- Streamlit payload used space-based keys while validator expected mixed/underscored fields.
- Added field aliases and removed obsolete required field mismatch to align UI and backend schema.

4. Dependency hardening for deployment
- Streamlit deployment failed at config import due to missing Pydantic packages.
- Added `pydantic` and `pydantic-settings` to requirements.

5. Streamlit production behavior update
- Cloud app failed when trying to call `http://localhost:8000` (no API process in hosted Streamlit runtime).
- Streamlit app updated to perform direct local inference and removed redundant API URL sidebar input.

6. Repository hygiene for release readiness
- Expanded `.gitignore` to exclude virtual environments, caches, logs, secrets, MLflow artifacts, and local model binaries.

## Notes for Docker Readiness

The current workspace does not contain container files (`Dockerfile`, `.dockerignore`) at root. If containerization is the next step, add those files based on the now-stable runtime path (direct Streamlit inference or API + UI split, depending on deployment strategy).

## Operational Notes

- If model file is missing, run training first: `python scripts/run_pipeline.py`.
- MLflow runs are stored under `mlruns/`.
- Use `.env` for environment-specific overrides where needed.

## Maintainer Guidance

When changing feature schema:

1. Update feature lists in `src/config/settings.py`.
2. Update validation model in `src/utils/validators.py`.
3. Retrain and regenerate artifacts.
4. Re-test both FastAPI and Streamlit flows.

---

Streamlit App: https://nito-msc-kfs-classifier.streamlit.app
