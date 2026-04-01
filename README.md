# Credit Risk MLOps System

An end-to-end **credit risk prediction system** built with **Python, scikit-learn, FastAPI, MLflow, Prometheus, Streamlit, and Docker**.

This project is designed to demonstrate how a machine learning model can move beyond notebook experimentation into a more production-oriented workflow. It includes:

- data loading and preprocessing
- train/validation/test split generation
- multiple candidate model pipelines
- threshold-based decision strategies
- probability calibration support
- model versioning with `latest` and `champion` pointers
- evaluation-based model promotion
- FastAPI model serving
- structured JSON logging
- Prometheus metrics
- production prediction logging
- feature drift and prediction drift monitoring
- SHAP-based prediction explanations
- Streamlit dashboard for operational visibility
- Dockerized local deployment
- basic automated tests

---

## Table of Contents

- [Project Overview](#project-overview)
- [Why This Project Matters](#why-this-project-matters)
- [Main Capabilities](#main-capabilities)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Data Pipeline](#data-pipeline)
- [Model Training Pipelines](#model-training-pipelines)
- [Thresholding Strategy](#thresholding-strategy)
- [Calibration Support](#calibration-support)
- [Artifact Management and Model Registry](#artifact-management-and-model-registry)
- [FastAPI Service](#fastapi-service)
- [API Endpoints](#api-endpoints)
- [Monitoring and Observability](#monitoring-and-observability)
- [Drift Monitoring](#drift-monitoring)
- [Explainability](#explainability)
- [Streamlit Dashboard](#streamlit-dashboard)
- [Environment Configuration](#environment-configuration)
- [End-to-End Workflow](#end-to-end-workflow)
- [Local Setup](#local-setup)
- [Docker Setup](#docker-setup)
- [Testing](#testing)
- [Example Prediction Request](#example-prediction-request)
- [What This Project Demonstrates](#what-this-project-demonstrates)
- [Known Gaps / Next Improvements](#known-gaps--next-improvements)

---

## Project Overview

This repository implements a full **credit default prediction system** using a classical tabular credit risk dataset and a production-style serving architecture.

The objective is not only to train a classifier, but also to answer practical MLOps questions such as:

- How is the data split and versioned?
- How are thresholds chosen?
- How can multiple candidate models be compared?
- How does the system decide which model is active?
- How are predictions logged in production?
- How is model behavior monitored after deployment?
- How can individual predictions be explained?
- How can a lightweight dashboard support ML operations?

This project answers those questions with a local, practical, engineering-focused design.

---

## Why This Project Matters

Many ML projects stop at:

- loading a dataset
- training one model
- printing a metric
- saving a notebook

This project goes further by showing the path toward a usable ML service:

- multiple candidate models instead of one
- explicit threshold policies instead of only `0.5`
- model calibration support
- registry-style `latest` / `champion` workflow
- promotion scripts
- operational metrics
- JSON production logs
- drift summaries
- explainability endpoints
- Dockerized deployment
- UI dashboard for model operations

The focus is on demonstrating **ML engineering and MLOps thinking**, not only model training.

---

## Main Capabilities

This system supports the following core capabilities:

### Data preparation
- load raw Excel dataset
- clean and standardize target column
- generate schema report
- generate data quality report
- create stratified train/validation/test splits

### Model development
- Logistic Regression baseline
- Random Forest candidate
- HistGradientBoosting candidate
- threshold search on validation set
- PR-AUC / F1 / Precision / Recall evaluation
- calibration metrics with Brier score
- calibrated Random Forest variant

### Model lifecycle
- save artifacts by model version
- track most recent model using `latest.json`
- track production champion using `champion.json`
- compare `latest` vs `champion`
- promotion script with evaluation gate
- promotion audit log

### Online serving
- FastAPI prediction service
- Pydantic request validation
- request ID middleware
- structured logs
- threshold-aware prediction modes
- health and metadata endpoints

### Monitoring
- Prometheus counters and histograms
- production JSONL prediction logs
- feature drift using PSI
- prediction drift against training default rate

### Explainability
- tree-model feature importance
- grouped feature importance
- SHAP local prediction explanations
- grouped positive/negative drivers

### Operations UI
- Streamlit dashboard
- health snapshot
- metrics snapshot
- prediction form
- SHAP explanation view
- drift summary
- model comparison
- feature importance view

---

## Architecture

The system can be understood in five layers.

### 1. Data Layer
Raw data is loaded from:

data/raw/default_of_credit_card_clients.xls

The loader is responsible for:

- removing useless index-like columns such as `Unnamed: 0`
- converting selected numeric fields to numeric
- standardizing the target column name to `default`
- enforcing integer target values

---

### 2. Training Layer

Three model training scripts generate separate artifact folders:

- model_v2 → Logistic Regression baseline  
- model_v3 → Random Forest candidate  
- model_v4 → HistGradientBoosting candidate  

Each training pipeline saves:

- serialized model  
- threshold file  
- metrics file  
- calibration report  
- baseline feature stats  
- run metadata  
- MLflow run association  
- pointer update for `latest.json`  

---

### 3. Registry Layer

The system uses a lightweight local model registry pattern.

- `latest.json` points to the most recently trained candidate  
- `champion.json` points to the active promoted model  

The API can load the model in three modes:

- latest  
- champion  
- manual  

---

### 4. Serving Layer

FastAPI loads the selected artifact bundle and serves:

- metadata  
- predictions  
- explanations  
- drift summaries  
- model comparison  
- calibration details  
- metrics for monitoring  

---

### 5. Monitoring Layer

Runtime behavior is tracked through:

- Prometheus counters and histograms  
- JSON structured logs  
- production prediction JSONL logs  
- drift summaries  
- dashboard snapshots  

---

## Project Structure

credit-risk-mlops/

- artifacts/
  - data/
    - data_report.md  
    - schema.json  
  - model_v1/  
  - model_v2/  
  - model_v3/  
  - model_v4/  
  - latest.json  
  - champion.json  
  - promotion_log.jsonl  

- data/
  - raw/
    - default_of_credit_card_clients.xls  
  - processed/
    - train.parquet  
    - val.parquet  
    - test.parquet  
  - prod/
    - predictions.jsonl  

- logs/
  - app.log  

- mlruns/

- src/
  - credit_risk/
    - dashboard/
      - app.py  
    - scripts/
      - evaluate_and_promote.py  
      - make_baseline_stats.py  
      - make_report.py  
      - make_schema.py  
      - make_split.py  
      - promote_model.py  
    - app.py  
    - baseline_stats.py  
    - config.py  
    - data_load.py  
    - data_report.py  
    - drift.py  
    - features.py  
    - logging_utils.py  
    - predict.py  
    - schema.py  
    - split.py  
    - train.py  
    - train_hgb.py  
    - train_random_forest.py  

- tests/
  - test_api.py  
  - test_drift.py  
  - test_smoke.py  

- .dockerignore  
- .env  
- docker-compose.yml  
- Dockerfile  
- pyproject.toml  
- README.md  

---

## Dataset

The project uses a credit default dataset stored as:

data/raw/default_of_credit_card_clients.xls

Feature mapping:

- X1 → credit limit  
- X2 → sex  
- X3 → education  
- X4 → marriage  
- X5 → age  
- X6–X11 → repayment status history  
- X12–X17 → bill amounts  
- X18–X23 → payment amounts  

Target column:

default  

---

## Data Pipeline

The data pipeline is implemented through reusable scripts.


### make_report.py

Generates a markdown data quality report including:

- row and column counts  
- target distribution  
- missing value analysis  
- numeric outlier ranges (1%–99%)  
- high-cardinality columns  
- recommended checks  

Output:

artifacts/data/data_report.md  

---

### make_schema.py

Builds a machine-readable schema describing:

- number of rows and columns  
- target column  
- each column's data type  
- nullability  
- missing value count  
- unique value count  
- numeric summary values or top categorical values  

Output:

artifacts/data/schema.json  

---

### make_split.py

Creates stratified train/validation/test datasets and stores them as parquet files.

Output:

- data/processed/train.parquet  
- data/processed/val.parquet  
- data/processed/test.parquet  

---

### make_baseline_stats.py

Computes baseline feature statistics for monitoring.

Output:

artifacts/model_v1/baseline_feature_stats.json  

---

## Model Training Pipelines

The project includes three separate training workflows.

---

### 1. Logistic Regression Baseline (train.py)

This script builds a baseline pipeline with:

- StandardScaler for numeric features  
- OneHotEncoder for categorical features  
- LogisticRegression  
- class_weight="balanced"  

It computes:

- validation and test PR-AUC  
- F1, precision, recall  
- thresholds for:
  - recall80  
  - maxf1  
  - fixed_0_5  
- calibration metrics  
- baseline feature stats  
- run metadata  
- MLflow logs  

Artifacts are saved to:

artifacts/model_v2/  

---

### 2. Random Forest Candidate (train_random_forest.py)

This script trains a stronger non-linear tree-based model using:

- StandardScaler  
- OneHotEncoder  
- RandomForestClassifier  

Additional capabilities:

- calibrated model using CalibratedClassifierCV  
- calibrated threshold computation  
- uncalibrated vs calibrated comparison  
- Brier score tracking  
- SHAP-compatible model for explanations  
- feature importance support  

Artifacts are saved to:

artifacts/model_v3/  

Saved files include:

- model.joblib  
- model_calibrated.joblib  
- thresholds.json  
- thresholds_calibrated.json  
- metrics.json  
- calibration.json  
- calibration_compare.json  
- baseline_feature_stats.json  
- run_meta.json  

---

### 3. HistGradientBoosting Candidate (train_hgb.py)

This script trains a boosted tree model using:

- StandardScaler  
- OrdinalEncoder  
- HistGradientBoostingClassifier  

It computes:

- validation and test metrics  
- threshold strategies  
- calibration metrics  
- baseline monitoring stats  
- MLflow run logs  

Artifacts are saved to:

artifacts/model_v4/  

---

## Thresholding Strategy

A key part of this repository is that model predictions are not reduced to a single hardcoded threshold.

The project supports three thresholding modes:

### recall80
Threshold selected on validation data to achieve approximately 80% recall.

### maxf1
Threshold selected on validation data to maximize F1 score.

### fixed_0_5
Traditional threshold of 0.5.

These thresholds are saved in:

- thresholds.json  
- thresholds_calibrated.json  

This design separates:

- probability estimation  
- decision policy  

This is closer to real-world risk systems.

---

## Calibration Support

The Random Forest candidate includes probability calibration.

Calibration is important because:

- a classifier may rank correctly  
- but still produce poorly calibrated probabilities  

This repository evaluates calibration using:

- calibration curves  
- Brier score  
- calibrated vs uncalibrated comparisons  

Relevant artifacts:

- calibration.json  
- calibration_compare.json  
- model_calibrated.joblib  
- thresholds_calibrated.json  

T## Artifact Management and Model Registry

This project uses a lightweight local artifact and registry design.

### Artifact Directories

Each model version has its own artifact directory:

- artifacts/model_v2  
- artifacts/model_v3  
- artifacts/model_v4  

### latest.json

Points to the most recently trained model.  
Automatically updated after training.

### champion.json

Points to the promoted production model.

---

### Promotion Scripts

#### promote_model.py

Promotes `latest.json` directly to `champion.json`.

#### evaluate_and_promote.py

- compares latest vs champion on a selected metric and split  
- promotes only if the latest model improves by at least `min_improvement`  
- appends promotion events to a JSONL log  

This registry pattern is simple but demonstrates core model lifecycle control.

---

## FastAPI Service

Main application:

src/credit_risk/app.py  

### Startup Behavior

On startup, the service:

- reads configuration from `.env`  
- validates `MODEL_SELECT_MODE`  
- loads `latest.json` and/or `champion.json`  
- resolves active artifact directory  
- decides calibrated vs uncalibrated model  
- verifies required files exist  
- validates threshold keys  
- validates baseline stats structure  
- verifies model supports `predict_proba`  

This prevents running with broken artifacts.

---

### Request Validation

Prediction requests are validated using Pydantic and include:

- numeric bounds  
- categorical constraints  
- explicit validators (e.g. sex, education, marriage)  

---

### Middleware

#### Request ID Middleware

- assigns `x-request-id` to every request  
- returned in response headers  
- logged for traceability  

#### Response Timing Middleware

- measures latency  
- adds `x-latency-ms` to response headers  

---

## API Endpoints

### Core Endpoints

#### GET /health

Returns:

- service status  
- active environment  
- active model version  
- threshold file used  
- calibration status  
- model type  
- feature count  
- run metadata  

---

#### GET /model/info

Returns:

- artifact path  
- model path  
- threshold values  
- feature lists  
- target name  
- selection mode  
- active / latest / champion metadata  

---

#### POST /predict

Returns:

- probability of default  
- threshold used  
- prediction mode  
- final binary prediction  

---

### Explainability Endpoints

#### GET /explain/feature_importance/raw

Returns raw feature importances.

#### GET /explain/feature_importance/grouped

Returns grouped feature importance mapped to original features.

#### GET /explain/feature_importance

Shortcut to grouped feature importance.

#### POST /explain/predict

Returns SHAP explanation (raw features).

#### POST /explain/predict/grouped

Returns grouped SHAP explanation:

- top positive drivers  
- top negative drivers  
- top overall contributors  

---

### Monitoring & Registry Endpoints

#### GET /metrics

Prometheus-compatible metrics endpoint.

#### GET /drift/summary

Returns:

- overall drift status  
- feature-level PSI  
- prediction drift  
- warning / alert features  

#### GET /registry/status

Returns:

- model selection mode  
- configured version  
- active version  
- latest version  
- champion version  

---

#### GET /models/compare

Returns:

- latest model score  
- champion model score  
- improvement value  
- decision suggestion  

---

#### GET /model/calibration

Returns calibration artifact for the active model.

---

## Debug Endpoint

#### GET /debug/data

Returns:

- raw dataset row count  
- column count  
- column names  
- target distribution  

---

## Monitoring and Observability

This repository includes several production-oriented observability features.

---

### 1. Prometheus Metrics

Tracks:

- total prediction requests  
- risky predictions  
- latency histogram  
- response counts  
- error counts  
- requests by mode  

Endpoint:

GET /metrics  

---

### 2. Structured JSON Logging

Logs include:

- timestamp  
- level  
- logger name  
- message  
- event type  
- request ID  
- method  
- path  
- status code  
- latency  

Log file:

logs/app.log  

---

### 3. Production Prediction Logging

Each request writes a JSONL record including:

- timestamp  
- request ID  
- mode  
- threshold  
- predicted probability  
- binary decision  
- input features  
- latency  

Log file:

data/prod/predictions.jsonl  

Used for:

- auditability  
- drift monitoring  
- debugging  

---

## Drift Monitoring

Drift detection is implemented using Population Stability Index (PSI).

### Numeric Drift

Compares production values with baseline quantile distributions.

### Categorical Drift

Compares category distributions between baseline and production.

### Prediction Drift

Compares:

- production prediction mean  
- training default rate  

### Severity Levels

- OK  
- WARN  
- ALERT  

Endpoint:

GET /drift/summary  

---

## Explainability

Explainability is available for tree-based models.

### Feature Importance

Returns:

- raw feature importances  
- grouped feature importances  

### SHAP Explanations

Supported models:

- RandomForestClassifier  
- HistGradientBoostingClassifier  

Provides:

- positive drivers  
- negative drivers  
- top contributors  

---

## Streamlit Dashboard

Location:

src/credit_risk/dashboard/app.py  

Features:

- API health check  
- model metadata view  
- Prometheus metrics summary  
- sample predictions  
- custom input prediction  
- threshold analysis  
- SHAP explanations  
- drift summary  
- model comparison  
- feature importance  

Provides a lightweight operational UI.

---

## Environment Configuration

Example `.env`:

APP_ENV=dev  
MODEL_VERSION=model_v2  
DEFAULT_MODE=recall80  
LOG_LEVEL=INFO  
MODEL_SELECT_MODE=champion  
USE_CALIBRATED_MODEL=true  

---

### Configuration Details

#### APP_ENV

- dev  
- prod  
- test  

#### MODEL_VERSION

Used when `MODEL_SELECT_MODE=manual`.

#### DEFAULT_MODE

- recall80  
- maxf1  
- fixed_0_5  

#### LOG_LEVEL

Controls logging verbosity.

#### MODEL_SELECT_MODE

- latest  
- champion  
- manual  

#### USE_CALIBRATED_MODEL

If true:

- model_calibrated.joblib  
- thresholds_calibrated.json  

Else:

- model.joblib  
- thresholds.json  

---

## End-to-End Workflow

### 1. Generate data artifacts

python -m credit_risk.scripts.make_report  
python -m credit_risk.scripts.make_schema  
python -m credit_risk.scripts.make_split  
python -m credit_risk.scripts.make_baseline_stats  

### 2. Train models

python -m credit_risk.train  
python -m credit_risk.train_random_forest  
python -m credit_risk.train_hgb  

### 3. Promote model

python -m credit_risk.scripts.promote_model  

or

python -m credit_risk.scripts.evaluate_and_promote --metric pr_auc --split test --min-improvement 0.0  

---

### 4. Run API

uvicorn credit_risk.app:app --reload  

### 5. Run Dashboard

streamlit run src/credit_risk/dashboard/app.py  

---

## Local Setup

### 1. Clone repo

git clone <your-repo-url>  
cd credit-risk-mlops  

### 2. Virtual env

python -m venv .venv  

### 3. Activate

Windows:  
.venv\Scripts\activate  

Linux/Mac:  
source .venv/bin/activate  

### 4. Install

pip install --upgrade pip  
pip install -e .  
pip install streamlit requests  

---

## Docker Setup

Includes:

- Dockerfile  
- docker-compose.yml  

Run:

docker compose up --build  

Services:

- API → http://localhost:8000  
- Dashboard → http://localhost:8501  

---

## Testing

Run:

pytest -q  

Covers:

- API health  
- prediction success/failure  
- drift checks  

---

## Example Prediction

JSON:

{
  "X1": 200000,
  "X2": 2,
  "X3": 2,
  "X4": 1,
  "X5": 35,
  ...
}

---

## What This Project Demonstrates

### Machine Learning

- tabular ML  
- imbalance handling  
- threshold optimization  
- calibration  
- model comparison  

### ML Engineering

- pipelines  
- artifact management  
- API serving  
- explainability  

### MLOps

- monitoring  
- logging  
- drift detection  
- promotion workflows  
- Docker deployment  

---
```text
data/raw/default_of_credit_card_clients.xls

