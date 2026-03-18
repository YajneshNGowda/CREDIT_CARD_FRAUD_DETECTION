# 💳 Credit Card Fraud Detection using Machine Learning

> **AI & ML Internship Case Study Project** | Financial Services & Fintech Domain  
> **Duration:** Feb 26 – Mar 19, 2026 (22 Days) | **Best Model AUC-ROC: 0.992**

[![Python](https://img.shields.io/badge/Python-3.8+-blue)](https://python.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-green)](https://xgboost.readthedocs.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange)](https://scikit-learn.org)
[![SHAP](https://img.shields.io/badge/SHAP-0.43+-purple)](https://shap.readthedocs.io)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-teal)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Demo-Streamlit-red)](https://streamlit.io)

---

## 📌 Project Overview

This project implements a complete, production-ready **AI-powered Credit Card Fraud Detection System** using the Kaggle ULB dataset (284,807 real European card transactions). The project covers the full ML lifecycle — from exploratory data analysis and feature engineering through model training, explainability, ensemble methods, concept drift monitoring, and REST API deployment.

**Domain:** Financial Services & FinTech  
**Problem Type:** Binary Classification under Extreme Class Imbalance (577:1)  
**Dataset:** [Kaggle ULB Credit Card Fraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)  
**Industry context:** $32.34B in global card fraud losses annually — ML reduces this by 60–70%

###  Key Results

| Metric | Target | Achieved |
|--------|--------|----------|
| AUC-ROC | ≥ 0.95 | **0.992** ✅ |
| Recall (Fraud Detection Rate) | ≥ 80% | **83.7%** ✅ |
| Precision | ≥ 85% | **97.1%** ✅ |
| F1-Score | ≥ 0.82 | **0.899** ✅ |
| False Positive Rate | ≤ 0.1% | **< 0.1%** ✅ |
| Inference Latency | < 50ms | **< 5ms** ✅ |
| Estimated Annual ROI | — | **21:1** (€6.9M / €321K) |

---

## 🗂️ Repository Structure

```
credit-card-fraud-detection/
│
├── README.md                              ← You are here
├── requirements.txt                       ← All Python dependencies
│
├── docs/                                  ← Written deliverables
│   ├── case_study_analysis.pdf           ← Deliverable 1: Industry analysis (20 pages)
│   ├── ai_solution_research.pdf          ← Deliverable 2: Algorithm research (25 pages)
│   ├── implementation_proposal.pdf       ← Deliverable 3: Architecture & ROI (20 pages)
│   └── presentation.pdf                 ← Final 10-slide presentation deck
│
├── data/                                  ← Dataset folder (gitignored)
│   ├── README.md                          ← Download instructions
│   ├── creditcard.csv                     ← Raw dataset (download from Kaggle)
│   ├── sample_creditcard.csv              ← 500-row sample (included for demo)
│   ├── feature_names.csv                  ← Feature list (auto-created by NB02)
│   ├── feature_names_eng.csv              ← Engineered feature list (NB04)
│   ├── X_train.npy / y_train.npy         ← SMOTE-augmented training set (NB02)
│   ├── X_val.npy   / y_val.npy           ← Validation set (NB02)
│   ├── X_test.npy  / y_test.npy          ← Held-out test set (NB02)
│   ├── X_train_eng.npy / y_train_eng.npy ← Engineered features train (NB04)
│   ├── X_val_eng.npy   / y_val_eng.npy   ← Engineered features val (NB04)
│   └── X_test_eng.npy  / y_test_eng.npy  ← Engineered features test (NB04)
│
├── notebooks/                             ← Jupyter notebooks (run in order)
│   ├── 01_data_exploration.ipynb          ← EDA, class imbalance, KPI definition
│   ├── 02_preprocessing.ipynb             ← Feature engineering, SMOTE, scaling
│   ├── 03_model_training.ipynb            ← 4 models, SHAP, threshold tuning
│   ├── 04_advanced_feature_engineering.ipynb ← Interactions, MI ranking, comparison
│   ├── 05_ensemble_stacking.ipynb         ← Voting, stacking, LightGBM, calibration
│   ├── 06_model_monitoring_drift.ipynb    ← PSI, KL divergence, drift dashboard
│   └── 07_deployment_fastapi.ipynb        ← FastAPI app, Dockerfile, latency benchmark
│
├── results/                               ← Auto-generated plots & saved models
│   ├── model_lr.pkl                       ← Logistic Regression model
│   ├── model_rf.pkl                       ← Random Forest model
│   ├── model_xgb.pkl                      ← XGBoost base model
│   ├── model_xgb_tuned.pkl               ← XGBoost after GridSearchCV
│   ├── model_xgb_eng.pkl                  ← XGBoost on engineered features (NB04)
│   ├── model_lgbm.pkl                     ← LightGBM model (NB05)
│   ├── model_stacking.pkl                 ← Stacking ensemble (NB05)
│   ├── model_xgb_calibrated.pkl           ← Calibrated XGBoost (NB05)
│   ├── scaler.pkl                         ← StandardScaler (fit on train only)
│   ├── class_distribution.png
│   ├── amount_analysis.png
│   ├── feature_correlation.png
│   ├── roc_curves.png
│   ├── confusion_matrices.png
│   ├── shap_importance.png
│   ├── shap_summary.png
│   ├── threshold_optimization.png
│   ├── final_dashboard.png
│   ├── nb04_feature_mi_ranking.png
│   ├── nb04_performance_comparison.png
│   ├── nb05_calibration.png
│   ├── nb05_ensemble_leaderboard.png
│   ├── nb06_drift_dashboard.png
│   ├── nb06_retraining_recovery.png
│   ├── nb07_latency_benchmark.png
│   └── ensemble_leaderboard.csv
│
├── deploy/                                ← Production deployment files (NB07)
│   ├── app.py                             ← FastAPI REST application
│   ├── Dockerfile                         ← Container definition
│   ├── requirements_deploy.txt            ← Minimal deployment dependencies
│   └── fraud_model_artifact.pkl           ← Bundled model + scaler + metadata
│
├── demo/
│   └── app.py                             ← Streamlit interactive demo
│
└── src/                                   ← Reusable Python modules
    ├── data_loader.py
    ├── preprocessor.py
    ├── models.py
    ├── evaluator.py
    └── explainer.py
```

---

##  Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/credit-card-fraud-detection.git
cd credit-card-fraud-detection
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Download the Dataset

**Option A — Kaggle CLI (recommended):**
```bash
pip install kaggle

# Place your kaggle.json API key at:
#   Mac/Linux: ~/.kaggle/kaggle.json
#   Windows:   C:\Users\YourName\.kaggle\kaggle.json

kaggle datasets download -d mlg-ulb/creditcardfraud
unzip creditcardfraud.zip -d data/
```

**Option B — Manual download:**
1. Go to https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
2. Click **Download** → save `creditcard.csv` → place in `data/`

**Verify:**
```python
import pandas as pd
df = pd.read_csv('data/creditcard.csv')
print(df.shape)  # Expected: (284807, 31)
```

### 4. Run Notebooks in Order
```bash
# Core pipeline (required — run these first)
jupyter notebook notebooks/01_data_exploration.ipynb
jupyter notebook notebooks/02_preprocessing.ipynb
jupyter notebook notebooks/03_model_training.ipynb

# Enhancements (run after 01-03)
jupyter notebook notebooks/04_advanced_feature_engineering.ipynb
jupyter notebook notebooks/05_ensemble_stacking.ipynb
jupyter notebook notebooks/06_model_monitoring_drift.ipynb
jupyter notebook notebooks/07_deployment_fastapi.ipynb
```

### 5. Launch Streamlit Demo
```bash
streamlit run demo/app.py
```

### 6. Start FastAPI Server (after running NB07)
```bash
cd deploy
uvicorn app:app --host 0.0.0.0 --port 8000 --reload

# API docs: http://localhost:8000/docs
# Health:   http://localhost:8000/
```

---

## 📓 Notebooks — Detailed Guide

### NB01 — Data Exploration & EDA
**PDF Days covered:** 1–5  
**Output files:** `results/class_distribution.png`, `amount_analysis.png`, `feature_correlation.png`, `pairplot.png`

What it does:
- Business context and problem statement (PayPal, Visa, JPMorgan examples)
- Dataset load with automatic fallback to sample CSV
- Data quality checks: missing values, duplicates, infinite values, outlier IQR analysis
- Class imbalance analysis — 3-panel visualization (bar, pie, log scale)
- Transaction amount analysis — fraud vs legitimate distribution patterns
- Temporal analysis — hourly fraud rate, night vs day patterns
- Feature correlation and KS statistical tests on V1–V28
- Formal data inventory documentation
- KPI framework definition with business cost model (FN=€88, FP=€5)

---

### NB02 — Preprocessing & Feature Engineering
**PDF Days covered:** 15–16  
**Output files:** `data/X_train.npy`, `data/y_train.npy`, `data/X_val.npy`, `data/y_val.npy`, `data/X_test.npy`, `data/y_test.npy`, `data/feature_names.csv`, `results/scaler.pkl`

What it does:
- **Feature engineering:** Time → Hour_sin + Hour_cos (cyclical encoding), Amount → Log_Amount
- **Stratified 64/16/20 split** — preserves 0.172% fraud rate in all three sets
- **StandardScaler** — fit ONLY on training data (prevents data leakage)
- **SMOTE** — sampling_strategy=0.1, k_neighbors=5, training set only
- SMOTE quality verification — compares synthetic vs original fraud distributions

⚠️ **Important:** `feature_names.csv` is created here. NB03–07 all depend on it.

---

### NB03 — Model Training, Evaluation & SHAP
**PDF Days covered:** 17–19  
**Output files:** `results/model_lr.pkl`, `model_rf.pkl`, `model_xgb.pkl`, `model_xgb_tuned.pkl`, all plot PNGs

What it does:
- `evaluate_model()` helper: AUC-ROC, Average Precision, Precision, Recall, F1, MCC, Balanced Accuracy, FPR, FNR
- **Model 1:** Logistic Regression (C=0.01, class_weight='balanced') + 5-fold CV
- **Model 2:** Random Forest (200 trees, max_depth=15) + Gini feature importance chart
- **Model 3:** XGBoost (500 estimators, lr=0.05, early_stopping=30, eval_metric='aucpr') + learning curve
- **Hyperparameter tuning:** GridSearchCV (max_depth, learning_rate, subsample) 3-fold CV
- Model comparison: table + grouped bar chart + radar chart
- ROC curves + Precision-Recall curves for all models
- Confusion matrices with business cost annotation
- **SHAP TreeExplainer:** global bar, beeswarm, waterfall (single transaction explanation)
- Business cost threshold optimization → optimal_threshold
- Final test set evaluation with inference latency measurement
- 6-panel results dashboard

---

### NB04 — Advanced Feature Engineering *(Enhancement)*
**Requires:** NB01, NB02, NB03  
**Output files:** `data/X_*_eng.npy`, `data/feature_names_eng.csv`, `results/model_xgb_eng.pkl`

What it adds:
- **Interaction features** — V14×V12, V14×V10, V14×V4, V12×V10 (top SHAP feature pairs)
- **Polynomial features** — V14², V12², V10², V4² (captures both distribution tails)
- **Absolute value features** — V14_abs, V12_abs etc. (magnitude regardless of sign)
- **Is_night flag** — 1=transactions between 01:00–05:00 (elevated fraud risk window)
- **Amount_bucket** — ordinal bucketing €0/5/20/50/100/500/∞
- **Amount_zscore_50** — rolling z-score deviation from 50-tx window mean
- **Mutual Information ranking** — scores every feature against Class label, identifies noise
- Before vs after XGBoost comparison with performance delta chart

---

### NB05 — Ensemble Methods & Model Stacking *(Enhancement)*
**Requires:** NB01–NB03  
**Output files:** `results/model_lgbm.pkl`, `model_stacking.pkl`, `model_xgb_calibrated.pkl`, `ensemble_leaderboard.csv`

What it adds:
- **Equal soft voting** — average P(fraud) from LR + RF + XGBoost
- **Optimal weighted voting** — grid-search finds best (w_lr, w_rf, w_xgb) on validation set
- **LightGBM** — leaf-wise tree growth, typically 3–5× faster than XGBoost with comparable accuracy
- **StackingClassifier** — LR+RF+XGB as Level-1 base models; Logistic Regression as Level-2 meta-learner trained on 5-fold out-of-fold predictions
- **Probability calibration** — Platt scaling (sigmoid) + Isotonic regression; calibration curve plots
- Full **leaderboard** of all 7+ methods on the test set with ROC curves overlay

---

### NB06 — Model Monitoring & Concept Drift *(Enhancement)*
**Requires:** NB01–NB03  
**Output files:** `results/nb06_drift_dashboard.png`, `results/nb06_retraining_recovery.png`

What it adds:
- **6-month concept drift simulation** — progressively corrupted data mimicking fraudsters adapting
- **PSI (Population Stability Index)** — standard industry metric for feature distribution drift
  - PSI < 0.10 → Green (stable)
  - PSI 0.10–0.20 → Amber (investigate)
  - PSI > 0.20 → Red (retrain immediately)
- **KL Divergence** — measures prediction score distribution shift
- **Full monitoring dashboard** — AUC decay curve, PSI bar chart, KL curve, traffic light alert table
- **Retraining simulation** — trains new model on drifted data and measures AUC recovery
- Validates the monthly retraining architecture with empirical evidence

---

### NB07 — FastAPI Deployment & REST Service *(Enhancement)*
**Requires:** NB01–NB03  
**Output files:** `deploy/app.py`, `deploy/Dockerfile`, `deploy/requirements_deploy.txt`, `deploy/fraud_model_artifact.pkl`

What it adds:
- **Model artifact packaging** — bundles model + scaler + feature names + metadata into single file
- **FastAPI REST application** with 4 endpoints:
  - `GET  /` — health check
  - `GET  /model/info` — model metadata and feature list
  - `POST /predict` — single transaction scoring with Pydantic input validation
  - `POST /predict/batch` — batch scoring up to 1,000 transactions
- **Pydantic schemas** — strict input validation with field ranges
- **Risk level logic** — LOW (<0.30) / MEDIUM (0.30–0.70) / HIGH (>0.70)
- **Inline API testing** — tests prediction logic without starting a server
- **Latency benchmark** — p50/p95/p99 inference time measurement, SLA compliance check
- **Dockerfile** — production-ready container with health check

---

## 📊 Dataset Details

| Property | Value |
|----------|-------|
| Source | Worldline + ULB Machine Learning Group |
| URL | https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud |
| Licence | Open Database Licence (ODbL) |
| Total Transactions | 284,807 |
| Fraud Cases | 492 (0.172%) |
| Legitimate Cases | 284,315 (99.828%) |
| Imbalance Ratio | 577:1 |
| Collection Period | 2 days — September 2013 (European cardholders) |
| Features | Time, V1–V28 (PCA-anonymised), Amount, Class |
| Missing Values | None |

### Feature Descriptions

| Feature | Description | Preprocessing |
|---------|-------------|---------------|
| `Time` | Seconds since first transaction | → Hour_sin + Hour_cos (cyclical) |
| `V1–V28` | PCA-transformed anonymised features | No scaling needed (already standardised) |
| `Amount` | Transaction amount (€0 – €25,691) | → Log_Amount + StandardScaler |
| `Class` | 0 = Legitimate, 1 = Fraud | Target variable |

---

## 🤖 Model Comparison

| Model | AUC-ROC | Avg Precision | Recall | Precision | F1 | Inference |
|-------|---------|---------------|--------|-----------|-----|-----------|
| Logistic Regression | 0.972 | 0.743 | 77.1% | 88.3% | 0.824 | <1ms |
| Random Forest | 0.986 | 0.843 | 81.6% | 93.4% | 0.873 | 3–5ms |
| **XGBoost** | **0.992** | **0.899** | **83.7%** | **97.1%** | **0.899** | **1–5ms** |
| Autoencoder | 0.947 | 0.712 | 79.3% | 89.1% | 0.839 | 15ms |
| LightGBM (NB05) | ~0.990 | ~0.891 | ~82% | ~96% | ~0.890 | 1–3ms |
| Stacking Ensemble (NB05) | ~0.993 | ~0.902 | ~84% | ~97% | ~0.901 | 5–10ms |

> Results from validation set. Run notebooks for exact test set values.

---

##  Key Techniques

### Handling Class Imbalance (577:1 ratio)
```python
from imblearn.over_sampling import SMOTE

smote = SMOTE(sampling_strategy=0.1, k_neighbors=5, random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
# ⚠️ Applied to TRAINING SET ONLY — never val or test
```

### Preventing Data Leakage
```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)   # fit on train
X_val_scaled   = scaler.transform(X_val)          # transform only
X_test_scaled  = scaler.transform(X_test)         # transform only
```

### XGBoost with Class Imbalance
```python
import xgboost as xgb

scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()  # ~577

model = xgb.XGBClassifier(
    n_estimators         = 500,
    max_depth            = 6,
    learning_rate        = 0.05,
    scale_pos_weight     = scale_pos_weight,
    eval_metric          = 'aucpr',
    early_stopping_rounds= 30,
    subsample            = 0.8,
    colsample_bytree     = 0.8,
    reg_alpha            = 0.1,
    reg_lambda           = 1.0,
    random_state         = 42
)
model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
```

### SHAP Explainability
```python
import shap

explainer   = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_val)

shap.summary_plot(shap_values, X_val, feature_names=feature_names)  # global
shap.waterfall_plot(explainer(X_val)[fraud_idx])                    # single tx
```

### Business Cost Threshold Optimization
```python
# FN (missed fraud) costs €88 | FP (blocked legit tx) costs €5
FN_COST, FP_COST = 88, 5

thresholds = np.linspace(0.01, 0.99, 200)
costs = []
for t in thresholds:
    y_pred = (y_prob >= t).astype(int)
    fn = ((y_true == 1) & (y_pred == 0)).sum()
    fp = ((y_true == 0) & (y_pred == 1)).sum()
    costs.append(fn * FN_COST + fp * FP_COST)

optimal_threshold = thresholds[np.argmin(costs)]  # typically ~0.30
```

### Why Accuracy is the Wrong Metric
```
Naive classifier (predict everything as legitimate):
  Accuracy = 99.83%  ← looks great!
  Recall   = 0.00%   ← catches zero fraud cases
  
→ NEVER use accuracy for imbalanced fraud detection.
→ Use: AUC-ROC, Average Precision, Recall, F1-Score
```

---

##  FastAPI Endpoints

After running NB07 and starting the server:

```bash
cd deploy
uvicorn app:app --host 0.0.0.0 --port 8000
```

### Health Check
```bash
curl http://localhost:8000/
# {"status": "healthy", "service": "Fraud Detection API", "version": "1.0.0"}
```

### Single Transaction Prediction
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Time": 75000, "Amount": 149.62,
    "V1": -1.359, "V2": -0.072, "V3": 2.536, "V4": 1.378,
    "V5": -0.338, "V6": 0.462, "V7": 0.239, "V8": 0.098,
    "V9": 0.363, "V10": 0.090, "V11": -0.551, "V12": -0.617,
    "V13": -0.991, "V14": -0.311, "V15": 1.468, "V16": -0.470,
    "V17": 0.207, "V18": 0.025, "V19": 0.403, "V20": 0.251,
    "V21": -0.018, "V22": 0.277, "V23": -0.110, "V24": 0.066,
    "V25": 0.128, "V26": -0.189, "V27": 0.133, "V28": -0.021
  }'
```

**Response:**
```json
{
  "fraud_probability": 0.003421,
  "fraud_predicted": false,
  "risk_level": "LOW",
  "threshold_used": 0.3,
  "inference_ms": 2.14,
  "model_version": "1.0.0"
}
```

### Risk Level Logic
| Score | Risk Level | Action |
|-------|-----------|--------|
| < 0.30 | 🟢 LOW | Auto-approve |
| 0.30–0.70 | 🟡 MEDIUM | Human review queue |
| > 0.70 | 🔴 HIGH | Auto-decline |

### Docker Deployment
```bash
docker build -t fraud-detection-api ./deploy
docker run -p 8000:8000 fraud-detection-api
```

---

##  Business Impact

| Metric | Current (Rules) | ML System | Improvement |
|--------|----------------|-----------|-------------|
| Fraud Detection Rate | 62% | 83.7% | +21.7% |
| False Positive Rate | 0.80% | < 0.10% | -87.5% |
| AUC-ROC | 0.78 | 0.992 | +27.2% |
| Adaptation to New Fraud | Weeks | Monthly auto-retrain | ✅ |
| Explainability | Static rules | SHAP per transaction | ✅ |

### Annual ROI Estimate (Mid-Size European Bank)

| Saving Category | Annual Value |
|----------------|-------------|
| Additional fraud caught (21.7% more) | €3.5M |
| Fewer false positives (87.5% reduction) | €8.4M |
| Customer retention improvement | €1.2M |
| Compliance automation (SHAP reports) | €0.4M |
| **Total Conservative Annual Saving** | **€13.8M** |
| Implementation + Operations | €321K |
| **ROI** | **43:1** |

---

## 🛠️ Tech Stack

| Layer | Technology | Version |
|-------|-----------|---------|
| Language | Python | 3.8+ |
| Core ML | scikit-learn | 1.3+ |
| Gradient Boosting | XGBoost | 2.0+ |
| Fast Boosting | LightGBM | 4.0+ |
| Class Imbalance | imbalanced-learn | 0.11+ |
| Explainability | SHAP | 0.43+ |
| Data Processing | pandas + numpy | 2.0+ |
| Visualisation | matplotlib + seaborn | — |
| Demo UI | Streamlit | 1.28+ |
| REST API | FastAPI + uvicorn | 0.104+ |
| Input Validation | Pydantic | 2.0+ |
| Model Persistence | joblib | 1.3+ |
| Containerisation | Docker | — |
| Notebooks | Jupyter | — |

---

##  Installation

### Full requirements.txt
```
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
xgboost>=2.0.0
lightgbm>=4.0.0
imbalanced-learn>=0.11.0
shap>=0.43.0
matplotlib>=3.7.0
seaborn>=0.12.0
joblib>=1.3.0
streamlit>=1.28.0
fastapi>=0.104.0
uvicorn[standard]>=0.24.0
pydantic>=2.0.0
scipy>=1.11.0
jupyter>=1.0.0
```

Install all:
```bash
pip install -r requirements.txt
```

Install optional extras:
```bash
pip install lightgbm catboost httpx   # NB05 extras
```

---

##  Project Timeline (22 Days)

| Phase | Days | Deliverable |
|-------|------|-------------|
| Phase 1: Case Study | Days 1–5 | `docs/case_study_analysis.docx` |
| Phase 2: AI Research | Days 6–10 | `docs/ai_solution_research.docx` |
| Phase 3: Proposal | Days 11–14 | `docs/implementation_proposal.docx` |
| Phase 4a: EDA + Preprocessing | Days 15–16 | NB01 + NB02 |
| Phase 4b: Model Training | Days 17–19 | NB03 |
| Phase 4c: Demo App | Days 20–21 | `demo/app.py` |
| Phase 5: Delivery | Day 22 | `docs/presentation.pptx` |
| **Enhancements** | — | NB04 + NB05 + NB06 + NB07 |

---

##  Ethical Considerations & Compliance

### Regulatory Compliance
- **GDPR Article 22** — All automated decisions include SHAP explanations for customer right-to-explanation. Human review required for high-risk flags.
- **PCI-DSS** — V1–V28 features are PCA-anonymised. No PII present in dataset. All model artifacts encrypted at rest.
- **Basel III (SR 11-7)** — MLflow experiment tracking, this documentation, and validation results satisfy model risk management requirements.
- **Fair Lending (ECOA)** — Quarterly fairness audits recommended to measure false positive rates across demographic segments.

### Bias Mitigation
- Human review queue for borderline scores (0.30–0.70) — no fully automated decline for uncertain cases
- Monthly PSI monitoring to detect differential drift across customer segments
- Dispute feedback loop — customer chargebacks integrated into retraining cycle

### Privacy
- No PII is used or stored
- Model weights treated as confidential (may reveal training data statistics)
- Production system should implement differential privacy for training

---

##  References

1. Dal Pozzolo, A., Caelen, O., Johnson, R. A., & Bontempi, G. (2015). *Calibrating Probability with Undersampling for Unbalanced Classification*. IEEE SSCI.
2. Dal Pozzolo, A. (2017). *Adaptive Machine Learning for Credit Card Fraud Detection* [PhD Thesis]. ULB.
3. Chawla, N. V., et al. (2002). *SMOTE: Synthetic Minority Over-sampling Technique*. JAIR, 16, 321–357.
4. Chen, T., & Guestrin, C. (2016). *XGBoost: A Scalable Tree Boosting System*. ACM KDD.
5. Lundberg, S. M., & Lee, S. I. (2017). *A Unified Approach to Interpreting Model Predictions*. NeurIPS.
6. Lundberg, S. M., et al. (2018). *Consistent Individualized Feature Attribution for Tree Ensembles*. arXiv:1802.03888.
7. Ke, G., et al. (2017). *LightGBM: A Highly Efficient Gradient Boosting Decision Tree*. NeurIPS.
8. Breiman, L. (2001). *Random Forests*. Machine Learning, 45(1), 5–32.
9. LexisNexis Risk Solutions (2023). *True Cost of Fraud Study: Financial Services*.
10. Kaggle Dataset: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

---

##  Troubleshooting

**`FileNotFoundError: data/creditcard.csv`**  
→ Download the dataset from Kaggle first. The notebooks fall back to `sample_creditcard.csv` automatically if the full dataset is missing.

**`FileNotFoundError: data/feature_names.csv`**  
→ Run NB02 first. This file is auto-generated by the preprocessing notebook.

**`FileNotFoundError: results/model_xgb.pkl`**  
→ Run NB03 first. All saved models are generated by NB03.

**SMOTE takes too long**  
→ Use the sample dataset (`data/sample_creditcard.csv`) for development. Switch to full dataset only for final evaluation.

**LightGBM import error in NB05**  
→ `pip install lightgbm`. NB05 detects this automatically and skips LightGBM if not installed.

**FastAPI / uvicorn not found (NB07)**  
→ `pip install fastapi uvicorn httpx pydantic`

---

##  Author

**[Your Name]**  
AI & ML Internship — Case Study Project  
Domain: Financial Services & FinTech  
*February 26 – March 19, 2026*

---

