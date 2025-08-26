# Water Quality Evaluation System

Advanced ML pipeline for analyzing water quality: preprocessing, feature engineering, clustering, regression,
ensembles, cross-validation, tuning, and explainability.

## Expected data schema
Place `data/raw_data/water_quality.csv` with columns:
- `timestamp` (e.g., `2024-05-01 10:00:00`)
- `ph` (float)
- `turbidity` (float)
- `nitrate` (float)
- `contamination_level` (float; target)

## Quickstart
```bash
python -m venv venv
# Windows: venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate
pip install -r requirements.txt

# 1) Preprocess -> cleaned CSV
python scripts/data_preprocessing.py

# 2) Optional imputations & outliers & validation
python scripts/imputation.py
python scripts/outlier_detection.py
python scripts/validation.py

# 3) Feature engineering & dimensionality reduction
python scripts/feature_engineering.py
python scripts/dimensionality_reduction.py

# 4) Models
python scripts/clustering_model.py
python scripts/regression_model.py
python scripts/neural_network_model.py
python scripts/ensemble_models.py

# 5) CV / tuning / comparison / explainability
python scripts/cross_validation.py
python scripts/hyperparameter_tuning.py
python scripts/model_comparison.py
python scripts/explainability.py
```

## Tests
```bash
pytest -q
```

## Outputs (saved to `visualization/graphs`)
- `contamination_patterns.png` (KMeans on PCA)
- `regression_scatter.png` (linear regression actual vs. pred)
- `model_comparison.png` (baseline models)
- `feature_importance.png` (RF feature importances)
- `loss_curve.png` (learning curve proxy)
- `shap_summary.png` + `shap_bar.png` (feature importance proxies produced via permutation importance)
