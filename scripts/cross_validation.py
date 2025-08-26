import os, pandas as pd, numpy as np
from sklearn.model_selection import KFold, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

INP = "data/processed_data/pca_water_quality.csv"
OUT = "data/processed_data/cv_scores.csv"

def run_cv(df: pd.DataFrame):
    X = df[["ph","turbidity","nitrate"]]; y = df["contamination_level"]
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    models = {"LinearRegression": LinearRegression(), "RandomForest": RandomForestRegressor(200, random_state=42)}
    rows=[]
    for name, m in models.items():
        scores = cross_val_score(m, X, y, scoring="neg_root_mean_squared_error", cv=kf)
        rows.append({"model":name, "rmse_mean": -np.mean(scores), "rmse_std": np.std(scores)})
    return pd.DataFrame(rows)

if __name__ == "__main__":
    df = pd.read_csv(INP)
    res = run_cv(df)
    os.makedirs("data/processed_data", exist_ok=True)
    res.to_csv(OUT, index=False)
    print("[OK] CV results saved:", OUT)
