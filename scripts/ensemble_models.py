import os, pandas as pd, matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

INP = "data/processed_data/pca_water_quality.csv"
FIG = "visualization/graphs/model_comparison.png"

def run_ensembles(df: pd.DataFrame) -> dict:
    X = df[["ph","turbidity","nitrate"]]; y = df["contamination_level"]
    Xtr, Xte, ytr, yte = train_test_split(X,y,test_size=0.2,random_state=42)
    models = {
        "RandomForest": RandomForestRegressor(n_estimators=200, random_state=42),
        "GradientBoosting": GradientBoostingRegressor(random_state=42),
    }
    scores = {}
    for name, m in models.items():
        m.fit(Xtr, ytr)
        yp = m.predict(Xte)
        rmse = mean_squared_error(yte, yp, squared=False)
        scores[name] = rmse
    return scores

def plot_scores(scores: dict, path=FIG):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    names = list(scores.keys())
    vals = [scores[k] for k in names]
    plt.figure(figsize=(7,4))
    plt.bar(names, vals)
    plt.ylabel("RMSE (lower better)")
    plt.title("Ensemble Model Comparison")
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()

if __name__ == "__main__":
    df = pd.read_csv(INP)
    scores = run_ensembles(df)
    plot_scores(scores)
    print("[OK] Ensembles:", scores)
