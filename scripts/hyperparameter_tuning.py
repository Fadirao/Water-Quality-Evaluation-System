import json, os, pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor

INP = "data/processed_data/pca_water_quality.csv"
OUT = "data/processed_data/best_params.json"

def tune(df: pd.DataFrame):
    X = df[["ph","turbidity","nitrate"]]; y = df["contamination_level"]
    rf = RandomForestRegressor(random_state=42)
    grid = {
        "n_estimators":[100,200],
        "max_depth":[None,10,20],
        "min_samples_split":[2,5],
        "min_samples_leaf":[1,2]
    }
    gs = GridSearchCV(rf, grid, scoring="neg_root_mean_squared_error", cv=3, n_jobs=-1)
    gs.fit(X,y)
    return gs.best_params_

if __name__ == "__main__":
    df = pd.read_csv(INP)
    best = tune(df)
    os.makedirs("data/processed_data", exist_ok=True)
    with open(OUT, "w") as f: json.dump(best, f, indent=2)
    print("[OK] Best params saved:", best)
