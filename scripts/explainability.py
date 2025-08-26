import os, pandas as pd, matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance

INP = "data/processed_data/pca_water_quality.csv"
SUM = "visualization/graphs/shap_summary.png"
BAR = "visualization/graphs/shap_bar.png"
FEAT = "visualization/graphs/feature_importance.png"

def permutation_plots(df: pd.DataFrame):
    X = df[["ph","turbidity","nitrate"]]; y = df["contamination_level"]
    model = RandomForestRegressor(n_estimators=200, random_state=42).fit(X,y)
    # Feature importances
    fi = getattr(model, "feature_importances_", None)
    names = X.columns.tolist()
    # Save feature importances
    plt.figure(figsize=(6,4))
    plt.bar(names, fi)
    plt.title("RandomForest Feature Importances")
    plt.tight_layout(); plt.savefig(FEAT, dpi=150); plt.close()

    # Permutation importance as SHAP proxy
    r = permutation_importance(model, X, y, n_repeats=10, random_state=42)
    imp = r.importances_mean
    # bar
    plt.figure(figsize=(6,4))
    plt.bar(names, imp)
    plt.title("Permutation Importance (proxy for SHAP bar)")
    plt.tight_layout(); plt.savefig(BAR, dpi=150); plt.close()

    # summary-like scatter (proxy): feature value vs permutation decrease
    plt.figure(figsize=(6,4))
    for i, n in enumerate(names):
        plt.scatter([i]*len(X), r.importances[i], s=4)
    plt.xticks(range(len(names)), names)
    plt.title("Permutation Importance Distribution (proxy for SHAP summary)")
    plt.tight_layout(); plt.savefig(SUM, dpi=150); plt.close()

if __name__ == "__main__":
    df = pd.read_csv(INP)
    permutation_plots(df)
    print("[OK] Saved permutation-importance plots.")
