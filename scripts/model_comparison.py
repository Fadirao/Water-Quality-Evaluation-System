import os, pandas as pd, matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR

INP = "data/processed_data/pca_water_quality.csv"
FIG = "visualization/graphs/model_comparison.png"

def compare_models(df: pd.DataFrame) -> pd.DataFrame:
    X = df[["ph","turbidity","nitrate"]]; y = df["contamination_level"]
    Xtr, Xte, ytr, yte = train_test_split(X,y,test_size=0.2,random_state=42)
    models = {
        "Linear": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "Lasso": Lasso(alpha=0.001, max_iter=10000),
        "SVR": SVR(kernel="rbf", C=5, epsilon=0.1)
    }
    rows=[]
    for name,m in models.items():
        m.fit(Xtr,ytr)
        yp = m.predict(Xte)
        rmse = mean_squared_error(yte, yp, squared=False)
        rows.append({"model":name, "rmse":rmse})
    return pd.DataFrame(rows)

def plot(df_scores: pd.DataFrame, path=FIG):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df_scores = df_scores.sort_values("rmse")
    plt.figure(figsize=(6,4))
    plt.bar(df_scores["model"], df_scores["rmse"])
    plt.ylabel("RMSE")
    plt.title("Baseline Model Comparison")
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()

if __name__ == "__main__":
    df = pd.read_csv(INP)
    scores = compare_models(df)
    plot(scores)
    print("[OK] Model comparison saved.")
