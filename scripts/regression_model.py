import os, pandas as pd, matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression

INP = "data/processed_data/pca_water_quality.csv"
PRED = "data/processed_data/regression_predictions.csv"
FIG = "visualization/graphs/regression_scatter.png"

def train_regression(df: pd.DataFrame):
    X = df[["ph","turbidity","nitrate"]]; y = df["contamination_level"]
    Xtr, Xte, ytr, yte = train_test_split(X,y,test_size=0.2,random_state=42)
    model = LinearRegression().fit(Xtr,ytr)
    yp = model.predict(Xte)
    rmse = mean_squared_error(yte, yp, squared=False)
    r2 = r2_score(yte, yp)
    return model, (Xte, yte, yp, rmse, r2)

def save_outputs(Xte, yte, yp, rmse, r2):
    os.makedirs("data/processed_data", exist_ok=True)
    out = pd.DataFrame({"y_true":yte, "y_pred":yp})
    out.to_csv(PRED, index=False)
    os.makedirs("visualization/graphs", exist_ok=True)
    plt.figure(figsize=(6,5))
    plt.scatter(yte, yp, s=10)
    plt.xlabel("Actual"); plt.ylabel("Predicted")
    plt.title(f"Linear Regression (RMSE={rmse:.3f}, R2={r2:.3f})")
    plt.tight_layout(); plt.savefig(FIG, dpi=150); plt.close()

if __name__ == "__main__":
    df = pd.read_csv(INP)
    model, (Xte,yte,yp,rmse,r2) = train_regression(df)
    save_outputs(Xte,yte,yp,rmse,r2)
    print(f"[OK] Regression R2={r2:.3f} RMSE={rmse:.3f}")
