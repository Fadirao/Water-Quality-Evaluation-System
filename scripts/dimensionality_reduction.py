import os, pandas as pd
from sklearn.decomposition import PCA

INP = "data/processed_data/enhanced_water_quality.csv"
OUT = "data/processed_data/pca_water_quality.csv"

def apply_pca(df: pd.DataFrame, cols=None, n=2) -> pd.DataFrame:
    if cols is None: cols = ["ph","turbidity","nitrate"]
    p = PCA(n_components=n, random_state=42)
    Xp = p.fit_transform(df[cols])
    for i in range(n):
        df[f"pca{i+1}"] = Xp[:, i]
    return df

if __name__ == "__main__":
    df = pd.read_csv(INP)
    df = apply_pca(df)
    os.makedirs("data/processed_data", exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"[OK] Saved: {OUT}")
