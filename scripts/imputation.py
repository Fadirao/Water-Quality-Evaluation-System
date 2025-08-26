import os, pandas as pd
from sklearn.impute import SimpleImputer

INP = "data/processed_data/cleaned_water_quality.csv"
OUT = "data/processed_data/imputed_water_quality.csv"

def impute(df: pd.DataFrame) -> pd.DataFrame:
    num_cols = ["ph","turbidity","nitrate","contamination_level"]
    imp = SimpleImputer(strategy="median")
    df[num_cols] = imp.fit_transform(df[num_cols])
    return df

if __name__ == "__main__":
    df = pd.read_csv(INP)
    df = impute(df)
    os.makedirs("data/processed_data", exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"[OK] Saved: {OUT}")
