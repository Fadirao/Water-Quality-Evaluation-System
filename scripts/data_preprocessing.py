import os, pandas as pd, numpy as np
from scipy import stats

RAW = "data/raw_data/water_quality.csv"
CLEAN = "data/processed_data/cleaned_water_quality.csv"
REQ_COLS = ["timestamp","ph","turbidity","nitrate","contamination_level"]

def load_data(path=RAW) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in REQ_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return df

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    for c in ["ph","turbidity","nitrate","contamination_level"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df[["ph","turbidity","nitrate","contamination_level"]] =         df[["ph","turbidity","nitrate","contamination_level"]].fillna(
            df[["ph","turbidity","nitrate","contamination_level"]].mean()
        )
    num = df[["ph","turbidity","nitrate","contamination_level"]]
    z = np.abs(stats.zscore(num, nan_policy="omit"))
    mask = (z < 5).all(axis=1)
    df = df[mask].reset_index(drop=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df

def save(df: pd.DataFrame, path=CLEAN):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"[OK] Saved: {path}  rows={len(df)}")

if __name__ == "__main__":
    df = load_data()
    df = clean_data(df)
    save(df)
