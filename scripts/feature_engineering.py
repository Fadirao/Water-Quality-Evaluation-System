import os, pandas as pd, numpy as np
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

INP = "data/processed_data/imputed_water_quality.csv"
OUT = "data/processed_data/enhanced_water_quality.csv"

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["hour_of_day"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.weekday
    df["is_weekend"] = (df["day_of_week"]>=5).astype(int)
    df["ph_turbidity"] = df["ph"] * df["turbidity"]
    poly = PolynomialFeatures(degree=2, include_bias=False)
    poly_cols = ["ph","turbidity","nitrate"]
    poly_arr = poly.fit_transform(df[poly_cols])
    poly_names = poly.get_feature_names_out(poly_cols)
    poly_df = pd.DataFrame(poly_arr, columns=poly_names, index=df.index)
    df = pd.concat([df, poly_df], axis=1)
    sc = StandardScaler()
    df[["ph","turbidity","nitrate"]] = sc.fit_transform(df[["ph","turbidity","nitrate"]])
    df["log_ph"] = np.log1p(df["ph"].clip(lower=0) + 1.0)
    return df

if __name__ == "__main__":
    df = pd.read_csv(INP)
    df = build_features(df)
    os.makedirs("data/processed_data", exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"[OK] Saved: {OUT}")
