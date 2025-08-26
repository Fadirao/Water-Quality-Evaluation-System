import pandas as pd

INP = "data/processed_data/imputed_water_quality.csv"

def validate_data(df: pd.DataFrame) -> bool:
    req = ["timestamp","ph","turbidity","nitrate","contamination_level"]
    if not all(c in df.columns for c in req): return False
    if (df["ph"]<=0).any() or (df["ph"]>14).any(): return False
    if (df["turbidity"]<0).any() or (df["nitrate"]<0).any(): return False
    if df["contamination_level"].isna().any(): return False
    return True

if __name__ == "__main__":
    df = pd.read_csv(INP)
    ok = validate_data(df)
    print("[OK] Validation passed." if ok else "[FAIL] Validation failed.")
