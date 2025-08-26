import pandas as pd

INP = "data/processed_data/imputed_water_quality.csv"

def detect_outliers_iqr(df: pd.DataFrame, col: str) -> pd.Series:
    q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - 1.5*iqr, q3 + 1.5*iqr
    return ~df[col].between(lower, upper)

def detect_outliers(df: pd.DataFrame) -> pd.Series:
    mask = pd.Series(False, index=df.index)
    for c in ["ph","turbidity","nitrate","contamination_level"]:
        mask = mask | detect_outliers_iqr(df, c)
    return mask

if __name__ == "__main__":
    df = pd.read_csv(INP)
    out_mask = detect_outliers(df)
    print(f"[INFO] Outliers detected: {out_mask.sum()} / {len(df)}")
