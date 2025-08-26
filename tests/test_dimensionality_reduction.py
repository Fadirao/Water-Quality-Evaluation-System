import pandas as pd
from scripts.dimensionality_reduction import apply_pca

def test_pca_cols():
    df = pd.read_csv("data/processed_data/enhanced_water_quality.csv")
    df2 = apply_pca(df.copy())
    assert "pca1" in df2.columns and "pca2" in df2.columns
