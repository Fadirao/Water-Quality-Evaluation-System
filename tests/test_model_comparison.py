import pandas as pd
from scripts.model_comparison import compare_models

def test_compare_models_df():
    df = pd.read_csv("data/processed_data/pca_water_quality.csv")
    res = compare_models(df)
    assert {"model","rmse"}.issubset(set(res.columns))
