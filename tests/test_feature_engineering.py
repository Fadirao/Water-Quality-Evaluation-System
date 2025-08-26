import pandas as pd
from scripts.feature_engineering import build_features

def test_features_added():
    df = pd.read_csv("data/processed_data/imputed_water_quality.csv")
    out = build_features(df.copy())
    assert "hour_of_day" in out.columns
    assert "ph_turbidity" in out.columns
