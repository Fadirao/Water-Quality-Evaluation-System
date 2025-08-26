import pandas as pd
from scripts.outlier_detection import detect_outliers

def test_outliers_mask_type():
    df = pd.read_csv("data/processed_data/imputed_water_quality.csv")
    mask = detect_outliers(df)
    assert mask.shape[0] == len(df)
    assert mask.dtype == bool
