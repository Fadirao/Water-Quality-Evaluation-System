import pandas as pd

def test_imputed_file_exists_after_run():
    df = pd.read_csv("data/processed_data/imputed_water_quality.csv")
    assert df.isna().sum().sum() == 0
