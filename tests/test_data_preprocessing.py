import pandas as pd
from scripts.data_preprocessing import load_data, clean_data

def test_load_and_clean():
    df = load_data("data/raw_data/water_quality.csv")
    df2 = clean_data(df)
    assert len(df2) > 0
    for c in ["ph","turbidity","nitrate","contamination_level"]:
        assert df2[c].isna().sum() == 0
