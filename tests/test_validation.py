import pandas as pd
from scripts.validation import validate_data

def test_validate_ok():
    df = pd.read_csv("data/processed_data/imputed_water_quality.csv")
    assert validate_data(df) is True
