import pandas as pd
from scripts.ensemble_models import run_ensembles

def test_ensembles_dict():
    df = pd.read_csv("data/processed_data/pca_water_quality.csv")
    scores = run_ensembles(df)
    assert isinstance(scores, dict)
    assert all(isinstance(v, float) for v in scores.values())
