import pandas as pd
from scripts.regression_model import train_regression

def test_regression_runs():
    df = pd.read_csv("data/processed_data/pca_water_quality.csv")
    model, (_Xte,_yte,yp,rmse,r2) = train_regression(df)
    assert hasattr(model, "predict")
    assert len(yp) > 0
