import pandas as pd
from scripts.clustering_model import train_kmeans

def test_kmeans_labels():
    df = pd.read_csv("data/processed_data/pca_water_quality.csv")
    km = train_kmeans(df, n_clusters=3)
    assert hasattr(km, "cluster_centers_")
    assert "cluster" in df.columns
