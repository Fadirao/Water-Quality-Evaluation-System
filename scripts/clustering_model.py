import os, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
from sklearn.cluster import KMeans

INP = "data/processed_data/pca_water_quality.csv"
FIG = "visualization/graphs/contamination_patterns.png"

def train_kmeans(df: pd.DataFrame, n_clusters=3) -> KMeans:
    X = df[["ph","turbidity","nitrate"]]
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df["cluster"] = km.fit_predict(X)
    return km

def save_plot(df: pd.DataFrame, path=FIG):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.figure(figsize=(7,5))
    sns.scatterplot(data=df, x="pca1", y="pca2", hue="cluster")
    plt.title("K-Means clusters on PCA projection")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()

if __name__ == "__main__":
    df = pd.read_csv(INP)
    km = train_kmeans(df)
    save_plot(df)
    print("[OK] Clustering complete.")
