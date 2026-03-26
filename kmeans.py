from __future__ import annotations

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def assign_kmeans_clusters(
    df: pd.DataFrame,
    feature_columns: list[str],
    n_clusters: int = 3,
    random_state: int = 42,
    n_init: int = 10,
) -> list[int]:
    """Return KMeans labels only (backward-compatible wrapper)."""
    labels, _, _ = assign_kmeans_clusters_with_pca(
        df=df,
        feature_columns=feature_columns,
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=n_init,
    )
    return labels


def assign_kmeans_clusters_with_pca(
    df: pd.DataFrame,
    feature_columns: list[str],
    n_clusters: int = 3,
    random_state: int = 42,
    n_init: int = 10,
) -> tuple[list[int], pd.DataFrame, pd.DataFrame]:
    """
    Assign KMeans clusters and generate 2D PCA coordinates for visualization.

    Returns:
        labels:
            List of cluster labels, one per row in df.

        plot_df:
            DataFrame aligned to df.index with columns:
            - pca_1
            - pca_2
            - cluster

        summary:
            DataFrame with cluster sizes.
    """
    valid_features = [col for col in feature_columns if col in df.columns]

    if len(df) == 0:
        empty_plot = pd.DataFrame(columns=["pca_1", "pca_2", "cluster"], index=df.index)
        empty_summary = pd.DataFrame(columns=["cluster", "size"])
        return [], empty_plot, empty_summary

    if len(df) < n_clusters or not valid_features:
        labels = [0] * len(df)
        plot_df = pd.DataFrame(
            {
                "pca_1": [0.0] * len(df),
                "pca_2": [0.0] * len(df),
                "cluster": labels,
            },
            index=df.index,
        )
        summary = pd.DataFrame({"cluster": [0], "size": [len(df)]})
        return labels, plot_df, summary

    cluster_frame = df[valid_features].copy()

    # Keep only numeric columns for KMeans/PCA
    cluster_frame = cluster_frame.select_dtypes(include="number")

    if cluster_frame.empty:
        labels = [0] * len(df)
        plot_df = pd.DataFrame(
            {
                "pca_1": [0.0] * len(df),
                "pca_2": [0.0] * len(df),
                "cluster": labels,
            },
            index=df.index,
        )
        summary = pd.DataFrame({"cluster": [0], "size": [len(df)]})
        return labels, plot_df, summary

    cluster_frame = cluster_frame.fillna(cluster_frame.median(numeric_only=True))

    # Scale before clustering/PCA
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(cluster_frame)

    # KMeans clustering
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=n_init,
    )
    labels = kmeans.fit_predict(scaled_data)

    # PCA projection for visualization
    pca = PCA(n_components=2, random_state=random_state)
    coords = pca.fit_transform(scaled_data)

    plot_df = pd.DataFrame(
        {
            "pca_1": coords[:, 0],
            "pca_2": coords[:, 1],
            "cluster": labels,
        },
        index=df.index,
    )

    summary = (
        pd.DataFrame({"cluster": labels})
        .groupby("cluster")
        .size()
        .reset_index(name="size")
        .sort_values("cluster")
        .reset_index(drop=True)
    )

    return labels.tolist(), plot_df, summary
