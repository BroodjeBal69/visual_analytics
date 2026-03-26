from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from data import cp_map, label_map, load_data, sex_map
from kmeans import assign_kmeans_clusters, assign_kmeans_clusters_with_pca
from palette import CLUSTER_COLOR_MAP, DISEASE_COLOR_MAP, PATIENT_COLOR


data = load_data()
RELATIONSHIP_NUMERIC_FEATURES = [
    col for col in data.columns if col != "target" and pd.api.types.is_numeric_dtype(data[col])
]


def _nearest_patient_index(patient_input: dict) -> int | None:
    """Approximate selected patient location by nearest record in numeric feature space."""
    if data.empty:
        return None
    usable = [col for col in RELATIONSHIP_NUMERIC_FEATURES if col in patient_input]
    if not usable:
        return None

    frame = data[usable].copy()
    for col in usable:
        std = float(frame[col].std()) or 1.0
        frame[col] = (frame[col] - float(patient_input[col])) / std
    distances = frame.abs().sum(axis=1)
    if distances.empty:
        return None
    return int(distances.idxmin())


def _build_color_column(df: pd.DataFrame, color_by: str) -> tuple[pd.DataFrame, str, dict | None]:
    plot_df = df.copy()
    palette = None

    if color_by == "disease":
        plot_df["Color Group"] = plot_df["target"].map({0: "No disease", 1: "Disease"})
        palette = DISEASE_COLOR_MAP
    elif color_by == "cluster":
        cluster_features = [col for col in RELATIONSHIP_NUMERIC_FEATURES if col in plot_df.columns]
        if len(plot_df) >= 3 and cluster_features:
            labels = assign_kmeans_clusters(plot_df, cluster_features, n_clusters=3, random_state=42, n_init=10)
            plot_df["Color Group"] = [f"Cluster {idx + 1}" for idx in labels]
        else:
            plot_df["Color Group"] = "Cluster 1"
        palette = CLUSTER_COLOR_MAP
    elif color_by == "sex":
        plot_df["Color Group"] = plot_df["sex"].map(sex_map).fillna(plot_df["sex"].astype(str))
    elif color_by == "cp":
        plot_df["Color Group"] = plot_df["cp"].map(cp_map).fillna(plot_df["cp"].astype(str))
    else:
        plot_df["Color Group"] = plot_df["target"].map({0: "No disease", 1: "Disease"})
        palette = DISEASE_COLOR_MAP

    return plot_df, "Color Group", palette


def make_feature_relationships_figure(
    x_feature: str,
    y_feature: str,
    color_by: str,
    view_space: str,
    show_density_contours: bool,
    highlight_patient: bool,
    patient_input: dict,
):
    selected_view = view_space if view_space in {"raw", "pca"} else "raw"

    x_col = x_feature if x_feature in RELATIONSHIP_NUMERIC_FEATURES else RELATIONSHIP_NUMERIC_FEATURES[0]
    y_col = y_feature if y_feature in RELATIONSHIP_NUMERIC_FEATURES else (
        RELATIONSHIP_NUMERIC_FEATURES[1] if len(RELATIONSHIP_NUMERIC_FEATURES) > 1 else RELATIONSHIP_NUMERIC_FEATURES[0]
    )
    if x_col == y_col and len(RELATIONSHIP_NUMERIC_FEATURES) > 1:
        y_col = next(col for col in RELATIONSHIP_NUMERIC_FEATURES if col != x_col)

    if selected_view == "pca":
        labels, pca_df, _ = assign_kmeans_clusters_with_pca(
            data,
            RELATIONSHIP_NUMERIC_FEATURES,
            n_clusters=3,
            random_state=42,
            n_init=10,
        )
        plot_df = data.copy()
        plot_df["pca_1"] = pca_df["pca_1"].values if "pca_1" in pca_df.columns else 0.0
        plot_df["pca_2"] = pca_df["pca_2"].values if "pca_2" in pca_df.columns else 0.0
        plot_df["Cluster PCA"] = [f"Cluster {idx + 1}" for idx in labels]

        if color_by == "cluster":
            color_col = "Cluster PCA"
            palette = CLUSTER_COLOR_MAP
        else:
            plot_df, color_col, palette = _build_color_column(plot_df, color_by)
        x_axis, y_axis = "pca_1", "pca_2"
        x_label, y_label = "PC 1", "PC 2"
        title = "Feature Relationships (PCA projection)"
    else:
        plot_df, color_col, palette = _build_color_column(data, color_by)
        x_axis, y_axis = x_col, y_col
        x_label = label_map.get(x_col, x_col.title())
        y_label = label_map.get(y_col, y_col.title())
        title = f"{y_label} vs {x_label}"

    fig = px.scatter(
        plot_df,
        x=x_axis,
        y=y_axis,
        color=color_col,
        color_discrete_map=palette,
        opacity=0.45,
        labels={
            x_axis: x_label,
            y_axis: y_label,
        },
        title=title,
    )

    if show_density_contours:
        contour = px.density_contour(
            plot_df,
            x=x_axis,
            y=y_axis,
            color=color_col,
            color_discrete_map=palette,
        )
        for trace in contour.data:
            trace.update(showlegend=False, opacity=0.55)
            fig.add_trace(trace)

    if highlight_patient:
        if selected_view == "raw" and x_col in patient_input and y_col in patient_input:
            fig.add_trace(
                go.Scatter(
                    x=[patient_input[x_col]],
                    y=[patient_input[y_col]],
                    mode="markers",
                    marker={"symbol": "star", "size": 14, "color": PATIENT_COLOR, "line": {"color": "white", "width": 1.5}},
                    name="Selected patient",
                    hovertemplate="Selected patient<extra></extra>",
                )
            )
        elif selected_view == "pca":
            nearest_idx = _nearest_patient_index(patient_input)
            if nearest_idx is not None and nearest_idx in plot_df.index:
                px_val = float(plot_df.loc[nearest_idx, "pca_1"])
                py_val = float(plot_df.loc[nearest_idx, "pca_2"])
                fig.add_trace(
                    go.Scatter(
                        x=[px_val],
                        y=[py_val],
                        mode="markers",
                        marker={"symbol": "star", "size": 14, "color": PATIENT_COLOR, "line": {"color": "white", "width": 1.5}},
                        name="Selected patient",
                        hovertemplate="Selected patient (nearest profile)<extra></extra>",
                    )
                )

    fig.update_layout(
        template="plotly_white",
        autosize=True,
        margin={"l": 20, "r": 20, "t": 60, "b": 20},
        legend_title_text="Group",
    )
    return fig
