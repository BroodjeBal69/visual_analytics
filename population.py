from __future__ import annotations

import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import html
from plotly.subplots import make_subplots

from data import (
    cp_map,
    exang_map,
    fbs_map,
    label_map,
    load_data,
    restecg_map,
    sex_map,
    slope_map,
    target_map,
)
from kmeans import assign_kmeans_clusters_with_pca
from models.rf import available_features, model_rf
from palette import (
    DEEMPHASIS_GREY,
    DISEASE_COLOR_MAP,
    CLUSTER_COLOR_MAP,
    NEGATIVE_COLOR,
    PATIENT_COLOR,
    POSITIVE_COLOR,
    PROFILE_DIRECTION_COLOR_MAP,
)


data = load_data()
DISPLAY_VALUE_MAPS = {
    "sex": sex_map,
    "cp": cp_map,
    "fbs": fbs_map,
    "restecg": restecg_map,
    "exang": exang_map,
    "slope": slope_map,
    "target": target_map,
}
SIMILARITY_NUMERIC_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak"]
SIMILARITY_CATEGORICAL_FEATURES = ["sex", "cp", "fbs", "restecg", "exang", "slope"]
POPULATION_CONTEXT_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak"]
SIMILAR_PATIENT_COLUMNS = [
    {"name": "Age", "id": "age"},
    {"name": "Sex", "id": "sex"},
    {"name": "Chest Pain", "id": "cp"},
    {"name": "Exercise Angina", "id": "exang"},
    {"name": "Observed Outcome", "id": "target"},
]
RELATIONSHIP_NUMERIC_FEATURES = [
    col for col in data.columns if col != "target" and pd.api.types.is_numeric_dtype(data[col])
]
CLUSTER_PROFILE_FEATURES = [
    feature for feature in ["age", "trestbps", "chol", "thalach", "oldpeak"] if feature in RELATIONSHIP_NUMERIC_FEATURES
]


def _nearest_patient_index(patient_input: dict) -> int | None:
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


def _build_clustered_population_df(n_clusters: int = 3) -> pd.DataFrame:
    labels, _, _ = assign_kmeans_clusters_with_pca(
        data,
        RELATIONSHIP_NUMERIC_FEATURES,
        n_clusters=n_clusters,
        random_state=42,
        n_init=10,
    )
    clustered = data.copy()
    clustered["cluster"] = labels
    return clustered


def get_kmeans_cluster_options(n_clusters: int = 3) -> list[dict]:
    clustered = _build_clustered_population_df(n_clusters=n_clusters)
    cluster_ids = sorted(clustered["cluster"].unique().tolist())
    options = [{"label": "All clusters", "value": "all"}] + [
        {"label": f"Cluster {int(cluster_id) + 1}", "value": int(cluster_id)}
        for cluster_id in cluster_ids
    ]
    return options


def make_cluster_overview_figure(selected_cluster: int | str | None, n_clusters: int = 3):
    clustered = _build_clustered_population_df(n_clusters=n_clusters)
    summary = (
        clustered.groupby("cluster", dropna=False)
        .agg(size=("cluster", "size"), disease_rate=("target", "mean"))
        .reset_index()
        .sort_values("cluster")
    )
    summary["Cluster"] = summary["cluster"].apply(lambda value: f"Cluster {int(value) + 1}")
    summary["Disease Rate"] = summary["disease_rate"].apply(lambda value: f"{value:.0%}")
    if selected_cluster is not None and str(selected_cluster) != "all":
        summary["is_selected"] = summary["cluster"].astype(int) == int(selected_cluster)
    else:
        summary["is_selected"] = False

    fig = px.bar(
        summary,
        x="Cluster",
        y="size",
        color="is_selected",
        text="Disease Rate",
        color_discrete_map={False: DEEMPHASIS_GREY, True: PATIENT_COLOR},
        labels={"size": "Patients", "Cluster": "KMeans clusters"},
        title="KMeans Cluster Sizes (Text = Disease Rate)",
    )
    fig.update_traces(
        textposition="outside",
        hovertemplate="%{x}<br>Patients: %{y}<br>Disease rate: %{text}<extra></extra>",
    )
    fig.update_layout(
        template="plotly_white",
        autosize=True,
        margin={"l": 20, "r": 20, "t": 60, "b": 20},
        showlegend=False,
    )
    return fig


def make_cluster_profile_figure(selected_cluster: int | str | None, n_clusters: int = 3):
    clustered = _build_clustered_population_df(n_clusters=n_clusters)
    if selected_cluster is None or str(selected_cluster) == "all":
        fig = go.Figure()
        fig.add_annotation(
            text="All clusters selected. Choose one cluster to view profile deviations.",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
            font={"size": 14},
        )
        fig.update_layout(
            template="plotly_white",
            autosize=True,
            margin={"l": 20, "r": 20, "t": 60, "b": 20},
            title="Cluster Profile",
            xaxis={"visible": False},
            yaxis={"visible": False},
        )
        return fig

    cluster_id = int(selected_cluster) if selected_cluster is not None else 0
    if cluster_id not in clustered["cluster"].unique():
        cluster_id = int(sorted(clustered["cluster"].unique())[0])

    features = CLUSTER_PROFILE_FEATURES or RELATIONSHIP_NUMERIC_FEATURES[:5]
    cluster_slice = clustered[clustered["cluster"] == cluster_id]
    overall_mean = clustered[features].mean()
    overall_std = clustered[features].std().replace(0, 1)
    cluster_mean = cluster_slice[features].mean()

    profile = pd.DataFrame(
        {
            "feature": features,
            "z_diff": ((cluster_mean - overall_mean) / overall_std).values,
        }
    )
    profile["abs_z_diff"] = profile["z_diff"].abs()
    profile = profile.sort_values("abs_z_diff", ascending=True).tail(5)
    profile["Feature"] = profile["feature"].apply(lambda feature: label_map.get(feature, feature.title()))
    profile["Direction"] = profile["z_diff"].apply(lambda value: "Higher than population" if value >= 0 else "Lower than population")

    fig = px.bar(
        profile,
        x="z_diff",
        y="Feature",
        orientation="h",
        color="Direction",
        color_discrete_map=PROFILE_DIRECTION_COLOR_MAP,
        labels={"z_diff": "Difference vs population (standard deviations)", "Feature": ""},
        title=f"Cluster {cluster_id + 1} Profile (Top Deviations)",
    )
    fig.add_vline(x=0, line_width=1.2, line_dash="dash", line_color=PATIENT_COLOR)
    fig.update_layout(
        template="plotly_white",
        autosize=True,
        margin={"l": 20, "r": 20, "t": 60, "b": 20},
        legend_title_text="",
    )
    return fig


def make_cluster_pca_figure(
    selected_cluster: int | str | None,
    color_by: str = "cluster",
    n_clusters: int = 3,
    patient_input: dict | None = None,
    matched_patient_ids: list[int] | None = None,
):
    labels, pca_df, _ = assign_kmeans_clusters_with_pca(
        data,
        RELATIONSHIP_NUMERIC_FEATURES,
        n_clusters=n_clusters,
        random_state=42,
        n_init=10,
    )

    plot_df = data.copy()
    plot_df["pca_1"] = pca_df["pca_1"].values if "pca_1" in pca_df.columns else 0.0
    plot_df["pca_2"] = pca_df["pca_2"].values if "pca_2" in pca_df.columns else 0.0
    plot_df["Cluster"] = [f"Cluster {int(idx) + 1}" for idx in labels]
    plot_df["Outcome"] = plot_df["target"].map({0: "No Disease", 1: "Disease"})
    plot_df["patient_id"] = plot_df.index

    color_mode = color_by if color_by in {"cluster", "disease"} else "cluster"
    if color_mode == "disease":
        color_col = "Outcome"
        color_map = DISEASE_COLOR_MAP
        title = "PCA Cluster Map (Color: Disease)"
    else:
        color_col = "Cluster"
        color_map = CLUSTER_COLOR_MAP
        title = "PCA Cluster Map (Color: Cluster)"

    plot_df[color_col] = plot_df[color_col].astype(str)
    color_groups = list(pd.unique(plot_df[color_col]))

    fig = px.scatter(
        plot_df,
        x="pca_1",
        y="pca_2",
        color=color_col,
        color_discrete_map=color_map,
        opacity=0.65,
        labels={"pca_1": "PCA 1", "pca_2": "PCA 2"},
        title=title,
    )

    matched_ids = set(matched_patient_ids or [])
    has_specific_cluster = selected_cluster is not None and str(selected_cluster) != "all"
    selected_label = f"Cluster {int(selected_cluster) + 1}" if has_specific_cluster else None

    if selected_label and color_mode == "cluster":
        for trace in fig.data:
            if trace.name == selected_label:
                trace.update(opacity=0.95, marker={"line": {"color": "#111111", "width": 1.2}})
            else:
                trace.update(opacity=0.14)
    elif selected_label:
        for trace in fig.data:
            trace.update(opacity=0.16)
    elif matched_ids:
        for trace in fig.data:
            trace.update(opacity=0.18)

    if has_specific_cluster:
        selected_points = plot_df[plot_df["Cluster"] == selected_label]
        if not selected_points.empty:
            if color_mode != "cluster":
                for idx, group in enumerate(color_groups):
                    group_df = selected_points[selected_points[color_col] == group]
                    if group_df.empty:
                        continue
                    group_color = color_map.get(group, PATIENT_COLOR)
                    fig.add_trace(
                        go.Scatter(
                            x=group_df["pca_1"],
                            y=group_df["pca_2"],
                            mode="markers",
                            marker={
                                "size": 10,
                                "symbol": "circle",
                                "color": group_color,
                                "line": {"color": "#111111", "width": 1.2},
                                "opacity": 0.96,
                            },
                            name=f"Selected: {selected_label}",
                            hovertemplate="Selected cluster member<extra></extra>",
                            showlegend=idx == 0,
                        )
                    )

    nearest_idx = _nearest_patient_index(patient_input or {})
    if nearest_idx is not None and nearest_idx in plot_df.index:
        patient_group = str(plot_df.loc[nearest_idx, color_col])
        patient_color = color_map.get(patient_group, PATIENT_COLOR)
        fig.add_trace(
            go.Scatter(
                x=[float(plot_df.loc[nearest_idx, "pca_1"])],
                y=[float(plot_df.loc[nearest_idx, "pca_2"])],
                mode="markers",
                marker={
                    "symbol": "star",
                    "size": 19,
                    "color": patient_color,
                    "line": {"color": "#111111", "width": 2.4},
                },
                name="Current patient",
                hovertemplate="Current patient (nearest profile)<extra></extra>",
                showlegend=True,
            )
        )

    if matched_ids:
        matched_df = plot_df[plot_df["patient_id"].isin(matched_ids)]
        if not matched_df.empty:
            for idx, group in enumerate(color_groups):
                group_df = matched_df[matched_df[color_col] == group]
                if group_df.empty:
                    continue
                group_color = color_map.get(group, PATIENT_COLOR)
                fig.add_trace(
                    go.Scatter(
                        x=group_df["pca_1"],
                        y=group_df["pca_2"],
                        mode="markers",
                        marker={
                            "size": 12,
                            "symbol": "circle-open",
                            "color": group_color,
                            "line": {"color": group_color, "width": 2.2},
                            "opacity": 1.0,
                        },
                        name="Rule-matched patients",
                        hovertemplate="Rule-matched patient<extra></extra>",
                        showlegend=idx == 0,
                    )
                )

    fig.update_layout(
        template="plotly_white",
        autosize=True,
        margin={"l": 20, "r": 20, "t": 60, "b": 20},
        legend_title_text="Group",
        uirevision="cluster-pca",
    )
    return fig


def make_cluster_explanation_summary(selected_cluster: int | str | None, n_clusters: int = 3) -> list:
    clustered = _build_clustered_population_df(n_clusters=n_clusters)
    if selected_cluster is None or str(selected_cluster) == "all":
        overall_rate = float(clustered["target"].mean()) if len(clustered) else 0.0
        return [
            html.Div("Cluster Explanation", className="fw-semibold mb-2"),
            html.Div("Selected: All clusters", className="small mb-1"),
            html.Div(f"Total clusters: {clustered['cluster'].nunique()}", className="small mb-1"),
            html.Div(f"Overall observed disease rate: {overall_rate:.0%}", className="small mb-1"),
            html.Div(
                "Tip: choose a specific cluster to see its profile deviations and focused interpretation.",
                className="small mb-1",
            ),
            html.Div(
                "This plot shows how patients group based on similarity across multiple features.",
                className="small mb-1",
            ),
            html.Div(
                "Interpretation: this is an unsupervised segment; combine with SHAP and rules for patient-level decisions.",
                className="small text-muted",
            ),
        ]

    cluster_id = int(selected_cluster) if selected_cluster is not None else 0
    if cluster_id not in clustered["cluster"].unique():
        cluster_id = int(sorted(clustered["cluster"].unique())[0])

    features = CLUSTER_PROFILE_FEATURES or RELATIONSHIP_NUMERIC_FEATURES[:5]
    cluster_slice = clustered[clustered["cluster"] == cluster_id]
    share = len(cluster_slice) / len(clustered) if len(clustered) else 0
    disease_rate = float(cluster_slice["target"].mean()) if len(cluster_slice) else 0.0

    overall_mean = clustered[features].mean()
    overall_std = clustered[features].std().replace(0, 1)
    cluster_mean = cluster_slice[features].mean()
    z_scores = ((cluster_mean - overall_mean) / overall_std).sort_values(key=lambda series: series.abs(), ascending=False)
    top_features = z_scores.head(2)
    signature = ", ".join(
        [
            f"{label_map.get(feature, feature.title())} ({'higher' if value >= 0 else 'lower'})"
            for feature, value in top_features.items()
        ]
    ) or "No strong feature differences"

    return [
        html.Div("Cluster Explanation", className="fw-semibold mb-2"),
        html.Div(f"Selected: Cluster {cluster_id + 1}", className="small mb-1"),
        html.Div(f"Population share: {share:.0%}", className="small mb-1"),
        html.Div(f"Observed disease rate: {disease_rate:.0%}", className="small mb-1"),
        html.Div(f"Signature features: {signature}", className="small mb-1"),
        html.Div(
            "This plot shows how patients group based on similarity across multiple features.",
            className="small mb-1",
        ),
        html.Div(
            "Interpretation: this is an unsupervised segment; combine with SHAP and rules for patient-level decisions.",
            className="small text-muted",
        ),
    ]


def format_display_value(feature, value):
    if feature in DISPLAY_VALUE_MAPS:
        return DISPLAY_VALUE_MAPS[feature].get(value, str(value))
    if isinstance(value, float):
        return f"{value:.1f}"
    return str(value)


def build_population_distribution_figure(patient_input, selected_features=None):
    features_to_plot = selected_features or POPULATION_CONTEXT_FEATURES
    if not features_to_plot:
        fig = go.Figure()
        fig.add_annotation(
            text="Select at least one continuous variable to display.",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
            font={"size": 15},
        )
        fig.update_layout(
            template="plotly_white",
            autosize=True,
            margin={"l": 20, "r": 20, "t": 40, "b": 20},
            xaxis={"visible": False},
            yaxis={"visible": False},
            title="Where This Patient Sits in the Population",
        )
        return fig

    feature_titles = {
        "age": "Age Distribution",
        "trestbps": "Resting BP Distribution",
        "chol": "Cholesterol Distribution",
        "thalach": "Max Heart Rate Distribution",
        "oldpeak": "ST Depression Distribution",
    }
    fig = make_subplots(
        rows=1,
        cols=len(features_to_plot),
        subplot_titles=[feature_titles.get(feature, label_map.get(feature, feature.title())) for feature in features_to_plot],
    )
    risk_groups = [
        (0, "No disease", NEGATIVE_COLOR),
        (1, "Disease", POSITIVE_COLOR),
    ]

    for idx, feature in enumerate(features_to_plot, start=1):
        for target_value, label, color in risk_groups:
            subset = data.loc[data["target"] == target_value, feature]
            fig.add_trace(
                go.Histogram(
                    x=subset,
                    name=label,
                    marker_color=color,
                    opacity=0.65,
                    showlegend=idx == 1,
                    nbinsx=18,
                ),
                row=1,
                col=idx,
            )
        fig.add_vline(
            x=float(patient_input[feature]),
            line_color=PATIENT_COLOR,
            line_width=3,
            row=1,
            col=idx,
        )
        fig.update_xaxes(title_text=label_map.get(feature, feature.title()), row=1, col=idx)
        fig.update_yaxes(title_text="Patients" if idx == 1 else "", row=1, col=idx)

    fig.update_layout(
        barmode="overlay",
        template="plotly_white",
        autosize=True,
        margin={"l": 20, "r": 20, "t": 50, "b": 20},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
        title="Where This Patient Sits in the Population",
    )
    return fig


def make_population_distribution_view_figure(
    feature: str | list[str] | None,
    mode: str,
    patient_input: dict | None = None,
    show_patient: bool = False,
    matched_patient_ids: list[int] | None = None,
):
    """Build one or more population distributions split by disease vs no disease."""
    if isinstance(feature, list):
        selected_features = [col for col in feature if col in data.columns and col != "target"]
    elif isinstance(feature, str) and feature in data.columns and feature != "target":
        selected_features = [feature]
    else:
        selected_features = ["age"]

    if not selected_features:
        selected_features = ["age"]

    selected_mode = mode if mode in {"histogram", "density"} else "histogram"

    plot_df = data.copy()
    plot_df["Outcome"] = plot_df["target"].map({0: "No Disease", 1: "Disease"})
    matched_ids = set(matched_patient_ids or [])
    
    # Pre-compute matched_patient_df once, outside the loop
    matched_patient_df = plot_df.loc[plot_df.index.isin(matched_ids)] if matched_ids else pd.DataFrame()
    
    fig = make_subplots(
        rows=1,
        cols=len(selected_features),
        subplot_titles=[label_map.get(col, col.title()) for col in selected_features],
        horizontal_spacing=0.05,
    )

    for idx, selected_feature in enumerate(selected_features, start=1):
        is_numeric = pd.api.types.is_numeric_dtype(plot_df[selected_feature])
        show_legend = idx == 1

        if is_numeric:
            histnorm = "probability density" if selected_mode == "density" else None
            for outcome_label in ["No Disease", "Disease"]:
                subset = plot_df.loc[plot_df["Outcome"] == outcome_label, selected_feature]
                fig.add_trace(
                    go.Histogram(
                        x=subset,
                        name=outcome_label,
                        marker_color=DISEASE_COLOR_MAP.get(outcome_label, PATIENT_COLOR),
                        opacity=0.6,
                        nbinsx=24,
                        histnorm=histnorm,
                        showlegend=show_legend,
                        legendgroup=outcome_label,
                    ),
                    row=1,
                    col=idx,
                )

            if show_patient and patient_input and patient_input.get(selected_feature) is not None:
                fig.add_vline(
                    x=float(patient_input[selected_feature]),
                    line_color=PATIENT_COLOR,
                    line_width=3,
                    row=1,
                    col=idx,
                )

            if matched_ids and not matched_patient_df.empty:
                # Use a binned approach instead of individual vlines for better performance
                matched_values = matched_patient_df[selected_feature].dropna()
                if len(matched_values) > 0:
                    # Add a single shape for the range instead of individual lines
                    bin_edges = np.histogram_bin_edges(matched_values, bins=12)
                    for edge in bin_edges[:-1]:
                        fig.add_vline(
                            x=float(edge),
                            line_color="#4b5563",
                            line_width=0.8,
                            opacity=0.15,
                            layer="above",
                            row=1,
                            col=idx,
                        )
        else:
            # Pre-compute counts for this feature once
            counts = (
                plot_df.groupby([selected_feature, "Outcome"], dropna=False)
                .size()
                .reset_index(name="count")
            )

            for outcome_label in ["No Disease", "Disease"]:
                subset = counts[counts["Outcome"] == outcome_label]
                if subset.empty:
                    continue
                fig.add_trace(
                    go.Bar(
                        x=subset[selected_feature].astype(str),
                        y=subset["count"],
                        name=outcome_label,
                        marker_color=DISEASE_COLOR_MAP.get(outcome_label, PATIENT_COLOR),
                        opacity=0.85,
                        showlegend=show_legend,
                        legendgroup=outcome_label,
                    ),
                    row=1,
                    col=idx,
                )

            if matched_ids and not matched_patient_df.empty:
                # Pre-compute matched counts for this feature
                matched_counts = (
                    matched_patient_df
                    .groupby([selected_feature, "Outcome"], dropna=False)
                    .size()
                    .reset_index(name="count")
                )
                for outcome_label in ["No Disease", "Disease"]:
                    subset = matched_counts[matched_counts["Outcome"] == outcome_label]
                    if subset.empty:
                        continue
                    outcome_color = DISEASE_COLOR_MAP.get(outcome_label, "#4b5563")
                    fig.add_trace(
                        go.Bar(
                            x=subset[selected_feature].astype(str),
                            y=subset["count"],
                            name=f"Rule-matched ({outcome_label})",
                            marker={
                                "color": "rgba(0,0,0,0)",
                                "line": {"color": outcome_color, "width": 1.4},
                                "pattern": {"shape": "/", "fgcolor": outcome_color, "solidity": 0.35},
                            },
                            opacity=0.95,
                            showlegend=show_legend,
                            legendgroup=f"rule-{outcome_label}",
                        ),
                        row=1,
                        col=idx,
                    )

        fig.update_xaxes(title_text=label_map.get(selected_feature, selected_feature.title()), row=1, col=idx)
        fig.update_yaxes(title_text="Patients" if selected_mode == "histogram" else "Density", row=1, col=idx)

    fig.update_layout(
        title=(
            f"{selected_mode.title()} by Outcome"
            if len(selected_features) > 1
            else f"{label_map.get(selected_features[0], selected_features[0].title())}: {selected_mode.title()} by Outcome"
        ),
        barmode="overlay",
        template="plotly_white",
        autosize=True,
        margin={"l": 20, "r": 20, "t": 60, "b": 20},
        legend_title_text="Outcome",
    )
    return fig


def make_feature_relationships_summary(x_feature: str, y_feature: str) -> list:
    x_col = x_feature if x_feature in RELATIONSHIP_NUMERIC_FEATURES else RELATIONSHIP_NUMERIC_FEATURES[0]
    y_col = y_feature if y_feature in RELATIONSHIP_NUMERIC_FEATURES else (
        RELATIONSHIP_NUMERIC_FEATURES[1] if len(RELATIONSHIP_NUMERIC_FEATURES) > 1 else RELATIONSHIP_NUMERIC_FEATURES[0]
    )
    if x_col == y_col and len(RELATIONSHIP_NUMERIC_FEATURES) > 1:
        y_col = next(col for col in RELATIONSHIP_NUMERIC_FEATURES if col != x_col)

    corr = float(data[x_col].corr(data[y_col])) if data[x_col].std() and data[y_col].std() else 0.0
    disease = data[data["target"] == 1][[x_col, y_col]].mean()
    no_disease = data[data["target"] == 0][[x_col, y_col]].mean()
    separation = float(np.linalg.norm(disease.values - no_disease.values))

    if abs(corr) >= 0.6:
        corr_note = "strong"
    elif abs(corr) >= 0.3:
        corr_note = "moderate"
    else:
        corr_note = "weak"

    if separation >= 40:
        sep_note = "clear"
    elif separation >= 15:
        sep_note = "moderate"
    else:
        sep_note = "limited"

    clinical_note = (
        "Clinical note: disease and no-disease groups show visible separation in this feature space."
        if sep_note in {"clear", "moderate"}
        else "Clinical note: this pair alone has limited group separation; use with other features/rules."
    )

    return [
        html.Div("Quick Insight", className="fw-semibold mb-2"),
        html.Div(f"Correlation ({label_map.get(x_col, x_col.title())} vs {label_map.get(y_col, y_col.title())}): {corr:+.2f} ({corr_note})", className="small mb-1"),
        html.Div(f"Group separation distance: {separation:.1f} ({sep_note})", className="small mb-1"),
        html.Div(clinical_note, className="small text-muted"),
    ]


def build_population_context(patient_input, selected_features=None):
    distance = 0
    for feature in SIMILARITY_NUMERIC_FEATURES:
        feature_std = float(data[feature].std()) or 1.0
        distance += (data[feature] - float(patient_input[feature])).abs() / feature_std
    for feature in SIMILARITY_CATEGORICAL_FEATURES:
        distance += (data[feature] != patient_input[feature]).astype(float)

    similar_patients = data.assign(_distance=distance).sort_values("_distance").head(8).copy()
    similar_rf_risk = float(model_rf.predict_proba(similar_patients[available_features])[:, 1].mean())
    similar_observed_rate = float(similar_patients["target"].mean())

    summary = [
        html.H5("Population Context", className="mb-3"),
        html.Div(f"Most similar cohort size: {len(similar_patients)} patients", className="mb-1"),
        html.Div(f"Average RF risk in similar patients: {similar_rf_risk:.1%}", className="mb-1"),
        html.Div(f"Observed disease rate in similar patients: {similar_observed_rate:.1%}", className="mb-0"),
    ]

    similar_records = [
        {
            "age": int(row.age),
            "sex": format_display_value("sex", row.sex),
            "cp": format_display_value("cp", row.cp),
            "exang": format_display_value("exang", row.exang),
            "target": format_display_value("target", row.target),
        }
        for row in similar_patients.itertuples()
    ]

    return summary, similar_records, build_population_distribution_figure(patient_input, selected_features)


def make_population_summary_alert(children, component_id: str = "population-summary"):
    return dbc.Alert(children, id=component_id, color="info", className="mb-3")
