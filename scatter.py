from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from data import cp_map, label_map, load_data, sex_map
from kmeans import assign_kmeans_clusters
from palette import CLUSTER_COLOR_MAP, DISEASE_COLOR_MAP, PATIENT_COLOR


data = load_data()
RELATIONSHIP_NUMERIC_FEATURES = [
    col for col in data.columns if col != "target" and pd.api.types.is_numeric_dtype(data[col])
]


def _default_discrete_map(groups: list[str]) -> dict[str, str]:
    base_colors = px.colors.qualitative.Safe
    return {group: base_colors[idx % len(base_colors)] for idx, group in enumerate(groups)}


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
    show_density_contours: bool,
    patient_input: dict | None,
    matched_patient_ids: list[int] | None = None,
    patient_outcome_label: str | None = None,
):
    patient_input = patient_input or {}
    x_col = x_feature if x_feature in RELATIONSHIP_NUMERIC_FEATURES else RELATIONSHIP_NUMERIC_FEATURES[0]
    y_col = y_feature if y_feature in RELATIONSHIP_NUMERIC_FEATURES else (
        RELATIONSHIP_NUMERIC_FEATURES[1] if len(RELATIONSHIP_NUMERIC_FEATURES) > 1 else RELATIONSHIP_NUMERIC_FEATURES[0]
    )
    if x_col == y_col and len(RELATIONSHIP_NUMERIC_FEATURES) > 1:
        y_col = next(col for col in RELATIONSHIP_NUMERIC_FEATURES if col != x_col)

    plot_df, color_col, palette = _build_color_column(data, color_by)
    color_groups = list(pd.unique(plot_df[color_col].astype(str)))
    color_map = palette or _default_discrete_map(color_groups)
    plot_df[color_col] = plot_df[color_col].astype(str)
    plot_df["patient_id"] = plot_df.index
    x_axis, y_axis = x_col, y_col
    x_label = label_map.get(x_col, x_col.title())
    y_label = label_map.get(y_col, y_col.title())
    title = f"{y_label} vs {x_label}"

    fig = px.scatter(
        plot_df,
        x=x_axis,
        y=y_axis,
        color=color_col,
        color_discrete_map=color_map,
        opacity=0.45,
        labels={
            x_axis: x_label,
            y_axis: y_label,
        },
        title=title,
    )

    matched_ids = set(matched_patient_ids or [])
    if matched_ids:
        for trace in fig.data:
            trace.update(opacity=0.16)

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

    if x_col in patient_input and y_col in patient_input:
        patient_color = DISEASE_COLOR_MAP.get(patient_outcome_label or "", PATIENT_COLOR)
        fig.add_trace(
            go.Scatter(
                x=[patient_input[x_col]],
                y=[patient_input[y_col]],
                mode="markers",
                marker={
                    "symbol": "circle-open-dot",
                    "size": 18,
                    "color": patient_color,
                    "line": {"color": "#111111", "width": 2.4},
                },
                name="Current patient",
                hovertemplate=f"Current patient<br>Outcome group: {patient_outcome_label or 'Unknown'}<extra></extra>",
            )
        )

    if matched_ids:
        matched_df = plot_df[plot_df["patient_id"].isin(matched_ids)].copy()
        if not matched_df.empty:
            matched_df["Outcome Group"] = matched_df["target"].map({0: "No disease", 1: "Disease"}).fillna("Unknown")
            outcome_groups = ["No disease", "Disease", "Unknown"]
            for idx, outcome_group in enumerate(outcome_groups):
                group_df = matched_df[matched_df["Outcome Group"] == outcome_group]
                if group_df.empty:
                    continue
                outcome_color = DISEASE_COLOR_MAP.get(outcome_group, PATIENT_COLOR)
                fig.add_trace(
                    go.Scatter(
                        x=group_df[x_axis],
                        y=group_df[y_axis],
                        mode="markers",
                        marker={
                            "size": 12,
                            "symbol": "circle-open",
                            "color": outcome_color,
                            "line": {"color": outcome_color, "width": 2.2},
                            "opacity": 1.0,
                        },
                        name=f"Rule-matched ({outcome_group})",
                        hovertemplate=f"Rule-matched patient<br>Outcome group: {outcome_group}<extra></extra>",
                        showlegend=True,
                    )
                )

    fig.update_layout(
        template="plotly_white",
        autosize=True,
        margin={"l": 20, "r": 20, "t": 60, "b": 20},
        legend_title_text="Group",
        uirevision="feature-relationships",
    )
    return fig
