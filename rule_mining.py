from __future__ import annotations

import math

import dash_bootstrap_components as dbc
import pandas as pd
from dash import html
from mlxtend.frequent_patterns import apriori, association_rules
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from data import cp_map, exang_map, fbs_map, restecg_map, slope_map, target_map
from models.rf import X_train, y_train
from palette import (
    DISEASE_COLOR_MAP,
    DIVERGING_COLOR_SCALE,
    NEGATIVE_COLOR,
    NEUTRAL_TEXT_COLOR,
    PATIENT_COLOR,
    POSITIVE_COLOR,
)


# ===== CONFIGURATION: Binning and Feature Mapping =====
# Initialize bins and labels for discretization
AGE_BINS = [-math.inf, 45, 55, 65, math.inf]
AGE_LABELS = ["<45", "45-54", "55-64", "65+"]
TRESTBPS_BINS = [-math.inf, 120, 140, 160, math.inf]
TRESTBPS_LABELS = ["<=120", "121-140", "141-160", ">160"]
CHOL_BINS = [-math.inf, 200, 240, math.inf]
CHOL_LABELS = ["<=200", "201-240", ">240"]
THALACH_BINS = [-math.inf, 120, 150, math.inf]
THALACH_LABELS = ["<=120", "121-150", ">150"]
OLDPEAK_BINS = [-math.inf, 0, 1, 2, math.inf]
OLDPEAK_LABELS = ["<=0", "0-1", "1-2", ">2"]

RULE_FEATURE_COLUMNS = [
    "age_bin",
    "sex",
    "cp",
    "trestbps_bin",
    "chol_bin",
    "fbs",
    "restecg",
    "thalach_bin",
    "exang",
    "oldpeak_bin",
    "slope",
    "target",
]
PLOT_RULE_LIMIT = 100
RULE_FIG_MIN_HEIGHT = 400
RULE_FIG_MAX_HEIGHT = 1000

ITEM_LABELS = {
    "target=1": "heart disease",
    "target=0": "no heart disease",
}

VALUE_LABEL_MAPS = {
    "sex": {0: "Female", 1: "Male", "0": "Female", "1": "Male"},
    "cp": cp_map,
    "fbs": {0: "<= 120 mg/dl", 1: "> 120 mg/dl", "0": "<= 120 mg/dl", "1": "> 120 mg/dl"},
    "restecg": restecg_map,
    "exang": exang_map,
    "slope": slope_map,
    "target": target_map,
}

FEATURE_LABELS = {
    "age_bin": "age",
    "sex": "sex",
    "cp": "chest pain type",
    "trestbps_bin": "resting blood pressure",
    "chol_bin": "cholesterol",
    "fbs": "fasting blood sugar",
    "restecg": "resting ECG",
    "thalach_bin": "max heart rate",
    "exang": "exercise angina",
    "oldpeak_bin": "ST depression",
    "slope": "ST slope",
    "target": "heart disease",
}


# ===== Base rules on traning set ======
def _build_train_rule_source_df() -> pd.DataFrame:
    train_df = X_train.copy()
    train_df["target"] = y_train
    return train_df.copy()


TRAIN_RULE_SOURCE_DF = _build_train_rule_source_df()


# ===== Binning and Feature Encoding =====
# Functions for binning continuous variables and discretizing dataframes
def _cut_value(value, bins, labels) -> str:
    if pd.isna(value):
        return "Unknown"
    return str(pd.cut([value], bins=bins, labels=labels, include_lowest=True)[0])


def bin_age(value) -> str:
    return _cut_value(value, AGE_BINS, AGE_LABELS)


def bin_trestbps(value) -> str:
    return _cut_value(value, TRESTBPS_BINS, TRESTBPS_LABELS)


def bin_chol(value) -> str:
    return _cut_value(value, CHOL_BINS, CHOL_LABELS)


def bin_thalach(value) -> str:
    return _cut_value(value, THALACH_BINS, THALACH_LABELS)


def bin_oldpeak(value) -> str:
    return _cut_value(value, OLDPEAK_BINS, OLDPEAK_LABELS)


def discretize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    discretized = df.copy()
    discretized["age_bin"] = discretized["age"].apply(bin_age)
    discretized["trestbps_bin"] = discretized["trestbps"].apply(bin_trestbps)
    discretized["chol_bin"] = discretized["chol"].apply(bin_chol)
    discretized["thalach_bin"] = discretized["thalach"].apply(bin_thalach)
    discretized["oldpeak_bin"] = discretized["oldpeak"].apply(bin_oldpeak)

    for feature in ["sex", "cp", "fbs", "restecg", "exang", "slope", "target"]:
        if feature in discretized.columns:
            discretized[feature] = discretized[feature].astype(str)

    return discretized[RULE_FEATURE_COLUMNS].copy()


def build_rule_dataset(df: pd.DataFrame | None = None) -> pd.DataFrame:
    source_df = TRAIN_RULE_SOURCE_DF if df is None else df.copy()
    return discretize_dataframe(source_df)


def _build_discretized_rule_source_df() -> pd.DataFrame:
    discretized = discretize_dataframe(TRAIN_RULE_SOURCE_DF)
    discretized["patient_id"] = TRAIN_RULE_SOURCE_DF.index.astype(int)
    return discretized


DISCRETIZED_RULE_SOURCE_DF = _build_discretized_rule_source_df()


# ===== RULE MINING: Apriori Algorithm and Rule Construction =====
# Build and filter association rules from training data
def build_rules(
    df: pd.DataFrame | None = None,
    min_support: float = 0.05,
    min_confidence: float = 0.75,
) -> pd.DataFrame:
    rule_df = build_rule_dataset(df)
    one_hot = pd.get_dummies(rule_df.astype(str), prefix_sep="=")

    freq_items = apriori(one_hot, min_support=min_support, use_colnames=True)
    if freq_items.empty:
        return pd.DataFrame(columns=["antecedents", "consequents", "support", "confidence", "lift"])

    rules = association_rules(freq_items, metric="confidence", min_threshold=min_confidence)
    if rules.empty:
        return pd.DataFrame(columns=["antecedents", "consequents", "support", "confidence", "lift"])

    rules = rules[rules["consequents"].apply(lambda items: {str(item) for item in items} in ({"target=1"}, {"target=0"}))].copy()
    rules["antecedents"] = rules["antecedents"].apply(lambda items: sorted(str(item) for item in items))
    rules["consequents"] = rules["consequents"].apply(lambda items: sorted(str(item) for item in items))
    rules = rules[["antecedents", "consequents", "support", "confidence", "lift"]]
    return rules.reset_index(drop=True)


RULES_DF = build_rules()


def build_plot_rules(rules_df: pd.DataFrame | None = None, limit: int = PLOT_RULE_LIMIT) -> pd.DataFrame:
    active_rules = RULES_DF if rules_df is None else rules_df.copy()
    if active_rules.empty:
        return active_rules

    ranked_rules = active_rules.copy()
    ranked_rules["rule_len"] = ranked_rules["antecedents"].apply(len)

    # Separate rules by outcome 
    target_1_rules = ranked_rules[ranked_rules["consequents"].apply(lambda items: items == ["target=1"])]
    target_0_rules = ranked_rules[ranked_rules["consequents"].apply(lambda items: items == ["target=0"])]

    # Filter both outcomes, but be more inclusive with disease rules (lower confidence threshold) 
    # -> high stake domain
    filtered_target_1 = target_1_rules[
        (target_1_rules["lift"] > 1.0) &
        (target_1_rules["confidence"] > 0.50) &  # Higher than 50% for disease rules
        (target_1_rules["rule_len"] <= 3)
    ].copy()

    filtered_target_0 = target_0_rules[
        (target_0_rules["lift"] > 1.0) &
        (target_0_rules["confidence"] >= 0.75) &  # Higher threshold for non-disease rules
        (target_0_rules["rule_len"] <= 3)
    ].copy()

    # Combine filtered rules
    plot_rules = pd.concat([filtered_target_1, filtered_target_0])

    # If either outcome is completely missing, add fallback rules (but filtered for quality)
    if filtered_target_1.empty and not target_1_rules.empty:
        fallback_1 = target_1_rules[
            (target_1_rules["lift"] > 0.9) &
            (target_1_rules["confidence"] > 0.50)
        ].sort_values(by=["lift", "confidence", "support"], ascending=False).head(15)
        plot_rules = pd.concat([plot_rules, fallback_1])

    if filtered_target_0.empty and not target_0_rules.empty:
        fallback_0 = target_0_rules[
            (target_0_rules["lift"] > 0.9) &
            (target_0_rules["confidence"] >= 0.60)
        ].sort_values(by=["lift", "confidence", "support"], ascending=False).head(15)
        plot_rules = pd.concat([plot_rules, fallback_0])

    plot_rules = (
        plot_rules
        .assign(
            _ante_key=plot_rules["antecedents"].apply(tuple),
            _cons_key=plot_rules["consequents"].apply(tuple),
        )
        .drop_duplicates(subset=["_ante_key", "_cons_key", "support", "confidence", "lift"])
        .drop(columns=["_ante_key", "_cons_key"])
    )

    sorted_rules = plot_rules.sort_values(
        by=["lift", "confidence", "support"],
        ascending=False,
    )

    if limit <= 0:
        return sorted_rules.head(0).reset_index(drop=True)

    positive_sorted = sorted_rules[sorted_rules["consequents"].apply(lambda items: items == ["target=1"])]
    negative_sorted = sorted_rules[sorted_rules["consequents"].apply(lambda items: items == ["target=0"])]

    if positive_sorted.empty or negative_sorted.empty:
        return sorted_rules.head(limit).reset_index(drop=True)

    # Balance outcomes: give priority to disease rules since they're clinically important
    # Use 55/45 split in favor of disease detection rules
    quota_disease = max(2, int(limit * 0.55))
    quota_non_disease = max(2, limit - quota_disease)

    balanced = pd.concat([
        positive_sorted.head(quota_disease),
        negative_sorted.head(quota_non_disease),
    ])

    if len(balanced) < limit:
        remainder = sorted_rules.loc[~sorted_rules.index.isin(balanced.index)].head(limit - len(balanced))
        balanced = pd.concat([balanced, remainder])

    return balanced.head(limit).reset_index(drop=True)


PLOT_RULES_DF = build_plot_rules(RULES_DF)


# ===== RULE MATCHING & RANKING: Patient-Specific Rule Selection =====
def filter_rules_by_outcome(rules_df: pd.DataFrame | None = None, outcome_mode: str = "all") -> pd.DataFrame:
    active_rules = RULES_DF if rules_df is None else rules_df.copy()
    if active_rules.empty or outcome_mode == "all":
        return active_rules
    if outcome_mode == "target=1":
        return active_rules[active_rules["consequents"].apply(lambda items: items == ["target=1"])].reset_index(drop=True)
    if outcome_mode == "target=0":
        return active_rules[active_rules["consequents"].apply(lambda items: items == ["target=0"])].reset_index(drop=True)
    return active_rules.reset_index(drop=True)


def discretize_patient(patient_dict: dict) -> dict:
    return {
        "age_bin": bin_age(patient_dict.get("age")),
        "sex": str(patient_dict.get("sex")),
        "cp": str(patient_dict.get("cp")),
        "trestbps_bin": bin_trestbps(patient_dict.get("trestbps")),
        "chol_bin": bin_chol(patient_dict.get("chol")),
        "fbs": str(patient_dict.get("fbs")),
        "restecg": str(patient_dict.get("restecg")),
        "thalach_bin": bin_thalach(patient_dict.get("thalach")),
        "exang": str(patient_dict.get("exang")),
        "oldpeak_bin": bin_oldpeak(patient_dict.get("oldpeak")),
        "slope": str(patient_dict.get("slope")),
    }


def patient_to_items(patient_dict: dict) -> set[str]:
    return {
        f"age_bin={patient_dict['age_bin']}",
        f"sex={patient_dict['sex']}",
        f"cp={patient_dict['cp']}",
        f"trestbps_bin={patient_dict['trestbps_bin']}",
        f"chol_bin={patient_dict['chol_bin']}",
        f"fbs={patient_dict['fbs']}",
        f"restecg={patient_dict['restecg']}",
        f"thalach_bin={patient_dict['thalach_bin']}",
        f"exang={patient_dict['exang']}",
        f"oldpeak_bin={patient_dict['oldpeak_bin']}",
        f"slope={patient_dict['slope']}",
    }


def matching_rules(rules_df: pd.DataFrame, patient_items: set[str]) -> pd.DataFrame:
    matches = []
    for _, row in rules_df.iterrows():
        if set(row["antecedents"]).issubset(patient_items):
            matches.append(row.to_dict())
    if not matches:
        return pd.DataFrame(columns=rules_df.columns)
    return pd.DataFrame(matches)


def rank_rules(matches_df: pd.DataFrame) -> pd.DataFrame:
    if matches_df.empty:
        return matches_df

    ranked = matches_df.copy()
    ranked["rule_len"] = ranked["antecedents"].apply(len)
    ranked["score"] = (
        0.5 * ranked["confidence"] +
        0.4 * ranked["lift"] -
        0.05 * ranked["rule_len"]
    )
    return ranked.sort_values(by=["score", "confidence", "lift"], ascending=False)


def humanize_item(item: str) -> str:
    if item in ITEM_LABELS:
        return ITEM_LABELS[item]

    if "=" not in item:
        return item.replace("_", " ")

    feature, value = item.split("=", 1)
    feature_label = FEATURE_LABELS.get(feature, feature.replace("_", " "))
    value_label = VALUE_LABEL_MAPS.get(feature, {}).get(value, VALUE_LABEL_MAPS.get(feature, {}).get(_safe_int(value), value))
    return f"{feature_label}: {value_label}"


def _safe_int(value: str):
    try:
        return int(value)
    except (TypeError, ValueError):
        return value


def rule_to_text(row: pd.Series) -> dict:
    conditions = [humanize_item(item) for item in row["antecedents"]]
    outcome = humanize_item(row["consequents"][0])
    return {
        "conditions": conditions,
        "if_text": " and ".join(conditions),
        "then_text": outcome,
        "confidence": float(row["confidence"]),
        "lift": float(row["lift"]),
        "support": float(row["support"]),
    }


def _build_rule_sentence(rule: dict):
    parts = [
        html.Span("IF", className="badge text-bg-primary me-2"),
    ]

    for idx, condition in enumerate(rule["conditions"]):
        parts.append(html.Span(condition, className="me-1 fw-semibold"))
        if idx < len(rule["conditions"]) - 1:
            parts.append(html.Span("AND", className="badge text-bg-secondary mx-2"))

    parts.extend(
        [
            html.Span("THEN", className="badge text-bg-info ms-2 me-2"),
            html.Span(rule["then_text"], className="fw-semibold text-info-emphasis"),
        ]
    )
    return html.Div(parts, className="mb-2")


def _build_rule_card(row: pd.Series, rule_index: int | None = None) -> dbc.Card:
    """Build a single rule card from a rule row. If rule_index is None, no index is shown."""
    rule = rule_to_text(row)
    title = f"Rule {rule_index + 1}" if rule_index is not None else "Selected Rule"
    return dbc.Card(
        dbc.CardBody(
            [
                html.H6(title, className="card-title"),
                _build_rule_sentence(rule),
                html.Small(
                    f"Support: {rule['support']:.2f} | "
                    f"Confidence: {rule['confidence']:.2f} | "
                    f"Lift: {rule['lift']:.2f}"
                ),
            ]
        ),
        className="mb-2 shadow-sm",
    )


def build_rule_detail_card(row: pd.Series, label: str | None = None):
    rule = rule_to_text(row)
    title = "Selected Rule" if label is None else label
    return dbc.Card(
        dbc.CardBody(
            [
                html.H6(title, className="card-title"),
                _build_rule_sentence(rule),
                html.Small(
                    f"Support: {rule['support']:.2f} | "
                    f"Confidence: {rule['confidence']:.2f} | "
                    f"Lift: {rule['lift']:.2f}"
                ),
            ]
        ),
        className="shadow-sm",
    )


def get_ranked_rules_for_patient(patient_input: dict, rules_df: pd.DataFrame | None = None) -> pd.DataFrame:
    patient_items = patient_to_items(discretize_patient(patient_input))
    active_rules = RULES_DF if rules_df is None else rules_df
    return rank_rules(matching_rules(active_rules, patient_items))


def build_rule_cards(ranked_rules: pd.DataFrame, top_n: int = 3):
    if ranked_rules.empty:
        return dbc.Alert("No strong matching rules found for this patient.", color="secondary")
    return [_build_rule_card(row, idx) for idx, (_, row) in enumerate(ranked_rules.head(top_n).iterrows())]


def build_selected_rule_details(rules_df: pd.DataFrame, selected_indices: list[int] | None):
    if not selected_indices:
        return dbc.Alert(
            "Select one or more bubbles in the rule distribution to inspect the corresponding rules.",
            color="light",
            className="mb-0",
        )

    cards = []
    for rule_index in selected_indices:
        if rule_index < 0 or rule_index >= len(rules_df):
            continue
        row = rules_df.iloc[rule_index]
        cards.append(_build_rule_card(row, rule_index))
    return cards or dbc.Alert("No valid rules selected.", color="light", className="mb-0")


def _rule_signature(row) -> tuple:
    return (
        tuple(row["antecedents"]),
        tuple(row["consequents"]),
        round(float(row["support"]), 6),
        round(float(row["confidence"]), 6),
        round(float(row["lift"]), 6),
    )


def get_patient_rule_selection(
    patient_input: dict,
    selected_rule_filters: list[str] | None,
    col_sort: str,
    rule_count: int,
    top_n: int = 5,
) -> list[int]:
    filtered_rules = filter_rules_by_antecedents(
        build_plot_rules(RULES_DF),
        selected_rule_filters,
    )
    matrix_rules = get_matrix_rules(filtered_rules, top_n=rule_count, col_sort=col_sort)
    if matrix_rules.empty:
        return []

    ranked = get_ranked_rules_for_patient(patient_input, rules_df=matrix_rules)
    if ranked.empty:
        return []

    selected_signatures = {
        _rule_signature(row)
        for _, row in ranked.head(top_n).iterrows()
    }

    selected_indices = []
    for idx, row in matrix_rules.reset_index(drop=True).iterrows():
        if _rule_signature(row) in selected_signatures:
            selected_indices.append(int(idx))
    return sorted(selected_indices)


# ===== VISUALIZATION: Rule Tables and Charts =====
def build_top_rules_table_data(rules_df: pd.DataFrame | None = None, limit: int = 50) -> pd.DataFrame:
    active_rules = PLOT_RULES_DF if rules_df is None else rules_df.copy()
    if active_rules.empty:
        return pd.DataFrame(columns=["rule", "if", "then", "support", "confidence", "lift"])

    top_rules = active_rules.sort_values(
        by=["lift", "confidence", "support"],
        ascending=False,
    ).head(limit).reset_index(drop=True)

    rows = []
    for idx, row in top_rules.iterrows():
        rule = rule_to_text(row)
        rows.append(
            {
                "rule": f"Rule {idx + 1}",
                "if": rule["if_text"],
                "then": rule["then_text"],
                "support": round(rule["support"], 3),
                "confidence": round(rule["confidence"], 3),
                "lift": round(rule["lift"], 3),
            }
        )
    return pd.DataFrame(rows)


def _add_empty_figure_annotation(fig: go.Figure, title: str, height: int = 320) -> go.Figure:
    """Add standard empty rules annotation to a figure."""
    fig.add_annotation(
        text="No rules available for the current filter.",
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False,
        font={"size": 15},
    )
    fig.update_layout(
        title=title,
        template="plotly_white",
        height=height,
        xaxis={"visible": False},
        yaxis={"visible": False},
        margin={"l": 20, "r": 20, "t": 50, "b": 20},
    )
    return fig


def make_rules_lollipop_figure(rules_df: pd.DataFrame | None = None, limit: int = 20) -> go.Figure:
    active_rules = PLOT_RULES_DF if rules_df is None else rules_df.copy()
    fig = go.Figure()

    if active_rules.empty:
        return _add_empty_figure_annotation(fig, "Rule Lift Overview", 320)

    top_rules = active_rules.sort_values(
        by=["lift", "confidence", "support"],
        ascending=False,
    ).head(limit).copy()
    top_rules = top_rules.iloc[::-1]
    labels = [f"Rule {idx + 1}" for idx in range(len(top_rules))]
    hover_text = []
    for _, row in top_rules.iterrows():
        rule = rule_to_text(row)
        hover_text.append(
            f"If {rule['if_text']}<br>Then {rule['then_text']}<br>"
            f"Lift: {rule['lift']:.2f}<br>Confidence: {rule['confidence']:.2f}<br>Support: {rule['support']:.2f}"
        )

    for y_value, lift_value in zip(labels, top_rules["lift"]):
        fig.add_shape(
            type="line",
            x0=0,
            x1=float(lift_value),
            y0=y_value,
            y1=y_value,
            line={"color": NEGATIVE_COLOR, "width": 2},
        )

    fig.add_trace(
        go.Scatter(
            x=top_rules["lift"],
            y=labels,
            mode="markers",
            marker={"size": 12, "color": POSITIVE_COLOR},
            text=hover_text,
            hovertemplate="%{text}<extra></extra>",
            showlegend=False,
        )
    )
    fig.update_layout(
        title="Rule Lift Overview",
        template="plotly_white",
        height=min(RULE_FIG_MAX_HEIGHT, max(RULE_FIG_MIN_HEIGHT, 28 * len(top_rules) + 100)),
        margin={"l": 90, "r": 20, "t": 50, "b": 30},
        xaxis_title="Lift",
        yaxis_title="",
    )
    return fig


# ===== RULE MATRIX: Visualization Helpers and Layout =====
# Helper functions for the association rule matrix visualization
def _feature_from_item(item: str) -> str:
    return item.split("=", 1)[0] if "=" in item else item


def _matrix_row_label(item: str, collapse_rows: bool) -> str:
    if not collapse_rows:
        return humanize_item(item)
    if str(item).startswith("target="):
        return humanize_item(item)
    return FEATURE_LABELS.get(_feature_from_item(item), _feature_from_item(item))


def _is_outcome_row_record(record: dict) -> bool:
    key = str(record.get("key", ""))
    feature = str(record.get("feature", ""))
    return key.startswith("target=") or feature == FEATURE_LABELS.get("target", "target")


def get_matrix_rules(
    rules_df: pd.DataFrame | None = None,
    top_n: int = 25,
    col_sort: str = "lift",
) -> pd.DataFrame:
    active_rules = PLOT_RULES_DF if rules_df is None else rules_df.copy()
    if active_rules.empty:
        return active_rules

    sortable = active_rules.copy()
    sortable["rule_len"] = sortable["antecedents"].apply(len)
    sort_column = col_sort if col_sort in {"lift", "confidence", "support", "rule_len"} else "lift"
    sortable = sortable.sort_values(
        by=[sort_column, "lift", "confidence", "support"],
        ascending=False,
    )

    if top_n <= 0:
        return sortable.head(0).reset_index(drop=True)

    # Keep both outcomes visible by selecting a balanced subset when possible.
    positive_rules = sortable[sortable["consequents"].apply(lambda items: items == ["target=1"])]
    negative_rules = sortable[sortable["consequents"].apply(lambda items: items == ["target=0"])]

    quota = max(1, top_n // 2)
    selected = pd.concat(
        [
            positive_rules.head(quota),
            negative_rules.head(quota),
        ]
    )

    if len(selected) < top_n:
        remainder = sortable.loc[~sortable.index.isin(selected.index)].head(top_n - len(selected))
        selected = pd.concat([selected, remainder])

    return selected.head(top_n).reset_index(drop=True)


def make_rule_matrix_figure(
    rules_df: pd.DataFrame | None = None,
    top_n: int = 25,
    row_sort: str = "frequency",
    col_sort: str = "lift",
    collapse_rows: bool = False,
) -> go.Figure:
    plot_rules = get_matrix_rules(rules_df, top_n=top_n, col_sort=col_sort)
    fig = make_subplots(
        rows=1,
        cols=2,
        column_widths=[0.07, 0.93],
        shared_yaxes=True,
        horizontal_spacing=0.01,
    )

    if plot_rules.empty:
        return _add_empty_figure_annotation(fig, "Association Rule Matrix", 320)

    row_membership = {}
    rule_metric_text = []
    for idx, row in plot_rules.iterrows():
        rule_metric_text.append(f"{row[col_sort]:.2f}" if col_sort in row else f"{row['lift']:.2f}")
        items = list(row["antecedents"]) + list(row["consequents"])
        row_keys = {_feature_from_item(item) if collapse_rows and not str(item).startswith("target=") else item for item in items}
        for key in row_keys:
            row_membership.setdefault(key, []).append(idx + 1)

    row_records = []
    for key, columns in row_membership.items():
        label = _matrix_row_label(key, collapse_rows)
        row_records.append(
            {
                "key": key,
                "label": label,
                "count": len(columns),
                "columns": sorted(columns),
                "feature": FEATURE_LABELS.get(_feature_from_item(key), _feature_from_item(key)),
            }
        )

    if row_sort == "alphabetical":
        row_records.sort(key=lambda record: record["label"])
    elif row_sort == "feature":
        row_records.sort(key=lambda record: (record["feature"], record["label"]))
    else:
        row_records.sort(key=lambda record: (-record["count"], record["label"]))

    if not collapse_rows and row_sort == "feature":
        grouped_rows = []
        current_feature = None
        for record in row_records:
            if record["feature"] != current_feature:
                current_feature = record["feature"]
                grouped_rows.append(
                    {
                        "key": f"header::{current_feature}",
                        "label": f"<b>{current_feature}</b>",
                        "count": 0,
                        "columns": [],
                        "feature": current_feature,
                        "is_header": True,
                    }
                )
            grouped_rows.append(
                {
                    **record,
                    "label": f"    {record['label'].split(': ', 1)[-1]}",
                    "is_header": False,
                }
            )
        row_records = grouped_rows
    else:
        row_records = [{**record, "is_header": False} for record in row_records]

    non_outcome_rows = [record for record in row_records if not _is_outcome_row_record(record)]
    outcome_rows = [record for record in row_records if _is_outcome_row_record(record)]
    row_records = non_outcome_rows + outcome_rows

    y_positions = {
        record["key"]: len(row_records) - idx - 1
        for idx, record in enumerate(row_records)
        if not record["is_header"]
    }

    for row_index in range(len(row_records)):
        if row_index % 2 == 0:
            fig.add_hrect(
                y0=row_index - 0.5,
                y1=row_index + 0.5,
                fillcolor="rgba(0,0,0,0.035)",
                line_width=0,
                layer="below",
                row=1,
                col=2,
            )

    max_count = max((record["count"] for record in row_records if not record["is_header"]), default=1)
    fig.add_trace(
        go.Bar(
            x=[-(record["count"] / max_count) if max_count else 0 for record in row_records],
            y=[len(row_records) - idx - 1 for idx in range(len(row_records))],
            orientation="h",
            marker_color=["rgba(0,0,0,0)" if record["is_header"] else NEGATIVE_COLOR for record in row_records],
            customdata=[
                ["header", record["feature"]] if record["is_header"] else ["item", record["key"]]
                for record in row_records
            ],
            hovertemplate="%{customdata[1]}<br>%{text} matching rules<extra></extra>",
            text=[record["count"] for record in row_records],
            textfont={"size": 9, "color": NEUTRAL_TEXT_COLOR},
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    antecedent_x = []
    antecedent_y = []
    antecedent_text = []
    antecedent_customdata = []
    consequent_points = {
        "target=1": {"x": [], "y": [], "text": [], "customdata": [], "name": "Heart disease", "color": DISEASE_COLOR_MAP["Heart disease"]},
        "target=0": {"x": [], "y": [], "text": [], "customdata": [], "name": "No heart disease", "color": DISEASE_COLOR_MAP["No heart disease"]},
    }

    for col_index, row in enumerate(plot_rules.itertuples(), start=1):
        row_text = rule_to_text(pd.Series(row._asdict()))
        hover_text = (
            f"Rule {col_index}<br>If {row_text['if_text']}<br>"
            f"Then {row_text['then_text']}<br>"
            f"Lift: {row.lift:.2f}<br>Confidence: {row.confidence:.2f}"
        )
        for item in row.antecedents:
            key = _feature_from_item(item) if collapse_rows else item
            antecedent_x.append(col_index)
            antecedent_y.append(y_positions[key])
            antecedent_text.append(hover_text)
            antecedent_customdata.append(["rule", col_index - 1])
        for item in row.consequents:
            key = item
            point_group = consequent_points.setdefault(
                item,
                {"x": [], "y": [], "text": [], "customdata": [], "name": humanize_item(item).title(), "color": NEGATIVE_COLOR},
            )
            point_group["x"].append(col_index)
            point_group["y"].append(y_positions[key])
            point_group["text"].append(hover_text)
            point_group["customdata"].append(["rule", col_index - 1])

    for col_index in range(1, len(plot_rules) + 1):
        fig.add_vline(x=col_index, line_color="rgba(80,80,80,0.35)", line_width=1, row=1, col=2)
        fig.add_annotation(
            x=col_index,
            y=len(row_records) - 0.15,
            text=rule_metric_text[col_index - 1],
            showarrow=False,
            font={"size": 10, "color": NEUTRAL_TEXT_COLOR},
            xref="x2",
            yref="y2",
        )

    fig.add_trace(
        go.Scatter(
            x=antecedent_x,
            y=antecedent_y,
            mode="markers",
            marker={"symbol": "square", "size": 12, "color": PATIENT_COLOR},
            hovertemplate="%{text}<extra></extra>",
            text=antecedent_text,
            customdata=antecedent_customdata,
            name="Conditions",
        ),
        row=1,
        col=2,
    )
    for item_key in ["target=1", "target=0"]:
        point_group = consequent_points.get(item_key)
        if not point_group or not point_group["x"]:
            continue
        fig.add_trace(
            go.Scatter(
                x=point_group["x"],
                y=point_group["y"],
                mode="markers",
                marker={"symbol": "circle", "size": 12, "color": point_group["color"]},
                hovertemplate="%{text}<extra></extra>",
                text=point_group["text"],
                customdata=point_group["customdata"],
                name=point_group["name"],
            ),
            row=1,
            col=2,
        )

    y_tickvals = list(range(len(row_records)))
    y_ticktext = [
        "" if record["is_header"] else record["label"]
        for record in reversed(row_records)
    ]
    fig.update_xaxes(
        title_text="Frequency",
        showgrid=False,
        zeroline=False,
        autorange="reversed",
        tickvals=[],
        range=[-1.05, 0],
        row=1,
        col=1,
    )
    fig.update_xaxes(
        title_text=f"Rules sorted by {col_sort}",
        tickmode="array",
        tickvals=list(range(1, len(plot_rules) + 1)),
        ticktext=[f"R{i}" for i in range(1, len(plot_rules) + 1)],
        side="top",
        showgrid=False,
        zeroline=False,
        row=1,
        col=2,
    )
    fig.update_yaxes(
        tickmode="array",
        tickvals=y_tickvals,
        ticktext=y_ticktext,
        showgrid=False,
        zeroline=False,
        row=1,
        col=1,
    )
    fig.update_yaxes(
        tickmode="array",
        tickvals=y_tickvals,
        ticktext=y_ticktext,
        showgrid=False,
        zeroline=False,
        row=1,
        col=2,
    )
    fig.update_layout(
        title="Association Rule Matrix",
        template="plotly_white",
        height=min(RULE_FIG_MAX_HEIGHT, max(360, 28 * len(row_records) + 120)),
        margin={"l": 150, "r": 20, "t": 100, "b": 30},
        legend={"orientation": "h", "yanchor": "top", "y": -0.08, "x": 0},
        clickmode="event+select",
        dragmode="select",
        hovermode="closest",
    )
    for record in row_records:
        if not record["is_header"]:
            continue
        fig.add_annotation(
            x=-0.015,
            y=y_positions.get(record["key"], None) if record["key"] in y_positions else len(row_records) - row_records.index(record) - 1,
            xref="paper",
            yref="y",
            text=record["feature"],
            showarrow=False,
            xanchor="right",
            align="left",
            font={"size": 11, "color": "#1f1f1f"},
            bgcolor="rgba(255,255,255,0.95)",
            bordercolor="#8c8c8c",
            borderwidth=1,
            borderpad=3,
        )
    return fig


# ===== RULE FILTERING & PATIENT MATCHING: Advanced Queries =====
# Functions for filtering rules by antecedents and finding matching patients
def get_rule_filter_options(rules_df: pd.DataFrame | None = None) -> list[dict]:
    active_rules = PLOT_RULES_DF if rules_df is None else rules_df
    items = set()
    for antecedents in active_rules["antecedents"]:
        items.update(antecedents)
    for consequents in active_rules["consequents"]:
        items.update(consequents)
    items = sorted(items, key=humanize_item)
    return [{"label": humanize_item(item), "value": item} for item in items]


def filter_rules_by_antecedents(rules_df: pd.DataFrame | None = None, selected_items: list[str] | None = None) -> pd.DataFrame:
    active_rules = PLOT_RULES_DF if rules_df is None else rules_df.copy()
    if not selected_items:
        return active_rules

    selected_set = set(selected_items)
    selected_outcomes = {item for item in selected_set if str(item).startswith("target=")}
    selected_antecedents = selected_set - selected_outcomes

    filtered = active_rules
    if selected_outcomes:
        filtered = filtered[
            filtered["consequents"].apply(lambda items: any(outcome in set(items) for outcome in selected_outcomes))
        ]

    if selected_antecedents:
        filtered = filtered[
            filtered["antecedents"].apply(lambda items: selected_antecedents.issubset(set(items)))
        ]

    return filtered.reset_index(drop=True)


def _patient_ids_for_rule_row(rule_row: pd.Series) -> set[int]:
    mask = pd.Series(True, index=DISCRETIZED_RULE_SOURCE_DF.index)
    rule_items = list(rule_row["antecedents"]) + list(rule_row["consequents"])
    for item in rule_items:
        if "=" not in str(item):
            continue
        feature, value = str(item).split("=", 1)
        if feature not in DISCRETIZED_RULE_SOURCE_DF.columns:
            continue
        mask &= DISCRETIZED_RULE_SOURCE_DF[feature].astype(str).eq(value)
    return set(DISCRETIZED_RULE_SOURCE_DF.loc[mask, "patient_id"].astype(int).tolist())


def get_selected_rule_matching_patient_ids(
    selected_indices: list[int] | None,
    selected_items: list[str] | None = None,
    col_sort: str = "lift",
    top_n: int = 25,
) -> list[int]:
    if not selected_indices:
        return []

    filtered_rules = filter_rules_by_antecedents(build_plot_rules(RULES_DF), selected_items)
    matrix_rules = get_matrix_rules(filtered_rules, top_n=top_n, col_sort=col_sort)

    matched_patient_ids: set[int] = set()
    for rule_index in selected_indices:
        if rule_index < 0 or rule_index >= len(matrix_rules):
            continue
        matched_patient_ids.update(_patient_ids_for_rule_row(matrix_rules.iloc[int(rule_index)]))

    return sorted(matched_patient_ids)


# ===== RULE DISTRIBUTION: Bubble Chart Visualization =====
# Functions for creating and formatting the rule distribution scatter plot
def make_top_rules_bar_figure(ranked_rules: pd.DataFrame, top_n: int = 8) -> go.Figure:
    fig = go.Figure()

    if ranked_rules.empty:
        fig.add_annotation(
            text="No matching rules found for this patient.",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
            font={"size": 15},
        )
        fig.update_layout(
            title="Top Rules by Lift",
            template="plotly_white",
            height=360,
            xaxis={"visible": False},
            yaxis={"visible": False},
            margin={"l": 20, "r": 20, "t": 50, "b": 20},
        )
        return fig

    top_rules = ranked_rules.sort_values(by=["lift", "confidence"], ascending=False).head(top_n).copy()
    top_rules = top_rules.iloc[::-1]
    labels = []
    hover_text = []
    for row in top_rules.itertuples():
        rule = rule_to_text(pd.Series(row._asdict()))
        labels.append(f"Rule {row.Index + 1}")
        hover_text.append(
            f"If {rule['if_text']}<br>Then {rule['then_text']}<br>"
            f"Lift: {row.lift:.2f}<br>Confidence: {row.confidence:.2f}<br>Support: {row.support:.2f}"
        )

    fig.add_trace(
        go.Bar(
            x=top_rules["lift"],
            y=labels,
            orientation="h",
            marker_color=NEGATIVE_COLOR,
            customdata=hover_text,
            hovertemplate="%{customdata}<extra></extra>",
        )
    )
    fig.update_layout(
        title="Top Rules by Lift",
        template="plotly_white",
        height=min(RULE_FIG_MAX_HEIGHT, max(RULE_FIG_MIN_HEIGHT, 44 * len(top_rules) + 80)),
        margin={"l": 90, "r": 20, "t": 50, "b": 30},
        xaxis_title="Lift",
        yaxis_title="",
    )
    return fig


def make_rule_distribution_figure(rules_df: pd.DataFrame | None = None) -> go.Figure:
    active_rules = PLOT_RULES_DF if rules_df is None else rules_df
    fig = go.Figure()

    if active_rules.empty:
        return _add_empty_figure_annotation(fig, "Rule Distribution", 560)

    plot_df = active_rules.copy()
    plot_df["rule_name"] = [f"Rule {idx + 1}" for idx in range(len(plot_df))]
    plot_df["outcome"] = plot_df["consequents"].apply(lambda items: humanize_item(items[0]) if items else "Outcome")
    plot_df["jitter_offset"] = [(((idx * 37) % 17) - 8) / 1000 for idx in range(len(plot_df))]
    plot_df["support_jitter"] = plot_df["support"] + plot_df["jitter_offset"]
    plot_df["confidence_jitter"] = plot_df["confidence"] - plot_df["jitter_offset"] * 0.6
    plot_df["hover_text"] = plot_df.apply(
        lambda row: (
            f"{row['rule_name']}<br>"
            f"If {' and '.join(humanize_item(item) for item in row['antecedents'])}<br>"
            f"Then {humanize_item(row['consequents'][0])}<br>"
            f"Support: {row['support']:.2f}<br>"
            f"Confidence: {row['confidence']:.2f}<br>"
            f"Lift: {row['lift']:.2f}"
        ),
        axis=1,
    )

    fig.add_trace(
        go.Scatter(
            x=plot_df["support_jitter"],
            y=plot_df["confidence_jitter"],
            mode="markers",
            marker={
                "size": plot_df["lift"].clip(lower=1.0) * 16,
                "color": plot_df["lift"],
                "colorscale": DIVERGING_COLOR_SCALE,
                "showscale": True,
                "colorbar": {"title": "Lift"},
                "line": {"color": "rgba(255,255,255,0.95)", "width": 1.2},
                "opacity": 0.72,
            },
            text=plot_df["hover_text"],
            customdata=plot_df.index,
            hovertemplate="%{text}<extra></extra>",
            selected={"marker": {"opacity": 1.0, "color": PATIENT_COLOR, "size": 20}},
            unselected={"marker": {"opacity": 0.22}},
        )
    )
    fig.update_layout(
        title="Rule Distribution",
        template="plotly_white",
        height=560,
        margin={"l": 70, "r": 20, "t": 50, "b": 50},
        xaxis_title="Support",
        yaxis_title="Confidence",
        clickmode="event+select",
        dragmode="lasso",
        hovermode="closest",
        uirevision="rule-distribution",
    )
    x_padding = 0.01
    y_padding = 0.01
    fig.update_xaxes(
        tickformat=".2f",
        title_font={"size": 15},
        tickfont={"size": 12},
        range=[
            max(0, float(plot_df["support_jitter"].min()) - x_padding),
            min(1, float(plot_df["support_jitter"].max()) + x_padding),
        ],
    )
    fig.update_yaxes(
        tickformat=".2f",
        title_font={"size": 15},
        tickfont={"size": 12},
        range=[
            max(0, float(plot_df["confidence_jitter"].min()) - y_padding),
            min(1, float(plot_df["confidence_jitter"].max()) + y_padding),
        ],
    )
    return fig


# ===== MODULE EXPORTS =====
# API for rule mining functionality
__all__ = [
    "RULES_DF",
    "PLOT_RULES_DF",
    "build_rule_dataset",
    "build_rules",
    "build_rule_cards",
    "build_rule_detail_card",
    "build_plot_rules",
    "build_top_rules_table_data",
    "build_selected_rule_details",
    "filter_rules_by_antecedents",
    "get_selected_rule_matching_patient_ids",
    "get_matrix_rules",
    "get_rule_filter_options",
    "make_rule_distribution_figure",
    "make_rule_matrix_figure",
    "make_top_rules_bar_figure",
    "discretize_dataframe",
    "discretize_patient",
    "get_ranked_rules_for_patient",
    "matching_rules",
    "patient_to_items",
    "rank_rules",
    "rule_to_text",
]
