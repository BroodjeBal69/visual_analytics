from dash import html
import dash_bootstrap_components as dbc

from models.rf import model_rf
from shap_card import compute_prediction_and_shap
from rule_mining import build_plot_rules, build_rule_cards, get_ranked_rules_for_patient, RULES_DF


def _build_shap_explanation(patient_df):
    """Create concise, readable RF SHAP explanation bullets."""
    try:
        _, _, shap_df = compute_prediction_and_shap(model_rf, patient_df)
    except RuntimeError:
        return html.Div("SHAP explanation unavailable.", className="text-muted small mb-2")

    positive = shap_df[shap_df["shap_value"] > 0].sort_values("shap_value", ascending=False).head(2)
    negative = shap_df[shap_df["shap_value"] < 0].sort_values("shap_value", ascending=True).head(2)

    if positive.empty and negative.empty:
        return html.Div("No strong SHAP drivers detected for this patient.", className="text-muted small mb-2")

    lines = []
    if not positive.empty:
        pos_text = "; ".join(
            [f"{row.label} ({row.shap_value:+.3f})" for row in positive.itertuples(index=False)]
        )
        lines.append(html.Li(f"Risk-increasing drivers: {pos_text}"))
    if not negative.empty:
        neg_text = "; ".join(
            [f"{row.label} ({row.shap_value:+.3f})" for row in negative.itertuples(index=False)]
        )
        lines.append(html.Li(f"Risk-reducing drivers: {neg_text}"))

    return html.Div(
        [
            html.Div("RF SHAP Explanation", className="fw-semibold mt-2 mb-1"),
            html.Ul(lines, className="mb-2 ps-3"),
        ]
    )


def build_clinical_summary(patient_df, probability, gam_probability):
    patient_input = patient_df.iloc[0].to_dict()
    ranked_rules = get_ranked_rules_for_patient(patient_input, build_plot_rules(RULES_DF))

    summary = [
        html.H5("Patient Rule Insights", className="mb-3"),
        html.Div(f"RF predicted risk: {probability:.1%}", className="mb-1"),
    ]
    if gam_probability is not None:
        summary.append(html.Div(f"GAM predicted risk: {gam_probability:.1%}", className="mb-2"))
    else:
        summary.append(html.Div("GAM predicted risk unavailable", className="mb-2 text-muted"))

    summary.append(_build_shap_explanation(patient_df))

    if ranked_rules.empty:
        summary.append(html.Div("No matching decision rules found for this patient.", className="text-muted"))
        return summary

    top_rules = ranked_rules.head(3).copy()
    disease_count = int(top_rules["consequents"].apply(lambda items: items == ["target=1"]).sum())
    no_disease_count = int(top_rules["consequents"].apply(lambda items: items == ["target=0"]).sum())

    disease_vote = float(
        top_rules[top_rules["consequents"].apply(lambda items: items == ["target=1"])]
        ["confidence"]
        .sum()
    )
    no_disease_vote = float(
        top_rules[top_rules["consequents"].apply(lambda items: items == ["target=0"])]
        ["confidence"]
        .sum()
    )
    rule_vote = "Heart disease" if disease_vote >= no_disease_vote else "No heart disease"
    rf_vote = "Heart disease" if probability >= 0.5 else "No heart disease"
    disagreement = rule_vote != rf_vote

    summary.extend(
        [
            html.Div(
                [
                    dbc.Badge(
                        f"Rules -> {rule_vote}",
                        color="danger" if rule_vote == "Heart disease" else "primary",
                        className="me-2",
                    ),
                    dbc.Badge(f"Counts: HD {disease_count} | No HD {no_disease_count}", color="light", text_color="dark"),
                ],
                className="mb-2",
            ),
            dbc.Alert(
                "Model and rules disagree on risk direction." if disagreement else "Model and rules agree on risk direction.",
                color="warning" if disagreement else "success",
                className="py-2 mb-3",
            ),
            html.Div("Top 3 Matching Rules", className="fw-semibold mb-2"),
        ]
    )
    summary.extend(build_rule_cards(top_rules, top_n=3))
    return summary
