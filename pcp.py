import plotly.express as px
import plotly.graph_objects as go
from dash import html
from palette import DIVERGING_COLOR_SCALE


FEATURE_OPTIONS = {
    'age': 'Age',
    'trestbps': 'Resting BP',
    'chol': 'Cholesterol',
    'thalach': 'Max Heart Rate',
    'oldpeak': 'ST Depression',
    'cp': 'Chest Pain Type',
    'exang': 'Exercise Angina',
    'target': 'Heart Disease',
}


def get_pcp_badges(selected_features):
    if not selected_features:
        selected_features = []
    badges = []
    for value, label in FEATURE_OPTIONS.items():
        is_selected = value in selected_features
        badges.append(html.Span(
            label,
            id={'type': 'pcp-dim-badge', 'index': value},
            className=f"badge {'bg-primary' if is_selected else 'bg-secondary'} me-1 mb-1",
            style={'cursor': 'pointer', 'fontSize': '0.9rem', 'padding': '0.38rem 0.6rem'},
        ))
    return badges


def toggle_pcp_feature(clicked_feature, selected_features):
    if selected_features is None:
        selected_features = []
    if clicked_feature is None:
        return selected_features
    if clicked_feature in selected_features:
        return [f for f in selected_features if f != clicked_feature]
    else:
        return selected_features + [clicked_feature]


def render_pcp(data, selected_features, age_range, target_values):
    if not selected_features or len(selected_features) < 3:
        fig = go.Figure()
        fig.update_layout(
            title='Select at least 3 features to create the parallel coordinates plot',
            xaxis={'visible': False},
            yaxis={'visible': False}
        )
        filtered_data = data[(data['age'] >= age_range[0]) & (data['age'] <= age_range[1]) & (data['target'].isin(target_values))]
        return fig, filtered_data.to_dict('records')

    filtered_data = data[
        (data['age'] >= age_range[0]) &
        (data['age'] <= age_range[1]) &
        (data['target'].isin(target_values))
    ]


    fig = px.parallel_coordinates(
        filtered_data,
        dimensions=selected_features,
        color='target',
        labels={
            'age': 'Age',
            'trestbps': 'Resting BP',
            'chol': 'Cholesterol',
            'thalach': 'Max HR',
            'oldpeak': 'ST Depression',
            'cp': 'Chest Pain Type',
            'exang': 'Exercise Angina',
            'target': 'Heart Disease',
        },
        color_continuous_scale=DIVERGING_COLOR_SCALE,
        color_continuous_midpoint=0.5,
    )

    return fig, filtered_data.to_dict('records')
