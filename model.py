import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

df = pd.read_csv(r"heart.csv")

features = ['Age', 'Cholesterol', 'RestingBP', 'Oldpeak', 'Sex', 'ST_Slope']
numeric_features = ['Age', 'Cholesterol', 'RestingBP', 'Oldpeak']
categorical_features = ['Sex', 'ST_Slope']
target = 'HeartDisease'

X = df[features]
y = df[target]

scaler = StandardScaler()
X_scaled = X.copy()
X_scaled[numeric_features] = scaler.fit_transform(X[numeric_features])
X_scaled = pd.get_dummies(X_scaled, columns=categorical_features, drop_first=True)

feature_names = X_scaled.columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

model = LogisticRegression()
model.fit(X_train, y_train)

st.title("Heart Disease Clinical Decision Support Tool")

st.sidebar.header(" Patient Profile")

age = st.sidebar.slider("Age", 20, 90, 50)
chol = st.sidebar.slider("Cholesterol", 100, 400, 200)
bp = st.sidebar.slider("Resting BP", 80, 200, 120)
oldpeak = st.sidebar.slider("Oldpeak", 0.0, 6.0, 1.0)

sex = st.sidebar.selectbox("Sex", ["M", "F"])
st_slope = st.sidebar.selectbox("ST Slope", ["Up", "Flat", "Down"])

threshold = st.sidebar.slider("Decision Threshold", 0.3, 0.7, 0.5)

input_df = pd.DataFrame({
    'Age': [age],
    'Cholesterol': [chol],
    'RestingBP': [bp],
    'Oldpeak': [oldpeak],
    'Sex': [sex],
    'ST_Slope': [st_slope]
})

input_scaled = input_df.copy()
input_scaled[numeric_features] = scaler.transform(input_df[numeric_features])
input_scaled = pd.get_dummies(input_scaled)

input_scaled = input_scaled.reindex(columns=feature_names, fill_value=0)

prob = model.predict_proba(input_scaled)[0][1]
prediction = int(prob >= threshold)

st.markdown("##Patient Risk Assessment")

col1, col2 = st.columns(2)

with col1:
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob * 100,
        title={'text': "Risk Score (%)"},
        gauge={
            'axis': {'range': [0, 100]},
            'steps': [
                {'range': [0, 30], 'color': "green"},
                {'range': [30, 70], 'color': "yellow"},
                {'range': [70, 100], 'color': "red"},
            ]
        }
    ))
    st.plotly_chart(fig_gauge)

with col2:
    st.metric("Predicted Risk", f"{prob*100:.1f}%")

    if prob > 0.7:
        st.error("High risk — immediate attention recommended")
    elif prob > 0.4:
        st.warning("Moderate risk — monitor closely")
    else:
        st.success("Low risk")

st.markdown("##Key Risk Drivers")

contributions = input_scaled.values[0] * model.coef_[0]

contrib_df = pd.DataFrame({
    'Feature': feature_names,
    'Contribution': contributions
}).sort_values(by='Contribution', key=abs, ascending=False)

fig_contrib = px.bar(
    contrib_df.head(10),
    x='Contribution',
    y='Feature',
    orientation='h',
    title="Top factors influencing prediction"
)

st.plotly_chart(fig_contrib)
st.markdown("##Model Performance")

probs_test = model.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, probs_test)
roc_auc = auc(fpr, tpr)

fig_roc = go.Figure()
fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines',
                             name=f"AUC = {roc_auc:.2f}"))
fig_roc.add_shape(type='line', line=dict(dash='dash'),
                  x0=0, x1=1, y0=0, y1=1)

st.plotly_chart(fig_roc)

st.markdown("##Population Insights")

df['AgeGroup'] = pd.cut(df['Age'], bins=[20, 40, 60, 80])
df['AgeGroup'] = df['AgeGroup'].astype(str)

group_risk = df.groupby('AgeGroup')[target].mean().reset_index()

fig_age = px.bar(
    group_risk,
    x='AgeGroup',
    y=target,
    title="Heart Disease Rate by Age Group"
)

st.plotly_chart(fig_age)

corr = df[numeric_features + [target]].corr()

fig_corr = px.imshow(
    corr,
    text_auto=True,
    title="Feature Correlation Matrix"
)

st.plotly_chart(fig_corr)
