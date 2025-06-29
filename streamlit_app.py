# Xiaomi Dashboard with Modeling and Advanced Analysis

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.cluster import KMeans
import xgboost
import os
import requests
from io import StringIO
import plotly.graph_objects as go

# --- CONFIG ---
st.set_page_config(page_title="Xiaomi Dashboard", layout="wide")
st.title("📱 Xiaomi Phones Dashboard")

# Xiaomi styling
XIAOMI_ORANGE = "#ff6900"
XIAOMI_BLACK = "#000000"
XIAOMI_WHITE = "#ffffff"

st.markdown("This dashboard provides insights into Xiaomi phone interactions, sales, and behavioral patterns for October 2019.")

# --- LOAD DATA ---
@st.cache_data
def load_data():
    url = "https://huggingface.co/datasets/genath3/Xiaomi/resolve/main/xiaomi_cleaned.csv"
    token = os.getenv("HF_TOKEN", "")
    headers = {"Authorization": f"Bearer {token}"}

    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        st.error(f"Failed to fetch file. Status code: {response.status_code}")
        st.stop()

    df = pd.read_csv(StringIO(response.text))
    df["event_time"] = pd.to_datetime(df["event_time"], errors="coerce")
    df["brand"] = df["brand"].astype(str).str.lower()
    df = df[df["brand"] == "xiaomi"]
    return df

# --- LOAD MODEL ---
@st.cache_resource
def get_model():
    from joblib import load
    from huggingface_hub import hf_hub_download

    token = os.getenv("HF_TOKEN", "")
    path = hf_hub_download(
        repo_id="genath3/xiaomi-purchase-model",
        filename="xiaomi_model_xgboost.joblib",
        token=token
    )
    model = load(path)
    return model

# --- MAIN EXECUTION ---
df = load_data()
model = get_model()

# Feature engineering
df = df.dropna(subset=["event_time", "event_type", "price"])
df["date"] = df["event_time"].dt.date
df["hour"] = df["event_time"].dt.hour
df["weekday"] = df["event_time"].dt.day_name()

views = df[df["event_type"] == "view"]
purchases = df[df["event_type"] == "purchase"]
total_views = len(views)
total_purchases = len(purchases)
conversion_rate = round((total_purchases / total_views * 100), 1) if total_views else 0
avg_price = int(purchases["price"].mean())

# --- TABS ---
tab1, tab2, tab3, tab4 = st.tabs([
    "1️⃣ Market Overview", "2️⃣ Time Analysis", "3️⃣ Basket & Pricing", "4️⃣ Predictive Insights"
])

# --- TAB 1 ---
with tab1:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("👁️ Total Views", f"{total_views:,}")
    col2.metric("🛒 Purchases", f"{total_purchases:,}")
    col3.metric("🎯 Conversion Rate", f"{conversion_rate:.1f}%")
    col4.metric("💲 Avg. Price", f"${avg_price:.2f}")

    daily_counts = df.groupby(["date", "event_type"]).size().reset_index(name="count")
    daily_counts = daily_counts[daily_counts["event_type"].isin(["view", "purchase"])]
    daily_counts["event_type"] = daily_counts["event_type"].replace({"view": "Viewed", "purchase": "Purchased"})
    daily_counts["count"] = daily_counts["count"].round(0).astype(int)

    fig_bar = px.bar(
        daily_counts,
        x="date",
        y="count",
        color="event_type",
        barmode="stack",
        title="📊 Daily Event Volume",
        color_discrete_map={"Viewed": XIAOMI_ORANGE, "Purchased": "#002f5f"},
        text_auto=True
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    funnel_data = df["event_type"].value_counts().reindex(["view", "cart", "purchase"]).fillna(0).astype(int)
    funnel_df = pd.DataFrame({"Stage": ["Viewed", "Added to Cart", "Purchased"], "Count": funnel_data.values})
    fig_funnel = px.funnel(funnel_df, y="Stage", x="Count", color="Stage",
                           color_discrete_map={"Viewed": XIAOMI_ORANGE, "Added to Cart": "gray", "Purchased": "#002f5f"},
                           title="🔁 Funnel: Views to Cart to Purchase")
    st.plotly_chart(fig_funnel, use_container_width=True)

# --- TAB 2 ---
with tab2:
    heatmap1 = views.groupby(["weekday", "hour"]).size().unstack(fill_value=0)
    heatmap2 = purchases.groupby(["weekday", "hour"]).size().unstack(fill_value=0)
    st.plotly_chart(px.imshow(heatmap1, title="Views Heatmap", color_continuous_scale="Blues"), use_container_width=True)
    st.plotly_chart(px.imshow(heatmap2, title="Purchases Heatmap", color_continuous_scale="Reds"), use_container_width=True)

# --- TAB 3 ---
with tab3:
    st.plotly_chart(px.histogram(purchases, x="price", nbins=30, color_discrete_sequence=[XIAOMI_ORANGE],
                                 title="💰 Price Distribution of Purchases"), use_container_width=True)
    st.plotly_chart(px.box(purchases, y="price", color_discrete_sequence=[XIAOMI_ORANGE],
                           title="📦 Price Range", points=False), use_container_width=True)
    stats = purchases['price'].describe().round(2).rename({"25%": "Q1", "50%": "Median", "75%": "Q3"})
    st.dataframe(stats)

# --- TAB 4 ---
with tab4:
    st.subheader("🎯 Purchase Probability Simulator")
    price_input = st.slider("Price", 0, 1000, 500)
    hour_input = st.slider("Hour", 0, 23, 12)
    try:
        prob = model.predict_proba(pd.DataFrame({"price": [price_input], "hour": [hour_input]}))[0][1] * 100
        st.metric("Predicted Purchase Probability", f"{prob:.1f}%")
        gauge_fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob,
            number={'suffix': "%"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': '#002f5f'},
                'steps': [
                    {'range': [0, 25], 'color': "#f9e4dc"},
                    {'range': [25, 50], 'color': "#fcd6bf"},
                    {'range': [50, 75], 'color': "#ffab7b"},
                    {'range': [75, 100], 'color': XIAOMI_ORANGE}
                ]
            }
        ))
        st.plotly_chart(gauge_fig, use_container_width=True)
    except Exception as e:
        st.error(f"Model prediction failed: {e}")
