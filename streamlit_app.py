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
st.title("Xiaomi Phones Dashboard")

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
df["weekday"] = pd.Categorical(df["event_time"].dt.day_name(), categories=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"], ordered=True)

views = df[df["event_type"] == "view"]
purchases = df[df["event_type"] == "purchase"]
total_views = len(views)
total_purchases = len(purchases)
conversion_rate = round((total_purchases / total_views * 100), 1) if total_views else 0
avg_price = 207.86

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
        title="Daily Event Volume",
        color_discrete_map={"Viewed": XIAOMI_ORANGE, "Purchased": "#002f5f"},
        text_auto=True
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    funnel_data = df["event_type"].value_counts().reindex(["view", "cart", "purchase"]).fillna(0).astype(int)
    funnel_df = pd.DataFrame({"Stage": ["Viewed", "Added to Cart", "Purchased"], "Count": funnel_data.values})
    funnel_df = funnel_df.sort_values(by="Count", ascending=True)

    fig_funnel = px.funnel(
        funnel_df,
        y="Stage",
        x="Count",
        color="Stage",
        color_discrete_map={"Viewed": XIAOMI_ORANGE, "Added to Cart": "gray", "Purchased": "#002f5f"},
        title="Funnel: Views to Cart to Purchase"
    )
    fig_funnel.update_traces(textinfo="value", textposition="outside")
    st.plotly_chart(fig_funnel, use_container_width=True)

    st.markdown(f"""
        <div style="background-color:#e6f4ff;padding:15px;border-radius:10px;">
         🧠 <b>Insight:</b> Conversion rate is {conversion_rate:.1f}%. There’s an opportunity to boost this through retargeting or urgency-based messaging.
        </div>
    """, unsafe_allow_html=True)

# --- TAB 2 ---
with tab2:
    heatmap1 = views.groupby(["weekday", "hour"]).size().unstack(fill_value=0)
    heatmap2 = purchases.groupby(["weekday", "hour"]).size().unstack(fill_value=0)
    fig_view = px.imshow(heatmap1, title="Products Viewed", color_continuous_scale="Blues",
                         labels=dict(x="Hour", y="Weekday", color="Views"))
    fig_purchase = px.imshow(heatmap2, title="Products Purchased", color_continuous_scale="Reds",
                             labels=dict(x="Hour", y="Weekday", color="Purchases"))
    st.plotly_chart(fig_view, use_container_width=True)
    st.plotly_chart(fig_purchase, use_container_width=True)

    hourly_df = pd.DataFrame({
        "Hour": range(24),
        "Views": views.groupby("hour").size().reindex(range(24), fill_value=0).values,
        "Purchases": purchases.groupby("hour").size().reindex(range(24), fill_value=0).values
    })
    fig_line = px.line(hourly_df, x="Hour", y=["Views", "Purchases"],
                       title="Hourly Xiaomi Activity",
                       labels={"value": "Event count", "variable": "Event type"})
    st.plotly_chart(fig_line, use_container_width=True)

    st.markdown("""
        <div style="background-color:#e6f4ff;padding:15px;border-radius:10px;">
        🧠 <b>Insight:</b> Most engagement happens between 18:00–22:00. These hours are optimal for targeted campaigns.
        </div>
    """, unsafe_allow_html=True)

# --- TAB 3 ---
with tab3:
    fig_price = px.histogram(purchases, x="price", nbins=30, color_discrete_sequence=[XIAOMI_ORANGE],
                             title="Price Distribution of Purchases", labels={"price": "Price (USD)", "Count": "Frequency"}, text_auto=True)
    st.plotly_chart(fig_price, use_container_width=True)

    bins = [0, 200, 400, 600, 800, 1000, np.inf]
    labels = ["<$200", "$200–400", "$400–600", "$600–800", "$800–1000", "$1000+"]
    df["price_bin"] = pd.cut(df["price"], bins=bins, labels=labels, include_lowest=True)
    conv_data = df[df["event_type"].isin(["view", "purchase"])]
    grouped = conv_data.groupby(["price_bin", "event_type"]).size().unstack(fill_value=0).reset_index()
    grouped["view"] = grouped["view"].round(0).astype(int)

    fig_price_buckets = px.bar(
        grouped,
        x="price_bin",
        y="view",
        text="view",
        color_discrete_sequence=[XIAOMI_ORANGE],
        labels={"view": "Views", "price_bin": "Price Range"},
        title="Views by Price Range"
    )
    st.plotly_chart(fig_price_buckets, use_container_width=True)

    fig_box = px.box(purchases, y="price", color_discrete_sequence=[XIAOMI_ORANGE],
                     title="Price Range", points=False)
    st.plotly_chart(fig_box, use_container_width=True)

    desc_stats = purchases['price'].describe()[["min", "25%", "50%", "75%", "max", "mean"]].round(2)
    desc_stats.index = ["Min", "Q1 (25%)", "Median", "Q3 (75%)", "Max", "Mean"]
    st.dataframe(
    desc_stats.reset_index().rename(columns={"index": "Statistic", "price": "USD"}),
    use_container_width=True,
    column_config={"USD": st.column_config.NumberColumn("USD")},
    height=300,
    title="Price Summary Table"
    )

    if "basket" in purchases.columns and purchases["basket"].notna().sum() > 0:
    basket_items = purchases["basket"].dropna().str.split(",").explode().str.strip()
    top_basket = basket_items.value_counts().head(10).reset_index()
    top_basket.columns = ["Item", "Frequency"]
    top_basket["Frequency"] = top_basket["Frequency"].astype(int)
    st.dataframe(
        top_basket,
        use_container_width=True,
        height=300,
        title="Top 10 Basket Items"
    )

    st.markdown("""
    <div style="background-color:#e6f4ff;padding:15px;border-radius:10px;">
    🧠 <b>Insight:</b> Promoting with common co-purchased items may improve total order value.
    </div>
    """, unsafe_allow_html=True)


# --- TAB 4 ---
with tab4:
    st.subheader("Purchase Probability Simulator")

    col1, col2 = st.columns(2)
    with col1:
        price_input = st.slider("Select product price (USD):", 0, 1000, 500, step=10)
    with col2:
        hour_input = st.slider("Select hour of day (0–23):", 0, 23, 14)

    input_df = pd.DataFrame({"price": [price_input], "hour": [hour_input]})
    prob = model.predict_proba(input_df)[0][1] * 100

    if prob >= 75:
        st.success("🔥 High likelihood of purchase")
    elif prob >= 50:
        st.info("👍 Moderate likelihood of purchase")
    elif prob >= 25:
        st.warning("⚠️ Low likelihood of purchase")
    else:
        st.error("❌ Very low likelihood of purchase")

    st.metric(label="Estimated purchase probability", value=f"{prob:.0f}%")

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
                {'range': [75, 100], 'color': "#ff6900"}
            ]
        }
    ))
    st.plotly_chart(gauge_fig, use_container_width=True)

    st.markdown("""
        ### Model Performance Summary
        - **Model**: XGBoost (balanced)
        - **Recall (class 1)**: 0.54
        - **Precision (class 1)**: 0.11
        - **ROC AUC**: 0.53
        - **PR AUC**: 0.11

        <div style='background-color:#e6f4ff;padding:15px;border-radius:10px;'>
        🧠 <b>Insight:</b> This model is designed to rank sessions by likelihood of purchase, prioritizing recall to avoid missing high-intent buyers. This tool should help to target customers based on the price of the item and time of day.
        </div>
    """, unsafe_allow_html=True)

