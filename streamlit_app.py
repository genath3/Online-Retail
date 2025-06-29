# Xiaomi Dashboard with Modeling and Advanced Analysis

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.cluster import KMeans

# --- CONFIG ---
import xgboost
st.set_page_config(page_title="Xiaomi Dashboard", layout="wide")
st.title("📱 Xiaomi Phones Dashboard")

# Xiaomi styling
XIAOMI_ORANGE = "#ff6900"
XIAOMI_BLACK = "#000000"
XIAOMI_WHITE = "#ffffff"


st.markdown("This dashboard provides insights into Xiaomi phone interactions, sales, and behavioral patterns for October 2019.")

# --- LOAD DATA ---
import requests
from io import StringIO

@st.cache_data
def load_data():
    url = "https://huggingface.co/datasets/genath3/Xiaomi/resolve/main/xiaomi_cleaned.csv"
    headers = {"Authorization": f"Bearer {st.secrets['huggingface']['token']}"}

    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        st.error(f"Failed to fetch file. Status code: {response.status_code}")
        st.stop()

    df = pd.read_csv(StringIO(response.text))
    df["event_time"] = pd.to_datetime(df["event_time"], errors="coerce")
    df["brand"] = df["brand"].astype(str).str.lower()
    df = df[df["brand"] == "xiaomi"]
    return df

df = load_data()
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

# --- TAB 1: MARKET OVERVIEW ---
with tab1:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("\U0001F440 Total Views", f"{total_views:,}")
    col2.metric("\U0001F6D2 Purchases", f"{total_purchases:,}")
    col3.metric("\U0001F3AF Conversion Rate", f"{conversion_rate:.1f}%")
    col4.metric("\U0001F4B8 Avg. Price", f"${avg_price:.2f}")

    daily_counts = df.groupby(["date", "event_type"]).size().reset_index(name="count")
    daily_counts = daily_counts[daily_counts["event_type"].isin(["view", "purchase"])]
    daily_counts["event_type"] = daily_counts["event_type"].replace({"view": "Viewed", "purchase": "Purchased"})
    daily_counts["count"] = daily_counts["count"].round(0).astype(int)

    
    chart_type = st.radio("Select chart type:", ["Absolute", "Percentage"], horizontal=True)

    if chart_type == "Percentage":
        pivot = daily_counts.pivot(index="date", columns="event_type", values="count").fillna(0)
        pivot_pct = pivot.div(pivot.sum(axis=1), axis=0) * 100
        daily_counts = pivot_pct.reset_index().melt(id_vars="date", var_name="event_type", value_name="count")
        yaxis_label = "Percentage of Events"
    else:
        yaxis_label = "Event Count"

    fig_bar = px.bar(
        daily_counts,
        x="date",
        y="count",
        color="event_type",
        color_discrete_map={"Viewed": "#ff6900", "Purchased": "#002f5f"},
        text="count",
        barmode="stack",
        labels={"date": "Date", "count": yaxis_label, "event_type": "Event Type"},
        category_orders={"event_type": ["View", "Purchase"]},
        text_auto=True,
        title="Daily Event Volume"
    )
    fig_bar.update_traces(texttemplate="%{text}")
    fig_bar.for_each_trace(lambda t: t.update(textposition="outside", textfont=dict(size=18)) if t.name == "purchase" else t.update(textposition="outside", textfont=dict(size=16)))
    st.plotly_chart(fig_bar, use_container_width=True)

    funnel_counts = df["event_type"].value_counts()
    funnel_counts = df["event_type"].value_counts()
    funnel_data = pd.DataFrame({
        "Stage": ["Viewed", "Added to Cart", "Purchased"],
        "Count": [
            funnel_counts.get("view", 0),
            funnel_counts.get("cart", 0),
            funnel_counts.get("purchase", 0)
        ]
    })
    funnel_data["Stage"] = pd.Categorical(funnel_data["Stage"], categories=["Viewed", "Added to Cart", "Purchased"], ordered=True)
    funnel_data["Stage"] = pd.Categorical(funnel_data["Stage"], categories=["Viewed", "Added to Cart", "Purchased"], ordered=True)
    funnel_data["Stage"] = pd.Categorical(funnel_data["Stage"], categories=["Viewed", "Added to Cart", "Purchased"], ordered=True)
    fig_funnel = px.funnel(
        funnel_data.sort_values(by="Stage", ascending=False),
        y="Stage",
        x="Count",
        color="Stage",
        color_discrete_map={"Viewed": XIAOMI_ORANGE, "Added to Cart": "#888888", "Purchased": "#002f5f"},
        title="🔁 Funnel: Views to Cart to Purchase"
    )
    fig_funnel.update_traces(texttemplate="%{y}: %{x}")
    st.plotly_chart(fig_funnel, use_container_width=True)

    st.markdown("""
        <div style="background-color:#e6f4ff;padding:15px;border-radius:10px;">
        \U0001F9E0 <b>Insight:</b> Conversion rate stands at {:.1f}%. Focused promotions or retargeting strategies may boost this further.
        </div>
    """.format(conversion_rate), unsafe_allow_html=True)

# --- TAB 2: TIME ANALYSIS ---
with tab2:
    heatmap1 = views.groupby(["weekday", "hour"]).size().unstack(fill_value=0)
    fig_view = px.imshow(heatmap1, aspect="auto", text_auto=True, color_continuous_scale="Blues",
                         title="Products Viewed")
    st.plotly_chart(fig_view, use_container_width=True)

    heatmap2 = purchases.groupby(["weekday", "hour"]).size().unstack(fill_value=0)
    fig_purchase = px.imshow(heatmap2, aspect="auto", text_auto=True, color_continuous_scale="Reds",
                             title="Products Purchased")
    st.plotly_chart(fig_purchase, use_container_width=True)

    hourly_df = pd.DataFrame({
        "Hour": range(24),
        "Views": views.groupby("hour").size().reindex(range(24), fill_value=0).values,
        "Purchases": purchases.groupby("hour").size().reindex(range(24), fill_value=0).values
    })
    fig_line = px.line(hourly_df, x="Hour", y=["Views", "Purchases"],
                       title="\U0001F550 Hourly Xiaomi Activity",
                       labels={"value": "Event Count", "variable": "Event Type"})
    st.plotly_chart(fig_line, use_container_width=True)

    st.markdown("""
        <div style="background-color:#e6f4ff;padding:15px;border-radius:10px;">
        \U0001F9E0 <b>Insight:</b> Engagement peaks in the evening. Time your marketing campaigns between 6–10 PM.
        </div>
    """, unsafe_allow_html=True)

# --- TAB 3: BASKET & PRICING ---
with tab3:
    st.markdown("### 💰 Price Distribution of Purchases")
    fig_price = px.histogram(purchases, x="price", nbins=30,
                             color_discrete_sequence=[XIAOMI_ORANGE],
                             text_auto=True,
                             labels={"price": "Price (USD)", "count": "Frequency"})
    st.plotly_chart(fig_price, use_container_width=True)

    st.markdown("### 📦 Price Range")
    box_fig = px.box(purchases, y="price", color_discrete_sequence=[XIAOMI_ORANGE], points=False)
    st.plotly_chart(box_fig, use_container_width=True)

    desc_stats = purchases['price'].describe()[["min", "25%", "50%", "75%", "max", "mean"]].round(2)
    desc_stats.index = ["Min", "Q1 (25%)", "Median", "Q3 (75%)", "Max", "Mean"]
    st.markdown("### 📊 Price Summary Table")
    st.dataframe(desc_stats.reset_index().rename(columns={"index": "Statistic", "price": "USD"}))

    st.markdown("### 🧮 Views by Price Range")
    bins = [0, 200, 400, 600, 800, 1000, np.inf]
    labels = ["<$200", "$200–400", "$400–600", "$600–800", "$800–1000", "$1000+"]
    df["price_bin"] = pd.cut(df["price"], bins=bins, labels=labels, include_lowest=True)
    conv_data = df[df["event_type"].isin(["view", "purchase"])]
    grouped = conv_data.groupby(["price_bin", "event_type"]).size().unstack(fill_value=0).reset_index()
    grouped["view"] = grouped["view"].round(0).astype(int)
    grouped["view"] = grouped["view"].astype(int)
    fig = px.bar(grouped, x="price_bin", y="view", text="view", color_discrete_sequence=[XIAOMI_ORANGE],
                 labels={"view": "Views", "price_bin": "Price Range"},)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 🛍️ Top Basket Items")
    if "basket" in purchases.columns and purchases["basket"].notna().sum() > 0:
        basket_items = purchases["basket"].dropna().str.split(",").explode().str.strip()
        top_basket = basket_items.value_counts().head(10).reset_index()
        top_basket.columns = ["Item", "Frequency"]
        top_basket["Frequency"] = top_basket["Frequency"].astype(int)
        st.dataframe(top_basket)

    st.markdown("""
        <div style="background-color:#e6f4ff;padding:15px;border-radius:10px;">
        \U0001F9E0 <b>Insight:</b> Popular co-purchases can be bundled to increase basket size.
        </div>
    """, unsafe_allow_html=True)

# --- TAB 4: PREDICTIVE INSIGHTS ---
@st.cache_resource
def get_model():
    from joblib import load
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        repo_id="genath3/xiaomi-purchase-model",
        filename="xiaomi_model_xgboost.joblib",
        token=st.secrets["huggingface"]["token"]
    )
    model = load(path)
    return model

with tab4:
    st.subheader("🎯 Purchase Probability Simulator")
    model = get_model()

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

    import plotly.graph_objects as go

    st.metric(label="Estimated Purchase Probability", value=f"{prob:.0f}%")

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
        ### 📊 Model Performance Summary
        - **Model**: XGBoost (balanced)
        - **Recall (class 1)**: 0.54
        - **Precision (class 1)**: 0.11
        - **ROC AUC**: 0.53
        - **PR AUC**: 0.11
        
        This model is designed to rank sessions by likelihood of purchase, prioritizing recall over precision to avoid missing high-intent buyers.
    """)










