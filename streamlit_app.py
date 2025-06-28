import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# --- CONFIG ---
st.set_page_config(page_title="Xiaomi Phone Dashboard", layout="wide")

# --- HEADER ---
st.title("Xiaomi Phones Dashboard")
st.markdown("This dashboard provides insights into Xiaomi phone user interactions and sales for the month of October 2019.")

# --- LOAD DATA ---
@st.cache_data
def load_data():
    url = "https://huggingface.co/datasets/7ng10dpE/Online-Retail/resolve/main/xiaomi_cleaned.csv"
    df = pd.read_csv(url)
    df["event_time"] = pd.to_datetime(df["event_time"], errors="coerce")
    df["brand"] = df["brand"].astype(str).str.lower()
    df = df[df["brand"] == "xiaomi"]
    return df

df = load_data()

# --- BASIC CLEANING ---
df = df.dropna(subset=["event_time", "event_type", "price"])  # price may be NaN for views
df["date"] = df["event_time"].dt.date
df["hour"] = df["event_time"].dt.hour
df["weekday"] = df["event_time"].dt.day_name()

# --- KPI METRICS ---
views = df[df["event_type"] == "view"]
purchases = df[df["event_type"] == "purchase"]
total_views = len(views)
total_purchases = len(purchases)
conversion_rate = (total_purchases / total_views * 100) if total_views else 0
avg_price = purchases["price"].mean()

col1, col2, col3, col4 = st.columns(4)
col1.metric("👀 Total Views", f"{total_views:,}")
col2.metric("🛒 Total Purchases", f"{total_purchases:,}")
col3.metric("🎯 Conversion Rate", f"{conversion_rate:.1f}%")
col4.metric("💸 Avg. Purchase Price", f"${avg_price:.2f}")

# --- TABS ---
tab1, tab2, tab3, tab4 = st.tabs(["📊 Daily Activity", "⏰ Time of Day", "🧺 Basket & Price", "📈 Funnel Analysis"])

# --- TAB 1: DAILY TRENDS ---
with tab1:
    daily_counts = df.groupby(["date", "event_type"]).size().reset_index(name="count")
    fig = px.line(daily_counts, x="date", y="count", color="event_type", markers=True,
                  title="📆 Daily Xiaomi Events")
    st.plotly_chart(fig, use_container_width=True)

# --- TAB 2: HOURLY HEATMAP ---
with tab2:
    heatmap_data = df.groupby(["weekday", "hour", "event_type"]).size().reset_index(name="count")
    weekdays_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    heatmap_data["weekday"] = pd.Categorical(heatmap_data["weekday"], categories=weekdays_order, ordered=True)

    for etype in heatmap_data["event_type"].unique():
        st.subheader(f"🔹 Hourly Heatmap - {etype.title()}")
        subset = heatmap_data[heatmap_data["event_type"] == etype]
        heatmap = subset.pivot_table(index="weekday", columns="hour", values="count", fill_value=0)
        fig = px.imshow(heatmap, text_auto=True, aspect="auto", color_continuous_scale="Blues",
                        labels={"color": "Count"}, title=f"Heatmap of {etype.title()} Events by Hour & Weekday")
        st.plotly_chart(fig, use_container_width=True)

# --- TAB 3: PRICE & BASKET ---
with tab3:
    st.subheader("💰 Price Distribution of Purchases")
    price_fig = px.histogram(purchases, x="price", nbins=30, title="Price Distribution (Purchased Xiaomi Phones)")
    st.plotly_chart(price_fig, use_container_width=True)

    if "basket" in df.columns and df["basket"].notna().sum() > 0:
        st.subheader("📦 Basket Items (Top 10 Most Common)")
        basket_items = df["basket"].dropna().str.split(",").explode().str.strip()
        basket_freq = basket_items.value_counts().head(10).reset_index()
        basket_freq.columns = ["Item", "Count"]
        st.dataframe(basket_freq)

# --- TAB 4: FUNNEL ---
with tab4:
    st.subheader("🧭 Conversion Funnel")
    funnel_counts = {
        "Viewed": total_views,
        "Purchased": total_purchases
    }
    funnel_df = pd.DataFrame(list(funnel_counts.items()), columns=["Stage", "Count"])
    fig = px.funnel(funnel_df, x="Count", y="Stage", title="Xiaomi Purchase Funnel")
    st.plotly_chart(fig, use_container_width=True)

# --- FOOTER ---
st.caption("Data source: Hugging Face · Month: October 2019 · Brand: Xiaomi")

