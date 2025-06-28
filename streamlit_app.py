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

df = load_data()
df = df.dropna(subset=["event_time", "event_type", "price"])
df["date"] = df["event_time"].dt.date
df["hour"] = df["event_time"].dt.hour
df["weekday"] = df["event_time"].dt.day_name()

# --- METRICS ---
views = df[df["event_type"] == "view"]
purchases = df[df["event_type"] == "purchase"]
total_views = len(views)
total_purchases = len(purchases)
conversion_rate = (total_purchases / total_views * 100) if total_views else 0
avg_price = purchases["price"].mean()

# --- TABS ---
tab1, tab2, tab3 = st.tabs(["📊 Market Overview", "⏰ Time Analysis", "🧺 Basket & Price"])

# --- TAB 1: MARKET OVERVIEW ---
with tab1:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("👀 Total Views", f"{total_views:,}")
    col2.metric("🛒 Purchases", f"{total_purchases:,}")
    col3.metric("🎯 Conversion Rate", f"{conversion_rate:.1f}%")
    col4.metric("💸 Avg. Price", f"${avg_price:.2f}")

    # --- Daily Events Stacked Bar ---
    st.subheader("Daily Xiaomi Events")
    st.markdown("This stacked bar chart shows how different types of user interactions (views, purchases) evolved each day.")
    daily_counts = df.groupby(["date", "event_type"]).size().reset_index(name="count")
    fig_bar = px.bar(daily_counts, x="date", y="count", color="event_type", barmode="stack",
                     labels={"date": "Date", "count": "Event Count", "event_type": "Event Type"},
                     title="📅 Daily Interaction Volume")
    st.plotly_chart(fig_bar, use_container_width=True)

    # --- Conversion Funnel ---
    st.subheader("Conversion Funnel")
    st.markdown("The funnel below illustrates the conversion from product views to purchases for Xiaomi phones.")
    funnel_data = pd.DataFrame({
        "Stage": ["Viewed", "Purchased"],
        "Count": [total_views, total_purchases]
    })
    fig_funnel = px.funnel(funnel_data, y="Stage", x="Count", title="🔁 Xiaomi Funnel: Views to Purchases")
    st.plotly_chart(fig_funnel, use_container_width=True)

    # --- Insight Box ---
    st.markdown(
        """
        <div style="background-color:#e6f4ff;padding:15px;border-radius:10px;">
        🧠 <b>Insight:</b> The overall conversion rate is {:.1f}%, suggesting potential to improve the purchase rate through better retargeting or pricing strategies.
        </div>
        """.format(conversion_rate), unsafe_allow_html=True
    )

# --- TAB 2: TIME ANALYSIS ---
with tab2:
    st.subheader("Products Viewed")
    view_heat = views.groupby(["weekday", "hour"]).size().reset_index(name="count")
    heatmap1 = view_heat.pivot(index="weekday", columns="hour", values="count").fillna(0)
    fig_view = px.imshow(heatmap1, aspect="auto", text_auto=True, color_continuous_scale="Blues",
                         title="Products Viewed")
    st.plotly_chart(fig_view, use_container_width=True)

    st.subheader("Products Purchased")
    purchase_heat = purchases.groupby(["weekday", "hour"]).size().reset_index(name="count")
    heatmap2 = purchase_heat.pivot(index="weekday", columns="hour", values="count").fillna(0)
    fig_purchase = px.imshow(heatmap2, aspect="auto", text_auto=True, color_continuous_scale="Reds",
                             title="Products Purchased")
    st.plotly_chart(fig_purchase, use_container_width=True)

    # --- Insight Box ---
    st.markdown(
        """
        <div style="background-color:#e6f4ff;padding:15px;border-radius:10px;">
        🧠 <b>Insight:</b> Most Xiaomi product views and purchases occur in the evening. Consider timing campaigns between 6–10 PM for maximum impact.
        </div>
        """, unsafe_allow_html=True
    )

# --- TAB 3: BASKET & PRICE ---
with tab3:
    st.subheader("Price Distribution of Purchases")
    st.markdown("This histogram shows how Xiaomi product prices are distributed among purchases.")
    fig_price = px.histogram(purchases, x="price", nbins=30, title="Distribution of Purchase Prices",
                             labels={"price": "Price (USD)", "count": "Frequency"})
    st.plotly_chart(fig_price, use_container_width=True)

    if "basket" in df.columns and df["basket"].notna().sum() > 0:
        st.subheader("Most Common Co-Purchased Items")
        st.markdown("These are the most frequent other items in the basket with Xiaomi phones.")
        basket_items = purchases["basket"].dropna().str.split(",").explode().str.strip()
        top_basket = basket_items.value_counts().head(10).reset_index()
        top_basket.columns = ["Item", "Frequency"]
        st.dataframe(top_basket)

    # --- Insight Box ---
    st.markdown(
        """
        <div style="background-color:#e6f4ff;padding:15px;border-radius:10px;">
        🧠 <b>Insight:</b> Xiaomi phones are often bundled with recurring accessories. Highlighting bundle deals could drive higher basket sizes.
        </div>
        """, unsafe_allow_html=True
    )
