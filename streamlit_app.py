import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Smartphone Dashboard", layout="wide")
st.title("Smartphone Sales Dashboard")

@st.cache_data
def load_data():
    url = "https://huggingface.co/datasets/7ng10dpE/Online-Retail/resolve/main/top10_brands_cleaned.csv.gz?raw=true"
    df = pd.read_csv(url, compression="gzip")
    df['event_time'] = pd.to_datetime(df['event_time'], errors='coerce')
    return df
df = load_data()

# Consistent brand colors
top_brands = df['brand'].unique().tolist()
palette = px.colors.qualitative.Plotly
brand_colors = {brand: palette[i % len(palette)] for i, brand in enumerate(top_brands)}

# ------------------ Tabs ------------------
tab1, tab2, tab3, tab4 = st.tabs(["Overall Market Overview", "Brand Deep Dive", "Time Analysis", "Basket & Segmentation"])

# ------------------ Tab 1: KPIs + Purchases ------------------
with tab1:
    st.subheader("Overall KPIs (Full Dataset Summary)")
    kpis = [
        ("Total Purchases", 337536),
        ("Total Views", 10597573),
        ("Users", 1299719),
        ("Sessions", 3097183),
        ("Brands", 41)
    ]
    cols = st.columns(len(kpis))
    for col, (label, value) in zip(cols, kpis):
        col.markdown(f"""
        <div style='background-color:#3b82f6;padding:20px;border-radius:15px;text-align:center;'>
            <h4 style='color:white;margin:0'>{label}</h4>
            <h2 style='color:white;margin:0'>{value:,}</h2>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("Top 10 Brands by Purchases")
    st.caption("Total number of purchases made for each of the top 10 smartphone brands.")
    purchase_counts = df[df['action'] == 'purchase'].groupby('brand').size().sort_values(ascending=False)
    fig = px.bar(purchase_counts, color=purchase_counts.index,
                 color_discrete_map=brand_colors,
                 labels={"value": "Number of Purchases", "index": "Brand"},
                 title="Total Purchases by Brand")
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("🧠 These charts show that brands like Samsung and Xiaomi dominate total purchases, therefore they should be prioritized in campaign targeting.")

    st.subheader("Views-to-Purchase Ratio by Brand (with Add-To-Cart)")
    st.caption("The ratio of total interactions (Views + Add-To-Cart) per Purchase.\n\n*Formula: ((Views + Add-To-Cart) / Purchases)")
    views = df[df['action'] == 'view'].groupby('brand').size()
    carts = df[df['action'] == 'cart'].groupby('brand').size()
    purchases = df[df['action'] == 'purchase'].groupby('brand').size()
    ratio = ((views + carts) / purchases).round(1)
    ratio_df = pd.DataFrame({"Brand": ratio.index, "Views-to-Purchase Ratio": ratio.values})
    st.dataframe(ratio_df.style.background_gradient(subset=['Views-to-Purchase Ratio'], cmap='Blues'), use_container_width=True)

    st.markdown("""
    <div style='background-color:#e0f2fe;padding:15px;border-radius:10px;'>
    🧠 <strong>These tables show that brands with lower ratios require fewer interactions to convert, therefore are more ad efficient.</strong>
    </div>
    """, unsafe_allow_html=True)

# ------------------ Tab 2: Brand Engagement ------------------
with tab2:
    st.subheader("Views to Purchase Ratio by Brand")
    st.caption("Shows how many product views are needed to trigger a purchase per brand. Lower is better.")
    view_purchase_ratio = ((views + 1) / (purchases + 1)).round(1)
    ratio_df = pd.DataFrame({"Brand": view_purchase_ratio.index, "View-to-Purchase Ratio": view_purchase_ratio.values})
    fig2 = px.bar(ratio_df.sort_values("View-to-Purchase Ratio"),
                  x="Brand", y="View-to-Purchase Ratio", color="Brand",
                  color_discrete_map=brand_colors,
                  title="View-to-Purchase Ratio by Brand")
    fig2.update_traces(width=0.6)
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("""
    <div style='background-color:#e0f2fe;padding:15px;border-radius:10px;'>
    🧠 <strong>This graph shows that some brands convert quickly with fewer views, therefore they may be undervalued in ROI analysis.</strong>
    </div>
    """, unsafe_allow_html=True)

# ------------------ Tab 3: Time Analysis ------------------
with tab3:
    st.subheader("Sessions by Hour and Day")
    st.caption("This heatmap highlights peak user activity across hours and weekdays.")
    df['hour'] = df['timestamp'].dt.hour
    df['weekday'] = pd.Categorical(df['timestamp'].dt.day_name(),
                                   categories=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
                                   ordered=True)
    heatmap_data = df.pivot_table(index="weekday", columns="hour", values="session_id", aggfunc="count").fillna(0)
    fig3 = px.imshow(heatmap_data,
                     labels=dict(x="Hour", y="Day", color="Sessions"),
                     color_continuous_scale="Blues",
                     title="Session Frequency by Hour and Day")
    st.plotly_chart(fig3, use_container_width=True)

    st.markdown("""
    <div style='background-color:#e0f2fe;padding:15px;border-radius:10px;'>
    🧠 <strong>This heatmap shows user peaks before midday and after 6pm, therefore ad timing should target those hours for highest visibility.</strong>
    </div>
    """, unsafe_allow_html=True)

# ------------------ Tab 4: Basket & Segmentation ------------------
with tab4:
    st.subheader("Market Basket Relationships")
    st.caption("The most common product combinations appearing in baskets during purchases.")

    basket_df = df[df["basket"].notna()].copy()
    top_baskets = basket_df["basket"].value_counts().head(10).reset_index()
    top_baskets.columns = ["Basket", "Frequency"]
    st.dataframe(top_baskets.style.background_gradient(subset=["Frequency"], cmap="Blues"), use_container_width=True)

    st.markdown("""
    <div style='background-color:#e0f2fe;padding:15px;border-radius:10px;'>
    🧠 <strong>These tables show recurring product pairings, therefore they can inform bundling and cross-sell promotion strategies.</strong>
    </div>
    """, unsafe_allow_html=True)
