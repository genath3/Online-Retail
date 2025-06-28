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
st.set_page_config(page_title="Xiaomi Dashboard", layout="wide")
st.title("\U0001F4F1 Xiaomi Phones")
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
conversion_rate = (total_purchases / total_views * 100) if total_views else 0
avg_price = purchases["price"].mean()

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
    fig_bar = px.bar(
        daily_counts,
        x="date",
        y="count",
        color="event_type",
        color_discrete_map={"view": "#636EFA", "purchase": "#EF553B"},
        text="count",
        barmode="stack",
        labels={"date": "Date", "count": "Event Count", "event_type": "Event Type"},
        category_orders={"event_type": ["view", "purchase"]},
        title="📅 Daily Interaction Volume"
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    funnel_data = pd.DataFrame({
        "Stage": ["Viewed", "Purchased"],
        "Count": [total_views, total_purchases]
    })
    fig_funnel = px.funnel(
        funnel_data[::-1],
        y="Stage",
        x="Count",
        color="Stage",
        color_discrete_map={"Viewed": "#636EFA", "Purchased": "#EF553B"},
        title="🔁 Xiaomi Funnel: Views to Purchases"
    )
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
    # --- Price Sensitivity: Conversion Rate by Price Band ---
    st.subheader("📊 Conversion Rate by Price Range")
    bins = [0, 200, 400, 600, 800, 1000, np.inf]
    labels = ["<$200", "$200-400", "$400-600", "$600-800", "$800-1000", "$1000+"]
    df["price_bin"] = pd.cut(df["price"], bins=bins, labels=labels, include_lowest=True)

    conv_data = df[df["event_type"].isin(["view", "purchase"])]
    summary = conv_data.groupby(["price_bin", "event_type"]).size().unstack(fill_value=0)
    summary["conversion_rate"] = summary["purchase"] / summary["view"] * 100
    summary = summary.reset_index()

    fig_conv = px.bar(summary, x="price_bin", y="conversion_rate", title="💸 Conversion Rate by Price Range",
                      labels={"conversion_rate": "Conversion Rate (%)", "price_bin": "Price Range"})
    st.plotly_chart(fig_conv, use_container_width=True)
    fig_price = px.histogram(purchases, x="price", nbins=30,
                             title="\U0001F4B2 Price Distribution of Purchases",
                             labels={"price": "Price (USD)", "count": "Frequency"})
    st.plotly_chart(fig_price, use_container_width=True)

    box_fig = px.box(purchases, y="price", title="Price Range (Box Plot)")
    st.plotly_chart(box_fig, use_container_width=True)

    if "basket" in purchases.columns and purchases["basket"].notna().sum() > 0:
        basket_items = purchases["basket"].dropna().str.split(",").explode().str.strip()
        top_basket = basket_items.value_counts().head(10).reset_index()
        top_basket.columns = ["Item", "Frequency"]
        top_basket["Category"] = top_basket["Item"].apply(lambda x: x.split("_")[0] if "_" in x else "other")
        st.dataframe(top_basket)

    st.markdown("""
        <div style="background-color:#e6f4ff;padding:15px;border-radius:10px;">
        \U0001F9E0 <b>Insight:</b> Popular co-purchases can be bundled to increase basket size.
        </div>
    """, unsafe_allow_html=True)

# --- TAB 4: PREDICTIVE INSIGHTS ---
with tab4:
    st.subheader("Purchase Prediction (View vs. Purchase)")
    clf_df = df[df["event_type"].isin(["view", "purchase"])].copy()
    clf_df["label"] = (clf_df["event_type"] == "purchase").astype(int)

    X = clf_df[["price", "hour"]].fillna(0)
    y = clf_df["label"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    report = classification_report(y_test, y_pred, output_dict=False)
    st.text(report)

    st.markdown("""
        <div style="background-color:#e6f4ff;padding:15px;border-radius:10px;">
        \U0001F9E0 <b>Insight:</b> A simple model using price and time can moderately predict purchase behavior. Integrate with campaigns.
        </div>
    """, unsafe_allow_html=True)








