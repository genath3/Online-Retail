import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.cluster import KMeans
from huggingface_hub import hf_hub_download

# --- CONFIG ---
st.set_page_config(page_title="Xiaomi Dashboard", layout="wide")
st.title("Xiaomi Phones")
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
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Market Overview", "⏰ Time Analysis", "🛒 Basket & Pricing", "👥 Customer Segments", "🔮 Predictive Insights"
])

# --- TAB 1 ---
with tab1:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("👀 Total Views", f"{total_views:,}")
    col2.metric("🛒 Purchases", f"{total_purchases:,}")
    col3.metric("🎯 Conversion Rate", f"{conversion_rate:.1f}%")
    col4.metric("💸 Avg. Price", f"${avg_price:.2f}")

    daily_counts = df.groupby(["date", "event_type"]).size().reset_index(name="count")
    st.plotly_chart(px.bar(daily_counts, x="date", y="count", color="event_type", barmode="stack", title="📅 Daily Interaction Volume"), use_container_width=True)

    funnel_data = pd.DataFrame({"Stage": ["Viewed", "Purchased"], "Count": [total_views, total_purchases]})
    st.plotly_chart(px.funnel(funnel_data, x="Count", y="Stage", color="Stage",
                              color_discrete_map={"Viewed": "#636EFA", "Purchased": "#EF553B"},
                              title="🔁 Xiaomi Funnel: Views to Purchases"), use_container_width=True)

    st.markdown("""<div style="background-color:#e6f4ff;padding:15px;border-radius:10px;">
        🧠 <b>Insight:</b> Conversion rate stands at {:.1f}%. Targeted promotions may improve this.
        </div>""".format(conversion_rate), unsafe_allow_html=True)

# --- TAB 2 ---
with tab2:
    heatmap1 = views.groupby(["weekday", "hour"]).size().unstack(fill_value=0)
    st.plotly_chart(px.imshow(heatmap1, aspect="auto", text_auto=True, color_continuous_scale="Blues", title="Products Viewed"), use_container_width=True)

    heatmap2 = purchases.groupby(["weekday", "hour"]).size().unstack(fill_value=0)
    st.plotly_chart(px.imshow(heatmap2, aspect="auto", text_auto=True, color_continuous_scale="Reds", title="Products Purchased"), use_container_width=True)

    hourly_df = pd.DataFrame({
        "Hour": range(24),
        "Views": views.groupby("hour").size().reindex(range(24), fill_value=0).values,
        "Purchases": purchases.groupby("hour").size().reindex(range(24), fill_value=0).values
    })
    st.plotly_chart(px.line(hourly_df, x="Hour", y=["Views", "Purchases"], title="⏰ Hourly Xiaomi Activity"), use_container_width=True)

    st.markdown("""<div style="background-color:#e6f4ff;padding:15px;border-radius:10px;">
        🧠 <b>Insight:</b> Evening hours (6–10 PM) are peak times for engagement.
        </div>""", unsafe_allow_html=True)

# --- TAB 3 ---
with tab3:
    st.plotly_chart(px.histogram(purchases, x="price", nbins=30, title="💰 Price Distribution"), use_container_width=True)
    st.plotly_chart(px.box(purchases, y="price", title="Price Range (Box Plot)"), use_container_width=True)

    if "basket" in purchases.columns and purchases["basket"].notna().sum() > 0:
        basket_items = purchases["basket"].dropna().str.split(",").explode().str.strip()
        top_basket = basket_items.value_counts().head(10).reset_index()
        top_basket.columns = ["Item", "Frequency"]
        st.dataframe(top_basket)

    st.markdown("""<div style="background-color:#e6f4ff;padding:15px;border-radius:10px;">
        🧠 <b>Insight:</b> Popular basket items present upsell opportunities.
        </div>""", unsafe_allow_html=True)

# --- TAB 4 ---
with tab4:
    segment_df = purchases[["price", "hour"]].dropna()
    if len(segment_df) >= 10:
        kmeans = KMeans(n_clusters=3, random_state=42).fit(segment_df)
        segment_df["Segment"] = kmeans.labels_
        st.plotly_chart(px.scatter(segment_df, x="hour", y="price", color="Segment", title="👥 Customer Segments by Hour & Price"), use_container_width=True)

        st.markdown("""<div style="background-color:#e6f4ff;padding:15px;border-radius:10px;">
            🧠 <b>Insight:</b> Distinct customer segments emerge by time and price.
            </div>""", unsafe_allow_html=True)
    else:
        st.warning("Not enough purchase data to segment.")

# --- TAB 5 ---
with tab5:
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

    st.markdown("""<div style="background-color:#e6f4ff;padding:15px;border-radius:10px;">
        🧠 <b>Insight:</b> Even simple models can predict purchase intent using price and hour.
        </div>""", unsafe_allow_html=True)









