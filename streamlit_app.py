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
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "\U0001F4CA Market Overview",
    "⏰ Time Analysis",
    "\U0001F6CD Basket & Pricing",
    "\U0001F9EA Customer Segments",
    "\U0001F52E Predictive Insights"
])

# --- TAB 1: MARKET OVERVIEW ---
with tab1:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("\U0001F440 Total Views", f"{total_views:,}")
    col2.metric("\U0001F6D2 Purchases", f"{total_purchases:,}")
    col3.metric("\U0001F3AF Conversion Rate", f"{conversion_rate:.1f}%")
    col4.metric("\U0001F4B8 Avg. Price", f"${avg_price:.2f}")

    daily_counts = df.groupby(["date", "event_type"]).size().reset_index(name="count")
    fig_bar = px.bar(daily_counts, x="date", y="count", color="event_type", barmode="stack",
                     labels={"date": "Date", "count": "Event Count", "event_type": "Event Type"},
                     title="\U0001F4C5 Daily Interaction Volume")
    st.plotly_chart(fig_bar, use_container_width=True)

    funnel_data = pd.DataFrame({
        "Stage": ["Viewed", "Purchased"],
        "Count": [total_views, total_purchases]
    })
    fig_funnel = px.funnel(funnel_data, y="Stage", x="Count",
                           color="Stage",
                           color_discrete_map={"Viewed": "#636EFA", "Purchased": "#EF553B"},
                           title="\U0001F501 Xiaomi Funnel: Views to Purchases")
    st.plotly_chart(fig_funnel, use_container_width=True)

    st.markdown("""
        <div style="background-color:#e6f4ff;padding:15px;border-radius:10px;">
        \U0001F9E0 <b>Insight:</b> Conversion rate stands at {:.1f}%. Focused promotions or retargeting strategies may boost this further.
        </div>
    """.format(conversion_rate), unsafe_allow_html=True)

# --- TAB 2: TIME ANALYSIS ---
with tab2:
    hourly_df = pd.DataFrame({
        "Hour": range(24),
        "Views": views.groupby("hour").size().reindex(range(24), fill_value=0).values,
        "Purchases": purchases.groupby("hour").size().reindex(range(24), fill_value=0).values
    })
    fig_line = px.line(hourly_df, x="Hour", y=["Views", "Purchases"],
                       title="\U0001F550 Hourly Xiaomi Activity",
                       labels={"value": "Event Count", "variable": "Event Type"})
    st.plotly_chart(fig_line, use_container_width=True)

    st.subheader("Products Viewed")
    heatmap1 = views.groupby(["weekday", "hour"]).size().unstack(fill_value=0)
    fig_view = px.imshow(heatmap1, aspect="auto", text_auto=True, color_continuous_scale="Blues",
                         title="Products Viewed")
    st.plotly_chart(fig_view, use_container_width=True)

    st.subheader("Products Purchased")
    heatmap2 = purchases.groupby(["weekday", "hour"]).size().unstack(fill_value=0)
    fig_purchase = px.imshow(heatmap2, aspect="auto", text_auto=True, color_continuous_scale="Reds",
                             title="Products Purchased")
    st.plotly_chart(fig_purchase, use_container_width=True)

    st.markdown("""
        <div style="background-color:#e6f4ff;padding:15px;border-radius:10px;">
        \U0001F9E0 <b>Insight:</b> Engagement peaks in the evening. Time your marketing campaigns between 6–10 PM.
        </div>
    """, unsafe_allow_html=True)

# --- TAB 3: BASKET & PRICING ---
with tab3:
    fig_price = px.histogram(purchases, x="price", nbins=30,
                             title="\U0001F4B2 Price Distribution of Purchases",
                             labels={"price": "Price (USD)", "count": "Frequency"})
    st.plotly_chart(fig_price, use_container_width=True)

    box_fig = px.box(purchases, y="price", title="Price Range (Box Plot)")
    st.plotly_chart(box_fig, use_container_width=True)

    avg_price_trend = purchases.groupby("date")["price"].mean().reset_index()
    fig_avg_price = px.line(avg_price_trend, x="date", y="price", title="Average Purchase Price Over Time")
    st.plotly_chart(fig_avg_price, use_container_width=True)

    if "basket" in purchases.columns and purchases["basket"].notna().sum() > 0:
        basket_items = purchases["basket"].dropna().str.split(",").explode().str.strip()
        top_basket = basket_items.value_counts().head(10).reset_index()
        top_basket.columns = ["Item", "Frequency"]
        st.dataframe(top_basket)

        # Basket Analysis
        transactions = purchases["basket"].dropna().apply(lambda x: list(set(map(str.strip, x.split(","))))).tolist()
        te = TransactionEncoder()
        te_ary = te.fit(transactions).transform(transactions)
        df_basket = pd.DataFrame(te_ary, columns=te.columns_)
        frequent_itemsets = apriori(df_basket, min_support=0.01, use_colnames=True)
        rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)
        st.write("Top Association Rules:")
        st.dataframe(rules.sort_values("confidence", ascending=False).head(10))

    st.markdown("""
        <div style="background-color:#e6f4ff;padding:15px;border-radius:10px;">
        \U0001F9E0 <b>Insight:</b> Popular co-purchases can be bundled to increase basket size.
        </div>
    """, unsafe_allow_html=True)

# --- TAB 4: CUSTOMER SEGMENTS ---
with tab4:
    cluster_data = purchases[["price", "hour"]].dropna()
    km = KMeans(n_clusters=3, n_init=10)
    cluster_data["Cluster"] = km.fit_predict(cluster_data)
    fig_cluster = px.scatter(cluster_data, x="hour", y="price", color="Cluster",
                             title="Customer Segments by Hour and Price",
                             labels={"hour": "Purchase Hour", "price": "Price (USD)"})
    st.plotly_chart(fig_cluster, use_container_width=True)

    st.markdown("""
        <div style="background-color:#e6f4ff;padding:15px;border-radius:10px;">
        \U0001F9E0 <b>Insight:</b> Different customer segments shop at different times and price points. Use this for targeted messaging.
        </div>
    """, unsafe_allow_html=True)

# --- TAB 5: PREDICTIVE INSIGHTS ---
with tab5:
    df_predict = df[df["event_type"].isin(["view", "purchase"])]
    df_predict["label"] = (df_predict["event_type"] == "purchase").astype(int)
    features = df_predict[["price", "hour"]]
    labels = df_predict["label"]

    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    st.text("\nModel: Random Forest Classifier")
    st.text(classification_report(y_test, y_pred))

    st.markdown("""
        <div style="background-color:#e6f4ff;padding:15px;border-radius:10px;">
        \U0001F9E0 <b>Insight:</b> Basic behavioral features can predict purchases with reasonable accuracy. Integrate this into campaign targeting.
        </div>
    """, unsafe_allow_html=True)

st.caption("Data: Hugging Face · Xiaomi only · October 2019")
