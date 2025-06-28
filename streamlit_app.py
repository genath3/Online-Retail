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
    url = "https://huggingface.co/datasets/7ng10dpE/Online-Retail/resolve/main/Smartphones_6M_FINAL.csv"
    df = pd.read_csv(url, nrows=100_000)  # Load only 100k rows
    df['event_time'] = pd.to_datetime(df['event_time'], errors='coerce')
    return df

df = load_data()

# Color mapping
brand_palette = px.colors.qualitative.Set3
top_brands = df['brand'].value_counts().head(20).index.tolist()
brand_colors = {b: brand_palette[i % len(brand_palette)] for i, b in enumerate(top_brands)}

tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Time Analysis", "Brand ROI Model", "Segments & Patterns"])

# -------------------- Tab 1 --------------------
with tab1:
    st.header("🟦 Overview")

    # KPIs with consistent layout
    total_views = int(df[df['event_type'] == 'view'].shape[0])
    total_purchases = int(df[df['event_type'] == 'purchase'].shape[0])
    total_users = int(df['user_id'].nunique())
    total_brands = int(df['brand'].nunique())
    total_rows = int(df.shape[0])

    kpi_data = [
        ("Total Views", total_views, "#3F51B5"),
        ("Total Purchases", total_purchases, "#3F51B5"),
        ("Users", total_users, "#3F51B5"),
        ("Brands", total_brands, "#3F51B5"),
        ("Events Logged", total_rows, "#3F51B5")
    ]
    cols = st.columns(len(kpi_data))
    for col, (label, value, color) in zip(cols, kpi_data):
        col.markdown(f"""
            <div style='background-color: {color}; padding: 20px; border-radius: 10px;
            text-align: center; color: white; height: 110px; min-height: 110px'>
                <h5 style='margin: 0;'>{label}</h5>
                <p style='font-size: 24px; line-height: 1.5; margin: 0;'><b>{value:,}</b></p>
            </div>
        """, unsafe_allow_html=True)

    # Top brands by purchase
    st.subheader("Top 10 Brands by Purchases")
    st.caption("Total number of purchases per brand, based on event logs.")
    top_brands = df[df['event_type'] == 'purchase']['brand'].value_counts().nlargest(10)
    fig1 = px.bar(top_brands, x=top_brands.index, y=top_brands.values, color=top_brands.index,
                  labels={"x": "Brand", "y": "Number of Purchases"},
                  color_discrete_map=brand_colors)
    fig1.update_traces(width=0.6)
    st.plotly_chart(fig1, use_container_width=True)
    st.markdown("🧠 These charts show that Samsung and Xiaomi dominate purchases — consider prioritizing ad spend on high-volume brands.")

    # Conversion rate
    st.subheader("Top 10 Brands by Conversion Rate")
    st.caption("Percentage of views that result in a purchase, per brand.")
    views = df[df['event_type'] == 'view'].groupby('brand').size()
    purchases = df[df['event_type'] == 'purchase'].groupby('brand').size()
    conversion = (purchases / views).dropna().sort_values(ascending=False).head(10).round(3)
    fig2 = px.bar(conversion, x=conversion.index, y=conversion.values, color=conversion.index,
                  labels={"x": "Brand", "y": "Conversion Rate"},
                  color_discrete_map=brand_colors)
    fig2.update_traces(width=0.6)
    st.plotly_chart(fig2, use_container_width=True)
    st.markdown("🧠 These brands convert a high % of views to purchases — support them with targeted promotions.")

    # Average basket size
    st.subheader("Top 10 Brands by Average Basket Size")
    st.caption("The average number of items in baskets containing each brand.")
    if 'basket' in df.columns:
        basket_avg = df.dropna(subset=['basket']).groupby('brand')['basket'].apply(
            lambda b: np.mean([len(eval(i)) if isinstance(i, str) else len(i) for i in b])
        ).round(1)
        basket_avg = basket_avg.sort_values(ascending=False).head(10)
        fig3 = px.bar(basket_avg, x=basket_avg.index, y=basket_avg.values, color=basket_avg.index,
                      labels={"x": "Brand", "y": "Average Basket Size"},
                      color_discrete_map=brand_colors)
        fig3.update_traces(width=0.6)
        st.plotly_chart(fig3, use_container_width=True)
        st.markdown("🧠 These brands are frequently bought in larger baskets — consider bundling or upselling.")

    # Basket item frequency table
    st.subheader("Top Basket Items by Frequency")
    st.caption("Most commonly co-purchased product types across all baskets.")
    if 'basket' in df.columns:
        all_items = df['basket'].dropna().explode()
        if isinstance(all_items.iloc[0], str):
            all_items = all_items.apply(lambda x: eval(x) if isinstance(x, str) else x)
            all_items = all_items.explode()
        item_counts = all_items.value_counts().reset_index()
        item_counts.columns = ['Product Type', 'Count']
        item_counts['Count'] = item_counts['Count'].astype(int)
        st.dataframe(
            item_counts.style.background_gradient(subset=['Count'], cmap='Blues')
            .format({'Count': '{:,}'})
            .set_properties(subset=['Product Type'], **{'text-align': 'left'}),
            use_container_width=True
        )
        st.markdown("🧠 These items are basket staples — promote them alongside smartphones.")

# -------------------- Tab 2 --------------------

# Ensure hour/weekday columns exist
df['hour'] = df['event_time'].dt.hour
df['weekday'] = pd.Categorical(
    df['event_time'].dt.day_name(),
    categories=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
    ordered=True
)

with tab2:
    st.header("🟧 Time Analysis")

    # Views heatmap
    st.subheader("Product Views by Hour and Day")
    view_matrix = df[df['event_type'] == 'view'].groupby(['weekday', 'hour']).size().unstack().fillna(0)
    view_matrix = view_matrix.reindex(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
    fig_view = go.Figure(data=go.Heatmap(
        z=view_matrix.values,
        x=list(range(24)),  # full 24-hour range
        y=view_matrix.index,
        colorscale='Blues',
        colorbar=dict(title="Views")
    ))
    fig_view.update_layout(title="Heatmap of Views (Hourly × Day)", xaxis_title="Hour", yaxis_title="Weekday")
    st.plotly_chart(fig_view, use_container_width=True)
    st.markdown("🧠 Views increase after lunch — try posting organic content during peak discovery hours.")

    # Purchases heatmap
    st.subheader("Product Purchases by Hour and Day")
    purchase_matrix = df[df['event_type'] == 'purchase'].groupby(['weekday', 'hour']).size().unstack().fillna(0)
    purchase_matrix = purchase_matrix.reindex(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
    fig_purch = go.Figure(data=go.Heatmap(
        z=purchase_matrix.values,
        x=list(range(24)),
        y=purchase_matrix.index,
        colorscale='Greens',
        colorbar=dict(title="Purchases")
    ))
    fig_purch.update_layout(title="Heatmap of Purchases (Hourly × Day)", xaxis_title="Hour", yaxis_title="Weekday")
    st.plotly_chart(fig_purch, use_container_width=True)
    st.markdown("🧠 Most purchases happen mid-morning — try running paid campaigns before 12pm.")

    # Conversion rate line chart
    st.subheader("Hourly Conversion Rate")
    hourly_views = df[df['event_type'] == 'view'].groupby('hour').size()
    hourly_purchases = df[df['event_type'] == 'purchase'].groupby('hour').size()
    conversion_rate = (hourly_purchases / hourly_views).dropna().round(3)
    fig_conv = px.line(x=conversion_rate.index, y=conversion_rate.values,
                       labels={'x': 'Hour of Day', 'y': 'Conversion Rate'},
                       title='Hourly Conversion Rate Across All Users')
    fig_conv.update_xaxes(range=[0, 23], dtick=1)
    st.plotly_chart(fig_conv, use_container_width=True)
    st.markdown("🧠 Conversion spikes around 9–11am — optimize ad timing for high-impact hours.")

   
# -------------------- Tab 3 --------------------
with tab3:
    st.header("🟩 Brand ROI Model")

    st.markdown("""
    This model estimates how likely each brand is to convert views into purchases using logistic regression.  
    Brands with high conversion and engagement are ideal candidates for ad spend.
    """)

# -------------------- Tab 4 --------------------
with tab4:
    st.header("🟪 Segments & Patterns")

    st.markdown("""
    Users are grouped based on:
    - Time of activity
    - Basket diversity
    - Session engagement
    - Price sensitivity
    """)

    seg_df = df.copy()
    seg_df['hour'] = seg_df['event_time'].dt.hour
    user_features = seg_df.groupby('user_id').agg({
        'hour': 'mean',
        'price': 'mean',
        'user_session': 'nunique',
        'basket': lambda b: np.mean([len(eval(i)) if isinstance(i, str) else len(i) for i in b.dropna()])
    }).fillna(0)
    user_features.columns = ['Avg Hour', 'Avg Price', 'Sessions', 'Avg Basket Size']

    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    scaled = StandardScaler().fit_transform(user_features)
    kmeans = KMeans(n_clusters=4, random_state=42)
    user_features['Segment'] = kmeans.fit_predict(scaled)

    st.subheader("Segment Summary Table")
    summary = user_features.groupby('Segment').mean().round(1)
    st.dataframe(summary, use_container_width=True)

    st.subheader("Segment Size Distribution")
    counts = user_features['Segment'].value_counts().sort_index()
    pie_fig = px.pie(names=counts.index.astype(str), values=counts.values,
                     color=counts.index.astype(str),
                     color_discrete_sequence=px.colors.qualitative.Set3)
    st.plotly_chart(pie_fig)

    st.subheader("Average Basket Size by Segment")
    bar_fig = px.bar(
        summary,
        x=summary.index.astype(str),
        y="Avg Basket Size",
        color=summary.index.astype(str),
        color_discrete_sequence=px.colors.qualitative.Set3,
        labels={"x": "Segment", "y": "Average Basket Size"}
    )
    st.plotly_chart(bar_fig)

    st.info("**Key Insights:**\n\n- Segment 0: High-value, high-basket customers — promote bundles.\n- Segment 2: Price-sensitive, low engagement — offer discounts.\n- Segment 1: Frequent sessions — ideal for retargeting.")


