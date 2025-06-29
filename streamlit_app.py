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

# --- CONFIG ---
st.markdown("&nbsp;")  # invisible output to trigger Hugging Face render
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

with tab1:
    st.subheader("📊 Overview goes here...")
    st.write(df.head())

with tab2:
    st.subheader("⏰ Time Analysis")
    st.write("Tab under development")

with tab3:
    st.subheader("📦 Basket & Pricing")
    st.write("Tab under development")

with tab4:
    st.subheader("🤖 Purchase Prediction")
    price = st.slider("Price", 0, 1000, 500)
    hour = st.slider("Hour", 0, 23, 12)
    try:
        prob = model.predict_proba(pd.DataFrame({"price": [price], "hour": [hour]}))[0][1]
        st.metric("Predicted Purchase Probability", f"{prob * 100:.1f}%")
    except Exception as e:
        st.error(f"Model prediction failed: {e}")




