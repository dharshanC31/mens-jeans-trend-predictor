import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Myntra Trend Predictor", layout="wide")
st.title("👖 Men's Jeans Trend Predictor — Myntra")
st.markdown("*Predicting trending products using price, discount & rating signals*")

@st.cache_data
def load_data():
    df = pd.read_csv("data/myntra_products.csv")
    df = df.dropna()
    df["discount_percent"] = df["discount_percent"].apply(
        lambda x: x/100 if x > 1 else x)
    df["trend_score"] = np.log1p(
        (df["ratings"] * df["number_of_ratings"] * df["discount_percent"])
        / df["price"])
    threshold = df["trend_score"].quantile(0.75)
    df["trending"] = df["trend_score"] > threshold
    return df

data = load_data()

st.sidebar.header("🔍 Filter Products")
max_price = st.sidebar.slider("Max Price (₹)", 300, 5000, 1500)
min_rating = st.sidebar.slider("Min Rating", 3.0, 5.0, 3.5)
only_trending = st.sidebar.checkbox("Show Trending Only", value=False)

filtered = data[(data["price"] <= max_price) & (data["ratings"] >= min_rating)]
if only_trending:
    filtered = filtered[filtered["trending"] == True]

st.markdown("### 📊 Overview")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Products", f"{len(filtered):,}")
col2.metric("Trending Products", f"{filtered['trending'].sum():,}")
col3.metric("Avg Price", f"₹{filtered['price'].mean():.0f}")
col4.metric("Avg Rating", f"{filtered['ratings'].mean():.2f}")

st.markdown("### 🔥 Top Trending Products")
top10 = filtered[filtered["trending"]==True].sort_values("trend_score", ascending=False)[["brand_name","pants_description","price","ratings","discount_percent","trend_score"]].head(10).reset_index(drop=True)
top10["discount_percent"] = (top10["discount_percent"]*100).round(1)
top10.columns = ["Brand","Product","Price (₹)","Rating","Discount %","Trend Score"]
st.dataframe(top10, use_container_width=True)

st.markdown("### 📈 Insights")
col1, col2 = st.columns(2)

with col1:
    st.markdown("**Price Distribution: Trending vs Non-Trending**")
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.kdeplot(data=filtered[filtered["trending"]==True]["price"], label="Trending", fill=True, color="#5cb85c", ax=ax)
    sns.kdeplot(data=filtered[filtered["trending"]==False]["price"], label="Not Trending", fill=True, color="#d9534f", ax=ax)
    ax.set_xlabel("Price (₹)")
    ax.legend()
    st.pyplot(fig)

with col2:
    st.markdown("**Top Brands by Trending Products**")
    top_brands = filtered[filtered["trending"]==True]["brand_name"].value_counts().head(8)
    fig2, ax2 = plt.subplots(figsize=(7, 4))
    top_brands.plot(kind="barh", color="steelblue", ax=ax2)
    ax2.invert_yaxis()
    ax2.set_xlabel("Trending Products Count")
    st.pyplot(fig2)

st.markdown("### 🤖 Predict: Will This Product Trend?")
st.markdown("Enter product details to get a trend prediction.")

c1, c2, c3, c4 = st.columns(4)
inp_price = c1.number_input("Price (₹)", 300, 5000, 999)
inp_mrp = c2.number_input("MRP (₹)", 300, 8000, 1999)
inp_discount = c3.slider("Discount %", 10, 80, 50)
inp_rating = c4.slider("Rating", 3.0, 5.0, 4.0)

if st.button("Predict Trend 🚀"):
    features2 = data[["price","MRP","discount_percent","ratings","trending"]].dropna()
    X = features2.drop("trending", axis=1)
    y = features2["trending"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    inp = pd.DataFrame([[inp_price, inp_mrp, inp_discount/100, inp_rating]], columns=["price","MRP","discount_percent","ratings"])
    pred = clf.predict(inp)[0]
    prob = clf.predict_proba(inp)[0][1]
    if pred:
        st.success(f"✅ TRENDING — {prob*100:.1f}% confidence")
    else:
        st.error(f"❌ NOT TRENDING — only {prob*100:.1f}% trend probability")

st.markdown("---")
st.caption("Built by Dharshan C | B.Com Business Analytics | Data: Myntra (Kaggle)")



