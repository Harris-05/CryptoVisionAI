import streamlit as st
import pandas as pd
import CrptoTool as ct
import warnings
import matplotlib.pyplot as plt
import joblib
warnings.filterwarnings('ignore')

# Page Config
st.set_page_config(page_title="Crypto Dashboard", layout="wide")
st.title("Crypto Investment Dashboard")

# --- METRIC DASHBOARD ---
st.subheader(" Key Market Metrics")
col1, col2, col3 = st.columns(3)
col1.metric("Total Coins Tracked", len(ct.df))
col2.metric("Highest Market Cap Coin", ct.df.loc[ct.df['market_cap'].idxmax(), 'name'])
col3.metric("Top Gainer (24h)", ct.gainer.iloc[0]['name'])

st.divider()

# --- SECTION 1: LONG-TERM INVESTMENTS ---
st.subheader(" Top 10 Long-Term Investment Coins")
if st.button("Get Best Long-Term Investments"):
    st.dataframe(ct.top_investments[['name', 'investment_score', 'growth', 'marketcap_change', 'circulating_change']])

st.divider()

# --- SECTION 2: SHORT-TERM INVESTMENTS ---
st.subheader(" Top 10 Short-Term Investment Coins")
if st.button("Get Best Short-Term Investments"):
    st.dataframe(ct.top_short_term[['name', 'short_investment_score', 'price_change_24h', 'total_volume', 'market_cap']])

st.divider()

st.subheader(" Coin Price History")
coin = st.selectbox("Select a coin to plot:", ct.df['id'])
if st.button("Show Price Chart"):
    fig = plt.figure(figsize=(8, 4))
    ct.plot_coin_price(coin)
    st.pyplot(fig)

st.divider()

st.subheader(" Details About Coin")
coin = st.selectbox("Select a coin:", ct.df['id'], key="coin_details")
if st.button("Coin Details"):
    st.dataframe(ct.df[ct.df["id"] == coin])

st.divider()

st.subheader(" Market Analysis")

analysis_options = {
    "Top 10 Coins by Market Cap": ct.topcoinsmarketcap,
    "Top 10 Coins by Market Rank": ct.topcoinsmarketrank,
    "Top 10 Cheapest Coins": ct.topcoinsprice,
    "Top 10 Most Expensive Coins": ct.leastcoinsprice,
    "Top 10 Losers (24h)": ct.loser,
    "Top 10 Gainers (24h)": ct.gainer,
    "Top 10 Most Volatile Coins": ct.mostvolatile,
    "Top 10 Rarest Coins": ct.rarity,
    "Top 10 Coins by Volume": ct.topvolume,
}

selected_analysis = st.selectbox("Select Market Analysis Type:", list(analysis_options.keys()))
st.dataframe(analysis_options[selected_analysis][['id', 'name', 'market_cap', 'current_price', 'price_change_24h']])

if st.button("Visualize"):
    selected_df = analysis_options[selected_analysis]
    fig, ax = plt.subplots(figsize=(10, 4))
    if 'price_change_24h' in selected_df.columns:
        colors = ['green' if val >= 0 else 'red' for val in selected_df['price_change_24h']]
        ax.bar(selected_df['id'], selected_df[selected_df.columns[2]], color=colors)
    else:
        ax.bar(selected_df['id'], selected_df[selected_df.columns[2]], color='blue')
    
    ax.set_xticklabels(selected_df['id'], rotation=45, ha="right")
    ax.set_title(selected_analysis)
    st.pyplot(fig)

st.divider()

st.subheader(" Market Movers & Growers")
col1, col2 = st.columns(2)

with col1:
    st.markdown("**Top Movers (Price Change Over Time)**")
    st.dataframe(ct.mover.head(10))

with col2:
    st.markdown("**Top Growers (% Growth Over Time)**")
    st.dataframe(ct.most_grow.head(10))

st.divider()

st.subheader(" Price Per Supply Analysis")
price_supply_df = ct.df[['id', 'name', 'market_cap', 'circulating_supply', 'price_per_supply']].sort_values(by="price_per_supply", ascending=False).head(10)
st.dataframe(price_supply_df)

st.subheader("AI Growth Predictor")


# Load trained model
try:
    model = joblib.load("crypto_predictor.pkl")
    features = joblib.load("crypto_features.pkl")
except:
    st.error("Train the model first by running crypto_predictor.py")
    model = None

if model:
    selected_coin = st.selectbox("Select a coin for prediction:", ct.df['id'], key="ai_predict")
    coin_data = ct.df[ct.df['id'] == selected_coin][features]

    if st.button("Predict Movement"):
        prediction = model.predict(coin_data)[0]
        probability = model.predict_proba(coin_data)[0]

        movement = "UP" if prediction == 1 else "DOWN "
        st.success(f"Prediction: **{movement}** (Confidence: {max(probability)*100:.2f}%)")

        # Visualization of probabilities
        st.bar_chart(pd.DataFrame({
            'Movement': ['DOWN', 'UP'],
            'Probability': probability
        }).set_index('Movement'))