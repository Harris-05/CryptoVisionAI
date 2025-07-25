# CryptoVisionAI
CryptoVision AI – Intelligent Crypto Analysis & Prediction Dashboard

CryptoVision AI is a data-driven crypto analytics dashboard powered by **Machine Learning, SQL Server, and Streamlit**.  
It provides real-time insights, investment rankings, market trends, and AI predictions of coin price movements, making it a perfect tool for traders and analysts.

## Features

### 1. Market Analytics
- Top 10 Long-Term Investment Coins  
  Ranked using custom scoring metrics that analyze growth rate, market cap trends, and circulating supply changes.
- Top 10 Short-Term Investment Coins  
  Uses 24h price changes, trading volume, and market volatility to highlight short-term opportunities.
- Comprehensive Market Overview
  - Top Gainers & Losers (24h)
  - Most Volatile Coins
  - Cheapest & Most Expensive Coins
  - Top Coins by Market Cap and Trading Volume

---

### 2. Interactive Visualizations
- Price History Charts:** Explore the price trends of any coin over time.
- Market Analysis Charts:** Bar charts with dynamic coloring (green for gains, red for losses).
- Customizable Filters:** Choose between different analysis types (e.g., volatility, rarity, volume).

---

### 3. AI Price Movement Prediction**
- Powered by Random Forest Classifier (Scikit-learn).**
- Predicts if the price of a coin is likely to go **UP or DOWN** in the next time interval.
- Uses historical data (price, market cap, volume, and supply) as features.
- Accuracy improves over time as more price history is stored.

---

### 4. SQL Server Integration
- All data fetched from CoinGecko API is stored in a SQL Server database.
- Historical price data is logged in a `pricehistory` table.
- Provides persistent storage for machine learning training and trend analysis.

---

## How It Works

1. Data Collection:  
   - Live data fetched from the CoinGecko API.
   - Data stored in `CryptoCoins` and `pricehistory` tables.

2. Data Analysis:  
   - `Pandas` is used for filtering, sorting, and ranking coins.
   - Volatility, growth rates, and price trends are computed dynamically.

3. Machine Learning:  
   - Random Forest Classifier is trained on historical data.  
   - Model predicts next movement: (1 = Price Increase, 0 = Price Decrease).
   - Model is saved with `joblib` for future predictions.

4. Visualization:  
   - Matplotlib and Streamlit render clean, interactive charts.

## Tech Stack

- Frontend: Streamlit (Python-based web framework)
- Backend: Python (Pandas, Matplotlib, Scikit-learn, Joblib)
- Database: SQL Server (via `pyodbc`)
- Data Source: CoinGecko API
- Machine Learning: Random Forest Classifier

