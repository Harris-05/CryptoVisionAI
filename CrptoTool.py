import pandas as pd
import requests as r
import matplotlib.pyplot as plt
import pyodbc
import warnings
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
warnings.filterwarnings("ignore")


def get_all_coin_ids(conn):
    query = "SELECT id FROM CryptoCoins"
    return pd.read_sql(query, conn)['id'].tolist()

def importdata():
    df = pd.DataFrame()
    for i in range(1, 3):
        url = "https://api.coingecko.com/api/v3/coins/markets"
        parameter = {
            "vs_currency": "usd",
            "order": "market_cap_desc",
            "per_page": 50,
            "page": i,
            "sparkline": False
        }
        response = r.get(url, params=parameter)
        data = response.json()
        df = pd.concat([df, pd.DataFrame(data)], ignore_index=True)
    return df


def exporttosql(df):
    conn = pyodbc.connect("DRIVER={SQL Server};SERVER=HARRIS-LAPTOP\\SQLEXPRESS;DATABASE=Crypto;Trusted_Connection=yes")
    cursor = conn.cursor()

    for index, row in df.iterrows():
        cursor.execute("""
            INSERT INTO CryptoCoins (id, name, symbol, currentprice, marketcap, marketrank, highesttday, lowesttday,
                                     pricechange, total_volume, circulating_supply, lastupdated)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        row['id'], row['name'], row['symbol'], row['current_price'], row['market_cap'], row['market_cap_rank'],
        row['high_24h'], row['low_24h'], row['price_change_24h'], row['total_volume'], row['circulating_supply'],
        row['last_updated'])

    conn.commit()
    cursor.close()
    conn.close()


def addupdatesql(df):
    conn = pyodbc.connect("DRIVER={SQL Server};SERVER=HARRIS-LAPTOP\\SQLEXPRESS;DATABASE=Crypto;Trusted_Connection=yes")
    cursor = conn.cursor()

    # Get existing coin IDs to avoid FK constraint error
    existing_coin_ids = get_all_coin_ids(conn)

    for index, row in df.iterrows():
        # If coin not in CryptoCoins, insert it
        if row['id'] not in existing_coin_ids:
            cursor.execute("""
                INSERT INTO CryptoCoins (id, name, symbol, currentprice, marketcap, marketrank, highesttday, lowesttday,
                                         pricechange, total_volume, circulating_supply, lastupdated)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            row['id'], row['name'], row['symbol'], row['current_price'], row['market_cap'], row['market_cap_rank'],
            row['high_24h'], row['low_24h'], row['price_change_24h'], row['total_volume'],
            row['circulating_supply'], row['last_updated'])

        # Always insert into pricehistory
        cursor.execute("""
            INSERT INTO pricehistory (cid, price, marketrank,marketcap,circulating_supply, timeentery)
            VALUES (?, ?, ?, ?,?,?)
        """,
        row['id'], row['current_price'], row['market_cap_rank'],row['market_cap'],row['circulating_supply'], row['last_updated'])

    conn.commit()
    cursor.close()
    conn.close()


def gethistory(coin_id, conn=None):
    close_conn = False
    if conn is None:
        conn = pyodbc.connect("DRIVER={SQL Server};SERVER=HARRIS-LAPTOP\\SQLEXPRESS;DATABASE=Crypto;Trusted_Connection=yes")
        close_conn = True

    query = """
    SELECT timeentery, price, marketrank,marketcap,circulating_supply
    FROM pricehistory
    WHERE cid = ?
    ORDER BY timeentery ASC
    """
    try:
        df = pd.read_sql(query, conn, params=[coin_id])
        if df.empty:
            return pd.DataFrame()  # early exit
        df['timeentery'] = pd.to_datetime(df['timeentery'], errors='coerce')
        df['circulating_supply'] = pd.to_numeric(df['circulating_supply'], errors='coerce')
        df.set_index('timeentery', inplace=True)
        df = df.dropna(subset=['price'])  # Remove rows with missing price
        return df
    except Exception as e:
        print(f"Error loading history for {coin_id}: {e}")
        return pd.DataFrame()
    finally:
        if close_conn:
            conn.close()

def build_full_history_dataset():
    conn = pyodbc.connect("DRIVER={SQL Server};SERVER=HARRIS-LAPTOP\\SQLEXPRESS;DATABASE=Crypto;Trusted_Connection=yes")
    all_ids = get_all_coin_ids(conn)
    master_df = pd.DataFrame()

    for coin_id in all_ids:
        try:
            df = gethistory(coin_id, conn)
            df["coin_id"] = coin_id
            master_df = pd.concat([master_df, df], ignore_index=False)
        except Exception as e:
            print(f"Error loading history for {coin_id}: {e}")

    conn.close()
    return master_df


def plot_price_change_history():
    conn = pyodbc.connect("DRIVER={SQL Server};SERVER=HARRIS-LAPTOP\\SQLEXPRESS;DATABASE=Crypto;Trusted_Connection=yes")
    all_ids = get_all_coin_ids(conn)
    for coin_id in all_ids:
        df = gethistory(coin_id, conn)
        if len(df) < 2:
            continue
        df["price_change"] = df["price"].diff()
        df["abs_change"] = df["price_change"].abs()
        print(f"\n=== {coin_id} ===")
        print(df[["price", "price_change", "marketrank"]].tail())


def plot_coin_price(coin_id):
    conn = pyodbc.connect("DRIVER={SQL Server};SERVER=HARRIS-LAPTOP\\SQLEXPRESS;DATABASE=Crypto;Trusted_Connection=yes")
    try:
        df = gethistory(coin_id, conn)
        if df.empty:
            print(f"No history found for coin: {coin_id}")
            return

        if 'price' not in df.columns:
            print(f"Price column missing for {coin_id}")
            return

        df = df.dropna(subset=['price'])  # Remove rows with NaN price
        if df.empty:
            print(f"No valid price data to plot for {coin_id}")
            return

        df['price'].plot(title=f"{coin_id} Price Over Time", figsize=(10, 4))
        plt.ylabel("Price")
        plt.xlabel("Time")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"Error plotting price for {coin_id}: {e}")

    finally:
        conn.close()



def top_movers():
    conn = pyodbc.connect("DRIVER={SQL Server};SERVER=HARRIS-LAPTOP\\SQLEXPRESS;DATABASE=Crypto;Trusted_Connection=yes")
    all_ids = get_all_coin_ids(conn)
    changes = []

    for coin_id in all_ids:
        df = gethistory(coin_id, conn)
        if len(df) < 2:
            continue
        change = df['price'].iloc[-1] - df['price'].iloc[0]
        changes.append((coin_id, change))

    sorted_changes = sorted(changes, key=lambda x: abs(x[1]), reverse=True)
    sort_ch=pd.DataFrame(sorted_changes)
    return sort_ch
    
def most_growers():
    conn = pyodbc.connect("DRIVER={SQL Server};SERVER=HARRIS-LAPTOP\\SQLEXPRESS;DATABASE=Crypto;Trusted_Connection=yes")
    all_ids = get_all_coin_ids(conn)
    growth_list = []

    for coin_id in all_ids:
        df = gethistory(coin_id, conn)
        if len(df) < 2:
            continue
        initial_price = df['price'].iloc[0]
        final_price = df['price'].iloc[-1]

        if initial_price > 0:
            growth_percentage = ((final_price - initial_price) / initial_price) * 100
            growth_list.append((coin_id, growth_percentage))

    # Sort by highest percentage growth
    sorted_growth = sorted(growth_list, key=lambda x: x[1], reverse=True)
    sr_gr= pd.DataFrame(sorted_growth)


    conn.close()
    return sr_gr

def investment_score(row):
    score = (
        row['growth'] * 0.3 +
        row['marketcap_change'] * 0.25 +
        (row['current_marketcap'] / 1e9) * 0.25 +  # Scale down (in billions)
        - row['circulating_change'] * 0.15 +
        (1 / row['current_circulating']) * 0.05
    )
    return round(score, 3)
def short_term_investment_score(row):
    price_change = row['price_change_24h'] if pd.notna(row['price_change_24h']) else 0
    volume_score = row['total_volume'] / 1e7  # Scaled to tens of millions
    marketcap_score = row['market_cap'] / 1e9  # Scaled to billions
    supply_score = 1 / (row['circulating_supply'] if row['circulating_supply'] > 0 else 1)

    score = (
        price_change * 0.55 +  
        volume_score * 0.2 +     
        marketcap_score * 0.15 +  
        supply_score * 0.1       
    )
    return round(score, 3)

def top_invest_long():
    growth_data = []

    for coin_id in full_history_df['coin_id'].unique():
        coin_df = full_history_df[full_history_df['coin_id'] == coin_id].copy()
        if len(coin_df) < 2:
            continue

    # Step 2: % growth in price
        initial_price = coin_df['price'].iloc[0]
        final_price = coin_df['price'].iloc[-1]
        growth = ((final_price - initial_price) / initial_price) * 100 if initial_price else 0

    # Step 3: % change in market cap
        initial_marketcap = coin_df['marketcap'].iloc[0]
        final_marketcap = coin_df['marketcap'].iloc[-1]  # <-- FIXED
        marketcap_change = ((final_marketcap - initial_marketcap) / initial_marketcap) * 100 if initial_marketcap else 0

    # Step 4: % change in circulating supply
        initial_supply = coin_df['circulating_supply'].iloc[0]
        final_supply = coin_df['circulating_supply'].iloc[-1]
        circulating_change = ((final_supply - initial_supply) / initial_supply) * 100 if initial_supply else 0

    # Step 5: Get latest circulating supply
        current_circulating = final_supply if final_supply > 0 else 1  # avoid div by 0

        growth_data.append({
            'id': coin_id,
            'growth': growth,
            'marketcap_change': marketcap_change,
            'current_marketcap': final_marketcap,  # <-- New field
            'circulating_change': circulating_change,
            'current_circulating': current_circulating
        })

# Step 6: Create DF and apply investment formula
    investment_df = pd.DataFrame(growth_data)
    investment_df['investment_score'] = investment_df.apply(investment_score, axis=1)

# Sort by best investment score
    top_investments = investment_df.sort_values(by='investment_score', ascending=False).head(10)

# Merge names for display
    coin_names = df[['id', 'name']]
    top_investments = top_investments.merge(coin_names, on='id', how='left')
    return top_investments

def top_invest_short():
    short_term_df = df[['id', 'name', 'price_change_24h', 'total_volume', 'market_cap', 'circulating_supply']].copy()
    short_term_df['short_investment_score'] = short_term_df.apply(short_term_investment_score, axis=1)

    top_short_term = short_term_df.sort_values(by='short_investment_score', ascending=False).head(10)
    return top_short_term
# ===================== MAIN EXECUTION ======================

df = importdata()

# First-time only
# exporttosql(df)

addupdatesql(df)

# ============ EDA ============
topcoinsmarketcap = df.sort_values(by="market_cap", ascending=False).head(10)
topcoinsmarketrank = df.sort_values(by="market_cap_rank", ascending=True).head(10)
topcoinsprice = df.sort_values(by="current_price", ascending=True).head(10)
leastcoinsprice = df.sort_values(by="current_price", ascending=False).head(10)
loser = df.sort_values(by="price_change_24h", ascending=True).head(10)
gainer = df.sort_values(by="price_change_24h", ascending=False).head(10)
df["volatility"] = df["high_24h"] - df["low_24h"]
mostvolatile = df.sort_values(by="volatility", ascending=False).head(10)
rarity = df.sort_values(by="circulating_supply", ascending=True).head(10)
topvolume = df.sort_values(by="total_volume", ascending=False).head(10)
df["price_change_percent"] = df["price_change_24h"] / df["current_price"] * 100

# ============ VISUALS ============
visuals = [
    (topcoinsmarketcap, "market_cap", "Top 10 Coins by Market Cap"),
    (topcoinsmarketrank, "market_cap_rank", "Top 10 Coins by Market Cap Rank"),
    (topcoinsprice, "current_price", "Top 10 Cheapest Coins"),
    (leastcoinsprice, "current_price", "Top 10 Most Expensive Coins"),
    (loser, "price_change_24h", "Top 10 Losers"),
    (gainer, "price_change_24h", "Top 10 Gainers"),
    (mostvolatile, "volatility", "Top 10 Most Volatile Coins"),
    (rarity, "circulating_supply", "Top 10 Rarest Coins"),
    (topvolume, "total_volume", "Top 10 Coins by Volume"),
]

'''for data, column, title in visuals:
    data.set_index("id")[column].plot(kind="bar")
    plt.xlabel("Coin ID")
    plt.ylabel(column.replace("_", " ").title())
    plt.title(title)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()'''

# ============ HISTORY DATA ============
full_history_df = build_full_history_dataset()
most_grow=most_growers()
mover=top_movers()
df["price_per_supply"]=df['market_cap']/df["circulating_supply"]

top_investments=top_invest_long()

top_short_term= top_invest_short()


df = df.copy()

df['target'] = (df['price_change_24h'] > 0).astype(int)

# Features to use
features = ['market_cap', 'total_volume', 'current_price', 'circulating_supply', 'volatility']
X = df[features]
y = df['target']

X = X.fillna(0)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {acc * 100:.2f}%")

joblib.dump(model, "crypto_predictor.pkl")
joblib.dump(features, "crypto_features.pkl")