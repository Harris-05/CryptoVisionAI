import pandas as pd
import requests as r
import pyodbc
import schedule
import time
import warnings
warnings.filterwarnings('ignore')
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


def update_sql_job():
    try:
        df = importdata()
        addupdatesql(df)
        print(f"SQL updated successfully at {pd.Timestamp.now()}")
    except Exception as e:
        print(f"Error during SQL update: {e}")

# Schedule job every 15 minutes
schedule.every(1).minutes.do(update_sql_job)

print("Scheduler started... (updates every 1 minutes)")
while True:
    schedule.run_pending()
    time.sleep(1)


