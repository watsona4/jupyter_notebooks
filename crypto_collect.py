import subprocess

import pandas as pd

from crypto import *

def save(df):
    filename = f"btc_data_5sec_{datetime.now():%Y%m%d-%H%M%S}.csv"
    df.to_csv(filename)
    subprocess.run(["git", "add", filename])
    subprocess.run(["git", "commit", "-m", f"added {filename}"])
    subprocess.run(["git", "push"])


login()

df = pd.DataFrame(columns=["time", "mark", "ask", "bid", "vol"])

while True:

    try:
        for i in range(4260):
            q = r.get_crypto_quote("BTC")
            quote = {
                "time": etime(datetime.now()),
                "mark": float(q["mark_price"]),
                "ask": float(q["ask_price"]),
                "bid": float(q["bid_price"]),
                "vol": float(q["volume"]),
            }
            print(quote)
            df = df.append(quote, ignore_index=True)
            sleep(5)
        save(df)
        df = pd.DataFrame(columns=["time", "mark", "ask", "bid", "vol"])
    except KeyboardInterrupt:
        save(df)
        break
    except Exception as e:
        print(e)
        continue
