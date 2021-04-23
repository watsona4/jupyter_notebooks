import pandas as pd
import getpass

from datetime import datetime

import robin_stocks.robinhood as r


username = "watsona4@gmail.com"
password = getpass.getpass()

login = r.login(username, password)

df = pd.DataFrame(
    r.get_crypto_historicals("BTC", interval="5minute", span="day")
)
df["begins_at"] = pd.to_datetime(df["begins_at"]).dt.tz_localize(None)

df.to_csv(f"btc_5min_{datetime.now():%Y-%m-%d_%H%M%S}.csv")

df = pd.DataFrame(
    r.get_crypto_historicals("BTC", interval="15second", span="day")
)
df["begins_at"] = pd.to_datetime(df["begins_at"]).dt.tz_localize(None)

df.to_csv(f"btc_15sec_{datetime.now():%Y-%m-%d_%H%M%S}.csv")
