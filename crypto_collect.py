import pandas as pd

from crypto import *

login()

df = pd.DataFrame(columns=["time", "mark", "ask", "bid", "vol"])

while True:

    try:
        quote = get_next_price()
        df = df.append(quote, ignore_index=True)
        sleep(5)
    except KeyboardInterrupt:
        df.to_csv(f"btc_data_5sec_{datetime.now():%Y%m%d-%H%M%S}.csv")
        break
    except:
        continue
