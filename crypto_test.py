import pickle
from datetime import datetime, timedelta

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import ConciseDateFormatter, AutoDateLocator

from crypto import *

def timefunc(x):
    return datetime(1900, 1, 1) + timedelta(days=x[0])


def func(x):
    minimum = np.amin([x["prev"], x["mark"], x["next"]])
    maximum = np.amax([x["prev"], x["mark"], x["next"]])
    if x["mark"] == minimum:
        return 0  # buy
    if x["mark"] == maximum:
        return 2  # sell
    return 1  # hold


def mean(df, name):
    res = 0
    for i in range(1, df.shape[0]):
        res += (df.iloc[i][name] + df.iloc[i-1][name]) * (df.iloc[i]["time"] - df.iloc[i-1]["time"]) / 2
    return res / (df.iloc[4]["time"] - df.iloc[0]["time"])


VALUE = 100
HOLDINGS = 0

def get_value():
    return VALUE

def get_holdings():
    return HOLDINGS

def buy(price, limit):
    global VALUE, HOLDINGS
    VALUE -= price
    HOLDINGS += price / limit

def sell(quantity, limit):
    global VALUE, HOLDINGS
    VALUE += quantity * limit
    HOLDINGS -= quantity

def round_price(price):
    price = float(price)
    if price <= 1e-2:
        returnPrice = round(price, 6)
    elif price < 1e0:
        returnPrice = round(price, 4)
    else:
        returnPrice = round(price, 2)

    return returnPrice


with open('mlpclassifier.pkl', 'rb') as pklfile:
    clf = pickle.load(pklfile)

with open("test_data_20210426-054749.csv") as csvfile:

    csvdf = pd.read_csv(csvfile, index_col=0)

    csvdf["time"] = csvdf.apply(timefunc, axis=1)
    csvdf["mark"] = pd.to_numeric(csvdf["mark"])
    csvdf["ask"] = pd.to_numeric(csvdf["ask"])
    csvdf["bid"] = pd.to_numeric(csvdf["bid"])

    df = pd.DataFrame(columns=["time", "mark", "ask", "bid"])
    for i in range(0, csvdf.shape[0], 5):
        take = csvdf.iloc[i:i+5]
        if take.shape[0] < 5:
            continue
        df = df.append({"time": take.iloc[4]["time"], "mark": mean(take, "mark"), "ask": mean(take, "ask"), "bid": mean(take, "bid")}, ignore_index=True)

x = []
y = []

i = 0
p0 = pp1 = pp2 = None

while True:

    value = get_value()
    holdings = get_holdings()

    p1 = p0
    try:
        p0 = df.iloc[i]
    except IndexError:
        break
    i += 1
    logger.debug(
        "time=%s, mark=%.6f, ask=%.6f, bid=%.6f", p0["time"], p0["mark"], p0["ask"], p0["bid"]
    )

    if p1 is None:
        continue

    dt = (p0["time"] - p1["time"]).total_seconds()

    pp1old = pp1
    pp1 = (p0["mark"] - p1["mark"]) / (p0["mark"] * dt)

    if pp1old is None:
        logger.debug("pp1=%.6g", pp1)
        continue

    pp2 = (pp1 - pp1old) / dt

    logger.debug(
        "pp1=%.6g, pp2=%.6g", pp1, pp2
    )

    action = ["BUY", "HOLD", "SELL"][clf.predict([[pp1, pp2]])[0]]

    logger.info(
        "action=%4s, shares=%.6f, value=%.2f", action, holdings, value
    )

    x.append(p0["time"])
    y.append(value + holdings * p0["mark"])

    if action == "BUY":
        if value > 1:
            buy(
                value,
                round_price(p0["mark"]),
            )
    elif action == "SELL":
        if holdings > 1e-6:
            sell(
                holdings,
                round_price(p0["mark"]),
            )

print(y)

plt.rc('font', size=12)
fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(x, y, color='tab:orange', label='Value')

ax.set_xlabel('Time')
ax.set_ylabel('Value')
ax.set_title('Portfolio Value')
ax.grid(True)
ax.legend(loc='upper left');

locator = AutoDateLocator()
ax.xaxis.set_major_locator(locator)
ax.xaxis.set_major_formatter(ConciseDateFormatter(locator))

plt.show()
