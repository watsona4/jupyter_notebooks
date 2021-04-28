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
        res += (
            (df.iloc[i][name] + df.iloc[i - 1][name])
            * (df.iloc[i]["time"] - df.iloc[i - 1]["time"])
            / 2
        )
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


with open("mlpclassifier.pkl", "rb") as pklfile:
    clf = pickle.load(pklfile)

with open("test_data_20210426-054749.csv") as csvfile:

    csvdf = pd.read_csv(csvfile, index_col=0)

    csvdf["time"] = csvdf.apply(timefunc, axis=1)
    csvdf["mark"] = pd.to_numeric(csvdf["mark"])
    csvdf["ask"] = pd.to_numeric(csvdf["ask"])
    csvdf["bid"] = pd.to_numeric(csvdf["bid"])

    df = pd.DataFrame(columns=["time", "mark", "ask", "bid"])
    for i in range(0, csvdf.shape[0], 5):
        take = csvdf.iloc[i : i + 5]
        if take.shape[0] < 5:
            continue
        df = df.append(
            {
                "time": take.iloc[4]["time"],
                "mark": mean(take, "mark"),
                "ask": mean(take, "ask"),
                "bid": mean(take, "bid"),
            },
            ignore_index=True,
        )

x = []
y = []
z = []

i = 0
p = p0 = p1 = p2 = None

while True:

    value = get_value()
    holdings = get_holdings()

    p2 = p1
    p1 = p0
    try:
        p = df.iloc[i]
        p0 = p["mark"]
    except IndexError:
        break
    if i == 0:
        pinit = p0
    i += 1
    logger.debug(
        "time=%s, mark=%.6f, ask=%.6f, bid=%.6f",
        p["time"],
        p["mark"],
        p["ask"],
        p["bid"],
    )

    if p1 is None or p2 is None:
        continue

    p3 = 3 * p0 - 3 * p1 + p2

    logger.debug("p0=%.6g, p1=%.6g, p2=%.6g, p3=%.6g", p0, p1, p2, p3)

    action = "HOLD"
    if p0 < p1 and p0 < p3:
        action = "BUY"
    elif p0 > p1 and p0 > p3:
        action = "SELL"

    logger.info(
        "action=%4s, shares=%.6f, value=%.2f, total=%.2f",
        action,
        holdings,
        value,
        value + holdings * p0,
    )

    x.append(p["time"])
    y.append(value + holdings * p0)
    z.append(p0 / pinit * 100)

    if action == "BUY":
        if value > 1:
            buy(
                0.9 * value, round_price(p0),
            )
    elif action == "SELL":
        if holdings > 1e-6:
            sell(
                0.9 * holdings, round_price(p0),
            )

plt.rc("font", size=12)
fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(x, y, color="tab:orange", label="Value")
ax.plot(x, z, color="tab:blue", label="Stock")

ax.set_xlabel("Time")
ax.set_ylabel("Value")
ax.set_title("Portfolio Value")
ax.grid(True)
ax.legend(loc="upper left")

locator = AutoDateLocator()
ax.xaxis.set_major_locator(locator)
ax.xaxis.set_major_formatter(ConciseDateFormatter(locator))

plt.show()
