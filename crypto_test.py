import pickle
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.dates import AutoDateLocator, ConciseDateFormatter

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
            * (df.iloc[i]["time"] - df.iloc[i - 1]["time"]).total_seconds()
            / 2
        )
    return res / (df.iloc[4]["time"] - df.iloc[0]["time"]).total_seconds()


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

    csvdf = csvdf.reindex(index=csvdf.index[::-1])

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

last_price = None
last_action = None

i = 0
p0 = pp1 = pp2 = pp3 = None

while True:

    value = get_value()
    holdings = get_holdings()

    p1 = p0
    try:
        p0 = df.iloc[i]
    except IndexError:
        break
    if i == 0:
        pinit = p0["mark"]
    i += 1
    logger.debug(
        "time=%s, mark=%.6f, ask=%.6f, bid=%.6f",
        p0["time"],
        p0["mark"],
        p0["ask"],
        p0["bid"],
    )

    if p1 is None:
        continue

    dt = (p0["time"] - p1["time"]).total_seconds()

    pp1old = pp1
    pp1 = (p0["mark"] - p1["mark"]) / (p0["mark"] * dt)

    if pp1old is None:
        logger.debug("pp1=%.6g", pp1)
        continue

    pp2old = pp2
    pp2 = (pp1 - pp1old) / dt

    if pp2old is None:
        logger.debug("pp1=%.6g, pp2=%.6g", pp1, pp2)
        continue

    pp3 = (pp2 - pp2old) / dt
    logger.debug("pp1=%.6g, pp2=%.6g, pp3=%.6g", pp1, pp2, pp3)

    action = ["BUY", "HOLD", "SELL"][clf.predict([[pp1, pp2, pp3]])[0]]

    x.append(p0["time"])
    y.append(value + holdings * p0["mark"])
    z.append(p0["mark"] / pinit * 100)

    price = round_price(p0["mark"])
    if last_action is not None and last_price is not None:
        if action == "SELL" and (
            last_action == "BUY" and price < 1.001 * last_price
        ):
            action = "HOLD"
        if action == "HOLD" and (
            last_action == "BUY" and price < 0.99 * last_price
        ):
            action = "SELL"

    logger.info(
        "action=%4s, shares=%.6f, value=%.2f, total=%2f",
        action,
        holdings,
        value,
        value + holdings * p0["mark"],
    )

    if action == "BUY":
        if value > 1:
            buy(value, price)
            last_price = price
            last_action = "BUY"
    elif action == "SELL":
        if holdings > 1e-6:
            sell(holdings, price)
            last_price = price
            last_action = "SELL"

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
