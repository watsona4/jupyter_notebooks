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


with open("mlpclassifier_5sec.pkl", "rb") as pklfile:
    clf = pickle.load(pklfile)

with open("btc_data_5sec_20210428-123100.csv") as csvfile:
    df = pd.read_csv(csvfile, index_col=0)

df["time"] = df.apply(timefunc, axis=1)
df["mark"] = pd.to_numeric(df["mark"])
df["ask"] = pd.to_numeric(df["ask"])
df["bid"] = pd.to_numeric(df["bid"])

# df = df.reindex(index=df.index[::-1])

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
