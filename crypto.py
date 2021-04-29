import getpass
import logging
import os
import pickle
from datetime import datetime
from random import SystemRandom
from time import sleep

try:
    import pyotp
    import robin_stocks.robinhood as r
    import robin_stocks.robinhood.helper as rh
    from discord_handler import DiscordHandler

    r.set_output(open(os.devnull, "w"))
except ImportError:
    pass

LOG_LEVEL = logging.INFO

logger = logging.getLogger()

handler = logging.FileHandler("crypto.log")
formatter = logging.Formatter("%(asctime)s: %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

handler = logging.StreamHandler()
handler.setFormatter(formatter)
logger.addHandler(handler)

# if LOG_LEVEL >= logging.INFO:
#     try:
#         webhook_url = "https://discord.com/api/webhooks/835509550881701898/ApE_MdvffnR6BX41L1T4l9iX9TZry4t3Fb6A97oMoWQu2Fin2ZXJaAGwKWGF6UNZthMj"
#         handler = DiscordHandler(webhook_url, "crypto")
#         handler.setFormatter(formatter)
#         logger.addHandler(handler)
#     except NameError:
#         pass

logger.setLevel(LOG_LEVEL)

with open("mlpclassifier.pkl", "rb") as pklfile:
    clf = pickle.load(pklfile)


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


def get_holdings():
    positions = r.get_crypto_positions()
    for pos in positions:
        if pos["currency"]["code"] == "BTC":
            return float(pos["quantity"])
    return 0.0


def get_value():
    profile = r.load_account_profile()
    return float(profile["portfolio_cash"])


def etime(timestamp):
    return (timestamp - datetime(1900, 1, 1)).total_seconds() / 3600 / 24


def get_next_price():
    t0 = t = t1 = datetime.now()
    v = v1 = float(r.get_crypto_quote("BTC")["mark_price"])
    val = 0
    for i in range(1, 12):
        sleep(5)
        quote = r.get_crypto_quote("BTC")
        t1, t = t, datetime.now()
        v1, v = v, float(quote["mark_price"])
        val += (v + v1) * (t - t1).total_seconds() / 2
    val = val / (t - t0).total_seconds()
    return {
        "time": t,
        "mark": val,
        "ask": float(quote["ask_price"]),
        "bid": float(quote["bid_price"]),
        "vol": float(quote["volume"]),
    }


def login():
    username = "watsona4@gmail.com"
    password = getpass.getpass()

    totp = pyotp.TOTP("C3NG54ZC7JXQGSGR").now()

    r.login(username, password, mfa_code=totp)


def main():

    login()

    prev_last_price = last_price = None
    prev_last_action = last_action = None

    p0 = pp1 = pp2 = pp3 = None

    while True:

        p1 = p0
        while True:
            try:
                p0 = get_next_price()
                break
            except:
                pass

        logger.info(
            "mark=%.6f, ask=%.6f, bid=%.6f", p0["mark"], p0["ask"], p0["bid"]
        )

        if p1 is None:
            continue

        dt = (p0["time"] - p1["time"]).total_seconds()

        pp1old = pp1
        pp1 = (p0["mark"] - p1["mark"]) / (p0["mark"] * dt)

        if pp1old is None:
            logger.info("pp1=%.6g", pp1)
            continue

        pp2old = pp2
        pp2 = (pp1 - pp1old) / dt

        if pp2old is None:
            logger.info("pp1=%.6g, pp2=%.6g", pp1, pp2)
            continue

        pp3 = (pp2 - pp2old) / dt
        logger.info("pp1=%.6g, pp2=%.6g, pp3=%.6g", pp1, pp2, pp3)

        action = ["BUY", "HOLD", "SELL"][clf.predict([[pp1, pp2, pp3]])[0]]

        for order in r.get_all_open_crypto_orders():
            logger.info("Cancelling order: %s", str(order))
            r.cancel_crypto_order(order["id"])
            last_price = prev_last_price
            last_action = prev_last_action

        value = get_value()
        holdings = get_holdings()

        quote = r.get_crypto_quote("BTC")

        price = rh.round_price(float(quote["mark_price"]))
        if last_action is not None and last_price is not None:
            if (
                action == "BUY"
                and last_action == "SELL"
                and price > last_price
            ) or (
                action == "SELL"
                and last_action == "BUY"
                and price < last_price
            ):
                action = "HOLD"

        logger.info(
            "action=%4s, shares=%.6f, value=%.2f, total=%.2f",
            action,
            holdings,
            value,
            holdings * float(quote["mark_price"]) + value,
        )

        if action == "BUY":
            if value > 1:
                order = r.order_buy_crypto_limit_by_price(
                    "BTC", rh.round_price(0.99 * value), price
                )
                if "account_id" not in order:
                    logger.info(str(order))
                else:
                    prev_last_price = last_price
                    last_price = price
                    prev_last_action = last_action
                    last_action = "BUY"
        elif action == "SELL":
            if holdings > 1e-6:
                order = r.order_sell_crypto_limit(
                    "BTC", round(0.99 * holdings, 6), price
                )
                if "account_id" not in order:
                    logger.info(str(order))
                else:
                    prev_last_price = last_price
                    last_price = price
                    prev_last_action = last_action
                    last_action = "SELL"


if __name__ == "__main__":
    main()
