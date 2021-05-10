import getpass
import locale
import logging
import os
import pickle
import numpy as np
from datetime import datetime
from random import SystemRandom
from time import sleep

try:
    import pyotp
    import robin_stocks.robinhood as r
    import robin_stocks.robinhood.helper as rh

    r.set_output(open(os.devnull, "w"))
except ImportError:
    pass

locale.setlocale(locale.LC_ALL, "")

LOG_LEVEL = logging.INFO

logger = logging.getLogger()

handler = logging.FileHandler("crypto.log")
formatter = logging.Formatter("%(asctime)s: %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

handler = logging.StreamHandler()
handler.setFormatter(formatter)
logger.addHandler(handler)

logger.setLevel(LOG_LEVEL)


def timefunc(x):
    return datetime(1900, 1, 1) + timedelta(days=x[0])


def get_holdings():
    positions = r.get_crypto_positions()
    for pos in positions:
        if pos["currency"]["code"] == "BTC":
            return float(pos["quantity"])
    return 0.0


def get_value():
    profile = r.load_account_profile()
    return float(profile["portfolio_cash"])


def get_next_price():
    # sleep(5)
    while True:
        try:
            quote = r.get_crypto_quote("BTC")
            break
        except:
            pass
    return {
        "time": datetime.now(),
        "mark": float(quote["mark_price"]),
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

    with open("mlpclassifier_candle.pkl", "rb") as pklfile:
        clf = pickle.load(pklfile)

    if os.path.exists("state.pkl"):
        with open("state.pkl", "rb") as statefile:
            last_price, last_action, last_time = pickle.load(statefile)
    else:
        last_price = last_action = None
        last_time = datetime.now()

    val = val1 = None
    while True:

        prices = [get_next_price()]
        for i in range(12):
            sleep(5)
            prices.append(get_next_price())

        open_p = prices[0]["mark"]
        close_p = prices[11]["mark"]
        dt = (prices[11]["time"] - prices[0]["time"]).total_seconds()

        logger.info(
            "    close price=%s, trend=%s/sec, last price=%s, last_action=%s",
            locale.currency(close_p, grouping=True),
            locale.currency(
                (close_p - open_p)
                / dt,
                grouping=True,
            ),
            locale.currency(last_price or 0, grouping=True),
            last_action or "None",
        )

        if val is None:
            val = close_p / open_p
            continue

        val1 = val
        val = close_p / open_p

        since = (datetime.now() - last_time).total_seconds()

        logger.info(
            "    %.6f  %.6f  %d",
            val,
            val1,
            since,
        )

        ["BUY", "HOLD", "SELL"][clf.predict([[val, val1, since_bs]])[0]]

        value = get_value()
        holdings = get_holdings()

        if (action == "BUY" and value < 1) or (
            action == "SELL" and holdings < 1e-6
        ):
            action = "HOLD"

        quote = r.get_crypto_quote("BTC")

        price = rh.round_price(float(quote["mark_price"]))
        if last_action is not None and last_price is not None:
            if action == "SELL" and (
                last_action == "BUY" and price < 1.001 * last_price
            ):
                action = "HOLD"
            # if action == "HOLD" and (
            #     last_action == "BUY" and price < 0.99 * last_price
            # ):
            #     action = "SELL"

        logger.info(
            "action=%4s, shares=%.6f, value=%s, total=%s",
            action,
            holdings,
            locale.currency(value, grouping=True),
            locale.currency(
                holdings * float(quote["mark_price"]) + value, grouping=True
            ),
        )

        if action == "BUY":
            while True:
                order = r.order_buy_crypto_by_price(
                    "BTC", rh.round_price(value),
                )
                if "non_field_errors" not in order:
                    break
            if "account_id" not in order:
                logger.info(str(order))
            else:
                last_price = price
                last_action = "BUY"
                last_time = datetime.now()
        elif action == "SELL":
            order = r.order_sell_crypto_by_quantity("BTC", round(holdings, 6),)
            if "account_id" not in order:
                logger.info(str(order))
            else:
                last_price = price
                last_action = "SELL"
                last_time = datetime.now()

        if last_action is not None and last_price is not None:
            with open("state.pkl", "wb") as statefile:
                pickle.dump((last_price, last_action, last_time), statefile)


if __name__ == "__main__":
    main()
