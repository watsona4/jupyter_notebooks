import getpass
import locale
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
    while True:
        try:
            profile = r.load_account_profile()
            return float(profile["portfolio_cash"])
        except Exception as e:
            print(e)


def get_next_price():
    sleep(5)
    while True:
        try:
            quote = r.get_crypto_quote("BTC")
            break
        except Exception as e:
            print(e)
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


def cur(x):
    return locale.currency(x, grouping=True)


def main():

    login()

    with open("mlpclassifier_5sec.pkl", "rb") as pklfile:
        gsc = pickle.load(pklfile)

    if os.path.exists("state.pkl"):
        with open("state.pkl", "rb") as statefile:
            last_price, last_action = pickle.load(statefile)
    else:
        last_price = last_action = None
    prev_last_price = prev_last_action = None

    p0 = pp1 = pp2 = pp3 = None

    while True:

        p1 = p0
        while True:
            try:
                p0 = get_next_price()
                break
            except:
                pass

        if p1 is None:
            logger.info(
                "    current price=%s, trend=%s/min, last price=%s, last_action=%s",
                cur(p0["mark"]),
                cur(60 * p0["mark"] * pp1 if pp1 is not None else 0),
                cur(last_price or 0),
                last_action or "None",
            )
            continue

        dt = (p0["time"] - p1["time"]).total_seconds()

        pp1old = pp1
        pp1 = (p0["mark"] - p1["mark"]) / (p0["mark"] * dt)

        logger.info(
            "    current price=%s, trend=%s/min, last price=%s, last_action=%s",
            cur(p0["mark"]),
            cur(60 * p0["mark"] * pp1),
            cur(last_price or 0),
            last_action or "None",
        )

        if pp1old is None:
            logger.info("    pp1=%.4g", pp1)
            continue

        pp2old = pp2
        pp2 = (pp1 - pp1old) / dt

        if pp2old is None:
            logger.info("    pp1=%.4g, pp2=%.4g", pp1, pp2)
            continue

        pp3old = pp3
        pp3 = (pp2 - pp2old) / dt

        if pp3old is None:
            logger.info("    pp1=%.4g, pp2=%.4g, pp3=%.4g", pp1, pp2, pp3)
            continue

        pp4 = (pp3 - pp3old) / dt

        logger.info(
            "    pp1=%.4g, pp2=%.4g, pp3=%.4g, pp4=%.4g", pp1, pp2, pp3, pp4
        )

        action = ["BUY", "HOLD", "SELL"][
            gsc.predict([[pp1, pp2, pp3, pp4]])[0]
        ]

        value = get_value()
        holdings = get_holdings()

        if (action == "BUY" and value < 1) or (
            action == "SELL" and holdings < 1e-6
        ):
            action = "HOLD"

        while True:
            try:
                quote = r.get_crypto_quote("BTC")
                break
            except Exception as e:
                print(e)

        price = rh.round_price(float(quote["mark_price"]))
        if last_action is not None and last_price is not None:
            if action == "SELL" and (
                last_action == "BUY" and price < 1.001 * last_price
            ):
                action = "HOLD"
            # if action == "HOLD" and (
            #     last_action == "BUY" and price < 0.97 * last_price
            # ):
            #     action = "SELL"

        logger.info(
            "action=%4s, shares=%.6f, value=%s, total=%s",
            action,
            holdings,
            cur(value),
            cur(holdings * float(quote["mark_price"]) + value),
        )

        while True:
            try:
                for order in r.get_all_open_crypto_orders():
                    r.cancel_crypto_order(order["id"])
                    last_price = prev_last_price
                    last_action = prev_last_action
                break
            except Exception as e:
                print(e)

        if action == "BUY":
            while True:
                order = r.order_buy_crypto_limit_by_price(
                    "BTC",
                    rh.round_price(0.99 * value),
                    rh.round_price(float(quote["ask_price"])),
                )
                # order = r.order_buy_crypto_by_price(
                #     "BTC",
                #     rh.round_price(value),
                # )
                if "non_field_errors" not in order:
                    break
            if "account_id" not in order:
                logger.info(str(order))
            else:
                prev_last_price = last_price
                prev_last_action = last_action
                last_price = price
                last_action = "BUY"
        elif action == "SELL":
            order = r.order_sell_crypto_limit(
                "BTC",
                round(holdings, 6),
                rh.round_price(float(quote["bid_price"])),
            )
            # order = r.order_sell_crypto_by_quantity(
            #     "BTC",
            #     round(holdings, 6),
            # )
            if "account_id" not in order:
                logger.info(str(order))
            else:
                prev_last_price = last_price
                prev_last_action = last_action
                last_price = price
                last_action = "SELL"

        if last_action is not None and last_price is not None:
            with open("state.pkl", "wb") as statefile:
                pickle.dump((last_price, last_action), statefile)


if __name__ == "__main__":
    main()
