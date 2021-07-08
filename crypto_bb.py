import getpass
import locale
import logging
import os
import pickle
from datetime import datetime, timedelta
from random import SystemRandom
from time import sleep

import numpy as np

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


def round_price(price):
    price = float(price)
    if price <= 1e-2:
        returnPrice = round(price, 6)
    elif price < 1e0:
        returnPrice = round(price, 4)
    else:
        returnPrice = round(price, 2)

    return returnPrice


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
    quote = r.get_crypto_quote("BTC")
    return {
        "time": datetime.now(),
        "mark": float(quote["mark_price"]),
        "ask": float(quote["ask_price"]),
        "bid": float(quote["bid_price"]),
        "vol": float(quote["volume"]),
    }


def get_bb(x, p, low, high):
    bb_mean = x.mean()
    bb_std = x.std()
    lower = bb_mean - low * bb_std
    upper = bb_mean + high * bb_std
    if upper == lower:
        return bb_mean, np.nan
    pct_bb = (p - lower) / (upper - lower)
    return bb_mean, pct_bb


def cancel_orders():
    for order in r.get_all_open_crypto_orders():
        cancel = r.cancel_crypto_order(order["id"])


def buy_limit(amount, ask):
    order = r.order_buy_crypto_limit_by_price(
        "BTC", round_price(amount), round_price(ask),
    )
    return order


def sell_limit(quantity, bid):
    order = r.order_sell_crypto_limit(
        "BTC", round(quantity, 6), round_price(bid),
    )
    return order


def login():
    totp = pyotp.TOTP("C3NG54ZC7JXQGSGR").now()
    r.login("watsona4@gmail.com", mfa_code=totp)


def cur(x):
    return locale.currency(x, grouping=True)


def main(
    period=3520,
    bb_low=3.3262,
    bb_high=1.6270,
    lo_zone=-0.07656,
    hi_zone=0.9172,
    lo_sigma=1.6562,
    hi_sigma=1.2812,
    protect_loss=True,
    stop_limit=False,
    no_login=False,
    no_sleep=False,
    save_last=True,
):

    if not no_login:
        login()

    if save_last and os.path.exists("state.pkl"):
        with open("state.pkl", "rb") as statefile:
            last_price, last_action = pickle.load(statefile)
    else:
        last_price = last_action = None
    prev_last_price = prev_last_action = None

    bb_prices = np.empty((0,))
    pct_bbs = np.empty((0,))

    while True:

        try:
            if not no_sleep:
                sleep(5)
            p = get_next_price()["mark"]

            bb_prices = np.append(bb_prices, p)

            if len(bb_prices) > period:
                bb_prices = bb_prices[1:]

            bb_mean, pct_bb = get_bb(bb_prices, p, bb_low, bb_high)
            logger.info(
                "    current price=%s, mean=%s, %%bb=%.2f, last_price=%s, last_action=%s",
                cur(p),
                cur(bb_mean),
                pct_bb,
                cur(last_price or 0),
                last_action or "None",
            )

            if len(bb_prices) < period:
                continue

            cancel_orders()

            value = get_value()
            holdings = get_holdings()

            if holdings < 1e-6:
                if last_action == "BUY":
                    logger.info("BUY FAILED!")
                    last_price = prev_last_price
                last_action = "SELL"
            else:
                if last_action == "SELL":
                    logger.info("SELL FAILED!")
                    last_price = prev_last_price
                last_action = "BUY"

            if last_action != "BUY" and pct_bb < lo_zone:
                pct_bbs = np.append(pct_bbs, pct_bb)
            elif last_action != "SELL" and pct_bb > hi_zone:
                pct_bbs = np.append(pct_bbs, pct_bb)

            action = "HOLD"
            if last_action != "BUY" and len(pct_bbs) > 0:
                target = pct_bbs.mean() + lo_sigma * pct_bbs.std()
                logger.info("    BUY zone: target=%.2f", target)
                if pct_bb > target:
                    action = "BUY"
                    pct_bbs = np.empty((0,))
            if last_action != "SELL" and len(pct_bbs) > 0:
                target = pct_bbs.mean() - hi_sigma * pct_bbs.std()
                logger.info("    SELL zone: target=%.2f", target)
                if pct_bb < target:
                    action = "SELL"
                    pct_bbs = np.empty((0,))

            if (action == "BUY" and value < 1) or (
                action == "SELL" and holdings < 1e-6
            ):
                action = "HOLD"

            quote = get_next_price()

            price = round_price(float(quote["mark"]))
            ask = float(quote["ask"])
            bid = float(quote["bid"])

            if last_action is not None and last_price is not None:
                if (
                    protect_loss
                    and action == "SELL"
                    and (last_action == "BUY" and bid < last_price)
                ):
                    action = "HOLD"
                if (
                    stop_limit
                    and action == "HOLD"
                    and (last_action == "BUY" and price < 0.99 * last_price)
                ):
                    action = "SELL"

            logger.info(
                "action=%4s, shares=%.6f, value=%s, total=%s",
                action,
                holdings,
                cur(value),
                cur(holdings * float(quote["mark"]) + value),
            )

            if action == "BUY":
                order = buy_limit(0.99 * value, ask)
                if "account_id" not in order:
                    logger.info(str(order))
                else:
                    prev_last_price = last_price
                    prev_last_action = last_action
                    last_price = ask
                    last_action = "BUY"
            elif action == "SELL":
                order = sell_limit(holdings, bid)
                if "account_id" not in order:
                    logger.info(str(order))
                else:
                    prev_last_price = last_price
                    prev_last_action = last_action
                    last_price = price
                    last_action = "SELL"

            if save_last and last_action is not None and last_price is not None:
                with open("state.pkl", "wb") as statefile:
                    pickle.dump((last_price, last_action), statefile)

        except Exception as e:
            print(e)

if __name__ == "__main__":
    main()
