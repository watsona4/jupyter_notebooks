import pickle
import getpass
import logging
import os

from datetime import datetime
from time import sleep
from random import SystemRandom

try:
    import pyotp
    import robin_stocks.robinhood as r
    import robin_stocks.robinhood.helper as rh
    from discord_handler import DiscordHandler
except ImportError:
    pass

# r.set_output(open(os.devnull, "w"))

logger = logging.getLogger()

handler = logging.FileHandler("crypto.log")
formatter = logging.Formatter("%(asctime)s: %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

handler = logging.StreamHandler()
handler.setFormatter(formatter)
logger.addHandler(handler)

try:
    webhook_url = "https://discord.com/api/webhooks/835509550881701898/ApE_MdvffnR6BX41L1T4l9iX9TZry4t3Fb6A97oMoWQu2Fin2ZXJaAGwKWGF6UNZthMj"
    handler = DiscordHandler(webhook_url, "crypto")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
except NameError:
    pass

logger.setLevel(logging.INFO)

RAND = SystemRandom()
SLEEP = 10

with open("crypto_actions.pkl", "rb") as pklfile:
    ACTIONS = pickle.load(pklfile)


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
    quote = r.get_crypto_quote("BTC")
    return {
        "time": etime(datetime.now()),
        "mark": float(quote["mark_price"]),
        "ask": float(quote["ask_price"]),
        "bid": float(quote["bid_price"]),
        "vol": float(quote["volume"]),
    }


def get_action(box):
    try:
        dist = ACTIONS[box]
        rand = RAND.random()
        if rand < dist[0]:
            return "BUY"
        if rand < dist[0] + dist[1]:
            return "HOLD"
        return "SELL"
    except KeyError:
        return "HOLD"


def boxpp(diff):
    if diff < -0.2:
        return 0
    if diff < 0.2:
        return 1
    return 2


def login():
    username = "watsona4@gmail.com"
    password = getpass.getpass()

    totp = pyotp.TOTP("C3NG54ZC7JXQGSGR").now()

    r.login(username, password, mfa_code=totp)


def main():

    login()

    mult = 150000 / SLEEP

    p0 = pp1 = pp2 = pp3 = None

    while True:

        sleep(SLEEP)

        for order in r.get_all_open_crypto_orders():
            # logger.info("Cancelling order: %s", str(order))
            r.cancel_crypto_order(order["id"])

        value = get_value()
        holdings = get_holdings()

        p1 = p0
        p0 = get_next_price()
        logger.info(
            "mark=%.6f, ask=%.6f, bid=%.6f", p0["mark"], p0["ask"], p0["bid"]
        )

        if p1 is None:
            continue

        dt = p0["time"] - p1["time"]

        pp1old = pp1
        pp1 = (p0["mark"] - p1["mark"]) / (p0["mark"] * dt)

        if pp1old is None:
            logger.info("pp1=%.4f", pp1)
            continue

        pp2old = pp2
        pp2 = (pp1 - pp1old) / (mult * dt)

        if pp2old is None:
            logger.info("pp1=%.4f, pp2=%.4f", pp1, pp2)
            continue

        pp3old = pp3
        pp3 = (pp2 - pp2old) / (mult * dt)

        if pp3old is None:
            logger.info("pp1=%.4f, pp2=%.4f, pp3=%.4f", pp1, pp2, pp3)
            continue

        pp4 = (pp3 - pp3old) / (mult * dt)

        logger.info(
            "pp1=%.4f, pp2=%.4f, pp3=%.4f, pp4=%.4f", pp1, pp2, pp3, pp4
        )

        box = (
            27 * boxpp(pp1) + 9 * boxpp(pp2) + 3 * boxpp(pp3) + boxpp(pp4) + 1
        )

        action = get_action(box)

        logger.info(
            "action=%4s, shares=%.6f, value=%.2f", action, holdings, value
        )

        if action == "BUY":
            if value > 1:
                #                while True:
                order = r.order_buy_crypto_limit_by_price(
                    "BTC",
                    rh.round_price(0.95 * value),
                    rh.round_price(p0["mark"]),
                )
                if "account_id" not in order:
                    logger.info(str(order))
        elif action == "SELL":
            if holdings > 1e-6:
                order = r.order_sell_crypto_limit(
                    "BTC",
                    round(0.95 * holdings, 6),
                    rh.round_price(p0["mark"]),
                )
                if "account_id" not in order:
                    logger.info(str(order))


if __name__ == "__main__":
    main()
