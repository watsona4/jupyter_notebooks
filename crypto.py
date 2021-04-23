import pickle
import getpass
import logging

from datetime import datetime
from time import sleep
from random import SystemRandom

import robin_stocks.robinhood as r

logging.basicConfig(format="%(asctime)s: %(message)s", level=logging.INFO)


RAND = SystemRandom()
SLEEP = 300

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
    sleep(SLEEP)
    quote = r.get_crypto_quote("BTC")
    return {
        "time": etime(datetime.now()),
        "mark": float(quote["mark_price"]),
        "ask": float(quote["ask_price"]),
        "bid": float(quote["bid_price"]),
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
    except IndexError:
        return "HOLD"


def boxpp(diff):
    if diff < -0.2:
        return 0
    if diff < 0.2:
        return 1
    return 2


def main():

    username = "watsona4@gmail.com"
    password = getpass.getpass()

    r.login(username, password)

    mult = 150000 / SLEEP

    p0 = pp1 = pp2 = pp3 = None

    while True:

        value = get_value()
        holdings = get_holdings()

        p1 = p0
        p0 = get_next_price()
        logging.info(
            "mark=%.6f, ask=%.6f, bid=%.6f", p0["mark"], p0["ask"], p0["bid"]
        )

        if p1 is None:
            continue

        dt = p0["time"] - p1["time"]

        pp1old = pp1
        pp1 = (p0["mark"] - p1["mark"]) / (p0["mark"] * dt)

        if pp1old is None:
            logging.info("pp1=%.4f", pp1)
            continue

        pp2old = pp2
        pp2 = (pp1 - pp1old) / (mult * dt)

        if pp2old is None:
            logging.info("pp1=%.4f, pp2=%.4f", pp1, pp2)
            continue

        pp3old = pp3
        pp3 = (pp2 - pp2old) / (mult * dt)

        if pp3old is None:
            logging.info("pp1=%.4f, pp2=%.4f, pp3=%.4f", pp1, pp2, pp3)
            continue

        pp4 = (pp3 - pp3old) / (mult * dt)

        logging.info(
            "pp1=%.4f, pp2=%.4f, pp3=%.4f, pp4=%.4f", pp1, pp2, pp3, pp4
        )

        box = (
            27 * boxpp(pp1) + 9 * boxpp(pp2) + 3 * boxpp(pp3) + boxpp(pp4) + 1
        )

        action = get_action(box)

        logging.info(
            "action=%4s, shares=%.6f, value=%.2f", action, holdings, value
        )

        if action == "BUY":
            for _ in enumerate(10):
                order = r.order_buy_crypto_by_price("BTC", value)
                if "account_id" in order:
                    break
        elif action == "SELL":
            if get_holdings() > 1e-6:
                order = r.order_sell_crypto_by_quantity("BTC", holdings)
                if "account_id" not in order:
                    logging.info(str(order))


if __name__ == "__main__":
    main()
