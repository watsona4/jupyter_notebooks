import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import numpy as np
from scipy import optimize
import argparse
import glob
import logging
import sys
import pandas as pd
from datetime import datetime, timedelta

import crypto_bb
from crypto_bb import timefunc

VALUE = 100
HOLDINGS = 0

i = 0


def get_holdings():
    return HOLDINGS


crypto_bb.get_holdings = get_holdings


def get_value():
    return VALUE


crypto_bb.get_value = get_value


def get_next_price():
    global i
    quote = df.iloc[i]
    i += 1
    return {
        "time": quote["time"],
        "mark": quote["mark"],
        "ask": quote["ask"],
        "bid": quote["bid"],
        "vol": 0,
    }


crypto_bb.get_next_price = get_next_price


def cancel_orders():
    pass


crypto_bb.cancel_orders = cancel_orders


def buy_limit(amount, ask):
    global VALUE, HOLDINGS
    VALUE -= amount
    HOLDINGS += amount / ask
    return {"account_id": None}


crypto_bb.buy_limit = buy_limit


def sell_limit(quantity, bid):
    global VALUE, HOLDINGS
    VALUE += quantity * bid
    HOLDINGS -= quantity
    return {"account_id": None}


crypto_bb.sell_limit = sell_limit


def run(x, period, bb_low, bb_high, lo_zone, hi_zone, lo_sigma, hi_sigma):

    global HOLDINGS, VALUE, i

    VALUE = 100
    HOLDINGS = 0
    i = 0

    crypto_bb.logger.setLevel(logging.ERROR)

    x = list(x)
    args = {}
    for arg in [
        "period",
        "bb_low",
        "bb_high",
        "lo_zone",
        "hi_zone",
        "lo_sigma",
        "hi_sigma",
    ]:
        if locals()[arg] is None:
            args[arg] = x.pop(0)
        else:
            args[arg] = locals()[arg]

    def formatdate():
        time = datetime.now()
        ms = int(time.microsecond / 1000)
        return f"{time:%Y-%m-%d %H:%M:%S},{ms:03d}"

    print(
        f"{formatdate()}: period={args['period']:.2f}, bb_low={args['bb_low']:.2f}, bb_high={args['bb_high']:.2f}, lo_zone={args['lo_zone']:.4f}, hi_zone={args['hi_zone']:.4f}, lo_sigma={args['lo_sigma']:.2f}, hi_sigma={args['hi_sigma']:.2f}"
    )

    try:
        crypto_bb.main(
            period=int(args["period"]),
            bb_low=args["bb_low"],
            bb_high=args["bb_high"],
            lo_zone=args["lo_zone"],
            hi_zone=args["hi_zone"],
            lo_sigma=args["lo_sigma"],
            hi_sigma=args["hi_sigma"],
            no_login=True,
            no_sleep=True,
            save_last=False,
            protect_loss=True,
        )
    except IndexError:
        pass

    if HOLDINGS > 0:
        sell_limit(HOLDINGS, df.iloc[df.shape[0] - 1]["bid"])

    dt = (
        df.iloc[df.shape[0] - 1]["time"] - df.iloc[0]["time"]
    ).total_seconds()

    result = (VALUE - 100) / dt * 3600 * 24

    ret = np.exp(-result)

    print(
        f"{formatdate()}:     Final value = {VALUE:.2f}, Return = {VALUE-100:.2f}% = {result:.2f}%/day, Score = {ret}"
    )

    return ret


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("glob")
    parser.add_argument("--period", type=int)
    parser.add_argument("--bb-low", type=float)
    parser.add_argument("--bb-high", type=float)
    parser.add_argument("--lo-zone", type=float)
    parser.add_argument("--hi-zone", type=float)
    parser.add_argument("--lo-sigma", type=float)
    parser.add_argument("--hi-sigma", type=float)
    parser.add_argument("--method", default="dual_annealing")

    args = parser.parse_args()

    df = pd.DataFrame(columns=["time", "mark", "ask", "bid"])

    for csvfile in glob.glob(args.glob):

        csvdf = pd.read_csv(csvfile, index_col=0)

        csvdf["time"] = csvdf.apply(timefunc, axis=1)
        csvdf["mark"] = pd.to_numeric(csvdf["mark"])
        csvdf["ask"] = pd.to_numeric(csvdf["ask"])
        csvdf["bid"] = pd.to_numeric(csvdf["bid"])

        if df.shape[0] > 0:
            prev_time = df.iloc[df.shape[0] - 1]["time"]
            prev_mark = df.iloc[df.shape[0] - 1]["mark"]

            dt = csvdf.iloc[0]["time"] - prev_time
            scale = csvdf.iloc[0]["mark"] - prev_mark

            csvdf["time"] = csvdf["time"] - dt
            csvdf["mark"] = csvdf["mark"] - scale
            csvdf["ask"] = csvdf["ask"] - scale
            csvdf["bid"] = csvdf["bid"] - scale

        df = df.append(csvdf, ignore_index=True)

    bounds_dict = {
        "period": (12, 2000),
        "bb_low": (0.25, 4),
        "bb_high": (0.25, 4),
        "lo_zone": (-0.1, 0.5),
        "hi_zone": (0.5, 1.1),
        "lo_sigma": (0, 4),
        "hi_sigma": (0, 4),
    }

    fixed = []
    bounds = []
    for arg in [
        "period",
        "bb_low",
        "bb_high",
        "lo_zone",
        "hi_zone",
        "lo_sigma",
        "hi_sigma",
    ]:
        if getattr(args, arg) is not None:
            fixed.append(getattr(args, arg))
        else:
            fixed.append(None)
            bounds.append(bounds_dict[arg])

    if len(bounds) == 0:
        run([], *fixed)

    elif args.method == "brute":

        x0, fval, grid, Jout = optimize.brute(
            func=run,
            Ns=20,
            args=tuple(fixed),
            ranges=bounds,
            full_output=True,
            # finish=None,
        )

        if len(grid.shape) == 2:
            fig = plt.figure(figsize=(10,6))
            ax1 = fig.add_subplot(111, projection='3d')

            mycmap = plt.get_cmap('gist_earth')
            surf1 = ax1.plot_surface(grid[0,:], grid[1,:], -np.log(Jout), cmap=mycmap)
            fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=5)

            plt.title(args.glob)
            plt.show()

        elif len(grid.shape) == 1:
            plt.plot(grid, -np.log(Jout))
            plt.title(args.glob)
            plt.show()


    else:
        # res = getattr(optimize, args.method)(
        #     func=run,
        #     args=tuple(fixed),
        #     bounds=bounds,
        #     local_search_options={"options": {"disp": True}},
        # )
        res = optimize.shgo(
            func=run,
            args=tuple(fixed),
            bounds=bounds,
            constraints=[
                {"type": "eq",
                 "fun": lambda x: x[0] - int(x[0])},
                {"type": "ineq",
                 "fun": lambda x: x[1]},
                {"type": "ineq",
                 "fun": lambda x: x[2]},
                {"type": "ineq",
                 "fun": lambda x: x[5]},
                {"type": "ineq",
                 "fun": lambda x: x[6]},
            ]
        )

        print(res)

    print(f"Glob = {args.glob}")
    print(f"Default = {df.iloc[df.shape[0] - 1]['mark']/df.iloc[0]['mark']}")
