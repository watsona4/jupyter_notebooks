import argparse
import glob
import logging
import sys
from datetime import datetime, timedelta
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from hyperopt import fmin, hp, space_eval, tpe
from mpl_toolkits.mplot3d import axes3d
from prettytable import PrettyTable
from scipy import optimize
from scipy.optimize import Bounds, NonlinearConstraint

import crypto_bb
from crypto_bb import timefunc

VALUE = 100
HOLDINGS = 0

ITER = None

I = 0
PTABLE = None
OPTIONS = None


def get_holdings():
    return HOLDINGS


crypto_bb.get_holdings = get_holdings


def get_value():
    return VALUE


crypto_bb.get_value = get_value


def get_next_price():
    try:
        quote = next(ITER)
    except StopIteration:
        raise IndexError
    return {
        "time": quote.time,
        "mark": quote.mark,
        "ask": quote.ask,
        "bid": quote.bid,
        "vol": quote.vol,
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


def run(
    x,
    period=None,
    bb_low=None,
    bb_high=None,
    lo_zone=None,
    hi_zone=None,
    lo_sigma=None,
    hi_sigma=None,
    protect_loss=None,
):

    global HOLDINGS, VALUE, ITER, I

    VALUE = 100
    HOLDINGS = 0
    ITER = DF.itertuples()

    crypto_bb.logger.setLevel(logging.ERROR)

    try:
        x = list(x)
    except TypeError:
        x = [x]

    args = {}
    for arg in [
        "period",
        "bb_low",
        "bb_high",
        "lo_zone",
        "hi_zone",
        "lo_sigma",
        "hi_sigma",
        "protect_loss",
    ]:
        if locals()[arg] is None:
            args[arg] = x.pop(0)
        else:
            args[arg] = locals()[arg]

    protect_loss = args["protect_loss"]
    if isinstance(protect_loss, (float, int)):
        protect_loss = bool(int(protect_loss + 0.5))

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
            protect_loss=protect_loss,
        )
    except IndexError:
        pass

    if HOLDINGS > 0:
        sell_limit(HOLDINGS, DF.iloc[DF.shape[0] - 1]["bid"])

    dt = (
        DF.iloc[DF.shape[0] - 1]["time"] - DF.iloc[0]["time"]
    ).total_seconds()

    result = (VALUE - 100) / dt * 3600 * 24

    ret = np.exp(-result)

    row = [
        I,
        f"{datetime.now():%X}",
        int(args["period"]),
        args["bb_low"],
        args["bb_high"],
        args["lo_zone"],
        args["hi_zone"],
        args["lo_sigma"],
        args["hi_sigma"],
        protect_loss,
        result,
    ]

    PTABLE.add_row(row)
    frows = PTABLE._format_rows(PTABLE._get_rows(OPTIONS), OPTIONS)
    PTABLE._compute_widths(frows, OPTIONS)

    print(PTABLE._stringify_row(frows[-1], OPTIONS))

    I += 1

    return ret


def main():

    global DF, PTABLE, OPTIONS

    parser = argparse.ArgumentParser()

    parser.add_argument("glob")
    parser.add_argument("--period", type=int)
    parser.add_argument("--bb-low", type=float)
    parser.add_argument("--bb-high", type=float)
    parser.add_argument("--lo-zone", type=float)
    parser.add_argument("--hi-zone", type=float)
    parser.add_argument("--lo-sigma", type=float)
    parser.add_argument("--hi-sigma", type=float)
    parser.add_argument("--protect-loss", type=bool)
    parser.add_argument("--method", default="dual_annealing")
    parser.add_argument("--finish", default=None)

    args = parser.parse_args()

    DF = pd.DataFrame(columns=["time", "mark", "ask", "bid"])

    for csvfile in glob.glob(args.glob):

        csvdf = pd.read_csv(csvfile, index_col=0)

        csvdf["time"] = csvdf.apply(timefunc, axis=1)
        csvdf["mark"] = pd.to_numeric(csvdf["mark"])
        csvdf["ask"] = pd.to_numeric(csvdf["ask"])
        csvdf["bid"] = pd.to_numeric(csvdf["bid"])

        if DF.shape[0] > 0:
            prev_time = DF.iloc[DF.shape[0] - 1]["time"]
            prev_mark = DF.iloc[DF.shape[0] - 1]["mark"]

            dt = csvdf.iloc[0]["time"] - prev_time
            scale = csvdf.iloc[0]["mark"] - prev_mark

            csvdf["time"] = csvdf["time"] - dt
            csvdf["mark"] = csvdf["mark"] - scale
            csvdf["ask"] = csvdf["ask"] - scale
            csvdf["bid"] = csvdf["bid"] - scale

        DF = DF.append(csvdf, ignore_index=True)

    bounds_dict = {
        "period": (12, 48 * 3600 / 5),
        "bb_low": (0.25, 4),
        "bb_high": (0.25, 4),
        "lo_zone": (-0.1, 0.5),
        "hi_zone": (0.5, 1.1),
        "lo_sigma": (0, 4),
        "hi_sigma": (0, 4),
        "protect_loss": (0, 1),
    }

    abs_dict = {
        "period": 1,
        "bb_low": 0.1,
        "bb_high": 0.1,
        "lo_zone": 0.01,
        "hi_zone": 0.01,
        "lo_sigma": 0.1,
        "hi_sigma": 0.1,
        "protect_loss": 1,
    }

    PTABLE = PrettyTable(
        [
            "Iteration",
            "Time",
            "Period",
            "BB Low",
            "BB High",
            "Low Zone",
            "High Zone",
            "Low Sigma",
            "High Sigma",
            "Protect",
            "Return",
        ]
    )
    PTABLE.float_format = ".4"

    bounds = []
    bounds.append((0, 100))
    bounds.append(
        (
            f"{datetime(2021, 1, 1, 0, 0, 0):%X}",
            f"{datetime(2021, 1, 1, 23, 59, 59):%X}",
        )
    )
    bounds.append([int(v) for v in bounds_dict["period"]])
    bounds.append([float(v) for v in bounds_dict["bb_low"]])
    bounds.append([float(v) for v in bounds_dict["bb_high"]])
    bounds.append([float(v) for v in bounds_dict["lo_zone"]])
    bounds.append([float(v) for v in bounds_dict["hi_zone"]])
    bounds.append([float(v) for v in bounds_dict["lo_sigma"]])
    bounds.append([float(v) for v in bounds_dict["hi_sigma"]])
    bounds.append((False, True))
    bounds.append((-99.0, 99.0))
    for i in product([0, 1], repeat=len(bounds)):
        PTABLE.add_row([bounds[j][i[j]] for j in range(len(bounds))])
    OPTIONS = PTABLE._get_options({})
    frows = PTABLE._format_rows(PTABLE._get_rows(OPTIONS), OPTIONS)
    PTABLE._compute_widths(frows, OPTIONS)
    PTABLE._hrule = PTABLE._stringify_hrule(OPTIONS)
    print(PTABLE._stringify_header(OPTIONS))

    fixed = []
    bounds = []
    abs_diff = []
    for arg in [
        "period",
        "bb_low",
        "bb_high",
        "lo_zone",
        "hi_zone",
        "lo_sigma",
        "hi_sigma",
        "protect_loss",
    ]:
        if getattr(args, arg) is not None:
            fixed.append(getattr(args, arg))
        else:
            fixed.append(None)
            bounds.append(bounds_dict[arg])
            abs_diff.append(abs_dict[arg])

    res = None
    if args.method == "brute":

        x0, fval, grid, Jout = optimize.brute(
            func=run,
            args=tuple(fixed),
            ranges=bounds,
            full_output=True,
            finish=args.finish,
        )

        if grid.ndim == 1:
            plt.plot(grid, -np.log(Jout))
            plt.title(args.glob)
            plt.show()

        elif grid.ndim == 3:
            fig = plt.figure(figsize=(10, 6))
            ax1 = fig.add_subplot(111, projection="3d")

            mycmap = plt.get_cmap("gist_earth")
            surf1 = ax1.plot_surface(
                grid[0, :], grid[1, :], -np.log(Jout), cmap=mycmap
            )
            fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=5)

            plt.title(args.glob)
            plt.show()

    elif args.method == "basinhopping":
        res = optimize.basinhopping(
            func=run,
            x0=tuple(fixed),
            minimizer_kwargs={"args": tuple(7 * [None])},
        )

    elif args.method == "shgo-sobol":

        constraints = []
        if args.period is None:
            constraints.append(
                {"type": "eq", "fun": lambda x: np.array([x[0] - int(x[0])])}
            )

        if args.protect_loss is None:
            constraints.append(
                {"type": "eq", "fun": lambda x: np.array([x[7] - int(x[7])])}
            )

        res = optimize.shgo(
            func=run,
            args=tuple(fixed),
            bounds=bounds,
            constraints=constraints,
            options={"disp": True},
            sampling_method="sobol",
            minimizer_kwargs={"options": {"eps": np.array(abs_diff)}},
        )

        tbl = PrettyTable(
            [
                "Period",
                "BB Low",
                "BB High",
                "Low Zone",
                "High Zone",
                "Low Sigma",
                "High Sigma",
                "Protect",
                "Return",
            ]
        )
        tbl.float_format = ".4"

        for minim in res.xl:
            row = []
            i = 0
            for val in fixed:
                if val is None:
                    row.append(minim[i])
                    i += 1
                else:
                    row.append(val)
            score = run(minim, *fixed)
            row.append(-np.log(score))
            tbl.add_row(row)

        print(PTABLE._hrule)
        print()

        print(tbl)

    elif args.method == "hyperopt":

        space = [
            hp.quniform("period", 12, 48 * 3600 / 5, 1),
            hp.uniform("bb_low", 0.25, 4),
            hp.uniform("bb_high", 0.25, 4),
            hp.uniform("lo_zone", -0.1, 0.5),
            hp.uniform("hi_zone", 0.5, 1.1),
            hp.uniform("lo_sigma", 0, 4),
            hp.uniform("hi_sigma", 0, 4),
            hp.quniform("protect_loss", 0, 1, 1),
        ]

        res = fmin(run, space, algo=tpe.suggest, max_evals=200)

        print(run(space_eval(space, res)))

    elif len(bounds) == 0:
        run([], *fixed)

    elif len(bounds) == 1:

        x0 = [(bounds[0][0] + bounds[0][1]) / 2]
        constraints = ()
        options = {"disp": True}
        if args.period is None:
            constraints = [
                {"type": "eq", "fun": lambda x: np.array([x[0] - int(x[0])])}
            ]
            options["finite_diff_rel_step"] = (1 / x0[0],)

        res = optimize.minimize(
            fun=run,
            x0=x0,
            method="trust-constr",
            args=tuple(fixed),
            bounds=Bounds(bounds[0][0], bounds[0][1]),
            constraints=constraints,
            options=options,
        )

    else:
        res = getattr(optimize, args.method)(
            func=run,
            args=tuple(fixed),
            bounds=bounds,
            maxiter=1000000,
            local_search_options={"options": {"disp": True}},
        )

    if res is not None:
        print(res)

    print(f"Glob = {args.glob}")
    print(f"Default = {DF.iloc[DF.shape[0] - 1]['mark']/DF.iloc[0]['mark']}")


if __name__ == "__main__":
    main()
