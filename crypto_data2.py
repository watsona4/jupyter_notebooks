import glob
import pickle
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


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
    return res / (df.iloc[11]["time"] - df.iloc[0]["time"]).total_seconds()


names = ["buy", "hold", "sell"]

grid_search = True

x = np.empty((0, 3), dtype=np.float64)
y = np.empty((0,), dtype=np.int64)

for csvfile in glob.glob("btc_data_5sec_*.csv"):

    print(f"Reading {csvfile}...")

    csvdf = pd.read_csv(csvfile, index_col=0)

    if csvdf.shape[0] < 12:
        continue

    csvdf["time"] = csvdf.apply(timefunc, axis=1)
    csvdf["mark"] = pd.to_numeric(csvdf["mark"])
    csvdf["ask"] = pd.to_numeric(csvdf["ask"])
    csvdf["bid"] = pd.to_numeric(csvdf["bid"])

    df = pd.DataFrame(columns=["time", "mark", "ask", "bid"])
    for i in range(0, csvdf.shape[0], 12):
        take = csvdf.iloc[i : i + 12]
        if take.shape[0] < 12:
            continue
        df = df.append(
            {
                "time": take.iloc[11]["time"],
                "mark": mean(take, "mark"),
                "ask": mean(take, "ask"),
                "bid": mean(take, "bid"),
            },
            ignore_index=True,
        )

    df["prev"] = df["mark"].shift(1)
    df["next"] = df["mark"].shift(-1)

    df["action"] = df.apply(func, axis=1)

    df["pp1"] = (
        df["mark"].diff() / df["mark"] / df["time"].diff().dt.total_seconds()
    )
    df["pp2"] = df["pp1"].diff() / df["time"].diff().dt.total_seconds()
    df["pp3"] = df["pp2"].diff() / df["time"].diff().dt.total_seconds()
    # df["pp4"] = df["pp3"].diff() / df["time"].diff().dt.total_seconds()

    x = np.append(x, df[["pp1", "pp2", "pp3"]][3:].to_numpy(), axis=0)
    y = np.append(y, df["action"][3:].to_numpy(), axis=0)

print(x.shape)

n_components = [None, 1, 2, 3, "mle"]
solvers = ["lbfgs"]
activations = ["identity", "logistic", "tanh", "relu"]
alphas = [1e-5, 1e-3, 1e-1, 10, 1000]
hiddens = [(), (1,), (2,), (3,), (4,), (5,)]

clf = make_pipeline(
    StandardScaler(), LinearDiscriminantAnalysis(), MLPClassifier(max_iter=100000)
)

gsc = GridSearchCV(
    clf,
    {
        "mlpclassifier__solver": solvers,
        "mlpclassifier__activation": activations,
        "mlpclassifier__alpha": alphas,
        "mlpclassifier__hidden_layer_sizes": hiddens,
    },
    n_jobs=4,
    verbose=3,
)

gsc.fit(x, y)

print(gsc.best_params_)

with open("mlpclassifier.pkl", "wb") as pklfile:
    pickle.dump(gsc, pklfile)
