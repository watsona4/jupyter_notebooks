import glob
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.decomposition import PCA
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


names = ["buy", "hold", "sell"]

grid_search = False

x = np.empty((0, 2), dtype=np.float64)
y = np.empty((0,), dtype=np.int64)

for csvfile in glob.glob("btc_data_*.csv"):

    print(f"Reading {csvfile}...")

    df = pd.read_csv(csvfile, index_col=0)

    if df.empty:
        continue

    df["mark"] = pd.to_numeric(df["mark"])
    df["time"] = df.apply(timefunc, axis=1)

    df["prev"] = df["mark"].shift(1)
    df["next"] = df["mark"].shift(-1)

    df["action"] = df.apply(func, axis=1)

    df["pp1"] = (
        df["mark"].diff() / df["mark"] / df["time"].diff().dt.total_seconds()
    )
    df["pp2"] = df["pp1"].diff() / df["time"].diff().dt.total_seconds()
    # df["pp3"] = df["pp2"].diff() / df["time"].diff().dt.total_seconds()
    # df["pp4"] = df["pp3"].diff() / df["time"].diff().dt.total_seconds()

    x = np.append(x, df[["pp1", "pp2"]][2:].to_numpy(), axis=0)
    y = np.append(y, df["action"][2:].to_numpy(), axis=0)

print(x.shape)

x_train, x_test, y_train, y_test = train_test_split(x, y)

if grid_search:

    solvers = ["lbfgs", "adam"]
    activations = ["logistic", "tanh", "relu"]
    alphas = [0.0001, 0.1, 10]
    hiddens = [
        i * [int((x_train.shape[0] / 5) ** (1 / i))] for i in range(1, 4)
    ]

    clf = make_pipeline(
        StandardScaler(), PCA(), MLPClassifier(max_iter=100000)
    )

    gsc = GridSearchCV(
        clf,
        {
            "pca__n_components": [None, 1, 2, 3, "mle"],
            "pca__whiten": [True, False],
            "mlpclassifier__solver": solvers,
            "mlpclassifier__activation": activations,
            "mlpclassifier__alpha": alphas,
            "mlpclassifier__hidden_layer_sizes": hiddens,
        },
        n_jobs=4,
        verbose=3,
    )

    gsc.fit(x_train, y_train)

    print(gsc.best_params_)

    y_true, y_pred = y_test, gsc.predict(x_test)

    print(metrics.classification_report(y_true, y_pred, target_names=names))

n_layers = 2

npl = int((x_train.shape[0] / 5) ** (1 / n_layers))

clf = make_pipeline(
    StandardScaler(),
    PCA(n_components=2),
    MLPClassifier(
        solver="adam",
        activation="tanh",
        alpha=0.0001,
        hidden_layer_sizes=n_layers * [npl],
        max_iter=100000
    ),
)

print(clf.get_params())
clf.fit(x_train, y_train)

y_true, y_pred = y_test, clf.predict(x_test)

print(metrics.classification_report(y_true, y_pred, target_names=names))
