# SPDX-License-Identifier: Apache-2.0

import os
import numpy
from pandas import read_csv
import matplotlib.pyplot as plt

HERE = os.path.abspath(os.path.dirname(__file__))


def autolabel(ax, rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(
            "%1.1fx" % height,
            xy=(rect.get_x() + rect.get_width() / 2, height),
            xytext=(0, 3),  # 3 points vertical offset
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )


def linear_models():
    filename1 = os.path.join(HERE, "bench_plot_onnxruntime_linreg.csv")
    filename2 = os.path.join(HERE, "bench_plot_onnxruntime_logreg.csv")
    if not os.path.exists(filename1) or not os.path.exists(filename2):
        return
    dfr = read_csv(filename1)
    dfr["speedup"] = dfr["time_skl"] / dfr["time_ort"]
    dfc = read_csv(filename2)
    dfc = dfc[(dfc.method == "predict_proba") & dfc.fit_intercept]
    dfc["speedup"] = dfc["time_skl"] / dfc["time_ort"]

    nfeats = [10, 50]
    fig, axs = plt.subplots(1, len(nfeats) * 2, figsize=(14, 4), sharey=True)

    names = ["LinearRegression", "LogisticRegression"]
    pos = 0
    for name, df in zip(names, [dfr, dfc]):
        for nf in nfeats:
            ax = axs[pos]
            sub = df[df.nfeat == nf]
            labels = sub.n_obs
            means = sub.speedup

            x = numpy.arange(len(labels))
            width = 0.90

            rects1 = ax.bar(x, means, width, label="Speedup")

            if pos == 0:
                ax.set_ylabel("Speedup")
            ax.set_title("%s %d features" % (name, nf))
            ax.set_xlabel("batch size")
            ax.set_xticks(x)
            ax.set_xticklabels(labels)
            autolabel(ax, rects1)
            for tick in ax.xaxis.get_major_ticks():
                tick.label.set_fontsize(8)
            for tick in ax.yaxis.get_major_ticks():
                tick.label.set_fontsize(8)
            pos += 1

    fig.tight_layout()
    fig.savefig("linear_model.png", dpi=1000)


def svm_models():
    filename = os.path.join(HERE, "bench_plot_onnxruntime_svm_reg.csv")
    if not os.path.exists(filename):
        return
    dfr = read_csv(filename)
    dfr["speedup"] = dfr["time_skl"] / dfr["time_ort"]
    print(dfr.tail())

    ncols = len(set(dfr["kernel"]))
    fig, axs = plt.subplots(1, ncols, figsize=(14, 4), sharey=True)

    name = "SVR"
    nf = 50
    pos = 0
    for kernel in sorted(set(dfr["kernel"])):
        sub = dfr[(dfr.kernel == kernel) & (dfr.nfeat == nf)]
        ax = axs[pos]
        labels = sub.n_obs
        means = sub.speedup

        x = numpy.arange(len(labels))
        width = 0.90

        rects1 = ax.bar(x, means, width, label="Speedup")

        if pos == 0:
            ax.set_ylabel("Speedup")
        ax.set_title("%s %s - %d features" % (name, kernel, nf))
        ax.set_xlabel("batch size")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        autolabel(ax, rects1)
        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(8)
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(8)
        pos += 1

    fig.tight_layout()
    fig.savefig("svm_model.png", dpi=1000)


def rf_models():
    filename = os.path.join(HERE, "bench_plot_onnxruntime_random_forest_reg.csv")
    if not os.path.exists(filename):
        return
    dfr = read_csv(filename)
    dfr["speedup"] = dfr["time_skl"] / dfr["time_ort"]
    print(dfr.tail().T)

    ncols = 4
    fig, axs = plt.subplots(1, ncols, figsize=(14, 4), sharey=True)

    name = "RandomForestRegressor"
    pos = 0
    for max_depth in [10]:
        for nf in [30, 100]:
            for est in [100, 200]:
                for n_jobs in [4]:
                    sub = dfr[
                        (dfr.max_depth == max_depth)
                        & (dfr.nfeat == nf)
                        & (dfr.n_estimators == est)
                        & (dfr.n_jobs == n_jobs)
                    ]
                    ax = axs[pos]
                    labels = sub.n_obs
                    means = sub.speedup

                    x = numpy.arange(len(labels))
                    width = 0.90

                    rects1 = ax.bar(x, means, width, label="Speedup")
                    if pos == 0:
                        ax.set_yscale("log")
                        ax.set_ylim([0.1, max(dfr["speedup"])])

                    if pos == 0:
                        ax.set_ylabel("Speedup")
                    ax.set_title(
                        "%s\ndepth %d - %d features\n %d estimators %d jobs"
                        "" % (name, max_depth, nf, est, n_jobs)
                    )
                    ax.set_xlabel("batch size")
                    ax.set_xticks(x)
                    ax.set_xticklabels(labels)
                    autolabel(ax, rects1)
                    for tick in ax.xaxis.get_major_ticks():
                        tick.label.set_fontsize(8)
                    for tick in ax.yaxis.get_major_ticks():
                        tick.label.set_fontsize(8)
                    pos += 1

    fig.tight_layout()
    fig.savefig("rf_model.png", dpi=1000)


if __name__ == "__main__":
    linear_models()
    svm_models()
    rf_models()
    # plt.show()
