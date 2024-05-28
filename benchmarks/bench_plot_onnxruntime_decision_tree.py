# SPDX-License-Identifier: Apache-2.0

# coding: utf-8
"""
Benchmark of onnxruntime on DecisionTree.
"""
# Authors: Xavier Dupré (benchmark)
from io import BytesIO
from time import perf_counter as time
import numpy as np
from numpy.random import rand
from numpy.testing import assert_almost_equal
import matplotlib.pyplot as plt
import pandas
from sklearn import config_context
from sklearn.tree import DecisionTreeClassifier

try:
    # scikit-learn >= 0.22
    from sklearn.utils._testing import ignore_warnings
except ImportError:
    # scikit-learn < 0.22
    from sklearn.utils.testing import ignore_warnings
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from onnxruntime import InferenceSession


##############################
# Implementations to benchmark.
##############################


def fcts_model(X, y, max_depth):
    "DecisionTreeClassifier."
    rf = DecisionTreeClassifier(max_depth=max_depth)
    rf.fit(X, y)

    initial_types = [("X", FloatTensorType([None, X.shape[1]]))]
    onx = convert_sklearn(
        rf,
        initial_types=initial_types,
        options={DecisionTreeClassifier: {"zipmap": False}},
    )
    f = BytesIO()
    f.write(onx.SerializeToString())
    content = f.getvalue()
    sess = InferenceSession(content, providers=["CPUExecutionProvider"])

    outputs = [o.name for o in sess.get_outputs()]

    def predict_skl_predict(X, model=rf):
        return rf.predict(X)

    def predict_skl_predict_proba(X, model=rf):
        return rf.predict_proba(X)

    def predict_onnxrt_predict(X, sess=sess):
        return sess.run(outputs[:1], {"X": X})[0]

    def predict_onnxrt_predict_proba(X, sess=sess):
        return sess.run(outputs[1:], {"X": X})[0]

    return {
        "predict": (predict_skl_predict, predict_onnxrt_predict),
        "predict_proba": (predict_skl_predict_proba, predict_onnxrt_predict_proba),
    }


##############################
# Benchmarks
##############################


def allow_configuration(**kwargs):
    return True


def bench(n_obs, n_features, max_depths, methods, repeat=10, verbose=False):
    res = []
    for nfeat in n_features:
        ntrain = 100000
        X_train = np.empty((ntrain, nfeat))
        X_train[:, :] = rand(ntrain, nfeat)[:, :].astype(np.float32)
        X_trainsum = X_train.sum(axis=1)
        eps = rand(ntrain) - 0.5
        X_trainsum_ = X_trainsum + eps
        y_train = (X_trainsum_ >= X_trainsum).ravel().astype(int)

        for max_depth in max_depths:
            fcts = fcts_model(X_train, y_train, max_depth)

            for n in n_obs:
                for method in methods:
                    fct1, fct2 = fcts[method]

                    if not allow_configuration(n=n, nfeat=nfeat, max_depth=max_depth):
                        continue

                    obs = dict(n_obs=n, nfeat=nfeat, max_depth=max_depth, method=method)

                    # creates different inputs to avoid caching in any ways
                    Xs = []
                    for r in range(repeat):
                        x = np.empty((n, nfeat))
                        x[:, :] = rand(n, nfeat)[:, :]
                        Xs.append(x.astype(np.float32))

                    # measures the baseline
                    with config_context(assume_finite=True):
                        st = time()
                        repeated = 0
                        for X in Xs:
                            p1 = fct1(X)
                            repeated += 1
                            if time() - st >= 1:
                                break  # stops if longer than a second
                        end = time()
                        obs["time_skl"] = (end - st) / repeated

                    # measures the new implementation
                    st = time()
                    r2 = 0
                    for X in Xs:
                        p2 = fct2(X)
                        r2 += 1
                        if r2 >= repeated:
                            break
                    end = time()
                    obs["time_ort"] = (end - st) / repeated
                    res.append(obs)
                    if verbose and (len(res) % 1 == 0 or n >= 10000):
                        print("bench", len(res), ":", obs)

                    # checks that both produce the same outputs
                    if n <= 10000:
                        if len(p1.shape) == 1 and len(p2.shape) == 2:
                            p2 = p2.ravel()
                        assert_almost_equal(p1, p2, decimal=5)
    return res


##############################
# Plots.
##############################


def plot_results(df, verbose=False):
    nrows = max(len(set(df.max_depth)) * len(set(df.n_obs)), 2)
    ncols = max(len(set(df.method)), 2)
    fig, ax = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 4))
    pos = 0
    row = 0
    for n_obs in sorted(set(df.n_obs)):
        for max_depth in sorted(set(df.max_depth)):
            pos = 0
            for method in sorted(set(df.method)):
                a = ax[row, pos]
                if row == ax.shape[0] - 1:
                    a.set_xlabel("N features", fontsize="x-small")
                if pos == 0:
                    a.set_ylabel(
                        "Time (s) n_obs={}\nmax_depth={}".format(n_obs, max_depth),
                        fontsize="x-small",
                    )

                color = "b"
                subset = df[
                    (df.method == method)
                    & (df.n_obs == n_obs)
                    & (df.max_depth == max_depth)
                ]
                if subset.shape[0] == 0:
                    continue
                subset = subset.sort_values("nfeat")
                if verbose:
                    print(subset)
                label = "skl"
                subset.plot(
                    x="nfeat",
                    y="time_skl",
                    label=label,
                    ax=a,
                    logx=True,
                    logy=True,
                    c=color,
                    style="--",
                )
                label = "ort"
                subset.plot(
                    x="nfeat",
                    y="time_ort",
                    label=label,
                    ax=a,
                    logx=True,
                    logy=True,
                    c=color,
                )

                a.legend(loc=0, fontsize="x-small")
                if row == 0:
                    a.set_title("method={}".format(method), fontsize="x-small")
                pos += 1
            row += 1

    plt.suptitle("Benchmark for DecisionTree sklearn/onnxruntime", fontsize=16)


@ignore_warnings(category=FutureWarning)
def run_bench(repeat=100, verbose=False):
    n_obs = [1, 10, 100, 1000, 10000, 100000]
    methods = ["predict", "predict_proba"]
    n_features = [1, 5, 10, 20, 50, 100, 200]
    max_depths = [2, 5, 10, 20]

    start = time()
    results = bench(
        n_obs, n_features, max_depths, methods, repeat=repeat, verbose=verbose
    )
    end = time()

    results_df = pandas.DataFrame(results)
    print("Total time = %0.3f sec\n" % (end - start))

    # plot the results
    plot_results(results_df, verbose=verbose)
    return results_df


if __name__ == "__main__":
    from datetime import datetime
    import sklearn
    import numpy
    import onnx
    import onnxruntime
    import skl2onnx

    df = pandas.DataFrame(
        [
            {"name": "date", "version": str(datetime.now())},
            {"name": "numpy", "version": numpy.__version__},
            {"name": "scikit-learn", "version": sklearn.__version__},
            {"name": "onnx", "version": onnx.__version__},
            {"name": "onnxruntime", "version": onnxruntime.__version__},
            {"name": "skl2onnx", "version": skl2onnx.__version__},
        ]
    )
    df.to_csv("bench_plot_onnxruntime_decision_tree.time.csv", index=False)
    print(df)
    df = run_bench(verbose=True)
    plt.savefig("bench_plot_onnxruntime_decision_tree.png")
    df.to_csv("bench_plot_onnxruntime_decision_tree.csv", index=False)
    plt.show()
