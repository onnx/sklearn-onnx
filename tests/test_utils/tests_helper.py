# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
import pickle
import os
import warnings
import traceback
import time
import sys
import platform
import numpy
import pandas
from sklearn.base import BaseEstimator
from .utils_backend import (
    compare_backend,
    extract_options,
    evaluate_condition,
    is_backend_enabled,
)
from skl2onnx.common.data_types import FloatTensorType


def _has_predict_proba(model):
    if hasattr(model, "voting") and model.voting == "hard":
        return False
    return hasattr(model, "predict_proba")


def _has_decision_function(model):
    if hasattr(model, "voting"):
        return False
    return hasattr(model, "decision_function")


def _has_transform_model(model):
    if hasattr(model, "voting"):
        return False
    return hasattr(model, "fit_transform") and hasattr(model, "score")


def dump_data_and_model(
        data,
        model,
        onnx=None,
        basename="model",
        folder=None,
        inputs=None,
        backend="onnxruntime",
        context=None,
        allow_failure=None,
        methods=None,
        dump_error_log=None,
        benchmark=None,
        comparable_outputs=None,
        verbose=False,
):
    """
    Saves data with pickle, saves the model with pickle and *onnx*,
    runs and saves the predictions for the given model.
    This function is used to test a backend (runtime) for *onnx*.

    :param data: any kind of data
    :param model: any model
    :param onnx: *onnx* model or *None* to use an onnx converters to convert it
        only if the model accepts one float vector
    :param basemodel: three files are writen ``<basename>.data.pkl``,
        ``<basename>.model.pkl``, ``<basename>.model.onnx``
    :param folder: files are written in this folder,
        it is created if it does not exist, if *folder* is None,
        it looks first in environment variable ``ONNXTESTDUMP``,
        otherwise, it is placed into ``'tests_dump'``.
    :param inputs: standard type or specific one if specified, only used is
        parameter *onnx* is None
    :param backend: backend used to compare expected output and runtime output.
        Two options are currently supported: None for no test,
        `'onnxruntime'` to use module *onnxruntime*.
    :param context: used if the model contains a custom operator such
        as a custom Keras function...
    :param allow_failure: None to raise an exception if comparison fails
        for the backends, otherwise a string which is then evaluated to check
        whether or not the test can fail, example:
        ``"StrictVersion(onnx.__version__) < StrictVersion('1.3.0')"``
    :param dump_error_log: if True, dumps any error message in a file
        ``<basename>.err``, if it is None, it checks the environment
         variable ``ONNXTESTDUMPERROR``
    :param benchmark: if True, runs a benchmark and stores the results
        into a file ``<basename>.bench``, if None, it checks the environment
        variable ``ONNXTESTBENCHMARK``
    :param verbose: additional information
    :param methods: ONNX may produce one or several results, each of them
        is equivalent to the output of a method from the model class,
        this parameter defines which methods is equivalent to ONNX outputs.
        If not specified, it falls back into a default behaviour implemented
        for classifiers, regressors, clustering.
    :param comparable_outputs: compares only these outputs
    :return: the created files

    Some convention for the name,
    *Bin* for a binary classifier, *Mcl* for a multiclass
    classifier, *Reg* for a regressor, *MRg* for a multi-regressor.
    The name can contain some flags. Expected outputs refer to the
    outputs computed with the original library, computed outputs
    refer to the outputs computed with a ONNX runtime.

    * ``-CannotLoad``: the model can be converted but the runtime
      cannot load it
    * ``-Dec3``: compares expected and computed outputs up to
      3 decimals (5 by default)
    * ``-Dec4``: compares expected and computed outputs up to
      4 decimals (5 by default)
    * ``-NoProb``: The original models computed probabilites for two classes
      *size=(N, 2)* but the runtime produces a vector of size *N*, the test
      will compare the second column to the column
    * ``-OneOff``: the ONNX runtime cannot compute the prediction for
      several inputs, it must be called for each of them.
    * ``-OneOffArray``: same as ``-OneOff`` but input is still a 2D array
      with one observation
    * ``-Out0``: only compares the first output on both sides
    * ``-Reshape``: merges all outputs into one single vector and resizes
      it before comparing
    * ``-SkipDim1``: before comparing expected and computed output,
      arrays with a shape like *(2, 1, 2)* becomes *(2, 2)*
    * ``-SklCol``: *scikit-learn* operator applies on a column and not a matrix

    If the *backend* is not None, the function either raises an exception
    if the comparison between the expected outputs and the backend outputs
    fails or it saves the backend output and adds it to the results.
    """
    runtime_test = dict(model=model, data=data)

    if folder is None:
        folder = os.environ.get("ONNXTESTDUMP", "tests_dump")
    if dump_error_log is None:
        dump_error_log = os.environ.get("ONNXTESTDUMPERROR", "0") in (
            "1",
            1,
            "True",
            "true",
            True,
        )
    if benchmark is None:
        benchmark = os.environ.get("ONNXTESTBENCHMARK", "0") in (
            "1",
            1,
            "True",
            "true",
            True,
        )
    if not os.path.exists(folder):
        os.makedirs(folder)

    lambda_original = None
    if isinstance(data, (numpy.ndarray, pandas.DataFrame)):
        dataone = data[:1].copy()
    else:
        dataone = data

    if methods is not None:
        prediction = []
        for method in methods:
            call = getattr(model, method)
            if callable(call):
                prediction.append(call(data))
                # we only take the last one for benchmark
                lambda_original = lambda: call(dataone)  # NOQA
            else:
                raise RuntimeError(
                    "Method '{0}' is not callable.".format(method))
    else:
        if hasattr(model, "predict"):
            if _has_predict_proba(model):
                # Classifier
                prediction = [model.predict(data), model.predict_proba(data)]
                lambda_original = lambda: model.predict_proba(dataone)  # NOQA
            elif _has_decision_function(model):
                # Classifier without probabilities
                prediction = [
                    model.predict(data),
                    model.decision_function(data),
                ]
                lambda_original = (
                    lambda: model.decision_function(dataone))  # NOQA
            elif _has_transform_model(model):
                # clustering
                prediction = [model.predict(data), model.transform(data)]
                lambda_original = lambda: model.transform(dataone)  # NOQA
            else:
                # Regressor or VotingClassifier
                prediction = [model.predict(data)]
                lambda_original = lambda: model.predict(dataone)  # NOQA

        elif hasattr(model, "transform"):
            options = extract_options(basename)
            SklCol = options.get("SklCol", False)
            if SklCol:
                prediction = model.transform(data.ravel())
                lambda_original = lambda: model.transform(dataone.ravel())  # NOQA
            else:
                prediction = model.transform(data)
                lambda_original = lambda: model.transform(dataone)  # NOQA
        else:
            raise TypeError(
                "Model has not predict or transform method: {0}".format(
                    type(model)))

    runtime_test["expected"] = prediction

    names = []
    dest = os.path.join(folder, basename + ".expected.pkl")
    names.append(dest)
    with open(dest, "wb") as f:
        pickle.dump(prediction, f)

    dest = os.path.join(folder, basename + ".data.pkl")
    names.append(dest)
    with open(dest, "wb") as f:
        pickle.dump(data, f)

    if hasattr(model, "save"):
        dest = os.path.join(folder, basename + ".model.keras")
        names.append(dest)
        model.save(dest)
    else:
        dest = os.path.join(folder, basename + ".model.pkl")
        names.append(dest)
        with open(dest, "wb") as f:
            pickle.dump(model, f)

    if dump_error_log:
        error_dump = os.path.join(folder, basename + ".err")

    if onnx is None:
        array = numpy.array(data)
        if inputs is None:
            inputs = [("input", FloatTensorType(list(array.shape)))]
        onnx, _ = convert_model(model, basename, inputs)

    dest = os.path.join(folder, basename + ".model.onnx")
    names.append(dest)
    with open(dest, "wb") as f:
        f.write(onnx.SerializeToString())
    if verbose:
        print("[dump_data_and_model] created '{}'.".format(dest))

    runtime_test["onnx"] = dest

    # backend
    if backend is not None:
        if not isinstance(backend, list):
            backend = [backend]
        for b in backend:
            if not is_backend_enabled(b):
                continue
            if isinstance(allow_failure, str):
                allow = evaluate_condition(b, allow_failure)
            else:
                allow = allow_failure
            if allow is None:
                output, lambda_onnx = compare_backend(
                    b,
                    runtime_test,
                    options=extract_options(basename),
                    context=context,
                    verbose=verbose,
                    comparable_outputs=comparable_outputs,
                )
            else:
                try:
                    output, lambda_onnx = compare_backend(
                        b,
                        runtime_test,
                        options=extract_options(basename),
                        context=context,
                        verbose=verbose,
                        comparable_outputs=comparable_outputs,
                    )
                except AssertionError as e:
                    if dump_error_log:
                        with open(error_dump, "w", encoding="utf-8") as f:
                            f.write(str(e) + "\n--------------\n")
                            traceback.print_exc(file=f)
                    if isinstance(allow, bool) and allow:
                        warnings.warn("Issue with '{0}' due to {1}".format(
                            basename,
                            str(e).replace("\n", " -- ")))
                        continue
                    else:
                        raise e

            if output is not None:
                dest = os.path.join(folder,
                                    basename + ".backend.{0}.pkl".format(b))
                names.append(dest)
                with open(dest, "wb") as f:
                    pickle.dump(output, f)
                if (benchmark and lambda_onnx is not None
                        and lambda_original is not None):
                    # run a benchmark
                    obs = compute_benchmark({
                        "onnxrt": lambda_onnx,
                        "original": lambda_original
                    })
                    df = pandas.DataFrame(obs)
                    df["input_size"] = sys.getsizeof(dataone)
                    dest = os.path.join(folder, basename + ".bench")
                    df.to_csv(dest, index=False)

    return names


def convert_model(model, name, input_types):
    """
    Runs the appropriate conversion method.

    :param model: model, *scikit-learn*, *keras*,
         or *coremltools* object
    :return: *onnx* model
    """
    from skl2onnx import convert_sklearn

    model, prefix = convert_sklearn(model, name, input_types), "Sklearn"
    if model is None:
        raise RuntimeError("Unable to convert model of type '{0}'.".format(
            type(model)))
    return model, prefix


def dump_one_class_classification(
        model,
        suffix="",
        folder=None,
        allow_failure=None,
        comparable_outputs=None,
        verbose=False,
):
    """
    Trains and dumps a model for a One Class outlier problem.
    The function trains a model and calls
    :func:`dump_data_and_model`.

    Every created filename will follow the pattern:
    ``<folder>/<prefix><task><classifier-name><suffix>.<data|expected|model|onnx>.<pkl|onnx>``.
    """
    X = [[0.0, 1.0], [1.0, 1.0], [2.0, 0.0]]
    X = numpy.array(X, dtype=numpy.float32)
    y = [1, 1, 1]
    model.fit(X, y)
    model_onnx, prefix = convert_model(model, "one_class",
                                       [("input", FloatTensorType([1, 2]))])
    return dump_data_and_model(
        X,
        model,
        model_onnx,
        folder=folder,
        allow_failure=allow_failure,
        basename=prefix + "One" + model.__class__.__name__ + suffix,
        verbose=verbose,
        comparable_outputs=comparable_outputs,
    )


def dump_binary_classification(
        model,
        suffix="",
        folder=None,
        allow_failure=None,
        comparable_outputs=None,
        verbose=False,
):
    """
    Trains and dumps a model for a binary classification problem.
    The function trains a model and calls
    :func:`dump_data_and_model`.

    Every created filename will follow the pattern:
    ``<folder>/<prefix><task><classifier-name><suffix>.<data|expected|model|onnx>.<pkl|onnx>``.
    """
    X = [[0, 1], [1, 1], [2, 0]]
    X = numpy.array(X, dtype=numpy.float32)
    y = ["A", "B", "A"]
    model.fit(X, y)
    model_onnx, prefix = convert_model(model, "binary classifier",
                                       [("input", FloatTensorType([1, 2]))])
    dump_data_and_model(
        X,
        model,
        model_onnx,
        folder=folder,
        allow_failure=allow_failure,
        basename=prefix + "Bin" + model.__class__.__name__ + suffix,
        verbose=verbose,
        comparable_outputs=comparable_outputs,
    )


def dump_multiple_classification(
        model,
        suffix="",
        folder=None,
        allow_failure=None,
        verbose=False,
        label_string=False,
        first_class=0,
        comparable_outputs=None,
):
    """
    Trains and dumps a model for a binary classification problem.
    The function trains a model and calls
    :func:`dump_data_and_model`.

    Every created filename will follow the pattern:
    ``<folder>/<prefix><task><classifier-name><suffix>.<data|expected|model|onnx>.<pkl|onnx>``.
    """
    X = [[0, 1], [1, 1], [2, 0], [0.5, 0.5], [1.1, 1.1], [2.1, 0.1]]
    X = numpy.array(X, dtype=numpy.float32)
    y = [0, 1, 2, 1, 1, 2]
    y = [i + first_class for i in y]
    if label_string:
        y = ["l%d" % i for i in y]
    model.fit(X, y)
    if verbose:
        print("[dump_multiple_classification] model '{}'".format(
            model.__class__.__name__))
    model_onnx, prefix = convert_model(model, "multi-class classifier",
                                       [("input", FloatTensorType([1, 2]))])
    if verbose:
        print("[dump_multiple_classification] model was converted")
    dump_data_and_model(
        X,
        model,
        model_onnx,
        folder=folder,
        allow_failure=allow_failure,
        basename=prefix + "Mcl" + model.__class__.__name__ + suffix,
        verbose=verbose,
        comparable_outputs=comparable_outputs,
    )


def dump_multiple_regression(
        model,
        suffix="",
        folder=None,
        allow_failure=None,
        comparable_outputs=None,
        verbose=False,
):
    """
    Trains and dumps a model for a multi regression problem.
    The function trains a model and calls
    :func:`dump_data_and_model`.

    Every created filename will follow the pattern:
    ``<folder>/<prefix><task><classifier-name><suffix>.<data|expected|model|onnx>.<pkl|onnx>``.
    """
    X = [[0, 1], [1, 1], [2, 0]]
    X = numpy.array(X, dtype=numpy.float32)
    y = numpy.array([[100, 50], [100, 49], [100, 99]], dtype=numpy.float32)
    model.fit(X, y)
    model_onnx, prefix = convert_model(model, "multi-regressor",
                                       [("input", FloatTensorType([1, 2]))])
    dump_data_and_model(
        X,
        model,
        model_onnx,
        folder=folder,
        allow_failure=allow_failure,
        basename=prefix + "MRg" + model.__class__.__name__ + suffix,
        verbose=verbose,
        comparable_outputs=comparable_outputs,
    )


def dump_single_regression(model,
                           suffix="",
                           folder=None,
                           allow_failure=None,
                           comparable_outputs=None):
    """
    Trains and dumps a model for a regression problem.
    The function trains a model and calls
    :func:`dump_data_and_model`.

    Every created filename will follow the pattern:
    ``<folder>/<prefix><task><classifier-name><suffix>.<data|expected|model|onnx>.<pkl|onnx>``.
    """
    X = [[0, 1], [1, 1], [2, 0]]
    X = numpy.array(X, dtype=numpy.float32)
    y = numpy.array([100, -10, 50], dtype=numpy.float32)
    model.fit(X, y)
    model_onnx, prefix = convert_model(model, "single regressor",
                                       [("input", FloatTensorType([1, 2]))])
    dump_data_and_model(
        X,
        model,
        model_onnx,
        folder=folder,
        allow_failure=allow_failure,
        basename=prefix + "Reg" + model.__class__.__name__ + suffix,
        comparable_outputs=comparable_outputs,
    )


def timeit_repeat(fct, number, repeat):
    """
    Returns a series of *repeat* time measures for
    *number* executions of *code* assuming *fct*
    is a function.
    """
    res = []
    for r in range(0, repeat):
        t1 = time.perf_counter()
        for i in range(0, number):
            fct()
        t2 = time.perf_counter()
        res.append(t2 - t1)
    return res


def timeexec(fct, number, repeat):
    """
    Measures the time for a given expression.

    :param fct: function to measure (as a string)
    :param number: number of time to run the expression
        (and then divide by this number to get an average)
    :param repeat: number of times to repeat the computation
        of the above average
    :return: dictionary
    """
    rep = timeit_repeat(fct, number=number, repeat=repeat)
    ave = sum(rep) / (number * repeat)
    std = (sum((x / number - ave)**2 for x in rep) / repeat)**0.5
    fir = rep[0] / number
    fir3 = sum(rep[:3]) / (3 * number)
    las3 = sum(rep[-3:]) / (3 * number)
    rep.sort()
    mini = rep[len(rep) // 20] / number
    maxi = rep[-len(rep) // 20] / number
    return dict(
        average=ave,
        deviation=std,
        first=fir,
        first3=fir3,
        last3=las3,
        repeat=repeat,
        min5=mini,
        max5=maxi,
        run=number,
    )


def compute_benchmark(fcts, number=10, repeat=100):
    """
    Compares the processing time several functions.

    :param fcts: dictionary ``{'name': fct}``
    :param number: number of time to run the expression
        (and then divide by this number to get an average)
    :param repeat: number of times to repeat the computation
        of the above average
    :return: list of [{'name': name, 'time': ...}]
    """
    obs = []
    for name, fct in fcts.items():
        res = timeexec(fct, number=number, repeat=repeat)
        res["name"] = name
        obs.append(res)
    return obs


def get_nb_skl_objects(obj):
    """
    Returns the number of *sklearn* objects.
    """
    ct = 0
    if isinstance(obj, BaseEstimator):
        ct += 1
    if isinstance(obj, (list, tuple)):
        for o in obj:
            ct += get_nb_skl_objects(o)
    elif isinstance(obj, dict):
        for o in obj.values():
            ct += get_nb_skl_objects(o)
    elif isinstance(obj, BaseEstimator):
        for o in obj.__dict__.values():
            ct += get_nb_skl_objects(o)
    return ct


def stat_model_skl(model):
    """
    Computes statistics on the sklearn model.
    """
    with open(model, "rb") as f:
        obj = pickle.load(f)
    return {"nb_estimators": get_nb_skl_objects(obj)}


def stat_model_onnx(model):
    """
    Computes statistics on the ONNX model.
    """
    import onnx

    gr = onnx.load(model)
    return {"nb_onnx_nodes": len(gr.graph.node)}


def make_report_backend(folder, as_df=False):
    """
    Looks into a folder for dumped files after
    the unit tests.

    :param folder: dump folder, it should contain files *.bench*
    :param as_df: returns a dataframe instread of a list of dictionary
    :return: time execution
    """
    import onnx
    import onnxruntime
    import cpuinfo

    res = {}
    benched = 0
    files = os.listdir(folder)
    for name in files:
        if name.endswith(".expected.pkl"):
            model = name.split(".")[0]
            if model not in res:
                res[model] = {}
            res[model]["_tested"] = True
        elif ".backend." in name:
            bk = name.split(".backend.")[-1].split(".")[0]
            model = name.split(".")[0]
            if model not in res:
                res[model] = {}
            res[model][bk] = True
        elif name.endswith(".err"):
            model = name.split(".")[0]
            fullname = os.path.join(folder, name)
            with open(fullname, "r", encoding="utf-8") as f:
                content = f.read()
            error = content.split("\n")[0].strip("\n\r ")
            if model not in res:
                res[model] = {}
            res[model]["stderr"] = error
        elif name.endswith(".model.pkl"):
            model = name.split(".")[0]
            if model not in res:
                res[model] = {}
            res[model].update(stat_model_skl(os.path.join(folder, name)))
        elif name.endswith(".model.onnx"):
            model = name.split(".")[0]
            if model not in res:
                res[model] = {}
            res[model].update(stat_model_onnx(os.path.join(folder, name)))
        elif name.endswith(".bench"):
            model = name.split(".")[0]
            fullname = os.path.join(folder, name)
            df = pandas.read_csv(fullname, sep=",")
            if model not in res:
                res[model] = {}
            for index, row in df.iterrows():
                name = row["name"]
                ave = row["average"]
                std = row["deviation"]
                size = row["input_size"]
                res[model]["{0}_time".format(name)] = ave
                res[model]["{0}_std".format(name)] = std
                res[model]["input_size"] = size
                benched += 1

    if benched == 0:
        raise RuntimeError("No benchmark files in '{0}', found:\n{1}".format(
            folder, "\n".join(files)))

    def dict_update(d, u):
        d.update(u)
        return d

    aslist = [dict_update(dict(_model=k), v) for k, v in res.items()]

    if as_df:
        from pandas import DataFrame

        df = DataFrame(aslist).sort_values(["_model"])
        df["numpy-version"] = numpy.__version__
        df["onnx-version"] = onnx.__version__
        df["onnxruntime-version"] = onnxruntime.__version__
        cols = list(df.columns)
        if "stderr" in cols:
            ind = cols.index("stderr")
            del cols[ind]
            cols += ["stderr"]
            df = df[cols]
        for col in ["onnxrt_time", "original_time"]:
            if col not in df.columns:
                raise RuntimeError("Column '{0}' is missing from {1}".format(
                    col, ", ".join(df.columns)))
        df["ratio"] = df["onnxrt_time"] / df["original_time"]
        df["ratio_nodes"] = df["nb_onnx_nodes"] / df["nb_estimators"]
        df["CPU"] = platform.processor()
        df["CPUI"] = cpuinfo.get_cpu_info()["brand"]
        return df
    else:
        cpu = cpuinfo.get_cpu_info()["brand"]
        proc = platform.processor()
        for row in aslist:
            try:
                row["ratio"] = row["onnxrt_time"] / row["original_time"]
            except KeyError:
                # execution failed
                pass
            try:
                row["ratio_nodes"] = (row["nb_onnx_nodes"] /
                                      row["nb_estimators"])
            except KeyError:
                # execution failed
                pass
            row["CPU"] = proc
            row["CPUI"] = cpu
            row["numpy-version"] = numpy.__version__
            row["onnx-version"] = onnx.__version__
            row["onnxruntime-version"] = onnxruntime.__version__
        return aslist
