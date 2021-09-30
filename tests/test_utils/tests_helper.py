# SPDX-License-Identifier: Apache-2.0

import pickle
import os
import warnings
import traceback
import time
import sys
import platform
import numpy
import pandas
from sklearn.datasets import (
    make_classification,
    make_multilabel_classification,
    make_regression,
)
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator
from sklearn.preprocessing import MultiLabelBinarizer
from .utils_backend import (
    compare_backend,
    extract_options,
    evaluate_condition,
    is_backend_enabled,
    OnnxRuntimeMissingNewOnnxOperatorException,
)
from skl2onnx.common.data_types import FloatTensorType


def _has_predict_proba(model):
    if hasattr(model, "voting") and model.voting == "hard":
        return False
    return hasattr(model, "predict_proba")


def _has_decision_function(model):
    if hasattr(model, "voting"):
        return False
    if hasattr(model, "dtype"):  # CastRegressor
        return False
    return hasattr(model, "decision_function")


def _has_transform_model(model):
    if hasattr(model, "voting"):
        return False
    return hasattr(model, "fit_transform") and hasattr(model, "score")


def fit_classification_model(model, n_classes, is_int=False,
                             pos_features=False, label_string=False,
                             random_state=42, is_bool=False,
                             n_features=20, n_redundant=None,
                             n_repeated=None):
    X, y = make_classification(
        n_classes=n_classes, n_features=n_features, n_samples=250,
        random_state=random_state, n_informative=min(7, n_features),
        n_redundant=n_redundant or min(2, n_features - min(7, n_features)),
        n_repeated=n_repeated or 0)
    if label_string:
        y = numpy.array(['cl%d' % cl for cl in y])
    X = X.astype(numpy.int64) if is_int or is_bool else X.astype(numpy.float32)
    if pos_features:
        X = numpy.abs(X)
    if is_bool:
        X = X.astype(bool)
    X_train, X_test, y_train, _ = train_test_split(X, y, test_size=0.5,
                                                   random_state=42)
    model.fit(X_train, y_train)
    return model, X_test


def fit_clustering_model(model, n_classes, is_int=False,
                         pos_features=False, label_string=False,
                         random_state=42, is_bool=False,
                         n_features=20, n_redundant=None,
                         n_repeated=None):
    X, y = make_classification(
        n_classes=n_classes, n_features=n_features, n_samples=250,
        random_state=random_state, n_informative=min(7, n_features),
        n_redundant=n_redundant or min(2, n_features - min(7, n_features)),
        n_repeated=n_repeated or 0)
    if label_string:
        y = numpy.array(['cl%d' % cl for cl in y])
    X = X.astype(numpy.int64) if is_int or is_bool else X.astype(numpy.float32)
    if pos_features:
        X = numpy.abs(X)
    if is_bool:
        X = X.astype(bool)
    X_train, X_test = train_test_split(X, test_size=0.5, random_state=42)
    model.fit(X_train)
    return model, X_test


def fit_multilabel_classification_model(model, n_classes=5, n_labels=2,
                                        n_samples=200, n_features=20,
                                        is_int=False):
    X, y = make_multilabel_classification(
        n_classes=n_classes, n_labels=n_labels, n_features=n_features,
        n_samples=n_samples, random_state=42)
    X = X.astype(numpy.int64) if is_int else X.astype(numpy.float32)
    X_train, X_test, y_train, _ = train_test_split(X, y, test_size=0.5,
                                                   random_state=42)
    model.fit(X_train, y_train)
    return model, X_test


def fit_regression_model(model, is_int=False, n_targets=1, is_bool=False,
                         factor=1., n_features=10, n_samples=250,
                         n_informative=10):
    X, y = make_regression(n_features=n_features, n_samples=n_samples,
                           n_targets=n_targets, random_state=42,
                           n_informative=n_informative)
    y *= factor
    X = X.astype(numpy.int64) if is_int or is_bool else X.astype(numpy.float32)
    if is_bool:
        X = X.astype(bool)
    X_train, X_test, y_train, _ = train_test_split(X, y, test_size=0.5,
                                                   random_state=42)
    model.fit(X_train, y_train)
    return model, X_test


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
        intermediate_steps=False,
        fail_evenif_notimplemented=False,
        verbose=False,
        classes=None,
        disable_optimisation=False):
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
    :param intermediate_steps: displays intermediate steps
        in case of an error
    :param fail_evenif_notimplemented: the test is considered as failing
        even if the error is due to onnxuntime missing the implementation
        of a new operator defiend in ONNX.
    :param classes: classes names
        (only for classifier, mandatory if option 'nocl' is used)
    :param disable_optimisation: disable all optimisations *onnxruntime*
        could do
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

    def _raw_score_binary_classification(model, X):
        scores = model.decision_function(X)
        if len(scores.shape) == 1:
            scores = scores.reshape(-1, 1)
        if len(scores.shape) != 2 or scores.shape[1] != 1:
            raise RuntimeError(
                "Unexpected shape {} for a binary classifiation".format(
                    scores.shape))
        return numpy.hstack([-scores, scores])

    if methods is not None:
        prediction = []
        for method in methods:
            if callable(method):
                def call(X, model=model): return method(model, X)  # noqa
            else:
                try:
                    call = getattr(model, method)
                except AttributeError as e:
                    if method == 'decision_function_binary':
                        call = (
                            lambda X, model=model:
                                _raw_score_binary_classification(model, X))
                    else:
                        raise e
            if callable(call):
                prediction.append(call(data))
                # we only take the last one for benchmark
                def lambda_original(): return call(dataone)  # noqa
            else:
                raise RuntimeError(
                    "Method '{0}' is not callable.".format(method))
    else:
        if hasattr(model, "predict"):
            if _has_predict_proba(model):
                # Classifier
                prediction = [model.predict(data), model.predict_proba(data)]
                def lambda_original(): return model.predict_proba(dataone)  # noqa
            elif _has_decision_function(model):
                # Classifier without probabilities
                prediction = [
                    model.predict(data),
                    model.decision_function(data),
                ]
                lambda_original = (
                    lambda: model.decision_function(dataone))  # noqa
            elif _has_transform_model(model):
                # clustering
                try:
                    prediction = [model.predict(data), model.transform(data)]
                    def lambda_original(): return model.transform(dataone)  # noqa
                except ValueError as e:
                    if "Buffer dtype mismatch" in str(e):
                        # scikit-learn does not cast anymore
                        data64 = data.astype(numpy.float64)
                        prediction = [model.predict(data64),
                                      model.transform(data64)]
                        dataone64 = dataone.astype(numpy.float64)
                        def lambda_original(): return model.transform(dataone64)  # noqa
                    else:
                        raise e
            else:
                # Regressor or VotingClassifier
                prediction = [model.predict(data)]
                def lambda_original(): return model.predict(dataone)  # noqa

        elif hasattr(model, "transform"):
            options = extract_options(basename)
            SklCol = options.get("SklCol", False)
            if SklCol:
                prediction = model.transform(data.ravel())
                def lambda_original(): return model.transform(dataone.ravel())  # noqa
            else:
                prediction = model.transform(data)
                def lambda_original(): return model.transform(dataone)  # noqa
        else:
            raise TypeError(
                "Model has no predict or transform method: {0}".format(
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

    dest = os.path.join(folder, basename + ".model.pkl")
    names.append(dest)
    load_pickle = True
    with open(dest, "wb") as f:
        try:
            pickle.dump(model, f)
        except AttributeError as e:
            print("[dump_data_and_model] cannot pickle model '{}'"
                  " due to {}.".format(dest, e))
            load_pickle = False
    if load_pickle and os.path.exists(dest):
        # Test unpickle works.
        with open(dest, "rb") as f:
            pickle.load(f)

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
                    intermediate_steps=intermediate_steps,
                    disable_optimisation=disable_optimisation,
                    classes=classes)
            else:
                try:
                    output, lambda_onnx = compare_backend(
                        b, runtime_test,
                        options=extract_options(basename),
                        context=context, verbose=verbose,
                        comparable_outputs=comparable_outputs,
                        intermediate_steps=intermediate_steps,
                        classes=classes,
                        disable_optimisation=disable_optimisation)
                except OnnxRuntimeMissingNewOnnxOperatorException as e:
                    if fail_evenif_notimplemented:
                        raise e
                    warnings.warn(str(e))
                    continue
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


def convert_model(model, name, input_types, target_opset=None):
    """
    Runs the appropriate conversion method.

    :param model: model created with *scikit-learn*
    :return: *onnx* model
    """
    from skl2onnx import convert_sklearn
    if target_opset is None:
        from . import TARGET_OPSET
        target_opset = TARGET_OPSET

    model, prefix = (
        convert_sklearn(model, name, input_types, target_opset=target_opset),
        "Sklearn")
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
        target_opset=None):
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
    model_onnx, prefix = convert_model(
        model, "one_class", [("input", FloatTensorType([None, 2]))],
        target_opset=target_opset)
    dump_data_and_model(
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
        label_string=True,
        target_opset=None):
    """
    Trains and dumps a model for a binary classification problem.
    The function trains a model and calls
    :func:`dump_data_and_model`.

    Every created filename will follow the pattern:
    ``<folder>/<prefix><task><classifier-name><suffix>.<data|expected|model|onnx>.<pkl|onnx>``.
    """
    X = [[0, 1], [1, 1], [2, 0]]
    X = numpy.array(X, dtype=numpy.float32)
    if label_string:
        y = ["A", "B", "A"]
    else:
        y = [0, 1, 0]
    model.fit(X, y)
    model_onnx, prefix = convert_model(
        model, "binary classifier", [("input", FloatTensorType([None, 2]))],
        target_opset=target_opset)
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

    X, y = make_classification(10, n_features=4, random_state=42)
    X = X[:, :2]
    model.fit(X, y)
    model_onnx, prefix = convert_model(model, "binary classifier",
                                       [("input", FloatTensorType([None, 2]))],
                                       target_opset=target_opset)
    dump_data_and_model(
        X.astype(numpy.float32),
        model,
        model_onnx,
        folder=folder,
        allow_failure=allow_failure,
        basename=prefix + "RndBin" + model.__class__.__name__ + suffix,
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
        label_uint8=False,
        first_class=0,
        comparable_outputs=None,
        target_opset=None):
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
        if label_uint8:
            raise AssertionError(
                "label_string and label_uint8 cannot be both True")
        y = ["l%d" % i for i in y]
        suffix += 'String'
    elif label_uint8:
        y = numpy.array(y).astype(numpy.uint8)
        suffix += 'Uint8'
    model.fit(X, y)
    if verbose:
        print("[dump_multiple_classification] model '{}'".format(
            model.__class__.__name__))
    model_onnx, prefix = convert_model(
        model, "multi-class classifier",
        [("input", FloatTensorType([None, 2]))],
        target_opset=target_opset)
    if verbose:
        print("[dump_multiple_classification] model was converted")
    dump_data_and_model(
        X.astype(numpy.float32),
        model,
        model_onnx,
        folder=folder,
        allow_failure=allow_failure,
        basename=prefix + "Mcl" + model.__class__.__name__ + suffix,
        verbose=verbose,
        comparable_outputs=comparable_outputs,
    )

    X, y = make_classification(40, n_features=4, random_state=42,
                               n_classes=3, n_clusters_per_class=1)
    X = X[:, :2]
    model.fit(X, y)
    if verbose:
        print("[dump_multiple_classification] model '{}'".format(
            model.__class__.__name__))
    model_onnx, prefix = convert_model(model, "multi-class classifier",
                                       [("input", FloatTensorType([None, 2]))],
                                       target_opset=target_opset)
    if verbose:
        print("[dump_multiple_classification] model was converted")
    dump_data_and_model(
        X[:10].astype(numpy.float32),
        model,
        model_onnx,
        folder=folder,
        allow_failure=allow_failure,
        basename=prefix + "RndMcl" + model.__class__.__name__ + suffix,
        verbose=verbose,
        comparable_outputs=comparable_outputs,
    )


def dump_multilabel_classification(
        model,
        suffix="",
        folder=None,
        allow_failure=None,
        verbose=False,
        label_string=False,
        first_class=0,
        comparable_outputs=None,
        target_opset=None):
    """
    Trains and dumps a model for a binary classification problem.
    The function trains a model and calls
    :func:`dump_data_and_model`.

    Every created filename will follow the pattern:
    ``<folder>/<prefix><task><classifier-name><suffix>.<data|expected|model|onnx>.<pkl|onnx>``.
    """
    X = [[0, 1], [1, 1], [2, 0], [0.5, 0.5], [1.1, 1.1], [2.1, 0.1]]
    X = numpy.array(X, dtype=numpy.float32)
    if label_string:
        y = [["l0"], ["l1"], ["l2"], ["l0", "l1"], ["l1"], ["l2"]]
    else:
        y = [[0 + first_class], [1 + first_class], [2 + first_class],
             [0 + first_class, 1 + first_class],
             [1 + first_class], [2 + first_class]]
    y = MultiLabelBinarizer().fit_transform(y)
    model.fit(X, y)
    if verbose:
        print("[make_multilabel_classification] model '{}'".format(
            model.__class__.__name__))
    model_onnx, prefix = convert_model(
        model, "multi-class classifier",
        [("input", FloatTensorType([None, 2]))],
        target_opset=target_opset)
    if verbose:
        print("[make_multilabel_classification] model was converted")
    dump_data_and_model(
        X.astype(numpy.float32),
        model,
        model_onnx,
        folder=folder,
        allow_failure=allow_failure,
        basename=prefix + "Mcl" + model.__class__.__name__ + suffix,
        verbose=verbose,
        comparable_outputs=comparable_outputs,
    )

    X, y = make_multilabel_classification(40, n_features=4, random_state=42,
                                          n_classes=3)
    X = X[:, :2]
    model.fit(X, y)
    if verbose:
        print("[make_multilabel_classification] model '{}'".format(
            model.__class__.__name__))
    model_onnx, prefix = convert_model(model, "multi-class classifier",
                                       [("input", FloatTensorType([None, 2]))])
    if verbose:
        print("[make_multilabel_classification] model was converted")
    dump_data_and_model(
        X[:10].astype(numpy.float32),
        model,
        model_onnx,
        folder=folder,
        allow_failure=allow_failure,
        basename=prefix + "RndMla" + model.__class__.__name__ + suffix,
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
        target_opset=None):
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
    model_onnx, prefix = convert_model(
        model, "multi-regressor", [("input", FloatTensorType([None, 2]))],
        target_opset=target_opset)
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
                           comparable_outputs=None,
                           target_opset=None):
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
    model_onnx, prefix = convert_model(
        model, "single regressor", [("input", FloatTensorType([None, 2]))],
        target_opset=target_opset)
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
    try:
        with open(model, "rb") as f:
            obj = pickle.load(f)
    except EOFError:
        return {"nb_estimators": 0}
    return {"nb_estimators": get_nb_skl_objects(obj)}


def stat_model_onnx(model):
    """
    Computes statistics on the ONNX model.
    """
    import onnx

    gr = onnx.load(model)
    return {"nb_onnx_nodes": len(gr.graph.node)}


def make_report_backend(folder, as_df=False, verbose=0):
    """
    Looks into a folder for dumped files after
    the unit tests.

    :param folder: dump folder, it should contain files *.bench*
    :param as_df: returns a dataframe instread of a list of dictionary
    :param verbose: display progress
    :return: time execution
    """
    import onnx
    import onnxruntime
    import cpuinfo

    res = {}
    benched = 0
    files = os.listdir(folder)
    for name in files:
        if verbose:
            print("[make_report_backend] name=%r" % name)
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
        info = cpuinfo.get_cpu_info()
        df["CPUI"] = info.get("brand", info.get('brand_raw', '?'))
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


def binary_array_to_string(mat):
    if not isinstance(mat, numpy.ndarray):
        raise NotImplementedError()
    if len(mat.shape) != 2:
        raise NotImplementedError()
    res = [[str(i) for i in row] for row in mat.tolist()]
    return [''.join(row) for row in res]


def path_to_leaf(tree, mat, tree_indices=None):
    if tree_indices is None:
        # single tree
        leave = set([i for i in range(tree.node_count)
                     if tree.children_left[i] <= i])
        res = []
        for row in range(mat.shape[0]):
            leaf = None
            for i in range(mat.shape[1]):
                if mat[row, i] == 1 and i in leave:
                    leaf = i
                    break
            if leaf is None:
                raise AssertionError("Path does not end with a leaf.")
            res.append(leaf)
        return numpy.array(res, numpy.int64)

    leaves = []
    for i in range(0, len(tree)):
        mm = mat[:, tree_indices[i]:tree_indices[i+1]]
        tt = tree[i].tree_ if hasattr(tree[i], 'tree_') else tree[i]
        res = path_to_leaf(tt, mm)
        leaves.append(numpy.array(res, dtype=numpy.int64))
    res = numpy.vstack(leaves)
    return res.T
