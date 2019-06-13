# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
from time import perf_counter
from importlib import import_module
import numpy as np
import pandas
from sklearn import __all__ as sklearn__all__, __version__ as sklearn_version
from sklearn.base import BaseEstimator
from sklearn.datasets import load_iris
from sklearn.ensemble import VotingClassifier
from sklearn.feature_selection import SelectFromModel, RFE, RFECV
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, NuSVC
from sklearn.tree import DecisionTreeRegressor
from .common.data_types import FloatTensorType
from ._supported_operators import build_sklearn_operator_name_map
from .convert import get_opset_number_from_onnx, convert_sklearn


def sklearn_operators():
    """
    Builds the list of supported and not supported
    *scikit-learn* models.
    """
    supported = set(build_sklearn_operator_name_map())
    found = []
    for sub in sklearn__all__:
        try:
            mod = import_module("{0}.{1}".format("sklearn", sub))
        except ModuleNotFoundError:
            continue
        cls = getattr(mod, "__all__", None)
        if cls is None:
            cls = list(mod.__dict__)
        cls = [mod.__dict__[cl] for cl in cls]
        for cl in cls:
            try:
                issub = issubclass(cl, BaseEstimator)
            except TypeError:
                continue
            if cl.__name__ in {'Pipeline', 'ColumnTransformer',
                               'FeatureUnion', 'BaseEstimator'}:
                continue
            if (sub in {'calibration', 'dummy', 'manifold'} and
                    'Calibrated' not in cl.__name__):
                continue
            if issub:
                found.append(dict(name=cl.__name__, subfolder=sub, cl=cl,
                                  supported=cl in supported))
    return found


def _problem_for_predictor_binary_classification():
    """
    Returns *X, y, intial_types, method, node name, X runtime* for a
    binary classification problem.
    It is based on Iris dataset.
    """
    data = load_iris()
    X = data.data
    y = data.target
    y[y == 2] = 1
    return (X, y, [('X', FloatTensorType((1, X.shape[1])))],
            'predict_proba', 1, X.astype(np.float32))


def _problem_for_predictor_multi_classification():
    """
    Returns *X, y, intial_types, method, node name, X runtime* for a
    multi-class classification problem.
    It is based on Iris dataset.
    """
    data = load_iris()
    X = data.data
    y = data.target
    return (X, y, [('X', FloatTensorType((1, X.shape[1])))],
            'predict_proba', 1, X.astype(np.float32))


def _problem_for_predictor_regression():
    """
    Returns *X, y, intial_types, method, name, X runtime* for a
    multi-class classification problem.
    It is based on Iris dataset.
    """
    data = load_iris()
    X = data.data
    y = data.target
    return (X, y.astype(float), [('X', FloatTensorType((1, X.shape[1])))],
            'predict', 0, X.astype(np.float32))


def _problem_for_numerical_transform():
    """
    Returns *X, y, intial_types, method, name, X runtime* for a
    multi-class classification problem.
    It is based on Iris dataset.
    """
    data = load_iris()
    X = data.data
    return (X, None, [('X', FloatTensorType((1, X.shape[1])))],
            'transform', 0, X.astype(np.float32))


def find_suitable_problem(model):
    """
    Finds a datasets suitable for a given operator.
    """
    if hasattr(model, 'predict_proba'):
        return ['bin-class', 'multi-class']

    if hasattr(model, 'predict'):
        return ['regression']

    if hasattr(model, 'transform'):
        return ['num-transform']

    raise RuntimeError("Unable to find problem for model '{}'."
                       "".format(model.__name__))


_problems = {
    "bin-class": _problem_for_predictor_binary_classification,
    "multi-class": _problem_for_predictor_multi_classification,
    "regression": _problem_for_predictor_regression,
    "num-transform": _problem_for_numerical_transform,
}


_extra_parameters = {
    LogisticRegression: [
        ('liblinear', {
            'solver': 'liblinear'
        }),
    ],
    NuSVC: [
        ('prob', {
            'probability': True,
        }),
    ],
    OneVsRestClassifier: [
        ('logreg', {
            'estimator': LogisticRegression(solver='liblinear'),
        })
    ],
    RFE: [
        ('logreg', {
            'estimator': LogisticRegression(solver='liblinear'),
        })
    ],
    RFECV: [
        ('logreg', {
            'estimator': LogisticRegression(solver='liblinear'),
        })
    ],
    SelectFromModel: [
        ('rf', {
            'estimator': DecisionTreeRegressor(),
        }),
    ],
    SGDClassifier: [
        ('log', {
            'loss': 'log',
        }),
    ],
    SVC: [
        ('prob', {
            'probability': True,
        }),
    ],
    VotingClassifier: [
        ('logreg', {
            'voting': 'soft',
            'estimators': [
                ('lr1', LogisticRegression(solver='liblinear')),
                ('lr2', LogisticRegression(solver='liblinear',
                                           fit_intercept=False)),
            ],
        })
    ],
}


def _measure_time(fct):
    begin = perf_counter()
    res = fct()
    end = perf_counter()
    return res, begin - end


def _measure_absolute_difference(skl_pred, ort_pred):
    """
    *Measures the differences between predictions
    from sickit-learn and onnxruntime.
    The functions returns nan if shapes are different.
    """
    ort_pred_ = ort_pred
    if isinstance(ort_pred, list):
        if isinstance(ort_pred[0], dict):
            ort_pred = pandas.DataFrame(ort_pred).values
        else:
            ort_pred = np.array(ort_pred)
    if skl_pred.shape != ort_pred.shape and len(ort_pred.shape) > 1:
        sh = list(set(ort_pred.shape[1:]))
        if len(sh) != 1 or sh[0] != 1:
            return np.nan
        ort_pred = ort_pred.ravel()

    if skl_pred.shape != ort_pred.shape:
        return np.nan

    if hasattr(skl_pred, 'todense'):
        skl_pred = skl_pred.todense()
    if hasattr(ort_pred, 'todense'):
        ort_pred = ort_pred.todense()

    diff = np.max(np.abs(skl_pred.ravel() - ort_pred.ravel()))

    if np.isnan(diff):
        raise RuntimeError("Unable to compute differences between\n{}\n"
                           "--------\n{}".format(skl_pred, ort_pred_))
    return diff


def enumerate_compatible_opset(model, opset_min=1, opset_max=None,
                               check_onnxruntime=True, debug=False):
    """
    Lists all compatiable opsets for a specific model.

    :param model: operator class
    :param opset_min: starts with this opset
    :param opset_max: ends with this opset (None to use
        current onnx opset)
    :param check_onnxruntime: checks that *onnxruntime* can
        consume the model
    :param debug: catch exception (True) or not (False)
    :return: dictionaries, each row has the following
        keys: opset, exception if any, conversion time,
        problem chosen to test the conversion...
    """
    try:
        problems = find_suitable_problem(model)
    except RuntimeError as e:
        yield {'name': model.__name__, 'skl_version': sklearn_version,
               'problem_exc': e}
        problems = []

    extras = _extra_parameters.get(model, [('default', {})])
    if opset_max is None:
        opset_max = get_opset_number_from_onnx()
    opsets = list(range(opset_min, opset_max + 1))
    opsets.append(None)

    if check_onnxruntime:
        from onnxruntime import __version__ as ort_version
        from onnxruntime import InferenceSession

    for prob in problems:
        X_, y_, init_types, method, output_index, Xort_ = _problems[prob]()
        if y_ is None:
            (X_train, X_test, Xort_train,
                Xort_test) = train_test_split(
                    X_, Xort_, random_state=42)
        else:
            (X_train, X_test, y_train, y_test,
                Xort_train, Xort_test) = train_test_split(
                    X_, y_, Xort_, random_state=42)

        for scenario, extra in extras:

            # training
            obs = {'scenario': scenario, 'name': model.__name__,
                   'skl_version': sklearn_version, 'problem': prob}
            try:
                inst = model(**extra)
            except TypeError as e:
                if debug:
                    raise
                import pprint
                raise RuntimeError(
                    "Unable to instantiate model '{}'.\nextra=\n{}".format(
                        model.__name__, pprint.pformat(extra))) from e

            try:
                if y_ is None:
                    t1 = _measure_time(lambda: inst.fit(X_train))[1]
                else:
                    t1 = _measure_time(lambda: inst.fit(X_train, y_train))[1]
            except (AttributeError, TypeError, ValueError) as e:
                obs["training_time_exc"] = str(e)
                yield obs
                continue

            obs["training_time"] = t1

            # runtime
            if check_onnxruntime:
                obs['ort_version'] = ort_version
                try:
                    meth = getattr(inst, method)
                except AttributeError as e:
                    if debug:
                        raise
                    raise AttributeError(
                        "Unable to get method '{}' for model "
                        "'{}'.".format(method, model.__class__)) from e
                try:
                    ypred, t4 = _measure_time(lambda: meth(X_test))
                except ValueError as e:
                    obs['prediction_exc'] = str(e)
                    yield obs
                    continue
                obs['prediction_time'] = t4

            # converting
            for opset in opsets:
                obs_op = obs.copy()
                if opset is not None:
                    obs_op['opset'] = opset
                fct = lambda: convert_sklearn(inst, initial_types=init_types,  # noqa
                                              target_opset=opset)
                try:
                    conv, t2 = _measure_time(fct)
                    obs_op["convert_time"] = t2
                except RuntimeError as e:
                    if debug:
                        raise
                    obs_op["convert_exc"] = e
                    yield obs_op
                    continue

                # opset_domain
                for op_imp in list(conv.opset_import):
                    obs_op['domain_opset_%s' % op_imp.domain] = op_imp.version

                # prediction
                if check_onnxruntime:
                    ser, t5 = _measure_time(lambda: conv.SerializeToString())
                    obs_op['tostring_time'] = t5

                    # load
                    try:
                        sess, t6 = _measure_time(lambda: InferenceSession(ser))
                        obs_op['tostring_time'] = t6
                    except RuntimeError as e:
                        if debug:
                            raise
                        obs_op['ort_load_exc'] = e
                        yield obs_op
                        continue

                    # compute batch
                    fct = lambda: sess.run(None,  # noqa
                                          {init_types[0][0]: Xort_test})
                    try:
                        opred, t7 = _measure_time(fct)
                        obs_op['ort_run_time_batch'] = t7
                    except (RuntimeError, TypeError) as e:
                        if debug:
                            raise
                        obs_op['ort_run_exc_batch'] = e

                    # difference
                    if 'ort_run_exc_batch' not in obs_op:
                        try:
                            opred = opred[output_index]
                        except IndexError:
                            if debug:
                                raise
                            obs_op['max_abs_diff_batch_exc'] = (
                                "Unable to fetch output {}/{} for model '{}'"
                                "".format(output_index, len(opred),
                                          model.__name__))
                            opred = None
                        if opred is not None:
                            max_abs_diff = _measure_absolute_difference(
                                ypred, opred)
                            if debug and np.isnan(max_abs_diff):
                                raise RuntimeError(
                                    "Unable to compute differences between"
                                    "\n{}\n--------\n{}".format(
                                        ypred, opred))
                            obs_op['max_abs_diff_batch'] = max_abs_diff

                    # compute single
                    fct = lambda: [  # noqa
                            sess.run(None, {init_types[0][0]: Xort_row})
                            for Xort_row in Xort_test
                    ]
                    try:
                        opred, t7 = _measure_time(fct)
                        obs_op['ort_run_time_single'] = t7
                    except (RuntimeError, TypeError) as e:
                        if debug:
                            raise
                        obs_op['ort_run_exc_single'] = e

                    # difference
                    if 'ort_run_exc_single' not in obs_op:
                        try:
                            opred = [o[output_index] for o in opred]
                        except IndexError:
                            if debug:
                                raise
                            obs_op['max_abs_diff_exc_single'] = (
                                "Unable to fetch output {}/{} for model '{}'"
                                "".format(output_index, len(opred),
                                          model.__name__))
                            opred = None
                        if opred is not None:
                            max_abs_diff = _measure_absolute_difference(
                                ypred, opred)
                            if debug and np.isnan(max_abs_diff):
                                raise RuntimeError(
                                    "Unable to compute differences between"
                                    "\n{}\n--------\n{}".format(
                                        ypred, opred))
                            obs_op['max_abs_diff_single'] = max_abs_diff

                    if debug:
                        import pprint
                        pprint.pprint(obs_op)
                    yield obs_op


def validate_operator_opsets(verbose=0, opset_min=1, opset_max=None,
                             check_onnxruntime=True, debug=None):
    """
    Tests all possible configuration for all possible
    operators and returns the results.
    """
    ops = [_ for _ in sklearn_operators() if _['supported']]

    if debug is not None:
        ops_ = [_ for _ in ops if _['name'] in debug]
        if len(ops) == 0:
            raise ValueError("Debug is wrong: {}\n{}".format(
                debug, ops[0]))
        ops = ops_

    if verbose > 0:
        try:
            from tqdm import tqdm
            loop = tqdm(ops)
        except ImportError:

            def iterate():
                for i, row in enumerate(ops):
                    print("{}/{} - {}".format(i + 1, len(ops), row))
                    yield row

            loop = iterate()
    else:
        loop = ops

    current_opset = get_opset_number_from_onnx()
    rows = []
    for row in loop:

        model = row['cl']

        for obs in enumerate_compatible_opset(
                model, opset_min=opset_min, opset_max=opset_max,
                check_onnxruntime=check_onnxruntime,
                debug=debug is not None):
            if verbose > 1:
                print("  ", obs)
            diff = obs.get('max_abs_diff_batch',
                           obs.get('max_abs_diff_single', None))
            batch = 'max_abs_diff_batch' in obs and diff is not None
            op1 = obs.get('domain_opset_', '')
            op2 = obs.get('domain_opset_ai.onnx.ml', '')
            op = '{}|{}'.format(op1, op2)
            if diff is not None:
                if diff < 1e-5:
                    obs['available'] = 'OK'
                elif diff < 0.01:
                    obs['available'] = 'e<0.01'
                elif diff < 0.1:
                    obs['available'] = 'e<0.1'
                else:
                    obs['available'] = 'ERROR'
                obs['available'] += '-' + op
                if not batch:
                    obs['available'] += "-NOBATCH"
            elif 'opset' in obs and obs['opset'] == current_opset:
                obs["available"] = 'ERROR'
            obs.update(row)
            rows.append(obs)

    return rows
