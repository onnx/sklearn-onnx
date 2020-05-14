# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
import os
from time import perf_counter
import pickle
import numpy
import pandas
import onnx
from sklearn import __all__ as sklearn__all__, __version__ as sklearn_version
from sklearn.utils.testing import all_estimators
from sklearn.model_selection import train_test_split
from onnxruntime import InferenceSession
import onnxruntime.capi.onnxruntime_pybind11_state as OrtErr
from . import __version__ as ort_version
from .convert import to_onnx
from ._validate_problems import _problems, find_suitable_problem
from ._validate_scenarios import _extra_parameters
from .helpers.onnx_helper import to_dot


def get_opset_number_from_onnx():
    """
    Retuns the current *onnx* opset
    based on the installed version of *onnx*.
    """
    return onnx.defs.onnx_opset_version()


def sklearn_operators(subfolder=None):
    """
    Builds the list of operators from *scikit-learn*.
    The function goes through the list of submodule
    and get the list of class which inherit from
    *BaseEstimator*.

    :param subfolder: look into only one subfolder
    """
    estims = all_estimators()

    found = []
    for clname, cl in estims:
        if cl.__name__ in {'Pipeline', 'ColumnTransformer',
                           'FeatureUnion', 'BaseEstimator'}:
            continue
        spl = cl.__module__.split('.')
        sub = spl[-1]
        if sub not in sklearn__all__:
            sub = spl[-2]
        if sub not in sklearn__all__:
            continue
        if (sub in {'calibration', 'dummy', 'manifold'} and
                'Calibrated' not in clname):
            continue
        if subfolder is not None and sub != subfolder:
            continue
        found.append(dict(name=cl.__name__, subfolder=sub, cl=cl))

    if subfolder is None:
        found.sort(key=lambda t: (t['subfolder'], t['name']))
    return found


def _measure_time(fct):
    """
    Measures the execution time for a function.
    """
    begin = perf_counter()
    res = fct()
    end = perf_counter()
    return res, end - begin


def _measure_absolute_difference(skl_pred, ort_pred):
    """
    *Measures the differences between predictions
    between two ways of computing them.
    The functions returns nan if shapes are different.
    """
    ort_pred_ = ort_pred
    if isinstance(ort_pred, list):
        if isinstance(ort_pred[0], dict):
            ort_pred = pandas.DataFrame(ort_pred).values
        elif (isinstance(ort_pred[0], list) and
                isinstance(ort_pred[0][0], dict)):
            if len(ort_pred) == 1:
                ort_pred = pandas.DataFrame(ort_pred[0]).values
            elif len(ort_pred[0]) == 1:
                ort_pred = pandas.DataFrame([o[0] for o in ort_pred]).values
            else:
                raise RuntimeError("Unable to compute differences between"
                                   "\n{}--------\n{}".format(
                                       skl_pred, ort_pred))
        else:
            ort_pred = numpy.array(ort_pred)

    if hasattr(skl_pred, 'todense'):
        skl_pred = skl_pred.todense()
    if hasattr(ort_pred, 'todense'):
        ort_pred = ort_pred.todense()

    if isinstance(ort_pred, list):
        raise RuntimeError("Issue with {}\n{}".format(ort_pred, ort_pred_))

    if skl_pred.shape != ort_pred.shape and skl_pred.size == ort_pred.size:
        ort_pred = ort_pred.ravel()
        skl_pred = skl_pred.ravel()

    if skl_pred.shape != ort_pred.shape:
        return 1e9

    diff = numpy.max(numpy.abs(skl_pred.ravel() - ort_pred.ravel()))

    if numpy.isnan(diff):
        raise RuntimeError("Unable to compute differences between {}-{}\n{}\n"
                           "--------\n{}".format(
                               skl_pred.shape, ort_pred.shape,
                               skl_pred, ort_pred))
    return diff


def _shape_exc(obj):
    if hasattr(obj, 'shape'):
        return obj.shape
    if isinstance(obj, (list, dict, tuple)):
        return "[{%d}]" % len(obj)
    return None


def dump_into_folder(dump_folder, obs_op=None, **kwargs):
    """
    Dumps information when an error was detected
    using *pickle*.
    """
    parts = (obs_op['name'], obs_op['scenario'],
             obs_op['problem'], obs_op.get('opset', '-'))
    name = "dump-ERROR-{}.pkl".format("-".join(map(str, parts)))
    name = os.path.join(dump_folder, name)
    kwargs.update({'obs_op': obs_op})
    with open(name, "wb") as f:
        pickle.dump(kwargs, f)


def enumerate_compatible_opset(model, opset_min=9, opset_max=None,
                               check_runtime=True, debug=False,
                               runtime='CPU', dump_folder=None,
                               store_models=False,
                               dot_graph=False, fLOG=print):
    """
    Lists all compatiable opsets for a specific model.

    :param model: operator class
    :param opset_min: starts with this opset
    :param opset_max: ends with this opset (None to use
        current onnx opset)
    :param check_runtime: checks that runtime can consume the
        model and compute predictions
    :param debug: catch exception (True) or not (False)
    :param runtime: test a specific runtime, by default ``'CPU'``
    :param dump_folder: dump information to replicate in case of mismatch
    :param store_models: if True, the function
        also stores the fitted model and its conversion
        into *ONNX*
    :param dot_graph: generate a DOT graph for every ONNX model
    :param fLOG: logging function
    :return: dictionaries, each row has the following
        keys: opset, exception if any, conversion time,
        problem chosen to test the conversion...

    The function requires *sklearn-onnx*.
    The outcome can be seen at page about :ref:`l-onnx-pyrun`.
    """
    try:
        problems = find_suitable_problem(model)
    except RuntimeError as e:
        yield {'name': model.__name__, 'skl_version': sklearn_version,
               '_0problem_exc': e}
        problems = []

    extras = _extra_parameters.get(model, [('default', {})])

    if opset_max is None:
        opset_max = get_opset_number_from_onnx()
    opsets = list(range(opset_min, opset_max + 1))
    opsets.append(None)

    if extras is None:
        problems = []
        yield {'name': model.__name__, 'skl_version': sklearn_version,
               '_0problem_exc': 'SKIPPED'}

    for prob in problems:
        X_, y_, init_types, method, output_index, Xort_ = _problems[prob]()
        if y_ is None:
            (X_train, X_test, Xort_train,  # pylint: disable=W0612
                Xort_test) = train_test_split(
                    X_, Xort_, random_state=42)
        else:
            (X_train, X_test, y_train, y_test,  # pylint: disable=W0612
                Xort_train, Xort_test) = train_test_split(
                    X_, y_, Xort_, random_state=42)

        for scenario, extra in extras:

            # training
            obs = {'scenario': scenario, 'name': model.__name__,
                   'skl_version': sklearn_version, 'problem': prob,
                   'method': method, 'output_index': output_index}
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
            except (AttributeError, TypeError, ValueError, IndexError) as e:
                if debug:
                    raise
                obs["_1training_time_exc"] = str(e)
                yield obs
                continue

            obs["training_time"] = t1
            if store_models:
                obs['MODEL'] = inst
                obs['X_test'] = X_test
                obs['Xort_test'] = Xort_test
                obs['init_types'] = init_types

            # runtime
            if check_runtime:

                # compute sklearn prediction
                obs['ort_version'] = ort_version
                try:
                    meth = getattr(inst, method)
                except AttributeError as e:
                    if debug:
                        raise
                    obs['_2skl_meth_exc'] = str(e)
                    yield obs
                    continue
                try:
                    ypred, t4 = _measure_time(lambda: meth(X_test))
                except (ValueError, AttributeError, TypeError) as e:
                    if debug:
                        raise
                    obs['_3prediction_exc'] = str(e)
                    yield obs
                    continue
                obs['prediction_time'] = t4

            # converting
            for opset in opsets:
                obs_op = obs.copy()
                if opset is not None:
                    obs_op['opset'] = opset

                if len(init_types) != 1:
                    raise NotImplementedError("Multiple types are "
                                              "is not implemented: "
                                              "{}.".format(init_types))

                def fct_skl(itt=inst, it=init_types[0][1], ops=opset):  # noqa
                    return to_onnx(itt, it, target_opset=ops)

                try:
                    conv, t2 = _measure_time(fct_skl)
                    obs_op["convert_time"] = t2
                except RuntimeError as e:
                    if debug:
                        raise
                    obs_op["_4convert_exc"] = e
                    yield obs_op
                    continue

                if dot_graph and "_4convert_exc" not in obs_op:
                    obs_op['DOT'] = to_dot(conv)

                if store_models:
                    obs_op['ONNX'] = conv

                # opset_domain
                for op_imp in list(conv.opset_import):
                    obs_op['domain_opset_%s' % op_imp.domain] = op_imp.version

                # prediction
                if check_runtime:
                    yield _call_runtime(obs_op=obs_op, conv=conv,
                                        opset=opset, debug=debug,
                                        runtime=runtime, inst=inst,
                                        X_=X_, y_=y_,
                                        init_types=init_types, method=method,
                                        output_index=output_index, Xort_=Xort_,
                                        ypred=ypred, Xort_test=Xort_test,
                                        model=model, dump_folder=dump_folder)
                else:
                    yield obs_op


def _call_runtime(obs_op, conv, opset, debug, inst, runtime,
                  X_, y_, init_types, method, output_index,
                  Xort_, ypred, Xort_test, model, dump_folder):
    """
    Private.
    """
    ser, t5 = _measure_time(lambda: conv.SerializeToString())
    obs_op['tostring_time'] = t5

    # load
    try:
        sess, t6 = _measure_time(
            lambda: InferenceSession(ser))
        obs_op['tostring_time'] = t6
    except (RuntimeError, ValueError, OrtErr.NotImplemented,
            OrtErr.Fail, OrtErr.InvalidGraph) as e:
        if debug:
            raise
        obs_op['_5ort_load_exc'] = e
        return obs_op

    # compute batch
    def fct_batch(se=sess, xo=Xort_test, it=init_types):  # noqa
        return se.run(None, {it[0][0]: xo})
    try:
        opred, t7 = _measure_time(fct_batch)
        obs_op['ort_run_time_batch'] = t7
    except (RuntimeError, TypeError, ValueError, KeyError) as e:
        if debug:
            raise
        obs_op['_6ort_run_batch_exc'] = e

    # difference
    if '_6ort_run_batch_exc' not in obs_op:
        if isinstance(opred, dict):
            ch = [(k, v) for k, v in sorted(opred.items())]
            # names = [_[0] for _ in ch]
            opred = [_[1] for _ in ch]

        try:
            opred = opred[output_index]
        except IndexError:
            if debug:
                raise
            obs_op['_8max_abs_diff_batch_exc'] = (
                "Unable to fetch output {}/{} for model '{}'"
                "".format(output_index, len(opred),
                          model.__name__))
            opred = None

        debug_exc = []
        if opred is not None:
            max_abs_diff = _measure_absolute_difference(
                ypred, opred)
            if numpy.isnan(max_abs_diff):
                obs_op['_8max_abs_diff_batch_exc'] = (
                    "Unable to compute differences between"
                    " {}-{}\n{}\n--------\n{}".format(
                        _shape_exc(
                            ypred), _shape_exc(opred),
                        ypred, opred))
                if debug:
                    debug_exc.append(RuntimeError(
                        obs_op['_8max_abs_diff_batch_exc']))
            else:
                obs_op['max_abs_diff_batch'] = max_abs_diff
                if dump_folder and max_abs_diff > 1e-5:
                    dump_into_folder(dump_folder, kind='batch', obs_op=obs_op,
                                     X_=X_, y_=y_, init_types=init_types,
                                     method=init_types,
                                     output_index=output_index,
                                     Xort_=Xort_)

    if debug and len(debug_exc) == 2:
        raise debug_exc[0]
    if debug:
        import pprint
        pprint.pprint(obs_op)
    return obs_op


def enumerate_validated_operator_opsets(verbose=0, opset_min=1, opset_max=None,
                                        check_runtime=True, debug=False,
                                        runtime='onnxruntime',
                                        models=None, dump_folder=None,
                                        store_models=False, dot_graph=False,
                                        fLOG=print):
    """
    Tests all possible configuration for all possible
    operators and returns the results.

    :param verbose: integer 0, 1, 2
    :param opset_min: checks conversion starting from the opset
    :param opset_max: checks conversion up to this opset,
        None means @see fn get_opset_number_from_onnx.
    :param check_runtime: checks the python runtime
    :param models: only process a small list of operators,
        set of model names
    :param debug: stops whenever an exception
        is raised
    :param runtime: test a specific runtime, by default ``'CPU'``
    :param dump_folder: dump information to replicate in case of mismatch
    :param store_models: if True, the function
        also stores the fitted model and its conversion
        into *onnx*
    :param dot_graph: generate a DOT graph for every ONNX model
    :param fLOG: logging function
    :return: list of dictionaries

    The function is available through command line
    :ref:`validate_runtime <l-cmd-validate_runtime>`.
    """
    ops = [_ for _ in sklearn_operators()]

    if models is not None:
        if not all(map(lambda m: isinstance(m, str), models)):
            raise ValueError("models must be a set of strings.")
        ops_ = [_ for _ in ops if _['name'] in models]
        if len(ops) == 0:
            raise ValueError("Parameter models is wrong: {}\n{}".format(
                models, ops[0]))
        ops = ops_

    if verbose > 0:
        try:
            from tqdm import tqdm
            loop = tqdm(ops)
        except ImportError:

            def iterate():
                for i, row in enumerate(ops):
                    fLOG("{}/{} - {}".format(i + 1, len(ops), row))
                    yield row

            loop = iterate()
    else:
        loop = ops

    current_opset = get_opset_number_from_onnx()
    for row in loop:

        model = row['cl']

        for obs in enumerate_compatible_opset(
                model, opset_min=opset_min, opset_max=opset_max,
                check_runtime=check_runtime, runtime=runtime,
                debug=debug, dump_folder=dump_folder,
                store_models=store_models, dot_graph=dot_graph,
                fLOG=fLOG):

            if verbose > 1:
                fLOG("  ", obs)
            elif verbose > 0 and "_0problem_exc" in obs:
                fLOG("  ???", obs)

            diff = obs.get('max_abs_diff_batch',
                           obs.get('max_abs_diff_single', None))
            batch = 'max_abs_diff_batch' in obs and diff is not None

            if diff is not None:
                if diff < 1e-5:
                    obs['available'] = 'OK'
                elif diff < 0.0001:
                    obs['available'] = 'e<0.0001'
                elif diff < 0.001:
                    obs['available'] = 'e<0.001'
                elif diff < 0.01:
                    obs['available'] = 'e<0.01'
                elif diff < 0.1:
                    obs['available'] = 'e<0.1'
                else:
                    obs['available'] = "ERROR->=%1.1f" % diff
                if not batch:
                    obs['available'] += "-NOBATCH"

            else:
                excs = []
                for k, v in sorted(obs.items()):
                    if k.endswith('_exc'):
                        excs.append((k, v))
                        break
                if 'opset' not in obs:
                    # It fails before the conversion happens.
                    obs['opset'] = current_opset
                if obs['opset'] == current_opset:
                    if len(excs) > 0:
                        k, v = excs[0]
                        obs['available'] = 'ERROR-%s' % k
                        obs['available-ERROR'] = v
                    else:
                        obs['available'] = 'ERROR-?'

            obs.update(row)
            yield obs


def summary_report(df):
    """
    Finalizes the results computed by function
    @see fn enumerate_validated_operator_opsets.

    :param df: dataframe
    :return: pivoted dataframe
    """

    def aggfunc(values):
        if len(values) != 1:
            vals = set(values)
            if len(vals) != 1:
                return " // ".join(map(str, values))
        val = values.iloc[0]
        if isinstance(val, float) and numpy.isnan(val):
            return ""
        else:
            return val

    if "_1training_time_exc" in df.columns:
        df = df[df["_1training_time_exc"].isnull()]
    if '_2skl_meth_exc' in df.columns:
        df = df[df["_2skl_meth_exc"].isnull()]
    piv = pandas.pivot_table(df, values="available",
                             index=['name', 'problem', 'scenario'],
                             columns='opset',
                             aggfunc=aggfunc).reset_index(drop=False)

    opmin = min(df['opset'].dropna())
    versions = ["opset%d" % (opmin + t - 1)
                for t in range(1, piv.shape[1] - 2)]
    indices = ["name", "problem", "scenario"]
    piv.columns = indices + versions
    piv = piv[indices + list(reversed(versions))].copy()
    new_col = "Opset"
    piv[new_col] = ""
    piv["Comment"] = ""
    poscol = {name: i for i, name in enumerate(piv.columns)}

    # simplification
    for i in range(piv.shape[0]):
        last = None
        for col in versions:
            val = piv.iloc[i, poscol[col]]
            if 'OK' == val:
                piv.iloc[i, poscol[new_col]] = col.replace('opset', '') + '+'
                break
            elif 'OK-NOBATCH' == val:
                piv.iloc[i, poscol[new_col]] = col.replace('opset', '') + '+'
                piv.iloc[i, poscol['Comment']] = "No batch prediction"
                break
            elif isinstance(val, str) and val.startswith("e"):
                piv.iloc[i, poscol[new_col]] = col.replace('opset', '') + '+'
                piv.iloc[i, poscol['Comment']] = (
                    "Still discrepancies " + val[1:])
                break
            elif isinstance(val, str) and val.startswith("ERR"):
                piv.iloc[i, poscol[new_col]] = col.replace('opset', '') + '+'
                piv.iloc[i, poscol['Comment']] = (
                    "Still significant discrepancies")
                break
            elif isinstance(val, str):
                last = val
        if val != 'OK' and not val.startswith('e'):
            piv.iloc[i, poscol[new_col]] = last

    piv = piv.drop(versions, axis=1)

    if "available-ERROR" in df.columns:

        from skl2onnx.common.exceptions import MissingShapeCalculator

        def replace_msg(text):
            if isinstance(text, MissingShapeCalculator):
                return "Not supported yet"
            if str(text).startswith("Unable to find a shape "
                                    "calculator for type '"):
                return "Not supported yet"
            return str(text)

        piv2 = pandas.pivot_table(df, values="available-ERROR",
                                  index=['name', 'problem', 'scenario'],
                                  columns='opset',
                                  aggfunc=aggfunc).reset_index(drop=False)

        col = piv2.iloc[:, piv2.shape[1] - 1]
        piv["ERROR-msg"] = col.apply(replace_msg)
        poscol = {name: i for i, name in enumerate(piv.columns)}
        for i in range(piv.shape[0]):
            err = piv.iloc[i, poscol['ERROR-msg']]
            if isinstance(err, str) and err != '':
                piv.iloc[i, poscol['Comment']] = err
                piv.iloc[i, poscol[new_col]] = ''
        piv = piv.drop('ERROR-msg', axis=1)

    return piv
