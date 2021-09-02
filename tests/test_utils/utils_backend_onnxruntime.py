# SPDX-License-Identifier: Apache-2.0

"""
Helpers to test runtimes.
"""
import numpy
import pandas
import warnings
import onnx as onnx_package
from skl2onnx.helpers.onnx_helper import (
    select_model_inputs_outputs, enumerate_model_node_outputs,
    enumerate_model_initializers)
from skl2onnx.algebra.type_helper import _guess_type

from .utils_backend import (
    load_data_and_model,
    extract_options,
    ExpectedAssertionError,
    OnnxRuntimeAssertionError,
    OnnxRuntimeMissingNewOnnxOperatorException,
    compare_outputs)


def _display_intermediate_steps(model_onnx, inputs, disable_optimisation):
    import onnxruntime
    print("[_display_intermediate_steps] BEGIN")
    if isinstance(model_onnx, str):
        import onnx
        model_onnx = onnx.load(model_onnx)

    for name, node in enumerate_model_initializers(model_onnx, add_node=True):
        print("INIT: {} - {}".format(name, _guess_type(node)))

    for out, node in enumerate_model_node_outputs(model_onnx, add_node=True):
        print('-')
        print("OUTPUT: {} from {}".format(out, node.name))
        step = select_model_inputs_outputs(model_onnx, out)
        if (disable_optimisation and
                hasattr(onnxruntime, 'GraphOptimizationLevel')):
            opts = onnxruntime.SessionOptions()
            opts.graph_optimization_level = (
                onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL)
        else:
            opts = None
        try:
            step_sess = onnxruntime.InferenceSession(
                step.SerializeToString(), sess_options=opts)
        except Exception as e:
            raise RuntimeError("Unable to load ONNX model with onnxruntime. "
                               "Last added node is:\n{}".format(node)) from e
        for o in step_sess.get_inputs():
            print("IN :", o)
        for o in step_sess.get_outputs():
            print("OUT: ", o)
        if inputs:
            res = step_sess.run(inputs)
            print(res)
    print("[_display_intermediate_steps] END")


def compare_runtime(test,
                    decimal=5,
                    options=None,
                    verbose=False,
                    context=None,
                    comparable_outputs=None,
                    intermediate_steps=False,
                    classes=None,
                    disable_optimisation=False):
    """
    The function compares the expected output (computed with
    the model before being converted to ONNX) and the ONNX output
    produced with module *onnxruntime*.

    :param test: dictionary with the following keys:
        - *onnx*: onnx model (filename or object)
        - *expected*: expected output (filename pkl or object)
        - *data*: input data (filename pkl or object)
    :param decimal: precision of the comparison
    :param options: comparison options
    :param context: specifies custom operators
    :param verbose: in case of error, the function may print
        more information on the standard output
    :param comparable_outputs: compare only these outputs
    :param intermediate_steps: displays intermediate steps
        in case of an error
    :param classes: classes names (if option 'nocl' is used)
    :param disable_optimisation: disable optimisation onnxruntime
        could do
    :return: tuple (outut, lambda function to run the predictions)

    The function does not return anything but raises an error
    if the comparison failed.
    """
    lambda_onnx = None
    if context is None:
        context = {}
    load = load_data_and_model(test, **context)
    if verbose:
        print("[compare_runtime] test '{}' loaded".format(test['onnx']))

    onx = test['onnx']
    if options is None:
        if isinstance(onx, str):
            options = extract_options(onx)
        else:
            options = {}
    elif options is None:
        options = {}
    elif not isinstance(options, dict):
        raise TypeError("options must be a dictionary.")

    try:
        import onnxruntime
    except ImportError:
        warnings.warn("Unable to import onnxruntime.")
        return None

    if verbose:
        print("[compare_runtime] InferenceSession('{}')".format(onx))

    if (disable_optimisation and
            hasattr(onnxruntime, 'GraphOptimizationLevel')):
        opts = onnxruntime.SessionOptions()
        opts.graph_optimization_level = (
            onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL)
    else:
        opts = None

    try:
        sess = onnxruntime.InferenceSession(onx, sess_options=opts)
    except ExpectedAssertionError as expe:
        raise expe
    except Exception as e:
        if "CannotLoad" in options:
            raise ExpectedAssertionError(
                "Unable to load onnx '{0}' due to\n{1}".format(onx, e))
        else:
            if intermediate_steps:
                _display_intermediate_steps(onx, None, disable_optimisation)
            if verbose:
                import onnx
                model = onnx.load(onx)
                smodel = "\nJSON ONNX\n" + str(model)
            else:
                smodel = ""
            if ("NOT_IMPLEMENTED : Could not find an implementation "
                    "for the node" in str(e)):
                # onnxruntime does not implement a specific node yet.
                raise OnnxRuntimeMissingNewOnnxOperatorException(
                    "onnxruntime does not implement a new operator "
                    "'{0}'\n{1}\nONNX\n{2}".format(
                        onx, e, smodel))
            if "is not a registered function/op" in str(e):
                content = onnx_package.load(onx)
                raise OnnxRuntimeAssertionError(
                    "Missing op? '{0}'\nONNX\n{1}\n{2}\n---\n{3}".format(
                        onx, smodel, e, content))
            raise OnnxRuntimeAssertionError(
                "Unable to load onnx '{0}'\nONNX\n{1}\n{2}".format(
                    onx, smodel, e))

    input = load["data"]
    DF = options.pop('DF', False)
    if DF:
        inputs = {c: input[c].values for c in input.columns}
        for k in inputs:
            if inputs[k].dtype == numpy.float64:
                inputs[k] = inputs[k].astype(numpy.float32)
            inputs[k] = inputs[k].reshape((inputs[k].shape[0], 1))
    else:
        if isinstance(input, dict):
            inputs = input
        elif isinstance(input, (list, numpy.ndarray, pandas.DataFrame)):
            inp = sess.get_inputs()
            if len(inp) == len(input):
                inputs = {i.name: v for i, v in zip(inp, input)}
            elif len(inp) == 1:
                inputs = {inp[0].name: input}
            elif isinstance(input, numpy.ndarray):
                shape = sum(i.shape[1] if len(i.shape) == 2 else i.shape[0]
                            for i in inp)
                if shape == input.shape[1]:
                    inputs = {n.name: input[:, i] for i, n in enumerate(inp)}
                else:
                    raise OnnxRuntimeAssertionError(
                        "Wrong number of inputs onnx {0} != "
                        "original shape {1}, onnx='{2}'"
                        .format(len(inp), input.shape, onx))
            elif isinstance(input, list):
                try:
                    array_input = numpy.array(input)
                except Exception:
                    raise OnnxRuntimeAssertionError(
                        "Wrong number of inputs onnx {0} != "
                        "original {1}, onnx='{2}'"
                        .format(len(inp), len(input), onx))
                shape = sum(i.shape[1] for i in inp)
                if shape == array_input.shape[1]:
                    inputs = {}
                    c = 0
                    for i, n in enumerate(inp):
                        d = c + n.shape[1]
                        inputs[n.name] = _create_column(
                            [row[c:d] for row in input], n.type)
                        c = d
                else:
                    raise OnnxRuntimeAssertionError(
                        "Wrong number of inputs onnx {0} != "
                        "original shape {1}, onnx='{2}'*"
                        .format(len(inp), array_input.shape, onx))
            elif isinstance(input, pandas.DataFrame):
                try:
                    array_input = numpy.array(input)
                except Exception:
                    raise OnnxRuntimeAssertionError(
                        "Wrong number of inputs onnx {0} != "
                        "original {1}, onnx='{2}'"
                        .format(len(inp), len(input), onx))
                shape = sum(i.shape[1] for i in inp)
                if shape == array_input.shape[1]:
                    inputs = {}
                    c = 0
                    for i, n in enumerate(inp):
                        d = c + n.shape[1]
                        inputs[n.name] = _create_column(
                            input.iloc[:, c:d], n.type)
                        c = d
                else:
                    raise OnnxRuntimeAssertionError(
                        "Wrong number of inputs onnx {0}={1} columns != "
                        "original shape {2}, onnx='{3}'*"
                        .format(len(inp), shape, array_input.shape, onx))
            else:
                raise OnnxRuntimeAssertionError(
                    "Wrong type of inputs onnx {0}, onnx='{1}'".format(
                        type(input), onx))
        else:
            raise OnnxRuntimeAssertionError(
                "Dict or list is expected, not {0}".format(type(input)))

        for k in inputs:
            if isinstance(inputs[k], list):
                inputs[k] = numpy.array(inputs[k])

    OneOff = options.pop('OneOff', False)
    OneOffArray = options.pop('OneOffArray', False)
    options.pop('SklCol', False)  # unused here but in dump_data_and_model
    if OneOff or OneOffArray:
        if verbose:
            print(
                "[compare_runtime] OneOff: type(inputs)={} "
                "len={} OneOffArray={}"
                .format(type(input), len(inputs), OneOffArray))
        if len(inputs) == 1 and not OneOffArray:
            name, values = list(inputs.items())[0]
            res = []
            for input in values:
                try:
                    one = sess.run(None, {name: input})
                    if lambda_onnx is None:
                        def lambda_onnx(): return sess.run(None, {name: input})  # noqa
                    if verbose:
                        import pprint
                        pprint.pprint(one)
                except ExpectedAssertionError as expe:
                    raise expe
                except Exception as e:
                    if intermediate_steps:
                        _display_intermediate_steps(
                            onx, {name: input}, disable_optimisation)
                    raise OnnxRuntimeAssertionError(
                        "Unable to run onnx '{0}' due to {1}".format(onx, e))
                res.append(one)
            if verbose:
                print("[compare_runtime] OneOff: _post_process_output1")
            output = _post_process_output(res)
        else:

            def to_array(vv):
                if isinstance(
                        vv, (numpy.ndarray, numpy.int64, numpy.float32, str)):
                    return numpy.array([vv])
                else:
                    return numpy.array([vv], dtype=numpy.float32)

            t = list(inputs.items())[0]
            res = []
            for i in range(0, len(t[1])):
                iii = {k: to_array(v[i]) for k, v in inputs.items()}
                try:
                    one = sess.run(None, iii)
                    if lambda_onnx is None:
                        def lambda_onnx(): return sess.run(None, iii)  # noqa
                    if verbose:
                        import pprint
                        pprint.pprint(one)
                except ExpectedAssertionError as expe:
                    raise expe
                except Exception as e:
                    if intermediate_steps:
                        _display_intermediate_steps(
                            onx, iii, disable_optimisation)
                    if verbose:
                        import onnx
                        model = onnx.load(onx)
                        smodel = "\nJSON ONNX\n" + str(model)
                    else:
                        smodel = ""
                    raise OnnxRuntimeAssertionError(
                        "Unable to run onnx '{0}' due to {1}{2}".format(
                            onx, e, smodel))
                res.append(one)
            if verbose:
                print("[compare_runtime] OneOff: _post_process_output2")
            output = _post_process_output(res)

            if OneOffArray:
                if isinstance(output, list):
                    pass
                elif not isinstance(output, numpy.ndarray):
                    raise TypeError("output must be an array, not {}".format(
                        type(output)))
                else:
                    output = [output]
    else:
        if verbose:
            print("[compare_runtime] type(inputs)={} len={} names={}".format(
                type(input), len(inputs), list(sorted(inputs))))
        if verbose:
            run_options = onnxruntime.RunOptions()
            if hasattr(run_options, 'run_log_verbosity_level'):
                run_options.run_log_verbosity_level = 5
            else:
                run_options.log_verbosity_level = 5
        else:
            run_options = None
        try:
            output = sess.run(None, inputs, run_options)
            def lambda_onnx(): return sess.run(None, inputs)  # noqa
            if verbose:
                import pprint
                pprint.pprint(output)
        except ExpectedAssertionError as expe:
            raise expe
        except RuntimeError as e:
            if intermediate_steps:
                _display_intermediate_steps(onx, inputs, disable_optimisation)
            if "-Fail" in onx:
                raise ExpectedAssertionError(
                    "onnxruntime cannot compute the prediction for '{0}'".
                    format(onx))
            else:
                if verbose:
                    import onnx
                    model = onnx.load(onx)
                    smodel = "\nJSON ONNX\n" + str(model)
                else:
                    smodel = ""
                raise OnnxRuntimeAssertionError(
                    "onnxruntime cannot compute the prediction"
                    " for '{0}' due to {1}{2}"
                    .format(onx, e, smodel))
        except Exception as e:
            raise OnnxRuntimeAssertionError(
                "Unable to run onnx '{0}' due to {1}".format(onx, e))
        if verbose:
            print("[compare_runtime] done type={}".format(type(output)))

    output0 = output.copy()

    if comparable_outputs:
        cmp_exp = [load["expected"][o] for o in comparable_outputs]
        cmp_out = [output[o] for o in comparable_outputs]
    else:
        cmp_exp = load["expected"]
        cmp_out = output

    try:
        _compare_expected(cmp_exp,
                          cmp_out,
                          sess,
                          onx,
                          decimal=decimal,
                          verbose=verbose,
                          classes=classes,
                          **options)
    except ExpectedAssertionError as expe:
        raise expe
    except Exception as e:
        if verbose:
            import onnx
            model = onnx.load(onx)
            smodel = "\nJSON ONNX\n" + str(model)
        else:
            smodel = ""
        raise OnnxRuntimeAssertionError(
            "Model '{0}' has discrepencies.\n{1}: {2}{3}".format(
                onx, type(e), e, smodel))

    return output0, lambda_onnx


def _post_process_output(res):
    """
    Applies post processings before running the comparison
    such as changing type from list to arrays.
    """
    if isinstance(res, list):
        if len(res) == 0:
            return res
        elif len(res) == 1:
            return _post_process_output(res[0])
        elif isinstance(res[0], numpy.ndarray):
            return numpy.array(res)
        elif isinstance(res[0], dict):
            import pandas
            return pandas.DataFrame(res).values
        else:
            ls = [len(r) for r in res]
            mi = min(ls)
            if mi != max(ls):
                raise NotImplementedError(
                    "Unable to postprocess various number of "
                    "outputs in [{0}, {1}]"
                    .format(min(ls), max(ls)))
            if mi > 1:
                output = []
                for i in range(mi):
                    output.append(_post_process_output([r[i] for r in res]))
                return output
            elif isinstance(res[0], list):
                # list of lists
                if isinstance(res[0][0], list):
                    return numpy.array(res)
                elif len(res[0]) == 1 and isinstance(res[0][0], dict):
                    return _post_process_output([r[0] for r in res])
                elif len(res) == 1:
                    return res
                else:
                    if len(res[0]) != 1:
                        raise NotImplementedError(
                            "Not conversion implemented for {0}".format(res))
                    st = [r[0] for r in res]
                    return numpy.vstack(st)
            else:
                return res
    else:
        return res


def _create_column(values, dtype):
    "Creates a column from values with dtype"
    if str(dtype) == "tensor(int64)":
        return numpy.array(values, dtype=numpy.int64)
    if str(dtype) == "tensor(float)":
        return numpy.array(values, dtype=numpy.float32)
    if str(dtype) == "tensor(string)":
        return numpy.array(values, dtype=numpy.str_)
    raise OnnxRuntimeAssertionError(
        "Unable to create one column from dtype '{0}'".format(dtype))


def _compare_expected(expected,
                      output,
                      sess,
                      onnx,
                      decimal=5,
                      verbose=False,
                      classes=None,
                      **kwargs):
    """
    Compares the expected output against the runtime outputs.
    This is specific to *onnxruntime* due to variable *sess*
    of type *onnxruntime.InferenceSession*.
    """
    tested = 0
    if isinstance(expected, list):
        if isinstance(output, list):
            if 'Out0' in kwargs:
                expected = expected[:1]
                output = output[:1]
                del kwargs['Out0']
            elif 'Out1' in kwargs:
                expected = expected[1:2]
                output = output[1:2]
                del kwargs['Out1']
            if 'Reshape' in kwargs:
                del kwargs['Reshape']
                output = numpy.hstack(output).ravel()
                output = output.reshape(
                    (len(expected), len(output.ravel()) // len(expected)))
            if len(expected) != len(output):
                raise OnnxRuntimeAssertionError(
                    "Unexpected number of outputs '{0}', expected={1}, got={2}"
                    .format(onnx, len(expected), len(output)))
            for exp, out in zip(expected, output):
                _compare_expected(exp,
                                  out,
                                  sess,
                                  onnx,
                                  decimal=5,
                                  verbose=verbose,
                                  classes=classes,
                                  **kwargs)
                tested += 1
        else:
            raise OnnxRuntimeAssertionError(
                "Type mismatch for '{0}', output type is {1}".format(
                    onnx, type(output)))
    elif isinstance(expected, dict):
        if not isinstance(output, dict):
            raise OnnxRuntimeAssertionError(
                "Type mismatch for '{0}'".format(onnx))
        for k, v in output.items():
            if k not in expected:
                continue
            msg = compare_outputs(expected[k],
                                  v,
                                  decimal=decimal,
                                  verbose=verbose,
                                  **kwargs)
            if msg:
                raise OnnxRuntimeAssertionError(
                    "Unexpected output '{0}' in model '{1}'\n{2}".format(
                        k, onnx, msg))
            tested += 1
    elif isinstance(expected, numpy.ndarray):
        if isinstance(output, list):
            if expected.shape[0] == len(output) and isinstance(
                    output[0], dict):
                import pandas
                output = pandas.DataFrame(output)
                output = output[list(sorted(output.columns))]
                output = output.values
        if isinstance(output, (dict, list)):
            if len(output) != 1:
                ex = str(output)
                if len(ex) > 170:
                    ex = ex[:170] + "..."
                raise OnnxRuntimeAssertionError(
                    "More than one output when 1 is expected "
                    "for onnx '{0}'\n{1}"
                    .format(onnx, ex))
            output = output[-1]
        if not isinstance(output, numpy.ndarray):
            raise OnnxRuntimeAssertionError(
                "output must be an array for onnx '{0}' not {1}".format(
                    onnx, type(output)))
        if (classes is not None and (
                expected.dtype == numpy.str or expected.dtype.char == 'U')):
            try:
                output = numpy.array([classes[cl] for cl in output])
            except IndexError as e:
                raise RuntimeError('Unable to handle\n{}\n{}\n{}'.format(
                    expected, output, classes)) from e
        msg = compare_outputs(expected,
                              output,
                              decimal=decimal,
                              verbose=verbose,
                              **kwargs)
        if isinstance(msg, ExpectedAssertionError):
            raise msg
        if msg:
            raise OnnxRuntimeAssertionError(
                "Unexpected output in model '{0}'\n{1}".format(onnx, msg))
        tested += 1
    else:
        from scipy.sparse.csr import csr_matrix
        if isinstance(expected, csr_matrix):
            # DictVectorizer
            one_array = numpy.array(output)
            dense = numpy.asarray(expected.todense())
            msg = compare_outputs(dense,
                                  one_array,
                                  decimal=decimal,
                                  verbose=verbose,
                                  **kwargs)
            if msg:
                raise OnnxRuntimeAssertionError(
                    "Unexpected output in model '{0}'\n{1}".format(onnx, msg))
            tested += 1
        else:
            raise OnnxRuntimeAssertionError(
                "Unexpected type for expected output ({1}) and onnx '{0}'".
                format(onnx, type(expected)))
    if tested == 0:
        raise OnnxRuntimeAssertionError("No test for onnx '{0}'".format(onnx))
