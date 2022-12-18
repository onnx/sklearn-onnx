# SPDX-License-Identifier: Apache-2.0

"""
Helpers to test runtimes.
"""
import numpy as np
from scipy.special import expit  # noqa
import pandas
import onnx as onnx_package
from onnx.defs import onnx_opset_version
try:
    from onnx.helper import tensor_dtype_to_string
except ImportError:
    tensor_dtype_to_string = None
from skl2onnx.helpers.onnx_helper import (
    select_model_inputs_outputs,
    enumerate_model_node_outputs,
    enumerate_model_initializers)
from skl2onnx.algebra.type_helper import _guess_type
from scipy.spatial.distance import cdist
from .utils_backend import (
    load_data_and_model,
    extract_options,
    ExpectedAssertionError,
    OnnxRuntimeAssertionError,
    OnnxRuntimeMissingNewOnnxOperatorException,
    compare_outputs)


if onnx_opset_version() >= 18:
    from onnx.reference import ReferenceEvaluator
    from onnx.reference.op_run import OpRun
    from onnx.reference.ops._op import OpRunReduceNumpy
    from onnx.reference.ops import load_op
    from .reference_implementation_text import Tokenizer

    class CDist(OpRun):
        op_domain = "com.microsoft"

        def _run(self, x, y, metric="euclidean"):
            return (cdist(x, y, metric=metric).astype(x.dtype),)

    additional_implementations = [CDist, Tokenizer]

    try:
        load_op("ai.onnx.ml", "Scaler")
        add_ops = False
    except Exception:
        add_ops = True

    if add_ops:
        # bugs in reference implementation not covered by a backend test
        from onnx.reference.ops.op_argmin import ArgMin_12 as _ArgMin, _argmin
        from onnx.reference.ops.op_argmax import ArgMax_12 as _ArgMax, _argmax
        from onnx.reference.ops.op_reduce_log_sum_exp import (
            compute_log_sum_exp)
        from .reference_implementation_ml import (
            Binarizer,
            DictVectorizer,
            FeatureVectorizer,
            FusedMatMul,
            Imputer,
            LabelEncoder,
            LinearClassifier,
            LinearRegressor,
            Normalizer,
            OneHotEncoder,
            Scaler,
        )
        from .reference_implementation_zipmap import ZipMap
        from .reference_implementation_afe import ArrayFeatureExtractor
        from .reference_implementation_tree import (
            TreeEnsembleClassifier,
            TreeEnsembleRegressor)
        from .reference_implementation_svm import (
            SVMClassifier,
            SVMRegressor)
        from .reference_implementation_text import TfIdfVectorizer

        class ArgMin(_ArgMin):
            def _run(self, data, axis=None, keepdims=None,
                     select_last_index=None):
                if select_last_index == 0:
                    if keepdims == 0:
                        return _ArgMin._run(
                            self, data, axis=axis, keepdims=keepdims,
                            select_last_index=0)
                    return (_argmin(data, axis=axis, keepdims=keepdims),)
                raise NotImplementedError("Unused in sklearn-onnx.")

        class ArgMax(_ArgMax):
            def _run(self, data, axis=None, keepdims=None,
                     select_last_index=None):
                if select_last_index == 0:
                    if keepdims == 0:
                        return _ArgMax._run(
                            self, data, axis=axis, keepdims=keepdims,
                            select_last_index=0)
                    return (_argmax(data, axis=axis, keepdims=keepdims),)
                raise NotImplementedError("Unused in sklearn-onnx.")

        class ReduceLogSumExp_1(OpRunReduceNumpy):
            def _run(self, data, axes=None, keepdims=None):  # type: ignore
                tax = tuple(axes) if axes else None
                return compute_log_sum_exp(data, tax, keepdims)

        class ReduceL2_18(OpRunReduceNumpy):
            def _run(self, data, axes=None, keepdims=1,
                     noop_with_empty_axes=0):
                if self.is_axes_empty(axes) and noop_with_empty_axes:
                    return (data,)

                axes = self.handle_axes(axes)
                keepdims = keepdims != 0  # type: ignore
                return (
                    np.sqrt(np.sum(np.square(data), axis=axes,
                            keepdims=keepdims)).astype(
                            dtype=data.dtype))

        class ConstantOfShape(OpRun):
            def __init__(self, onnx_node, run_params):  # type: ignore
                OpRun.__init__(self, onnx_node, run_params)
                self.cst = (
                    self.value[0] if isinstance(self.value, np.ndarray)
                    else self.value)
                if isinstance(self.cst, int):
                    self.cst = np.int64(self.cst)
                elif isinstance(self.cst, float):
                    self.cst = np.float64(self.cst)
                elif self.cst is None:
                    self.cst = np.float32(0)
                if not isinstance(
                    self.cst, (np.float32, np.float64, np.int64,
                               np.int32, np.bool_, np.float16)):
                    raise TypeError(f"cst must be a real not {type(self.cst)}")

            def _run(self, data, value=None):
                try:
                    res = np.full(tuple(data), self.cst)
                except TypeError as e:
                    raise RuntimeError(
                        f"Unable to create a constant of shape {data!r} "
                        f"with value {self.cst!r} "
                        f"(raw value={value!r}).") from e
                return (res,)

        class Where(OpRun):
            def _run(self, condition, x, y):  # type: ignore
                if (x.dtype != y.dtype and
                    x.dtype not in (np.object_,) and
                    not (x.dtype.type is np.str_ and
                         y.dtype.type is np.str_)):
                    raise RuntimeError(
                        f"x and y should share the same dtype "
                        f"{x.dtype} != {y.dtype}")
                return (np.where(condition, x, y).astype(x.dtype),)

        additional_implementations.extend([
            # ai.onnx
            ArgMax,
            ArgMin,
            ConstantOfShape,
            ReduceL2_18,
            ReduceLogSumExp_1,
            Where,
            # ai.onnx.ml
            ArrayFeatureExtractor,
            Binarizer,
            DictVectorizer,
            FeatureVectorizer,
            FusedMatMul,
            Imputer,
            LabelEncoder,
            LinearClassifier,
            LinearRegressor,
            Normalizer,
            OneHotEncoder,
            TfIdfVectorizer,
            TreeEnsembleClassifier,
            TreeEnsembleRegressor,
            Scaler,
            SVMClassifier,
            SVMRegressor,
            ZipMap,
        ])

        class ReferenceEvaluatorEx(ReferenceEvaluator):
            def __init__(self, *args, new_ops=None, **kwargs):
                if new_ops is None:
                    new_ops = additional_implementations
                else:
                    new_ops = new_ops + additional_implementations
                super().__init__(*args, new_ops=new_ops, **kwargs)

            def get_inputs(self):
                res = [InputDef(n, list(get_shape(t, True)), get_type(t))
                       for n, t in zip(self.input_names,
                                       self.input_types)]
                return res

            def get_outputs(self):
                res = [InputDef(n, list(get_shape(t, True)), get_type(t))
                       for n, t in zip(self.output_names,
                                       self.output_types)]
                return res


def _display_intermediate_steps(model_onnx, inputs, disable_optimisation):
    import onnx.reference

    print("[_display_intermediate_steps] BEGIN")
    if isinstance(model_onnx, str):
        import onnx

        model_onnx = onnx.load(model_onnx)

    for name, node in enumerate_model_initializers(model_onnx, add_node=True):
        print("INIT: {} - {}".format(name, _guess_type(node)))

    for out, node in enumerate_model_node_outputs(model_onnx, add_node=True):
        print("-")
        print("OUTPUT: {} from {}".format(out, node.name))
        step = select_model_inputs_outputs(model_onnx, out)
        try:
            step_sess = onnx.reference.ReferenceEvaluator(step)
        except Exception as e:
            raise RuntimeError(
                "Unable to load ONNX model with ReferenceEvaluator. "
                "Last added node is:\n{}".format(node)
            ) from e
        for o in step_sess.get_inputs():
            print("IN :", o)
        for o in step_sess.get_outputs():
            print("OUT: ", o)
        if inputs:
            res = step_sess.run(inputs)
            print(res)
    print("[_display_intermediate_steps] END")


class InputDef:
    def __init__(self, name, shape, dtype):
        self.name = name
        self.shape = shape
        self.type = dtype


def get_shape(t, use_none=False):
    if t.tensor_type:
        dims = [getattr(d, 'dim_value', None)
                for d in t.tensor_type.shape.dim]
        if use_none:
            return tuple(r if r != 0 else None for r in dims)
        return tuple(dims)
    return None


def get_type(t):
    if t.tensor_type and str(t).startswith("tensor_type"):
        if tensor_dtype_to_string is None:
            res = ""
        else:
            res = tensor_dtype_to_string(t.tensor_type.elem_type)
        maps = {
            'TensorProto.STRING': 'tensor(string)',
            'TensorProto.INT64': 'tensor(int64)',
            'TensorProto.INT32': 'tensor(int32)',
            'TensorProto.DOUBLE': 'tensor(double)',
            'TensorProto.FLOAT': 'tensor(float)',
            'TensorProto.BOOL': 'tensor(bool)',
        }
        return maps[res]
    return None


def get_inputs(sess):
    return [InputDef(n, get_shape(t), get_type(t))
            for n, t in zip(sess.input_names,
                            sess.input_types)]


def compare_runtime(
        test,
        decimal=5,
        options=None,
        verbose=0,
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
        print("[compare_runtime] test '{}' loaded".format(test["onnx"]))

    onx = test["onnx"]
    if options is None:
        if isinstance(onx, str):
            options = extract_options(onx)
        else:
            options = {}
    elif not isinstance(options, dict):
        raise TypeError("options must be a dictionary.")

    import onnx.reference

    if verbose:
        print("[compare_runtime] InferenceSession('{}')".format(onx))

    try:
        sess = onnx.reference.ReferenceEvaluator(
            onx, new_ops=additional_implementations, verbose=verbose)
    except ExpectedAssertionError as expe:
        raise expe
    except Exception as e:
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
                "ReferenceEvaluator does not implement a new operator "
                "'{0}'\n{1}\nONNX\n{2}".format(onx, e, smodel))
        if "is not a registered function/op" in str(e):
            content = onnx_package.load(onx)
            raise OnnxRuntimeAssertionError(
                "Missing op? '{0}'\nONNX\n{1}\n{2}\n---\n{3}".format(
                    onx, smodel, e, content))
        raise OnnxRuntimeAssertionError(
            "Unable to load onnx '{0}'\nONNX\n{1}\n{2}"
            ".".format(onx, smodel, e))

    input = load["data"]
    DF = options.pop("DF", False)
    if DF:
        inputs = {c: input[c].values for c in input.columns}
        for k in inputs:
            if inputs[k].dtype == np.float64:
                inputs[k] = inputs[k].astype(np.float32)
            inputs[k] = inputs[k].reshape((inputs[k].shape[0], 1))
    else:
        if isinstance(input, dict):
            inputs = input
        elif isinstance(input, (list, np.ndarray, pandas.DataFrame)):
            inp = get_inputs(sess)
            if len(inp) == len(input):
                inputs = {i.name: v for i, v in zip(inp, input)}
            elif len(inp) == 1:
                inputs = {inp[0].name: input}
            elif isinstance(input, np.ndarray):
                shape = sum(
                    i.shape[1] if len(i.shape) == 2
                    else i.shape[0] for i in inp)
                if shape == input.shape[1]:
                    inputs = {n.name: input[:, i] for i, n in enumerate(inp)}
                else:
                    raise OnnxRuntimeAssertionError(
                        "Wrong number of inputs onnx {0} != "
                        "original shape {1}, onnx='{2}'".format(
                            len(inp), input.shape, onx))
            elif isinstance(input, list):
                try:
                    array_input = np.array(input)
                except Exception:
                    raise OnnxRuntimeAssertionError(
                        "Wrong number of inputs onnx {0} != "
                        "original {1}, onnx='{2}'".format(
                            len(inp), len(input), onx))
                if hasattr(inp[0], 'shape'):
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
                            "original shape {1}, onnx='{2}'*".format(
                                len(inp), array_input.shape, onx))
                else:
                    array_input = array_input.reshape((-1, len(inp)))
                    inputs = {i.name: r for i, r in zip(inp, array_input.T)}
            elif isinstance(input, pandas.DataFrame):
                try:
                    array_input = np.array(input)
                except Exception:
                    raise OnnxRuntimeAssertionError(
                        "Wrong number of inputs onnx {0} != "
                        "original {1}, onnx='{2}'".format(
                            len(inp), len(input), onx
                        )
                    )
                if hasattr(inp[0], 'shape'):
                    shape = sum(i.shape[1] for i in inp)
                    if shape == array_input.shape[1]:
                        inputs = {}
                        c = 0
                        for i, n in enumerate(inp):
                            d = c + n.shape[1]
                            inputs[n.name] = _create_column(
                                input.iloc[:, c:d], n.type
                            )
                            c = d
                    else:
                        raise OnnxRuntimeAssertionError(
                            "Wrong number of inputs onnx {0}={1} columns != "
                            "original shape {2}, onnx='{3}'*".format(
                                len(inp), shape, array_input.shape, onx))
                else:
                    array_input = array_input.reshape((-1, len(inp)))
                    inputs = {i.name: r for i, r in zip(inp, array_input.T)}
            else:
                raise OnnxRuntimeAssertionError(
                    "Wrong type of inputs onnx {0}, onnx='{1}'".format(
                        type(input), onx))
        else:
            raise OnnxRuntimeAssertionError(
                "Dict or list is expected, not {0}".format(type(input)))

        for k in inputs:
            if isinstance(inputs[k], list):
                inputs[k] = np.array(inputs[k])

    OneOff = options.pop("OneOff", False)
    OneOffArray = options.pop("OneOffArray", False)
    options.pop("SklCol", False)  # unused here but in dump_data_and_model
    if OneOff or OneOffArray:
        if verbose:
            print(
                "[compare_runtime] OneOff: type(inputs)={} "
                "len={} OneOffArray={}".format(
                    type(input), len(inputs), OneOffArray))
        if len(inputs) == 1 and not OneOffArray:
            name, values = list(inputs.items())[0]
            res = []
            for input in values:
                try:
                    one = sess.run(None, {name: input})
                    if lambda_onnx is None:
                        lambda_onnx = (
                            lambda sess=sess, input=input: sess.run(  # noqa
                                None, {name: input}))
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
                        vv, (np.ndarray, np.int64, np.float32, str)):
                    return np.array([vv])
                return np.array([vv], dtype=np.float32)

            t = list(inputs.items())[0]
            res = []
            for i in range(0, len(t[1])):
                iii = {k: to_array(v[i]) for k, v in inputs.items()}
                try:
                    one = sess.run(None, iii)
                    if lambda_onnx is None:
                        lambda_onnx = (
                            lambda sess=sess, iii=iii: sess.run(  # noqa
                                None, iii))
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
                elif not isinstance(output, np.ndarray):
                    raise TypeError(
                        "output must be an array, not {}".format(type(output)))
                else:
                    output = [output]
    else:
        if verbose:
            print(
                "[compare_runtime] type(inputs)={} len={} names={}".format(
                    type(input), len(inputs), list(sorted(inputs))))
        try:
            output = sess.run(None, inputs)

            def lambda_onnx():
                return sess.run(None, inputs)  # noqa

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
                    "onnxruntime cannot compute the "
                    "prediction for '{0}'".format(onx))
            else:
                if verbose:
                    import onnx
                    model = onnx.load(onx)
                    smodel = "\nJSON ONNX\n" + str(model)
                else:
                    smodel = ""
                raise OnnxRuntimeAssertionError(
                    "ReferenceEvaluator cannot compute the prediction"
                    " for '{0}' due to {1}{2}".format(onx, e, smodel))
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
        _compare_expected(
            cmp_exp,
            cmp_out,
            sess,
            onx,
            decimal=decimal,
            verbose=verbose,
            classes=classes,
            **options)
    except OnnxRuntimeAssertionError as de:
        if isinstance(onx, str):
            import onnx

            model = onnx.load(onx)
        else:
            model = onx
        opset_version = None
        for imp in model.opset_import:
            if imp.domain == "":
                opset_version = imp.version
        if opset_version is None or opset_version < 15:
            return None, None
        if "support for domain ai.onnx is till opset 17" in str(de):
            return None, None
        raise de
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
            "Model '{0}' has discrepencies with backend="
            "'onnx'.\n{1}: {2}{3}".format(onx, type(e), e, smodel))

    return output0, lambda_onnx


def _post_process_output(res):
    """
    Applies post processings before running the comparison
    such as changing type from list to arrays.
    """
    if isinstance(res, list):
        if len(res) == 0:
            return res
        if len(res) == 1:
            return _post_process_output(res[0])
        if isinstance(res[0], np.ndarray):
            return np.array(res)
        if isinstance(res[0], dict):
            import pandas

            return pandas.DataFrame(res).values
        ls = [len(r) for r in res]
        mi = min(ls)
        if mi != max(ls):
            raise NotImplementedError(
                "Unable to postprocess various number of "
                "outputs in [{0}, {1}]".format(min(ls), max(ls)))
        if mi > 1:
            output = []
            for i in range(mi):
                output.append(_post_process_output([r[i] for r in res]))
            return output
        if isinstance(res[0], list):
            # list of lists
            if isinstance(res[0][0], list):
                return np.array(res)
            if len(res[0]) == 1 and isinstance(res[0][0], dict):
                return _post_process_output([r[0] for r in res])
            if len(res) == 1:
                return res
            if len(res[0]) != 1:
                raise NotImplementedError(
                    "Not conversion implemented for {0}".format(res))
            st = [r[0] for r in res]
            return np.vstack(st)
        return res
    return res


def _create_column(values, dtype):
    "Creates a column from values with dtype"
    if str(dtype) == "tensor(int64)":
        return np.array(values, dtype=np.int64)
    if str(dtype) == "tensor(float)":
        return np.array(values, dtype=np.float32)
    if str(dtype) == "tensor(string)":
        return np.array(values, dtype=np.str_)
    raise OnnxRuntimeAssertionError(
        "Unable to create one column from dtype '{0}'".format(dtype))


def _compare_expected(
        expected,
        output,
        sess,
        onnx,
        decimal=5,
        verbose=False,
        classes=None,
        **kwargs):
    """
    Compares the expected output against the runtime outputs.
    This is specific to *ReferenceEvaluator* due to variable *sess*
    of type *onnx.reference.ReferenceEvaluator*.
    """
    tested = 0
    if isinstance(expected, list):
        if isinstance(output, (list, np.ndarray)):
            if "Out0" in kwargs:
                expected = expected[:1]
                output = output[:1]
                del kwargs["Out0"]
            elif "Out1" in kwargs:
                expected = expected[1:2]
                output = output[1:2]
                del kwargs["Out1"]
            if "Reshape" in kwargs:
                del kwargs["Reshape"]
                output = np.hstack(output).ravel()
                output = output.reshape(
                    (len(expected), len(output.ravel()) // len(expected)))
            if len(expected) != len(output):
                raise OnnxRuntimeAssertionError(
                    "Unexpected number of outputs '{0}', "
                    "expected={1}, got={2}".format(
                        onnx, len(expected), len(output)))
            for exp, out in zip(expected, output):
                _compare_expected(
                    exp,
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
            msg = compare_outputs(
                expected[k], v, decimal=decimal, verbose=verbose, **kwargs
            )
            if msg:
                raise OnnxRuntimeAssertionError(
                    "Unexpected output '{0}' in model '{1}'\n{2}".format(
                        k, onnx, msg))
            tested += 1
    elif isinstance(expected, np.ndarray):
        if isinstance(output, list):
            if (expected.shape[0] == len(output) and
                    isinstance(output[0], dict)):
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
                    "for onnx '{0}'\n{1}".format(onnx, ex))
            output = output[-1]
        if not isinstance(output, np.ndarray):
            raise OnnxRuntimeAssertionError(
                "output must be an array for onnx '{0}' not {1}".format(
                    onnx, type(output)))
        if (classes is not None and (
                expected.dtype == np.str_ or
                expected.dtype.char == "U")):
            try:
                output = np.array([classes[cl] for cl in output])
            except IndexError as e:
                raise RuntimeError(
                    "Unable to handle\n{}\n{}\n{}".format(
                        expected, output, classes)) from e
        msg = compare_outputs(
            expected, output, decimal=decimal, verbose=verbose, **kwargs)
        if isinstance(msg, ExpectedAssertionError):
            raise msg
        if msg:
            raise OnnxRuntimeAssertionError(
                f"Unexpected output in model {onnx}\n{msg}")
        tested += 1
    else:
        from scipy.sparse import csr_matrix

        if isinstance(expected, csr_matrix):
            # DictVectorizer
            one_array = np.array(output)
            dense = np.asarray(expected.todense())
            msg = compare_outputs(
                dense, one_array, decimal=decimal, verbose=verbose, **kwargs)
            if msg:
                raise OnnxRuntimeAssertionError(
                    "Unexpected output in model '{0}'\n{1}".format(onnx, msg))
            tested += 1
        else:
            raise OnnxRuntimeAssertionError(
                "Unexpected type for expected output ({1}) "
                "and onnx '{0}'".format(onnx, type(expected)))
    if tested == 0:
        raise OnnxRuntimeAssertionError("No test for onnx '{0}'".format(onnx))
