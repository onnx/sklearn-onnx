# SPDX-License-Identifier: Apache-2.0

import pprint
from collections import OrderedDict
import hashlib
import numpy as np
from onnx.numpy_helper import from_array
from .data_types import TensorType

_unique_index = {"subgraph": 0}


def sklearn_installed():
    """
    Checks that *scikit-learn* is available.
    """
    try:
        import sklearn  # noqa: F401

        return True
    except ImportError:
        return False


def get_unique_subgraph():
    "Returns a unique identifier integer for subgraph."
    global _unique_index
    _unique_index["subgraph"] += 1
    return _unique_index["subgraph"]


def get_producer():
    """
    Internal helper function to return the producer
    """
    from .. import __producer__

    return __producer__


def get_producer_version():
    """
    Internal helper function to return the producer version
    """
    from .. import __producer_version__

    return __producer_version__


def get_domain():
    """
    Internal helper function to return the model domain
    """
    from .. import __domain__

    return __domain__


def get_model_version():
    """
    Internal helper function to return the model version
    """
    from .. import __model_version__

    return __model_version__


def get_column_index(i, inputs):
    """
    Returns a tuples (variable index, column index in that variable).
    The function has two different behaviours, one when *i* (column index)
    is an integer, another one when *i* is a string (column name).
    If *i* is a string, the function looks for input name with
    this name and returns (index, 0).
    If *i* is an integer, let's assume first we have two inputs
    *I0 = FloatTensorType([None, 2])* and *I1 = FloatTensorType([None, 3])*,
    in this case, here are the results:

    ::

        get_column_index(0, inputs) -> (0, 0)
        get_column_index(1, inputs) -> (0, 1)
        get_column_index(2, inputs) -> (1, 0)
        get_column_index(3, inputs) -> (1, 1)
        get_column_index(4, inputs) -> (1, 2)
    """
    if isinstance(i, int):
        if i == 0:
            # Useful shortcut, skips the case when end is None
            # (unknown dimension)
            return 0, 0
        vi = 0
        pos = 0
        end = inputs[0].type.shape[1] if isinstance(inputs[0].type, TensorType) else 1
        if end is None:
            raise RuntimeError(
                "Cannot extract a specific column {0} when "
                "one input ('{1}') has unknown "
                "dimension.".format(i, inputs[0])
            )
        while True:
            if pos <= i < end:
                return (vi, i - pos)
            vi += 1
            pos = end
            if vi >= len(inputs):
                raise RuntimeError(
                    "Input {} (i={}, end={}) is not available in\n{}".format(
                        vi, i, end, pprint.pformat(inputs)
                    )
                )
            rel_end = (
                inputs[vi].type.shape[1]
                if isinstance(inputs[vi].type, TensorType)
                else 1
            )
            if rel_end is None:
                raise RuntimeError(
                    "Cannot extract a specific column {0} when "
                    "one input ('{1}') has unknown "
                    "dimension.".format(i, inputs[vi])
                )
            end += rel_end
    else:
        for ind, inp in enumerate(inputs):
            if inp.raw_name == i:
                return ind, 0
        raise RuntimeError(
            "Unable to find column name %r among names %r. "
            "Make sure the input names specified with parameter "
            "initial_types fits the column names specified in the "
            "pipeline to convert. This may happen because a "
            "ColumnTransformer follows a transformer without "
            "any mapped converter in a pipeline." % (i, [n.raw_name for n in inputs])
        )


def get_column_indices(indices, inputs, multiple):
    """
    Returns the requested graph inpudes based on their
    indices or names. See :func:`get_column_index`.

    :param indices: variables indices or names
    :param inputs: graph inputs
    :param multiple: allows column to come from multiple variables
    :return: a tuple *(variable name, list of requested indices)* if
        *multiple* is False, a dictionary *{ var_index: [ list of
        requested indices ] }*
        if *multiple* is True
    """
    if multiple:
        res = OrderedDict()
        for p in indices:
            ov, onnx_i = get_column_index(p, inputs)
            if ov not in res:
                res[ov] = []
            res[ov].append(onnx_i)
        return res
    else:
        onnx_var = None
        onnx_is = []
        for p in indices:
            ov, onnx_i = get_column_index(p, inputs)
            onnx_is.append(onnx_i)
            if onnx_var is None:
                onnx_var = ov
            elif onnx_var != ov:
                cols = [onnx_var, ov]
                raise NotImplementedError(
                    "sklearn-onnx is not able to merge multiple columns from "
                    "multiple variables ({0}). You should think about merging "
                    "initial types.".format(cols)
                )
        return onnx_var, onnx_is


def hash_array(value, length=15):
    "Computes a hash identifying the value."
    try:
        onx = from_array(value)
    except (AttributeError, TypeError) as e:
        # sparse matrix for example
        if hasattr(value, "tocoo"):
            coo = value.tocoo()
            arrs = [coo.data, coo.row, coo.col, np.array(coo.shape)]
            m = hashlib.sha256()
            for arr in arrs:
                m.update(from_array(arr).SerializeToString())
            return m.hexdigest()[:length]

        raise ValueError(
            "Unable to compute hash for type %r (value=%r)." % (type(value), value)
        ) from e
    except RuntimeError as ee:
        # cannot be serialized
        if isinstance(value, (np.ndarray, list)):
            b = str(value).encode("utf-8")
            m = hashlib.sha256()
            m.update(b)
            return m.hexdigest()[:length]
        raise RuntimeError(
            "Unable to convert value type %r, (value=%r)." % (type(value), value)
        ) from ee

    m = hashlib.sha256()
    m.update(onx.SerializeToString())
    return m.hexdigest()[:length]


def check_input_and_output_numbers(
    operator, input_count_range=None, output_count_range=None
):
    """
    Check if the number of input(s)/output(s) is correct

    :param operator: A Operator object
    :param input_count_range: A list of two integers or an integer.
        If it's a list the first/second element is the
        minimal/maximal number of inputs. If it's an integer,
        it is equivalent to specify that number twice in a list. For
        infinite ranges like 5 to infinity, you need to use [5, None].
    :param output_count_range: A list of two integers or an integer.
        See input_count_range for its format.
    """
    if isinstance(input_count_range, list):
        min_input_count = input_count_range[0]
        max_input_count = input_count_range[1]
    elif isinstance(input_count_range, int) or input_count_range is None:
        min_input_count = input_count_range
        max_input_count = input_count_range
    else:
        raise RuntimeError("input_count_range must be a list or an integer")

    if isinstance(output_count_range, list):
        min_output_count = output_count_range[0]
        max_output_count = output_count_range[1]
    elif isinstance(output_count_range, int) or output_count_range is None:
        min_output_count = output_count_range
        max_output_count = output_count_range
    else:
        raise RuntimeError("output_count_range must be a list or an integer")

    if min_input_count is not None and len(operator.inputs) < min_input_count:
        raise RuntimeError(
            (
                "For operator %s (type: %s), at least %s input(s) is(are) "
                "required but we got %s input(s) which are %s"
            )
            % (
                operator.full_name,
                operator.type,
                min_input_count,
                len(operator.inputs),
                operator.input_full_names,
            )
        )

    if max_input_count is not None and len(operator.inputs) > max_input_count:
        raise RuntimeError(
            (
                "For operator %s (type: %s), at most %s input(s) is(are) "
                "supported but we got %s input(s) which are %s"
            )
            % (
                operator.full_name,
                operator.type,
                max_input_count,
                len(operator.inputs),
                operator.input_full_names,
            )
        )

    if min_output_count is not None and len(operator.outputs) < min_output_count:
        raise RuntimeError(
            (
                "For operator %s (type: %s), at least %s output(s) "
                "is(are) produced but we got %s output(s) which are %s"
            )
            % (
                operator.full_name,
                operator.type,
                min_output_count,
                len(operator.outputs),
                operator.output_full_names,
            )
        )

    if max_output_count is not None and len(operator.outputs) > max_output_count:
        raise RuntimeError(
            (
                "For operator %s (type: %s), at most %s outputs(s) "
                "is(are) supported but we got %s output(s) which are %s"
            )
            % (
                operator.full_name,
                operator.type,
                max_output_count,
                len(operator.outputs),
                operator.output_full_names,
            )
        )


def check_input_and_output_types(
    operator, good_input_types=None, good_output_types=None
):
    """
    Check if the type(s) of input(s)/output(s) is(are) correct

    :param operator: A Operator object
    :param good_input_types: A list of allowed input types
        (e.g., [FloatTensorType, Int64TensorType]) or None. None
        means that we skip the check of the input types.
    :param good_output_types: A list of allowed output types.
        See good_input_types for its format.
    """
    if good_input_types is not None:
        for variable in operator.inputs:
            if type(variable.type) not in good_input_types:
                raise RuntimeError(
                    (
                        "Operator %s (type: %s) got an input %s "
                        "with a wrong type %s. Only %s are allowed"
                    )
                    % (
                        operator.full_name,
                        operator.type,
                        variable.full_name,
                        type(variable.type),
                        good_input_types,
                    )
                )

    if good_output_types is not None:
        for variable in operator.outputs:
            if type(variable.type) not in good_output_types:
                raise RuntimeError(
                    (
                        "Operator %s (type: %s) got an output %s "
                        "with a wrong type %s. Only %s are allowed"
                    )
                    % (
                        operator.full_name,
                        operator.type,
                        variable.full_name,
                        type(variable.type),
                        good_output_types,
                    )
                )
