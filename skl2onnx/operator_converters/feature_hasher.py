# SPDX-License-Identifier: Apache-2.0

import numpy as np
from onnx import TensorProto
from onnx.helper import make_tensor
from onnx.numpy_helper import from_array
from ..common._registration import register_converter
from ..common._topology import Scope, Operator
from ..common._container import ModelComponentContainer


def convert_sklearn_feature_hasher(
    scope: Scope, operator: Operator, container: ModelComponentContainer
):
    out = operator.outputs
    op = operator.raw_operator
    if op.input_type != "string":
        raise RuntimeError(
            f"The converter for FeatureHasher only supports "
            f"input_type='string' not {op.input_type!r}."
        )

    # If option separator is not None, the converter assumes the input
    # is one string column, each element is a list of strings concatenated
    # with this separator.
    options = container.get_options(op, dict(separator=None))
    separator = options.get("separator", None)

    if separator is not None:
        # Let's split the columns
        delimiter = scope.get_unique_variable_name("delimiter")
        container.add_initializer(
            delimiter, TensorProto.STRING, [], [separator.encode("utf-8")]
        )
        empty_string = scope.get_unique_variable_name("empty_string")
        container.add_initializer(
            empty_string, TensorProto.STRING, [], ["".encode("utf-8")]
        )
        skip_empty = scope.get_unique_variable_name("delimiter")
        container.add_initializer(skip_empty, TensorProto.BOOL, [], [False])
        flat_shape = scope.get_unique_variable_name("flat_shape")
        container.add_initializer(flat_shape, TensorProto.INT64, [1], [-1])
        zero = scope.get_unique_variable_name("zero")
        container.add_initializer(zero, TensorProto.INT64, [1], [0])
        one = scope.get_unique_variable_name("one")
        container.add_initializer(one, TensorProto.INT64, [1], [1])

        to_concat = []
        for i, col_to_split in enumerate(operator.inputs):
            reshaped = scope.get_unique_variable_name(f"reshaped{i}")
            container.add_node(
                "Reshape", [col_to_split.full_name, flat_shape], [reshaped]
            )
            out_indices = scope.get_unique_variable_name(f"out_indices{i}")
            out_text = scope.get_unique_variable_name(f"out_text{i}")
            out_shape = scope.get_unique_variable_name(f"out_shape{i}")
            if len(separator) <= 1:
                container.add_node(
                    "StringSplit",
                    [reshaped, delimiter, skip_empty],
                    [out_indices, out_text, out_shape],
                    op_domain="ai.onnx.contrib",
                    op_version=1,
                )
            else:
                raise RuntimeError(
                    f"Only one character separators are supported but delimiter is {separator!r}."
                )
            shape = scope.get_unique_variable_name(f"shape{i}")
            container.add_node("Shape", [col_to_split.full_name], [shape])

            emptyi = scope.get_unique_variable_name(f"emptyi{i}")
            container.add_node(
                "ConstantOfShape",
                [out_shape],
                [emptyi],
                value=from_array(np.array([0], dtype=np.int64)),
            )
            emptyb = scope.get_unique_variable_name(f"emptyb{i}")
            container.add_node("Cast", [emptyi], [emptyb], to=TensorProto.BOOL)
            emptys = scope.get_unique_variable_name(f"emptys{i}")
            container.add_node("Where", [emptyb, empty_string, empty_string], [emptys])
            flat_split = scope.get_unique_variable_name(f"flat_split{i}")
            container.add_node(
                "ScatterND", [emptys, out_indices, out_text], [flat_split]
            )
            # shape_1 = scope.get_unique_variable_name(f"shape_1{i}")
            # container.add_node("Concat", [shape, flat_shape], [shape_1], axis=0)

            split = scope.get_unique_variable_name(f"split{i}")
            to_concat.append(split)
            # container.add_node("Reshape", [flat_split, shape_1], [split])
            container.add_node("Identity", [flat_split], [split])
        if len(to_concat) == 1:
            input_hasher = to_concat[0]
        else:
            input_hasher = scope.get_unique_variable_name("concatenated")
            container.add_node("Concat", to_concat, [input_hasher], axis=1)
    elif len(operator.inputs) == 1:
        X = operator.inputs[0]
        input_hasher = X.full_name
    else:
        raise RuntimeError(
            f"Only one input is expected but received "
            f"{[i.name for i in operator.inputs]}."
        )

    hashed_ = scope.get_unique_variable_name("hashed_")
    container.add_node(
        "MurmurHash3",
        input_hasher,
        hashed_,
        positive=0,
        seed=0,
        op_domain="com.microsoft",
        op_version=1,
    )
    hashed = scope.get_unique_variable_name("hashed")
    container.add_node("Cast", hashed_, hashed, to=TensorProto.INT64)

    if op.dtype in (np.float32, np.float64, np.int64):
        cst_neg = -1
    else:
        cst_neg = 4294967295

    infinite = scope.get_unique_variable_name("infinite")
    container.add_initializer(infinite, TensorProto.INT64, [1], [-2147483648])

    infinite2 = scope.get_unique_variable_name("infinite2")
    container.add_initializer(infinite2, TensorProto.INT64, [1], [cst_neg])

    infinite_n = scope.get_unique_variable_name("infinite_n")
    container.add_initializer(
        infinite_n, TensorProto.INT64, [1], [2147483647 - (op.n_features - 1)]
    )

    zero = scope.get_unique_variable_name("zero")
    container.add_initializer(zero, TensorProto.INT64, [1], [0])

    one = scope.get_unique_variable_name("one")
    container.add_initializer(one, TensorProto.INT64, [1], [1])

    mone = scope.get_unique_variable_name("mone")
    container.add_initializer(mone, TensorProto.INT64, [1], [-1])

    mtwo = scope.get_unique_variable_name("mtwo")
    container.add_initializer(mtwo, TensorProto.INT64, [1], [-2])

    nf = scope.get_unique_variable_name("nf")
    container.add_initializer(nf, TensorProto.INT64, [1], [op.n_features])

    new_shape = scope.get_unique_variable_name("new_shape")
    container.add_initializer(new_shape, TensorProto.INT64, [2], [-1, 1])

    # values
    if op.alternate_sign:
        cmp = scope.get_unique_variable_name("cmp")
        container.add_node("GreaterOrEqual", [hashed, zero], cmp)
        values = scope.get_unique_variable_name("values")
        container.add_node("Where", [cmp, one, infinite2], values)
    else:
        mul = scope.get_unique_variable_name("mul")
        container.add_node("Mul", [hashed, zero], mul)
        values = scope.get_unique_variable_name("values")
        container.add_node("Add", [mul, one], values)

    values_reshaped = scope.get_unique_variable_name("values_reshaped")
    container.add_node("Reshape", [values, new_shape], values_reshaped)

    # indices
    cmp = scope.get_unique_variable_name("cmp_ind")
    container.add_node("Equal", [hashed, infinite], cmp)
    values_abs = scope.get_unique_variable_name("values_abs")
    container.add_node("Abs", hashed, values_abs)
    values_ind = scope.get_unique_variable_name("values_ind")
    container.add_node("Where", [cmp, infinite_n, values_abs], values_ind)
    indices = scope.get_unique_variable_name("indices")
    container.add_node("Mod", [values_ind, nf], indices)
    indices_reshaped = scope.get_unique_variable_name("indices_reshaped")
    container.add_node("Reshape", [indices, new_shape], indices_reshaped)

    # scatter
    zerot_ = scope.get_unique_variable_name("zerot_")
    container.add_node(
        "ConstantOfShape",
        [nf],
        zerot_,
        value=make_tensor("value", TensorProto.INT64, [1], [0]),
    )
    zerot = scope.get_unique_variable_name("zerot")
    container.add_node("Mul", [indices_reshaped, zerot_], zerot)

    final = scope.get_unique_variable_name("final")
    container.add_node(
        "ScatterElements", [zerot, indices_reshaped, values_reshaped], final, axis=1
    )

    # at this point, every string has been processed as if it were in
    # in a single columns.
    # in case there is more than one column, we need to reduce over
    # the last dimension
    input_shape = scope.get_unique_variable_name("input_shape")
    container.add_node("Shape", input_hasher, input_shape)
    shape_not_last = scope.get_unique_variable_name("shape_not_last")
    container.add_node("Slice", [input_shape, zero, mone], shape_not_last)
    final_shape = scope.get_unique_variable_name("final_last")
    container.add_node("Concat", [shape_not_last, mone, nf], final_shape, axis=0)
    final_reshaped = scope.get_unique_variable_name("final_reshaped")
    container.add_node("Reshape", [final, final_shape], final_reshaped)

    if op.dtype == np.float32:
        to = TensorProto.FLOAT
    elif op.dtype == np.float64:
        to = TensorProto.DOUBLE
    elif op.dtype in (np.int32, np.uint32, np.int64):
        to = TensorProto.INT64
    else:
        raise RuntimeError(
            f"Converter is not implemented for FeatureHasher.dtype={op.dtype}."
        )
    final_reshaped_cast = scope.get_unique_variable_name("final_reshaped_cast")
    container.add_node("Cast", [final_reshaped], final_reshaped_cast, to=to)
    container.add_node(
        "ReduceSum", [final_reshaped_cast, mtwo], out[0].full_name, keepdims=0
    )


register_converter(
    "SklearnFeatureHasher", convert_sklearn_feature_hasher, options={"separator": None}
)
