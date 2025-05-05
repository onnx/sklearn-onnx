# SPDX-License-Identifier: Apache-2.0

import numpy as np

try:
    from onnx.helper import np_dtype_to_tensor_dtype
except ImportError:
    from onnx.mapping import NP_TYPE_TO_TENSOR_TYPE

    def np_dtype_to_tensor_dtype(dtype):
        return NP_TYPE_TO_TENSOR_TYPE[dtype]


from ..proto import onnx_proto


def _apply_basic_numerical_operation(
    scope, op_type, input_names, output_name, container, operator_name, axis, broadcast
):
    name = _create_name_or_use_existing_one(scope, op_type, operator_name)

    attrs = {}
    if container.target_opset < 7:
        # Before ONNX-1.2 (opset 7), broadcasting behavior is Caffe2-like.
        if axis is not None:
            attrs["axis"] = axis
        if broadcast is not None:
            attrs["broadcast"] = broadcast

        if container.target_opset < 6:
            attrs["consumed_inputs"] = [0, 0]
            op_version = 1
        else:
            op_version = 6
    else:
        # Since ONNX-1.2 (opset 7), broadcasting behavior is Numpy-like,
        # so we don't need to specify any attributes
        op_version = 7

    container.add_node(
        op_type, input_names, output_name, op_version=op_version, name=name, **attrs
    )


def _apply_unary_operation(
    scope, op_type, input_name, output_name, container, operator_name, **attrs
):
    name = _create_name_or_use_existing_one(scope, op_type, operator_name)

    attrs["name"] = name
    if container.target_opset < 6:
        attrs["consumed_inputs"] = [0]
        op_version = 1
    else:
        op_version = 6

    container.add_node(op_type, input_name, output_name, op_version=op_version, **attrs)


def apply_div(
    scope,
    input_names,
    output_name,
    container,
    operator_name=None,
    axis=None,
    broadcast=None,
):
    _apply_basic_numerical_operation(
        scope,
        "Div",
        input_names,
        output_name,
        container,
        operator_name=operator_name,
        axis=axis,
        broadcast=broadcast,
    )


def apply_cast(scope, input_name, output_name, container, operator_name=None, to=None):
    """
    :param to: enum defined in ONNX TensorProto.DataType,
        for example, TensorProto.FLOAT and TensorProto.INT64.
    """
    name = _create_name_or_use_existing_one(scope, "Cast", operator_name)
    attrs = {"name": name}

    d = onnx_proto.TensorProto.DataType.DESCRIPTOR
    allowed_type_name_and_type_enum_pairs = {
        v.number: k for k, v in d.values_by_name.items()
    }
    if to not in allowed_type_name_and_type_enum_pairs:
        raise ValueError(
            'Attribute "to" must be one of %s'
            % allowed_type_name_and_type_enum_pairs.keys()
        )

    if container.target_opset < 9:
        if to in [
            onnx_proto.TensorProto.STRING,
            onnx_proto.TensorProto.COMPLEX64,
            onnx_proto.TensorProto.COMPLEX128,
        ]:
            raise ValueError(
                'Attribute "to" cannot correspond to a String or Complex TensorProto type.'
            )

        if container.target_opset < 6:
            # Convert enum to string, for example, TensorProto.INT64 to 'INT64'
            attrs["to"] = allowed_type_name_and_type_enum_pairs[to]
            op_version = 1
        else:
            # Enum, for example, TensorProto.INT64
            attrs["to"] = to
            op_version = 6
    else:
        # Enum value, for example, TensorProto.INT64
        # String casting is supported in opset 9
        if to in [onnx_proto.TensorProto.COMPLEX64, onnx_proto.TensorProto.COMPLEX128]:
            raise ValueError(
                'Attribute "to" cannot correspond to a Complex TensorProto type.'
            )
        attrs["to"] = to
        op_version = 9

    container.add_node("Cast", input_name, output_name, op_version=op_version, **attrs)


def apply_reshape(
    scope, input_name, output_name, container, operator_name=None, desired_shape=None
):
    if (
        not isinstance(desired_shape, str)
        and len([i for i in desired_shape if i is not None and i < 0]) > 1
    ):
        raise ValueError(
            "There can only be one -1 in the targeted shape of a Reshape but got %s"
            % desired_shape
        )

    name = _create_name_or_use_existing_one(scope, "Reshape", operator_name)

    if container.target_opset < 5:
        container.add_node(
            "Reshape",
            input_name,
            output_name,
            op_version=1,
            name=name,
            shape=desired_shape,
            consumed_inputs=[0],
        )
    else:
        if isinstance(desired_shape, str):
            desired_shape_name = desired_shape
        else:
            desired_shape_name = scope.get_unique_variable_name("shape_tensor")
            container.add_initializer(
                desired_shape_name,
                onnx_proto.TensorProto.INT64,
                [len(desired_shape)],
                desired_shape,
            )

        # Create ONNX Reshape operator
        if isinstance(input_name, list):
            input_name.append(desired_shape_name)
        else:
            input_name = [input_name, desired_shape_name]
        container.add_node("Reshape", input_name, output_name, op_version=5, name=name)


def apply_normalizer(scope, inputs, outputs, container, norm, use_float):
    """
    Adds operator Normalizer if *use_float* is true,
    otherwise, uses *ReduceSum* + *Div*. *Normalizer*
    always produces float according to ONNX speciciations.
    """
    input = inputs[0] if isinstance(inputs, list) else inputs
    output = outputs[0] if isinstance(outputs, list) else outputs
    use_normalizer = container.is_allowed({"Normalizer"})

    if use_normalizer and use_float:
        container.add_node(
            "Normalizer",
            input,
            output,
            op_domain="ai.onnx.ml",
            norm=norm,
            name=scope.get_unique_operator_name("Normalizer"),
        )
    else:
        # Normalizer only produces floats.
        if norm == "L1":
            norm = scope.get_unique_variable_name("norm")
            norm_abs = scope.get_unique_variable_name("norm_abs")
            container.add_node(
                "Abs", input, norm_abs, name=scope.get_unique_operator_name("Abs")
            )

            if container.target_opset < 13:
                container.add_node(
                    "ReduceSum",
                    norm_abs,
                    norm,
                    axes=[1],
                    keepdims=1,
                    name=scope.get_unique_operator_name("ReduceSum"),
                )
            else:
                axis_name = scope.get_unique_variable_name("axis")
                container.add_initializer(
                    axis_name, onnx_proto.TensorProto.INT64, [1], [1]
                )
                container.add_node(
                    "ReduceSum",
                    [norm_abs, axis_name],
                    norm,
                    keepdims=1,
                    name=scope.get_unique_operator_name("ReduceSum"),
                )
            apply_div(
                scope,
                [input, norm],
                output,
                container,
                operator_name=scope.get_unique_operator_name("NormalizerNorm"),
            )
        elif norm == "L2":
            norm = scope.get_unique_variable_name("norm")
            norm2 = scope.get_unique_variable_name("norm2")
            if container.target_opset < 18:
                container.add_node(
                    "ReduceSumSquare",
                    input,
                    norm,
                    axes=[1],
                    keepdims=1,
                    name=scope.get_unique_operator_name("ReduceSumSquare"),
                )
            else:
                axis_name = scope.get_unique_variable_name("axis")
                container.add_initializer(
                    axis_name, onnx_proto.TensorProto.INT64, [1], [1]
                )
                container.add_node(
                    "ReduceSumSquare",
                    [input, axis_name],
                    norm,
                    keepdims=1,
                    name=scope.get_unique_operator_name("ReduceSumSquare"),
                )
            container.add_node(
                "Sqrt", [norm], norm2, name=scope.get_unique_operator_name("Sqrt")
            )
            apply_div(
                scope,
                [input, norm2],
                output,
                container,
                operator_name=scope.get_unique_operator_name("NormalizerNorm"),
            )
        else:
            raise NotImplementedError(
                "Normalization not implemented for norm %r." % norm
            )


def _create_name_or_use_existing_one(scope, op_type, name):
    if name is None:
        return scope.get_unique_operator_name(op_type)
    return name


def apply_clip(
    scope, input_name, output_name, container, operator_name=None, max=None, min=None
):
    name = _create_name_or_use_existing_one(scope, "Clip", operator_name)
    attrs = {"name": name}

    if container.target_opset < 11:
        if max is not None:
            attrs["max"] = float(max)
        if min is not None:
            attrs["min"] = float(min)

        if container.target_opset < 6:
            attrs["consumed_inputs"] = [0]
            op_version = 1
        else:
            op_version = 6

        container.add_node(
            "Clip", input_name, output_name, op_version=op_version, **attrs
        )
    else:
        if container.target_opset < 12:
            op_version = 11
        else:
            op_version = 12
        if min is None and max is not None:
            raise RuntimeError("Operator 'Clip': min must be specified if max is.")
        inputs = [input_name]

        if min is not None:
            if isinstance(
                min,
                (np.ndarray, float, int, np.float32, np.float64, np.int64, np.int32),
            ):
                # add initializer
                if isinstance(min, np.ndarray):
                    if len(min.shape) == 0:
                        min = [min]
                    elif min.shape == (1,):
                        min = list(min[0]) if hasattr(min[0], "__iter__") else list(min)
                    else:
                        raise RuntimeError("min must be an array of one element.")
                else:
                    min = [min]

                # container in sklearn-onnx stores the computation type in
                # container.dtype.
                min_name = scope.get_unique_variable_name("clip_min")
                if op_version < 12:
                    min = np.array(min, dtype=getattr(container, "dtype", np.float32))
                    container.add_initializer(
                        min_name,
                        getattr(container, "proto_dtype", onnx_proto.TensorProto.FLOAT),
                        [],
                        [min[0]],
                    )
                else:
                    min = np.array(min)
                    container.add_initializer(
                        min_name, np_dtype_to_tensor_dtype(min.dtype), [], [min[0]]
                    )
                min = min_name
            if isinstance(min, str):
                inputs.append(min)
            else:
                raise RuntimeError("Parameter 'min' must be a string or a float.")

        if max is not None:
            if min is None:
                raise RuntimeError("Parameter 'min' must be specified if 'max' is.")
            if isinstance(
                max,
                (np.ndarray, float, int, np.float32, np.float64, np.int64, np.int32),
            ):
                # add initializer
                if isinstance(max, np.ndarray):
                    if len(max.shape) == 0:
                        max = [max]
                    elif max.shape == (1,):
                        max = list(max[0]) if hasattr(max[0], "__iter__") else list(max)
                    else:
                        raise RuntimeError("max must be an array of one element.")
                else:
                    max = [max]

                max_name = scope.get_unique_variable_name("clip_max")
                if op_version < 12:
                    max = np.array(max, dtype=getattr(container, "dtype", np.float32))
                    container.add_initializer(
                        max_name,
                        getattr(container, "proto_dtype", onnx_proto.TensorProto.FLOAT),
                        [],
                        [max[0]],
                    )
                else:
                    max = np.array(max)
                    container.add_initializer(
                        max_name, np_dtype_to_tensor_dtype(max.dtype), [], [max[0]]
                    )
                max = max_name
            if isinstance(max, str):
                inputs.append(max)
            else:
                raise RuntimeError("Parameter 'max' must be a string or a float.")

        container.add_node("Clip", inputs, output_name, op_version=op_version, **attrs)


def apply_add(
    scope,
    input_names,
    output_name,
    container,
    operator_name=None,
    axis=None,
    broadcast=None,
):
    _apply_basic_numerical_operation(
        scope,
        "Add",
        input_names,
        output_name,
        container,
        operator_name=operator_name,
        axis=axis,
        broadcast=broadcast,
    )


def apply_concat(
    scope, input_names, output_name, container, operator_name=None, axis=0
):
    name = _create_name_or_use_existing_one(scope, "Concat", operator_name)

    if container.target_opset < 4:
        op_version = 1
    elif container.target_opset < 11:
        op_version = 4
    else:
        op_version = 11

    container.add_node(
        "Concat", input_names, output_name, op_version=op_version, name=name, axis=axis
    )


def apply_exp(scope, input_name, output_name, container, operator_name=None):
    _apply_unary_operation(
        scope, "Exp", input_name, output_name, container, operator_name=operator_name
    )


def apply_mul(
    scope,
    input_names,
    output_name,
    container,
    operator_name=None,
    axis=None,
    broadcast=None,
):
    _apply_basic_numerical_operation(
        scope,
        "Mul",
        input_names,
        output_name,
        container,
        operator_name=operator_name,
        axis=axis,
        broadcast=broadcast,
    )


def apply_sub(
    scope,
    input_names,
    output_name,
    container,
    operator_name=None,
    axis=None,
    broadcast=0,
):
    _apply_basic_numerical_operation(
        scope,
        "Sub",
        input_names,
        output_name,
        container,
        operator_name=operator_name,
        axis=axis,
        broadcast=broadcast,
    )


def apply_topk(scope, input_name, output_names, container, k, operator_name=None):
    name = _create_name_or_use_existing_one(scope, "TopK", operator_name)

    if container.target_opset < 10:
        if isinstance(k, str):
            raise ValueError("topk k cannot be string type before opset 10")
        container.add_node(
            "TopK", input_name, output_names, name=name, k=k, op_version=1
        )
    else:
        if container.target_opset == 10:
            op_version = 10
        else:
            op_version = 11

        if isinstance(k, str):
            k_value_name = k
        else:
            k_value_name = scope.get_unique_variable_name("k_value")
            container.add_initializer(
                k_value_name, onnx_proto.TensorProto.INT64, [1], [k]
            )
        container.add_node(
            "TopK",
            input_name + [k_value_name],
            output_names,
            name=name,
            op_version=op_version,
        )


def apply_transpose(
    scope, input_name, output_name, container, operator_name=None, perm=None
):
    name = _create_name_or_use_existing_one(scope, "Transpose", operator_name)
    container.add_node("Transpose", input_name, output_name, name=name, perm=perm)


def apply_abs(scope, input_name, output_name, container, operator_name=None):
    _apply_unary_operation(
        scope, "Abs", input_name, output_name, container, operator_name=operator_name
    )


def apply_reducesum(
    scope,
    input_name,
    output_name,
    container,
    operator_name=None,
    axes=None,
    keepdims=1,
    rank=0,
):
    name = _create_name_or_use_existing_one(scope, "ReduceSum", operator_name)
    if axes is None:
        axes = []
    if container.target_opset < 13:
        if container.target_opset < 11:
            op_version = 1
            axes = [axis if axis >= 0 else axis + rank for axis in axes]
        else:
            op_version = 11
        container.add_node(
            "ReduceSum",
            input_name,
            output_name,
            name=name,
            op_version=op_version,
            axes=axes,
            keepdims=keepdims,
        )
    else:
        if not isinstance(input_name, list):
            input_name = [input_name]
        op_version = 13
        if isinstance(axes, str):
            container.add_node(
                "ReduceSum",
                input_name + [axes],
                output_name,
                op_version=op_version,
                name=name,
                keepdims=keepdims,
            )
        elif axes is None or len(axes) == 0:
            container.add_node(
                "ReduceSum",
                input_name,
                output_name,
                op_version=op_version,
                name=name,
                keepdims=keepdims,
            )
        else:
            axes_name = scope.get_unique_variable_name(name + "_reducesum")
            container.add_initializer(
                axes_name, onnx_proto.TensorProto.INT64, [len(axes)], axes
            )
            container.add_node(
                "ReduceSum",
                input_name + [axes_name],
                output_name,
                op_version=op_version,
                name=name,
                keepdims=keepdims,
            )


def apply_sqrt(scope, input_name, output_name, container, operator_name=None):
    _apply_unary_operation(
        scope, "Sqrt", input_name, output_name, container, operator_name=operator_name
    )


def apply_identity(scope, input_name, output_name, container, operator_name=None):
    name = _create_name_or_use_existing_one(scope, "Identity", operator_name)
    container.add_node("Identity", input_name, output_name, name=name)


def apply_sigmoid(scope, input_name, output_name, container, operator_name=None):
    _apply_unary_operation(
        scope, "Sigmoid", input_name, output_name, container, operator_name
    )


def apply_softmax(
    scope, input_name, output_name, container, operator_name=None, axis=None
):
    name = _create_name_or_use_existing_one(scope, "Softmax", operator_name)
    if axis is None:
        axis = 1 if container.target_opset < 13 else -1
    container.add_node("Softmax", input_name, output_name, name=name, axis=axis)


def apply_log(scope, input_name, output_name, container, operator_name=None):
    _apply_unary_operation(
        scope, "Log", input_name, output_name, container, operator_name=operator_name
    )


def apply_pow(
    scope,
    input_names,
    output_name,
    container,
    operator_name=None,
    axis=None,
    broadcast=None,
):
    name = _create_name_or_use_existing_one(scope, "Pow", operator_name)

    attrs = {"name": name}
    if container.target_opset < 7:
        # Before ONNX-1.2, broadcasting behavior is Caffe2-like.
        if axis is not None:
            attrs["axis"] = axis
        if broadcast is not None:
            attrs["broadcast"] = broadcast
        op_version = 1
    elif container.target_opset < 12:
        # Since ONNX-1.2, broadcasting behavior is Numpy-like,
        # so we don't need to specify any attributes
        op_version = 7
    else:
        op_version = 12

    container.add_node("Pow", input_names, output_name, op_version=op_version, **attrs)


def apply_normalization(
    scope, input_name, output_name, container, operator_name=None, axis=1, p=2
):
    name = _create_name_or_use_existing_one(scope, "LpNormalization", operator_name)
    container.add_node(
        "LpNormalization", input_name, output_name, name=name, p=p, axis=axis
    )


def apply_slice(
    scope,
    input_name,
    output_name,
    container,
    starts,
    ends,
    axes=None,
    steps=None,
    operator_name=None,
):
    name = _create_name_or_use_existing_one(scope, "Slice", operator_name)

    if container.target_opset < 10:
        if axes is None:
            container.add_node(
                "Slice",
                input_name,
                output_name,
                name=name,
                starts=starts,
                ends=ends,
                op_version=1,
            )
        else:
            container.add_node(
                "Slice",
                input_name,
                output_name,
                name=name,
                starts=starts,
                ends=ends,
                axes=axes,
                op_version=1,
            )
    else:
        if container.target_opset == 10:
            op_version = 10
        else:
            op_version = 11
        inputs = input_name if isinstance(input_name, list) else [input_name]
        if isinstance(starts, str):
            starts_name = starts
        else:
            starts_name = scope.get_unique_variable_name("starts")
            container.add_initializer(
                starts_name, onnx_proto.TensorProto.INT64, [len(starts)], starts
            )

        if isinstance(ends, str):
            ends_name = ends
        else:
            ends_name = scope.get_unique_variable_name("ends")
            container.add_initializer(
                ends_name, onnx_proto.TensorProto.INT64, [len(ends)], ends
            )

        inputs.append(starts_name)
        inputs.append(ends_name)
        if axes:
            if isinstance(axes, str):
                axes_name = axes
            else:
                axes_name = scope.get_unique_variable_name("axes")
                container.add_initializer(
                    axes_name, onnx_proto.TensorProto.INT64, [len(axes)], axes
                )
            inputs.append(axes_name)
        if steps:
            if not axes:
                inputs.append("")
            if isinstance(steps, str):
                steps_name = steps
            else:
                steps_name = scope.get_unique_variable_name("steps")
                container.add_initializer(
                    steps_name, onnx_proto.TensorProto.INT64, [len(steps)], steps
                )
            inputs.append(steps_name)
        container.add_node(
            "Slice", inputs, output_name, name=name, op_version=op_version
        )


def apply_argmax(
    scope,
    input_name,
    output_name,
    container,
    operator_name=None,
    axis=0,
    keepdims=1,
    select_last_index=0,
):
    name = _create_name_or_use_existing_one(scope, "ArgMax", operator_name)
    attrs = {"axis": axis, "keepdims": keepdims}
    if container.target_opset < 11:
        op_version = 1
    elif container.target_opset < 12:
        op_version = 11
    else:
        op_version = 12
        attrs["select_last_index"] = select_last_index
    container.add_node(
        "ArgMax", input_name, output_name, op_version=op_version, name=name, **attrs
    )


def apply_argmin(
    scope,
    input_name,
    output_name,
    container,
    operator_name=None,
    axis=0,
    keepdims=1,
    select_last_index=0,
):
    name = _create_name_or_use_existing_one(scope, "ArgMin", operator_name)
    attrs = {"axis": axis, "keepdims": keepdims}
    if container.target_opset < 11:
        op_version = 1
    elif container.target_opset < 12:
        op_version = 11
    else:
        op_version = 12
        attrs["select_last_index"] = select_last_index
    container.add_node(
        "ArgMin", input_name, output_name, op_version=op_version, name=name, **attrs
    )


def apply_matmul(scope, input_names, output_name, container, operator_name=None):
    op_type = "MatMul"
    name = _create_name_or_use_existing_one(scope, op_type, operator_name)
    if container.target_opset <= 9:
        op_version = 1
    else:
        op_version = 9
    container.add_node(
        op_type, input_names, output_name, op_version=op_version, name=name
    )


def apply_reciprocal(scope, input_name, output_name, container, operator_name=None):
    _apply_unary_operation(
        scope,
        "Reciprocal",
        input_name,
        output_name,
        container,
        operator_name=operator_name,
    )


def apply_less(scope, input_names, output_name, container, operator_name=None):
    name = _create_name_or_use_existing_one(scope, "Less", operator_name)
    if container.target_opset < 7:
        op_version = 1
    elif container.target_opset < 9:
        op_version = 7
    else:
        op_version = 9

    container.add_node(
        "Less", input_names, output_name, name=name, op_version=op_version
    )


def apply_greater(scope, input_names, output_name, container, operator_name=None):
    name = _create_name_or_use_existing_one(scope, "Greater", operator_name)
    if container.target_opset < 7:
        op_version = 1
    elif container.target_opset < 9:
        op_version = 7
    else:
        op_version = 9

    container.add_node(
        "Greater", input_names, output_name, name=name, op_version=op_version
    )
