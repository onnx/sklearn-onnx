# SPDX-License-Identifier: Apache-2.0


from onnxconverter_common.onnx_ops import *  # noqa
from ..proto import onnx_proto


def apply_normalizer(scope, inputs, outputs, container,
                     norm, use_float):
    """
    Adds operator Normalizer if *use_float* is true,
    otherwise, uses *ReduceSum* + *Div*. *Normalizer*
    always produces float according to ONNX speciciations.
    """
    input = inputs[0] if isinstance(inputs, list) else inputs
    output = outputs[0] if isinstance(outputs, list) else outputs

    if use_float:
        container.add_node(
            'Normalizer', input, output,
            op_domain='ai.onnx.ml', norm=norm,
            name=scope.get_unique_operator_name('Normalizer'))
    else:
        # Normalizer only produces floats.
        if norm == 'L1':
            norm = scope.get_unique_variable_name('norm')
            norm_abs = scope.get_unique_variable_name('norm_abs')
            container.add_node(
                'Abs', input, norm_abs,
                name=scope.get_unique_operator_name('Abs'))

            if container.target_opset < 13:
                container.add_node(
                    'ReduceSum', norm_abs, norm, axes=[1], keepdims=1,
                    name=scope.get_unique_operator_name('ReduceSum'))
            else:
                axis_name = scope.get_unique_variable_name('axis')
                container.add_initializer(
                    axis_name, onnx_proto.TensorProto.INT64, [1], [1])
                container.add_node(
                    'ReduceSum', [norm_abs, axis_name], norm, keepdims=1,
                    name=scope.get_unique_operator_name('ReduceSum'))
            apply_div(  # noqa
                scope, [input, norm], output, container,
                operator_name=scope.get_unique_operator_name(
                    'NormalizerNorm'))
        elif norm == 'L2':
            norm = scope.get_unique_variable_name('norm')
            norm2 = scope.get_unique_variable_name('norm2')
            container.add_node(
                'ReduceSumSquare', input, norm, axes=[1], keepdims=1,
                name=scope.get_unique_operator_name('ReduceSumSquare'))
            container.add_node(
                'Sqrt', [norm], norm2,
                name=scope.get_unique_operator_name('Sqrt'))
            apply_div(  # noqa
                scope, [input, norm2], output, container,
                operator_name=scope.get_unique_operator_name(
                    'NormalizerNorm'))
        else:
            raise NotImplementedError(
                "Normalization not implemented for norm %r." % norm)
