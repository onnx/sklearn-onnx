# SPDX-License-Identifier: Apache-2.0


from ..common._apply_operation import (
    apply_add, apply_argmax, apply_cast, apply_concat, apply_div, apply_exp,
    apply_log, apply_matmul, apply_mul, apply_pow,
    apply_reducesum, apply_reshape, apply_sub, apply_transpose)
from ..common.data_types import (
    BooleanTensorType, Int64TensorType, guess_proto_type)
from ..common._registration import register_converter
from ..common._topology import Scope, Operator
from ..common._container import ModelComponentContainer
from ..proto import onnx_proto


def convert_quadratic_discriminant_analysis_classifier(
    scope: Scope, operator: Operator, container: ModelComponentContainer):

    input_name = operator.inputs[0].full_name
    model = operator.raw_operator

    n_classes = len(model.classes_)

    proto_dtype = guess_proto_type(operator.inputs[0].type)
    if proto_dtype != onnx_proto.TensorProto.DOUBLE:
        proto_dtype = onnx_proto.TensorProto.FLOAT

    if isinstance(operator.inputs[0].type,
                  (BooleanTensorType, Int64TensorType)):
        cast_input_name = scope.get_unique_variable_name('cast_input')
        apply_cast(scope, operator.input_full_names, cast_input_name,
                   container, to=proto_dtype)
        input_name = cast_input_name

    norm_array_name = []
    sum_array_name = []

    container.add_initializer(
        'const_n05', onnx_proto.TensorProto.FLOAT, [], [-0.5])
    container.add_initializer(
        'const_p2', onnx_proto.TensorProto.FLOAT, [], [2])

    for i in range(n_classes):
        R = model.rotations_[i]
        rotation_name = scope.get_unique_variable_name('rotations')
        container.add_initializer(
            rotation_name, onnx_proto.TensorProto.FLOAT, R.shape, R)

        S = model.scalings_[i]
        scaling_name = scope.get_unique_variable_name('scalings')
        container.add_initializer(
            scaling_name, onnx_proto.TensorProto.FLOAT, S.shape, S)

        mean = model.means_[i]
        mean_name = scope.get_unique_variable_name('means')
        container.add_initializer(
            mean_name, onnx_proto.TensorProto.FLOAT, mean.shape, mean)
        #Xm = X - self.means_[i]
        Xm_name = scope.get_unique_variable_name('Xm')
        apply_sub(scope, [input_name, mean_name], [Xm_name], container)

        s_pow_name = scope.get_unique_variable_name('s_pow_n05')
        apply_pow(scope, [scaling_name, 'const_n05'], [s_pow_name], container)

        mul_name = scope.get_unique_variable_name('mul')
        apply_mul(scope, [rotation_name, s_pow_name], [mul_name], container)

        x2_name = scope.get_unique_variable_name('matmul')
        apply_matmul(scope, [Xm_name, mul_name], [x2_name], container)

        pow_x2_name = scope.get_unique_variable_name('pow_x2')
        apply_pow(scope, [x2_name, 'const_p2'], [pow_x2_name], container)

        # np.sum(X2**2, axis=1)
        sum_name = scope.get_unique_variable_name('sum')
        apply_reducesum(scope, [pow_x2_name], [sum_name], container, axes=[1], keepdims=1)
        # norm2.append(np.sum(X2**2, axis=1))
        norm_array_name.append(sum_name)

        # u = np.asarray([np.sum(np.log(s)) for s in self.scalings_])
        log_name = scope.get_unique_variable_name('log')
        apply_log(scope, [scaling_name], [log_name], container)

        sum_log_name = scope.get_unique_variable_name('sum_log')
        apply_reducesum(scope, [log_name], [sum_log_name], container, keepdims=1)
        sum_array_name.append(sum_log_name)

    # end for

    concat_norm_name = scope.get_unique_variable_name('concat_norm')
    apply_concat(scope, norm_array_name, [concat_norm_name], container)

    reshape_norm_name = scope.get_unique_variable_name('reshape_concat_norm')
    apply_reshape(scope, [concat_norm_name], [reshape_norm_name], container, desired_shape=[n_classes, -1])

    transpose_norm_name = scope.get_unique_variable_name('transpose_norm')
    apply_transpose(scope, [reshape_norm_name], [transpose_norm_name], container, perm=(1, 0))

    apply_concat(scope, sum_array_name, ['concat_logsum'], container)

    add_norm2_u_name = scope.get_unique_variable_name('add_norm2_u')
    apply_add(scope, [transpose_norm_name, 'concat_logsum'], [add_norm2_u_name], container)

    norm2_u_n05_name = scope.get_unique_variable_name('norm2_u_n05')
    apply_mul(scope, ['const_n05', add_norm2_u_name], [norm2_u_n05_name], container)


    container.add_initializer(
        'priors', onnx_proto.TensorProto.FLOAT, [n_classes,], model.priors_)
    apply_log(scope, ['priors'], ['log_p'], container)

    apply_add(scope, [norm2_u_n05_name, 'log_p'], ['decision_fun'], container)

    apply_argmax(scope, ['decision_fun'], ['argmax_out'], container, axis=1)

    container.add_initializer(
        'classes', onnx_proto.TensorProto.INT64, [n_classes], model.classes_)

    container.add_node(
        'ArrayFeatureExtractor',
        ['classes', 'argmax_out'],
        [operator.outputs[0].full_name], 
        op_domain='ai.onnx.ml'
    )

    attr = {'axes': [1]}
    container.add_node(
        'ReduceMax', ['decision_fun'], ['df_max'], **attr)

    apply_sub(scope, ['decision_fun', 'df_max'], ['df_sub_max'], container)

    apply_exp(scope, ['df_sub_max'], ['likelihood'], container)

    apply_reducesum(scope, ['likelihood'], ['likelihood_sum'], container, axes=[1], keepdims=1)

    apply_div(scope, ['likelihood', 'likelihood_sum'], [operator.outputs[1].full_name], container)  


register_converter('SklearnQuadraticDiscriminantAnalysis',
                   convert_quadratic_discriminant_analysis_classifier,
                   options={'zipmap': [True, False, 'columns'],
                            'nocl': [True, False],
                            'output_class_labels': [False, True],
                            'raw_scores': [True, False]})
