# SPDX-License-Identifier: Apache-2.0

import numpy as np
from onnx import TensorProto
from onnx.helper import make_tensor
from ..common._registration import register_converter
from ..common._topology import Scope, Operator
from ..common._container import ModelComponentContainer


def convert_sklearn_feature_hasher(scope: Scope, operator: Operator,
                                   container: ModelComponentContainer):
    X = operator.inputs[0]
    out = operator.outputs
    op = operator.raw_operator
    if op.input_type != "string":
        raise RuntimeError(
            f"The converter for FeatureHasher only supports "
            f"input_type='string' not {op.input_type!r}.")

    hashed_ = scope.get_unique_variable_name('hashed_')
    container.add_node('MurmurHash3', X.full_name, hashed_,
                       positive=0, seed=0, op_domain="com.microsoft",
                       op_version=1)
    hashed = scope.get_unique_variable_name('hashed')
    container.add_node('Cast', hashed_, hashed, to=TensorProto.INT64)

    if op.dtype in (np.float32, np.float64, np.int64):
        cst_neg = -1
    else:
        cst_neg = 4294967295

    infinite = scope.get_unique_variable_name('infinite')
    container.add_initializer(infinite, TensorProto.INT64, [1],
                              [-2147483648])

    infinite2 = scope.get_unique_variable_name('infinite2')
    container.add_initializer(infinite2, TensorProto.INT64, [1],
                              [cst_neg])

    infinite_n = scope.get_unique_variable_name('infinite_n')
    container.add_initializer(infinite_n, TensorProto.INT64, [1],
                              [2147483647 - (op.n_features - 1)])

    zero = scope.get_unique_variable_name('zero')
    container.add_initializer(zero, TensorProto.INT64, [1], [0])

    one = scope.get_unique_variable_name('one')
    container.add_initializer(one, TensorProto.INT64, [1], [1])

    mone = scope.get_unique_variable_name('mone')
    container.add_initializer(mone, TensorProto.INT64, [1], [-1])

    mtwo = scope.get_unique_variable_name('mtwo')
    container.add_initializer(mtwo, TensorProto.INT64, [1], [-2])

    nf = scope.get_unique_variable_name('nf')
    container.add_initializer(nf, TensorProto.INT64, [1], [op.n_features])

    new_shape = scope.get_unique_variable_name('new_shape')
    container.add_initializer(new_shape, TensorProto.INT64, [2], [-1, 1])
    new_shape2 = scope.get_unique_variable_name('new_shape2')
    container.add_initializer(new_shape2, TensorProto.INT64, [2], [1, -1])

    # values
    if op.alternate_sign:
        cmp = scope.get_unique_variable_name('cmp')
        container.add_node('GreaterOrEqual', [hashed, zero], cmp)
        values = scope.get_unique_variable_name('values')
        container.add_node('Where', [cmp, one, infinite2], values)
    else:
        mul = scope.get_unique_variable_name('mul')
        container.add_node('Mul', [hashed, zero], mul)
        values = scope.get_unique_variable_name('values')
        container.add_node('Add', [mul, one], values)

    values_reshaped = scope.get_unique_variable_name('values_reshaped')
    container.add_node('Reshape', [values, new_shape], values_reshaped)

    # indices
    cmp = scope.get_unique_variable_name('cmp_ind')
    container.add_node('Equal', [hashed, infinite], cmp)
    values_abs = scope.get_unique_variable_name('values_abs')
    container.add_node('Abs', hashed, values_abs)
    values_ind = scope.get_unique_variable_name('values_ind')
    container.add_node('Where', [cmp, infinite_n, values_abs], values_ind)
    indices = scope.get_unique_variable_name('indices')
    container.add_node('Mod', [values_ind, nf], indices)
    indices_reshaped = scope.get_unique_variable_name('indices_reshaped')
    container.add_node('Reshape', [indices, new_shape], indices_reshaped)

    # scatter
    zerot_ = scope.get_unique_variable_name('zerot_')
    container.add_node('ConstantOfShape', [nf], zerot_,
                       value=make_tensor("value",
                                         TensorProto.INT64, [1], [0]))
    zerot = scope.get_unique_variable_name('zerot')
    container.add_node('Mul', [indices_reshaped, zerot_], zerot)

    final = scope.get_unique_variable_name('final')
    container.add_node('ScatterElements',
                       [zerot, indices_reshaped, values_reshaped],
                       final, axis=1)

    # at this point, every string has been processed as if it was in
    # in a single columns.
    # in case there is more than one column, we need to reduce over
    # the last dimension
    input_shape = scope.get_unique_variable_name('input_shape')
    container.add_node('Shape', X.full_name, input_shape)
    shape_not_last = scope.get_unique_variable_name('shape_not_last')
    container.add_node('Slice', [input_shape, zero, mone], shape_not_last)
    final_shape = scope.get_unique_variable_name('final_last')
    container.add_node('Concat', [shape_not_last, mone, nf],
                       final_shape, axis=0)
    final_reshaped = scope.get_unique_variable_name('final_reshaped')
    container.add_node('Reshape', [final, final_shape], final_reshaped)
    container.add_node('ReduceSum', [final_reshaped, mtwo],
                       out[0].full_name, keepdims=0)


register_converter('SklearnFeatureHasher', convert_sklearn_feature_hasher)
