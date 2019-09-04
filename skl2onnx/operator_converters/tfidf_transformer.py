# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import numpy as np
from ..common._registration import register_converter
from ..common._apply_operation import apply_log, apply_add
from ..common._apply_operation import apply_mul, apply_identity


def convert_sklearn_tfidf_transformer(scope, operator, container):
    # TODO: use sparse containers when available
    float_type = container.dtype
    # onnx_proto.TensorProto.FLOAT
    proto_type = container.proto_dtype
    op = operator.raw_operator
    data = operator.input_full_names
    final = operator.output_full_names
    C = operator.inputs[0].type.shape[1]

    if op.sublinear_tf:
        # code scikit-learn
        # np.log(X.data, X.data) --> does not apply on null coefficient
        # X.data += 1
        raise RuntimeError(
            "ONNX does not support sparse tensors before opset < 11, "
            "sublinear_tf must be False.")

        logged = scope.get_unique_variable_name('logged')
        apply_log(scope, data, logged, container)

        if not op.use_idf and op.norm is None:
            loggedplus1 = final
        else:
            loggedplus1 = scope.get_unique_variable_name('loggedplus1')
        ones = scope.get_unique_variable_name('ones')
        cst = np.ones((C,), dtype=float_type)
        container.add_initializer(ones, proto_type, [C], cst.flatten())
        apply_add(scope, [logged, ones], loggedplus1, container, broadcast=1)

        data = [loggedplus1]

    if op.use_idf:
        cst = op.idf_.astype(float_type)
        if len(cst.shape) > 1:
            cst = np.diag(cst)
        cst = cst.ravel().flatten()
        shape = [len(cst)]
        idfcst = scope.get_unique_variable_name('idfcst')
        container.add_initializer(idfcst, proto_type, shape, cst)
        idfed = (final if op.norm is None
                 else scope.get_unique_variable_name('idfed'))
        apply_mul(scope, data + [idfcst], idfed, container, broadcast=1)
        data = [idfed]

    if op.norm is not None:
        op_type = 'Normalizer'
        norm_map = {'max': 'MAX', 'l1': 'L1', 'l2': 'L2'}
        attrs = {'name': scope.get_unique_operator_name(op_type)}
        if op.norm in norm_map:
            attrs['norm'] = norm_map[op.norm]
        else:
            raise RuntimeError("Invalid norm '%s'. "
                               "You may raise an issue at "
                               "https://github.com/onnx/sklearn-onnx/"
                               "issues." % op.norm)

        container.add_node(op_type, data, operator.output_full_names,
                           op_domain='ai.onnx.ml', **attrs)
        data = None

    if data == operator.input_full_names:
        # Nothing happened --> identity
        apply_identity(scope, data, final, container)


register_converter('SklearnTfidfTransformer',
                   convert_sklearn_tfidf_transformer)
