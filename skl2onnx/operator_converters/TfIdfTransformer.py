# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import numpy

from ..common._registration import register_converter
from ..common._apply_operation import apply_mul, apply_identity
from ..proto import onnx_proto


def convert_sklearn_tfidf_transformer(scope, operator, container):
    # TODO: use sparse containers when available
    op = operator.raw_operator
    data = operator.input_full_names
    final = operator.output_full_names

    if op.sublinear_tf:
        # code scikit-learn
        # np.log(X.data, X.data) --> does not apply on null coefficient
        # X.data += 1
        raise RuntimeError("ONNX does not support sparse tensors, sublinear_tf must be False")

    if op.use_idf:
        # code scikit-learn
        # X = X * self._idf_diag
        cst = op._idf_diag.astype(numpy.float32)
        if not isinstance(cst, numpy.ndarray):
            cst = numpy.array(cst.todense())
        if len(cst.shape) > 1:
            cst = numpy.diag(cst)
        cst = cst.ravel().flatten()
        shape = [len(cst)]
        idfcst = scope.get_unique_variable_name('idfcst')
        container.add_initializer(idfcst, onnx_proto.TensorProto.FLOAT, shape, cst)
        idfed = final if op.norm is None else scope.get_unique_variable_name('idfed')
        apply_mul(scope, data + [idfcst], idfed, container, broadcast=1)
        data = [idfed]

    if op.norm is not None:
        op_type = 'Normalizer'
        norm_map = {'max': 'MAX', 'l1': 'L1', 'l2': 'L2'}
        attrs = {'name': scope.get_unique_operator_name(op_type)}
        if op.norm in norm_map:
            attrs['norm'] = norm_map[op.norm]
        else:
            raise RuntimeError('Invalid norm: %s' % op.norm)

        container.add_node(op_type, data, operator.output_full_names, op_domain='ai.onnx.ml', **attrs)
        data = None

    if data == operator.input_full_names:
        # Nothing happened --> identity
        apply_identity(scope, data, final, container)


register_converter('SklearnTfidfTransformer', convert_sklearn_tfidf_transformer)
