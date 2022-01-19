# SPDX-License-Identifier: Apache-2.0

"""
Two ways to implement a converter
=================================

.. index:: syntax

There are two ways to write a converter. The first one
is less verbose and easier to understand
(see `k_means.py <https://github.com/onnx/sklearn-onnx/blob/
master/skl2onnx/operator_converters/k_means.py>`_). The other is very verbose (see `ada_boost.py <https://github.com/onnx/
sklearn-onnx/blob/master/skl2onnx/operator_converters/ada_boost.py>`_
for an example).

The first way is used in :ref:`l-plot-custom-converter`.
This one demonstrates the second way which is usually the one
used in other converter library. It is more verbose.

.. contents::
    :local:


Custom model
++++++++++++

It basically copies what is in example
`:ref:`l-plot-custom-converter`.
"""
from skl2onnx.common.data_types import guess_proto_type
from onnxconverter_common.onnx_ops import apply_sub
from onnxruntime import InferenceSession
from skl2onnx import update_registered_converter
from skl2onnx import to_onnx
import numpy
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.datasets import load_iris


class DecorrelateTransformer(TransformerMixin, BaseEstimator):
    """
    Decorrelates correlated gaussian features.

    :param alpha: avoids non inversible matrices
        by adding *alpha* identity matrix

    *Attributes*

    * `self.mean_`: average
    * `self.coef_`: square root of the coveriance matrix
    """

    def __init__(self, alpha=0.):
        BaseEstimator.__init__(self)
        TransformerMixin.__init__(self)
        self.alpha = alpha

    def fit(self, X, y=None, sample_weights=None):
        if sample_weights is not None:
            raise NotImplementedError(
                "sample_weights != None is not implemented.")
        self.mean_ = numpy.mean(X, axis=0, keepdims=True)
        X = X - self.mean_
        V = X.T @ X / X.shape[0]
        if self.alpha != 0:
            V += numpy.identity(V.shape[0]) * self.alpha
        L, P = numpy.linalg.eig(V)
        Linv = L ** (-0.5)
        diag = numpy.diag(Linv)
        root = P @ diag @ P.transpose()
        self.coef_ = root
        return self

    def transform(self, X):
        return (X - self.mean_) @ self.coef_


data = load_iris()
X = data.data

dec = DecorrelateTransformer()
dec.fit(X)
pred = dec.transform(X[:5])
print(pred)


############################################
# Conversion into ONNX
# ++++++++++++++++++++
#
# The shape calculator does not change.

def decorrelate_transformer_shape_calculator(operator):
    op = operator.raw_operator
    input_type = operator.inputs[0].type.__class__
    # The shape may be unknown. *get_first_dimension*
    # returns the appropriate value, None in most cases
    # meaning the transformer can process any batch of observations.
    input_dim = operator.inputs[0].get_first_dimension()
    output_type = input_type([input_dim, op.coef_.shape[1]])
    operator.outputs[0].type = output_type


###################################
# The converter is different.


def decorrelate_transformer_converter(scope, operator, container):
    op = operator.raw_operator
    out = operator.outputs

    # We retrieve the unique input.
    X = operator.inputs[0]

    # In most case, computation happen in floats.
    # But it might be with double. ONNX is very strict
    # about types, every constant should have the same
    # type as the input.
    proto_dtype = guess_proto_type(X.type)

    mean_name = scope.get_unique_variable_name('mean')
    container.add_initializer(mean_name, proto_dtype,
                              op.mean_.shape, list(op.mean_.ravel()))

    coef_name = scope.get_unique_variable_name('coef')
    container.add_initializer(coef_name, proto_dtype,
                              op.coef_.shape, list(op.coef_.ravel()))

    op_name = scope.get_unique_operator_name('sub')
    sub_name = scope.get_unique_variable_name('sub')
    # This function is defined in package onnxconverter_common.
    # Most common operators can be added to the graph with
    # these functions. It handles the case when specifications
    # changed accross opsets (a parameter becomes an input
    # for example).
    apply_sub(scope, [X.full_name, mean_name], sub_name, container,
              operator_name=op_name)

    op_name = scope.get_unique_operator_name('matmul')
    container.add_node(
        'MatMul', [sub_name, coef_name],
        out[0].full_name, name=op_name)


##########################################
# We need to let *skl2onnx* know about the new converter.

update_registered_converter(
    DecorrelateTransformer, "SklearnDecorrelateTransformer",
    decorrelate_transformer_shape_calculator,
    decorrelate_transformer_converter)


onx = to_onnx(dec, X.astype(numpy.float32))

sess = InferenceSession(onx.SerializeToString())

exp = dec.transform(X.astype(numpy.float32))
got = sess.run(None, {'X': X.astype(numpy.float32)})[0]


def diff(p1, p2):
    p1 = p1.ravel()
    p2 = p2.ravel()
    d = numpy.abs(p2 - p1)
    return d.max(), (d / numpy.abs(p1)).max()


print(diff(exp, got))

#####################################
# Let's check it works as well with double.

onx = to_onnx(dec, X.astype(numpy.float64))

sess = InferenceSession(onx.SerializeToString())

exp = dec.transform(X.astype(numpy.float64))
got = sess.run(None, {'X': X.astype(numpy.float64)})[0]
print(diff(exp, got))

#############################################
# The differences are smaller with double as expected.
