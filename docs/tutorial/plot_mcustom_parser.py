# SPDX-License-Identifier: Apache-2.0

"""
Change the number of outputs by adding a parser
===============================================

.. index:: parser

By default, :epkg:`sklearn-onnx` assumes that a classifier
has two outputs (label and probabilities), a regressor
has one output (prediction), a transform has one output
(the transformed data). What if it is not the case?
The following example creates a custom converter
and a custom parser which defines the number of outputs
expected by the converted model.

Example :ref:`l-plot-custom-options` shows a converter
which selects two ways to compute the same outputs.
In this one, the converter produces both. That would not
be a very efficient converter but that's just for the sake
of using a parser. By default, a transformer only returns
one output but both are needed.

.. contents::
    :local:

A new transformer
+++++++++++++++++
"""
from pyquickhelper.helpgen.graphviz_helper import plot_graphviz
from mlprodict.onnxrt import OnnxInference
import numpy
from onnxruntime import InferenceSession
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.datasets import load_iris
from skl2onnx import update_registered_converter
from skl2onnx.common.data_types import guess_numpy_type
from skl2onnx.algebra.onnx_ops import (
    OnnxSub, OnnxMatMul, OnnxGemm)
from skl2onnx import to_onnx, get_model_alias


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
# Conversion into ONNX with two outputs
# +++++++++++++++++++++++++++++++++++++
#
# Let's try to convert it and see what happens.


def decorrelate_transformer_shape_calculator(operator):
    op = operator.raw_operator
    input_type = operator.inputs[0].type.__class__
    input_dim = operator.inputs[0].type.shape[0]
    output_type = input_type([input_dim, op.coef_.shape[1]])
    operator.outputs[0].type = output_type


def decorrelate_transformer_converter(scope, operator, container):
    op = operator.raw_operator
    opv = container.target_opset
    out = operator.outputs

    X = operator.inputs[0]

    dtype = guess_numpy_type(X.type)

    Y1 = OnnxMatMul(
        OnnxSub(X, op.mean_.astype(dtype), op_version=opv),
        op.coef_.astype(dtype),
        op_version=opv, output_names=out[:1])

    Y2 = OnnxGemm(X, op.coef_.astype(dtype),
                  (- op.mean_ @ op.coef_).astype(dtype),
                  op_version=opv, alpha=1., beta=1.,
                  output_names=out[1:2])

    Y1.add_to(scope, container)
    Y2.add_to(scope, container)


def decorrelate_transformer_parser(
        scope, model, inputs, custom_parsers=None):
    alias = get_model_alias(type(model))
    this_operator = scope.declare_local_operator(alias, model)

    # inputs
    this_operator.inputs.append(inputs[0])

    # outputs
    cls_type = inputs[0].type.__class__
    val_y1 = scope.declare_local_variable('nogemm', cls_type())
    val_y2 = scope.declare_local_variable('gemm', cls_type())
    this_operator.outputs.append(val_y1)
    this_operator.outputs.append(val_y2)

    # ends
    return this_operator.outputs

###################################
# The registration needs to declare the parser as well.


update_registered_converter(
    DecorrelateTransformer, "SklearnDecorrelateTransformer",
    decorrelate_transformer_shape_calculator,
    decorrelate_transformer_converter,
    parser=decorrelate_transformer_parser)


#############################################
# And conversion.

onx = to_onnx(dec, X.astype(numpy.float32),
              target_opset=14)

sess = InferenceSession(onx.SerializeToString())

exp = dec.transform(X.astype(numpy.float32))
results = sess.run(None, {'X': X.astype(numpy.float32)})
y1 = results[0]
y2 = results[1]


def diff(p1, p2):
    p1 = p1.ravel()
    p2 = p2.ravel()
    d = numpy.abs(p2 - p1)
    return d.max(), (d / numpy.abs(p1)).max()


print(diff(exp, y1))
print(diff(exp, y2))


################################
# It works. The final looks like the following.

oinf = OnnxInference(onx, runtime="python_compiled")
print(oinf)

#############################
# Final graph
# +++++++++++

ax = plot_graphviz(oinf.to_dot())
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
