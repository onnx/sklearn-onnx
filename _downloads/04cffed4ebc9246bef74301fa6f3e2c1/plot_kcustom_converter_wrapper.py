# SPDX-License-Identifier: Apache-2.0

"""
.. _l-plot-custom-converter-wrapper:

Implement a new converter using other converters
================================================

.. index:: custom converter

In many cases, a custom models leverages existing models
which already have an associated converter. To convert this
patchwork, existing converters must be called. This example
shows how to do that. Example :ref:`l-plot-custom-converter`
can be rewritten by using a `PCA <https://scikit-learn.org/
stable/modules/generated/sklearn.decomposition.PCA.html>`_.
We could then reuse the converter associated to this model.

.. contents::
    :local:

Custom model
++++++++++++

Let's implement a simple custom model using
:epkg:`scikit-learn` API. The model is preprocessing
which decorrelates correlated random variables.
If *X* is a matrix of features, :math:`V=\\frac{1}{n}X'X`
is the covariance matrix. We compute :math:`X V^{1/2}`.
"""
from mlprodict.onnxrt import OnnxInference
from pyquickhelper.helpgen.graphviz_helper import plot_graphviz
import pickle
from io import BytesIO
import numpy
from numpy.testing import assert_almost_equal
from onnxruntime import InferenceSession
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from skl2onnx import update_registered_converter
from skl2onnx.algebra.onnx_operator import OnnxSubEstimator
from skl2onnx import to_onnx


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
        self.pca_ = PCA(X.shape[1])
        self.pca_.fit(X)
        return self

    def transform(self, X):
        return self.pca_.transform(X)


def test_decorrelate_transformer():
    data = load_iris()
    X = data.data

    dec = DecorrelateTransformer()
    dec.fit(X)
    pred = dec.transform(X)
    cov = pred.T @ pred
    for i in range(cov.shape[0]):
        cov[i, i] = 1.
    assert_almost_equal(numpy.identity(4), cov)

    st = BytesIO()
    pickle.dump(dec, st)
    dec2 = pickle.load(BytesIO(st.getvalue()))
    assert_almost_equal(dec.transform(X), dec2.transform(X))


test_decorrelate_transformer()

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
# Let's try to convert it and see what happens.


try:
    to_onnx(dec, X.astype(numpy.float32))
except Exception as e:
    print(e)

############################
# This error means there is no converter associated
# to *DecorrelateTransformer*. Let's do it.
# It requires to implement the two following
# functions, a shape calculator and a converter
# with the same signature as below.
# First the shape calculator. We retrieve the input type
# add tells the output type has the same type,
# the same number of rows and a specific number of columns.


def decorrelate_transformer_shape_calculator(operator):
    op = operator.raw_operator
    input_type = operator.inputs[0].type.__class__
    input_dim = operator.inputs[0].type.shape[0]
    output_type = input_type([input_dim, op.pca_.components_.shape[1]])
    operator.outputs[0].type = output_type


###################################
# The converter. One thing we need to pay attention to
# is the target opset. This information is important
# to make sure that every node is defined following the
# specifications of that opset.


def decorrelate_transformer_converter(scope, operator, container):
    op = operator.raw_operator
    opv = container.target_opset
    out = operator.outputs

    # We retrieve the unique input.
    X = operator.inputs[0]

    # We tell in ONNX language how to compute the unique output.
    # op_version=opv tells which opset is requested
    Y = OnnxSubEstimator(op.pca_, X, op_version=opv, output_names=out[:1])
    Y.add_to(scope, container)


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


#############################
# Final graph
# +++++++++++

oinf = OnnxInference(onx)
ax = plot_graphviz(oinf.to_dot())
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
