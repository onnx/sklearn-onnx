# SPDX-License-Identifier: Apache-2.0


"""
.. _l-convert-syntax:

Different ways to convert a model
=================================

This example leverages some code added to implement custom converters
in an easy way.

.. contents::
    :local:

Predict with onnxruntime
++++++++++++++++++++++++

Simple function to check the converted model
works fine.
"""
import onnxruntime
import onnx
import numpy
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.pipeline import make_pipeline
from onnxruntime import InferenceSession
from skl2onnx import convert_sklearn, to_onnx, wrap_as_onnx_mixin
from skl2onnx.common.data_types import FloatTensorType
from skl2onnx.algebra.onnx_ops import OnnxSub, OnnxDiv
from skl2onnx.algebra.onnx_operator_mixin import OnnxOperatorMixin


def predict_with_onnxruntime(onx, X):
    sess = InferenceSession(onx.SerializeToString())
    input_name = sess.get_inputs()[0].name
    res = sess.run(None, {input_name: X.astype(np.float32)})
    return res[0]

#################################
# Simple KMeans
# +++++++++++++
#
# The first way: :func:`convert_sklearn`.


X = np.arange(20).reshape(10, 2)
tr = KMeans(n_clusters=2)
tr.fit(X)

onx = convert_sklearn(
    tr, initial_types=[('X', FloatTensorType((None, X.shape[1])))],
    target_opset=12)
print(predict_with_onnxruntime(onx, X))

#################################
# The second way: :func:`to_onnx`: no need to play with
# :class:`FloatTensorType` anymore.

X = np.arange(20).reshape(10, 2)
tr = KMeans(n_clusters=2)
tr.fit(X)

onx = to_onnx(tr, X.astype(np.float32), target_opset=12)
print(predict_with_onnxruntime(onx, X))


#################################
# The third way: :func:`wrap_as_onnx_mixin`: wraps
# the machine learned model into a new class
# inheriting from :class:`OnnxOperatorMixin`.

X = np.arange(20).reshape(10, 2)
tr = KMeans(n_clusters=2)
tr.fit(X)

tr_mixin = wrap_as_onnx_mixin(tr, target_opset=12)

onx = tr_mixin.to_onnx(X.astype(np.float32))
print(predict_with_onnxruntime(onx, X))

#################################
# The fourth way: :func:`wrap_as_onnx_mixin`: can be called
# before fitting the model.

X = np.arange(20).reshape(10, 2)
tr = wrap_as_onnx_mixin(KMeans(n_clusters=2),
                        target_opset=12)
tr.fit(X)

onx = tr.to_onnx(X.astype(np.float32))
print(predict_with_onnxruntime(onx, X))

##########################################
# Pipeline and a custom object
# ++++++++++++++++++++++++++++
#
# This is a simple scaler.


class CustomOpTransformer(BaseEstimator, TransformerMixin,
                          OnnxOperatorMixin):

    def __init__(self):
        BaseEstimator.__init__(self)
        TransformerMixin.__init__(self)
        self.op_version = 12

    def fit(self, X, y=None):
        self.W_ = np.mean(X, axis=0)
        self.S_ = np.std(X, axis=0)
        return self

    def transform(self, X):
        return (X - self.W_) / self.S_

    def onnx_shape_calculator(self):
        def shape_calculator(operator):
            operator.outputs[0].type = operator.inputs[0].type
        return shape_calculator

    def to_onnx_operator(self, inputs=None, outputs=('Y', ),
                         target_opset=None, **kwargs):
        if inputs is None:
            raise RuntimeError("Parameter inputs should contain at least "
                               "one name.")
        opv = target_opset or self.op_version
        i0 = self.get_inputs(inputs, 0)
        W = self.W_.astype(np.float32)
        S = self.S_.astype(np.float32)
        return OnnxDiv(OnnxSub(i0, W, op_version=12), S,
                       output_names=outputs,
                       op_version=opv)

#############################
# Way 1


X = np.arange(20).reshape(10, 2)
tr = make_pipeline(CustomOpTransformer(), KMeans(n_clusters=2))
tr.fit(X)

onx = convert_sklearn(
    tr, initial_types=[('X', FloatTensorType((None, X.shape[1])))],
    target_opset=12)
print(predict_with_onnxruntime(onx, X))

#############################
# Way 2

X = np.arange(20).reshape(10, 2)
tr = make_pipeline(CustomOpTransformer(), KMeans(n_clusters=2))
tr.fit(X)

onx = to_onnx(tr, X.astype(np.float32), target_opset=12)
print(predict_with_onnxruntime(onx, X))

#############################
# Way 3

X = np.arange(20).reshape(10, 2)
tr = make_pipeline(CustomOpTransformer(), KMeans(n_clusters=2))
tr.fit(X)

tr_mixin = wrap_as_onnx_mixin(tr, target_opset=12)
tr_mixin.to_onnx(X.astype(np.float32))

print(predict_with_onnxruntime(onx, X))

#############################
# Way 4

X = np.arange(20).reshape(10, 2)
tr = wrap_as_onnx_mixin(
    make_pipeline(CustomOpTransformer(), KMeans(n_clusters=2)),
    target_opset=12)

tr.fit(X)

onx = tr.to_onnx(X.astype(np.float32))
print(predict_with_onnxruntime(onx, X))

##################################
# Display the ONNX graph
# ++++++++++++++++++++++
#
# Finally, let's see the graph converted with *sklearn-onnx*.

from onnx.tools.net_drawer import GetPydotGraph, GetOpNodeProducer  # noqa
pydot_graph = GetPydotGraph(onx.graph, name=onx.graph.name, rankdir="TB",
                            node_producer=GetOpNodeProducer(
                                "docstring", color="yellow",
                                fillcolor="yellow", style="filled"))
pydot_graph.write_dot("pipeline_onnx_mixin.dot")

import os  # noqa
os.system('dot -O -Gdpi=300 -Tpng pipeline_onnx_mixin.dot')

import matplotlib.pyplot as plt  # noqa
image = plt.imread("pipeline_onnx_mixin.dot.png")
fig, ax = plt.subplots(figsize=(40, 20))
ax.imshow(image)
ax.axis('off')

#################################
# **Versions used for this example**

import sklearn  # noqa
print("numpy:", numpy.__version__)
print("scikit-learn:", sklearn.__version__)
import skl2onnx  # noqa
print("onnx: ", onnx.__version__)
print("onnxruntime: ", onnxruntime.__version__)
print("skl2onnx: ", skl2onnx.__version__)
