# SPDX-License-Identifier: Apache-2.0


"""
Custom Operator for NMF Decomposition
=====================================

`NMF <https://scikit-learn.org/stable/modules/generated/
sklearn.decomposition.NMF.html>`_ factorizes an input matrix
into two matrices *W, H* of rank *k* so that :math:`WH \\sim M``.
:math:`M=(m_{ij})` may be a binary matrix where *i* is a user
and *j* a product he bought. The prediction
function depends on whether or not the user needs a
recommandation for an existing user or a new user.
This example addresses the first case.

The second case is more complex as it theoretically
requires the estimation of a new matrix *W* with a
gradient descent.

.. contents::
    :local:

Building a simple model
+++++++++++++++++++++++

"""

import os
import skl2onnx
import onnxruntime
import sklearn
from sklearn.decomposition import NMF
import numpy as np
import matplotlib.pyplot as plt
from onnx.tools.net_drawer import GetPydotGraph, GetOpNodeProducer
import onnx
from skl2onnx.algebra.onnx_ops import (
    OnnxArrayFeatureExtractor, OnnxMul, OnnxReduceSum)
from skl2onnx.common.data_types import FloatTensorType
from onnxruntime import InferenceSession


mat = np.array([[1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0],
                [1, 0, 0, 0], [1, 0, 0, 0]], dtype=np.float64)
mat[:mat.shape[1], :] += np.identity(mat.shape[1])

mod = NMF(n_components=2)
W = mod.fit_transform(mat)
H = mod.components_
pred = mod.inverse_transform(W)

print("original predictions")
exp = []
for i in range(mat.shape[0]):
    for j in range(mat.shape[1]):
        exp.append((i, j, pred[i, j]))

print(exp)

#######################
# Let's rewrite the prediction in a way it is closer
# to the function we need to convert into ONNX.


def predict(W, H, row_index, col_index):
    return np.dot(W[row_index, :], H[:, col_index])


got = []
for i in range(mat.shape[0]):
    for j in range(mat.shape[1]):
        got.append((i, j, predict(W, H, i, j)))

print(got)


#################################
# Conversion into ONNX
# ++++++++++++++++++++
#
# There is no implemented converter for
# `NMF <https://scikit-learn.org/stable/modules/generated/
# sklearn.decomposition.NMF.html>`_ as the function we plan
# to convert is not transformer or a predictor.
# The following converter does not need to be registered,
# it just creates an ONNX graph equivalent to function
# *predict* implemented above.


def nmf_to_onnx(W, H, op_version=12):
    """
    The function converts a NMF described by matrices
    *W*, *H* (*WH* approximate training data *M*).
    into a function which takes two indices *(i, j)*
    and returns the predictions for it. It assumes
    these indices applies on the training data.
    """
    col = OnnxArrayFeatureExtractor(H, 'col')
    row = OnnxArrayFeatureExtractor(W.T, 'row')
    dot = OnnxMul(col, row, op_version=op_version)
    res = OnnxReduceSum(dot, output_names="rec", op_version=op_version)
    indices_type = np.array([0], dtype=np.int64)
    onx = res.to_onnx(inputs={'col': indices_type,
                              'row': indices_type},
                      outputs=[('rec', FloatTensorType((None, 1)))],
                      target_opset=op_version)
    return onx


model_onnx = nmf_to_onnx(W.astype(np.float32),
                         H.astype(np.float32))
print(model_onnx)

########################################
# Let's compute prediction with it.

sess = InferenceSession(model_onnx.SerializeToString())


def predict_onnx(sess, row_indices, col_indices):
    res = sess.run(None,
                   {'col': col_indices,
                    'row': row_indices})
    return res


onnx_preds = []
for i in range(mat.shape[0]):
    for j in range(mat.shape[1]):
        row_indices = np.array([i], dtype=np.int64)
        col_indices = np.array([j], dtype=np.int64)
        pred = predict_onnx(sess, row_indices, col_indices)[0]
        onnx_preds.append((i, j, pred[0, 0]))

print(onnx_preds)


###################################
# The ONNX graph looks like the following.
pydot_graph = GetPydotGraph(
    model_onnx.graph, name=model_onnx.graph.name,
    rankdir="TB", node_producer=GetOpNodeProducer("docstring"))
pydot_graph.write_dot("graph_nmf.dot")
os.system('dot -O -Tpng graph_nmf.dot')
image = plt.imread("graph_nmf.dot.png")
plt.imshow(image)
plt.axis('off')

#################################
# **Versions used for this example**

print("numpy:", np.__version__)
print("scikit-learn:", sklearn.__version__)
print("onnx: ", onnx.__version__)
print("onnxruntime: ", onnxruntime.__version__)
print("skl2onnx: ", skl2onnx.__version__)
