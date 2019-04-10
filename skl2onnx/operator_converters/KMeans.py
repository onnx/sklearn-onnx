# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
import numpy as np
from sklearn.utils.extmath import row_norms
from ..common._registration import register_converter
from ..algebra.onnx_ops import OnnxReduceSumSquare, OnnxGemm
from ..algebra.onnx_ops import OnnxAdd, OnnxArgMin, OnnxSqrt


def convert_sklearn_kmeans(scope, operator, container):
    """
    Computation graph of distances to all centroids for a batch of examples.
    Note that a centriod is just the center of a cluster. We use ``[]`` to
    denote the dimension of a variable; for example, ``X[3, 2]`` means that
    *X* is a *3-by-2* tensor. In addition, for a matrix *X*, $X'$ denotes its
    transpose.

    Symbols:

    * *l*: # of examples.
    * *n*: # of features per input example.
    * *X*: input examples, l-by-n tensor.
    * *C*: centroids, k-by-n tensor.
    * :math:`C^2`: 2-norm of all centriod vectors, its shape is ``[k]``.
    * *Y*: 2-norm of difference between examples and centroids,
      *l-by-k* tensor. The value at i-th row and k-th column row,
      ``Y[i,k]``,is the distance from example *i* to centroid *k*.
    * *L*: the id of the nearest centroid for each input example,
      its shape is ``[l]``.

    ::

         .------------------------------------------------------.
         |                                                      |
         |                                                      v
        X [l, n] --> ReduceSumSquare -> X^2 [l]   Gemm (alpha=-2, transB=1)
                                         |                  |  |- C [k, n]
                                         |                  |
                                         |                  v
                                         `------> Add <-- -2XC' [l, k]
                                                   |
                                                   v
                     C^2 [k] --------> Add <----- Z [l, k]
                                        |
                                        v
                 L [l] <-- ArgMin <--  Y2 [l, k] --> Sqrt --> Y2 [l, k]

    *scikit-learn* code:

    ::

        X = data
        Y = model.cluster_centers_
        XX = row_norms(X, squared=True)
        YY = row_norms(Y, squared=True)
        distances = safe_sparse_dot(X, Y.T, dense_output=True)
        distances *= -2
        distances += XX[:, numpy.newaxis]
        distances += YY[numpy.newaxis, :]
        numpy.sqrt(distances, out=distances)
    """
    X = operator.inputs[0]
    out = operator.outputs
    op = operator.raw_operator

    C = op.cluster_centers_
    C2 = row_norms(C, squared=True)

    N = X.type.shape[0]
    zeros = np.zeros((N, ))

    rs = OnnxReduceSumSquare(X, axes=[1], keepdims=1)
    z = OnnxAdd(rs, OnnxGemm(X, C, zeros, alpha=-2., transB=1))
    y2 = OnnxAdd(C2, z)
    ll = OnnxArgMin(y2, axis=1, keepdims=0, output_names=out[:1])
    y2s = OnnxSqrt(y2, output_names=out[1:])

    ll.add_to(scope, container)
    y2s.add_to(scope, container)


register_converter('SklearnKMeans', convert_sklearn_kmeans)
register_converter('SklearnMiniBatchKMeans', convert_sklearn_kmeans)
