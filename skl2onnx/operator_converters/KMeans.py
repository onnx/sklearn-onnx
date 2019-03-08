# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

from ..proto import onnx_proto
from ..common._apply_operation import apply_add, apply_gemm, apply_sqrt
from ..common._registration import register_converter
import numpy as np
from sklearn.utils.extmath import row_norms


def to_onnx_kmeans(scope, operator, container):
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
    * $C^2$: 2-norm of all centriod vectors, its shape is ``[k]``.
    * *Y*: 2-norm of difference between examples and centroids,
      *l-by-k* tensor. The value at i-th row and k-th column row,
      ``Y[i,k]``,is the distance from example *i* to centroid *k*.
    * *L*: the id of the nearest centroid for each input example,
      its shape is ``[l]``.

    ::

     .------------------------------------------------------.
     |                                                      |
     |                                                      v
X [l, n] --> ReduceSumSquare -> X^2 [l]   Gemm (alpha=-2, transB=1) <- C [k, n]
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
    op = operator.raw_operator
    variable = operator.inputs[0]
    N = variable.type.shape[0]

    # centroids
    shapeC = list(op.cluster_centers_.shape)
    nameC = scope.get_unique_variable_name('centroid')
    container.add_initializer(nameC, onnx_proto.TensorProto.FLOAT,
                              shapeC, op.cluster_centers_.flatten())

    nameX2 = scope.get_unique_variable_name('X2')
    nameX = operator.inputs[0].full_name
    container.add_node('ReduceSumSquare', [nameX], [nameX2], axes=[1],
                       keepdims=1,
                       name=scope.get_unique_operator_name('ReduceSumSquare'))

    # Compute -2XC'
    zero_name = scope.get_unique_variable_name('zero')
    zeros = np.zeros((N, ))
    container.add_initializer(zero_name, onnx_proto.TensorProto.FLOAT,
                              list(zeros.shape), zeros)
    nameXC2 = scope.get_unique_variable_name('XC2')
    apply_gemm(scope, [nameX, nameC, zero_name], [nameXC2], container,
               alpha=-2., transB=1)

    # Compute Z = X^2 - 2XC'
    nameZ = scope.get_unique_variable_name("Z")
    apply_add(scope, [nameXC2, nameX2], [nameZ], container)

    # centroids ^2
    nameC2 = scope.get_unique_variable_name('C2')
    c2 = row_norms(op.cluster_centers_, squared=True)
    container.add_initializer(nameC2, onnx_proto.TensorProto.FLOAT,
                              [1, shapeC[0]], c2.flatten())

    # Compute Y2 = Z + C^2
    nameY2 = scope.get_unique_variable_name('Y2')
    apply_add(scope, [nameZ, nameC2], [nameY2], container)

    # Compute Y = sqrt(Y2)
    nameY = operator.outputs[1].full_name
    apply_sqrt(scope, [nameY2], [nameY], container)

    # Compute the most-matched cluster index, L
    nameL = operator.outputs[0].full_name
    container.add_node('ArgMin', [nameY2], [nameL],
                       name=scope.get_unique_operator_name('ArgMin'),
                       axis=1, keepdims=0)


register_converter('SklearnKMeans', to_onnx_kmeans)
register_converter('SklearnMiniBatchKMeans', to_onnx_kmeans)
