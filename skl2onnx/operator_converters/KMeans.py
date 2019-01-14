# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

from ..proto import onnx_proto
from ..common._apply_operation import apply_add, apply_cast, apply_exp, apply_reshape, apply_sub
from ..common._registration import register_converter
import numpy as np


def convert_sklearn_kmeans(scope, operator, container):
    """
    Computation graph of distances to all centriods for a batch of examples. Note that a centriod is just
    the center of a cluster. We use ``[]`` to denote the dimension of a variable; for example, ``X[3, 2]`` means
    that *X* is a *3-by-2* tensor. In addition, for a matrix *X*, $X'$ denotes its transpose.    
    Symbols:
    
    * *l*: # of examples.
    * *n*: # of features per input example.
    * *X*: input examples, l-by-n tensor.
    * *C*: centroids, k-by-n tensor.
    * $C^2$: 2-norm of all centriod vectors, its shape is ``[k]``.
    * *Y*: 2-norm of difference between examples and centriods, *l-by-k* tensor.
      The value at i-th row and k-th column row, ``Y[i,k]``,
      is the distance from example *i* to centroid *k*.
    * *L*: the id of the nearest centroid for each input example, its shape is ``[l]``.
    
    ::
    
        .------------------------------------------------------.
        |                                                      |
        |                                                      v
        X [l, n] --> ReduceSumSquare --> X^2 [l]             Gemm (alpha=-2, transB=1) <-- C [k, n]
                                          |                    |
                                          |                    v
                                          `------> Add <---- -2XC' [l, k]
                                                    |
                                                    v
                                                    Z [l, k] ----------> Add <------------C^2 [k]
                                                                          |
                                                                          v
                                                  L [l] <--- ArgMin <---  Y [l, k]
    """
    op = operator.raw_operator
    variable = operator.inputs[0]
    N = variable.type.shape[0]
    
    # centroids        
    shapeC = list(op.cluster_centers_.shape)
    nameC = scope.get_unique_variable_name('centroid')
    container.add_initializer(nameC, onnx_proto.TensorProto.FLOAT,
                              shapeC, op.cluster_centers_.flatten())    

    #centroids ^2
    nameC2 = scope.get_unique_variable_name('C2')
    c2 = (op.cluster_centers_ ** 2).sum(axis=1)
    shapeC2 = list(c2.shape)
    container.add_initializer(nameC2, onnx_proto.TensorProto.FLOAT,
                              [shapeC[0]], c2.flatten())    

    nameX2 = scope.get_unique_variable_name('X2')
    nameX = operator.inputs[0].full_name
    container.add_node('ReduceSumSquare', [nameX], [nameX2],
                        name=scope.get_unique_operator_name('ReduceSumSquare'))
    
    # Compute -2XC'
    zero_name = scope.get_unique_variable_name('zero')
    zeros = np.zeros((N, ))
    container.add_initializer(zero_name, onnx_proto.TensorProto.FLOAT,
                              list(zeros.shape), zeros)   
    
    if container.target_opset < 5:
        attrs['broadcast'] = 1
        op_version = 1
    elif container.target_opset < 7:
        attrs['broadcast'] = 1
        op_version = 6
    else:
        op_version = 7
        
    nameXC2 = scope.get_unique_variable_name('XC2')
    container.add_node("Gemm", [nameX, nameC, zero_name], [nameXC2], 
                       name=scope.get_unique_operator_name('Gemm'),
                       alpha=-2., transB=1, op_version=op_version)

    # Compute Z = X^2 - 2XC'
    nameZ = scope.get_unique_variable_name("Z")
    container.add_node("Add", [nameX2, nameXC2], [nameZ], name=scope.get_unique_operator_name('Add'))

    # Compute Y = Z + C^2
    nameY = operator.outputs[1].full_name
    container.add_node("Add", [nameZ, nameC2], [nameY], name=scope.get_unique_operator_name('Add'))

    # Compute the most-matched cluster index, L
    nameL = operator.outputs[0].full_name
    container.add_node("ArgMin", [nameY], [nameL], name=scope.get_unique_operator_name('ArgMin'),
                       axis=1, keepdims=0)


register_converter('SklearnKMeans', convert_sklearn_kmeans)
