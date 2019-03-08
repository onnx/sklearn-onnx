# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
from .onnx_operator import OnnxOperator


class Gemm(OnnxOperator):
    """
    `General Matrix multiplication
    <https://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms#Level_3>`_::

      A' = transpose(A) if transA else A
      B' = transpose(B) if transB else B


    Compute *Y = alpha * A' * B' + beta * C*, where input tensor *A* has shape *(M, K)* or *(K, M)*,
    input tensor *B* has shape *(K, N)* or *(N, K)*, input tensor *C* is broadcastable to shape *(M, N)*,
    and output tensor Y has shape *(M, N)*. A will be transposed before doing the
    computation if attribute *transA* is non-zero, same for *B* and *transB*.
    This operator supports **unidirectional broadcasting**
    (tensor *C* should be unidirectional broadcastable to tensor *A * B*);
    for more details please check [the doc](Broadcasting.md).

    Version
    -------

    This version of the operator has been available since version 9 of
    the default ONNX operator set.

    Attributes
    ----------

    .. html::

        <dl>
        <dt><tt>alpha</tt> : float (default is 1.0)</dt>
        <dd>Scalar multiplier for the product of input tensors A * B.</dd>
        <dt><tt>beta</tt> : float (default is 1.0)</dt>
        <dd>Scalar multiplier for input tensor C.</dd>
        <dt><tt>transA</tt> : int (default is 0)</dt>
        <dd>Whether A should be transposed</dd>
        <dt><tt>transB</tt> : int (default is 0)</dt>
        <dd>Whether B should be transposed</dd>
        </dl>

    Inputs
    ------

    .. html::

        <dl>
        <dt><tt>A</tt> : T</dt>
        <dd>Input tensor A. The shape of A should be (M, K) if transA is 0, or (K, M) if transA is non-zero.</dd>
        <dt><tt>B</tt> : T</dt>
        <dd>Input tensor B. The shape of B should be (K, N) if transB is 0, or (N, K) if transB is non-zero.</dd>
        <dt><tt>C</tt> : T</dt>
        <dd>Input tensor C. The shape of C should be unidirectional broadcastable to (M, N).</dd>
        </dl>

    Outputs
    -------

    .. html::

        <dl>
        <dt><tt>Y</tt> : T</dt>
        <dd>Output tensor of shape (M, N).</dd>
        </dl>

    Type Constraints
    ----------------

    .. html::

        <dl>
        <dt><tt>T</tt> : tensor(float16), tensor(float), tensor(double), tensor(uint32), tensor(uint64), tensor(int32), tensor(int64)</dt>
        <dd>Constrain input and output types to float/int tensors.</dd>
        </dl>


    Examples
    --------

    .. runpython::
        :showcode:

        import onnx.helper
        import numpy as np
        node = onnx.helper.make_node(
            'Gemm',
            inputs=['a', 'b', 'c'],
            outputs=['y'],
            alpha=0.5,
            beta=0.5
        )
        a = np.random.ranf([3, 6]).astype(np.float32)
        b = np.random.ranf([6, 4]).astype(np.float32)
        c = np.random.ranf([3, 4]).astype(np.float32)
        y = 0.5 * np.dot(a, b) + 0.5 * c
        print(y)
        expect(node, inputs=[a, b, c], outputs=[y],
               name='test_gemm_nobroadcast')

    .. runpython::
        :showcode:

        import onnx.helper
        import numpy as np
        node = onnx.helper.make_node(
            'Gemm',
            inputs=['a', 'b', 'c'],
            outputs=['y'],
            alpha=0.5,
            beta=0.5,
            transA=1,
            transB=1
        )
        a = np.random.ranf([6, 3]).astype(np.float32)
        b = np.random.ranf([4, 6]).astype(np.float32)
        c = np.random.ranf([1, 1]).astype(np.float32)
        y = 0.5 * np.dot(a.T, b.T) + 0.5 * c
        print(y)
        expect(node, inputs=[a, b, c], outputs=[y],
               name='test_gemm_broadcast')
    """
    pass
