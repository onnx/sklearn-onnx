# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
from .onnx_operator import OnnxOperator


class Mul(OnnxOperator):
    """
    Performs element-wise multiplication (with Numpy-style broadcasting support).
    This operator supports **multidirectional (i.e., Numpy-style) broadcasting**;
    for more details please check :epkg:`broadcasting`.

    Version
    -------

    This version of the operator has been available since version 7
    of the default ONNX operator set.

    Inputs
    ------

    .. html::

        <dl>
        <dt><tt>A</tt> : T</dt>
        <dd>First operand.</dd>
        <dt><tt>B</tt> : T</dt>
        <dd>Second operand.</dd>
        </dl>

    Outputs
    -------

    .. html::

        <dl>
        <dt><tt>C</tt> : T</dt>
        <dd>Result, has same element type as two inputs</dd>
        </dl>

    Type Constraints
    ----------------

    .. html::

        <dl>
        <dt><tt>T</tt> : tensor(uint32), tensor(uint64), tensor(int32), tensor(int64), tensor(float16), tensor(float), tensor(double)</dt>
        <dd>Constrain input and output types to high-precision numeric tensors.</dd>
        </dl>


    Examples
    --------

    .. runpython::
        :showcode:

        import onnx.helper
        import numpy as np

        node = onnx.helper.make_node(
            'Mul',
            inputs=['x', 'y'],
            outputs=['z'],
        )

        x = np.array([1, 2, 3]).astype(np.float32)
        y = np.array([3, 2, 1]).astype(np.float32)
        z = x * y
        print(z)
        expect(node, inputs=[x, y], outputs=[z],
               name='test_mul_example')

        x = np.random.randn(3, 4, 5).astype(np.float32)
        y = np.random.randn(3, 4, 5).astype(np.float32)
        z = x - y
        expect(node, inputs=[x, y], outputs=[z],
               name='test_mul')

    .. runpython::
        :showcode:

        import onnx.helper
        import numpy as np

        node = onnx.helper.make_node(
            'Mul',
            inputs=['x', 'y'],
            outputs=['z'],
        )

        x = np.random.randn(3, 4, 5).astype(np.float32)
        y = np.random.randn(5).astype(np.float32)
        z = x * y
        print(z)
        expect(node, inputs=[x, y], outputs=[z],
               name='test_mul_bcast')
    """
    pass
